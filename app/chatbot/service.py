"""Chatbot service: a multi-step tool-calling loop.

One question may take several tool calls. The service:

1. sends the question + tool schemas to the model,
2. runs any tool calls the model requests (validated + executed by the registry),
3. feeds the results back and repeats,
4. stops when the model returns a final answer (or a step limit is reached).

Complex questions (compare two floors, "which floor has the most …") are
answered by the model *composing* the atomic tools and reasoning over the
results — there is no bespoke function per question. The returned
:class:`ChatResponse` records every step and is reused by the evaluation runner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.llm.base import LLMClient
from app.llm.groq_client import make_client
from app.llm.prompt_templates import SYSTEM_PROMPT
from app.tools.fm_functions import build_registry
from app.tools.models import ErrorCategory, ToolCall, ToolResult, strip_internal_keys
from app.tools.registry import ToolRegistry

# Max model<->tools round trips before forcing an answer (guards against loops).
MAX_STEPS = 6
# Trim long lists in tool results fed back to the model, to keep the context small.
_MAX_LIST_ITEMS = 25


@dataclass
class Step:
    """One executed tool call within a question."""

    call: ToolCall
    result: ToolResult

    @property
    def name(self) -> str:
        return self.call.name


@dataclass
class ChatResponse:
    """Full trace of one question through the multi-step loop."""

    user_query: str
    model_name: str
    final_answer: str
    steps: list[Step] = field(default_factory=list)
    error_category: ErrorCategory = ErrorCategory.NONE
    latency_planning_ms: float = 0.0
    latency_tools_ms: float = 0.0
    latency_total_ms: float = 0.0

    @property
    def made_tool_call(self) -> bool:
        return bool(self.steps)

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def selected_functions(self) -> list[str]:
        """The ordered list of functions the model actually called."""
        return [s.name for s in self.steps]

    @property
    def calls(self) -> list[dict[str, Any]]:
        """Per-step (function, raw args, normalized args, ok, error) for traces."""
        out: list[dict[str, Any]] = []
        for s in self.steps:
            out.append(
                {
                    "function": s.call.name,
                    "arguments": s.call.arguments,
                    "normalized_arguments": s.result.normalized_arguments,
                    "ok": s.result.ok,
                    "error_category": s.result.error_category.value,
                    "result": strip_internal_keys(s.result.data) if s.result.ok else None,
                }
            )
        return out


def _trim_for_prompt(data: Any) -> Any:
    """Shorten long lists so the tool results fed back stay compact."""
    if isinstance(data, dict):
        return {k: _trim_for_prompt(v) for k, v in data.items() if k != "_normalized_arguments"}
    if isinstance(data, list) and len(data) > _MAX_LIST_ITEMS:
        return data[:_MAX_LIST_ITEMS] + [f"… (+{len(data) - _MAX_LIST_ITEMS} more)"]
    return data


def _result_to_json(result: ToolResult) -> str:
    if result.ok:
        payload: Any = _trim_for_prompt(result.data)
    else:
        payload = {
            "ok": False,
            "error_category": result.error_category.value,
            "error_message": result.error_message,
        }
    return json.dumps(payload, default=str, ensure_ascii=False)


class ChatbotService:
    """Stateless service: one call answers one question (possibly multi-step)."""

    def __init__(self, client: LLMClient, registry: ToolRegistry | None = None) -> None:
        self.client = client
        self.registry = registry if registry is not None else build_registry()
        self._schemas = self.registry.openai_schemas()

    @classmethod
    def for_model(cls, model_name: str, registry: ToolRegistry | None = None) -> ChatbotService:
        """Build a service for a named model using the configured provider."""
        return cls(make_client(model_name), registry)

    def answer(self, user_query: str) -> ChatResponse:
        """Run the multi-step loop for one question."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]
        steps: list[Step] = []
        planning_ms = 0.0
        tools_ms = 0.0
        worst_error = ErrorCategory.NONE

        for _ in range(MAX_STEPS):
            turn = self.client.chat(messages, self._schemas)
            planning_ms += turn.latency_ms

            if turn.request_error is not None:
                return ChatResponse(
                    user_query=user_query,
                    model_name=self.client.model_name,
                    final_answer="(The language model request failed.)",
                    steps=steps,
                    error_category=ErrorCategory.INVALID_JSON_OR_SCHEMA,
                    latency_planning_ms=planning_ms,
                    latency_tools_ms=tools_ms,
                    latency_total_ms=planning_ms + tools_ms,
                )

            # No tool calls -> the model produced its final answer.
            if not turn.wants_tools:
                return ChatResponse(
                    user_query=user_query,
                    model_name=self.client.model_name,
                    final_answer=turn.content or "",
                    steps=steps,
                    error_category=worst_error,
                    latency_planning_ms=planning_ms,
                    latency_tools_ms=tools_ms,
                    latency_total_ms=planning_ms + tools_ms,
                )

            # Execute each requested tool call and feed results back.
            messages.append(turn.assistant_message)
            for planned in turn.planned_calls:
                result = self.registry.execute(
                    planned.call,
                    model_name=self.client.model_name,
                    user_query=user_query,
                )
                tools_ms += result.latency_ms or 0.0
                if result.error_category is not ErrorCategory.NONE:
                    worst_error = result.error_category
                steps.append(Step(call=planned.call, result=result))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": planned.id,
                        "content": _result_to_json(result),
                    }
                )

        # Step limit reached: ask once more for an answer without tools.
        messages.append(
            {
                "role": "user",
                "content": "Please give your best final answer now using the results above.",
            }
        )
        turn = self.client.chat(messages, self._schemas)
        planning_ms += turn.latency_ms
        return ChatResponse(
            user_query=user_query,
            model_name=self.client.model_name,
            final_answer=turn.content or "(No answer produced within the step limit.)",
            steps=steps,
            error_category=worst_error,
            latency_planning_ms=planning_ms,
            latency_tools_ms=tools_ms,
            latency_total_ms=planning_ms + tools_ms,
        )
