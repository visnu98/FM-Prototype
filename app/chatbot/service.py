"""Chatbot service pipeline (Phase 9).

Ties the LLM, the tool registry and the FM functions into one grounded
question-answering flow:

    question -> LLM selects a tool -> registry validates + executes ->
    LLM phrases a grounded final answer (from the tool result only).

The returned :class:`ChatResponse` carries the full trace (selected function,
raw + normalized arguments, tool result, per-step latency, error category). The
Phase 14 evaluation runner reuses this exact pipeline so the chatbot and the
experiment cannot diverge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.llm.base import LLMClient
from app.llm.groq_client import make_client
from app.tools.fm_functions import build_registry
from app.tools.models import ErrorCategory, ToolCall, ToolResult
from app.tools.registry import ToolRegistry

# Keep the final-answer prompt small: trim long lists in the tool result.
_MAX_LIST_ITEMS = 25


@dataclass
class ChatResponse:
    """Full trace of one question through the pipeline."""

    user_query: str
    model_name: str
    final_answer: str
    made_tool_call: bool
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    normalized_arguments: dict[str, Any] = field(default_factory=dict)
    error_category: ErrorCategory = ErrorCategory.NONE
    latency_tool_call_ms: float = 0.0
    latency_sql_ms: float = 0.0
    latency_final_answer_ms: float = 0.0
    latency_total_ms: float = 0.0

    @property
    def selected_function(self) -> str | None:
        return self.tool_call.name if self.tool_call else None

    @property
    def arguments(self) -> dict[str, Any]:
        return self.tool_call.arguments if self.tool_call else {}


def _trim_for_prompt(data: Any) -> Any:
    """Shorten long lists so the final-answer prompt stays compact."""
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            if k == "_normalized_arguments":
                continue
            out[k] = _trim_for_prompt(v)
        return out
    if isinstance(data, list) and len(data) > _MAX_LIST_ITEMS:
        return data[:_MAX_LIST_ITEMS] + [f"… (+{len(data) - _MAX_LIST_ITEMS} more)"]
    return data


def _result_to_json(result: ToolResult) -> str:
    if result.ok:
        payload = _trim_for_prompt(result.data)
    else:
        payload = {
            "ok": False,
            "error_category": result.error_category.value,
            "error_message": result.error_message,
        }
    return json.dumps(payload, default=str, ensure_ascii=False)


class ChatbotService:
    """Stateless service: one call answers one question."""

    def __init__(self, client: LLMClient, registry: ToolRegistry | None = None) -> None:
        self.client = client
        self.registry = registry if registry is not None else build_registry()
        self._schemas = self.registry.openai_schemas()

    @classmethod
    def for_model(cls, model_name: str, registry: ToolRegistry | None = None) -> ChatbotService:
        """Build a service for a named model using the configured provider."""
        return cls(make_client(model_name), registry)

    def answer(self, user_query: str) -> ChatResponse:
        """Run the full pipeline for one question."""
        # 1) Tool selection.
        decision = self.client.generate_tool_call(user_query, self._schemas)

        # 2) No tool call: the model answered directly or asked to clarify.
        if not decision.made_tool_call or decision.tool_call is None:
            answer = decision.raw_content or (
                "I could not determine which tool to use for this question."
            )
            return ChatResponse(
                user_query=user_query,
                model_name=self.client.model_name,
                final_answer=answer,
                made_tool_call=False,
                error_category=(
                    ErrorCategory.NONE
                    if decision.parse_error is None
                    else ErrorCategory.INVALID_JSON_OR_SCHEMA
                ),
                latency_tool_call_ms=decision.latency_ms,
                latency_total_ms=decision.latency_ms,
            )

        # 3) Validate + execute the proposed tool call.
        result = self.registry.execute(
            decision.tool_call,
            model_name=self.client.model_name,
            user_query=user_query,
        )
        sql_latency = result.latency_ms or 0.0

        # 4) Grounded final answer (covers both success and failure results).
        result_json = _result_to_json(result)
        final_answer, final_latency = self.client.generate_final_answer(
            user_query, decision.tool_call.name, result_json
        )

        total = decision.latency_ms + sql_latency + final_latency
        return ChatResponse(
            user_query=user_query,
            model_name=self.client.model_name,
            final_answer=final_answer,
            made_tool_call=True,
            tool_call=decision.tool_call,
            tool_result=result,
            normalized_arguments=result.normalized_arguments,
            error_category=result.error_category,
            latency_tool_call_ms=decision.latency_ms,
            latency_sql_ms=sql_latency,
            latency_final_answer_ms=final_latency,
            latency_total_ms=total,
        )
