"""Groq multi-step function-calling client.

Wraps the Groq SDK with native (parallel) tool calling and a strict-JSON
fallback for models that emit a tool call as text. The same system prompt, tool
schemas and temperature are used for every model so the comparison is fair. The
API key is read from settings and never logged.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from groq import APIError, Groq, RateLimitError

from app.config import get_settings
from app.llm.base import AssistantTurn, LLMClient, PlannedToolCall
from app.llm.json_tool_parser import ToolCallParseError, parse_tool_call
from app.tools.models import ToolCall

logger = logging.getLogger(__name__)

# Hard-coded decoding temperature. 0.0 = deterministic output, which the thesis
# requires for reproducible evaluation. This is intentionally not configurable.
TEMPERATURE = 0.0


class GroqClient(LLMClient):
    """An :class:`LLMClient` backed by Groq's chat completions API."""

    def __init__(self, model_name: str) -> None:
        settings = get_settings()
        if settings.groq_api_key is None:
            raise RuntimeError("GROQ_API_KEY is not set in the environment/.env.")
        super().__init__(model_name=model_name, temperature=TEMPERATURE)
        self._client = Groq(api_key=settings.groq_api_key.get_secret_value())
        # Wall-time (ms) of the last *successful* API call, excluding any
        # rate-limit backoff sleeps — so reported latency reflects model speed.
        self._last_api_ms = 0.0
        # Reasoning models (qwen3, deepseek, gpt-oss) emit chain-of-thought into
        # the content. Hide it so final answers are clean and tokens are saved.
        lname = model_name.lower()
        self._reasoning_model = any(k in lname for k in ("qwen3", "deepseek", "gpt-oss"))

    def _extra_params(self) -> dict[str, Any]:
        return {"reasoning_format": "hidden"} if self._reasoning_model else {}

    def _create(self, **kwargs: Any) -> Any:
        """Call Groq with bounded retry/backoff on rate limits (free tier).

        Per-minute (TPM/RPM) limits clear within ~60s, so back off up to that.
        Daily token caps cannot be waited out within a run and will surface as a
        final RateLimitError. Backoff sleeps are NOT counted in ``_last_api_ms``.
        """
        max_attempts = 6
        for attempt in range(max_attempts):
            try:
                t0 = time.perf_counter()
                resp = self._client.chat.completions.create(**kwargs)
                self._last_api_ms = (time.perf_counter() - t0) * 1000
                return resp
            except RateLimitError:
                wait = min(2**attempt, 60)
                logger.warning("Groq rate limit; retrying in %ss", wait)
                time.sleep(wait)
            except APIError as exc:
                if attempt < 2 and getattr(exc, "status_code", 500) >= 500:
                    time.sleep(2**attempt)
                    continue
                raise
        return self._client.chat.completions.create(**kwargs)

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AssistantTurn:
        """One turn: return the model's tool calls, or its final answer."""
        try:
            response = self._create(
                model=self.model_name,
                temperature=self.temperature,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                **self._extra_params(),
            )
        except Exception as exc:  # pragma: no cover - network dependent
            logger.error("Groq request failed: %s", type(exc).__name__)
            return AssistantTurn(request_error=f"request_failed: {type(exc).__name__}")

        latency = self._last_api_ms  # excludes rate-limit backoff sleeps
        message = response.choices[0].message
        content = message.content

        # 1) Native (possibly parallel) tool calls.
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            planned: list[PlannedToolCall] = []
            raw_calls: list[dict[str, Any]] = []
            for tc in tool_calls:
                planned.append(PlannedToolCall(id=tc.id, call=_to_tool_call(tc.function)))
                raw_calls.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                )
            return AssistantTurn(
                planned_calls=planned,
                latency_ms=latency,
                assistant_message={
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": raw_calls,
                },
            )

        # 2) Fallback: a tool call emitted as JSON text (some weaker models).
        if content:
            try:
                parsed = parse_tool_call(content)
                synthetic_id = "fallback_0"
                return AssistantTurn(
                    planned_calls=[PlannedToolCall(id=synthetic_id, call=parsed)],
                    latency_ms=latency,
                    assistant_message={
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": synthetic_id,
                                "type": "function",
                                "function": {
                                    "name": parsed.name,
                                    "arguments": json.dumps(parsed.arguments),
                                },
                            }
                        ],
                    },
                )
            except ToolCallParseError:
                pass

        # 3) Final answer (no tool calls).
        return AssistantTurn(
            content=content or "",
            latency_ms=latency,
            assistant_message={"role": "assistant", "content": content or ""},
        )


def _to_tool_call(function: Any) -> ToolCall:
    try:
        args = json.loads(function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    if not isinstance(args, dict):
        args = {}
    return ToolCall(name=function.name, arguments=args)


def make_client(model_name: str) -> LLMClient:
    """Factory used by the chatbot/evaluation to build a client by provider."""
    provider = get_settings().llm_provider.lower()
    if provider == "groq":
        return GroqClient(model_name)
    raise RuntimeError(f"Unsupported LLM_PROVIDER '{provider}'. Only 'groq' is implemented.")
