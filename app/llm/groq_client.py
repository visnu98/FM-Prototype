"""Groq function-calling client (Phase 8).

Wraps the Groq SDK with native tool calling and a strict-JSON fallback. The same
system prompt, tool schemas and temperature are used for every model so the H3
comparison is fair. The API key is read from settings and never logged.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from groq import APIError, Groq, RateLimitError

from app.config import get_settings
from app.llm.base import LLMClient, ToolCallDecision
from app.llm.json_tool_parser import ToolCallParseError, parse_tool_call
from app.llm.prompt_templates import SYSTEM_PROMPT, final_answer_prompt
from app.tools.models import ToolCall

logger = logging.getLogger(__name__)


class GroqClient(LLMClient):
    """An :class:`LLMClient` backed by Groq's chat completions API."""

    def __init__(self, model_name: str, temperature: float | None = None) -> None:
        settings = get_settings()
        if settings.groq_api_key is None:
            raise RuntimeError("GROQ_API_KEY is not set in the environment/.env.")
        temp = settings.llm_temperature if temperature is None else temperature
        super().__init__(model_name=model_name, temperature=temp)
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
        final RateLimitError (handled by the caller). Backoff sleeps are NOT
        counted towards :attr:`_last_api_ms`.
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
                # Transient server errors: retry a couple of times.
                if attempt < 2 and getattr(exc, "status_code", 500) >= 500:
                    time.sleep(2**attempt)
                    continue
                raise
        # Final attempt without catching, to surface the error.
        return self._client.chat.completions.create(**kwargs)

    # ── Tool selection ───────────────────────────────────────────────────────

    def generate_tool_call(self, user_query: str, tools: list[dict[str, Any]]) -> ToolCallDecision:
        start = time.perf_counter()
        try:
            response = self._create(
                model=self.model_name,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_query},
                ],
                tools=tools,
                tool_choice="auto",
                **self._extra_params(),
            )
        except Exception as exc:  # pragma: no cover - network dependent
            latency = (time.perf_counter() - start) * 1000
            logger.error("Groq tool-call request failed: %s", type(exc).__name__)
            return ToolCallDecision(
                tool_call=None,
                made_tool_call=False,
                latency_ms=latency,
                parse_error=f"request_failed: {type(exc).__name__}",
            )

        latency = self._last_api_ms  # excludes rate-limit backoff sleeps
        message = response.choices[0].message
        content = message.content

        # 1) Native tool call.
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            first = tool_calls[0]
            try:
                arguments = json.loads(first.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            return ToolCallDecision(
                tool_call=ToolCall(name=first.function.name, arguments=arguments),
                made_tool_call=True,
                raw_content=content,
                latency_ms=latency,
            )

        # 2) Fallback: try to parse a JSON tool call from the content.
        if content:
            try:
                parsed = parse_tool_call(content)
                return ToolCallDecision(
                    tool_call=parsed,
                    made_tool_call=True,
                    raw_content=content,
                    latency_ms=latency,
                )
            except ToolCallParseError:
                pass

        # 3) No tool call (model answered directly or asked for clarification).
        return ToolCallDecision(
            tool_call=None,
            made_tool_call=False,
            raw_content=content,
            latency_ms=latency,
        )

    # ── Final answer ─────────────────────────────────────────────────────────

    def generate_final_answer(
        self, user_query: str, tool_name: str, tool_result_json: str
    ) -> tuple[str, float]:
        start = time.perf_counter()
        prompt = final_answer_prompt(user_query, tool_name, tool_result_json)
        try:
            response = self._create(
                model=self.model_name,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                **self._extra_params(),
            )
            answer = response.choices[0].message.content or ""
            latency = self._last_api_ms  # excludes rate-limit backoff sleeps
        except Exception as exc:  # pragma: no cover - network dependent
            logger.error("Groq final-answer request failed: %s", type(exc).__name__)
            answer = f"(Could not generate a final answer: {type(exc).__name__}.)"
            latency = (time.perf_counter() - start) * 1000
        return answer.strip(), latency


def make_client(model_name: str, temperature: float | None = None) -> LLMClient:
    """Factory used by the chatbot/evaluation to build a client by provider."""
    provider = get_settings().llm_provider.lower()
    if provider == "groq":
        return GroqClient(model_name, temperature)
    raise RuntimeError(f"Unsupported LLM_PROVIDER '{provider}'. Only 'groq' is implemented.")
