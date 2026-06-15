"""Strict JSON tool-call parser (Phase 8).

Used as a fallback when a model emits a tool call as JSON text instead of via
native tool-calling. It extracts and validates a ``{"name", "arguments"}``
object, rejecting malformed output. It does NOT validate against the registry —
that is the registry's job; this only guarantees a well-formed ToolCall.
"""

from __future__ import annotations

import json
import re

from app.tools.models import ToolCall

# Matches a ```json ... ``` or ``` ... ``` fenced block.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class ToolCallParseError(ValueError):
    """Raised when text cannot be parsed into a valid tool call."""


def _candidate_json_strings(text: str) -> list[str]:
    """Yield plausible JSON substrings from free-form model output."""
    candidates: list[str] = []
    candidates.extend(m.group(1).strip() for m in _FENCE_RE.finditer(text))
    # First balanced {...} block as a last resort.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])
    candidates.append(text.strip())
    return candidates


def parse_tool_call(text: str) -> ToolCall:
    """Parse model text into a :class:`ToolCall` or raise ToolCallParseError."""
    if not text or not text.strip():
        raise ToolCallParseError("Empty model output.")

    last_error: str | None = None
    for candidate in _candidate_json_strings(text):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = f"invalid JSON: {exc}"
            continue
        if not isinstance(obj, dict):
            last_error = "JSON is not an object."
            continue
        # Accept both {"name","arguments"} and OpenAI-style {"function":{...}}.
        if "function" in obj and isinstance(obj["function"], dict):
            obj = obj["function"]
        name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
        if not isinstance(name, str) or not name:
            last_error = "missing 'name'."
            continue
        arguments = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        return ToolCall(name=name, arguments=arguments)

    raise ToolCallParseError(last_error or "No valid tool call found.")
