"""Pydantic models for the controlled function-calling layer (Phase 5).

These models are the shared vocabulary between the tool registry, the LLM
layer, the chatbot and the evaluation harness. They make every function call
explicit, typed and loggable — which is the core of the thesis's "secure,
controlled retrieval" argument (Sub-RQ 1).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


def strip_internal_keys(data: Any) -> Any:
    """Drop the internal ``_normalized_arguments`` key from tool-result data.

    Tools embed normalization bookkeeping under ``_normalized_arguments``; it is
    useful for traces but should not appear in answers or ground-truth values.
    """
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if k != "_normalized_arguments"}
    return data


class ErrorCategory(StrEnum):
    """Error taxonomy shared by the registry and the evaluation (Phase 13).

    Only the subset that the registry itself can detect is used here; the
    evaluation layer adds answer-level categories (e.g. correct call but wrong
    final response).
    """

    NONE = "none"
    UNKNOWN_FUNCTION = "hallucinated_or_unsupported_function"
    MISSING_REQUIRED_PARAMETER = "missing_required_parameter"
    UNSUPPORTED_PARAMETER = "unsupported_parameter"
    INVALID_PARAMETER_TYPE = "invalid_parameter_type"
    INVALID_ENUM_VALUE = "incorrect_parameter_value"
    EXECUTION_FAILURE = "execution_or_runtime_failure"
    INVALID_JSON_OR_SCHEMA = "invalid_json_or_schema_error"


class ToolParameter(BaseModel):
    """One parameter of a tool, as advertised to the LLM."""

    name: str
    type: str = Field(description="JSON-schema type, e.g. 'string', 'integer'.")
    description: str = ""
    required: bool = False
    enum: list[str] | None = Field(
        default=None, description="Allowed values, if the parameter is constrained."
    )


class ToolDefinition(BaseModel):
    """A tool the LLM is allowed to call, plus its JSON schema."""

    name: str
    description: str
    parameters: list[ToolParameter] = Field(default_factory=list)

    def to_openai_schema(self) -> dict[str, Any]:
        """Render an OpenAI/Ollama-compatible tool schema (function-calling)."""
        properties: dict[str, Any] = {}
        required: list[str] = []
        for p in self.parameters:
            # Optional parameters must also accept null: some models emit e.g.
            # `"floor": null`, which strict server-side tool validation rejects
            # unless the schema type explicitly allows null.
            ptype: str | list[str] = p.type if p.required else [p.type, "null"]
            prop: dict[str, Any] = {"type": ptype, "description": p.description}
            if p.enum is not None:
                prop["enum"] = p.enum
            properties[p.name] = prop
            if p.required:
                required.append(p.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class ToolCall(BaseModel):
    """A request to call one tool with arguments (as proposed by the LLM)."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The structured outcome of executing (or rejecting) a tool call."""

    tool_name: str
    ok: bool
    data: Any = None
    error_category: ErrorCategory = ErrorCategory.NONE
    error_message: str | None = None
    # Normalized arguments actually used (after synonym resolution).
    normalized_arguments: dict[str, Any] = Field(default_factory=dict)
    row_count: int | None = None
    latency_ms: float | None = None

    def summary(self, max_len: int = 300) -> str:
        """Short, log-friendly description of the result payload."""
        if not self.ok:
            return f"ERROR[{self.error_category.value}]: {self.error_message}"
        text = str(self.data)
        return text[:max_len] + ("…" if len(text) > max_len else "")


class FunctionCallLog(BaseModel):
    """Audit record for a single function call (traceability requirement)."""

    timestamp: datetime = Field(default_factory=_utcnow)
    model_name: str | None = None
    user_query: str | None = None
    selected_function: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    normalized_arguments: dict[str, Any] = Field(default_factory=dict)
    validation_ok: bool = False
    execution_ok: bool = False
    result_summary: str = ""
    row_count: int | None = None
    latency_ms: float | None = None
    error_category: ErrorCategory = ErrorCategory.NONE

    def to_jsonl(self) -> str:
        """Serialise to a single JSON line (secrets are never stored here)."""
        return self.model_dump_json()
