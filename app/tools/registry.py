"""Tool registry: the single gate through which functions are called (Phase 5).

Responsibilities:
- Register callable FM functions together with a Pydantic *arguments model*.
- Advertise tool JSON schemas to the LLM.
- Validate every proposed call BEFORE execution:
    * reject unknown / hallucinated functions,
    * reject missing required parameters,
    * reject unsupported (extra) parameters,
    * reject wrong parameter types,
    * reject invalid enum values (raised by the normalization layer).
- Execute only approved functions and return a structured ``ToolResult``.
- Classify errors using the shared taxonomy and log every call.

The LLM never touches SQL or the database directly; it can only ask this
registry to run a named, validated function.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from app.config import PROJECT_ROOT
from app.tools.models import (
    ErrorCategory,
    FunctionCallLog,
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)

logger = logging.getLogger(__name__)

LOG_DIR = PROJECT_ROOT / "data" / "logs"
LOG_FILE = LOG_DIR / "function_calls.jsonl"


class ValueNotAllowedError(ValueError):
    """Raised by tool functions when an argument value is not allowed.

    The normalization layer raises this for unrecognised floors / component
    types / attributes so the registry can classify it as an invalid enum
    value rather than a generic execution failure.
    """

    def __init__(self, message: str, *, parameter: str | None = None) -> None:
        super().__init__(message)
        self.parameter = parameter


# Type of an FM function: receives validated kwargs, returns JSON-serialisable
# data. May raise ValueNotAllowedError for bad enum-like values.
ToolFunc = Callable[..., Any]


@dataclass
class RegisteredTool:
    """A registered function plus its schema and argument validator."""

    name: str
    description: str
    func: ToolFunc
    args_model: type[BaseModel]
    definition: ToolDefinition


class ToolRegistry:
    """Holds registered tools and runs validated calls."""

    def __init__(self, log_file: Path | None = None) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self._log_file = log_file if log_file is not None else LOG_FILE

    # ── Registration ────────────────────────────────────────────────────────

    def register(
        self,
        *,
        name: str,
        description: str,
        func: ToolFunc,
        args_model: type[BaseModel],
        parameters: list[ToolParameter] | None = None,
    ) -> None:
        """Register a tool. ``parameters`` advertises the schema to the LLM."""
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered.")
        definition = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters or [],
        )
        self._tools[name] = RegisteredTool(
            name=name,
            description=description,
            func=func,
            args_model=args_model,
            definition=definition,
        )
        logger.debug("Registered tool '%s'", name)

    # ── Introspection ───────────────────────────────────────────────────────

    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def openai_schemas(self) -> list[dict[str, Any]]:
        """All tool schemas in OpenAI/Ollama function-calling format."""
        return [t.definition.to_openai_schema() for t in self._tools.values()]

    # ── Validation + execution ──────────────────────────────────────────────

    def _validate_arguments(
        self, tool: RegisteredTool, arguments: dict[str, Any]
    ) -> tuple[BaseModel | None, ErrorCategory, str | None]:
        """Validate raw arguments against the tool's Pydantic model."""
        try:
            model = tool.args_model.model_validate(arguments)
            return model, ErrorCategory.NONE, None
        except ValidationError as exc:
            # Map the first error to a taxonomy category.
            first = exc.errors()[0]
            etype = first.get("type", "")
            loc = ".".join(str(x) for x in first.get("loc", ()))
            msg = f"{loc}: {first.get('msg', 'invalid')}"
            if etype in {"missing"}:
                return None, ErrorCategory.MISSING_REQUIRED_PARAMETER, msg
            if etype in {"extra_forbidden"}:
                return None, ErrorCategory.UNSUPPORTED_PARAMETER, msg
            return None, ErrorCategory.INVALID_PARAMETER_TYPE, msg

    def execute(
        self,
        tool_call: ToolCall,
        *,
        model_name: str | None = None,
        user_query: str | None = None,
    ) -> ToolResult:
        """Validate and run a proposed tool call; always returns a ToolResult."""
        start = time.perf_counter()
        tool = self._tools.get(tool_call.name)

        # 1) Unknown / hallucinated function.
        if tool is None:
            result = ToolResult(
                tool_name=tool_call.name,
                ok=False,
                error_category=ErrorCategory.UNKNOWN_FUNCTION,
                error_message=(
                    f"Unknown function '{tool_call.name}'. " f"Allowed: {', '.join(self.names())}."
                ),
            )
            self._log(result, tool_call, model_name, user_query, start)
            return result

        # 2) Argument validation.
        model, category, msg = self._validate_arguments(tool, tool_call.arguments)
        if model is None:
            result = ToolResult(
                tool_name=tool.name,
                ok=False,
                error_category=category,
                error_message=msg,
            )
            self._log(result, tool_call, model_name, user_query, start)
            return result

        # 3) Execute the approved function.
        kwargs = model.model_dump()
        try:
            data = tool.func(**kwargs)
            latency_ms = (time.perf_counter() - start) * 1000
            row_count = len(data) if isinstance(data, list) else None
            # Functions may return a (data, normalized_args) tuple to surface the
            # arguments actually used after synonym resolution.
            normalized = {}
            if isinstance(data, dict) and "_normalized_arguments" in data:
                normalized = data.get("_normalized_arguments", {})
            result = ToolResult(
                tool_name=tool.name,
                ok=True,
                data=data,
                normalized_arguments=normalized,
                row_count=row_count,
                latency_ms=latency_ms,
            )
        except ValueNotAllowedError as exc:
            result = ToolResult(
                tool_name=tool.name,
                ok=False,
                error_category=ErrorCategory.INVALID_ENUM_VALUE,
                error_message=str(exc),
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Execution failed for tool '%s'", tool.name)
            result = ToolResult(
                tool_name=tool.name,
                ok=False,
                error_category=ErrorCategory.EXECUTION_FAILURE,
                error_message=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        self._log(result, tool_call, model_name, user_query, start)
        return result

    # ── Logging ─────────────────────────────────────────────────────────────

    def _log(
        self,
        result: ToolResult,
        tool_call: ToolCall,
        model_name: str | None,
        user_query: str | None,
        start: float,
    ) -> None:
        """Append an audit record. Never writes secrets."""
        entry = FunctionCallLog(
            model_name=model_name,
            user_query=user_query,
            selected_function=tool_call.name,
            arguments=tool_call.arguments,
            normalized_arguments=result.normalized_arguments,
            validation_ok=result.error_category
            not in {
                ErrorCategory.UNKNOWN_FUNCTION,
                ErrorCategory.MISSING_REQUIRED_PARAMETER,
                ErrorCategory.UNSUPPORTED_PARAMETER,
                ErrorCategory.INVALID_PARAMETER_TYPE,
            },
            execution_ok=result.ok,
            result_summary=result.summary(),
            row_count=result.row_count,
            latency_ms=(
                result.latency_ms
                if result.latency_ms is not None
                else (time.perf_counter() - start) * 1000
            ),
            error_category=result.error_category,
        )
        try:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
            with self._log_file.open("a", encoding="utf-8") as fh:
                fh.write(entry.to_jsonl() + "\n")
        except OSError:  # pragma: no cover - logging must not crash calls
            logger.warning("Could not write function-call log.")
