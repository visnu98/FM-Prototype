"""Tests for the tool registry validation + execution (no database)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from app.tools.models import ErrorCategory, ToolCall
from app.tools.registry import ToolRegistry, ValueNotAllowedError


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int
    label: str | None = None


def _add(n: int, label: str | None = None) -> dict[str, object]:
    return {"echo": n, "label": label}


def _make_registry(tmp_path) -> ToolRegistry:  # type: ignore[no-untyped-def]
    reg = ToolRegistry(log_file=tmp_path / "calls.jsonl")
    reg.register(name="add", description="d", func=_add, args_model=_Args)
    return reg


def test_successful_call(tmp_path) -> None:  # type: ignore[no-untyped-def]
    reg = _make_registry(tmp_path)
    res = reg.execute(ToolCall(name="add", arguments={"n": 5}))
    assert res.ok
    assert res.data == {"echo": 5, "label": None}
    assert res.error_category is ErrorCategory.NONE


def test_unknown_function(tmp_path) -> None:  # type: ignore[no-untyped-def]
    reg = _make_registry(tmp_path)
    res = reg.execute(ToolCall(name="nope", arguments={}))
    assert not res.ok
    assert res.error_category is ErrorCategory.UNKNOWN_FUNCTION


def test_missing_required_parameter(tmp_path) -> None:  # type: ignore[no-untyped-def]
    reg = _make_registry(tmp_path)
    res = reg.execute(ToolCall(name="add", arguments={}))
    assert res.error_category is ErrorCategory.MISSING_REQUIRED_PARAMETER


def test_unsupported_parameter(tmp_path) -> None:  # type: ignore[no-untyped-def]
    reg = _make_registry(tmp_path)
    res = reg.execute(ToolCall(name="add", arguments={"n": 1, "extra": 2}))
    assert res.error_category is ErrorCategory.UNSUPPORTED_PARAMETER


def test_invalid_parameter_type(tmp_path) -> None:  # type: ignore[no-untyped-def]
    reg = _make_registry(tmp_path)
    res = reg.execute(ToolCall(name="add", arguments={"n": "not-an-int"}))
    assert res.error_category is ErrorCategory.INVALID_PARAMETER_TYPE


def test_value_not_allowed_is_enum_error(tmp_path) -> None:  # type: ignore[no-untyped-def]
    reg = ToolRegistry(log_file=tmp_path / "calls.jsonl")

    def _raises(n: int, label: str | None = None) -> None:
        raise ValueNotAllowedError("bad value", parameter="n")

    reg.register(name="boom", description="d", func=_raises, args_model=_Args)
    res = reg.execute(ToolCall(name="boom", arguments={"n": 1}))
    assert res.error_category is ErrorCategory.INVALID_ENUM_VALUE


def test_execution_failure(tmp_path) -> None:  # type: ignore[no-untyped-def]
    reg = ToolRegistry(log_file=tmp_path / "calls.jsonl")

    def _crash(n: int, label: str | None = None) -> None:
        raise RuntimeError("kaboom")

    reg.register(name="crash", description="d", func=_crash, args_model=_Args)
    res = reg.execute(ToolCall(name="crash", arguments={"n": 1}))
    assert res.error_category is ErrorCategory.EXECUTION_FAILURE


def test_call_is_logged(tmp_path) -> None:  # type: ignore[no-untyped-def]
    log_file = tmp_path / "calls.jsonl"
    reg = ToolRegistry(log_file=log_file)
    reg.register(name="add", description="d", func=_add, args_model=_Args)
    reg.execute(ToolCall(name="add", arguments={"n": 2}), model_name="m", user_query="q")
    assert log_file.exists()
    assert '"selected_function":"add"' in log_file.read_text(encoding="utf-8")


def test_openai_schema_shape(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from app.tools.models import ToolParameter

    reg = ToolRegistry(log_file=tmp_path / "calls.jsonl")
    reg.register(
        name="add",
        description="d",
        func=_add,
        args_model=_Args,
        parameters=[ToolParameter(name="n", type="integer", required=True)],
    )
    schema = reg.openai_schemas()[0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "add"
    assert schema["function"]["parameters"]["required"] == ["n"]
