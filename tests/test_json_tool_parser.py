"""Tests for the strict JSON tool-call parser (no network)."""

from __future__ import annotations

import pytest

from app.llm.json_tool_parser import ToolCallParseError, parse_tool_call


def test_plain_json() -> None:
    tc = parse_tool_call('{"name": "count_components", "arguments": {"component_type": "windows"}}')
    assert tc.name == "count_components"
    assert tc.arguments == {"component_type": "windows"}


def test_fenced_json() -> None:
    text = 'Sure!\n```json\n{"name": "list_queryable_floors", "arguments": {}}\n```'
    tc = parse_tool_call(text)
    assert tc.name == "list_queryable_floors"
    assert tc.arguments == {}


def test_openai_style_function_wrapper() -> None:
    text = '{"function": {"name": "count_components", "arguments": "{\\"component_type\\": \\"doors\\"}"}}'
    tc = parse_tool_call(text)
    assert tc.name == "count_components"
    assert tc.arguments == {"component_type": "doors"}


def test_missing_name_raises() -> None:
    with pytest.raises(ToolCallParseError):
        parse_tool_call('{"arguments": {"x": 1}}')


def test_garbage_raises() -> None:
    with pytest.raises(ToolCallParseError):
        parse_tool_call("this is not json at all")


def test_empty_raises() -> None:
    with pytest.raises(ToolCallParseError):
        parse_tool_call("   ")
