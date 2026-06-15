"""Tests for the evaluation logic (metrics, taxonomy, statistics). No DB/LLM."""

from __future__ import annotations

import pandas as pd
import pytest

from app.evaluation import metrics as M
from app.evaluation.error_taxonomy import EvalErrorCategory, classify_error
from app.evaluation.statistics import evaluate_h1, evaluate_h3, evaluate_h4
from app.tools.models import ErrorCategory

# ── metrics: number / string / list extraction ───────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        ("There are 52 windows.", [52.0]),
        ("Total area is 1,787.95 m².", [1787.95]),
        ("first: 79, second: 52", [79.0, 52.0]),
    ],
)
def test_extract_numbers(text, expected) -> None:  # type: ignore[no-untyped-def]
    assert M.extract_numbers(text) == expected


def test_answer_number_with_tolerance() -> None:
    ok, _ = M.answer_correctness("number", [1787.953], 1.0, "About 1787.95 m².", True)
    assert ok
    bad, _ = M.answer_correctness("number", [1787.953], 1.0, "About 1700 m².", True)
    assert not bad


def test_answer_string_with_floor_alias() -> None:
    ok, _ = M.answer_correctness("string", ["1. Obergeschoss"], 0, "the first floor wins", True)
    assert ok


def test_answer_list_recall() -> None:
    ok, recall = M.answer_correctness(
        "list",
        ["Untergeschoss", "Erdgeschoss"],
        0,
        "Floors: Untergeschoss and Erdgeschoss.",
        True,
    )
    assert ok and recall == 1.0
    partial_ok, partial_recall = M.answer_correctness(
        "list", ["Untergeschoss", "Erdgeschoss"], 0, "Only Untergeschoss.", True
    )
    assert not partial_ok and partial_recall == 0.5


def test_parameter_accuracy() -> None:
    ok, acc = M.parameter_accuracy(
        {"component_type": "IfcWindow", "floor": "2. Obergeschoss"},
        {"component_type": "IfcWindow", "floor": "2. Obergeschoss"},
    )
    assert ok and acc == 1.0
    ok2, acc2 = M.parameter_accuracy(
        {"component_type": "IfcWindow", "floor": "2. Obergeschoss"},
        {"component_type": "IfcWindow", "floor": "Erdgeschoss"},
    )
    assert not ok2 and acc2 == 0.5


def test_flatten_normalized_args() -> None:
    nested = {
        "component_type": {"original": "windows", "normalized": "IfcWindow", "method": "synonym"}
    }
    assert M.flatten_normalized_args(nested) == {"component_type": "IfcWindow"}


# ── error taxonomy ────────────────────────────────────────────────────────────


def test_classify_correct_is_none() -> None:
    cat = classify_error(
        made_tool_call=True,
        registry_error=ErrorCategory.NONE,
        function_correct=True,
        parameters_correct=True,
        execution_success=True,
        answer_correct=True,
    )
    assert cat is EvalErrorCategory.NONE


def test_classify_no_tool_call() -> None:
    cat = classify_error(
        made_tool_call=False,
        registry_error=ErrorCategory.NONE,
        function_correct=False,
        parameters_correct=False,
        execution_success=False,
        answer_correct=False,
    )
    assert cat is EvalErrorCategory.NO_TOOL_CALL_WHEN_TOOL_REQUIRED


def test_classify_wrong_function() -> None:
    cat = classify_error(
        made_tool_call=True,
        registry_error=ErrorCategory.NONE,
        function_correct=False,
        parameters_correct=False,
        execution_success=True,
        answer_correct=False,
    )
    assert cat is EvalErrorCategory.WRONG_FUNCTION_SELECTED


def test_classify_correct_call_wrong_answer() -> None:
    cat = classify_error(
        made_tool_call=True,
        registry_error=ErrorCategory.NONE,
        function_correct=True,
        parameters_correct=True,
        execution_success=True,
        answer_correct=False,
    )
    assert cat is EvalErrorCategory.CORRECT_CALL_INCORRECT_FINAL_RESPONSE


def test_classify_hallucinated() -> None:
    cat = classify_error(
        made_tool_call=True,
        registry_error=ErrorCategory.UNKNOWN_FUNCTION,
        function_correct=False,
        parameters_correct=False,
        execution_success=False,
        answer_correct=False,
    )
    assert cat is EvalErrorCategory.HALLUCINATED_OR_UNSUPPORTED_FUNCTION


# ── statistics on synthetic data ──────────────────────────────────────────────


def _synthetic_metrics() -> pd.DataFrame:
    rows = []
    # Model A: perfect; Model B: fails the L4 queries.
    for level in ["L1", "L2", "L3", "L4"]:
        for i in range(5):
            qid = f"{level}_{i}"
            rows.append(
                {
                    "query_id": qid,
                    "model": "A",
                    "complexity_level": level,
                    "fully_correct_call": True,
                    "function_correct": True,
                    "parameters_correct": True,
                    "answer_correct": True,
                    "execution_success": True,
                    "latency_total": 500.0,
                    "is_standard_wording": i == 0,
                    "paraphrase_group_id": f"g_{level}",
                    "category": "counting",
                }
            )
            b_ok = level != "L4"
            rows.append(
                {
                    "query_id": qid,
                    "model": "B",
                    "complexity_level": level,
                    "fully_correct_call": b_ok,
                    "function_correct": b_ok,
                    "parameters_correct": b_ok,
                    "answer_correct": b_ok,
                    "execution_success": True,
                    "latency_total": 200.0,
                    "is_standard_wording": i == 0,
                    "paraphrase_group_id": f"g_{level}",
                    "category": "counting",
                }
            )
    return pd.DataFrame(rows)


def test_h1_supported_when_high() -> None:
    df = _synthetic_metrics()
    res = evaluate_h1(df[df["model"] == "A"])
    assert res.supported is True
    assert res.result_value["overall_rate"] == 1.0


def test_h3_detects_difference() -> None:
    df = _synthetic_metrics()
    res = evaluate_h3(df, "A", "B")
    # A is perfect, B fails all L4 -> discordant pairs exist.
    assert res.result_value["correct_rate_a"] > res.result_value["correct_rate_b"]


def test_h4_detects_decrease() -> None:
    df = _synthetic_metrics()
    res = evaluate_h4(df[df["model"] == "B"])
    rates = res.result_value["rate_by_level"]
    assert rates["L1"] >= rates["L4"]
