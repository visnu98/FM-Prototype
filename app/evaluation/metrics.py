"""Evaluation metrics (Phase 12).

Computes, per query, the metrics the thesis reports:
- answer correctness (exact for counts, tolerance for areas, set recall for lists),
- function-selection correctness,
- parameter accuracy (per-parameter and fully-correct-call),
- execution success,
- latency (already measured by the pipeline).

The functions take primitive inputs so they are unit-testable without a DB/LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.evaluation.error_taxonomy import EvalErrorCategory, classify_error
from app.tools.models import ErrorCategory

# Aliases accepted when matching a canonical floor name in a free-text answer.
_FLOOR_ALIASES: dict[str, set[str]] = {
    "Untergeschoss": {"untergeschoss", "basement", "ug", "lower ground"},
    "Erdgeschoss": {"erdgeschoss", "ground floor", "ground", "eg"},
    "1. Obergeschoss": {
        "1. obergeschoss",
        "first floor",
        "first upper",
        "level one",
        "level 1",
        "1og",
        "1. og",
    },
    "2. Obergeschoss": {
        "2. obergeschoss",
        "second floor",
        "second upper",
        "level two",
        "level 2",
        "2og",
        "2. og",
    },
    "Dachgeschoss": {"dachgeschoss", "attic", "top floor", "roof", "dg"},
}


# ── Argument flattening / comparison ─────────────────────────────────────────


def flatten_normalized_args(normalized: dict[str, Any]) -> dict[str, Any]:
    """Turn nested {param: {normalized: v}} into {param: v}."""
    out: dict[str, Any] = {}
    for key, val in normalized.items():
        if isinstance(val, dict) and "normalized" in val:
            out[key] = val["normalized"]
        else:
            out[key] = val
    return out


def parameter_accuracy(
    expected: dict[str, Any], actual_normalized: dict[str, Any]
) -> tuple[bool, float]:
    """Return (all_correct, per-parameter accuracy) over the expected keys."""
    if not expected:
        # No parameters expected: correct iff the call also passed none meaningful.
        return True, 1.0
    correct = 0
    for k, v in expected.items():
        if str(actual_normalized.get(k)) == str(v):
            correct += 1
    return correct == len(expected), correct / len(expected)


# ── Answer correctness ───────────────────────────────────────────────────────

_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def extract_numbers(text: str) -> list[float]:
    """Extract numeric values from free text (commas treated as thousands)."""
    values: list[float] = []
    for tok in _NUMBER_RE.findall(text):
        cleaned = tok.replace(",", "").rstrip(".")
        try:
            values.append(float(cleaned))
        except ValueError:
            continue
    return values


def _number_present(expected: float, answer: str, tol: float) -> bool:
    nums = extract_numbers(answer)
    return any(abs(n - expected) <= tol for n in nums)


def _string_present(expected: str, answer: str) -> bool:
    low = answer.lower()
    if expected.lower() in low:
        return True
    # Accept known aliases (e.g. floor names phrased differently).
    for canonical, aliases in _FLOOR_ALIASES.items():
        if expected == canonical and any(a in low for a in aliases):
            return True
    return False


def answer_correctness(
    answer_kind: str,
    expected_values: list[Any],
    tolerance: float,
    final_answer: str,
    execution_success: bool,
) -> tuple[bool, float | None]:
    """Return (answer_correct, list_recall|None) for one query."""
    if not final_answer:
        return False, None
    if answer_kind == "number":
        ok = all(
            _number_present(float(v), final_answer, tolerance)
            for v in expected_values
            if v is not None
        )
        return ok, None
    if answer_kind == "string":
        ok = all(_string_present(str(v), final_answer) for v in expected_values if v is not None)
        return ok, None
    if answer_kind == "list":
        present = sum(1 for v in expected_values if _string_present(str(v), final_answer))
        recall = present / len(expected_values) if expected_values else 1.0
        return recall == 1.0, recall
    # "any"/"result_count-any": cannot verify specific text -> grounded if it ran.
    return execution_success, None


# ── Per-query metrics record ─────────────────────────────────────────────────


@dataclass
class QueryMetrics:
    query_id: str
    model: str
    category: str
    complexity_level: str
    paraphrase_group_id: str
    is_standard_wording: bool

    expected_function: str
    actual_function: str | None
    expected_parameters: dict[str, Any]
    actual_parameters: dict[str, Any]

    function_correct: bool
    parameters_correct: bool
    parameter_level_accuracy: float
    execution_success: bool
    fully_correct_call: bool
    answer_correct: bool
    list_recall: float | None = None

    latency_total: float = 0.0
    latency_tool_call: float = 0.0
    latency_sql: float = 0.0
    latency_final_answer: float = 0.0

    error_category: str = EvalErrorCategory.NONE.value
    final_answer: str = ""
    expected_answer_values: list[Any] = field(default_factory=list)


def evaluate_query(
    *,
    gt: Any,  # GroundTruth
    model: str,
    made_tool_call: bool,
    actual_function: str | None,
    actual_normalized_arguments: dict[str, Any],
    registry_error: ErrorCategory,
    execution_success: bool,
    final_answer: str,
    latency_total: float,
    latency_tool_call: float,
    latency_sql: float,
    latency_final_answer: float,
) -> QueryMetrics:
    """Compute all metrics for one (query, model) outcome."""
    actual_params = flatten_normalized_args(actual_normalized_arguments)
    function_correct = actual_function == gt.expected_function
    params_ok, param_acc = parameter_accuracy(gt.expected_arguments, actual_params)
    # Parameters only count when the right function was chosen.
    parameters_correct = function_correct and params_ok

    answer_correct, list_recall = answer_correctness(
        gt.answer_kind,
        gt.expected_answer_values,
        gt.answer_tolerance,
        final_answer,
        execution_success,
    )
    fully_correct_call = function_correct and parameters_correct and execution_success

    error_category = classify_error(
        made_tool_call=made_tool_call,
        registry_error=registry_error,
        function_correct=function_correct,
        parameters_correct=parameters_correct,
        execution_success=execution_success,
        answer_correct=answer_correct,
    )

    return QueryMetrics(
        query_id=gt.query_id,
        model=model,
        category=gt.category,
        complexity_level=gt.complexity_level,
        paraphrase_group_id=gt.paraphrase_group_id,
        is_standard_wording=gt.is_standard_wording,
        expected_function=gt.expected_function,
        actual_function=actual_function,
        expected_parameters=gt.expected_arguments,
        actual_parameters=actual_params,
        function_correct=function_correct,
        parameters_correct=parameters_correct,
        parameter_level_accuracy=param_acc,
        execution_success=execution_success,
        fully_correct_call=fully_correct_call,
        answer_correct=answer_correct,
        list_recall=list_recall,
        latency_total=latency_total,
        latency_tool_call=latency_tool_call,
        latency_sql=latency_sql,
        latency_final_answer=latency_final_answer,
        error_category=error_category.value,
        final_answer=final_answer,
        expected_answer_values=gt.expected_answer_values,
    )
