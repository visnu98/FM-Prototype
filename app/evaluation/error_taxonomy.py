"""Error taxonomy and classification (Phase 13).

Every query that is not fully correct (correct function + parameters +
execution + final answer) is classified into exactly one category. The first
five categories mirror registry-level failures; the rest are answer/selection
level and are derived during evaluation.
"""

from __future__ import annotations

from enum import StrEnum

from app.tools.models import ErrorCategory


class EvalErrorCategory(StrEnum):
    """The nine evaluation error categories (plus NONE for correct queries)."""

    NONE = "none"
    WRONG_FUNCTION_SELECTED = "wrong_function_selected"
    MISSING_REQUIRED_PARAMETER = "missing_required_parameter"
    INCORRECT_PARAMETER_VALUE = "incorrect_parameter_value"
    HALLUCINATED_OR_UNSUPPORTED_FUNCTION = "hallucinated_or_unsupported_function"
    EXECUTION_OR_RUNTIME_FAILURE = "execution_or_runtime_failure"
    CORRECT_CALL_INCORRECT_FINAL_RESPONSE = "correct_call_incorrect_final_response"
    AMBIGUITY_OR_UNDERSPECIFIED_QUERY = "ambiguity_or_underspecified_query"
    NO_TOOL_CALL_WHEN_TOOL_REQUIRED = "no_tool_call_when_tool_required"
    INVALID_JSON_OR_SCHEMA_ERROR = "invalid_json_or_schema_error"


# Map registry-level categories to evaluation categories.
_REGISTRY_MAP = {
    ErrorCategory.UNKNOWN_FUNCTION: EvalErrorCategory.HALLUCINATED_OR_UNSUPPORTED_FUNCTION,
    ErrorCategory.MISSING_REQUIRED_PARAMETER: EvalErrorCategory.MISSING_REQUIRED_PARAMETER,
    ErrorCategory.UNSUPPORTED_PARAMETER: EvalErrorCategory.INCORRECT_PARAMETER_VALUE,
    ErrorCategory.INVALID_PARAMETER_TYPE: EvalErrorCategory.INCORRECT_PARAMETER_VALUE,
    ErrorCategory.INVALID_ENUM_VALUE: EvalErrorCategory.INCORRECT_PARAMETER_VALUE,
    ErrorCategory.EXECUTION_FAILURE: EvalErrorCategory.EXECUTION_OR_RUNTIME_FAILURE,
    ErrorCategory.INVALID_JSON_OR_SCHEMA: EvalErrorCategory.INVALID_JSON_OR_SCHEMA_ERROR,
}


def classify_error(
    *,
    made_tool_call: bool,
    registry_error: ErrorCategory,
    function_correct: bool,
    parameters_correct: bool,
    execution_success: bool,
    answer_correct: bool,
) -> EvalErrorCategory:
    """Assign one evaluation error category using a fixed priority order."""
    # Fully correct → no error.
    if function_correct and parameters_correct and execution_success and answer_correct:
        return EvalErrorCategory.NONE

    # 1) The model did not call any tool although one was required.
    if not made_tool_call:
        if registry_error is ErrorCategory.INVALID_JSON_OR_SCHEMA:
            return EvalErrorCategory.INVALID_JSON_OR_SCHEMA_ERROR
        return EvalErrorCategory.NO_TOOL_CALL_WHEN_TOOL_REQUIRED

    # 2) Registry rejected or failed the call → map directly.
    if registry_error is not ErrorCategory.NONE:
        return _REGISTRY_MAP.get(registry_error, EvalErrorCategory.EXECUTION_OR_RUNTIME_FAILURE)

    # 3) Call executed, but the wrong function was chosen.
    if not function_correct:
        return EvalErrorCategory.WRONG_FUNCTION_SELECTED

    # 4) Right function, wrong parameter value(s).
    if not parameters_correct:
        return EvalErrorCategory.INCORRECT_PARAMETER_VALUE

    # 5) Execution failed despite a valid-looking call.
    if not execution_success:
        return EvalErrorCategory.EXECUTION_OR_RUNTIME_FAILURE

    # 6) Correct call + result, but the final natural-language answer is wrong.
    if not answer_correct:
        return EvalErrorCategory.CORRECT_CALL_INCORRECT_FINAL_RESPONSE

    return EvalErrorCategory.NONE


ALL_CATEGORIES: tuple[EvalErrorCategory, ...] = tuple(EvalErrorCategory)
