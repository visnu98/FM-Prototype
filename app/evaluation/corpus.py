"""Evaluation corpus (multi-step aware).

78 natural-language queries across the five question categories and four
complexity levels. Organised into 26 paraphrase groups of exactly three
wordings each (one ``is_standard_wording`` plus two rewordings) so the
paraphrase-robustness hypothesis (H2) is tested on balanced groups.

This module is the *authoring* source for the questions and their
ground-truth specs. It is consumed only when (re)generating the committed
committed ground truth (``app.evaluation.ground_truth``); the evaluation runner
reads the frozen ``data/evaluation/ground_truth.json`` and never recomputes.

Because the assistant answers by *composing* atomic tools (there is no bespoke
function per question), each query records:

- ``expected_functions``: the primitive function(s) a correct solution must use
  (checked as a subset of the functions actually called — extra helper calls
  such as ``list_queryable_floors`` are allowed),
- ``expected_arguments``: for single-call (atomic) queries only, the normalized
  arguments that one call should use (used for the parameter-accuracy metric),
- ``gt_calls``: the deterministic registry calls used to compute ground truth,
- ``answer_spec``: how to turn the ground-truth call results into the expected
  answer value(s).

The expected *values* are not hard-coded here; ground truth runs ``gt_calls``
against the database (parameterised SQL, never the LLM).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.config import PROJECT_ROOT

EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"

# Canonical normalized values for facility 124851 (so expected args are stable).
W = "IfcWindow"
D = "IfcDoor"
H = "IfcSpaceHeater"
EG = "Erdgeschoss"
OG1 = "1. Obergeschoss"
OG2 = "2. Obergeschoss"


@dataclass(frozen=True)
class EvalQuery:
    """One evaluation query and its (multi-step aware) ground-truth spec."""

    query_id: str
    query_text: str
    category: str  # metadata | attribute | counting | aggregation | multistep
    complexity_level: str  # L1 | L2 | L3 | L4
    paraphrase_group_id: str
    is_standard_wording: bool
    expected_functions: list[str]
    gt_calls: list[dict[str, Any]]
    answer_spec: dict[str, Any]
    expected_arguments: dict[str, Any] = field(default_factory=dict)


def _call(function: str, **arguments: Any) -> dict[str, Any]:
    return {"function": function, "arguments": arguments}


def _q(
    qid: str,
    text: str,
    category: str,
    level: str,
    group: str,
    standard: bool,
    expected_functions: list[str],
    gt_calls: list[dict[str, Any]],
    answer_spec: dict[str, Any],
    expected_arguments: dict[str, Any] | None = None,
) -> EvalQuery:
    return EvalQuery(
        query_id=qid,
        query_text=text,
        category=category,
        complexity_level=level,
        paraphrase_group_id=group,
        is_standard_wording=standard,
        expected_functions=expected_functions,
        gt_calls=gt_calls,
        answer_spec=answer_spec,
        expected_arguments=expected_arguments or {},
    )


def _count(qid, text, group, standard, *, ctype, floor=None):  # type: ignore[no-untyped-def]
    """Shorthand for an atomic count query."""
    args = {"component_type": ctype} | ({"floor": floor} if floor else {})
    return _q(
        qid,
        text,
        "counting",
        "L2",
        group,
        standard,
        ["count_components"],
        [_call("count_components", **args)],
        {"kind": "number", "field": "count"},
        expected_arguments=args,
    )


def _area(qid, text, group, standard, *, ctype, floor=None):  # type: ignore[no-untyped-def]
    """Shorthand for an atomic component-area query."""
    args = {"component_type": ctype} | ({"floor": floor} if floor else {})
    return _q(
        qid,
        text,
        "aggregation",
        "L3",
        group,
        standard,
        ["calculate_total_component_area"],
        [_call("calculate_total_component_area", **args)],
        {"kind": "number", "field": "total_area", "tol": 1.0},
        expected_arguments=args,
    )


CORPUS: list[EvalQuery] = [
    # ── Metadata (L1) ────────────────────────────────────────────────────────
    _q(
        "q01",
        "Which floors can I query for this building?",
        "metadata",
        "L1",
        "g01",
        True,
        ["list_queryable_floors"],
        [_call("list_queryable_floors")],
        {"kind": "list", "field": "name"},
    ),
    _q(
        "q02",
        "What floors are available?",
        "metadata",
        "L1",
        "g01",
        False,
        ["list_queryable_floors"],
        [_call("list_queryable_floors")],
        {"kind": "list", "field": "name"},
    ),
    _q(
        "q03",
        "Show me the storeys in this facility.",
        "metadata",
        "L1",
        "g01",
        False,
        ["list_queryable_floors"],
        [_call("list_queryable_floors")],
        {"kind": "list", "field": "name"},
    ),
    _q(
        "q04",
        "What component types are available?",
        "metadata",
        "L1",
        "g02",
        True,
        ["list_queryable_component_types"],
        [_call("list_queryable_component_types")],
        {"kind": "any"},
    ),
    _q(
        "q05",
        "Which BIM element types can I ask about?",
        "metadata",
        "L1",
        "g02",
        False,
        ["list_queryable_component_types"],
        [_call("list_queryable_component_types")],
        {"kind": "any"},
    ),
    _q(
        "q06",
        "What asset component categories exist?",
        "metadata",
        "L1",
        "g02",
        False,
        ["list_queryable_component_types"],
        [_call("list_queryable_component_types")],
        {"kind": "any"},
    ),
    _q(
        "q07",
        "What can this assistant help me with?",
        "metadata",
        "L1",
        "g03",
        True,
        ["get_database_capabilities"],
        [_call("get_database_capabilities")],
        {"kind": "any"},
    ),
    _q(
        "q08",
        "What kinds of questions can you answer?",
        "metadata",
        "L1",
        "g03",
        False,
        ["get_database_capabilities"],
        [_call("get_database_capabilities")],
        {"kind": "any"},
    ),
    # ── Counting (L2) ────────────────────────────────────────────────────────
    _count("q09", "How many windows are on the second floor?", "g04", True, ctype=W, floor=OG2),
    _count("q10", "Count the windows on level two.", "g04", False, ctype=W, floor=OG2),
    _count(
        "q11",
        "What is the number of window elements on the second upper floor?",
        "g04",
        False,
        ctype=W,
        floor=OG2,
    ),
    _count("q12", "How many doors does the building have in total?", "g05", True, ctype=D),
    _count("q13", "Count all doors in the facility.", "g05", False, ctype=D),
    _count("q14", "What is the total number of door components?", "g05", False, ctype=D),
    _count("q15", "How many windows are on the ground floor?", "g06", True, ctype=W, floor=EG),
    _count("q16", "Count the windows on the Erdgeschoss.", "g06", False, ctype=W, floor=EG),
    _count(
        "q17", "How many space heaters are on the first floor?", "g07", True, ctype=H, floor=OG1
    ),
    _count("q18", "Count the radiators on the 1. Obergeschoss.", "g07", False, ctype=H, floor=OG1),
    _count("q19", "How many windows are there in total?", "g08", True, ctype=W),
    _count("q20", "What is the total window count for the building?", "g08", False, ctype=W),
    _count("q21", "How many columns does the building have?", "g09", True, ctype="IfcColumn"),
    _count("q22", "Count all the columns in the facility.", "g09", False, ctype="IfcColumn"),
    _count("q23", "How many doors are on the second floor?", "g10", True, ctype=D, floor=OG2),
    _count("q24", "Count the doors on level two.", "g10", False, ctype=D, floor=OG2),
    _count("q25", "How many windows are on the first floor?", "g11", True, ctype=W, floor=OG1),
    _count("q26", "Count the windows on the 1. Obergeschoss.", "g11", False, ctype=W, floor=OG1),
    _count("q27", "How many sensors does the building have?", "g12", True, ctype="IfcSensor"),
    _count("q28", "Count all sensors in the facility.", "g12", False, ctype="IfcSensor"),
    # ── Attribute retrieval (L2) ─────────────────────────────────────────────
    _q(
        "q29",
        "What are the heights and widths of the windows on the second floor?",
        "attribute",
        "L2",
        "g13",
        True,
        ["get_component_attributes"],
        [
            _call(
                "get_component_attributes",
                component_type=W,
                floor=OG2,
                attributes=["height", "width"],
            )
        ],
        {"kind": "result_count"},
        expected_arguments={"component_type": W, "floor": OG2},
    ),
    _q(
        "q30",
        "Show me the dimensions of windows on level two.",
        "attribute",
        "L2",
        "g13",
        False,
        ["get_component_attributes"],
        [
            _call(
                "get_component_attributes",
                component_type=W,
                floor=OG2,
                attributes=["height", "width"],
            )
        ],
        {"kind": "result_count"},
        expected_arguments={"component_type": W, "floor": OG2},
    ),
    _q(
        "q31",
        "What are the dimensions of the doors on the ground floor?",
        "attribute",
        "L2",
        "g14",
        True,
        ["get_component_attributes"],
        [
            _call(
                "get_component_attributes",
                component_type=D,
                floor=EG,
                attributes=["height", "width"],
            )
        ],
        {"kind": "result_count"},
        expected_arguments={"component_type": D, "floor": EG},
    ),
    _q(
        "q32",
        "Show me the height and width of doors in the Erdgeschoss.",
        "attribute",
        "L2",
        "g14",
        False,
        ["get_component_attributes"],
        [
            _call(
                "get_component_attributes",
                component_type=D,
                floor=EG,
                attributes=["height", "width"],
            )
        ],
        {"kind": "result_count"},
        expected_arguments={"component_type": D, "floor": EG},
    ),
    # ── Aggregation (L3): component area ─────────────────────────────────────
    _area("q33", "What is the total window area in the building?", "g15", True, ctype=W),
    _area("q34", "Calculate the overall area of all windows.", "g15", False, ctype=W),
    _area(
        "q35", "What is the total window area on the second floor?", "g16", True, ctype=W, floor=OG2
    ),
    _area("q36", "Calculate the window area on level two.", "g16", False, ctype=W, floor=OG2),
    _area("q37", "What is the total door area in the building?", "g17", True, ctype=D),
    _area("q38", "Calculate the combined area of all doors.", "g17", False, ctype=D),
    _area(
        "q39", "What is the total window area on the first floor?", "g18", True, ctype=W, floor=OG1
    ),
    _area("q40", "Calculate the window area on level one.", "g18", False, ctype=W, floor=OG1),
    # ── Aggregation (L3): floor area ─────────────────────────────────────────
    _q(
        "q41",
        "What is the floor area of the ground floor?",
        "aggregation",
        "L3",
        "g19",
        True,
        ["calculate_floor_area"],
        [_call("calculate_floor_area", floor=EG)],
        {"kind": "number", "field": "gross_area", "tol": 1.0},
        expected_arguments={"floor": EG},
    ),
    _q(
        "q42",
        "How big is the Erdgeschoss?",
        "aggregation",
        "L3",
        "g19",
        False,
        ["calculate_floor_area"],
        [_call("calculate_floor_area", floor=EG)],
        {"kind": "number", "field": "gross_area", "tol": 1.0},
        expected_arguments={"floor": EG},
    ),
    _q(
        "q43",
        "What is the floor area of the first floor?",
        "aggregation",
        "L3",
        "g20",
        True,
        ["calculate_floor_area"],
        [_call("calculate_floor_area", floor=OG1)],
        {"kind": "number", "field": "gross_area", "tol": 1.0},
        expected_arguments={"floor": OG1},
    ),
    _q(
        "q44",
        "How many square metres is the 1. Obergeschoss?",
        "aggregation",
        "L3",
        "g20",
        False,
        ["calculate_floor_area"],
        [_call("calculate_floor_area", floor=OG1)],
        {"kind": "number", "field": "gross_area", "tol": 1.0},
        expected_arguments={"floor": OG1},
    ),
    # ── Multi-step (L4): compare counts between floors ───────────────────────
    _q(
        "q45",
        "Are there more windows on the first or the second upper floor?",
        "multistep",
        "L4",
        "g21",
        True,
        ["count_components"],
        [
            _call("count_components", component_type=W, floor=OG1),
            _call("count_components", component_type=W, floor=OG2),
        ],
        {"kind": "compare_winner", "field": "count", "floor_key": "floor"},
    ),
    _q(
        "q46",
        "Compare window counts between the first and second floor.",
        "multistep",
        "L4",
        "g21",
        False,
        ["count_components"],
        [
            _call("count_components", component_type=W, floor=OG1),
            _call("count_components", component_type=W, floor=OG2),
        ],
        {"kind": "compare_winner", "field": "count", "floor_key": "floor"},
    ),
    _q(
        "q47",
        "Which has more windows, level one or level two?",
        "multistep",
        "L4",
        "g21",
        False,
        ["count_components"],
        [
            _call("count_components", component_type=W, floor=OG1),
            _call("count_components", component_type=W, floor=OG2),
        ],
        {"kind": "compare_winner", "field": "count", "floor_key": "floor"},
    ),
    _q(
        "q48",
        "Are there more doors on the ground floor or the second floor?",
        "multistep",
        "L4",
        "g22",
        True,
        ["count_components"],
        [
            _call("count_components", component_type=D, floor=EG),
            _call("count_components", component_type=D, floor=OG2),
        ],
        {"kind": "compare_winner", "field": "count", "floor_key": "floor"},
    ),
    _q(
        "q49",
        "Compare the number of doors between the Erdgeschoss and level two.",
        "multistep",
        "L4",
        "g22",
        False,
        ["count_components"],
        [
            _call("count_components", component_type=D, floor=EG),
            _call("count_components", component_type=D, floor=OG2),
        ],
        {"kind": "compare_winner", "field": "count", "floor_key": "floor"},
    ),
    # ── Multi-step (L4): compare areas between floors ────────────────────────
    _q(
        "q50",
        "Is the window area larger on the first or the second floor?",
        "multistep",
        "L4",
        "g23",
        True,
        ["calculate_total_component_area"],
        [
            _call("calculate_total_component_area", component_type=W, floor=OG1),
            _call("calculate_total_component_area", component_type=W, floor=OG2),
        ],
        {"kind": "compare_winner", "field": "total_area", "floor_key": "floor"},
    ),
    _q(
        "q51",
        "Compare the window area between level one and level two.",
        "multistep",
        "L4",
        "g23",
        False,
        ["calculate_total_component_area"],
        [
            _call("calculate_total_component_area", component_type=W, floor=OG1),
            _call("calculate_total_component_area", component_type=W, floor=OG2),
        ],
        {"kind": "compare_winner", "field": "total_area", "floor_key": "floor"},
    ),
    _q(
        "q52",
        "Which floor has more window surface, the first or second upper floor?",
        "multistep",
        "L4",
        "g23",
        False,
        ["calculate_total_component_area"],
        [
            _call("calculate_total_component_area", component_type=W, floor=OG1),
            _call("calculate_total_component_area", component_type=W, floor=OG2),
        ],
        {"kind": "compare_winner", "field": "total_area", "floor_key": "floor"},
    ),
    # ── Multi-step (L4): which floor has the largest component area ──────────
    _q(
        "q53",
        "Which floor has the largest total window area?",
        "multistep",
        "L4",
        "g24",
        True,
        ["calculate_area_by_floor"],
        [_call("calculate_area_by_floor", component_type=W)],
        {"kind": "max_floor", "by": "by_floor", "value": "total_area", "label": "floor"},
    ),
    _q(
        "q54",
        "Where is the biggest amount of window area?",
        "multistep",
        "L4",
        "g24",
        False,
        ["calculate_area_by_floor"],
        [_call("calculate_area_by_floor", component_type=W)],
        {"kind": "max_floor", "by": "by_floor", "value": "total_area", "label": "floor"},
    ),
    _q(
        "q55",
        "Which storey has the most glazing?",
        "multistep",
        "L4",
        "g24",
        False,
        ["calculate_area_by_floor"],
        [_call("calculate_area_by_floor", component_type=W)],
        {"kind": "max_floor", "by": "by_floor", "value": "total_area", "label": "floor"},
    ),
    _q(
        "q56",
        "Which floor has the largest total door area?",
        "multistep",
        "L4",
        "g25",
        True,
        ["calculate_area_by_floor"],
        [_call("calculate_area_by_floor", component_type=D)],
        {"kind": "max_floor", "by": "by_floor", "value": "total_area", "label": "floor"},
    ),
    _q(
        "q57",
        "On which storey is the door area the biggest?",
        "multistep",
        "L4",
        "g25",
        False,
        ["calculate_area_by_floor"],
        [_call("calculate_area_by_floor", component_type=D)],
        {"kind": "max_floor", "by": "by_floor", "value": "total_area", "label": "floor"},
    ),
    # ── Multi-step (L4): areas of two floors at once ─────────────────────────
    _q(
        "q58",
        "What is the floor area of the first and the second floor?",
        "multistep",
        "L4",
        "g26",
        True,
        ["calculate_floor_area"],
        [_call("calculate_floor_area", floor=OG1), _call("calculate_floor_area", floor=OG2)],
        {"kind": "values", "field": "gross_area", "tol": 1.0},
    ),
    _q(
        "q59",
        "Give me the area of both the first and second upper floor.",
        "multistep",
        "L4",
        "g26",
        False,
        ["calculate_floor_area"],
        [_call("calculate_floor_area", floor=OG1), _call("calculate_floor_area", floor=OG2)],
        {"kind": "values", "field": "gross_area", "tol": 1.0},
    ),
    # ── Third paraphrase per group (H2: every group has exactly 3 wordings) ───
    _q(
        "q60",
        "What are you able to do?",
        "metadata",
        "L1",
        "g03",
        False,
        ["get_database_capabilities"],
        [_call("get_database_capabilities")],
        {"kind": "any"},
    ),
    _count("q61", "What's the window count on the ground floor?", "g06", False, ctype=W, floor=EG),
    _count(
        "q62", "How many radiators are on the first upper floor?", "g07", False, ctype=H, floor=OG1
    ),
    _count("q63", "Across the whole building, how many windows are there?", "g08", False, ctype=W),
    _count(
        "q64",
        "What's the total number of columns in the building?",
        "g09",
        False,
        ctype="IfcColumn",
    ),
    _count(
        "q65", "What's the door count on the second upper floor?", "g10", False, ctype=D, floor=OG2
    ),
    _count("q66", "What's the window count on the first floor?", "g11", False, ctype=W, floor=OG1),
    _count("q67", "What's the total number of sensors installed?", "g12", False, ctype="IfcSensor"),
    _q(
        "q68",
        "List the height and width of each window on the second floor.",
        "attribute",
        "L2",
        "g13",
        False,
        ["get_component_attributes"],
        [
            _call(
                "get_component_attributes",
                component_type=W,
                floor=OG2,
                attributes=["height", "width"],
            )
        ],
        {"kind": "result_count"},
        expected_arguments={"component_type": W, "floor": OG2},
    ),
    _q(
        "q69",
        "List the height and width of the ground-floor doors.",
        "attribute",
        "L2",
        "g14",
        False,
        ["get_component_attributes"],
        [
            _call(
                "get_component_attributes",
                component_type=D,
                floor=EG,
                attributes=["height", "width"],
            )
        ],
        {"kind": "result_count"},
        expected_arguments={"component_type": D, "floor": EG},
    ),
    _area("q70", "Sum up the area of all windows in the building.", "g15", False, ctype=W),
    _area("q71", "Sum the window area on the second floor.", "g16", False, ctype=W, floor=OG2),
    _area("q72", "What is the combined door area across the building?", "g17", False, ctype=D),
    _area("q73", "Sum the window area on the first floor.", "g18", False, ctype=W, floor=OG1),
    _q(
        "q74",
        "How many square metres does the ground floor cover?",
        "aggregation",
        "L3",
        "g19",
        False,
        ["calculate_floor_area"],
        [_call("calculate_floor_area", floor=EG)],
        {"kind": "number", "field": "gross_area", "tol": 1.0},
        expected_arguments={"floor": EG},
    ),
    _q(
        "q75",
        "What is the size of the first upper floor in square metres?",
        "aggregation",
        "L3",
        "g20",
        False,
        ["calculate_floor_area"],
        [_call("calculate_floor_area", floor=OG1)],
        {"kind": "number", "field": "gross_area", "tol": 1.0},
        expected_arguments={"floor": OG1},
    ),
    _q(
        "q76",
        "Which has more doors, the ground floor or the second floor?",
        "multistep",
        "L4",
        "g22",
        False,
        ["count_components"],
        [
            _call("count_components", component_type=D, floor=EG),
            _call("count_components", component_type=D, floor=OG2),
        ],
        {"kind": "compare_winner", "field": "count", "floor_key": "floor"},
    ),
    _q(
        "q77",
        "Which floor has the most door area?",
        "multistep",
        "L4",
        "g25",
        False,
        ["calculate_area_by_floor"],
        [_call("calculate_area_by_floor", component_type=D)],
        {"kind": "max_floor", "by": "by_floor", "value": "total_area", "label": "floor"},
    ),
    _q(
        "q78",
        "What are the floor areas of the first and second upper floors?",
        "multistep",
        "L4",
        "g26",
        False,
        ["calculate_floor_area"],
        [_call("calculate_floor_area", floor=OG1), _call("calculate_floor_area", floor=OG2)],
        {"kind": "values", "field": "gross_area", "tol": 1.0},
    ),
]


def get_corpus() -> list[EvalQuery]:
    """Return the full evaluation corpus."""
    return list(CORPUS)


if __name__ == "__main__":
    from collections import Counter

    corpus = get_corpus()
    groups = {q.paraphrase_group_id for q in corpus}
    print(f"Corpus: {len(corpus)} queries across {len(groups)} paraphrase groups")
    for level, n in sorted(Counter(q.complexity_level for q in corpus).items()):
        print(f"  {level}: {n}")
    for cat, n in sorted(Counter(q.category for q in corpus).items()):
        print(f"  {cat}: {n}")
