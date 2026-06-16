"""Evaluation corpus (multi-step aware).

~60 natural-language queries across the five question categories and four
complexity levels, with paraphrase groups for H2.

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

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import PROJECT_ROOT

EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"

# Canonical normalized values for facility 124851 (so expected args are stable).
W = "IfcWindow"
D = "IfcDoor"
H = "IfcSpaceHeater"
EG = "Erdgeschoss"
UG = "Untergeschoss"
OG1 = "1. Obergeschoss"
OG2 = "2. Obergeschoss"
DG = "Dachgeschoss"


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

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


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
]


def get_corpus() -> list[EvalQuery]:
    """Return the full evaluation corpus."""
    return list(CORPUS)


def corpus_dataframe() -> pd.DataFrame:
    """Corpus as a DataFrame (lists/dicts kept as objects)."""
    return pd.DataFrame([q.to_row() for q in CORPUS])


def export_corpus(out_dir: Path = EVAL_DIR) -> Path:
    """Write the corpus to CSV for inspection; return the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "corpus.csv"
    corpus_dataframe().to_csv(path, index=False, encoding="utf-8")
    return path


if __name__ == "__main__":
    p = export_corpus()
    df = corpus_dataframe()
    print(f"Corpus: {len(df)} queries across {df['paraphrase_group_id'].nunique()} groups")
    print(df.groupby("complexity_level").size().to_string())
    print(df.groupby("category").size().to_string())
    print("Written to", p)
