"""Evaluation corpus (Phase 10).

~70 natural-language queries covering the five question categories and four
complexity levels, with paraphrase groups for H2. Each query records the
*expected* function and the *canonical normalized* arguments a correct call
should produce, plus an ``answer_spec`` describing how to extract the expected
answer value from the tool result (used to build ground truth in Phase 11).

Complexity levels:
- L1: simple metadata / no parameters
- L2: single function with one or more normalized parameters
- L3: aggregation / calculation
- L4: multi-step comparison, grouping or chained logic

The expected *values* are NOT hard-coded here; ground truth computes them by
running the expected function deterministically against the database.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import PROJECT_ROOT

EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"


@dataclass(frozen=True)
class EvalQuery:
    """One evaluation query and its ground-truth specification."""

    query_id: str
    query_text: str
    category: str  # metadata | attribute | counting | aggregation | multistep
    complexity_level: str  # L1 | L2 | L3 | L4
    paraphrase_group_id: str
    is_standard_wording: bool
    expected_function: str
    expected_arguments: dict[str, Any] = field(default_factory=dict)
    # How to read the expected answer value from the tool result.
    answer_spec: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["expected_arguments"] = self.expected_arguments
        row["answer_spec"] = self.answer_spec
        return row


# Canonical normalized values for facility 124851 (so expected args are stable).
W = "IfcWindow"
D = "IfcDoor"
H = "IfcSpaceHeater"
EG = "Erdgeschoss"
UG = "Untergeschoss"
OG1 = "1. Obergeschoss"
OG2 = "2. Obergeschoss"
DG = "Dachgeschoss"


def _q(
    qid: str,
    text: str,
    category: str,
    level: str,
    group: str,
    standard: bool,
    func: str,
    args: dict[str, Any] | None = None,
    answer_spec: dict[str, Any] | None = None,
) -> EvalQuery:
    return EvalQuery(
        query_id=qid,
        query_text=text,
        category=category,
        complexity_level=level,
        paraphrase_group_id=group,
        is_standard_wording=standard,
        expected_function=func,
        expected_arguments=args or {},
        answer_spec=answer_spec or {},
    )


# ── The corpus ───────────────────────────────────────────────────────────────

CORPUS: list[EvalQuery] = [
    # ─ Group 1: list floors (metadata, L1) ─
    _q(
        "q01",
        "Which floors can I query for this building?",
        "metadata",
        "L1",
        "g01",
        True,
        "list_queryable_floors",
        {},
        {"kind": "list", "field": "name"},
    ),
    _q(
        "q02",
        "What floors are available?",
        "metadata",
        "L1",
        "g01",
        False,
        "list_queryable_floors",
        {},
        {"kind": "list", "field": "name"},
    ),
    _q(
        "q03",
        "Show me the storeys in this facility.",
        "metadata",
        "L1",
        "g01",
        False,
        "list_queryable_floors",
        {},
        {"kind": "list", "field": "name"},
    ),
    # ─ Group 2: list component types (metadata, L1) ─
    _q(
        "q04",
        "What component types are available?",
        "metadata",
        "L1",
        "g02",
        True,
        "list_queryable_component_types",
        {},
        {"kind": "any"},
    ),
    _q(
        "q05",
        "Which BIM element types can I ask about?",
        "metadata",
        "L1",
        "g02",
        False,
        "list_queryable_component_types",
        {},
        {"kind": "any"},
    ),
    _q(
        "q06",
        "What asset component categories exist?",
        "metadata",
        "L1",
        "g02",
        False,
        "list_queryable_component_types",
        {},
        {"kind": "any"},
    ),
    # ─ Group 3: capabilities (metadata, L1) ─
    _q(
        "q07",
        "What can this assistant help me with?",
        "metadata",
        "L1",
        "g03",
        True,
        "get_database_capabilities",
        {},
        {"kind": "any"},
    ),
    _q(
        "q08",
        "What kinds of questions can you answer?",
        "metadata",
        "L1",
        "g03",
        False,
        "get_database_capabilities",
        {},
        {"kind": "any"},
    ),
    # ─ Group 4: count windows on 2nd floor (counting, L2) ─
    _q(
        "q09",
        "How many windows are on the second floor?",
        "counting",
        "L2",
        "g04",
        True,
        "count_components",
        {"component_type": W, "floor": OG2},
        {"kind": "number", "field": "count"},
    ),
    _q(
        "q10",
        "Count the windows on level two.",
        "counting",
        "L2",
        "g04",
        False,
        "count_components",
        {"component_type": W, "floor": OG2},
        {"kind": "number", "field": "count"},
    ),
    _q(
        "q11",
        "What is the number of window elements on the second upper floor?",
        "counting",
        "L2",
        "g04",
        False,
        "count_components",
        {"component_type": W, "floor": OG2},
        {"kind": "number", "field": "count"},
    ),
    # ─ Group 5: count doors total (counting, L2) ─
    _q(
        "q12",
        "How many doors does the building have in total?",
        "counting",
        "L2",
        "g05",
        True,
        "count_components",
        {"component_type": D},
        {"kind": "number", "field": "count"},
    ),
    _q(
        "q13",
        "Count all doors in the facility.",
        "counting",
        "L2",
        "g05",
        False,
        "count_components",
        {"component_type": D},
        {"kind": "number", "field": "count"},
    ),
    _q(
        "q14",
        "What is the total number of door components?",
        "counting",
        "L2",
        "g05",
        False,
        "count_components",
        {"component_type": D},
        {"kind": "number", "field": "count"},
    ),
    # ─ Group 6: count windows on ground floor (counting, L2) ─
    _q(
        "q15",
        "How many windows are on the ground floor?",
        "counting",
        "L2",
        "g06",
        True,
        "count_components",
        {"component_type": W, "floor": EG},
        {"kind": "number", "field": "count"},
    ),
    _q(
        "q16",
        "Count the windows on the Erdgeschoss.",
        "counting",
        "L2",
        "g06",
        False,
        "count_components",
        {"component_type": W, "floor": EG},
        {"kind": "number", "field": "count"},
    ),
    # ─ Group 7: count heaters on first floor (counting, L2) ─
    _q(
        "q17",
        "How many space heaters are on the first floor?",
        "counting",
        "L2",
        "g07",
        True,
        "count_components",
        {"component_type": H, "floor": OG1},
        {"kind": "number", "field": "count"},
    ),
    _q(
        "q18",
        "Count the radiators on the 1. Obergeschoss.",
        "counting",
        "L2",
        "g07",
        False,
        "count_components",
        {"component_type": H, "floor": OG1},
        {"kind": "number", "field": "count"},
    ),
    # ─ Group 8: count windows total (counting, L2) ─
    _q(
        "q19",
        "How many windows are there in total?",
        "counting",
        "L2",
        "g08",
        True,
        "count_components",
        {"component_type": W},
        {"kind": "number", "field": "count"},
    ),
    _q(
        "q20",
        "What is the total window count for the building?",
        "counting",
        "L2",
        "g08",
        False,
        "count_components",
        {"component_type": W},
        {"kind": "number", "field": "count"},
    ),
    # ─ Group 9: count columns total (counting, L2) ─
    _q(
        "q21",
        "How many columns does the building have?",
        "counting",
        "L2",
        "g09",
        True,
        "count_components",
        {"component_type": "IfcColumn"},
        {"kind": "number", "field": "count"},
    ),
    _q(
        "q22",
        "Count all the columns in the facility.",
        "counting",
        "L2",
        "g09",
        False,
        "count_components",
        {"component_type": "IfcColumn"},
        {"kind": "number", "field": "count"},
    ),
    # ─ Group 10: count doors on second floor (counting, L2) ─
    _q(
        "q23",
        "How many doors are on the second floor?",
        "counting",
        "L2",
        "g10",
        True,
        "count_components",
        {"component_type": D, "floor": OG2},
        {"kind": "number", "field": "count"},
    ),
    _q(
        "q24",
        "Count the doors on level two.",
        "counting",
        "L2",
        "g10",
        False,
        "count_components",
        {"component_type": D, "floor": OG2},
        {"kind": "number", "field": "count"},
    ),
    # ─ Group 11: window attributes on 2nd floor (attribute, L2) ─
    _q(
        "q25",
        "What are the heights and widths of the windows on the second floor?",
        "attribute",
        "L2",
        "g11",
        True,
        "get_component_attributes",
        {"component_type": W, "floor": OG2},
        {"kind": "result_count"},
    ),
    _q(
        "q26",
        "Show me the dimensions of windows on level two.",
        "attribute",
        "L2",
        "g11",
        False,
        "get_component_attributes",
        {"component_type": W, "floor": OG2},
        {"kind": "result_count"},
    ),
    _q(
        "q27",
        "List the window heights on the 2. Obergeschoss.",
        "attribute",
        "L2",
        "g11",
        False,
        "get_component_attributes",
        {"component_type": W, "floor": OG2},
        {"kind": "result_count"},
    ),
    # ─ Group 12: door attributes (attribute, L2) ─
    _q(
        "q28",
        "What are the dimensions of the doors on the ground floor?",
        "attribute",
        "L2",
        "g12",
        True,
        "get_component_attributes",
        {"component_type": D, "floor": EG},
        {"kind": "result_count"},
    ),
    _q(
        "q29",
        "Show me the height and width of doors in the Erdgeschoss.",
        "attribute",
        "L2",
        "g12",
        False,
        "get_component_attributes",
        {"component_type": D, "floor": EG},
        {"kind": "result_count"},
    ),
    # ─ Group 13: total window area (aggregation, L3) ─
    _q(
        "q30",
        "What is the total window area in the building?",
        "aggregation",
        "L3",
        "g13",
        True,
        "calculate_total_component_area",
        {"component_type": W},
        {"kind": "number", "field": "total_area", "tol": 1.0},
    ),
    _q(
        "q31",
        "Calculate the overall area of all windows.",
        "aggregation",
        "L3",
        "g13",
        False,
        "calculate_total_component_area",
        {"component_type": W},
        {"kind": "number", "field": "total_area", "tol": 1.0},
    ),
    _q(
        "q32",
        "How many square meters of windows are there in total?",
        "aggregation",
        "L3",
        "g13",
        False,
        "calculate_total_component_area",
        {"component_type": W},
        {"kind": "number", "field": "total_area", "tol": 1.0},
    ),
    # ─ Group 14: window area on 2nd floor (aggregation, L3) ─
    _q(
        "q33",
        "What is the total window area on the second floor?",
        "aggregation",
        "L3",
        "g14",
        True,
        "calculate_total_component_area",
        {"component_type": W, "floor": OG2},
        {"kind": "number", "field": "total_area", "tol": 1.0},
    ),
    _q(
        "q34",
        "Calculate the window area on level two.",
        "aggregation",
        "L3",
        "g14",
        False,
        "calculate_total_component_area",
        {"component_type": W, "floor": OG2},
        {"kind": "number", "field": "total_area", "tol": 1.0},
    ),
    _q(
        "q35",
        "How many square meters of windows are on the second upper floor?",
        "aggregation",
        "L3",
        "g14",
        False,
        "calculate_total_component_area",
        {"component_type": W, "floor": OG2},
        {"kind": "number", "field": "total_area", "tol": 1.0},
    ),
    # ─ Group 15: door area total (aggregation, L3) ─
    _q(
        "q36",
        "What is the total door area in the building?",
        "aggregation",
        "L3",
        "g15",
        True,
        "calculate_total_component_area",
        {"component_type": D},
        {"kind": "number", "field": "total_area", "tol": 1.0},
    ),
    _q(
        "q37",
        "Calculate the combined area of all doors.",
        "aggregation",
        "L3",
        "g15",
        False,
        "calculate_total_component_area",
        {"component_type": D},
        {"kind": "number", "field": "total_area", "tol": 1.0},
    ),
    # ─ Group 16: window area by floor (aggregation, L3) ─
    _q(
        "q38",
        "What is the window area on each floor?",
        "aggregation",
        "L3",
        "g16",
        True,
        "calculate_area_by_floor",
        {"component_type": W},
        {"kind": "any"},
    ),
    _q(
        "q39",
        "Break down the window area by storey.",
        "aggregation",
        "L3",
        "g16",
        False,
        "calculate_area_by_floor",
        {"component_type": W},
        {"kind": "any"},
    ),
    # ─ Group 17: floor with largest window area (multistep, L4) ─
    _q(
        "q40",
        "Which floor has the largest total window area?",
        "multistep",
        "L4",
        "g17",
        True,
        "find_floor_with_largest_component_area",
        {"component_type": W},
        {"kind": "string", "field": "largest_floor"},
    ),
    _q(
        "q41",
        "Where is the biggest amount of window area?",
        "multistep",
        "L4",
        "g17",
        False,
        "find_floor_with_largest_component_area",
        {"component_type": W},
        {"kind": "string", "field": "largest_floor"},
    ),
    _q(
        "q42",
        "Which storey has the most glazing?",
        "multistep",
        "L4",
        "g17",
        False,
        "find_floor_with_largest_component_area",
        {"component_type": W},
        {"kind": "string", "field": "largest_floor"},
    ),
    # ─ Group 18: compare window counts first vs second (multistep, L4) ─
    _q(
        "q43",
        "Are there more windows on the first or the second upper floor?",
        "multistep",
        "L4",
        "g18",
        True,
        "compare_component_count_between_floors",
        {"component_type": W, "floor_a": OG1, "floor_b": OG2},
        {"kind": "string", "field": "more_on"},
    ),
    _q(
        "q44",
        "Compare window counts between the first and second floor.",
        "multistep",
        "L4",
        "g18",
        False,
        "compare_component_count_between_floors",
        {"component_type": W, "floor_a": OG1, "floor_b": OG2},
        {"kind": "string", "field": "more_on"},
    ),
    _q(
        "q45",
        "Which has more windows, level one or level two?",
        "multistep",
        "L4",
        "g18",
        False,
        "compare_component_count_between_floors",
        {"component_type": W, "floor_a": OG1, "floor_b": OG2},
        {"kind": "string", "field": "more_on"},
    ),
    # ─ Group 19: compare window area first vs second (multistep, L4) ─
    _q(
        "q46",
        "Is the window area larger on the first or the second floor?",
        "multistep",
        "L4",
        "g19",
        True,
        "compare_component_area_between_floors",
        {"component_type": W, "floor_a": OG1, "floor_b": OG2},
        {"kind": "string", "field": "more_on"},
    ),
    _q(
        "q47",
        "Compare the window area between level one and level two.",
        "multistep",
        "L4",
        "g19",
        False,
        "compare_component_area_between_floors",
        {"component_type": W, "floor_a": OG1, "floor_b": OG2},
        {"kind": "string", "field": "more_on"},
    ),
    _q(
        "q48",
        "Which floor has more window surface, the first or second upper floor?",
        "multistep",
        "L4",
        "g19",
        False,
        "compare_component_area_between_floors",
        {"component_type": W, "floor_a": OG1, "floor_b": OG2},
        {"kind": "string", "field": "more_on"},
    ),
    # ─ Group 20: compare door counts ground vs second (multistep, L4) ─
    _q(
        "q49",
        "Are there more doors on the ground floor or the second floor?",
        "multistep",
        "L4",
        "g20",
        True,
        "compare_component_count_between_floors",
        {"component_type": D, "floor_a": EG, "floor_b": OG2},
        {"kind": "string", "field": "more_on"},
    ),
    _q(
        "q50",
        "Compare the number of doors between the Erdgeschoss and level two.",
        "multistep",
        "L4",
        "g20",
        False,
        "compare_component_count_between_floors",
        {"component_type": D, "floor_a": EG, "floor_b": OG2},
        {"kind": "string", "field": "more_on"},
    ),
    # ─ Group 21: count windows top floor (counting, L2) ─
    _q(
        "q51",
        "How many windows are on the top floor?",
        "counting",
        "L2",
        "g21",
        True,
        "count_components",
        {"component_type": W, "floor": DG},
        {"kind": "number", "field": "count"},
    ),
    _q(
        "q52",
        "Count the windows in the attic.",
        "counting",
        "L2",
        "g21",
        False,
        "count_components",
        {"component_type": W, "floor": DG},
        {"kind": "number", "field": "count"},
    ),
    # ─ Group 22: count windows basement (counting, L2) ─
    _q(
        "q53",
        "How many windows are in the basement?",
        "counting",
        "L2",
        "g22",
        True,
        "count_components",
        {"component_type": W, "floor": UG},
        {"kind": "number", "field": "count"},
    ),
    _q(
        "q54",
        "Count the windows on the Untergeschoss.",
        "counting",
        "L2",
        "g22",
        False,
        "count_components",
        {"component_type": W, "floor": UG},
        {"kind": "number", "field": "count"},
    ),
    # ─ Group 23: window area ground floor (aggregation, L3) ─
    _q(
        "q55",
        "What is the total window area on the ground floor?",
        "aggregation",
        "L3",
        "g23",
        True,
        "calculate_total_component_area",
        {"component_type": W, "floor": EG},
        {"kind": "number", "field": "total_area", "tol": 1.0},
    ),
    _q(
        "q56",
        "Calculate the window area in the Erdgeschoss.",
        "aggregation",
        "L3",
        "g23",
        False,
        "calculate_total_component_area",
        {"component_type": W, "floor": EG},
        {"kind": "number", "field": "total_area", "tol": 1.0},
    ),
    # ─ Group 24: count sanitary terminals (counting, L2) ─
    _q(
        "q57",
        "How many sanitary terminals are in the building?",
        "counting",
        "L2",
        "g24",
        True,
        "count_components",
        {"component_type": "IfcSanitaryTerminal"},
        {"kind": "number", "field": "count"},
    ),
    _q(
        "q58",
        "Count the sanitary fixtures in the facility.",
        "counting",
        "L2",
        "g24",
        False,
        "count_components",
        {"component_type": "IfcSanitaryTerminal"},
        {"kind": "number", "field": "count"},
    ),
    # ─ Group 25: count sensors total (counting, L2) ─
    _q(
        "q59",
        "How many sensors does the building have?",
        "counting",
        "L2",
        "g25",
        True,
        "count_components",
        {"component_type": "IfcSensor"},
        {"kind": "number", "field": "count"},
    ),
    _q(
        "q60",
        "Count all sensors in the facility.",
        "counting",
        "L2",
        "g25",
        False,
        "count_components",
        {"component_type": "IfcSensor"},
        {"kind": "number", "field": "count"},
    ),
    # ─ Group 26: floor with largest door area (multistep, L4) ─
    _q(
        "q61",
        "Which floor has the largest total door area?",
        "multistep",
        "L4",
        "g26",
        True,
        "find_floor_with_largest_component_area",
        {"component_type": D},
        {"kind": "string", "field": "largest_floor"},
    ),
    _q(
        "q62",
        "On which storey is the door area the biggest?",
        "multistep",
        "L4",
        "g26",
        False,
        "find_floor_with_largest_component_area",
        {"component_type": D},
        {"kind": "string", "field": "largest_floor"},
    ),
    # ─ Group 27: slabs count (counting, L2) ─
    _q(
        "q63",
        "How many slabs are in the building?",
        "counting",
        "L2",
        "g27",
        True,
        "count_components",
        {"component_type": "IfcSlab"},
        {"kind": "number", "field": "count"},
    ),
    _q(
        "q64",
        "Count the floor slabs in the facility.",
        "counting",
        "L2",
        "g27",
        False,
        "count_components",
        {"component_type": "IfcSlab"},
        {"kind": "number", "field": "count"},
    ),
    # ─ Group 28: door area by floor (aggregation, L3) ─
    _q(
        "q65",
        "What is the door area on each floor?",
        "aggregation",
        "L3",
        "g28",
        True,
        "calculate_area_by_floor",
        {"component_type": D},
        {"kind": "any"},
    ),
    _q(
        "q66",
        "Break down door area per storey.",
        "aggregation",
        "L3",
        "g28",
        False,
        "calculate_area_by_floor",
        {"component_type": D},
        {"kind": "any"},
    ),
    # ─ Group 29: window count first floor (counting, L2) ─
    _q(
        "q67",
        "How many windows are on the first floor?",
        "counting",
        "L2",
        "g29",
        True,
        "count_components",
        {"component_type": W, "floor": OG1},
        {"kind": "number", "field": "count"},
    ),
    _q(
        "q68",
        "Count the windows on the 1. Obergeschoss.",
        "counting",
        "L2",
        "g29",
        False,
        "count_components",
        {"component_type": W, "floor": OG1},
        {"kind": "number", "field": "count"},
    ),
    # ─ Group 30: window area first floor (aggregation, L3) ─
    _q(
        "q69",
        "What is the total window area on the first floor?",
        "aggregation",
        "L3",
        "g30",
        True,
        "calculate_total_component_area",
        {"component_type": W, "floor": OG1},
        {"kind": "number", "field": "total_area", "tol": 1.0},
    ),
    _q(
        "q70",
        "Calculate the window area on level one.",
        "aggregation",
        "L3",
        "g30",
        False,
        "calculate_total_component_area",
        {"component_type": W, "floor": OG1},
        {"kind": "number", "field": "total_area", "tol": 1.0},
    ),
]


def get_corpus() -> list[EvalQuery]:
    """Return the full evaluation corpus."""
    return list(CORPUS)


def corpus_dataframe() -> pd.DataFrame:
    """Corpus as a DataFrame (arguments/specs kept as objects)."""
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
    print(df.groupby(["complexity_level"]).size().to_string())
    print(df.groupby(["category"]).size().to_string())
    print("Written to", p)
