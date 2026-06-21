from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT

EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"
GROUND_TRUTH_JSON = EVAL_DIR / "ground_truth.json"


@dataclass
class GroundTruth:
    """Expected outcome for one query, loaded from the committed reference file."""

    query_id: str
    query_text: str
    category: str
    complexity_level: str
    paraphrase_group_id: str
    is_standard_wording: bool
    expected_functions: list[str]
    expected_arguments: dict[str, Any]
    expected_answer_values: list[Any] = field(default_factory=list)
    answer_kind: str = "any"  # number | string | list | any
    answer_tolerance: float = 0.0
    execution_ok: bool = True
    note: str | None = None


def load_ground_truth(path: Path = GROUND_TRUTH_JSON) -> list[GroundTruth]:
    """Load the committed ground-truth file; raise clearly if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Ground truth not found at {path}.")
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [GroundTruth(**row) for row in rows]
