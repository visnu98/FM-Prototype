"""Ground-truth generation (Phase 11).

For every corpus query, ground truth is produced by executing the *expected*
function with the *canonical* arguments through the registry — i.e. by
deterministic, parameterised SQL, never by the LLM. The salient expected answer
value(s) are extracted via the query's ``answer_spec``.

Outputs:
- data/evaluation/ground_truth.json
- data/evaluation/ground_truth.csv
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.evaluation.corpus import EVAL_DIR, get_corpus
from app.tools.fm_functions import build_registry
from app.tools.models import ToolCall
from app.tools.registry import ToolRegistry


@dataclass
class GroundTruth:
    """Deterministic expected outcome for one query."""

    query_id: str
    query_text: str
    category: str
    complexity_level: str
    paraphrase_group_id: str
    is_standard_wording: bool
    expected_function: str
    expected_arguments: dict[str, Any]
    expected_result: Any
    expected_answer_values: list[Any] = field(default_factory=list)
    answer_kind: str = "any"
    answer_tolerance: float = 0.0
    execution_ok: bool = True
    note: str | None = None


def _extract_expected_values(spec: dict[str, Any], data: Any) -> tuple[list[Any], str, float]:
    """Apply an answer_spec to a tool result to get expected value(s)."""
    kind = spec.get("kind", "any")
    tol = float(spec.get("tol", 0.0))
    if not isinstance(data, dict) and kind not in {"list", "any"}:
        return [], kind, tol

    if kind == "number":
        return [data.get(spec["field"])], "number", tol
    if kind == "string":
        return [data.get(spec["field"])], "string", tol
    if kind == "result_count":
        return [data.get("result_count")], "number", tol
    if kind == "list":
        items = data if isinstance(data, list) else data.get("items", [])
        field_name = spec.get("field")
        values = [it.get(field_name) if isinstance(it, dict) else it for it in items]
        return values, "list", tol
    # "any": no specific scalar to check.
    return [], "any", tol


def build_ground_truth(registry: ToolRegistry | None = None) -> list[GroundTruth]:
    """Compute ground truth for every corpus query."""
    reg = registry if registry is not None else build_registry()
    out: list[GroundTruth] = []
    for q in get_corpus():
        result = reg.execute(ToolCall(name=q.expected_function, arguments=q.expected_arguments))
        values, kind, tol = (
            _extract_expected_values(q.answer_spec, result.data) if result.ok else ([], "any", 0.0)
        )
        out.append(
            GroundTruth(
                query_id=q.query_id,
                query_text=q.query_text,
                category=q.category,
                complexity_level=q.complexity_level,
                paraphrase_group_id=q.paraphrase_group_id,
                is_standard_wording=q.is_standard_wording,
                expected_function=q.expected_function,
                expected_arguments=q.expected_arguments,
                expected_result=_strip_internal(result.data) if result.ok else None,
                expected_answer_values=values,
                answer_kind=kind,
                answer_tolerance=tol,
                execution_ok=result.ok,
                note=None if result.ok else result.error_message,
            )
        )
    return out


def _strip_internal(data: Any) -> Any:
    """Drop the `_normalized_arguments` bookkeeping key from stored results."""
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if k != "_normalized_arguments"}
    return data


def save_ground_truth(gt: list[GroundTruth], out_dir: Path = EVAL_DIR) -> tuple[Path, Path]:
    """Write ground_truth.json and .csv; return both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ground_truth.json"
    csv_path = out_dir / "ground_truth.csv"
    rows = [asdict(g) for g in gt]
    json_path.write_text(
        json.dumps(rows, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )
    # Flatten complex fields for the CSV view.
    flat = []
    for r in rows:
        flat.append(
            {
                **{
                    k: v
                    for k, v in r.items()
                    if k not in {"expected_result", "expected_arguments", "expected_answer_values"}
                },
                "expected_arguments": json.dumps(r["expected_arguments"], ensure_ascii=False),
                "expected_answer_values": json.dumps(
                    r["expected_answer_values"], default=str, ensure_ascii=False
                ),
            }
        )
    pd.DataFrame(flat).to_csv(csv_path, index=False, encoding="utf-8")
    return json_path, csv_path


def main() -> int:
    gt = build_ground_truth()
    json_path, csv_path = save_ground_truth(gt)
    ok = sum(1 for g in gt if g.execution_ok)
    print(f"Ground truth for {len(gt)} queries ({ok} executed OK).")
    print("Written:", json_path.name, "and", csv_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
