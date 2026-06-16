"""Ground-truth generation (multi-step aware).

For every corpus query, ground truth is produced by executing the query's
``gt_calls`` through the registry — deterministic, parameterised SQL, never the
LLM — and reducing the results with the query's ``answer_spec`` to the expected
answer value(s).

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
    expected_functions: list[str]
    expected_arguments: dict[str, Any]
    expected_answer_values: list[Any] = field(default_factory=list)
    answer_kind: str = "any"  # number | string | list | any
    answer_tolerance: float = 0.0
    execution_ok: bool = True
    note: str | None = None


def _strip_internal(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if k != "_normalized_arguments"}
    return data


def _reduce(
    spec: dict[str, Any], gt_calls: list[dict[str, Any]], results: list[Any]
) -> tuple[list[Any], str, float]:
    """Apply an answer_spec to ground-truth call results -> (values, kind, tol)."""
    kind = spec.get("kind", "any")
    tol = float(spec.get("tol", 0.0))

    if kind == "any":
        return [], "any", tol
    if kind == "list":
        data = results[0]
        items = data if isinstance(data, list) else []
        f = spec.get("field")
        return [it.get(f) if isinstance(it, dict) else it for it in items], "list", tol
    if kind == "result_count":
        data = results[0]
        return [data.get("result_count") if isinstance(data, dict) else None], "number", tol
    if kind == "number":
        data = results[0]
        return [data.get(spec["field"]) if isinstance(data, dict) else None], "number", tol
    if kind == "string":
        data = results[0]
        return [data.get(spec["field"]) if isinstance(data, dict) else None], "string", tol
    if kind == "values":
        f = spec["field"]
        return [d.get(f) if isinstance(d, dict) else None for d in results], "number", tol
    if kind == "compare_winner":
        f = spec["field"]
        floor_key = spec.get("floor_key", "floor")
        a, b = results[0], results[1]
        va = a.get(f) if isinstance(a, dict) else None
        vb = b.get(f) if isinstance(b, dict) else None
        fa = gt_calls[0]["arguments"].get(floor_key)
        fb = gt_calls[1]["arguments"].get(floor_key)
        if va is None or vb is None:
            return [], "any", tol
        winner = fa if va > vb else (fb if vb > va else "equal")
        return [winner], "string", tol
    if kind == "max_floor":
        data = results[0]
        items = data.get(spec["by"], []) if isinstance(data, dict) else []
        valued = [it for it in items if it.get(spec["value"]) is not None]
        if not valued:
            return [], "any", tol
        top = max(valued, key=lambda it: it[spec["value"]])
        return [top.get(spec["label"])], "string", tol
    return [], "any", tol


def build_ground_truth(registry: ToolRegistry | None = None) -> list[GroundTruth]:
    """Compute ground truth for every corpus query."""
    reg = registry if registry is not None else build_registry()
    out: list[GroundTruth] = []
    for q in get_corpus():
        results: list[Any] = []
        ok = True
        note: str | None = None
        for gc in q.gt_calls:
            res = reg.execute(ToolCall(name=gc["function"], arguments=gc["arguments"]))
            if not res.ok:
                ok = False
                note = res.error_message
                results.append(None)
            else:
                results.append(_strip_internal(res.data))
        values, kind, tol = _reduce(q.answer_spec, q.gt_calls, results) if ok else ([], "any", 0.0)
        out.append(
            GroundTruth(
                query_id=q.query_id,
                query_text=q.query_text,
                category=q.category,
                complexity_level=q.complexity_level,
                paraphrase_group_id=q.paraphrase_group_id,
                is_standard_wording=q.is_standard_wording,
                expected_functions=q.expected_functions,
                expected_arguments=q.expected_arguments,
                expected_answer_values=values,
                answer_kind=kind,
                answer_tolerance=tol,
                execution_ok=ok,
                note=note,
            )
        )
    return out


def save_ground_truth(gt: list[GroundTruth], out_dir: Path = EVAL_DIR) -> tuple[Path, Path]:
    """Write ground_truth.json and .csv; return both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ground_truth.json"
    csv_path = out_dir / "ground_truth.csv"
    rows = [asdict(g) for g in gt]
    json_path.write_text(
        json.dumps(rows, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )
    flat = [
        {
            **{
                k: v
                for k, v in r.items()
                if k not in {"expected_arguments", "expected_answer_values", "expected_functions"}
            },
            "expected_functions": json.dumps(r["expected_functions"], ensure_ascii=False),
            "expected_arguments": json.dumps(r["expected_arguments"], ensure_ascii=False),
            "expected_answer_values": json.dumps(
                r["expected_answer_values"], default=str, ensure_ascii=False
            ),
        }
        for r in rows
    ]
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
