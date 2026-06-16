"""Ground-truth generation (multi-step aware).

The committed ground truth is produced ONCE from the corpus: for every query
the ``gt_calls`` are executed through the registry — deterministic, parameterised
SQL, never the LLM — and reduced with the query's ``answer_spec`` to the expected
answer value(s). The result is written to ``ground_truth.json`` and committed.

The evaluation runner then *loads* that frozen file (:func:`load_ground_truth`)
and compares model output against it, rather than recomputing ground truth on
every run. Regenerate (and re-validate) only when the corpus or the underlying
database snapshot changes::

    python -m app.evaluation.ground_truth

Output (committed; the researcher reviews/validates it):
- data/evaluation/ground_truth.json   ← single source the runner compares to
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.evaluation.corpus import EVAL_DIR, get_corpus
from app.tools.fm_functions import build_registry
from app.tools.models import ToolCall, strip_internal_keys
from app.tools.registry import ToolRegistry

GROUND_TRUTH_JSON = EVAL_DIR / "ground_truth.json"


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
                results.append(strip_internal_keys(res.data))
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


def save_ground_truth(gt: list[GroundTruth], out_dir: Path = EVAL_DIR) -> Path:
    """Write ground_truth.json (the committed reference); return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / GROUND_TRUTH_JSON.name
    rows = [asdict(g) for g in gt]
    json_path.write_text(
        json.dumps(rows, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )
    return json_path


def load_ground_truth(path: Path = GROUND_TRUTH_JSON) -> list[GroundTruth]:
    """Load the committed ground truth the evaluation compares against.

    Raises a clear error if the file is missing so the runner never silently
    falls back to recomputing ground truth. Regenerate with
    ``python -m app.evaluation.ground_truth``.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Ground truth not found at {path}. Generate it once with "
            "`python -m app.evaluation.ground_truth` (runs deterministic SQL, no LLM)."
        )
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [GroundTruth(**row) for row in rows]


def main() -> int:
    gt = build_ground_truth()
    json_path = save_ground_truth(gt)
    ok = sum(1 for g in gt if g.execution_ok)
    print(f"Ground truth for {len(gt)} queries ({ok} executed OK).")
    print("Written:", json_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
