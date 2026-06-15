"""Evaluation runner (Phase 14).

Runs the full corpus through each model using the SAME pipeline, tools, prompt
and temperature, scores every answer against ground truth, and writes raw +
aggregated results to ``data/evaluation/runs/<timestamp>/``.

Run::

    python -m app.evaluation.runner                 # both models, full corpus
    python -m app.evaluation.runner --limit 8       # quick smoke (first 8 queries)
    python -m app.evaluation.runner --models llama-3.1-8b-instant
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import get_settings
from app.evaluation.corpus import EVAL_DIR, get_corpus
from app.evaluation.error_taxonomy import ALL_CATEGORIES
from app.evaluation.ground_truth import GroundTruth, build_ground_truth, save_ground_truth
from app.evaluation.metrics import QueryMetrics, evaluate_query
from app.tools.fm_functions import build_registry

logger = logging.getLogger(__name__)

RUNS_DIR = EVAL_DIR / "runs"


def _run_one_model(
    model: str,
    gt_by_id: dict[str, GroundTruth],
    delay: float,
    limit: int | None,
) -> tuple[list[QueryMetrics], list[dict[str, Any]]]:
    """Run every corpus query through one model; return metrics + raw rows."""
    # Import here so the rest of the module imports without the LLM stack.
    from app.chatbot.service import ChatbotService

    registry = build_registry()
    service = ChatbotService.for_model(model, registry=registry)

    metrics: list[QueryMetrics] = []
    raw_rows: list[dict[str, Any]] = []

    queries = get_corpus()[:limit] if limit else get_corpus()
    for i, q in enumerate(queries, 1):
        gt = gt_by_id[q.query_id]
        logger.info("[%s] %d/%d %s", model, i, len(queries), q.query_id)
        response = service.answer(q.query_text)
        execution_success = bool(response.tool_result and response.tool_result.ok)

        qm = evaluate_query(
            gt=gt,
            model=model,
            made_tool_call=response.made_tool_call,
            actual_function=response.selected_function,
            actual_normalized_arguments=response.normalized_arguments,
            registry_error=response.error_category,
            execution_success=execution_success,
            final_answer=response.final_answer,
            latency_total=response.latency_total_ms,
            latency_tool_call=response.latency_tool_call_ms,
            latency_sql=response.latency_sql_ms,
            latency_final_answer=response.latency_final_answer_ms,
        )
        metrics.append(qm)

        actual_result = (
            {k: v for k, v in response.tool_result.data.items() if k != "_normalized_arguments"}
            if (response.tool_result and isinstance(response.tool_result.data, dict))
            else (response.tool_result.data if response.tool_result else None)
        )
        raw_rows.append(
            {
                "query_id": q.query_id,
                "query_text": q.query_text,
                "model": model,
                "category": q.category,
                "complexity_level": q.complexity_level,
                "paraphrase_group_id": q.paraphrase_group_id,
                "is_standard_wording": q.is_standard_wording,
                "expected_function": gt.expected_function,
                "actual_function": qm.actual_function,
                "expected_parameters": gt.expected_arguments,
                "actual_parameters": qm.actual_parameters,
                "expected_result": gt.expected_result,
                "actual_result": actual_result,
                "expected_answer_values": gt.expected_answer_values,
                "final_answer": response.final_answer,
                "answer_correct": qm.answer_correct,
                "function_correct": qm.function_correct,
                "parameters_correct": qm.parameters_correct,
                "fully_correct_call": qm.fully_correct_call,
                "execution_success": qm.execution_success,
                "latency_total": qm.latency_total,
                "latency_tool_call": qm.latency_tool_call,
                "latency_sql": qm.latency_sql,
                "latency_final_answer": qm.latency_final_answer,
                "error_category": qm.error_category,
            }
        )
        if delay:
            time.sleep(delay)

    return metrics, raw_rows


# ── Aggregation ──────────────────────────────────────────────────────────────


def _aggregate(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Per-model aggregate metrics."""
    grouped = metrics_df.groupby("model")
    agg = grouped.agg(
        n=("query_id", "count"),
        fully_correct_call_rate=("fully_correct_call", "mean"),
        function_accuracy=("function_correct", "mean"),
        parameter_accuracy=("parameters_correct", "mean"),
        answer_accuracy=("answer_correct", "mean"),
        execution_success_rate=("execution_success", "mean"),
        mean_latency_total_ms=("latency_total", "mean"),
        median_latency_total_ms=("latency_total", "median"),
    ).reset_index()
    return agg


def _error_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Counts of each error category per model (wide format)."""
    counts = metrics_df.groupby(["model", "error_category"]).size().reset_index(name="count")
    wide = counts.pivot(index="model", columns="error_category", values="count").fillna(0)
    # Ensure all categories appear as columns.
    for cat in ALL_CATEGORIES:
        if cat.value not in wide.columns:
            wide[cat.value] = 0
    return wide.reset_index()


def _summary_md(agg: pd.DataFrame, metrics_df: pd.DataFrame, timestamp: str) -> str:
    lines = [f"# Evaluation run {timestamp}\n"]
    lines.append(f"- Queries per model: **{metrics_df['query_id'].nunique()}**")
    lines.append(f"- Models: **{', '.join(agg['model'])}**\n")
    lines.append("## Aggregate metrics\n")
    lines.append(agg.to_markdown(index=False, floatfmt=".3f"))
    lines.append("\n## Fully-correct-call rate by complexity level\n")
    pivot = metrics_df.groupby(["model", "complexity_level"])["fully_correct_call"].mean().unstack()
    lines.append(pivot.to_markdown(floatfmt=".3f"))
    lines.append("\n## Answer accuracy by complexity level\n")
    pivot2 = metrics_df.groupby(["model", "complexity_level"])["answer_correct"].mean().unstack()
    lines.append(pivot2.to_markdown(floatfmt=".3f"))
    return "\n".join(lines)


# ── Orchestration ────────────────────────────────────────────────────────────


def run_evaluation(
    models: list[str] | None = None,
    *,
    limit: int | None = None,
    delay: float = 0.0,
    out_dir: Path | None = None,
) -> Path:
    """Run the evaluation and write all output files; return the run directory."""
    settings = get_settings()
    models = models or settings.models()

    # Ground truth (deterministic) — also persisted for traceability.
    gt = build_ground_truth()
    save_ground_truth(gt)
    gt_by_id = {g.query_id: g for g in gt}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (out_dir or RUNS_DIR) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[QueryMetrics] = []
    raw_path = run_dir / "raw_results.jsonl"
    with raw_path.open("w", encoding="utf-8") as fh:
        for model in models:
            model_metrics, raw_rows = _run_one_model(model, gt_by_id, delay, limit)
            all_metrics.extend(model_metrics)
            for row in raw_rows:
                fh.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")

    metrics_df = pd.DataFrame([dataclasses.asdict(m) for m in all_metrics])
    metrics_df.to_csv(run_dir / "metrics.csv", index=False, encoding="utf-8")

    agg = _aggregate(metrics_df)
    agg.to_csv(run_dir / "model_comparison.csv", index=False, encoding="utf-8")
    _error_table(metrics_df).to_csv(run_dir / "error_taxonomy.csv", index=False, encoding="utf-8")
    (run_dir / "summary.md").write_text(_summary_md(agg, metrics_df, timestamp), encoding="utf-8")

    logger.info("Evaluation written to %s", run_dir)
    return run_dir


def latest_run(runs_dir: Path = RUNS_DIR) -> Path | None:
    """Return the most recent run directory, if any."""
    if not runs_dir.exists():
        return None
    dirs = sorted((p for p in runs_dir.iterdir() if p.is_dir()), reverse=True)
    return dirs[0] if dirs else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LLM function-calling evaluation.")
    parser.add_argument(
        "--models", nargs="*", default=None, help="Model names (default: MODEL_A, MODEL_B)."
    )
    parser.add_argument("--limit", type=int, default=None, help="Only the first N queries.")
    parser.add_argument(
        "--delay", type=float, default=0.0, help="Seconds to sleep between queries."
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    run_dir = run_evaluation(args.models, limit=args.limit, delay=args.delay)
    print(f"Evaluation complete. Results in: {run_dir}")
    print("Next: python -m app.evaluation.report  (generates thesis-ready reports)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
