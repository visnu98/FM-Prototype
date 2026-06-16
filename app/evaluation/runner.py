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

from app.core.config import get_settings
from app.evaluation.corpus import EVAL_DIR
from app.evaluation.ground_truth import GroundTruth, load_ground_truth
from app.evaluation.metrics import QueryMetrics, aggregate_by_model, error_pivot, evaluate_query
from app.tools.fm_functions import build_registry

logger = logging.getLogger(__name__)

RUNS_DIR = EVAL_DIR / "runs"


def _run_one_model(
    model: str,
    ground_truth: list[GroundTruth],
    delay: float,
    limit: int | None,
) -> tuple[list[QueryMetrics], list[dict[str, Any]]]:
    """Run every ground-truth query through one model; return metrics + raw rows."""
    # Import here so the rest of the module imports without the LLM stack.
    from app.chatbot.service import ChatbotService

    registry = build_registry()
    service = ChatbotService.for_model(model, registry=registry)

    metrics: list[QueryMetrics] = []
    raw_rows: list[dict[str, Any]] = []

    queries = ground_truth[:limit] if limit else ground_truth
    for i, gt in enumerate(queries, 1):
        logger.info("[%s] %d/%d %s", model, i, len(queries), gt.query_id)
        response = service.answer(gt.query_text)
        # A question succeeds at the tool level if it made >=1 call and none errored.
        execution_success = response.made_tool_call and all(c["ok"] for c in response.calls)

        qm = evaluate_query(
            gt=gt,
            model=model,
            made_tool_call=response.made_tool_call,
            actual_functions=response.selected_functions,
            actual_calls=response.calls,
            registry_error=response.error_category,
            execution_success=execution_success,
            final_answer=response.final_answer,
            latency_total=response.latency_total_ms,
            latency_planning=response.latency_planning_ms,
            latency_tools=response.latency_tools_ms,
        )
        metrics.append(qm)

        raw_rows.append(
            {
                "query_id": gt.query_id,
                "query_text": gt.query_text,
                "model": model,
                "category": gt.category,
                "complexity_level": gt.complexity_level,
                "paraphrase_group_id": gt.paraphrase_group_id,
                "is_standard_wording": gt.is_standard_wording,
                "expected_functions": gt.expected_functions,
                "actual_functions": response.selected_functions,
                "num_steps": response.num_steps,
                "calls": response.calls,  # full per-step trace (function, args, result)
                "expected_arguments": gt.expected_arguments,
                "expected_answer_values": gt.expected_answer_values,
                "final_answer": response.final_answer,
                "answer_correct": qm.answer_correct,
                "function_correct": qm.function_correct,
                "parameters_correct": qm.parameters_correct,
                "fully_correct_call": qm.fully_correct_call,
                "execution_success": qm.execution_success,
                "latency_total": qm.latency_total,
                "latency_planning": qm.latency_planning,
                "latency_tools": qm.latency_tools,
                "error_category": qm.error_category,
            }
        )
        if delay:
            time.sleep(delay)

    return metrics, raw_rows


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

    # Load the committed ground truth and compare against it (no recompute).
    ground_truth = load_ground_truth()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (out_dir or RUNS_DIR) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[QueryMetrics] = []
    raw_path = run_dir / "raw_results.jsonl"
    with raw_path.open("w", encoding="utf-8") as fh:
        for model in models:
            model_metrics, raw_rows = _run_one_model(model, ground_truth, delay, limit)
            all_metrics.extend(model_metrics)
            for row in raw_rows:
                fh.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")

    metrics_df = pd.DataFrame([dataclasses.asdict(m) for m in all_metrics])
    metrics_df.to_csv(run_dir / "metrics.csv", index=False, encoding="utf-8")

    aggregate_by_model(metrics_df).to_csv(run_dir / "model_comparison.csv", encoding="utf-8")
    error_pivot(metrics_df).to_csv(run_dir / "error_taxonomy.csv", encoding="utf-8")

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
