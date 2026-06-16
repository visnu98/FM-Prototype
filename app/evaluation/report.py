"""Thesis-ready report generation (Phase 16).

Reads a run's metrics + ground truth, runs the H1-H4 tests, and writes:
- summary.md            (overall metrics)
- hypothesis_results.md (H1-H4 supported/not, with tests and interpretation)
- model_comparison.md   (side-by-side model metrics)
- error_analysis.md     (error category distribution)
- metrics.csv           (already written by the runner; copied/kept)
- plots/*.png           (if matplotlib is available)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from app.core.config import get_settings
from app.evaluation.metrics import aggregate_by_model, error_pivot
from app.evaluation.runner import latest_run
from app.evaluation.statistics import HypothesisResult, run_all, save_results


def generate_summary_md(df: pd.DataFrame) -> str:
    lines = ["# Evaluation Summary\n"]
    lines.append(f"- Total query runs: **{len(df)}**")
    lines.append(f"- Models: **{', '.join(sorted(df['model'].unique()))}**")
    lines.append(f"- Unique queries: **{df['query_id'].nunique()}**\n")
    lines.append("## Metrics by model\n")
    lines.append(aggregate_by_model(df).to_markdown(floatfmt=".3f"))
    lines.append("\n## Metrics by complexity level (answer accuracy)\n")
    lines.append(
        df.groupby(["complexity_level", "model"])["answer_correct"]
        .mean()
        .unstack()
        .to_markdown(floatfmt=".3f")
    )
    lines.append("\n## Metrics by category (fully-correct call)\n")
    lines.append(
        df.groupby(["category", "model"])["fully_correct_call"]
        .mean()
        .unstack()
        .to_markdown(floatfmt=".3f")
    )
    return "\n".join(lines)


def generate_hypothesis_md(results: list[HypothesisResult]) -> str:
    lines = ["# Hypothesis Results\n"]
    verdict = {True: "**supported**", False: "**not supported**", None: "inconclusive"}
    lines.append("| Hypothesis | Verdict | Test | p-value |")
    lines.append("| --- | --- | --- | --- |")
    for r in results:
        p = "-" if r.p_value is None else f"{r.p_value:.4f}"
        lines.append(f"| {r.hypothesis} | {verdict[r.supported]} | {r.test} | {p} |")
    lines.append("")
    for r in results:
        lines.append(f"## {r.hypothesis} — {verdict[r.supported]}\n")
        lines.append(f"- **Metric:** {r.metric}")
        lines.append(f"- **Test:** {r.test}")
        if r.statistic is not None:
            lines.append(f"- **Statistic:** {r.statistic:.4f}")
        if r.p_value is not None:
            lines.append(f"- **p-value:** {r.p_value:.4f}")
        lines.append(f"- **Result:** {r.result_value}")
        lines.append(f"- **Interpretation:** {r.interpretation}")
        lines.append(f"- **Limitations:** {r.limitations}\n")
    return "\n".join(lines)


def generate_model_comparison_md(df: pd.DataFrame) -> str:
    lines = ["# Model Comparison\n"]
    lines.append(aggregate_by_model(df).to_markdown(floatfmt=".3f"))
    lines.append("\n## Per-model latency breakdown (mean ms)\n")
    lat = df.groupby("model")[["latency_planning", "latency_tools", "latency_total"]].mean()
    lines.append(lat.to_markdown(floatfmt=".1f"))
    return "\n".join(lines)


def generate_error_analysis_md(df: pd.DataFrame) -> str:
    lines = ["# Error Analysis\n"]
    lines.append(error_pivot(df).to_markdown())
    lines.append("\n## Most common failure cases\n")
    failures = df[df["error_category"] != "none"]
    if failures.empty:
        lines.append("_No failures recorded._")
    else:
        for _, row in failures.head(20).iterrows():
            lines.append(
                f"- `{row['query_id']}` [{row['model']}] **{row['error_category']}** — "
                f"expected `{row['expected_functions']}`, got `{row['actual_functions']}`."
            )
    return "\n".join(lines)


def _make_plots(df: pd.DataFrame, plots_dir: Path) -> list[str]:
    """Generate PNG plots; return filenames. Silently skips if matplotlib absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    plots_dir.mkdir(parents=True, exist_ok=True)
    made: list[str] = []

    # 1) Fully-correct-call rate by model.
    fig, ax = plt.subplots(figsize=(6, 4))
    df.groupby("model")["fully_correct_call"].mean().plot.bar(ax=ax, color="#4C72B0")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fully-correct-call rate")
    ax.set_title("Correct-call rate by model")
    ax.axhline(0.9, color="red", linestyle="--", label="H1 target 90%")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "correct_call_by_model.png", dpi=120)
    plt.close(fig)
    made.append("correct_call_by_model.png")

    # 2) Answer accuracy by complexity level.
    fig, ax = plt.subplots(figsize=(6, 4))
    pivot = df.groupby(["complexity_level", "model"])["answer_correct"].mean().unstack()
    pivot = pivot.reindex(["L1", "L2", "L3", "L4"])
    pivot.plot.line(ax=ax, marker="o")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Answer accuracy")
    ax.set_title("Answer accuracy by complexity level (H4)")
    fig.tight_layout()
    fig.savefig(plots_dir / "accuracy_by_complexity.png", dpi=120)
    plt.close(fig)
    made.append("accuracy_by_complexity.png")

    # 3) Error category distribution.
    fig, ax = plt.subplots(figsize=(8, 4))
    err = df[df["error_category"] != "none"]
    if not err.empty:
        err.groupby(["error_category", "model"]).size().unstack(fill_value=0).plot.bar(ax=ax)
        ax.set_ylabel("count")
        ax.set_title("Error categories by model")
        fig.tight_layout()
        fig.savefig(plots_dir / "error_categories.png", dpi=120)
        made.append("error_categories.png")
    plt.close(fig)

    return made


def generate_reports(run_dir: Path, model_a: str, model_b: str) -> Path:
    """Generate all thesis reports for a run; return the run directory."""
    metrics_csv = run_dir / "metrics.csv"
    df = pd.read_csv(metrics_csv)

    results = run_all(metrics_csv, model_a, model_b)
    save_results(results, run_dir / "hypothesis_results.json")

    (run_dir / "summary.md").write_text(generate_summary_md(df), encoding="utf-8")
    (run_dir / "hypothesis_results.md").write_text(
        generate_hypothesis_md(results), encoding="utf-8"
    )
    (run_dir / "model_comparison.md").write_text(generate_model_comparison_md(df), encoding="utf-8")
    (run_dir / "error_analysis.md").write_text(generate_error_analysis_md(df), encoding="utf-8")
    _make_plots(df, run_dir / "plots")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate thesis-ready evaluation reports.")
    parser.add_argument("--run", type=Path, default=None, help="Run directory (default: latest).")
    args = parser.parse_args(argv)

    run_dir = args.run or latest_run()
    if run_dir is None:
        print("No evaluation run found. Run `python -m app.evaluation.runner` first.")
        return 1
    settings = get_settings()
    generate_reports(run_dir, settings.model_a, settings.model_b)
    print(f"Reports written to: {run_dir}")
    for name in [
        "summary.md",
        "hypothesis_results.md",
        "model_comparison.md",
        "error_analysis.md",
        "plots/",
    ]:
        print("  -", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
