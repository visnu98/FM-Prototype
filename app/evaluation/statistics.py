"""Statistical hypothesis testing for H1-H4 (Phase 15).

Consumes a run's ``metrics.csv`` and tests the four thesis hypotheses:

- H1: the system produces correct function calls for >= 90% of queries.
- H2: rewording (paraphrase) does not lower the correct-call rate.
- H3: function-calling reliability differs between the two models.
- H4: answer correctness decreases as complexity rises from L1 to L4.

"Correct call" = ``fully_correct_call`` (right function + parameters +
execution). "Answer correctness" = ``answer_correct``.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

ALPHA = 0.05
H1_THRESHOLD = 0.90


@dataclass
class HypothesisResult:
    hypothesis: str
    metric: str
    test: str
    statistic: float | None
    p_value: float | None
    result_value: dict[str, Any]
    supported: bool | None
    interpretation: str
    limitations: str = ""


# ── Helpers ──────────────────────────────────────────────────────────────────


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (float(max(0.0, centre - margin)), float(min(1.0, centre + margin)))


def _paired_pivot(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Pivot to query_id × model for a boolean column (paired by query)."""
    return df.pivot_table(index="query_id", columns="model", values=column, aggfunc="first")


# ── H1 ───────────────────────────────────────────────────────────────────────


def evaluate_h1(df: pd.DataFrame, threshold: float = H1_THRESHOLD) -> HypothesisResult:
    """One-sided binomial test that the correct-call rate >= threshold."""
    per_model: dict[str, Any] = {}
    for model, g in df.groupby("model"):
        successes = int(g["fully_correct_call"].sum())
        n = int(len(g))
        rate = successes / n if n else 0.0
        ci = _wilson_ci(successes, n)
        # H0: p = threshold; alternative: p > threshold (one-sided "greater").
        bt = stats.binomtest(successes, n, threshold, alternative="greater")
        per_model[model] = {
            "successes": successes,
            "n": n,
            "rate": round(float(rate), 3),
            "wilson_ci": [round(ci[0], 3), round(ci[1], 3)],
            "p_value_greater": float(bt.pvalue),
            "meets_threshold": bool(rate >= threshold),
        }
    # Overall (pooled).
    successes = int(df["fully_correct_call"].sum())
    n = int(len(df))
    rate = successes / n if n else 0.0
    bt = stats.binomtest(successes, n, threshold, alternative="greater")
    ci = _wilson_ci(successes, n)
    supported = rate >= threshold
    return HypothesisResult(
        hypothesis="H1",
        metric="fully_correct_call rate",
        test="binomial (one-sided, p>0.90), Wilson CI",
        statistic=None,
        p_value=bt.pvalue,
        result_value={
            "overall_rate": round(float(rate), 3),
            "overall_ci": [round(ci[0], 3), round(ci[1], 3)],
            "per_model": per_model,
        },
        supported=bool(supported),
        interpretation=(
            f"Pooled correct-call rate is {rate:.1%} "
            f"({'meets' if supported else 'below'} the 90% target)."
        ),
        limitations="Single dataset/facility; corpus authored by the developer.",
    )


# ── H2 ───────────────────────────────────────────────────────────────────────


def evaluate_h2(df: pd.DataFrame, metric: str = "fully_correct_call") -> HypothesisResult:
    """McNemar test pairing standard wording vs paraphrases within groups."""
    per_model: dict[str, Any] = {}
    pooled_b = pooled_c = 0  # discordant counts across models
    for model, g in df.groupby("model"):
        b = c = 0  # b: std correct & para wrong; c: std wrong & para correct
        std_correct = para_correct = std_n = para_n = 0
        for _, grp in g.groupby("paraphrase_group_id"):
            std_rows = grp[grp["is_standard_wording"]]
            para_rows = grp[~grp["is_standard_wording"]]
            if std_rows.empty or para_rows.empty:
                continue
            s = bool(std_rows.iloc[0][metric])
            std_correct += int(s)
            std_n += 1
            for _, prow in para_rows.iterrows():
                p = bool(prow[metric])
                para_correct += int(p)
                para_n += 1
                if s and not p:
                    b += 1
                elif (not s) and p:
                    c += 1
        pooled_b += b
        pooled_c += c
        table = [[0, b], [c, 0]]
        res = mcnemar(table, exact=True)
        per_model[model] = {
            "standard_rate": round(std_correct / std_n, 3) if std_n else None,
            "paraphrase_rate": round(para_correct / para_n, 3) if para_n else None,
            "discordant_std_correct_para_wrong": b,
            "discordant_std_wrong_para_correct": c,
            "p_value": float(res.pvalue),
        }
    pooled_table = [[0, pooled_b], [pooled_c, 0]]
    pooled = mcnemar(pooled_table, exact=True)
    # H2 supported if paraphrasing does NOT significantly lower correctness.
    supported = pooled.pvalue >= ALPHA
    return HypothesisResult(
        hypothesis="H2",
        metric=f"{metric} (standard vs paraphrase)",
        test="McNemar (exact), paired within paraphrase groups",
        statistic=float(pooled.statistic) if pooled.statistic is not None else None,
        p_value=float(pooled.pvalue),
        result_value={
            "pooled_discordant": {
                "std_correct_para_wrong": pooled_b,
                "std_wrong_para_correct": pooled_c,
            },
            "per_model": per_model,
        },
        supported=bool(supported),
        interpretation=(
            "No significant difference between standard and reworded queries "
            f"(p={pooled.pvalue:.3f}); paraphrasing did not lower correctness."
            if supported
            else f"Paraphrasing significantly changed correctness (p={pooled.pvalue:.3f})."
        ),
        limitations="Few discordant pairs make McNemar low-powered on a small corpus.",
    )


# ── H3 ───────────────────────────────────────────────────────────────────────


def evaluate_h3(
    df: pd.DataFrame, model_a: str, model_b: str, metric: str = "fully_correct_call"
) -> HypothesisResult:
    """McNemar on correctness + Wilcoxon on latency between the two models."""
    correct = _paired_pivot(df, metric)
    latency = _paired_pivot(df, "latency_total")
    for m in (model_a, model_b):
        if m not in correct.columns:
            return HypothesisResult(
                "H3",
                metric,
                "McNemar + Wilcoxon",
                None,
                None,
                {"error": f"model '{m}' not in results"},
                None,
                "Both models are required for H3.",
                "",
            )
    paired = correct[[model_a, model_b]].dropna().astype(bool)
    b = int((paired[model_a] & ~paired[model_b]).sum())
    c = int((~paired[model_a] & paired[model_b]).sum())
    mc = mcnemar([[0, b], [c, 0]], exact=True)

    lat = latency[[model_a, model_b]].dropna()
    if len(lat) >= 1 and not np.allclose(lat[model_a], lat[model_b]):
        try:
            w_stat, w_p = stats.wilcoxon(lat[model_a], lat[model_b])
        except ValueError:
            w_stat, w_p = (float("nan"), float("nan"))
    else:
        w_stat, w_p = (float("nan"), float("nan"))

    supported = mc.pvalue < ALPHA
    rate_a = float(paired[model_a].mean())
    rate_b = float(paired[model_b].mean())
    return HypothesisResult(
        hypothesis="H3",
        metric=f"{metric} + latency_total",
        test="McNemar (exact) on correctness; Wilcoxon signed-rank on latency",
        statistic=float(mc.statistic) if mc.statistic is not None else None,
        p_value=float(mc.pvalue),
        result_value={
            "model_a": model_a,
            "model_b": model_b,
            "correct_rate_a": round(rate_a, 3),
            "correct_rate_b": round(rate_b, 3),
            "discordant_a_correct_b_wrong": b,
            "discordant_b_correct_a_wrong": c,
            "latency_wilcoxon_stat": None if np.isnan(w_stat) else float(w_stat),
            "latency_wilcoxon_p": None if np.isnan(w_p) else float(w_p),
            "median_latency_a_ms": round(float(lat[model_a].median()), 1) if len(lat) else None,
            "median_latency_b_ms": round(float(lat[model_b].median()), 1) if len(lat) else None,
        },
        supported=bool(supported),
        interpretation=(
            f"Reliability differs between models (McNemar p={mc.pvalue:.3f})."
            if supported
            else f"No significant reliability difference (McNemar p={mc.pvalue:.3f})."
        ),
        limitations="McNemar needs discordant pairs; latency depends on network/load.",
    )


# ── H4 ───────────────────────────────────────────────────────────────────────


def evaluate_h4(df: pd.DataFrame, metric: str = "answer_correct") -> HypothesisResult:
    """Trend test for decreasing correctness across L1->L4 (logistic on level)."""
    level_map = {"L1": 1, "L2": 2, "L3": 3, "L4": 4}
    d = df.copy()
    d["level_num"] = d["complexity_level"].map(level_map)
    d = d.dropna(subset=["level_num"])
    rates = d.groupby("complexity_level")[metric].mean().reindex(["L1", "L2", "L3", "L4"]).dropna()

    slope = p_value = None
    test_name = "logistic regression (correctness ~ level)"
    try:
        import statsmodels.api as sm

        x = sm.add_constant(d["level_num"].astype(float))
        y = d[metric].astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore separation/convergence on small data
            model = sm.Logit(y, x).fit(disp=False)
        slope = float(model.params["level_num"])
        p_value = float(model.pvalues["level_num"])
    except Exception:
        # Fallback: Spearman correlation between level and correctness.
        test_name = "Spearman correlation (level vs correctness)"
        if d["level_num"].nunique() > 1:
            rho, p_value = stats.spearmanr(d["level_num"], d[metric].astype(int))
            slope = float(rho)

    monotonic_decrease = bool(all(np.diff(rates.values) <= 1e-9)) if len(rates) > 1 else False
    significant_decrease = (slope is not None and slope < 0) and (
        p_value is not None and p_value < ALPHA
    )
    supported = bool(significant_decrease or monotonic_decrease)
    return HypothesisResult(
        hypothesis="H4",
        metric=f"{metric} by complexity level",
        test=test_name,
        statistic=slope,
        p_value=p_value,
        result_value={
            "rate_by_level": {k: round(float(v), 3) for k, v in rates.items()},
            "monotonic_decrease": monotonic_decrease,
            "slope_negative": (slope is not None and slope < 0),
        },
        supported=supported,
        interpretation=(
            "Answer correctness decreases with complexity "
            f"(slope={slope:.3f}, p={p_value if p_value is None else round(p_value, 3)})."
            if supported
            else "No clear decreasing trend in correctness across complexity levels."
        ),
        limitations="Few queries per level limit statistical power.",
    )


# ── Orchestration ────────────────────────────────────────────────────────────


def run_all(metrics_csv: Path, model_a: str, model_b: str) -> list[HypothesisResult]:
    df = pd.read_csv(metrics_csv)
    return [
        evaluate_h1(df),
        evaluate_h2(df),
        evaluate_h3(df, model_a, model_b),
        evaluate_h4(df),
    ]


def save_results(results: list[HypothesisResult], out_path: Path) -> None:
    out_path.write_text(
        json.dumps([asdict(r) for r in results], indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    from app.config import get_settings
    from app.evaluation.runner import latest_run

    parser = argparse.ArgumentParser(description="Run H1-H4 statistical tests on a run.")
    parser.add_argument("--run", type=Path, default=None, help="Run directory (default: latest).")
    args = parser.parse_args(argv)

    run_dir = args.run or latest_run()
    if run_dir is None:
        print("No evaluation run found. Run `python -m app.evaluation.runner` first.")
        return 1
    settings = get_settings()
    results = run_all(run_dir / "metrics.csv", settings.model_a, settings.model_b)
    save_results(results, run_dir / "hypothesis_results.json")
    for r in results:
        print(f"{r.hypothesis}: supported={r.supported} | {r.interpretation}")
    print("Saved:", run_dir / "hypothesis_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
