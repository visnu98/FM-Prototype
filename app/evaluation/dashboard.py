"""Streamlit evaluation dashboard.

Browse an evaluation run interactively: headline metrics, charts, a per-query
explorer (with the tool chain each model used and its final answer), and the
error taxonomy.

Run with::

    streamlit run app/evaluation/dashboard.py

It is read-only: it loads ``data/evaluation/runs/<timestamp>/raw_results.jsonl``
written by the evaluation runner.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.core.config import PROJECT_ROOT

RUNS_DIR = PROJECT_ROOT / "data" / "evaluation" / "runs"


# ── Loading ──────────────────────────────────────────────────────────────────


def _list_runs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted((p for p in RUNS_DIR.iterdir() if p.is_dir()), reverse=True)


@st.cache_data(show_spinner=False)
def _load_raw(run: str) -> pd.DataFrame:
    path = Path(run) / "raw_results.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return pd.DataFrame(rows)


# ── Sections ─────────────────────────────────────────────────────────────────


def _overview(df: pd.DataFrame) -> None:
    st.subheader("Headline metrics by model")
    agg = df.groupby("model").agg(
        n=("query_id", "count"),
        fully_correct_call=("fully_correct_call", "mean"),
        function_accuracy=("function_correct", "mean"),
        parameter_accuracy=("parameters_correct", "mean"),
        answer_accuracy=("answer_correct", "mean"),
        execution_success=("execution_success", "mean"),
        mean_latency_ms=("latency_total", "mean"),
    )
    st.dataframe(agg.style.format("{:.3f}", subset=agg.columns[1:]), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Fully-correct-call rate by model (H1 target = 0.90)")
        st.bar_chart(df.groupby("model")["fully_correct_call"].mean())
    with c2:
        st.caption("Answer accuracy by complexity level (H4)")
        pivot = (
            df.groupby(["complexity_level", "model"])["answer_correct"]
            .mean()
            .unstack()
            .reindex(["L1", "L2", "L3", "L4"])
        )
        st.line_chart(pivot)

    st.caption("Fully-correct-call rate by category")
    st.bar_chart(df.groupby(["category", "model"])["fully_correct_call"].mean().unstack())


def _explorer(df: pd.DataFrame) -> None:
    st.subheader("Per-query explorer")
    models = sorted(df["model"].unique())
    categories = sorted(df["category"].unique())
    levels = sorted(df["complexity_level"].unique())

    f1, f2, f3, f4 = st.columns(4)
    sel_models = f1.multiselect("Model", models, default=models)
    sel_cats = f2.multiselect("Category", categories, default=categories)
    sel_levels = f3.multiselect("Complexity", levels, default=levels)
    only_fail = f4.checkbox("Only failures", value=False)

    view = df[
        df["model"].isin(sel_models)
        & df["category"].isin(sel_cats)
        & df["complexity_level"].isin(sel_levels)
    ]
    if only_fail:
        view = view[~view["fully_correct_call"] | ~view["answer_correct"]]

    display = view.copy()
    display["expected_functions"] = display["expected_functions"].apply(_join)
    display["actual_functions"] = display["actual_functions"].apply(_join)
    cols = [
        "query_id",
        "model",
        "category",
        "complexity_level",
        "expected_functions",
        "actual_functions",
        "num_steps",
        "function_correct",
        "answer_correct",
        "fully_correct_call",
        "error_category",
    ]
    st.dataframe(display[cols], width="stretch", height=360, hide_index=True)
    st.caption(
        f"{len(view)} rows · "
        f"{int(view['fully_correct_call'].sum())} fully-correct · "
        f"{int(view['answer_correct'].sum())} answer-correct"
    )

    # Drill-down on one query.
    st.markdown("#### Drill-down")
    qids = sorted(view["query_id"].unique())
    if not qids:
        return
    qid = st.selectbox("Query", qids)
    rows = view[view["query_id"] == qid]
    st.write(f"**Question:** {rows.iloc[0]['query_text']}")
    for _, row in rows.iterrows():
        ok = "✅" if row["fully_correct_call"] and row["answer_correct"] else "⚠️"
        with st.expander(f"{ok} {row['model']} — {row['error_category']}", expanded=True):
            st.write(
                f"**Expected functions:** `{_join(row['expected_functions'])}`  ·  "
                f"**Actual chain:** `{_join(row['actual_functions'])}`"
            )
            st.caption(f"expected answer value(s): {row.get('expected_answer_values')}")
            st.markdown("**Tool calls (steps):**")
            st.json(row.get("calls") if isinstance(row.get("calls"), list) else [])
            st.markdown("**Model's final answer:**")
            st.info(row["final_answer"])


def _join(value) -> str:  # type: ignore[no-untyped-def]
    """Render a list (or scalar) of function names as 'a → b'."""
    if isinstance(value, list):
        return " → ".join(str(v) for v in value)
    return str(value)


def _errors(df: pd.DataFrame) -> None:
    st.subheader("Error taxonomy")
    counts = (
        df[df["error_category"] != "none"]
        .groupby(["error_category", "model"])
        .size()
        .unstack(fill_value=0)
    )
    if counts.empty:
        st.success("No errors recorded in this run.")
        return
    st.bar_chart(counts)
    st.dataframe(counts, width="stretch")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(page_title="FM Evaluation Dashboard", page_icon="📊", layout="wide")
    st.title("📊 Function-Calling Evaluation Dashboard")

    runs = _list_runs()
    if not runs:
        st.warning("No evaluation runs found. Run `python -m app.evaluation.runner` first.")
        return

    run = st.sidebar.selectbox("Run", [str(p) for p in runs], format_func=lambda p: Path(p).name)
    st.sidebar.caption(f"Loaded: {Path(run).name}")

    df = _load_raw(run)
    if df.empty:
        st.error("This run has no raw_results.jsonl.")
        return

    st.sidebar.metric("Queries", df["query_id"].nunique())
    st.sidebar.metric("Models", df["model"].nunique())
    st.sidebar.metric("Total runs", len(df))

    tabs = st.tabs(["Overview", "Per-query explorer", "Errors"])
    with tabs[0]:
        _overview(df)
    with tabs[1]:
        _explorer(df)
    with tabs[2]:
        _errors(df)


if __name__ == "__main__":
    main()
