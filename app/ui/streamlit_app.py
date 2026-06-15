"""Streamlit chatbot UI (Phase 9).

Run with::

    streamlit run app/ui/streamlit_app.py

Features: chat interface, model selector (MODEL_A / MODEL_B), and an optional
function-call trace panel (selected function, raw + normalized arguments, tool
result, per-step latency, error category). No secrets are ever displayed.
"""

from __future__ import annotations

import json

import streamlit as st

from app.chatbot.service import ChatbotService
from app.config import get_settings


@st.cache_resource(show_spinner=False)
def _service(model_name: str) -> ChatbotService:
    """Cache one service per model (registry + client are reused)."""
    return ChatbotService.for_model(model_name)


def main() -> None:
    settings = get_settings()
    st.set_page_config(page_title="FM Function-Calling Assistant", page_icon="🏢")
    st.title("🏢 Facility Management Assistant")
    st.caption(
        "Natural-language retrieval of structured FM/BIM data via controlled "
        f"function calling. Facility: {settings.default_facility_id}."
    )

    with st.sidebar:
        st.header("Settings")
        model_name = st.selectbox("Model", settings.models(), index=0)
        show_trace = st.checkbox("Show function-call trace", value=True)
        if st.button("Clear conversation"):
            st.session_state.history = []
        st.markdown(
            "**Examples**\n"
            "- What floors can I query?\n"
            "- How many windows are on the second floor?\n"
            "- Total window area on the first floor?\n"
            "- Which floor has the largest window area?"
        )

    if "history" not in st.session_state:
        st.session_state.history = []

    # Replay history.
    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["query"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if show_trace and turn.get("trace"):
                _render_trace(turn["trace"])

    prompt = st.chat_input("Ask about floors, components, counts or areas…")
    if not prompt:
        return

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Thinking with {model_name}…"):
            response = _service(model_name).answer(prompt)
        st.write(response.final_answer)
        trace = _trace_dict(response)
        if show_trace:
            _render_trace(trace)

    st.session_state.history.append(
        {"query": prompt, "answer": response.final_answer, "trace": trace}
    )


def _trace_dict(response) -> dict:  # type: ignore[no-untyped-def]
    tr = response.tool_result
    return {
        "model": response.model_name,
        "selected_function": response.selected_function,
        "arguments": response.arguments,
        "normalized_arguments": response.normalized_arguments,
        "error_category": response.error_category.value,
        "made_tool_call": response.made_tool_call,
        "result_ok": (tr.ok if tr else None),
        "result_data": (tr.data if tr else None),
        "latency_ms": {
            "tool_call": round(response.latency_tool_call_ms),
            "sql": round(response.latency_sql_ms),
            "final_answer": round(response.latency_final_answer_ms),
            "total": round(response.latency_total_ms),
        },
    }


def _render_trace(trace: dict) -> None:
    with st.expander("🔎 Function-call trace", expanded=False):
        st.markdown(
            f"**Function:** `{trace['selected_function']}`  \n"
            f"**Error category:** `{trace['error_category']}`  \n"
            f"**Latency (ms):** {trace['latency_ms']}"
        )
        st.markdown("**Arguments (raw):**")
        st.json(trace["arguments"])
        if trace.get("normalized_arguments"):
            st.markdown("**Normalized arguments:**")
            st.json(trace["normalized_arguments"])
        st.markdown("**Tool result:**")
        st.json(_safe(trace.get("result_data")))


def _safe(data) -> object:  # type: ignore[no-untyped-def]
    """Make data JSON-displayable (Streamlit handles dict/list natively)."""
    try:
        json.dumps(data, default=str)
        return data
    except (TypeError, ValueError):
        return str(data)


if __name__ == "__main__":
    main()
