"""Function-calling FM/BIM question-answering prototype.

Package layout:

- ``core``                   — settings (``core.config``) and the read-only
                               PostgreSQL layer (``core.db``).
- ``tools``                  — the controlled function-calling core (models,
                               registry, normalization, SQL-backed FM functions).
- ``llm``                    — the model-agnostic tool-calling clients (Groq).
- ``chatbot``                — the question -> tool -> grounded-answer pipeline,
                               plus the Streamlit chat UI (``chatbot.ui``).
- ``evaluation``             — corpus, ground truth, metrics, runner, statistics,
                               report generation and the results dashboard
                               (``evaluation.dashboard``) for H1-H4.
"""
