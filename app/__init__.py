"""Function-calling FM/BIM question-answering prototype.

Package layout:

- ``core``                   — settings (``core.config``) and the read-only
                               PostgreSQL layer (``core.db``).
- ``tools``                  — the controlled function-calling core (models,
                               registry, normalization, SQL-backed FM functions).
- ``llm``                    — the model-agnostic tool-calling clients (Groq).
- ``chatbot``                — the question -> tool -> grounded-answer pipeline.
- ``ui``                     — Streamlit apps (chatbot + evaluation dashboard).
- ``evaluation``             — corpus, ground truth, metrics, runner, statistics
                               and report generation for H1-H4.
"""
