"""One-time database understanding (Phases 3-4).

These scripts are run once, early, to learn the live database before any FM
functions are written. They are not part of the request-time chatbot pipeline.

- ``schema_discovery`` — inspect the live DB -> CSV reports + Markdown summary.
- ``data_dictionary``  — turn those reports into a human-readable data dictionary.
"""
