"""Tests for candidate detection (no database required)."""

from __future__ import annotations

import pandas as pd

from app.discovery.schema_discovery import detect_candidates


def test_detect_candidates_flags_bim_terms() -> None:
    columns = pd.DataFrame(
        [
            {
                "table_schema": "public",
                "table_name": "asset_component",
                "column_name": "ext_object",
                "data_type": "text",
            },
            {
                "table_schema": "public",
                "table_name": "asset_component",
                "column_name": "area",
                "data_type": "numeric",
            },
            {
                "table_schema": "public",
                "table_name": "unrelated",
                "column_name": "note",
                "data_type": "text",
            },
        ]
    )
    result = detect_candidates(columns)
    flagged = set(zip(result["table_name"], result["column_name"], strict=False))
    assert ("asset_component", "ext_object") in flagged
    assert ("asset_component", "area") in flagged
    # "unrelated.note" matches no BIM/FM term and must not be flagged.
    assert ("unrelated", "note") not in flagged


def test_detect_candidates_empty_input() -> None:
    result = detect_candidates(
        pd.DataFrame(columns=["table_schema", "table_name", "column_name", "data_type"])
    )
    assert result.empty
