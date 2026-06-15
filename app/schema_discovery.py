"""Database schema discovery and analysis (Phase 3).

Inspects a live PostgreSQL database and produces machine-readable CSV reports
plus a human-readable Markdown summary. NOTHING about the schema is assumed:
table names, column names, floor labels and component types are all derived
from the actual database.

Run it as a module::

    python -m app.schema_discovery

All catalog queries are read-only and parameterised. Row sampling is limited
and columns whose names look sensitive (password/token/secret/...) are redacted.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import PROJECT_ROOT
from app.db import execute_read_query, test_connection

logger = logging.getLogger(__name__)

REPORTS_DIR = PROJECT_ROOT / "data" / "schema_reports"

# Maximum number of rows sampled per table.
SAMPLE_ROW_LIMIT = 5

# Substrings that mark a column as potentially sensitive -> values redacted.
_SENSITIVE_COLUMN_TOKENS = (
    "password",
    "passwd",
    "pwd",
    "secret",
    "token",
    "apikey",
    "api_key",
    "private",
    "credential",
    "hash",
    "salt",
    "ssn",
)

# Domain vocabulary that hints a column/table is BIM/FM-relevant (Sub-RQ 1/2).
# Derived from the thesis brief; used only to *flag candidates*, never to assume.
BIM_FM_TERMS = (
    "facility",
    "project",
    "building",
    "floor",
    "storey",
    "story",
    "level",
    "geschoss",  # German: Geschoss / Obergeschoss
    "etage",
    "room",
    "space",
    "raum",
    "asset",
    "component",
    "element",
    "equipment",
    "ifc",
    "object",
    "guid",
    "globalid",
    "global_id",
    "externalid",
    "external_id",
    "extobject",
    "type",
    "category",
    "attribute",
    "property",
    "propertyset",
    "pset",
    "width",
    "height",
    "length",
    "area",
    "flaeche",  # German: Fläche
    "volume",
    "maintenance",
    "wartung",
    "inspection",
    "schedule",
    "date",
)


# ── Data container ───────────────────────────────────────────────────────────


@dataclass
class DiscoveryResult:
    """In-memory holder of all discovery dataframes, for tests and reuse."""

    schemas: pd.DataFrame = field(default_factory=pd.DataFrame)
    tables: pd.DataFrame = field(default_factory=pd.DataFrame)
    columns: pd.DataFrame = field(default_factory=pd.DataFrame)
    primary_keys: pd.DataFrame = field(default_factory=pd.DataFrame)
    foreign_keys: pd.DataFrame = field(default_factory=pd.DataFrame)
    indexes: pd.DataFrame = field(default_factory=pd.DataFrame)
    row_counts: pd.DataFrame = field(default_factory=pd.DataFrame)
    sample_values: pd.DataFrame = field(default_factory=pd.DataFrame)
    relevant_candidates: pd.DataFrame = field(default_factory=pd.DataFrame)


# ── Catalog queries ──────────────────────────────────────────────────────────


def list_schemas() -> pd.DataFrame:
    """Return all non-system schemas."""
    rows = execute_read_query("""
        SELECT schema_name
        FROM information_schema.schemata
        WHERE schema_name NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
          AND schema_name NOT LIKE 'pg_temp_%'
          AND schema_name NOT LIKE 'pg_toast_temp_%'
        ORDER BY schema_name
        """)
    return pd.DataFrame(rows, columns=["schema_name"])


def list_tables() -> pd.DataFrame:
    """Return all base tables and views in non-system schemas."""
    rows = execute_read_query("""
        SELECT table_schema, table_name, table_type
        FROM information_schema.tables
        WHERE table_schema NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
          AND table_schema NOT LIKE 'pg_temp_%'
          AND table_schema NOT LIKE 'pg_toast_temp_%'
        ORDER BY table_schema, table_name
        """)
    return pd.DataFrame(rows, columns=["table_schema", "table_name", "table_type"])


def list_columns() -> pd.DataFrame:
    """Return every column with type / nullability / default / size metadata."""
    rows = execute_read_query("""
        SELECT
            table_schema,
            table_name,
            column_name,
            ordinal_position,
            data_type,
            is_nullable,
            column_default,
            character_maximum_length,
            numeric_precision,
            numeric_scale
        FROM information_schema.columns
        WHERE table_schema NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
          AND table_schema NOT LIKE 'pg_temp_%'
        ORDER BY table_schema, table_name, ordinal_position
        """)
    return pd.DataFrame(
        rows,
        columns=[
            "table_schema",
            "table_name",
            "column_name",
            "ordinal_position",
            "data_type",
            "is_nullable",
            "column_default",
            "character_maximum_length",
            "numeric_precision",
            "numeric_scale",
        ],
    )


def list_primary_keys() -> pd.DataFrame:
    """Return primary-key columns per table (ordered)."""
    rows = execute_read_query("""
        SELECT
            tc.table_schema,
            tc.table_name,
            kcu.column_name,
            kcu.ordinal_position,
            tc.constraint_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
        WHERE tc.constraint_type = 'PRIMARY KEY'
          AND tc.table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY tc.table_schema, tc.table_name, kcu.ordinal_position
        """)
    return pd.DataFrame(
        rows,
        columns=[
            "table_schema",
            "table_name",
            "column_name",
            "ordinal_position",
            "constraint_name",
        ],
    )


def list_foreign_keys() -> pd.DataFrame:
    """Return foreign-key relationships (child -> parent)."""
    rows = execute_read_query("""
        SELECT
            tc.table_schema      AS table_schema,
            tc.table_name        AS table_name,
            kcu.column_name      AS column_name,
            ccu.table_schema     AS foreign_table_schema,
            ccu.table_name       AS foreign_table_name,
            ccu.column_name      AS foreign_column_name,
            tc.constraint_name   AS constraint_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage ccu
          ON ccu.constraint_name = tc.constraint_name
         AND ccu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY tc.table_schema, tc.table_name, kcu.column_name
        """)
    return pd.DataFrame(
        rows,
        columns=[
            "table_schema",
            "table_name",
            "column_name",
            "foreign_table_schema",
            "foreign_table_name",
            "foreign_column_name",
            "constraint_name",
        ],
    )


def list_indexes() -> pd.DataFrame:
    """Return index definitions (uses the pg_indexes catalog view)."""
    rows = execute_read_query("""
        SELECT schemaname AS table_schema,
               tablename   AS table_name,
               indexname   AS index_name,
               indexdef    AS index_def
        FROM pg_indexes
        WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
        ORDER BY schemaname, tablename, indexname
        """)
    return pd.DataFrame(rows, columns=["table_schema", "table_name", "index_name", "index_def"])


# ── Identifier quoting (for dynamic but safe object names) ───────────────────


def _quote_ident(identifier: str) -> str:
    """Safely quote a SQL identifier (schema/table name).

    Table and schema names come from the catalog, not user input, but we still
    quote defensively by doubling embedded double-quotes.
    """
    return '"' + identifier.replace('"', '""') + '"'


def count_rows(tables: pd.DataFrame) -> pd.DataFrame:
    """Count rows per base table. Views are skipped (could be expensive)."""
    records: list[dict[str, Any]] = []
    base_tables = tables[tables["table_type"] == "BASE TABLE"]
    for _, row in base_tables.iterrows():
        schema, table = row["table_schema"], row["table_name"]
        qualified = f"{_quote_ident(schema)}.{_quote_ident(table)}"
        try:
            # COUNT(*) on a quoted, catalog-sourced identifier; no user input.
            result = execute_read_query(f"SELECT COUNT(*) AS n FROM {qualified}")
            n = int(result[0]["n"]) if result else 0
        except Exception as exc:  # pragma: no cover - permission/lock dependent
            logger.warning("Row count failed for %s.%s: %s", schema, table, type(exc).__name__)
            n = -1
        records.append({"table_schema": schema, "table_name": table, "row_count": n})
    df = pd.DataFrame(records, columns=["table_schema", "table_name", "row_count"])
    return df.sort_values("row_count", ascending=False, ignore_index=True)


def _is_sensitive_column(column_name: str) -> bool:
    lower = column_name.lower()
    return any(tok in lower for tok in _SENSITIVE_COLUMN_TOKENS)


def sample_table_rows(
    tables: pd.DataFrame, columns: pd.DataFrame, limit: int = SAMPLE_ROW_LIMIT
) -> pd.DataFrame:
    """Safely sample up to ``limit`` rows per base table.

    Output is long-format: one row per (table, sample_index, column, value).
    Sensitive columns are redacted; long values are truncated.
    """
    records: list[dict[str, Any]] = []
    base_tables = tables[tables["table_type"] == "BASE TABLE"]
    cols_by_table = {
        (s, t): grp["column_name"].tolist()
        for (s, t), grp in columns.groupby(["table_schema", "table_name"])
    }

    for _, row in base_tables.iterrows():
        schema, table = row["table_schema"], row["table_name"]
        qualified = f"{_quote_ident(schema)}.{_quote_ident(table)}"
        table_cols = cols_by_table.get((schema, table), [])
        sensitive = {c for c in table_cols if _is_sensitive_column(c)}
        try:
            sample = execute_read_query(f"SELECT * FROM {qualified} LIMIT {int(limit)}")
        except Exception as exc:  # pragma: no cover
            logger.warning("Sampling failed for %s.%s: %s", schema, table, type(exc).__name__)
            continue
        for idx, sample_row in enumerate(sample):
            for col, value in sample_row.items():
                rendered = "***redacted***" if col in sensitive else _truncate(value)
                records.append(
                    {
                        "table_schema": schema,
                        "table_name": table,
                        "sample_index": idx,
                        "column_name": col,
                        "value": rendered,
                    }
                )
    return pd.DataFrame(
        records,
        columns=["table_schema", "table_name", "sample_index", "column_name", "value"],
    )


def _truncate(value: Any, max_len: int = 120) -> str:
    text_value = "" if value is None else str(value)
    if len(text_value) > max_len:
        return text_value[:max_len] + "…"
    return text_value


# ── Candidate detection ──────────────────────────────────────────────────────


def detect_candidates(columns: pd.DataFrame) -> pd.DataFrame:
    """Flag columns whose table or column name matches BIM/FM vocabulary.

    Produces one row per matched column with the matching term(s), so the data
    dictionary (Phase 4) can group these into candidate domain tables.
    """
    records: list[dict[str, Any]] = []
    for _, row in columns.iterrows():
        haystack = f"{row['table_name']} {row['column_name']}".lower()
        matched = sorted({term for term in BIM_FM_TERMS if term in haystack})
        if matched:
            records.append(
                {
                    "table_schema": row["table_schema"],
                    "table_name": row["table_name"],
                    "column_name": row["column_name"],
                    "data_type": row["data_type"],
                    "matched_terms": ", ".join(matched),
                }
            )
    return pd.DataFrame(
        records,
        columns=[
            "table_schema",
            "table_name",
            "column_name",
            "data_type",
            "matched_terms",
        ],
    )


# ── Orchestration ────────────────────────────────────────────────────────────


def discover() -> DiscoveryResult:
    """Run the full discovery pipeline and return all dataframes."""
    logger.info("Listing schemas…")
    schemas = list_schemas()
    logger.info("Listing tables…")
    tables = list_tables()
    logger.info("Listing columns…")
    columns = list_columns()
    logger.info("Detecting primary keys…")
    primary_keys = list_primary_keys()
    logger.info("Detecting foreign keys…")
    foreign_keys = list_foreign_keys()
    logger.info("Listing indexes…")
    indexes = list_indexes()
    logger.info("Counting rows (%d base tables)…", (tables["table_type"] == "BASE TABLE").sum())
    row_counts = count_rows(tables)
    logger.info("Sampling rows…")
    sample_values = sample_table_rows(tables, columns)
    logger.info("Detecting BIM/FM candidate columns…")
    relevant_candidates = detect_candidates(columns)

    return DiscoveryResult(
        schemas=schemas,
        tables=tables,
        columns=columns,
        primary_keys=primary_keys,
        foreign_keys=foreign_keys,
        indexes=indexes,
        row_counts=row_counts,
        sample_values=sample_values,
        relevant_candidates=relevant_candidates,
    )


# ── Reporting ────────────────────────────────────────────────────────────────


def _candidate_tables_by_terms(candidates: pd.DataFrame, terms: tuple[str, ...]) -> list[str]:
    """Return distinct 'schema.table' whose matched terms intersect ``terms``."""
    if candidates.empty:
        return []
    term_set = set(terms)
    hits: set[str] = set()
    for _, row in candidates.iterrows():
        matched = {t.strip() for t in str(row["matched_terms"]).split(",")}
        if matched & term_set:
            hits.add(f"{row['table_schema']}.{row['table_name']}")
    return sorted(hits)


def build_summary_markdown(result: DiscoveryResult) -> str:
    """Assemble the human-readable schema_summary.md content."""
    lines: list[str] = []
    lines.append("# Database Schema Summary\n")
    lines.append(
        "_Auto-generated by `python -m app.schema_discovery`. "
        "Nothing here is assumed — every fact is read from the live database._\n"
    )

    n_schemas = len(result.schemas)
    n_tables = len(result.tables)
    n_base = int((result.tables["table_type"] == "BASE TABLE").sum()) if n_tables else 0
    n_columns = len(result.columns)
    lines.append("## Overview\n")
    lines.append(f"- Total schemas: **{n_schemas}**")
    lines.append(f"- Total tables/views: **{n_tables}** (base tables: {n_base})")
    lines.append(f"- Total columns: **{n_columns}**")
    lines.append(f"- Foreign-key relationships: **{len(result.foreign_keys)}**\n")

    # Largest tables
    lines.append("## Largest tables by row count\n")
    if not result.row_counts.empty:
        top = result.row_counts.head(15)
        lines.append("| schema.table | rows |")
        lines.append("| --- | ---: |")
        for _, r in top.iterrows():
            lines.append(f"| {r['table_schema']}.{r['table_name']} | {r['row_count']:,} |")
        lines.append("")
    else:
        lines.append("_No row counts available._\n")

    # Candidate groupings
    groups = {
        "Candidate BIM/FM tables": BIM_FM_TERMS,
        "Candidate asset/component tables": (
            "asset",
            "component",
            "element",
            "object",
            "ifc",
            "extobject",
        ),
        "Candidate floor/storey tables": (
            "floor",
            "storey",
            "story",
            "level",
            "geschoss",
            "etage",
        ),
        "Candidate attribute/property tables": (
            "attribute",
            "property",
            "propertyset",
            "pset",
        ),
        "Candidate maintenance/equipment tables": (
            "maintenance",
            "wartung",
            "inspection",
            "equipment",
            "schedule",
        ),
        "Candidate geometry columns (area/width/height/length/volume)": (
            "area",
            "flaeche",
            "width",
            "height",
            "length",
            "volume",
        ),
    }
    for title, terms in groups.items():
        lines.append(f"## {title}\n")
        hits = _candidate_tables_by_terms(result.relevant_candidates, terms)
        if hits:
            for h in hits:
                lines.append(f"- {h}")
        else:
            lines.append("_None detected by name matching._")
        lines.append("")

    # Foreign-key relationships
    lines.append("## Possible relationships (foreign keys)\n")
    if not result.foreign_keys.empty:
        lines.append("| child table.column | → | parent table.column |")
        lines.append("| --- | :-: | --- |")
        for _, r in result.foreign_keys.iterrows():
            child = f"{r['table_schema']}.{r['table_name']}.{r['column_name']}"
            parent = (
                f"{r['foreign_table_schema']}.{r['foreign_table_name']}"
                f".{r['foreign_column_name']}"
            )
            lines.append(f"| {child} | → | {parent} |")
        lines.append("")
    else:
        lines.append(
            "_No declared foreign keys found. Relationships may be implicit "
            "(by naming convention) and must be confirmed manually._\n"
        )

    # Open questions
    lines.append("## Open questions requiring manual confirmation\n")
    lines.append(
        "- Which table holds **asset components** and which column carries the "
        "**IFC type** (e.g. `IfcWindow`, `IfcDoor`)?"
    )
    lines.append(
        "- How is the **floor / storey** of a component represented — a foreign "
        "key, a label column, or encoded in the component name?"
    )
    lines.append(
        "- Where are component **attributes/properties** (area, width, height) "
        "stored — wide columns or a key/value attribute table?"
    )
    lines.append(
        "- Are there usable **equipment** and **maintenance** tables, or are "
        "those functions out of scope for this dataset?"
    )
    lines.append(
        "- Which columns scope data to a **facility/project** (for multi-tenant " "filtering)?\n"
    )

    return "\n".join(lines)


def export_reports(result: DiscoveryResult, reports_dir: Path = REPORTS_DIR) -> None:
    """Write all CSV reports and the Markdown summary to ``reports_dir``."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_map = {
        "schemas.csv": result.schemas,
        "tables.csv": result.tables,
        "columns.csv": result.columns,
        "primary_keys.csv": result.primary_keys,
        "foreign_keys.csv": result.foreign_keys,
        "indexes.csv": result.indexes,
        "row_counts.csv": result.row_counts,
        "sample_values.csv": result.sample_values,
        "relevant_candidates.csv": result.relevant_candidates,
    }
    for filename, df in csv_map.items():
        path = reports_dir / filename
        df.to_csv(path, index=False, encoding="utf-8")
        logger.info("Wrote %s (%d rows)", path.name, len(df))

    summary_path = reports_dir / "schema_summary.md"
    summary_path.write_text(build_summary_markdown(result), encoding="utf-8")
    logger.info("Wrote %s", summary_path.name)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect the PostgreSQL database and generate schema reports."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR,
        help="Directory for the generated reports.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not test_connection():
        logger.error("Could not connect to the database. Check your .env settings.")
        return 1

    result = discover()
    export_reports(result, args.output)
    print(f"\nSchema discovery complete. Reports written to: {args.output}")
    print("Next: run `python -m app.data_dictionary` to build the data dictionary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
