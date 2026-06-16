"""Data dictionary generation (Phase 4).

Reads the Phase 3 CSV reports and produces a human-readable data dictionary
(`data/schema_reports/data_dictionary.md`) that maps the *discovered* database
onto the FM/BIM concepts the prototype needs.

The dictionary is deliberately cautious: every candidate carries a confidence
level and a "manual confirmation needed" flag. It does NOT assume the schema is
correct — it states what is uncertain so the Phase 7 functions can be designed
against confirmed structures only.

Run after schema discovery::

    python -m app.discovery.schema_discovery
    python -m app.discovery.data_dictionary
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.core.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

REPORTS_DIR = PROJECT_ROOT / "data" / "schema_reports"

# Each FM concept maps to the vocabulary that suggests a table/column serves it.
# Concept -> (search terms, human explanation of why it matters).
CONCEPTS: dict[str, tuple[tuple[str, ...], str]] = {
    "Facilities": (("facility",), "Top-level scoping of building data per facility."),
    "Projects": (("project",), "Project scoping; often a sibling of facility."),
    "Buildings": (("building",), "Physical building entity."),
    "Floors / storeys / levels": (
        ("floor", "storey", "story", "level", "geschoss", "etage"),
        "Vertical structure used to scope counts and areas by floor.",
    ),
    "Rooms / spaces": (
        ("room", "space", "raum"),
        "Spatial subdivision within a floor.",
    ),
    "Asset components": (
        ("asset", "component", "element"),
        "The core inventory of building elements (windows, doors, …).",
    ),
    "Component types": (
        ("type", "category", "extobject", "ifc"),
        "The IFC/type classification used to filter components.",
    ),
    "IFC object identifiers": (
        ("ifc", "guid", "globalid", "global_id", "externalid", "external_id", "object"),
        "Stable identifiers for traceability of returned components.",
    ),
    "Component attributes / properties": (
        ("attribute", "property", "propertyset", "pset"),
        "Where geometric/descriptive values live (possibly key/value).",
    ),
    "Geometry — area": (("area", "flaeche"), "Stored area for area aggregation."),
    "Geometry — width": (("width",), "Width; fallback for area = width × height."),
    "Geometry — height": (("height",), "Height; fallback for area = width × height."),
    "Geometry — length": (("length",), "Length dimension."),
    "Geometry — volume": (("volume",), "Volume dimension."),
    "Equipment": (("equipment",), "Operable equipment (optional scope)."),
    "Maintenance records": (
        ("maintenance", "wartung", "inspection", "schedule", "date"),
        "Maintenance / inspection scheduling (optional scope).",
    ),
}


@dataclass
class Reports:
    """Loaded Phase 3 CSV reports."""

    tables: pd.DataFrame
    columns: pd.DataFrame
    primary_keys: pd.DataFrame
    foreign_keys: pd.DataFrame
    row_counts: pd.DataFrame
    sample_values: pd.DataFrame


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.warning("Missing report: %s (treating as empty)", path.name)
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_reports(reports_dir: Path = REPORTS_DIR) -> Reports:
    """Load the CSV reports produced by Phase 3."""
    return Reports(
        tables=_read_csv(reports_dir / "tables.csv"),
        columns=_read_csv(reports_dir / "columns.csv"),
        primary_keys=_read_csv(reports_dir / "primary_keys.csv"),
        foreign_keys=_read_csv(reports_dir / "foreign_keys.csv"),
        row_counts=_read_csv(reports_dir / "row_counts.csv"),
        sample_values=_read_csv(reports_dir / "sample_values.csv"),
    )


# ── Per-table fact lookups ───────────────────────────────────────────────────


def _matching_columns(columns: pd.DataFrame, terms: tuple[str, ...]) -> pd.DataFrame:
    """Columns whose table or column name contains any of ``terms``."""
    if columns.empty:
        return columns
    terms_lower = tuple(t.lower() for t in terms)
    mask = columns.apply(
        lambda r: any(t in f"{r['table_name']} {r['column_name']}".lower() for t in terms_lower),
        axis=1,
    )
    return columns[mask]


def _row_count(reports: Reports, schema: str, table: str) -> str:
    if reports.row_counts.empty:
        return "?"
    match = reports.row_counts[
        (reports.row_counts["table_schema"] == schema) & (reports.row_counts["table_name"] == table)
    ]
    return match.iloc[0]["row_count"] if not match.empty else "?"


def _pk_columns(reports: Reports, schema: str, table: str) -> list[str]:
    if reports.primary_keys.empty:
        return []
    match = reports.primary_keys[
        (reports.primary_keys["table_schema"] == schema)
        & (reports.primary_keys["table_name"] == table)
    ]
    return match["column_name"].tolist()


def _fk_links(reports: Reports, schema: str, table: str) -> list[str]:
    if reports.foreign_keys.empty:
        return []
    fk = reports.foreign_keys
    out: list[str] = []
    child = fk[(fk["table_schema"] == schema) & (fk["table_name"] == table)]
    for _, r in child.iterrows():
        out.append(
            f"{r['column_name']} → {r['foreign_table_schema']}."
            f"{r['foreign_table_name']}.{r['foreign_column_name']}"
        )
    parent = fk[(fk["foreign_table_schema"] == schema) & (fk["foreign_table_name"] == table)]
    for _, r in parent.iterrows():
        out.append(
            f"{r['table_schema']}.{r['table_name']}.{r['column_name']} → "
            f"{r['foreign_column_name']} (incoming)"
        )
    return out


def _sample_for_column(
    reports: Reports, schema: str, table: str, column: str, limit: int = 3
) -> list[str]:
    if reports.sample_values.empty:
        return []
    sv = reports.sample_values
    match = sv[
        (sv["table_schema"] == schema) & (sv["table_name"] == table) & (sv["column_name"] == column)
    ]
    values = [v for v in match["value"].tolist() if v not in ("", "None")]
    return values[:limit]


def _confidence(n_matched_cols: int, row_count: str) -> tuple[str, str]:
    """Heuristic confidence + manual-confirmation flag for a candidate table."""
    try:
        rows = int(row_count)
    except (ValueError, TypeError):
        rows = -1
    if n_matched_cols >= 2 and rows > 0:
        return "medium", "yes"
    if n_matched_cols >= 1 and rows > 0:
        return "low", "yes"
    return "low", "yes"


# ── Markdown assembly ────────────────────────────────────────────────────────


def build_dictionary_markdown(reports: Reports) -> str:
    lines: list[str] = []
    lines.append("# Data Dictionary (candidate mapping)\n")
    lines.append(
        "_Generated by `python -m app.discovery.data_dictionary` from the Phase 3 reports. "
        "This maps the **discovered** database onto the FM/BIM concepts the "
        "prototype needs. Confidence is heuristic; treat every 'manual "
        "confirmation needed: yes' as a task before writing Phase 7 functions._\n"
    )

    if reports.columns.empty:
        lines.append(
            "> **No column report found.** Run `python -m app.discovery.schema_discovery` first.\n"
        )
        return "\n".join(lines)

    for concept, (terms, why) in CONCEPTS.items():
        lines.append(f"## {concept}\n")
        lines.append(f"_Why relevant:_ {why}\n")

        matched = _matching_columns(reports.columns, terms)
        if matched.empty:
            lines.append(
                "**No candidate columns** matched by name. "
                "This concept may be absent, named differently, or encoded "
                "inside another field (manual confirmation needed).\n"
            )
            continue

        # Group candidate columns by their owning table.
        for (schema, table), grp in matched.groupby(["table_schema", "table_name"]):
            cols = grp["column_name"].tolist()
            rc = _row_count(reports, schema, table)
            pks = _pk_columns(reports, schema, table)
            fks = _fk_links(reports, schema, table)
            conf, manual = _confidence(len(cols), rc)

            lines.append(f"### `{schema}.{table}`  ·  rows: {rc}\n")
            lines.append(f"- **Why relevant:** matched columns for *{concept}*.")
            lines.append(f"- **Matched columns:** {', '.join(f'`{c}`' for c in cols)}")
            lines.append(
                f"- **Possible primary key:** {', '.join(pks) if pks else '— (none declared)'}"
            )
            if fks:
                lines.append("- **Possible joins / foreign keys:**")
                for link in fks:
                    lines.append(f"    - {link}")
            else:
                lines.append("- **Possible joins / foreign keys:** — (none declared)")

            # A couple of sample values for the first matched column.
            samples = _sample_for_column(reports, schema, table, cols[0])
            if samples:
                rendered = ", ".join(f"`{s}`" for s in samples)
                lines.append(f"- **Sample values (`{cols[0]}`):** {rendered}")

            lines.append(f"- **Confidence:** {conf}")
            lines.append(f"- **Manual confirmation needed:** {manual}\n")

    # Closing guidance.
    lines.append("## How to use this dictionary\n")
    lines.append(
        "1. Confirm the **asset-component table** and its **IFC type column** "
        "against `sample_values.csv`.\n"
        "2. Confirm how **floor** is represented (FK, label column, or encoded "
        "in a name).\n"
        "3. Confirm whether geometry (**area/width/height**) is stored as "
        "columns or in a key/value **attribute** table.\n"
        "4. Only then design the Phase 7 SQL-backed functions against the "
        "**confirmed** structures.\n"
    )
    return "\n".join(lines)


def generate(reports_dir: Path = REPORTS_DIR) -> Path:
    """Build the data dictionary and write it to disk; return the output path."""
    reports = load_reports(reports_dir)
    content = build_dictionary_markdown(reports)
    out_path = reports_dir / "data_dictionary.md"
    out_path.write_text(content, encoding="utf-8")
    logger.info("Wrote %s", out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a data dictionary from the Phase 3 schema reports."
    )
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    out_path = generate(args.reports)
    print(f"Data dictionary written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
