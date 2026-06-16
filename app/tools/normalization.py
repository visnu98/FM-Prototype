"""Normalization layer (Phase 6).

Users and LLMs phrase things differently from the database. This module maps
natural-language terms onto the *discovered* database values:

- floors / storeys  ->  the real `floor.name` for the facility
- component types    ->  the real `asset_component.ext_object` (IFC) value
- attribute names    ->  the real (German) `attribute.name`, or a computed marker

Design choices (documented for the thesis):
- Mappings are derived from **live DB values** wherever possible. Floor labels
  and component types are validated against what actually exists for the
  facility, so an unknown value yields a clean "invalid value" error.
- A small, clearly-located synonym table bridges English/German wording and
  positional phrases ("ground floor", "top floor", "second floor"). Floor
  ordering is derived from the German naming convention because `floor.elevation`
  is mostly NULL in this dataset.
- Every resolution returns BOTH the original and the normalized value (plus the
  method used) so the evaluation can log how normalization behaved.

The decimal handling note: numeric attribute values are stored as German text
(e.g. ``"0,995"``); parsing lives in :func:`parse_german_number`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from app.core.config import get_settings
from app.core.db import execute_read_query
from app.tools.registry import ValueNotAllowedError

# ── Result type ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Normalized:
    """The outcome of normalizing one term."""

    original: str
    normalized: str
    method: str  # "exact" | "synonym" | "positional" | "db_value"

    def as_dict(self) -> dict[str, str]:
        return {"original": self.original, "normalized": self.normalized, "method": self.method}


@dataclass(frozen=True)
class FloorInfo:
    """A floor available for a facility, with a derived ordering index."""

    floor_id: int
    name: str
    order_index: int
    elevation: float | None
    source_table: str = "floor"


# ── Facility default ─────────────────────────────────────────────────────────


def default_facility_id() -> int:
    """Return the configured DEFAULT_FACILITY_ID or raise a clear error."""
    fid = get_settings().default_facility_id
    if fid is None:
        raise ValueNotAllowedError(
            "No facility selected. Set DEFAULT_FACILITY_ID in .env.",
            parameter="facility_id",
        )
    return fid


# ── Floors ───────────────────────────────────────────────────────────────────

# Positional / synonym phrases -> a semantic key resolved against the facility's
# ordered floors. Kept here as the single source of hard-coded floor vocabulary.
_FLOOR_GROUND_WORDS = {"erdgeschoss", "eg", "ground floor", "ground", "groundfloor"}
_FLOOR_BASEMENT_WORDS = {"untergeschoss", "ug", "basement", "cellar", "souterrain"}
_FLOOR_TOP_WORDS = {"dachgeschoss", "dg", "attic", "top floor", "topfloor", "top", "roof"}

# "<n>th floor" / "level <n>" / "<n>. Obergeschoss" -> upper-floor number n.
_UPPER_ORDINAL_WORDS = {
    "first": 1,
    "1st": 1,
    "one": 1,
    "second": 2,
    "2nd": 2,
    "two": 2,
    "third": 3,
    "3rd": 3,
    "three": 3,
    "fourth": 4,
    "4th": 4,
    "four": 4,
}


def floor_order_index(name: str) -> int:
    """Derive a sortable order index from a German floor name.

    Basement < ground < 1.OG < 2.OG < ... < attic. Used because elevation is
    mostly NULL in this dataset.
    """
    n = name.strip().lower()
    if any(w in n for w in ("unter", "basement")) or n == "ug":
        return -1
    if any(w in n for w in ("erd", "ground")) or n == "eg":
        return 0
    if any(w in n for w in ("dach", "attic", "roof")) or n == "dg":
        return 1000
    m = re.search(r"(\d+)\s*\.?\s*(?:og|obergeschoss)", n)
    if not m:
        m = re.search(r"(\d+)", n)
    return int(m.group(1)) if m else 999


@lru_cache(maxsize=8)
def list_floors(facility_id: int) -> tuple[FloorInfo, ...]:
    """Return the facility's floors ordered bottom-to-top (cached)."""
    rows = execute_read_query(
        """
        SELECT id, name, elevation
        FROM floor
        WHERE facility_id = :fid AND name IS NOT NULL
        """,
        {"fid": facility_id},
    )
    floors = [
        FloorInfo(
            floor_id=int(r["id"]),
            name=str(r["name"]),
            order_index=floor_order_index(str(r["name"])),
            elevation=float(r["elevation"]) if r["elevation"] is not None else None,
        )
        for r in rows
    ]
    # De-duplicate by name (some facilities have repeated labels); keep first.
    seen: set[str] = set()
    unique: list[FloorInfo] = []
    for f in sorted(floors, key=lambda x: x.order_index):
        if f.name not in seen:
            seen.add(f.name)
            unique.append(f)
    return tuple(unique)


def resolve_floor(label: str, facility_id: int | None = None) -> Normalized:
    """Map any accepted floor label to the facility's canonical `floor.name`."""
    fid = facility_id if facility_id is not None else default_facility_id()
    floors = list_floors(fid)
    if not floors:
        raise ValueNotAllowedError(f"No floors found for facility {fid}.", parameter="floor")

    raw = label.strip()
    key = raw.lower()
    names_by_order = {f.order_index: f for f in floors}

    # 1) Exact match against a real floor name.
    for f in floors:
        if f.name.lower() == key:
            return Normalized(raw, f.name, "exact")

    # 2) Positional words: ground / basement / top.
    if key in _FLOOR_GROUND_WORDS and 0 in names_by_order:
        return Normalized(raw, names_by_order[0].name, "positional")
    if key in _FLOOR_BASEMENT_WORDS:
        bottom = min(floors, key=lambda x: x.order_index)
        return Normalized(raw, bottom.name, "positional")
    if key in _FLOOR_TOP_WORDS:
        top = max(floors, key=lambda x: x.order_index)
        return Normalized(raw, top.name, "positional")

    # 3) Upper-floor number: "second floor", "level 2", "2. Obergeschoss", "2og".
    n = _extract_upper_floor_number(key)
    if n is not None and n in names_by_order:
        return Normalized(raw, names_by_order[n].name, "positional")

    valid = ", ".join(f.name for f in floors)
    raise ValueNotAllowedError(
        f"Floor '{label}' not recognised for facility {fid}. Valid floors: {valid}. "
        "Call list_queryable_floors() to see options.",
        parameter="floor",
    )


def _extract_upper_floor_number(key: str) -> int | None:
    """Extract an upper-floor number from phrases like 'second floor', '2og'."""
    for word, num in _UPPER_ORDINAL_WORDS.items():
        if re.search(rf"\b{word}\b", key):
            return num
    m = re.search(r"(\d+)\s*\.?\s*(?:og|obergeschoss)", key)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:level|floor|stock|etage)\s*(\d+)", key)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*(?:st|nd|rd|th)\s*floor", key)
    if m:
        return int(m.group(1))
    return None


# ── Component types ──────────────────────────────────────────────────────────

# Synonyms -> canonical IFC ext_object value. The single hard-coded location for
# component-type vocabulary; validated against live DB values before use.
_TYPE_SYNONYMS: dict[str, str] = {
    "window": "IfcWindow",
    "windows": "IfcWindow",
    "fenster": "IfcWindow",
    "door": "IfcDoor",
    "doors": "IfcDoor",
    "tür": "IfcDoor",
    "tueren": "IfcDoor",
    "türen": "IfcDoor",
    "tur": "IfcDoor",
    "wall": "IfcWallStandardCase",
    "walls": "IfcWallStandardCase",
    "wand": "IfcWallStandardCase",
    "wände": "IfcWallStandardCase",
    "slab": "IfcSlab",
    "slabs": "IfcSlab",
    "decke": "IfcSlab",
    "column": "IfcColumn",
    "columns": "IfcColumn",
    "stütze": "IfcColumn",
    "stützen": "IfcColumn",
    "pillar": "IfcColumn",
    "stair": "IfcStair",
    "stairs": "IfcStair",
    "treppe": "IfcStair",
    "railing": "IfcRailing",
    "railings": "IfcRailing",
    "geländer": "IfcRailing",
    "beam": "IfcBeam",
    "beams": "IfcBeam",
    "balken": "IfcBeam",
    "space heater": "IfcSpaceHeater",
    "heater": "IfcSpaceHeater",
    "heaters": "IfcSpaceHeater",
    "radiator": "IfcSpaceHeater",
    "heizung": "IfcSpaceHeater",
    "heizkörper": "IfcSpaceHeater",
    "sensor": "IfcSensor",
    "sensors": "IfcSensor",
    "fan": "IfcFan",
    "fans": "IfcFan",
    "lüftung": "IfcFan",
    "ventilator": "IfcFan",
    "air terminal": "IfcAirTerminal",
    "air outlet": "IfcAirTerminal",
    "sanitary": "IfcSanitaryTerminal",
    "sanitary terminal": "IfcSanitaryTerminal",
    "curtain wall": "IfcCurtainWall",
    "curtainwall": "IfcCurtainWall",
    "tank": "IfcTank",
    "tanks": "IfcTank",
    "boiler": "IfcBoiler",
    "boilers": "IfcBoiler",
    "damper": "IfcDamper",
    "actuator": "IfcActuator",
    "controller": "IfcController",
    "covering": "IfcCovering",
    "pipe": "IfcPipeSegment",
}


@lru_cache(maxsize=8)
def list_component_types(facility_id: int) -> tuple[tuple[str, int], ...]:
    """Return (ext_object, count) available for the facility (cached)."""
    rows = execute_read_query(
        """
        SELECT ext_object, COUNT(*) AS n
        FROM asset_component
        WHERE owning_facility_id = :fid AND ext_object IS NOT NULL
        GROUP BY ext_object
        ORDER BY n DESC
        """,
        {"fid": facility_id},
    )
    return tuple((str(r["ext_object"]), int(r["n"])) for r in rows)


def resolve_component_type(label: str, facility_id: int | None = None) -> Normalized:
    """Map a component label to a real IFC ``ext_object`` value for the facility."""
    fid = facility_id if facility_id is not None else default_facility_id()
    available = {t for t, _ in list_component_types(fid)}
    available_lower = {t.lower(): t for t in available}

    raw = label.strip()
    key = raw.lower()

    # 1) Exact IFC value (case-insensitive).
    if key in available_lower:
        return Normalized(raw, available_lower[key], "exact")

    # 2) Synonym -> IFC value, then confirm it exists for this facility.
    mapped = _TYPE_SYNONYMS.get(key)
    if mapped and mapped in available:
        return Normalized(raw, mapped, "synonym")
    if mapped and mapped.lower() in available_lower:
        return Normalized(raw, available_lower[mapped.lower()], "synonym")

    valid = ", ".join(sorted(available))
    raise ValueNotAllowedError(
        f"Component type '{label}' not available for facility {fid}. "
        f"Available types: {valid}. Call list_queryable_component_types().",
        parameter="component_type",
    )


# ── Attributes ───────────────────────────────────────────────────────────────

# Maps an attribute request to a real (German) attribute.name. "area" is special:
# it is computed from a stored area attribute or width*height (see fm_functions).
AREA_MARKER = "__AREA__"

_ATTRIBUTE_SYNONYMS: dict[str, str] = {
    "height": "Höhe",
    "höhe": "Höhe",
    "hoehe": "Höhe",
    "width": "Breite",
    "breite": "Breite",
    "length": "Länge",
    "länge": "Länge",
    "laenge": "Länge",
    "area": AREA_MARKER,
    "fläche": AREA_MARKER,
    "flaeche": AREA_MARKER,
    "surface": AREA_MARKER,
    "oberfläche": AREA_MARKER,
    "material": "Material",
    "name": "Name",
}

# Candidate stored area attribute names, in preference order.
STORED_AREA_NAMES: tuple[str, ...] = ("Fläche", "Oberfläche", "NetSideArea", "GrossSideArea")


def resolve_attribute(label: str) -> Normalized:
    """Map an attribute request to a DB attribute name or the AREA marker."""
    raw = label.strip()
    key = raw.lower()
    mapped = _ATTRIBUTE_SYNONYMS.get(key)
    if mapped is not None:
        method = "synonym"
        return Normalized(raw, mapped, method)
    # Unknown attribute: pass through verbatim (the query simply may match none).
    return Normalized(raw, raw, "db_value")


# ── German numeric parsing ───────────────────────────────────────────────────


def parse_german_number(value: str | None) -> float | None:
    """Parse a German-formatted number string (``"1.234,56"`` -> 1234.56).

    Returns ``None`` if the value is missing or not numeric.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    # Remove thousands separators, convert decimal comma to dot.
    s = s.replace(".", "") if ("," in s and "." in s) else s
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        # Keep only the leading numeric portion if a unit got appended.
        m = re.match(r"-?\d+(?:\.\d+)?", s)
        return float(m.group(0)) if m else None


def clear_caches() -> None:
    """Clear DB-derived caches (use in tests or after data changes)."""
    list_floors.cache_clear()
    list_component_types.cache_clear()
