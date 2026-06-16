"""SQL-backed Facility Management functions (Phase 7).

These are the ONLY functions the LLM may call. Each one:
- has a clear docstring and typed (Pydantic-validated) parameters,
- normalizes free-text arguments via the Phase 6 layer,
- runs parameterised, read-only SQL (no string concatenation of user input),
- returns JSON-serialisable data with source IDs and traceability metadata.

Domain facts (verified against the live `libalv2` DB, facility 124851):
- Component IFC type lives in ``asset_component.ext_object``.
- A component's floor is resolved through the space placement:
  ``asset_component -> space_components -> space.floor_id -> floor``.
  Components not placed in a space are therefore invisible to floor-scoped
  queries; this is a documented limitation of the dataset.
- Geometry is stored as German key/value attributes (``Breite``/``Höhe``);
  area is computed as width*height when no stored area attribute exists.

Use :func:`build_registry` to obtain a populated :class:`ToolRegistry`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.db import execute_read_query
from app.tools import normalization as norm
from app.tools.models import ToolParameter
from app.tools.registry import ToolRegistry

# Safety cap on rows returned by listing functions.
RESULT_LIMIT = 500

# Reusable SQL fragment: component ids placed on a given floor (by name).
_FLOOR_SCOPE_SUBQUERY = """
    ac.id IN (
        SELECT sc.components_id
        FROM space_components sc
        JOIN space s ON s.id = sc.spaces_id
        JOIN floor f ON f.id = s.floor_id
        WHERE f.name = :floor
    )
"""


# ── Argument models (extra fields forbidden -> clean "unsupported param") ─────


class _StrictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmptyArgs(_StrictArgs):
    pass


class ComponentTypeArgs(_StrictArgs):
    component_type: str = Field(description="Component type, e.g. 'windows' or 'IfcWindow'.")


class ComponentTypeFloorArgs(_StrictArgs):
    component_type: str = Field(description="Component type, e.g. 'windows'.")
    floor: str | None = Field(default=None, description="Floor label, e.g. 'second floor'.")


class ComponentAttributesArgs(_StrictArgs):
    component_type: str
    floor: str | None = None
    attributes: list[str] | None = Field(
        default=None, description="Attribute names, e.g. ['height', 'width', 'area']."
    )


class FloorArgs(_StrictArgs):
    floor: str | None = Field(default=None, description="Floor label; omit for all floors.")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _facility() -> int:
    return norm.default_facility_id()


def _count_components(fid: int, ext_object: str, floor_name: str | None) -> int:
    sql = (
        "SELECT COUNT(DISTINCT ac.id) AS n FROM asset_component ac "
        "WHERE ac.owning_facility_id = :fid AND ac.ext_object = :t"
    )
    params: dict[str, Any] = {"fid": fid, "t": ext_object}
    if floor_name is not None:
        sql += " AND " + _FLOOR_SCOPE_SUBQUERY
        params["floor"] = floor_name
    rows = execute_read_query(sql, params)
    return int(rows[0]["n"]) if rows else 0


def _area_rows(fid: int, ext_object: str, floor_name: str | None) -> list[dict[str, Any]]:
    """Fetch per-component area inputs (stored area, width, height)."""
    area_names = ", ".join(f"'{n}'" for n in norm.STORED_AREA_NAMES)
    sql = f"""
        SELECT ac.id AS id,
               MAX(CASE WHEN a.name IN ({area_names}) THEN a.jhi_value_txt END) AS area_txt,
               MAX(CASE WHEN a.name = 'Breite' THEN a.jhi_value_txt END) AS width_txt,
               MAX(CASE WHEN a.name = 'Höhe'  THEN a.jhi_value_txt END) AS height_txt
        FROM asset_component ac
        JOIN asset_component_attributes aca ON aca.asset_components_id = ac.id
        JOIN attribute a ON a.id = aca.attributes_id
        WHERE ac.owning_facility_id = :fid AND ac.ext_object = :t
    """
    params: dict[str, Any] = {"fid": fid, "t": ext_object}
    if floor_name is not None:
        sql += " AND " + _FLOOR_SCOPE_SUBQUERY
        params["floor"] = floor_name
    sql += " GROUP BY ac.id"
    return execute_read_query(sql, params)


def _compute_area(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum component areas, preferring stored area then width*height."""
    total = 0.0
    by_method = {"stored_area": 0, "width_x_height": 0, "no_area_data": 0}
    source_ids: list[int] = []
    for r in rows:
        stored = norm.parse_german_number(r.get("area_txt"))
        if stored is not None:
            total += stored
            by_method["stored_area"] += 1
            source_ids.append(int(r["id"]))
            continue
        w = norm.parse_german_number(r.get("width_txt"))
        h = norm.parse_german_number(r.get("height_txt"))
        if w is not None and h is not None:
            total += w * h
            by_method["width_x_height"] += 1
            source_ids.append(int(r["id"]))
        else:
            by_method["no_area_data"] += 1
    method = "stored_area" if by_method["stored_area"] else "width_x_height"
    return {
        "total_area": round(total, 3),
        "unit": "m²",  # length attributes are in metres; area is m² (unit not stored)
        "method": method,
        "method_breakdown": by_method,
        "components_with_area": len(source_ids),
        "source_component_ids": source_ids[:RESULT_LIMIT],
    }


# ── The FM functions ─────────────────────────────────────────────────────────


def get_database_capabilities() -> dict[str, Any]:
    """Describe what the prototype can answer for the selected facility.

    Returns the facility, the queryable floors and component types, the
    supported question kinds, and what is explicitly NOT supported.
    """
    fid = _facility()
    floors = [f.name for f in norm.list_floors(fid)]
    types = [t for t, _ in norm.list_component_types(fid)]
    return {
        "facility_id": fid,
        "queryable_floors": floors,
        "queryable_component_types": types,
        "supported_questions": [
            "list floors / component types",
            "count components (optionally per floor)",
            "component attributes (height, width, area)",
            "total component area (stored area or width×height)",
            "area per floor / floor with largest area",
            "floor area (sum of room/space areas per floor)",
            "compare counts or areas between two floors",
        ],
        "not_supported": [
            "equipment inventories (no dedicated equipment table)",
            "maintenance / inspection schedules (no maintenance table)",
        ],
        "floor_resolution": "via space placement (asset_component→space→floor)",
    }


def list_queryable_floors() -> list[dict[str, Any]]:
    """List the floors/storeys available for the selected facility."""
    fid = _facility()
    return [
        {
            "floor_id": f.floor_id,
            "name": f.name,
            "order_index": f.order_index,
            "elevation": f.elevation,
            "source_table": f.source_table,
        }
        for f in norm.list_floors(fid)
    ]


def list_queryable_component_types() -> list[dict[str, Any]]:
    """List the component/IFC types available for the selected facility."""
    fid = _facility()
    return [
        {
            "component_type": t,
            "count": n,
            "source_table": "asset_component",
        }
        for t, n in norm.list_component_types(fid)
    ]


def count_components(component_type: str, floor: str | None = None) -> dict[str, Any]:
    """Count asset components of a type, optionally scoped to one floor."""
    fid = _facility()
    t = norm.resolve_component_type(component_type, fid)
    floor_norm = norm.resolve_floor(floor, fid) if floor else None
    floor_name = floor_norm.normalized if floor_norm else None
    n = _count_components(fid, t.normalized, floor_name)
    return {
        "count": n,
        "component_type": t.normalized,
        "floor": floor_name,
        "facility_id": fid,
        "_normalized_arguments": {
            "component_type": t.as_dict(),
            **({"floor": floor_norm.as_dict()} if floor_norm else {}),
        },
    }


def get_component_attributes(
    component_type: str,
    floor: str | None = None,
    attributes: list[str] | None = None,
) -> dict[str, Any]:
    """Retrieve descriptive/geometric attributes for components of a type.

    ``attributes`` defaults to ['height', 'width']. 'area' is reported as a
    computed value (stored area or width×height). Values are parsed from the
    German-formatted attribute text.
    """
    fid = _facility()
    t = norm.resolve_component_type(component_type, fid)
    floor_norm = norm.resolve_floor(floor, fid) if floor else None
    floor_name = floor_norm.normalized if floor_norm else None

    requested = attributes or ["height", "width"]
    resolved = [norm.resolve_attribute(a) for a in requested]
    wants_area = any(r.normalized == norm.AREA_MARKER for r in resolved)
    db_attr_names = [r.normalized for r in resolved if r.normalized != norm.AREA_MARKER]

    # Pull the requested key/value attributes for the matching components.
    area_rows = _area_rows(fid, t.normalized, floor_name) if wants_area else []
    area_by_id = {int(r["id"]): r for r in area_rows}

    components: list[dict[str, Any]] = []
    if db_attr_names or wants_area:
        placeholders = ", ".join(f":a{i}" for i in range(len(db_attr_names)))
        sql = """
            SELECT ac.id AS id, ac.name AS name, a.name AS attr, a.jhi_value_txt AS val
            FROM asset_component ac
            JOIN asset_component_attributes aca ON aca.asset_components_id = ac.id
            JOIN attribute a ON a.id = aca.attributes_id
            WHERE ac.owning_facility_id = :fid AND ac.ext_object = :t
        """
        params: dict[str, Any] = {"fid": fid, "t": t.normalized}
        if db_attr_names:
            sql += f" AND a.name IN ({placeholders})"
            for i, name in enumerate(db_attr_names):
                params[f"a{i}"] = name
        else:
            sql += " AND FALSE"  # no key/value attrs requested; area handled below
        if floor_name is not None:
            sql += " AND " + _FLOOR_SCOPE_SUBQUERY
            params["floor"] = floor_name
        rows = execute_read_query(sql, params) if db_attr_names else []

        grouped: dict[int, dict[str, Any]] = {}
        for r in rows:
            cid = int(r["id"])
            entry = grouped.setdefault(
                cid, {"component_id": cid, "name": r["name"], "attributes": {}}
            )
            entry["attributes"][r["attr"]] = norm.parse_german_number(r["val"]) or r["val"]

        # Add computed area per component when requested.
        if wants_area:
            for cid, ar in area_by_id.items():
                entry = grouped.setdefault(
                    cid, {"component_id": cid, "name": None, "attributes": {}}
                )
                stored = norm.parse_german_number(ar.get("area_txt"))
                if stored is None:
                    w = norm.parse_german_number(ar.get("width_txt"))
                    h = norm.parse_german_number(ar.get("height_txt"))
                    stored = round(w * h, 4) if (w is not None and h is not None) else None
                entry["attributes"]["area"] = stored
        components = list(grouped.values())[:RESULT_LIMIT]

    return {
        "component_type": t.normalized,
        "floor": floor_name,
        "requested_attributes": requested,
        "resolved_attributes": [r.as_dict() for r in resolved],
        "result_count": len(components),
        "components": components,
        "facility_id": fid,
        "_normalized_arguments": {
            "component_type": t.as_dict(),
            **({"floor": floor_norm.as_dict()} if floor_norm else {}),
        },
    }


def calculate_total_component_area(component_type: str, floor: str | None = None) -> dict[str, Any]:
    """Total area for a component type, optionally on one floor.

    Prefers a stored area attribute; otherwise computes width×height. The method
    used and a per-method breakdown are returned for transparency.
    """
    fid = _facility()
    t = norm.resolve_component_type(component_type, fid)
    floor_norm = norm.resolve_floor(floor, fid) if floor else None
    floor_name = floor_norm.normalized if floor_norm else None

    rows = _area_rows(fid, t.normalized, floor_name)
    area = _compute_area(rows)
    area.update(
        {
            "component_type": t.normalized,
            "floor": floor_name,
            "facility_id": fid,
            "_normalized_arguments": {
                "component_type": t.as_dict(),
                **({"floor": floor_norm.as_dict()} if floor_norm else {}),
            },
        }
    )
    return area


def calculate_area_by_floor(component_type: str) -> dict[str, Any]:
    """Total component area grouped by floor (for the selected facility)."""
    fid = _facility()
    t = norm.resolve_component_type(component_type, fid)
    per_floor: list[dict[str, Any]] = []
    for f in norm.list_floors(fid):
        rows = _area_rows(fid, t.normalized, f.name)
        area = _compute_area(rows)
        per_floor.append(
            {
                "floor": f.name,
                "order_index": f.order_index,
                "total_area": area["total_area"],
                "unit": area["unit"],
                "components_with_area": area["components_with_area"],
                "method_breakdown": area["method_breakdown"],
            }
        )
    return {
        "component_type": t.normalized,
        "facility_id": fid,
        "by_floor": per_floor,
        "_normalized_arguments": {"component_type": t.as_dict()},
    }


def calculate_floor_area(floor: str | None = None) -> dict[str, Any]:
    """Total floor area for the facility, from the spaces placed on each floor.

    'Floor area' is the sum of the gross (and net) areas of the spaces/rooms on
    a floor (``space.gross_area`` / ``space.net_area`` grouped by ``floor``).
    With ``floor`` given, returns that floor only; otherwise a per-floor
    breakdown. Spaces without a recorded area are excluded from the totals but
    still counted in ``spaces_total`` (so partial coverage is visible).
    """
    fid = _facility()
    floor_norm = norm.resolve_floor(floor, fid) if floor else None
    floor_name = floor_norm.normalized if floor_norm else None

    sql = """
        SELECT f.name AS floor, f.id AS floor_id,
               COUNT(s.id) AS spaces_total,
               COUNT(s.gross_area) AS spaces_with_area,
               SUM(s.gross_area) AS gross_area,
               SUM(s.net_area) AS net_area
        FROM floor f
        LEFT JOIN space s ON s.floor_id = f.id
        WHERE f.facility_id = :fid
    """
    params: dict[str, Any] = {"fid": fid}
    if floor_name is not None:
        sql += " AND f.name = :floor"
        params["floor"] = floor_name
    sql += " GROUP BY f.id, f.name"
    rows = execute_read_query(sql, params)

    def _round(value: Any) -> float | None:
        return round(float(value), 3) if value is not None else None

    by_floor = [
        {
            "floor": r["floor"],
            "floor_id": int(r["floor_id"]),
            "spaces_total": int(r["spaces_total"]),
            "spaces_with_area": int(r["spaces_with_area"]),
            "gross_area": _round(r["gross_area"]),
            "net_area": _round(r["net_area"]),
            "unit": "m²",
        }
        for r in rows
    ]
    order = {fi.name: fi.order_index for fi in norm.list_floors(fid)}
    by_floor.sort(key=lambda x: order.get(str(x["floor"]), 999))

    result: dict[str, Any] = {
        "facility_id": fid,
        "source": "sum of space.gross_area / net_area grouped by floor",
    }
    if floor_name is not None:
        single = (
            by_floor[0]
            if by_floor
            else {
                "floor": floor_name,
                "gross_area": None,
                "net_area": None,
                "spaces_total": 0,
                "spaces_with_area": 0,
                "unit": "m²",
            }
        )
        result.update(single)
        result["_normalized_arguments"] = {"floor": floor_norm.as_dict()}  # type: ignore[union-attr]
    else:
        result["by_floor"] = by_floor
    return result


# ── Registry assembly ────────────────────────────────────────────────────────


def build_registry(registry: ToolRegistry | None = None) -> ToolRegistry:
    """Create (or populate) a registry with all FM tools and their schemas."""
    reg = registry if registry is not None else ToolRegistry()

    p_type = ToolParameter(
        name="component_type",
        type="string",
        description="Component type, e.g. 'windows', 'doors', or an IFC value like 'IfcWindow'.",
        required=True,
    )
    p_floor = ToolParameter(
        name="floor",
        type="string",
        description="Floor label, e.g. 'ground floor', 'second floor', '2. Obergeschoss'.",
        required=False,
    )

    reg.register(
        name="get_database_capabilities",
        description="Describe what the prototype can answer for the selected facility.",
        func=get_database_capabilities,
        args_model=EmptyArgs,
    )
    reg.register(
        name="list_queryable_floors",
        description="List the floors/storeys available for the selected facility.",
        func=list_queryable_floors,
        args_model=EmptyArgs,
    )
    reg.register(
        name="list_queryable_component_types",
        description="List the component/IFC types available for the selected facility.",
        func=list_queryable_component_types,
        args_model=EmptyArgs,
    )
    reg.register(
        name="count_components",
        description="Count asset components of a type, optionally on a specific floor.",
        func=count_components,
        args_model=ComponentTypeFloorArgs,
        parameters=[p_type, p_floor],
    )
    reg.register(
        name="get_component_attributes",
        description="Get attributes (height, width, area, …) for components of a type.",
        func=get_component_attributes,
        args_model=ComponentAttributesArgs,
        parameters=[
            p_type,
            p_floor,
            ToolParameter(
                name="attributes",
                type="array",
                description="Attribute names, e.g. ['height','width','area'].",
                required=False,
            ),
        ],
    )
    reg.register(
        name="calculate_total_component_area",
        description="Total area for a component type, optionally on one floor "
        "(stored area or width×height).",
        func=calculate_total_component_area,
        args_model=ComponentTypeFloorArgs,
        parameters=[p_type, p_floor],
    )
    reg.register(
        name="calculate_area_by_floor",
        description="Total component area grouped by floor.",
        func=calculate_area_by_floor,
        args_model=ComponentTypeArgs,
        parameters=[p_type],
    )
    reg.register(
        name="calculate_floor_area",
        description="Total floor area (sum of room/space areas) for the facility, "
        "optionally for one floor. This is the area of the floor itself, not of "
        "components.",
        func=calculate_floor_area,
        args_model=FloorArgs,
        parameters=[p_floor],
    )
    return reg
