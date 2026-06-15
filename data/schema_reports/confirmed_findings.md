# Confirmed Schema Findings (manually verified)

_These facts were verified by direct queries against the live `libalv2` database
on 2026-06-13, on top of the auto-generated Phase 3/4 reports. They are the
**confirmed** structures that the Phase 7 SQL-backed functions should target._

## 1. The database is multi-facility — scoping is mandatory

`public.facility` holds 14+ buildings (e.g. `Albrechtstraße 20`, `Campus
Zollikofen`, `Barracks 101`, `Facility SBIS`). Floor labels are **not unique**
across facilities (many duplicate `Erdgeschoss` rows with different ids).

**Implication:** every FM function must filter by a facility id. The old
prototype used facility **124851** (`Albrechtstraße 20`), whose floors are
`Untergeschoss`, `Erdgeschoss`, `1. Obergeschoss`, `2. Obergeschoss`,
`Dachgeschoss`. Set `DEFAULT_FACILITY_ID` accordingly.

Component counts per facility (top): 84787500 → 6149, 124851 → 1857,
67020000 → 1853, 72242500 → 1171, 84787701 → 1036.

## 2. Asset components and their IFC type — CONFIRMED

- Table: `public.asset_component` (PK `id`, scope column `owning_facility_id`).
- **IFC type lives in `asset_component.ext_object`** — values include
  `IfcDoor` (1260), `IfcWindow` (788), `IfcWall` (603), `IfcWallStandardCase`
  (536), `IfcSlab` (503), `IfcSpaceHeater` (354), `IfcColumn` (209),
  `IfcCurtainWall` (972), `IfcFlowSegment` (1499)… Note: 3413 rows have
  `ext_object = NULL`.
- Also useful: `type_id` (→ `type.id`), `category`, `predefined_type`,
  `name`, `ifc_reference`, `model_id`.

➡ Enables `count_components(component_type)`, `list_queryable_component_types()`.

## 3. Floor of a component — via SPACE, not a direct column — CONFIRMED

`asset_component` has **no** `floor_id`. The floor is reached through the
space association:

```
asset_component.id
  → space_components.components_id / space_components.spaces_id
  → space.id  (space.floor_id, space.gross_area, space.net_area)
  → floor.id  (floor.name, floor.elevation, floor.height, floor.facility_id)
```

Verified for facility 124851 / `IfcWindow`: `1. Obergeschoss` = 158,
`2. Obergeschoss` = 104, `Erdgeschoss` = 32, `Untergeschoss` = 30 — matching
the old prototype's expected floor labels.

➡ Enables floor-scoped functions (`count_components(type, floor)`,
`compare_component_count_between_floors`, `calculate_area_by_floor`).

**Caveat:** components not associated with any space are invisible to the
floor join. The old prototype instead parsed the floor code out of the
component `name` (e.g. `…_OG1_…`). Both strategies should be evaluated; the
space join is the cleaner, schema-backed one.

## 4. Attributes / geometry — key/value model — CONFIRMED

- `public.attribute` (309,423 rows): `name`, `jhi_value_txt` (the value),
  `unit_id` (→ `unit`), `property_id` (→ `property`), `category`,
  `owning_facility_id`.
- Join table: `public.asset_component_attributes(attributes_id,
  asset_components_id)` links attributes to components.
- Attribute names are **German**: `Höhe` (height, 3792), `Breite` (width,
  3606), area appears as `Fläche` (1942), `Oberfläche`, `NetSideArea` (1313),
  `GrossSideArea` (1313). Values are stored as text in `jhi_value_txt` and must
  be cast/parsed; units come from `unit_id`.

➡ Enables `get_component_attributes(...)` and area aggregation, but requires a
**normalization layer** (Phase 6) mapping `area/height/width` → the German
attribute names, and careful numeric parsing of `jhi_value_txt`.

## 5. Room/space area — alternative area source

`public.space` carries `gross_area`, `net_area`, `usable_height`, `floor_id`.
This is room area, distinct from component (e.g. window) area. Keep the two
clearly separated in function docs.

## 6. Types catalog

`public.type` (PK `id`): `ext_object`, `category`, `nominal_length`,
`nominal_width`, `nominal_height`, `material`, plus warranty fields
(`warranty_duration_labor/parts`, `expected_life`, `replacement_cost`).

## 7. Equipment & maintenance — NOT directly supported

No dedicated `equipment` or `maintenance`/`wartung`/`inspection` table was
found by name. Closest signals: `type.warranty_*`/`expected_life`,
`technical_system`, `warranty_reminder`, `job`/`task` (work orders).

➡ **Do not fake** equipment/maintenance functions (Phase 7 items 11–14).
Document them as unsupported for this dataset, or scope them narrowly to
`warranty_reminder` / `job` only after manual confirmation.

## 8. No declared primary/foreign keys

The catalog reports 0 PKs and 0 FKs. Relationships above were inferred from
column naming and **verified by sample joins**. Treat join columns as the
contract; there is no DB-level referential integrity to rely on.

## Recommended next steps (Phase 5+)

1. Set `DEFAULT_FACILITY_ID` (start with 124851 to mirror the old prototype, or
   pick the richest facility 84787500 for more data).
2. Build normalization (Phase 6) from **live values**: floor labels from
   `floor.name` (per facility), component types from
   `asset_component.ext_object`, attribute names from `attribute.name`.
3. Implement Phase 7 functions against the joins confirmed here; parse
   `jhi_value_txt` numerically and surface `unit`.
4. Mark equipment/maintenance functions as unsupported unless confirmed.
