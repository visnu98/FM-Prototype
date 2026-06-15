"""Live-DB tests for the FM functions (facility 124851). Skip if no DB."""

from __future__ import annotations

import pytest

from app.tools.fm_functions import build_registry
from app.tools.models import ToolCall

pytestmark = pytest.mark.usefixtures("require_db")


@pytest.fixture(scope="module")
def registry():  # type: ignore[no-untyped-def]
    return build_registry()


def _ok(registry, name, **args):  # type: ignore[no-untyped-def]
    res = registry.execute(ToolCall(name=name, arguments=args))
    assert res.ok, res.error_message
    return res.data


def test_list_floors(registry) -> None:  # type: ignore[no-untyped-def]
    floors = _ok(registry, "list_queryable_floors")
    names = [f["name"] for f in floors]
    assert "Erdgeschoss" in names and "2. Obergeschoss" in names
    # ordered bottom-to-top
    assert names[0] == "Untergeschoss"


def test_count_components_all(registry) -> None:  # type: ignore[no-untyped-def]
    data = _ok(registry, "count_components", component_type="windows")
    assert data["component_type"] == "IfcWindow"
    assert data["count"] == 272


def test_count_components_on_floor(registry) -> None:  # type: ignore[no-untyped-def]
    data = _ok(registry, "count_components", component_type="windows", floor="second floor")
    assert data["floor"] == "2. Obergeschoss"
    assert data["count"] == 52


def test_total_area_uses_width_x_height(registry) -> None:  # type: ignore[no-untyped-def]
    data = _ok(registry, "calculate_total_component_area", component_type="windows")
    assert data["method"] == "width_x_height"
    assert data["total_area"] == pytest.approx(1787.95, abs=0.5)


def test_area_by_floor(registry) -> None:  # type: ignore[no-untyped-def]
    data = _ok(registry, "calculate_area_by_floor", component_type="windows")
    floors = {f["floor"]: f["total_area"] for f in data["by_floor"]}
    assert floors["1. Obergeschoss"] > floors["Erdgeschoss"]


def test_largest_area_floor(registry) -> None:  # type: ignore[no-untyped-def]
    data = _ok(registry, "find_floor_with_largest_component_area", component_type="windows")
    assert data["largest_floor"] == "1. Obergeschoss"


def test_compare_counts(registry) -> None:  # type: ignore[no-untyped-def]
    data = _ok(
        registry,
        "compare_component_count_between_floors",
        component_type="windows",
        floor_a="first floor",
        floor_b="second floor",
    )
    assert data["count_a"] == 79 and data["count_b"] == 52
    assert data["more_on"] == "1. Obergeschoss"


def test_invalid_component_type_is_enum_error(registry) -> None:  # type: ignore[no-untyped-def]
    from app.tools.models import ErrorCategory

    res = registry.execute(
        ToolCall(name="count_components", arguments={"component_type": "spaceships"})
    )
    assert not res.ok
    assert res.error_category is ErrorCategory.INVALID_ENUM_VALUE


def test_capabilities_lists_unsupported(registry) -> None:  # type: ignore[no-untyped-def]
    data = _ok(registry, "get_database_capabilities")
    assert "IfcWindow" in data["queryable_component_types"]
    assert any("maintenance" in s for s in data["not_supported"])


def test_floor_area_single(registry) -> None:  # type: ignore[no-untyped-def]
    data = _ok(registry, "calculate_floor_area", floor="ground floor")
    assert data["floor"] == "Erdgeschoss"
    assert data["gross_area"] == pytest.approx(556.74, abs=1.0)
    assert data["unit"] == "m²"


def test_floor_area_all(registry) -> None:  # type: ignore[no-untyped-def]
    data = _ok(registry, "calculate_floor_area")
    floors = {f["floor"]: f["gross_area"] for f in data["by_floor"]}
    assert "Untergeschoss" in floors and "Erdgeschoss" in floors
    # Basement is the largest floor by area in this facility.
    assert floors["Untergeschoss"] > floors["Erdgeschoss"]
