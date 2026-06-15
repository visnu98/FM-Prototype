"""Tests for the normalization layer.

Pure-logic tests run without a database; resolution tests use the live DB and
skip if it is unreachable.
"""

from __future__ import annotations

import pytest

from app.tools import normalization as norm

# ── Pure logic (no DB) ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0,995", 0.995),
        ("0,59", 0.59),
        ("1.234,56", 1234.56),
        ("42", 42.0),
        ("", None),
        (None, None),
        ("abc", None),
    ],
)
def test_parse_german_number(raw, expected) -> None:  # type: ignore[no-untyped-def]
    assert norm.parse_german_number(raw) == expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Untergeschoss", -1),
        ("Erdgeschoss", 0),
        ("1. Obergeschoss", 1),
        ("2. Obergeschoss", 2),
        ("Dachgeschoss", 1000),
    ],
)
def test_floor_order_index(name, expected) -> None:  # type: ignore[no-untyped-def]
    assert norm.floor_order_index(name) == expected


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("second floor", 2),
        ("2. obergeschoss", 2),
        ("2og", 2),
        ("level 1", 1),
        ("first floor", 1),
        ("3rd floor", 3),
    ],
)
def test_extract_upper_floor_number(phrase, expected) -> None:  # type: ignore[no-untyped-def]
    assert norm._extract_upper_floor_number(phrase) == expected


def test_resolve_attribute_synonyms() -> None:
    assert norm.resolve_attribute("height").normalized == "Höhe"
    assert norm.resolve_attribute("width").normalized == "Breite"
    assert norm.resolve_attribute("area").normalized == norm.AREA_MARKER
    # Unknown attribute passes through verbatim.
    assert norm.resolve_attribute("Foo").normalized == "Foo"


# ── DB-backed resolution (facility 124851) ───────────────────────────────────

FID = 124851


def test_resolve_floor_positional(require_db) -> None:  # type: ignore[no-untyped-def]
    norm.clear_caches()
    assert norm.resolve_floor("ground floor", FID).normalized == "Erdgeschoss"
    assert norm.resolve_floor("second floor", FID).normalized == "2. Obergeschoss"
    assert norm.resolve_floor("top floor", FID).normalized == "Dachgeschoss"
    assert norm.resolve_floor("basement", FID).normalized == "Untergeschoss"


def test_resolve_floor_exact_and_german(require_db) -> None:  # type: ignore[no-untyped-def]
    assert norm.resolve_floor("1. Obergeschoss", FID).normalized == "1. Obergeschoss"
    assert norm.resolve_floor("2og", FID).normalized == "2. Obergeschoss"


def test_resolve_floor_invalid_raises(require_db) -> None:  # type: ignore[no-untyped-def]
    from app.tools.registry import ValueNotAllowedError

    with pytest.raises(ValueNotAllowedError):
        norm.resolve_floor("planet mars", FID)


def test_resolve_component_type(require_db) -> None:  # type: ignore[no-untyped-def]
    assert norm.resolve_component_type("windows", FID).normalized == "IfcWindow"
    assert norm.resolve_component_type("doors", FID).normalized == "IfcDoor"
    assert norm.resolve_component_type("IfcWindow", FID).normalized == "IfcWindow"


def test_resolve_component_type_invalid_raises(require_db) -> None:  # type: ignore[no-untyped-def]
    from app.tools.registry import ValueNotAllowedError

    with pytest.raises(ValueNotAllowedError):
        norm.resolve_component_type("spaceship", FID)
