"""View and cube rows embed their description; model rows list every column."""

from __future__ import annotations

import pytest

from wren.memory.schema_indexer import extract_schema_items

_MANIFEST = {
    "catalog": "c",
    "schema": "s",
    "models": [
        {
            "name": "wide",
            "columns": [{"name": f"c{i}", "type": "int"} for i in range(30)],
        }
    ],
    "relationships": [],
    "views": [
        {
            "name": "v_orders",
            "statement": "SELECT " + ", ".join(f"col{i}" for i in range(80)),
            "properties": {"description": "Orders enriched with customer region"},
        }
    ],
    "cubes": [
        {
            "name": "sales",
            "baseObject": "orders",
            "measures": [{"name": "revenue", "type": "DOUBLE", "expression": "SUM(x)"}],
            "dimensions": [],
            "timeDimensions": [],
            "properties": {"description": "Revenue by region and period"},
        }
    ],
}


def _by_type(items, t):
    return [i for i in items if i["item_type"] == t]


@pytest.mark.unit
def test_view_record_includes_description_before_sql():
    (view,) = _by_type(extract_schema_items(_MANIFEST), "view")
    assert view["text"].startswith(
        "View 'v_orders': Orders enriched with customer region. SQL: SELECT"
    )
    assert view["text"].endswith("…")
    assert view["expression"] == _MANIFEST["views"][0]["statement"]


@pytest.mark.unit
def test_cube_record_includes_description():
    (cube,) = _by_type(extract_schema_items(_MANIFEST), "cube")
    assert cube["text"].startswith(
        "Cube 'sales' over 'orders': Revenue by region and period. Measures: revenue"
    )


@pytest.mark.unit
def test_model_record_lists_all_columns():
    (model,) = _by_type(extract_schema_items(_MANIFEST), "model")
    for i in range(30):
        assert f"c{i} (int)" in model["text"]
