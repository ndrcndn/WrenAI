"""split_row_text: over-budget MDL rows split into parts that keep their identity."""

from __future__ import annotations

import pytest

from wren.memory.schema_indexer import extract_schema_items
from wren.memory.store import split_row_text


def _words(text: str) -> int:
    return len(text.split())


@pytest.mark.unit
def test_model_row_splits_on_column_boundaries_with_prefix():
    manifest = {
        "models": [
            {
                "name": "wide",
                "primaryKey": "c0",
                "properties": {"description": "Very wide fact table"},
                "columns": [
                    {"name": f"c{i}", "type": "decimal(18, 4)"} for i in range(60)
                ],
            }
        ]
    }
    (row,) = [i for i in extract_schema_items(manifest) if i["item_type"] == "model"]
    parts = split_row_text(row["text"], "model", _words, 40)
    assert parts[0] == "Model 'wide': Very wide fact table. Primary key: c0."
    col_parts = parts[1:]
    # Budget 40 words, prefix reserves 7, entries are 3 words → 11 per part.
    assert len(col_parts) == 6
    assert all(p.startswith("Model 'wide' (columns part ") for p in col_parts)
    assert col_parts[0].startswith(
        "Model 'wide' (columns part 1 of 6): c0 (decimal(18, 4)), c1"
    )
    assert col_parts[-1].startswith("Model 'wide' (columns part 6 of 6): ")
    assert all(p.endswith(").") for p in col_parts)
    # Type-internal commas are not split points.
    assert all(p.count("(decimal(18, 4))") <= 25 for p in col_parts)
    joined = " ".join(parts)
    assert all(f"c{i} (decimal(18, 4))" in joined for i in range(60))
    assert all(_words(p) <= 40 for p in parts)


@pytest.mark.unit
def test_model_row_caps_columns_per_part_when_budget_is_large():
    manifest = {
        "models": [
            {
                "name": "wide",
                "columns": [{"name": f"c{i}", "type": "int"} for i in range(60)],
            }
        ]
    }
    (row,) = [i for i in extract_schema_items(manifest) if i["item_type"] == "model"]
    parts = split_row_text(row["text"], "model", _words, 10_000)
    assert parts[0] == "Model 'wide'."
    assert [p.count("(int)") for p in parts[1:]] == [25, 25, 10]
    assert parts[1].startswith("Model 'wide' (columns part 1 of 3): c0 (int), c1 (int)")


@pytest.mark.unit
def test_cube_row_parts_carry_lead_clause():
    long_desc = " ".join(f"w{i}" for i in range(50))
    manifest = {
        "cubes": [
            {
                "name": "sales",
                "baseObject": "orders",
                "properties": {"description": long_desc},
                "measures": [
                    {"name": "revenue", "type": "DOUBLE", "expression": "SUM(x)"}
                ],
                "dimensions": [],
                "timeDimensions": [],
            }
        ]
    }
    (row,) = [i for i in extract_schema_items(manifest) if i["item_type"] == "cube"]
    parts = split_row_text(row["text"], "cube", _words, 20)
    assert len(parts) > 1
    assert all(p.startswith("Cube 'sales' over 'orders' (part ") for p in parts)
    assert all(_words(p) <= 20 for p in parts)
    joined = " ".join(parts)
    assert all(f"w{i}" in joined for i in range(50))
    assert "revenue" in joined


@pytest.mark.unit
def test_text_without_lead_falls_back_to_plain_chunking():
    parts = split_row_text(" ".join(f"w{i}" for i in range(30)), "model", _words, 8)
    assert parts and all(_words(p) <= 8 for p in parts)
    assert not any("(part" in p for p in parts)
