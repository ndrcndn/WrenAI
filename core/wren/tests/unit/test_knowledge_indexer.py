"""knowledge_indexer: every source file validated against the token budget, chunked or skipped with a reason."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from wren.memory.knowledge_indexer import (
    CUBE_SOURCE_TYPE,
    MODEL_SOURCE_TYPE,
    RULE_TYPE,
    SOURCE_TYPES,
    VIEW_SOURCE_TYPE,
    chunk_text,
    collect_source_files,
    extract_source_items,
    summarize_reports,
)


def _words(text: str) -> int:
    """Deterministic stand-in for a tokenizer: one token per whitespace word."""
    return len(text.split())


_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


# ── chunk_text ────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_chunk_text_whole_when_fits():
    assert chunk_text("a b c", _words, 10) == ["a b c"]


@pytest.mark.unit
def test_chunk_text_empty():
    assert chunk_text("   \n\n ", _words, 10) == []


@pytest.mark.unit
def test_chunk_text_splits_on_headings_first():
    text = "# Title\nintro words here\n\n## Section A\nalpha beta gamma\n\n## Section B\ndelta epsilon"
    chunks = chunk_text(text, _words, 6)
    assert chunks == [
        "# Title\nintro words here",
        "## Section A\nalpha beta gamma",
        "## Section B\ndelta epsilon",
    ]


@pytest.mark.unit
def test_chunk_text_folds_heading_only_sections_into_next():
    # "# Concepts" has no body of its own: it must not become a chunk.
    text = "# Concepts\n\n## Account Manager\n" + " ".join(f"w{i}" for i in range(9))
    chunks = chunk_text(text, _words, 8)
    assert all(not c.strip().startswith("# Concepts\n\n") for c in chunks)
    assert not any(c.strip() == "# Concepts" for c in chunks)
    assert chunks[0].startswith("# Concepts\n## Account Manager\n")
    assert all(_words(c) <= 8 for c in chunks)
    # Sub-chunks keep the heading context when it fits.
    assert all(
        c.startswith("# Concepts\n## Account Manager") or "w" in c for c in chunks
    )


@pytest.mark.unit
def test_chunk_text_falls_through_paragraph_line_sentence_word():
    long_sentence = " ".join(f"w{i}" for i in range(25))  # 25 tokens, no punctuation
    text = f"## H\n{long_sentence}\n\nshort para. another one."
    chunks = chunk_text(text, _words, 8)
    assert chunks, "must produce chunks"
    # Post-condition: nothing over budget survives.
    assert all(_words(c) <= 8 for c in chunks)
    # Every word of the long sentence is retained somewhere.
    joined = " ".join(chunks)
    assert all(f"w{i}" in joined for i in range(25))
    assert "short para." in joined and "another one." in joined


@pytest.mark.unit
def test_chunk_text_yaml_splits_on_lines():
    yaml = "\n".join(f"- name: col{i}\n  type: int" for i in range(10))  # 40 words
    chunks = chunk_text(yaml, _words, 12)
    assert len(chunks) > 1
    assert all(_words(c) <= 12 for c in chunks)
    # Lines stay intact — no line is cut mid-way.
    assert all(ln in yaml.split("\n") for c in chunks for ln in c.split("\n"))


@pytest.mark.unit
def test_chunk_text_drops_single_oversized_word_only():
    text = "ok " + "x" * 50 + " fine"

    def count(s: str) -> int:
        # Pretend the giant word alone is over budget.
        return sum(3 if len(w) > 40 else 1 for w in s.split())

    chunks = chunk_text(text, count, 2)
    assert chunks == ["ok fine"]


# ── collect / extract ─────────────────────────────────────────────────────


def _project(tmp_path, files: dict[str, str | bytes]):
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content, encoding="utf-8")
    return tmp_path


@pytest.mark.unit
def test_collect_source_files_rules_models_views_cubes_only(tmp_path):
    proj = _project(
        tmp_path,
        {
            "knowledge/rules/01.md": "r",
            "knowledge/rules/notes.txt": "not md",
            "knowledge/glossary/g.md": "not collected",
            "knowledge/metrics/m.md": "not collected",
            "knowledge/caveats/c.md": "not collected",
            "knowledge/sql/q.md": "not collected",
            "models/orders/metadata.yml": "name: orders",
            "models/orders/ref_sql.sql": "not collected: SQL body",
            "models/orders/README.md": "not collected",
            "views/top/metadata.yml": "name: top",
            "views/top/sql.yml": "not collected: SQL body",
            "cubes/sales/metadata.yml": "name: sales",
            "cubes/stray.yml": "not under an entity dir",
        },
    )
    got = collect_source_files(proj)
    assert [(s.item_type, s.entity, s.path.name) for s in got] == [
        (RULE_TYPE, "", "01.md"),
        (MODEL_SOURCE_TYPE, "orders", "metadata.yml"),
        (VIEW_SOURCE_TYPE, "top", "metadata.yml"),
        (CUBE_SOURCE_TYPE, "sales", "metadata.yml"),
    ]
    assert SOURCE_TYPES == {
        RULE_TYPE,
        MODEL_SOURCE_TYPE,
        VIEW_SOURCE_TYPE,
        CUBE_SOURCE_TYPE,
    }


@pytest.mark.unit
def test_collect_source_files_empty_project(tmp_path):
    assert collect_source_files(tmp_path) == []


@pytest.mark.unit
def test_extract_reports_every_file_and_never_exceeds_budget(tmp_path):
    proj = _project(
        tmp_path,
        {
            "knowledge/rules/short.md": "one two three",
            "knowledge/rules/long.md": "# T\n" + " ".join(f"w{i}" for i in range(40)),
            "knowledge/rules/empty.md": "   \n",
            "knowledge/rules/bad.md": b"\xff\xfe\x00garbage",
            "models/orders/metadata.yml": "\n".join(
                f"- name: c{i}\n  type: int" for i in range(8)
            ),
        },
    )
    files = collect_source_files(proj)
    items, reports = extract_source_items(files, proj, "h1", _NOW, _words, 10)

    by_path = {r.path: r for r in reports}
    assert len(reports) == len(files) == 5

    assert by_path["knowledge/rules/short.md"].status == "embedded"
    assert by_path["knowledge/rules/short.md"].chunks == 1
    assert by_path["knowledge/rules/short.md"].tokens == 3

    long = by_path["knowledge/rules/long.md"]
    assert long.status == "chunked"
    assert long.chunks > 1
    assert long.largest_chunk <= 10

    assert by_path["knowledge/rules/empty.md"].status == "skipped"
    assert by_path["knowledge/rules/empty.md"].reason == "empty file"

    bad = by_path["knowledge/rules/bad.md"]
    assert bad.status == "skipped"
    assert bad.reason.startswith("not UTF-8")

    model = by_path["models/orders/metadata.yml"]
    assert model.item_type == MODEL_SOURCE_TYPE
    assert model.status == "chunked"

    # Items: one per chunk, shaped like schema_items rows.
    assert len(items) == 1 + long.chunks + model.chunks
    for it in items:
        assert _words(it["text"]) <= 10
        assert it["item_type"] in SOURCE_TYPES
        assert it["mdl_hash"] == "h1"
        assert it["indexed_at"] == _NOW
        assert it["is_calculated"] is False
        assert it["expression"] == it["item_name"].split("#")[0]

    rule_names = [it["item_name"] for it in items if it["item_type"] == RULE_TYPE]
    assert "knowledge/rules/short.md" in rule_names
    assert "knowledge/rules/long.md#1" in rule_names
    assert all(it["model_name"] == "" for it in items if it["item_type"] == RULE_TYPE)
    # Source rows carry their entity so model_name filters find them.
    assert all(
        it["model_name"] == "orders"
        for it in items
        if it["item_type"] == MODEL_SOURCE_TYPE
    )

    summary = summarize_reports(reports)
    assert summary == {
        "files": 5,
        "chunks": 1 + long.chunks + model.chunks,
        "by_type": {RULE_TYPE: 1 + long.chunks, MODEL_SOURCE_TYPE: model.chunks},
        "embedded": 1,
        "chunked": 2,
        "skipped": 2,
    }


@pytest.mark.unit
def test_extract_skips_when_nothing_fits(tmp_path):
    proj = _project(tmp_path, {"knowledge/rules/x.md": "supercalifragilistic"})
    files = collect_source_files(proj)

    def count(_: str) -> int:
        return 99

    items, reports = extract_source_items(files, proj, "h", _NOW, count, 5)
    assert items == []
    assert reports[0].status == "skipped"
    assert "no chunk fits within 5 tokens" == reports[0].reason
