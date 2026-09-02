"""knowledge_indexer: every file validated against the token budget, chunked or skipped with a reason."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from wren.memory.knowledge_indexer import (
    KNOWLEDGE_ITEM_TYPES,
    KNOWLEDGE_TYPES,
    chunk_text,
    collect_knowledge_files,
    extract_knowledge_items,
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
def test_chunk_text_falls_through_paragraph_sentence_word():
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
def test_collect_knowledge_files_covers_every_subdir_but_sql(tmp_path):
    proj = _project(
        tmp_path,
        {
            "knowledge/rules/01.md": "r",
            "knowledge/glossary/g.md": "g",
            "knowledge/metrics/m.md": "m",
            "knowledge/caveats/c.md": "c",
            "knowledge/sql/q.md": "not collected",
            "knowledge/rules/notes.txt": "not md",
        },
    )
    got = collect_knowledge_files(proj)
    assert [(t, p.name) for t, p in got] == [
        ("rule", "01.md"),
        ("glossary", "g.md"),
        ("metric", "m.md"),
        ("caveat", "c.md"),
    ]
    assert set(KNOWLEDGE_ITEM_TYPES.values()) == KNOWLEDGE_TYPES


@pytest.mark.unit
def test_collect_knowledge_files_no_knowledge_dir(tmp_path):
    assert collect_knowledge_files(tmp_path) == []


@pytest.mark.unit
def test_extract_reports_every_file_and_never_exceeds_budget(tmp_path):
    proj = _project(
        tmp_path,
        {
            "knowledge/rules/short.md": "one two three",
            "knowledge/rules/long.md": "# T\n" + " ".join(f"w{i}" for i in range(40)),
            "knowledge/rules/empty.md": "   \n",
            "knowledge/glossary/bad.md": b"\xff\xfe\x00garbage",
        },
    )
    files = collect_knowledge_files(proj)
    items, reports = extract_knowledge_items(files, proj, "h1", _NOW, _words, 10)

    by_path = {r.path: r for r in reports}
    assert len(reports) == len(files) == 4

    assert by_path["knowledge/rules/short.md"].status == "embedded"
    assert by_path["knowledge/rules/short.md"].chunks == 1
    assert by_path["knowledge/rules/short.md"].tokens == 3

    long = by_path["knowledge/rules/long.md"]
    assert long.status == "chunked"
    assert long.chunks > 1
    assert long.largest_chunk <= 10

    assert by_path["knowledge/rules/empty.md"].status == "skipped"
    assert by_path["knowledge/rules/empty.md"].reason == "empty file"

    bad = by_path["knowledge/glossary/bad.md"]
    assert bad.status == "skipped"
    assert bad.reason.startswith("not UTF-8")

    # Items: one per chunk, shaped like schema_items rows.
    assert len(items) == 1 + long.chunks
    for it in items:
        assert _words(it["text"]) <= 10
        assert it["item_type"] in KNOWLEDGE_TYPES
        assert it["mdl_hash"] == "h1"
        assert it["indexed_at"] == _NOW
        assert it["model_name"] == ""
        assert it["is_calculated"] is False
        assert it["expression"] == it["item_name"].split("#")[0]

    names = [it["item_name"] for it in items if it["item_type"] == "rule"]
    assert "knowledge/rules/short.md" in names
    assert "knowledge/rules/long.md#1" in names

    summary = summarize_reports(reports)
    assert summary == {
        "files": 4,
        "chunks": 1 + long.chunks,
        "embedded": 1,
        "chunked": 1,
        "skipped": 2,
    }


@pytest.mark.unit
def test_extract_skips_when_nothing_fits(tmp_path):
    proj = _project(tmp_path, {"knowledge/caveats/x.md": "supercalifragilistic"})
    files = collect_knowledge_files(proj)

    def count(_: str) -> int:
        return 99

    items, reports = extract_knowledge_items(files, proj, "h", _NOW, count, 5)
    assert items == []
    assert reports[0].status == "skipped"
    assert "no chunk fits within 5 tokens" == reports[0].reason
