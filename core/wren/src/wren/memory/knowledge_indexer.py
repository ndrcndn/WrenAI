"""Turn ``knowledge/{rules,glossary,metrics,caveats}/*.md`` into embeddable rows.

Every file is validated against the embedding model's token budget before it
is embedded. A file that fits is stored as one row; a file that does not is
split on markdown headings, then paragraphs, then sentences, then words until
every chunk fits. Nothing is handed to the embedder unchecked, and every file
ends up in the report with one of three statuses: ``embedded`` (one row),
``chunked`` (several rows) or ``skipped`` (with a reason). The report is the
caller's evidence of what was — and was not — indexed.

``knowledge/sql`` is deliberately excluded: those NL→SQL pairs are indexed
into ``query_history`` by :mod:`wren.memory.markdown` / ``load_queries``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# knowledge/<subdir>  →  schema_items.item_type
KNOWLEDGE_ITEM_TYPES: dict[str, str] = {
    "rules": "rule",
    "glossary": "glossary",
    "metrics": "metric",
    "caveats": "caveat",
}

# Reverse lookup used by callers that need to select/delete knowledge rows.
KNOWLEDGE_TYPES: frozenset[str] = frozenset(KNOWLEDGE_ITEM_TYPES.values())

TokenCounter = Callable[[str], int]

_HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class KnowledgeFileReport:
    """Outcome for one knowledge file."""

    path: str  # project-relative, e.g. knowledge/rules/01_concepts.md
    item_type: str
    status: str  # embedded | chunked | skipped
    chunks: int = 0
    tokens: int = 0  # whole-file token count (0 when unreadable)
    largest_chunk: int = 0  # tokens in the largest emitted chunk
    reason: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def collect_knowledge_files(project_path: Path) -> list[tuple[str, Path]]:
    """Return ``(item_type, path)`` for every knowledge markdown file, sorted."""
    root = project_path / "knowledge"
    found: list[tuple[str, Path]] = []
    if not root.is_dir():
        return found
    for subdir, item_type in KNOWLEDGE_ITEM_TYPES.items():
        d = root / subdir
        if not d.is_dir():
            continue
        found.extend((item_type, p) for p in sorted(d.glob("*.md")) if p.is_file())
    return found


# ── Chunking ──────────────────────────────────────────────────────────────


def _pack(
    parts: list[str], count: TokenCounter, max_tokens: int, sep: str
) -> list[str]:
    """Greedily pack *parts* (each already ≤ max_tokens) into chunks ≤ max_tokens."""
    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}{sep}{part}" if current else part
        if current and count(candidate) > max_tokens:
            chunks.append(current)
            current = part
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _split_words(text: str, count: TokenCounter, max_tokens: int) -> list[str]:
    """Last resort: window over words. A single word over budget is dropped."""
    words = text.split()
    fitting = [w for w in words if count(w) <= max_tokens]
    return _pack(fitting, count, max_tokens, " ")


def _split_sentences(text: str, count: TokenCounter, max_tokens: int) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    parts: list[str] = []
    for s in sentences:
        if count(s) <= max_tokens:
            parts.append(s)
        else:
            parts.extend(_split_words(s, count, max_tokens))
    return _pack(parts, count, max_tokens, " ")


def _split_paragraphs(text: str, count: TokenCounter, max_tokens: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    parts: list[str] = []
    for p in paragraphs:
        if count(p) <= max_tokens:
            parts.append(p)
        else:
            parts.extend(_split_sentences(p, count, max_tokens))
    return _pack(parts, count, max_tokens, "\n\n")


def _split_sections(text: str) -> list[str]:
    """Split on markdown headings; each section keeps its heading line."""
    starts = [m.start() for m in _HEADING_RE.finditer(text)]
    if not starts:
        return [text]
    if starts[0] != 0:
        starts.insert(0, 0)
    starts.append(len(text))
    return [text[a:b].strip() for a, b in zip(starts, starts[1:]) if text[a:b].strip()]


def chunk_text(text: str, count: TokenCounter, max_tokens: int) -> list[str]:
    """Return chunks of *text* that each fit within *max_tokens*.

    Fast path: the whole text fits → one chunk. Otherwise split by heading,
    then paragraph, then sentence, then word. Every returned chunk satisfies
    ``count(chunk) <= max_tokens`` — verified before return, so the embedder
    never receives a string it would silently truncate.
    """
    text = text.strip()
    if not text:
        return []
    if count(text) <= max_tokens:
        return [text]

    chunks: list[str] = []
    for section in _split_sections(text):
        if count(section) <= max_tokens:
            chunks.append(section)
        else:
            chunks.extend(_split_paragraphs(section, count, max_tokens))
    # Post-condition: nothing over budget leaves this function.
    return [c for c in chunks if c and count(c) <= max_tokens]


# ── Items ─────────────────────────────────────────────────────────────────


def extract_knowledge_items(
    files: list[tuple[str, Path]],
    project_path: Path,
    mdl_hash: str,
    now: datetime,
    count: TokenCounter,
    max_tokens: int,
) -> tuple[list[dict], list[KnowledgeFileReport]]:
    """Validate, chunk and shape every knowledge file into ``schema_items`` rows.

    Returns ``(items, reports)``. ``items`` match the ``schema_items`` table
    columns (minus ``vector``); ``reports`` has exactly one entry per input
    file, so the caller can show what happened to each one.
    """
    items: list[dict] = []
    reports: list[KnowledgeFileReport] = []

    for item_type, path in files:
        try:
            rel = str(path.relative_to(project_path))
        except ValueError:
            rel = str(path)

        try:
            raw = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            reports.append(
                KnowledgeFileReport(
                    rel, item_type, "skipped", reason=f"not UTF-8: {e.reason}"
                )
            )
            continue
        except OSError as e:
            reports.append(
                KnowledgeFileReport(
                    rel, item_type, "skipped", reason=f"unreadable: {e.strerror}"
                )
            )
            continue

        text = raw.strip()
        if not text:
            reports.append(
                KnowledgeFileReport(rel, item_type, "skipped", reason="empty file")
            )
            continue

        total_tokens = count(text)
        chunks = chunk_text(text, count, max_tokens)
        if not chunks:
            reports.append(
                KnowledgeFileReport(
                    rel,
                    item_type,
                    "skipped",
                    tokens=total_tokens,
                    reason=f"no chunk fits within {max_tokens} tokens",
                )
            )
            continue

        largest = max(count(c) for c in chunks)
        status = "embedded" if len(chunks) == 1 else "chunked"
        reports.append(
            KnowledgeFileReport(
                rel,
                item_type,
                status,
                chunks=len(chunks),
                tokens=total_tokens,
                largest_chunk=largest,
            )
        )
        for i, chunk in enumerate(chunks):
            items.append(
                {
                    "text": chunk,
                    "item_type": item_type,
                    "model_name": "",
                    "item_name": rel if len(chunks) == 1 else f"{rel}#{i + 1}",
                    "data_type": None,
                    "expression": rel,
                    "is_calculated": False,
                    "mdl_hash": mdl_hash,
                    "indexed_at": now,
                }
            )

    return items, reports


def summarize_reports(reports: list[KnowledgeFileReport]) -> dict:
    """Roll a report list up into counts for CLI output / JSON."""
    by_status = {"embedded": 0, "chunked": 0, "skipped": 0}
    for r in reports:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    return {
        "files": len(reports),
        "chunks": sum(r.chunks for r in reports),
        **by_status,
    }
