"""Turn project source files into embeddable rows.

Sources:

* ``knowledge/rules/*.md``            → ``item_type="rule"``
* ``models/<name>/metadata.yml``      → ``item_type="model_source"`` (model_name=<name>)
* ``views/<name>/metadata.yml``       → ``item_type="view_source"``  (model_name=<name>)
* ``cubes/<name>/metadata.yml``       → ``item_type="cube_source"``  (model_name=<name>)

SQL bodies (``views/<name>/sql.yml``, ``models/<name>/*.sql``) are not embedded:
they are fragments with no natural-language signal; the agent reads exact SQL
through ``describe_model`` / the ``wren://knowledge`` resources.

These complement the compiled-MDL rows (model/column/view/cube summaries): the
MDL rows are short synthesised descriptions, the source rows are the full
file content, so a question can match on anything written in the definition.

Every file is validated against the embedding model's token budget before it
is embedded. A file that fits is stored as one row; a file that does not is
split on markdown headings, then paragraphs, then lines, then sentences, then
words until every chunk fits. Nothing is handed to the embedder unchecked, and
every file ends up in the report with one of three statuses: ``embedded`` (one
row), ``chunked`` (several rows) or ``skipped`` (with a reason). The report is
the caller's evidence of what was — and was not — indexed.

``knowledge/sql`` is deliberately excluded: those NL→SQL pairs are indexed
into ``query_history`` by :mod:`wren.memory.markdown` / ``load_queries``.
Other ``knowledge/`` subdirectories are not embedded.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# item_type values written by this module; used to select/replace its rows.
RULE_TYPE = "rule"
MODEL_SOURCE_TYPE = "model_source"
VIEW_SOURCE_TYPE = "view_source"
CUBE_SOURCE_TYPE = "cube_source"
SOURCE_TYPES: frozenset[str] = frozenset(
    {RULE_TYPE, MODEL_SOURCE_TYPE, VIEW_SOURCE_TYPE, CUBE_SOURCE_TYPE}
)

# <project>/<dir>/<entity>/<file>  →  item_type
_ENTITY_DIRS: dict[str, str] = {
    "models": MODEL_SOURCE_TYPE,
    "views": VIEW_SOURCE_TYPE,
    "cubes": CUBE_SOURCE_TYPE,
}
_ENTITY_FILES = ("metadata.yml", "metadata.yaml")

TokenCounter = Callable[[str], int]

_HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class SourceFile:
    """One file to embed: its row type, path and owning entity (if any)."""

    item_type: str
    path: Path
    entity: str = ""  # model/view/cube name; "" for rules


@dataclass
class SourceFileReport:
    """Outcome for one source file."""

    path: str  # project-relative, e.g. knowledge/rules/01_concepts.md
    item_type: str
    status: str  # embedded | chunked | skipped
    chunks: int = 0
    tokens: int = 0  # whole-file token count (0 when unreadable)
    largest_chunk: int = 0  # tokens in the largest emitted chunk
    reason: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def collect_source_files(project_path: Path) -> list[SourceFile]:
    """Return every embeddable source file, sorted by path within each group."""
    found: list[SourceFile] = []

    rules_dir = project_path / "knowledge" / "rules"
    if rules_dir.is_dir():
        found.extend(
            SourceFile(RULE_TYPE, p)
            for p in sorted(rules_dir.glob("*.md"))
            if p.is_file()
        )

    for dirname, item_type in _ENTITY_DIRS.items():
        root = project_path / dirname
        if not root.is_dir():
            continue
        for entity_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for f in sorted(entity_dir.iterdir()):
                if f.is_file() and f.name.lower() in _ENTITY_FILES:
                    found.append(SourceFile(item_type, f, entity_dir.name))
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


def _split_lines(text: str, count: TokenCounter, max_tokens: int) -> list[str]:
    """Line-level split — the natural unit for YAML and SQL bodies."""
    lines = [ln.rstrip() for ln in text.split("\n") if ln.strip()]
    parts: list[str] = []
    for ln in lines:
        if count(ln) <= max_tokens:
            parts.append(ln)
        else:
            parts.extend(_split_sentences(ln, count, max_tokens))
    return _pack(parts, count, max_tokens, "\n")


def _split_paragraphs(text: str, count: TokenCounter, max_tokens: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    parts: list[str] = []
    for p in paragraphs:
        if count(p) <= max_tokens:
            parts.append(p)
        else:
            parts.extend(_split_lines(p, count, max_tokens))
    return _pack(parts, count, max_tokens, "\n\n")


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split on markdown headings into ``(heading_lines, body)`` pairs.

    A section whose body is empty (a bare ``# Title`` followed directly by a
    sub-heading) is folded into the next section's heading context instead of
    becoming its own chunk — a heading alone carries nothing to embed. Text
    without headings (YAML, SQL) is a single section with no heading.
    """
    starts = [m.start() for m in _HEADING_RE.finditer(text)]
    if not starts:
        return [("", text.strip())]
    if starts[0] != 0:
        starts.insert(0, 0)
    starts.append(len(text))

    sections: list[tuple[str, str]] = []
    pending: list[str] = []
    for a, b in zip(starts, starts[1:]):
        raw = text[a:b].strip()
        if not raw:
            continue
        first, _, rest = raw.partition("\n")
        if _HEADING_RE.match(first):
            heading, body = first.strip(), rest.strip()
        else:
            heading, body = "", raw
        if not body:
            pending.append(heading)
            continue
        heading_lines = "\n".join([*pending, heading]).strip() if pending else heading
        pending = []
        sections.append((heading_lines, body))
    return sections


def _with_heading(heading: str, part: str, count: TokenCounter, max_tokens: int) -> str:
    """Prefix *part* with its heading context when the result still fits."""
    if not heading:
        return part
    candidate = f"{heading}\n{part}"
    return candidate if count(candidate) <= max_tokens else part


def chunk_text(text: str, count: TokenCounter, max_tokens: int) -> list[str]:
    """Return chunks of *text* that each fit within *max_tokens*.

    Fast path: the whole text fits → one chunk. Otherwise split by heading,
    then paragraph, then line, then sentence, then word; every sub-chunk keeps
    its heading line(s) as context when that still fits. Every returned chunk
    satisfies ``count(chunk) <= max_tokens`` — verified before return, so the
    embedder never receives a string it would silently truncate.
    """
    text = text.strip()
    if not text:
        return []
    if count(text) <= max_tokens:
        return [text]

    chunks: list[str] = []
    for heading, body in _split_sections(text):
        whole = f"{heading}\n{body}" if heading else body
        if count(whole) <= max_tokens:
            chunks.append(whole)
            continue
        # Reserve room for the heading so packed parts can carry it.
        budget = max_tokens - (count(heading) if heading else 0)
        if budget < max_tokens // 4:
            budget = max_tokens  # heading too long to repeat; parts go bare
        for part in _split_paragraphs(body, count, budget):
            chunks.append(_with_heading(heading, part, count, max_tokens))
    # Post-condition: nothing over budget leaves this function.
    return [c for c in chunks if c and count(c) <= max_tokens]


# ── Items ─────────────────────────────────────────────────────────────────


def extract_source_items(
    files: list[SourceFile],
    project_path: Path,
    mdl_hash: str,
    now: datetime,
    count: TokenCounter,
    max_tokens: int,
) -> tuple[list[dict], list[SourceFileReport]]:
    """Validate, chunk and shape every source file into ``schema_items`` rows.

    Returns ``(items, reports)``. ``items`` match the ``schema_items`` table
    columns (minus ``vector``); ``reports`` has exactly one entry per input
    file, so the caller can show what happened to each one.
    """
    items: list[dict] = []
    reports: list[SourceFileReport] = []

    for src in files:
        item_type, path = src.item_type, src.path
        try:
            rel = str(path.relative_to(project_path))
        except ValueError:
            rel = str(path)

        try:
            raw = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            reports.append(
                SourceFileReport(
                    rel, item_type, "skipped", reason=f"not UTF-8: {e.reason}"
                )
            )
            continue
        except OSError as e:
            reports.append(
                SourceFileReport(
                    rel, item_type, "skipped", reason=f"unreadable: {e.strerror}"
                )
            )
            continue

        text = raw.strip()
        if not text:
            reports.append(
                SourceFileReport(rel, item_type, "skipped", reason="empty file")
            )
            continue

        total_tokens = count(text)
        chunks = chunk_text(text, count, max_tokens)
        if not chunks:
            reports.append(
                SourceFileReport(
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
            SourceFileReport(
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
                    "model_name": src.entity,
                    "item_name": rel if len(chunks) == 1 else f"{rel}#{i + 1}",
                    "data_type": None,
                    "expression": rel,
                    "is_calculated": False,
                    "mdl_hash": mdl_hash,
                    "indexed_at": now,
                }
            )

    return items, reports


def summarize_reports(reports: list[SourceFileReport]) -> dict:
    """Roll a report list up into counts for CLI output / JSON."""
    by_status = {"embedded": 0, "chunked": 0, "skipped": 0}
    by_type: dict[str, int] = {}
    for r in reports:
        by_status[r.status] = by_status.get(r.status, 0) + 1
        by_type[r.item_type] = by_type.get(r.item_type, 0) + r.chunks
    return {
        "files": len(reports),
        "chunks": sum(r.chunks for r in reports),
        "by_type": by_type,
        **by_status,
    }
