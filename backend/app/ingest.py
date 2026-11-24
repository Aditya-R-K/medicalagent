"""Turn PDFs and text files into searchable passages stored in Postgres.

Passages are built by packing paragraphs and never span a page boundary, so the
page number attached to every citation is exact rather than approximate.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from . import db, store
from .providers import EmbeddingModel, count_tokens
from .retrieve import vector_literal

log = logging.getLogger(__name__)

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED = TEXT_SUFFIXES | PDF_SUFFIXES
PAGE_BATCH = 40

HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
YEAR = re.compile(r"(?:19|20)\d{2}")

_lock = threading.Lock()
_running = False


@dataclass
class Block:
    text: str
    page: int
    section: str
    tokens: int = 0
    line_start: int | None = None   # line numbers on the page, for exact references
    line_end: int | None = None

    def __post_init__(self) -> None:
        self.tokens = self.tokens or count_tokens(self.text)


@dataclass
class Parsed:
    pages: int = 0
    year: int | None = None
    blocks: list[Block] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:32]


def page_count(path: Path) -> int:
    if path.suffix.lower() not in PDF_SUFFIXES:
        return 1
    try:
        import pymupdf

        with pymupdf.open(str(path)) as document:
            return document.page_count
    except Exception:
        return 0


def publication_year(filename: str, body: str) -> int | None:
    """Filename first: a superseded edition often cites the one that replaced it."""
    limit = datetime.date.today().year + 1

    def found(text: str) -> list[int]:
        return [int(y) for y in YEAR.findall(text) if 1900 <= int(y) <= limit]

    from_name = found(filename)
    if from_name:
        return max(from_name)
    from_body = found(body[:2000])
    return max(from_body) if from_body else None


def document_metadata(path: Path) -> tuple[str, str]:
    """(title, author) from a PDF's own metadata, when it holds anything useful.

    Many PDFs carry junk here ("Untitled", "Microsoft Word - ch3.docx"); that
    is discarded and the reference falls back to a title made from the name.
    """
    if path.suffix.lower() not in PDF_SUFFIXES:
        return "", ""
    try:
        import pymupdf

        with pymupdf.open(str(path)) as document:
            meta = document.metadata or {}
    except Exception:
        return "", ""
    title = " ".join((meta.get("title") or "").split())
    author = " ".join((meta.get("author") or "").split())
    junk = ("untitled", "microsoft word", ".doc", ".pdf", "unknown", "title")
    if len(title) < 4 or any(j in title.lower() for j in junk):
        title = ""
    if len(author) < 3 or author.lower() in ("unknown", "author", "admin", "user"):
        author = ""
    return title[:300], author[:300]


def unwrap_lines(text: str) -> list[tuple[str, int, int]]:
    """Rejoin PDF line wrapping into paragraphs, keeping where each came from.

    Returns (paragraph, first line, last line), lines numbered from 1 as they
    sit on the page. Those numbers are what lets a reference say "p. 412,
    lines 14-31" rather than just the page.
    """
    output: list[tuple[str, int, int]] = []
    buffer: list[str] = []
    first = 0
    for number, raw in enumerate(text.split("\n"), start=1):
        line = raw.strip()
        if not line:
            if buffer:
                output.append((" ".join(buffer), first, number - 1))
                buffer = []
            continue
        if not buffer:
            first = number
        if buffer and buffer[-1].endswith("-"):
            buffer[-1] = buffer[-1][:-1] + line
            continue
        buffer.append(line)
        if line.endswith((".", ":", ";", "?", "!")):
            output.append((" ".join(buffer), first, number))
            buffer = []
    if buffer:
        output.append((" ".join(buffer), first, len(text.split("\n"))))
    return output


def unwrap(text: str) -> str:
    """Rejoin PDF line wrapping into sentences and repair split words."""
    output: list[str] = []
    buffer: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            if buffer:
                output.append(" ".join(buffer))
                buffer = []
            continue
        if buffer and buffer[-1].endswith("-"):
            buffer[-1] = buffer[-1][:-1] + line
            continue
        buffer.append(line)
        if line.endswith((".", ":", ";", "?", "!")):
            output.append(" ".join(buffer))
            buffer = []
    if buffer:
        output.append(" ".join(buffer))
    return "\n\n".join(output)


def read_pages(path: Path, mode: str, on_page=None
               ) -> Iterator[tuple[int, str, list[tuple[int, int]] | None]]:
    """Yield (page number, text, line spans), reading in batches from one handle.

    Line spans give, for each non-empty line of `text`, the range of original
    lines it came from; None where the original lines are not known (the
    'thorough' markdown conversion rewrites the layout).

    'thorough' keeps headings and reads scanned pages, at around a second per
    page. 'fast' is roughly two hundred times quicker for about 97% of the text.
    """
    if path.suffix.lower() not in PDF_SUFFIXES:
        body = path.read_text(encoding="utf-8", errors="replace")
        spans = [(n, n) for n, line in enumerate(body.splitlines(), start=1) if line.strip()]
        yield 1, body, spans
        return

    import pymupdf

    to_markdown = None
    if mode != "fast":
        try:
            import pymupdf4llm

            to_markdown = pymupdf4llm.to_markdown
        except ImportError:
            log.warning("pymupdf4llm missing; using fast extraction")

    with pymupdf.open(str(path)) as document:
        total = document.page_count
        for start in range(0, total, PAGE_BATCH):
            indices = list(range(start, min(start + PAGE_BATCH, total)))
            texts: list[str] = []
            spans: list = []
            if to_markdown is not None:
                try:
                    texts = [
                        item.get("text", "")
                        for item in to_markdown(document, pages=indices, page_chunks=True)
                    ]
                except Exception as exc:
                    log.warning("%s: %s; falling back to fast extraction", path.name, exc)
            if not texts:
                for i in indices:
                    paragraphs = unwrap_lines(document.load_page(i).get_text())
                    texts.append("\n\n".join(p[0] for p in paragraphs))
                    spans.append([(p[1], p[2]) for p in paragraphs])
            for offset, text in enumerate(texts):
                yield indices[offset] + 1, text, (spans[offset] if spans else None)
            if on_page:
                on_page(min(start + PAGE_BATCH, total), total)


def parse(path: Path, mode: str, on_page=None) -> Parsed:
    result = Parsed()
    stack: list[str] = []

    for number, text, spans in read_pages(path, mode, on_page):
        result.pages = number
        buffer: list[str] = []
        lines: list[tuple[int, int]] = []
        seen = 0  # non-empty lines so far, to look up each one's span

        def flush() -> None:
            joined = "\n".join(buffer).strip()
            buffer.clear()
            first = min((a for a, _ in lines), default=None)
            last = max((b for _, b in lines), default=None)
            lines.clear()
            if joined:
                result.blocks.append(Block(joined, number, " > ".join(stack),
                                           line_start=first, line_end=last))

        for line in text.splitlines():
            if line.strip():
                span = spans[seen] if spans and seen < len(spans) else None
                seen += 1
            else:
                span = None
            heading = HEADING.match(line.strip())
            if heading:
                flush()
                depth = len(heading.group(1))
                del stack[depth - 1 :]
                stack.append(heading.group(2))
            elif not line.strip():
                flush()
            else:
                buffer.append(line.rstrip())
                if span:
                    lines.append(span)
        flush()

    result.year = publication_year(
        path.name, "\n\n".join(b.text for b in result.blocks[:6])
    )
    return result


# --------------------------------------------------------------------------- #
# passages
# --------------------------------------------------------------------------- #


@dataclass
class Passage:
    text: str
    parent_text: str
    section: str
    page_start: int
    page_end: int
    line_start: int | None = None
    line_end: int | None = None


def _pack(blocks: list[Block], budget: int) -> list[list[Block]]:
    groups: list[list[Block]] = []
    current: list[Block] = []
    used = 0
    for block in blocks:
        if current and used + block.tokens > budget:
            groups.append(current)
            current, used = [], 0
        current.append(block)
        used += block.tokens
    if current:
        groups.append(current)
    return groups


def _split(block: Block, budget: int, overlap: int) -> list[Block]:
    if block.tokens <= budget:
        return [block]
    words = block.text.split()
    size = max(1, int(budget / 1.3))
    step = max(1, size - max(1, int(overlap / 1.3)))
    pieces = []
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if not window:
            break
        pieces.append(Block(" ".join(window), block.page, block.section,
                            line_start=block.line_start, line_end=block.line_end))
        if start + size >= len(words):
            break
    return pieces


def to_passages(parsed: Parsed, settings) -> list[Passage]:
    passages: list[Passage] = []
    for parent in _pack(parsed.blocks, settings.parent_tokens):
        parent_text = "\n\n".join(b.text for b in parent)
        expanded: list[Block] = []
        for block in parent:
            expanded.extend(_split(block, settings.child_tokens, settings.child_overlap))

        tail = ""
        for group in _pack(expanded, settings.child_tokens):
            body = "\n\n".join(b.text for b in group)
            pages = [b.page for b in group]
            passages.append(
                Passage(
                    text=f"{tail}\n\n{body}".strip() if tail else body,
                    parent_text=parent_text,
                    section=group[0].section,
                    page_start=min(pages),
                    page_end=max(pages),
                    line_start=group[0].line_start,
                    line_end=group[-1].line_end,
                )
            )
            tail = " ".join(body.split()[-max(1, int(settings.child_overlap / 1.3)) :])
    return passages


# --------------------------------------------------------------------------- #
# scanning and indexing
# --------------------------------------------------------------------------- #


def scan(settings) -> list[dict]:
    """Reconcile the documents folder with the database and return the listing."""
    root = settings.document_root
    root.mkdir(parents=True, exist_ok=True)

    # Keyed by the path relative to the folder, so a file in a subfolder is
    # found again at indexing time and two files with the same name in
    # different subfolders do not collide.
    on_disk = {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED and path.name != "README.md"
    }
    known = {row["filename"]: row for row in db.query("SELECT * FROM documents")}

    for name in set(known) - set(on_disk):
        db.execute("DELETE FROM documents WHERE filename = %s", (name,))
        log.info("removed %s: no longer on disk", name)

    for name, path in on_disk.items():
        digest = file_hash(path)
        record = known.get(name)
        if record is None:
            title, author = document_metadata(path)
            db.execute(
                """
                INSERT INTO documents (filename, content_hash, pages, size_bytes, title, author)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (name, digest, page_count(path), path.stat().st_size, title, author),
            )
        elif record["content_hash"] != digest:
            # An excluded document stays excluded even when its file changes;
            # re-including it is the operator's decision, not a side effect.
            status = "excluded" if record["status"] == "excluded" else "pending"
            db.execute(
                """
                UPDATE documents
                SET content_hash = %s, pages = %s, size_bytes = %s,
                    status = %s, chunk_count = 0, progress = 0,
                    indexed_at = NULL, error = NULL
                WHERE id = %s
                """,
                (digest, page_count(path), path.stat().st_size, status, record["id"]),
            )
            db.execute("DELETE FROM chunks WHERE document_id = %s", (record["id"],))

    return db.query("SELECT * FROM documents ORDER BY filename")


def is_running() -> bool:
    return _running


def index_pending(settings, rebuild: bool = False) -> None:
    """Index every document that is not already indexed. Safe to call twice."""
    global _running
    with _lock:
        if _running:
            log.info("indexing already in progress")
            return
        _running = True

    try:
        scan(settings)
        if rebuild:
            db.execute("TRUNCATE chunks")
            db.execute(
                "UPDATE documents SET status = 'pending', chunk_count = 0, "
                "progress = 0, indexed_at = NULL, error = NULL"
            )

        pending = db.query(
            "SELECT * FROM documents WHERE status IN ('pending', 'failed') ORDER BY size_bytes"
        )
        if not pending:
            log.info("nothing to index")
            return

        embedder = EmbeddingModel(settings)
        log.info("indexing %d document(s) with %s", len(pending), embedder.name)
        for record in pending:
            _index_one(settings, embedder, record)
    finally:
        _running = False


def _index_one(settings, embedder: EmbeddingModel, record: dict) -> None:
    path = settings.document_root / record["filename"]
    document_id = record["id"]

    if not path.exists():
        db.execute(
            "UPDATE documents SET status = 'failed', error = %s WHERE id = %s",
            ("file is missing from the documents folder", document_id),
        )
        return

    db.execute(
        "UPDATE documents SET status = 'indexing', progress = 0, error = NULL WHERE id = %s",
        (document_id,),
    )

    try:
        def on_page(done: int, total: int) -> None:
            db.execute(
                "UPDATE documents SET progress = %s WHERE id = %s",
                (round(done / max(total, 1) * 0.5, 4), document_id),
            )

        parsed = parse(path, settings.pdf_extraction, on_page)
        passages = to_passages(parsed, settings)
        if not passages:
            raise ValueError("no text could be extracted; the file may be scanned images")

        log.info("%s: %d pages, %d passages", record["filename"], parsed.pages, len(passages))
        db.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))

        written = 0
        batch = 256
        refused_before = embedder.rejected
        spent: list[dict] = []
        embedder.meter = spent.append
        for start in range(0, len(passages), batch):
            window = passages[start : start + batch]
            # A passage the provider's content screening refused comes back as
            # None and is stored without a vector: keyword search still finds it.
            vectors = embedder.embed([p.text for p in window])
            with db.connection() as conn:
                with conn.cursor() as cursor:
                    cursor.executemany(
                        """
                        INSERT INTO chunks
                            (document_id, text, parent_text, section,
                             page_start, page_end, line_start, line_end, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                        """,
                        [
                            (
                                document_id, p.text, p.parent_text, p.section,
                                p.page_start, p.page_end, p.line_start, p.line_end,
                                vector_literal(vector) if vector is not None else None,
                            )
                            for p, vector in zip(window, vectors)
                        ],
                    )
            written += len(window)
            db.execute(
                "UPDATE documents SET progress = %s, chunk_count = %s WHERE id = %s",
                (round(0.5 + written / len(passages) * 0.5, 4), written, document_id),
            )

        refused = embedder.rejected - refused_before
        embedder.meter = None
        if spent:
            # One analytics row per document: what indexing it cost.
            store.record_usage([{
                "step": "index", "provider": spent[0]["provider"], "model": spent[0]["model"],
                "input": sum(r["input"] for r in spent), "output": 0, "cached": 0,
                "cache_write": 0, "calls": len(spent),
            }], document_id=document_id)
        title, author = document_metadata(path)
        if title or author:
            db.execute("UPDATE documents SET title = %s, author = %s WHERE id = %s",
                       (title, author, document_id))
        note = (
            f"{refused} of {written} passages were refused by {embedder.name}'s content "
            "safety screening; they are searchable by keyword only."
            if refused else None
        )
        db.execute(
            """
            UPDATE documents
            SET status = 'indexed', progress = 1, chunk_count = %s, pages = %s,
                year = %s, indexed_at = now(), error = %s
            WHERE id = %s
            """,
            (written, parsed.pages, parsed.year, note, document_id),
        )
        log.info("%s: indexed %d passages%s", record["filename"], written,
                 f" ({refused} refused by content screening)" if refused else "")

    except Exception as exc:
        log.exception("failed to index %s", record["filename"])
        db.execute(
            "UPDATE documents SET status = 'failed', error = %s WHERE id = %s",
            (str(exc)[:500], document_id),
        )
