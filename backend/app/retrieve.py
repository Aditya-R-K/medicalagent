"""Hybrid retrieval over Postgres: vector search plus full-text search.

Both are needed. Vector search finds wording the question did not use; full-text
search finds the exact strings vector search blurs, such as doses and codes.
Results are combined by rank rather than score, so the two scales never have to
be reconciled.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass

from . import db
from .providers import EmbeddingModel, is_safety_rejection

log = logging.getLogger(__name__)

RRF_CONSTANT = 60
MAX_TERMS = 12

CITATION = re.compile(r"\[(\d+)\]")
# Two letters and up: DH, MI, TB and other abbreviations carry the meaning.
WORD = re.compile(r"[A-Za-z][A-Za-z0-9-]+|\d+[A-Za-z]*")

CHUNK_COLUMNS = """
    c.id, c.document_id, c.text, c.parent_text, c.section, c.page_start, c.page_end,
    c.line_start, c.line_end, d.filename, d.year, d.title, d.author
"""


def readable_title(filename: str) -> str:
    """A book-like title from a file name, for PDFs that carry no title."""
    stem = re.sub(r"\.(pdf|txt|md|markdown)$", "", filename.rsplit("/", 1)[-1], flags=re.I)
    return re.sub(r"[_]+", " ", stem).strip()


def any_words_query(text: str) -> str:
    """A tsquery matching ANY content word, for whole sentences.

    websearch_to_tsquery ANDs every word, so a full question almost never
    matches; OR-ing the words and ranking by cover density rewards passages
    that hold many of them close together instead. Stop words are dropped by
    Postgres itself.
    """
    # Hyphenated terms (anti-TNF) are split: a hyphen is tsquery syntax.
    words = dict.fromkeys(
        part.lower() for w in WORD.findall(text) for part in w.split("-") if part
    )
    return " | ".join(w for w in words if w)[:2000]


def vector_literal(values) -> str:
    """pgvector accepts its own text form, which avoids depending on an adapter."""
    return "[" + ",".join(f"{float(v):.7g}" for v in values) + "]"


@dataclass
class Passage:
    id: int
    text: str
    parent_text: str
    section: str
    page_start: int
    page_end: int
    filename: str
    year: int | None
    score: float = 0.0
    signals: list[str] = None
    document_id: int = 0
    line_start: int | None = None   # first line on page_start
    line_end: int | None = None     # last line on page_end
    title: str = ""
    author: str = ""

    def __post_init__(self) -> None:
        self.signals = self.signals or []

    @property
    def pages(self) -> str:
        return (
            f"p.{self.page_start}"
            if self.page_start == self.page_end
            else f"pp.{self.page_start}-{self.page_end}"
        )

    @property
    def location(self) -> str:
        """Page and line: "p. 412, lines 14-31" or "pp. 411-412 (l. 30 to l. 12)"."""
        if not (self.line_start and self.line_end):
            return self.pages.replace("p.", "p. ").replace("pp. ", "pp. ")
        if self.page_start == self.page_end:
            lines = (f"line {self.line_start}" if self.line_start == self.line_end
                     else f"lines {self.line_start}-{self.line_end}")
            return f"p. {self.page_start}, {lines}"
        return (f"pp. {self.page_start}-{self.page_end} "
                f"(p. {self.page_start} l. {self.line_start} to p. {self.page_end} l. {self.line_end})")

    @property
    def book(self) -> str:
        return self.title or readable_title(self.filename)

    @property
    def citation(self) -> str:
        year = f", {self.year}" if self.year else ""
        return f"{self.book}{year}, {self.location}"

    def vancouver(self) -> str:
        """A reference in Vancouver style, the convention in medical writing."""
        author = f"{self.author}. " if self.author else ""
        year = f" {self.year}" if self.year else ""
        pages = (f"p. {self.page_start}" if self.page_start == self.page_end
                 else f"p. {self.page_start}-{self.page_end}")
        return f"{author}{self.book}.{year}. {pages}."

    def to_dict(self, index: int) -> dict:
        return {
            "index": index,
            "source": self.filename,
            "document_id": self.document_id,
            "title": self.book,
            "author": self.author,
            "year": self.year,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "location": self.location,
            "section": self.section,
            "citation": self.citation,
            "vancouver": self.vancouver(),
            "score": round(self.score, 4),
            "signals": self.signals,
            "excerpt": self.text[:2000],
        }


class Retriever:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.embedder = EmbeddingModel(settings)

    def _nearest(self, vector: list[float], limit: int) -> list[int]:
        rows = db.query(
            """
            SELECT c.id
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.status = 'indexed' AND c.embedding IS NOT NULL
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            (vector_literal(vector), limit),
        )
        return [row["id"] for row in rows]

    def _by_keyword(self, query: str, limit: int) -> list[int]:
        """Every word must match: right for short index terms like "dapsone dose"."""
        rows = db.query(
            """
            SELECT c.id
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.status = 'indexed'
              AND c.search @@ websearch_to_tsquery('english', %s)
            ORDER BY ts_rank(c.search, websearch_to_tsquery('english', %s)) DESC
            LIMIT %s
            """,
            (query, query, limit),
        )
        return [row["id"] for row in rows]

    def _by_any_word(self, sentence: str, limit: int) -> list[int]:
        """Any word may match, ranked by how many appear close together: right
        for whole questions, which an all-words match almost never finds."""
        expression = any_words_query(sentence)
        if not expression:
            return []
        try:
            rows = db.query(
                """
                SELECT c.id
                FROM chunks c
                JOIN documents d ON d.id = c.document_id,
                     to_tsquery('english', %s) q
                WHERE d.status = 'indexed' AND c.search @@ q
                ORDER BY ts_rank_cd(c.search, q) DESC
                LIMIT %s
                """,
                (expression, limit),
            )
        except Exception as exc:  # a malformed expression must not sink the search
            log.info("any-word search skipped: %s", exc)
            return []
        return [row["id"] for row in rows]

    def search(
        self,
        question: str,
        variants: list[str] | None = None,
        hypothetical: str | None = None,
        terms: list[str] | None = None,
        top_k: int | None = None,
        exclude: set[int] | None = None,
    ) -> list[Passage]:
        """Search with every angle available.

        `hypothetical` is a written-out guess at what the answer looks like. A
        textbook passage resembles an answer far more than it resembles a
        question, so embedding the guess and searching with that finds material
        the question's own wording misses.

        `terms` are short index-style phrases searched one at a time:
        websearch_to_tsquery ANDs every word it is given, so a whole sentence
        over-constrains the match while a two-word term is exactly the kind of
        string the tsvector holds.
        """
        queries = [question] + [v for v in (variants or []) if v and v != question]
        limit = self.settings.candidates
        top_k = top_k or self.settings.passages

        fused: dict[int, float] = {}
        signals: dict[int, set[str]] = {}

        def absorb(name: str, ids: list[int], weight: float = 1.0) -> None:
            for rank, identifier in enumerate(ids):
                fused[identifier] = fused.get(identifier, 0.0) + weight / (
                    RRF_CONSTANT + rank + 1
                )
                signals.setdefault(identifier, set()).add(name)

        # One embeddings request for every query at once, instead of a round
        # trip per variant.
        texts = queries + ([hypothetical] if hypothetical else [])
        try:
            vectors = self.embedder.embed(texts, kind="query")
        except Exception as exc:
            if not is_safety_rejection(exc):
                raise
            # The provider's content screening refused the question itself.
            # Keyword search needs no provider, so the student still gets an
            # answer from the documents.
            log.warning("query embedding refused by content screening; keyword search only")
            vectors = []

        for position, query in enumerate(queries):
            if vectors:
                absorb("meaning", self._nearest(vectors[position], limit))
            absorb("keyword", self._by_any_word(query, limit), weight=0.8)
        if hypothetical and vectors:
            absorb("hypothesis", self._nearest(vectors[-1], limit), weight=1.2)

        cleaned = dict.fromkeys(t.strip() for t in (terms or []) if t and t.strip())
        for term in list(cleaned)[:MAX_TERMS]:
            absorb("term", self._by_keyword(term, limit), weight=0.9)

        for seen in exclude or ():
            fused.pop(seen, None)
        if not fused:
            return []

        best = sorted(fused, key=fused.get, reverse=True)[:top_k]
        rows = db.query(
            f"""
            SELECT {CHUNK_COLUMNS}
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id = ANY(%s)
            """,
            (best,),
        )

        top = max(fused.values()) or 1.0
        passages = []
        for row in rows:
            passage = self._passage(row)
            passage.score = fused[row["id"]] / top
            passage.signals = sorted(signals[row["id"]])
            passages.append(passage)
        passages.sort(key=lambda p: p.score, reverse=True)
        return passages

    def neighbours(self, ids: list[int], radius: int = 1) -> dict[int, Passage]:
        """The passages either side of each given one, in the same document.

        Passages are stored in reading order, so ids next to each other in one
        document are consecutive text: expanding a strong hit with its
        neighbours restores the sentences around it without sending the whole
        surrounding section.
        """
        if not ids:
            return {}
        rows = db.query(
            f"""
            SELECT DISTINCT {CHUNK_COLUMNS}
            FROM chunks h
            JOIN chunks c ON c.document_id = h.document_id
                         AND c.id BETWEEN h.id - %s AND h.id + %s
            JOIN documents d ON d.id = c.document_id
            WHERE h.id = ANY(%s) AND d.status = 'indexed'
            """,
            (radius, radius, ids),
        )
        return {row["id"]: self._passage(row) for row in rows}

    @staticmethod
    def _passage(row: dict) -> Passage:
        return Passage(
            id=row["id"], text=row["text"], parent_text=row["parent_text"],
            section=row["section"], page_start=row["page_start"],
            page_end=row["page_end"], filename=row["filename"], year=row["year"],
            document_id=row["document_id"],
            line_start=row.get("line_start"), line_end=row.get("line_end"),
            title=row.get("title") or "", author=row.get("author") or "",
        )

    def fetch(self, ids: list[int]) -> list[Passage]:
        if not ids:
            return []
        rows = db.query(
            f"""
            SELECT {CHUNK_COLUMNS}
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE c.id = ANY(%s)
            """,
            (ids,),
        )
        order = {identifier: position for position, identifier in enumerate(ids)}
        found = [self._passage(row) for row in rows]
        found.sort(key=lambda p: order.get(p.id, 1_000_000))
        return found


class Sources:
    """Stable citation numbers shared by every member of the team.

    Specialists retrieve independently and often land on the same page. Numbering
    centrally means [4] refers to one passage across the whole consultation, in
    the findings, in the draft, and in what the reader finally sees.
    """

    def __init__(self) -> None:
        self._numbers: dict[int, int] = {}
        self._passages: dict[int, Passage] = {}
        # Specialists register from parallel threads.
        self._lock = threading.Lock()

    def register(self, passages: list[Passage]) -> list[tuple[int, Passage]]:
        numbered = []
        with self._lock:
            for passage in passages:
                if passage.id not in self._numbers:
                    self._numbers[passage.id] = len(self._numbers) + 1
                    self._passages[passage.id] = passage
                numbered.append((self._numbers[passage.id], passage))
        return numbered

    def cited(self, text: str) -> list[dict]:
        """Only the passages the finished answer actually refers to."""
        used = {int(n) for n in CITATION.findall(text)}
        out = []
        for passage_id, number in sorted(self._numbers.items(), key=lambda kv: kv[1]):
            if number in used:
                out.append(self._passages[passage_id].to_dict(number))
        return out

    def referenced(self, *texts: str) -> list[tuple[int, Passage]]:
        """The numbered passages the given texts cite, in citation order."""
        wanted: set[int] = set()
        for text in texts:
            wanted.update(int(n) for n in CITATION.findall(text or ""))
        by_number = {number: pid for pid, number in self._numbers.items()}
        return [
            (number, self._passages[by_number[number]])
            for number in sorted(wanted)
            if number in by_number
        ]

    def sanitize(self, text: str) -> str:
        """Strip citation markers that point at nothing.

        A model under instruction to cite everything occasionally invents a
        number. Removing it is strictly better than letting the reader click a
        reference that does not exist.
        """
        valid = set(self._numbers.values())
        return CITATION.sub(
            lambda m: m.group(0) if int(m.group(1)) in valid else "", text
        )

    def all(self) -> list[dict]:
        return [
            self._passages[i].to_dict(n)
            for i, n in sorted(self._numbers.items(), key=lambda kv: kv[1])
        ]

    def __len__(self) -> int:
        return len(self._numbers)

    def passages(self) -> dict[int, Passage]:
        """Number -> passage, for checking an answer against what it cites."""
        return {n: self._passages[i] for i, n in self._numbers.items()}

    @staticmethod
    def format(numbered: list[tuple[int, Passage]], compact: bool = False) -> str:
        """Compact uses the passage itself; full includes the wider section."""
        def attr(value: str) -> str:
            return value.replace('"', "'")

        return "\n\n".join(
            f'<source id="{number}" ref="{attr(passage.citation)}"'
            + (f' section="{attr(passage.section)}"' if passage.section else "")
            + f">\n{passage.text if compact else (passage.parent_text or passage.text)}\n</source>"
            for number, passage in numbered
        )
