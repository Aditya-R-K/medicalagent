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
from .providers import EmbeddingModel

log = logging.getLogger(__name__)

RRF_CONSTANT = 60
MAX_TERMS = 12

CITATION = re.compile(r"\[(\d+)\]")


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
    def citation(self) -> str:
        year = f", {self.year}" if self.year else ""
        return f"{self.filename}{year}, {self.pages}"

    def to_dict(self, index: int) -> dict:
        return {
            "index": index,
            "source": self.filename,
            "year": self.year,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section": self.section,
            "citation": self.citation,
            "score": round(self.score, 4),
            "signals": self.signals,
            "excerpt": self.text[:600],
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
            WHERE d.status = 'indexed'
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            (vector_literal(vector), limit),
        )
        return [row["id"] for row in rows]

    def _by_keyword(self, query: str, limit: int) -> list[int]:
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

    def search(
        self,
        question: str,
        variants: list[str] | None = None,
        hypothetical: str | None = None,
        terms: list[str] | None = None,
        top_k: int | None = None,
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
        vectors = self.embedder.embed(texts, kind="query")

        for query, vector in zip(queries, vectors):
            absorb("meaning", self._nearest(vector, limit))
            absorb("keyword", self._by_keyword(query, limit))
        if hypothetical:
            absorb("hypothesis", self._nearest(vectors[-1], limit), weight=1.2)

        cleaned = dict.fromkeys(t.strip() for t in (terms or []) if t and t.strip())
        for term in list(cleaned)[:MAX_TERMS]:
            absorb("term", self._by_keyword(term, limit), weight=0.9)

        if not fused:
            return []

        best = sorted(fused, key=fused.get, reverse=True)[:top_k]
        rows = db.query(
            """
            SELECT c.id, c.text, c.parent_text, c.section, c.page_start, c.page_end,
                   d.filename, d.year
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id = ANY(%s)
            """,
            (best,),
        )

        top = max(fused.values()) or 1.0
        passages = [
            Passage(
                id=row["id"],
                text=row["text"],
                parent_text=row["parent_text"],
                section=row["section"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                filename=row["filename"],
                year=row["year"],
                score=fused[row["id"]] / top,
                signals=sorted(signals[row["id"]]),
            )
            for row in rows
        ]
        passages.sort(key=lambda p: p.score, reverse=True)
        return passages

    def fetch(self, ids: list[int]) -> list[Passage]:
        if not ids:
            return []
        rows = db.query(
            """
            SELECT c.id, c.text, c.parent_text, c.section, c.page_start, c.page_end,
                   d.filename, d.year
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE c.id = ANY(%s)
            """,
            (ids,),
        )
        order = {identifier: position for position, identifier in enumerate(ids)}
        found = [
            Passage(
                id=row["id"], text=row["text"], parent_text=row["parent_text"],
                section=row["section"], page_start=row["page_start"],
                page_end=row["page_end"], filename=row["filename"], year=row["year"],
            )
            for row in rows
        ]
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

    @staticmethod
    def format(numbered: list[tuple[int, Passage]], compact: bool = False) -> str:
        """Compact uses the passage itself; full includes the wider section."""
        return "\n\n---\n\n".join(
            f"[{number}] {passage.citation}"
            + (f" - {passage.section}" if passage.section else "")
            + f"\n{passage.text if compact else (passage.parent_text or passage.text)}"
            for number, passage in numbered
        )
