"""Runs one question through the tutor and streams what happens.

    plan -> search -> grade -> (one corrective search if something is missing)
         -> pack evidence -> write (streamed) -> verify -> correct if needed

A typical question costs four to six model calls, and the evidence is sent as
one identical block that providers cache across the writer, checker and
corrector. Greetings and thanks skip the documents entirely.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

from .providers import ChatModel
from .retrieve import CITATION, Retriever
from .team import CHAT, Consultation, Team

log = logging.getLogger(__name__)

NOTHING_FOUND = (
    "I couldn't find this in your documents. Try rephrasing with the terms your "
    "textbook would use, or check in **Documents** that the relevant file has "
    "finished indexing."
)


def _subquestion(entry: dict) -> int:
    """The 1-based sub-question a grader's "missing" entry refers to, or 0."""
    try:
        return int(str(entry.get("subquestion") or 0).strip())
    except ValueError:
        return 0


def evidence_strength(graded, missing) -> str:
    strong = sum(1 for g in graded if g.score == 3)
    if not graded:
        return "none"
    if not missing and strong >= 1:
        return "strong"
    return "partial" if strong or len(graded) >= 2 else "limited"


class Pipeline:
    def __init__(self, settings, chat: ChatModel | None = None,
                 retriever: Retriever | None = None) -> None:
        self.settings = settings
        self.team = Team(settings, chat, retriever)

    def _usage(self) -> dict:
        """Token use so far, for the live counter in the interface."""
        return {"type": "usage", **self.team.meter.summary()}

    def _finish(self, result: Consultation) -> dict:
        result.usage = self.team.meter.summary()
        result.records = list(self.team.meter.records)
        return {"type": "complete", "result": result}

    def answer(self, question: str, history: list[dict] | None = None,
               summary: str = "") -> Iterator[dict]:
        """`history` is the last few messages; `summary` covers everything
        before them, so a long conversation costs no more than a short one."""
        # Earlier answers carry citation numbers from their own sources;
        # stripped, so the writer cannot reuse a [3] that now means another.
        history = [
            {"role": m["role"], "content": CITATION.sub("", m["content"])}
            for m in (history or [])
        ]
        team = self.team
        result = Consultation()

        # -- plan ------------------------------------------------------------ #
        yield {"type": "status", "text": "Understanding the question"}
        try:
            plan = team.plan(question, history, summary)
        except Exception as exc:
            log.exception("planning failed")
            yield {"type": "error", "text": str(exc)}
            return
        result.plan = plan
        yield {"type": "plan", "reading": plan.standalone, "mode": plan.mode,
               "subquestions": plan.subquestions}
        yield self._usage()

        if plan.mode == "chat":
            yield from self._chat(question, history, result)
            return
        if plan.case:
            yield {"type": "reasoning", **plan.case}

        # -- search and grade ---------------------------------------------- #
        yield {"type": "status", "text": "Searching your documents"}
        candidates = team.search(plan)
        result.candidates = len(candidates)
        yield {"type": "status", "text": f"Ranking {len(candidates)} passages"}
        graded, missing = team.grade(plan, candidates)
        yield self._usage()

        # One corrective search, only for what grading found missing. Gaps are
        # found before writing, which is far cheaper than drafting, reviewing
        # and searching again afterwards.
        if missing and self.settings.enable_review:
            queries = [str(m["search"]).strip() for m in missing]
            result.retried = queries
            yield {"type": "status", "text": "Looking again for: " + "; ".join(queries)}
            seen = {p.id for p in candidates}
            extra = team.search(plan, queries=queries, exclude=seen)
            result.candidates += len(extra)
            if extra:
                more, still = team.grade(plan, extra)
                graded += more
                still_open = {_subquestion(m) for m in still}
                if more:
                    missing = [m for m in missing if _subquestion(m) in still_open]

        result.missing = list(dict.fromkeys(
            plan.subquestions[_subquestion(m) - 1]
            if 1 <= _subquestion(m) <= len(plan.subquestions)
            else str(m.get("search"))
            for m in missing
        ))

        spans = team.pack(graded) if graded else []
        result.kept = len(spans)
        strength = evidence_strength(graded, missing)
        yield {"type": "evidence", "candidates": result.candidates, "sources": len(spans),
               "strength": strength, "missing": result.missing, "retried": result.retried}
        yield self._usage()

        if not spans:
            yield {"type": "citations", "citations": []}
            yield {"type": "token", "text": NOTHING_FOUND}
            result.answer = NOTHING_FOUND
            yield self._finish(result)
            return

        numbered = team.sources.register(spans)
        evidence = team.evidence_block(numbered)
        yield {"type": "citations", "citations": team.sources.all()}

        # -- write ----------------------------------------------------------- #
        yield {"type": "status", "text": "Writing the answer"}
        collected: list[str] = []
        try:
            for piece in team.chat.stream(
                team.write_system(plan.mode), team.brief(plan, result.missing, summary),
                history[-4:], context=evidence, step="write",
            ):
                collected.append(piece)
                yield {"type": "token", "text": piece}
        except Exception as exc:
            log.exception("writing failed")
            yield {"type": "error", "text": str(exc)}
            return

        raw = "".join(collected)
        answer = team.sources.sanitize(raw)
        yield self._usage()

        # -- verify (and suggest, in parallel) ------------------------------ #
        check = self.settings.enable_review
        if check:
            yield {"type": "status", "text": "Checking every claim against the sources"}
        with ThreadPoolExecutor(max_workers=2) as pool:
            verifying = pool.submit(team.verify, answer, evidence) if check else None
            suggesting = (pool.submit(team.suggest, question, answer)
                          if self.settings.enable_suggestions else None)
            issues = []
            if verifying:
                try:
                    issues = verifying.result()
                except Exception as exc:
                    log.warning("verification failed: %s", exc)
            result.issues = issues

            if issues:
                yield {"type": "status",
                       "text": f"Correcting {len(issues)} point{'s' if len(issues) > 1 else ''}"}
                try:
                    fixed = team.sources.sanitize(team.repair(answer, issues, evidence))
                    if fixed.strip():
                        answer, result.repaired = fixed, True
                except Exception as exc:
                    log.warning("correction failed: %s", exc)

            if suggesting:
                try:
                    result.suggestions = suggesting.result()
                except Exception as exc:
                    log.info("suggestions failed: %s", exc)

        result.answer = answer
        if answer != raw:
            # The stream carried the uncorrected text; hand the interface the
            # final version to re-render and store.
            yield {"type": "answer", "text": answer}
        if check:
            yield {"type": "verify", "issues": issues, "repaired": result.repaired}
        yield {"type": "citations", "citations": team.sources.cited(answer)}
        if result.suggestions:
            yield {"type": "suggestions", "questions": result.suggestions}
        yield self._usage()
        yield self._finish(result)

    def _chat(self, question: str, history: list[dict], result: Consultation) -> Iterator[dict]:
        """Greetings and thanks: a short reply, no documents, no citations."""
        yield {"type": "citations", "citations": []}
        collected: list[str] = []
        try:
            for piece in self.team.chat.stream(CHAT, question, history[-4:], step="chat"):
                collected.append(piece)
                yield {"type": "token", "text": piece}
        except Exception as exc:
            yield {"type": "error", "text": str(exc)}
            return
        result.answer = "".join(collected)
        yield self._usage()
        yield self._finish(result)
