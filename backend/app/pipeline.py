"""Runs a question past the specialist panel and streams what happens.

Simple lookups take the short path: one specialist, straight to the answer.
Anything needing more than one perspective goes through the full consultation -
parallel investigation, a draft, a consultant's review, and a second search for
whatever the review found missing. Case vignettes get a bedside-reasoning step
first, so the library is searched for the differential rather than the stem.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator

from .providers import ChatModel
from .retrieve import CITATION, Retriever
from .team import DRAFT, FINAL, MODE_GUIDANCE, SPECIALISTS, Consultation, Team

log = logging.getLogger(__name__)

NOTHING_FOUND = (
    "Nothing in the indexed documents covers this. Try rephrasing, or check "
    "that the relevant document has finished indexing."
)


class Pipeline:
    def __init__(self, settings, chat: ChatModel | None = None,
                 retriever: Retriever | None = None) -> None:
        self.settings = settings
        self.team = Team(settings, chat, retriever)

    def answer(self, question: str, history: list[dict] | None = None) -> Iterator[dict]:
        # Earlier answers carry citation numbers from their own consultations.
        # Stripped, so the writer cannot reuse a [3] that now means a different
        # passage.
        history = [
            {"role": m["role"], "content": CITATION.sub("", m["content"])}
            for m in (history or [])
        ]
        team = self.team
        result = Consultation()

        yield {"type": "status", "text": "Reading the question"}
        try:
            result = team.triage(question, history)
        except Exception as exc:
            log.exception("triage failed")
            yield {"type": "error", "text": str(exc)}
            return

        members = [
            {"key": a["specialist"], "title": SPECIALISTS[a["specialist"]][0],
             "brief": a["brief"]}
            for a in result.assignments
        ]
        yield {"type": "team", "reading": result.reading, "mode": result.mode,
               "members": members}

        # -- bedside reasoning first, for case vignettes --------------------- #
        if result.mode == "vignette":
            yield {"type": "status", "text": "Working through the case"}
            try:
                result.reasoning = team.analyze_vignette(question)
            except Exception as exc:
                log.warning("vignette analysis failed: %s", exc)
            if result.reasoning:
                yield {"type": "reasoning", **result.reasoning}

        # Topic terms from triage, plus the differential's diagnosis names for a
        # vignette - the stem never names the diagnosis, so those terms are what
        # make the library searchable at all.
        terms = result.terms + result.reasoning.get("lookups", [])

        # -- specialists work in parallel; report each as it lands ----------- #
        pending = len(result.assignments)
        yield {"type": "status",
               "text": f"Consulting {pending} specialist{'s' if pending > 1 else ''}"}

        with ThreadPoolExecutor(max_workers=min(4, pending)) as pool:
            futures = {
                pool.submit(team.investigate, a["specialist"], a["brief"], terms): a
                for a in result.assignments
            }
            for future in as_completed(futures):
                assignment = futures[future]
                try:
                    finding = future.result()
                except Exception as exc:
                    log.exception("specialist %s failed", assignment["specialist"])
                    yield {"type": "status",
                           "text": f"{assignment['specialist']} could not report: {exc}"}
                    continue
                result.findings.append(finding)
                found = finding.text.strip() != "NOTHING RELEVANT"
                yield {
                    "type": "finding",
                    "specialist": finding.specialist,
                    "title": finding.title,
                    "passages": finding.passages,
                    "found": found,
                }

        collated = team.collate(result.findings)
        if not collated:
            searched = "; ".join(dict.fromkeys(a["brief"] for a in result.assignments))
            text = NOTHING_FOUND + (
                f"\n\nThe panel searched for: {searched}" if searched else ""
            )
            yield {"type": "citations", "citations": []}
            yield {"type": "token", "text": text}
            result.answer = text
            yield {"type": "complete", "result": result}
            return

        # The panel's own reading of a case is context for the writers, but it
        # is not evidence: claims still need passage citations.
        context = f"Question: {question}\n\n"
        reasoning = team.reasoning_text(result.reasoning)
        if reasoning:
            context += (
                "CLINICAL REASONING (the panel's own reading of the case, not "
                f"library evidence)\n{reasoning}\n\n"
            )

        # -- draft and review, for anything beyond a direct lookup ----------- #
        single = result.depth == "single" or len(result.findings) == 1
        if not single and self.settings.enable_review:
            yield {"type": "status", "text": "Drafting the answer"}
            try:
                result.draft = team.chat.complete(DRAFT, f"{context}FINDINGS\n{collated}")
            except Exception as exc:
                log.warning("draft failed: %s", exc)

            if result.draft:
                yield {"type": "status", "text": "Senior review"}
                # A failed review must not discard the panel's work: the
                # findings and draft still make an answer.
                try:
                    sound, problems, follow_up = team.review(question, result.draft)
                except Exception as exc:
                    log.warning("review failed: %s", exc)
                    sound, problems, follow_up = True, [], []
                result.problems, result.follow_up = problems, follow_up
                yield {"type": "review", "sound": sound, "problems": problems,
                       "follow_up": follow_up}

                if follow_up:
                    result.revisited = True
                    yield {"type": "status",
                           "text": f"Going back for {len(follow_up)} open point"
                                   f"{'s' if len(follow_up) > 1 else ''}"}
                    extra = team.follow_up_search(follow_up)
                    result.findings.extend(extra)
                    collated = team.collate(result.findings)
                    yield {"type": "status",
                           "text": f"{len(extra)} further finding"
                                   f"{'s' if len(extra) != 1 else ''}"}

        # -- final answer ---------------------------------------------------- #
        yield {"type": "citations", "citations": team.sources.all()}
        yield {"type": "status", "text": "Writing the answer"}

        system = FINAL.replace(
            "{mode_guidance}", MODE_GUIDANCE.get(result.mode, MODE_GUIDANCE["concept"])
        )
        prompt = f"{context}FINDINGS\n{collated}"
        # The writer quotes doses from the passages themselves, not from the
        # findings' paraphrase of them.
        evidence = team.evidence(collated, result.draft)
        if evidence:
            prompt += f"\n\nPASSAGES (quote figures and doses from these)\n{evidence}"
        if result.draft:
            prompt += f"\n\nDRAFT\n{result.draft}"
        if result.problems:
            prompt += "\n\nCONSULTANT'S PROBLEMS\n- " + "\n- ".join(result.problems)

        collected: list[str] = []
        try:
            for piece in team.chat.stream(system, prompt, history[-4:]):
                collected.append(piece)
                yield {"type": "token", "text": piece}
        except Exception as exc:
            log.exception("final answer failed")
            yield {"type": "error", "text": str(exc)}
            return

        raw = "".join(collected)
        result.answer = team.sources.sanitize(raw)
        if result.answer != raw:
            # The stream already carried the invented markers; hand the
            # interface the corrected text to re-render and store.
            yield {"type": "answer", "text": result.answer}

        # Narrow the citation list to what the answer actually refers to.
        yield {"type": "citations", "citations": team.sources.cited(result.answer)}

        if self.settings.enable_suggestions:
            try:
                result.suggestions = team.suggest(question, result.answer)
            except Exception as exc:
                log.info("suggestions failed: %s", exc)
            if result.suggestions:
                yield {"type": "suggestions", "questions": result.suggestions}

        yield {"type": "complete", "result": result}
