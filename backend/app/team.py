"""The tutor: plans, gathers evidence, writes, checks.

    plan      one call: a standalone question, its kind, sub-questions, a
              textbook-style search probe, rephrasings and index terms, and -
              for a case or an exam question - a differential before anything
              is looked up
    search    hybrid retrieval for every angle at once (no model calls)
    grade     one call scores every candidate 0-3 and says which sub-questions
              are still uncovered; one targeted second search runs only then
    pack      the best passages, widened with their neighbours, merged where
              they touch, and trimmed to a token budget
    write     one call writes the answer straight from the sources
    verify    a free numeric check plus one call against the sources; a
              correction is written only when something is actually wrong

Every model call sees the sources themselves, never another call's paraphrase
of them, so nothing drifts between reading and writing. The sources are sent as
one identical block to the writer, checker and corrector, which providers cache.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field

from .providers import ChatModel, count_tokens
from .retrieve import CITATION, Passage, Retriever, Sources

log = logging.getLogger(__name__)

MODES = ("lookup", "concept", "management", "comparison", "vignette", "mcq", "chat")

# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #

PLAN = """\
You plan how to answer a medical student's question from their own documents \
(the textbooks, notes and guidelines they uploaded). You do not answer it.

Return JSON:
{
  "standalone"  : the question rewritten to stand alone. Resolve "it", "this", \
"what about..." from the conversation. Keep every detail of a case or exam \
question, including all answer options, verbatim.
  "mode"        : lookup | concept | management | comparison | vignette | mcq | chat
  "subquestions": 1-4 standalone questions whose answers together answer it; \
exactly 1 for a lookup. For a vignette or mcq: what must be looked up to confirm \
the leading diagnosis and to decide between the options.
  "probe"       : the paragraph a textbook would contain in answer. Confident \
textbook prose, technical vocabulary, at most 4 sentences. A search probe only, \
never shown to anyone: sounding like the passage matters more than being right.
  "variants"    : up to {variants} rephrasings in different vocabulary: expand \
every abbreviation, use synonyms, generic and brand drug names.
  "terms"       : 4-8 short index-style terms (2-3 words each), abbreviations \
expanded.
  "case"        : vignette or mcq only, else omit: {"findings": salient positives \
and negatives, "differential": [{"diagnosis": "...", "for": "...", "against": \
"..."}] (3-5, most likely first), "leading": the most likely diagnosis or best \
option}
}

Modes:
  lookup      one fact: a dose, value, definition, criterion, classification
  concept     how or why: mechanism, physiology, pharmacology, pathology
  management  treatment, work-up or monitoring of a condition
  comparison  weighing two or more drugs, options or conditions
  vignette    a clinical case to reason through
  mcq         a multiple-choice question with answer options
  chat        greetings, thanks, or questions about you; needs no documents \
(return only "standalone" and "mode")

Work only from details the question contains; never invent case findings."""

GRADE = """\
You grade search results for a medical student's question. The numbered \
PASSAGES are snippets from the student's own documents.

Sub-questions to answer:
{subquestions}

Score each passage that helps answer any sub-question:
  3  directly answers part of it: the fact, dose, criterion or mechanism itself
  2  relevant supporting detail or necessary context
  1  on topic but little substance
Leave out passages that are irrelevant, and index pages, tables of contents, \
reference lists, or figure captions with no substance.

Then judge coverage: a sub-question is covered when passages you scored 2-3 \
answer it.

Return JSON:
{
  "scores" : {"<passage number>": <1-3>, ...},
  "missing": [{"subquestion": <number>, "search": "a search query in textbook \
vocabulary for exactly what is missing"}]
}
"missing" is empty when every sub-question is covered."""

WRITE = """\
You are an expert medical tutor answering a medical student from their own \
documents. The <source> blocks above are the only admissible evidence: your \
own knowledge may organise and explain, never add facts.

Rules:
  - Lead with the direct answer; its key point in **bold**.
  - Cite every clinical statement with its source id, like [2] or [2][5]. \
A sentence you cannot cite is a sentence you leave out.
  - Copy doses, values, thresholds and criteria exactly as the source gives \
them: same numbers, same units. Never round, convert or combine figures.
  - Where sources disagree, give both and attribute each.
  - If the documents do not cover part of the question, say so in one plain \
sentence ("Your documents don't cover ..."). Never fill a gap from memory.
  - Explain the why: link the facts into reasoning the student can reproduce \
in an exam or on the ward.
  - End with **High-yield:** one to three short cited takeaways worth \
remembering. Skip it for a one-line lookup.
  - Answer in the language the student wrote in.

{profile}{mode_guidance}

Markdown: ### headings only when the answer has distinct parts, **bold** for \
key terms, a table only for a genuine comparison, prose over bullets unless the \
content is a list. No preamble; never restate the question."""

MODE_GUIDANCE = {
    "lookup": (
        "Shape: a direct lookup. The fact in the first sentence, qualifiers "
        "right after. A few tight sentences, no headings."
    ),
    "concept": (
        "Shape: teach the mechanism as a chain of causes - what happens, why, "
        "and what that produces clinically. Make every causal link explicit."
    ),
    "management": (
        "Shape: the plan in the order a clinician acts - first-line with exact "
        "doses, then alternatives and when to switch, then monitoring and "
        "safety checks (contraindications, pregnancy, interactions) where the "
        "sources give them."
    ),
    "comparison": (
        "Shape: compare head to head - a markdown table for the facts side by "
        "side, then when to choose which."
    ),
    "vignette": (
        "Shape: walk the case as a clinician thinks - the key findings, the "
        "differential and how the findings sort it, the leading diagnosis and "
        "what confirms it, then management. Teach the reasoning at each step."
    ),
    "mcq": (
        "Shape: an exam question. First line: **Answer: <letter>. <option>**. "
        "Then why it is right, citing sources. Then one line per other option "
        "on why it is wrong (cite where the documents cover it; say so where "
        "they do not). Finish with the concept the question is testing."
    ),
}

VERIFY = """\
You check a tutor's answer against the <source> blocks above before a medical \
student relies on it. Judge only against the sources, never your own knowledge.

Report:
  - a claim whose cited source does not say it, or says something different
  - a number, dose, unit or threshold that differs from its source
  - a clinical claim with no citation that the sources do not support
Ignore style, structure and omissions. Do not report a claim that is supported.

Return JSON: {"issues": [{"quote": "the exact phrase at fault", "problem": \
"what is wrong", "fix": "the correction with the right citation, or 'remove'"}]}
An empty list when the answer is faithful."""

REPAIR = """\
You are correcting a tutor's answer so that it is faithful to the <source> \
blocks above. Fix every listed issue. Change only what the issues require and \
keep everything else exactly as written, citations included. Output only the \
full corrected answer."""

SUGGEST = """\
A medical student has just read the answer below. Suggest the follow-up \
questions a good tutor would nudge them toward next: the natural next step in \
understanding the topic, likely answerable from the same documents.

Return JSON: {"questions": ["...", "...", "..."]}
Exactly 3, each short, standalone and specific. No numbering."""

MEMORY = """\
You keep the running memory of a tutoring conversation with a medical student. \
Update the summary with the new turns below.

Keep: the topics and conditions discussed, the student's goals and level as \
they reveal them, key facts already established (with doses and figures as \
given), what they found confusing, and open threads they may return to. Drop \
pleasantries and anything superseded.

Write at most 180 words of plain notes. Output only the updated summary."""

CHAT = """\
You are a medical study assistant. You answer a medical student's questions \
from the documents they have uploaded, citing the page each fact comes from. \
Reply briefly and warmly to what they said; if it helps, suggest the kind of \
question you can answer. Answer in the language they wrote in."""


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


@dataclass
class Plan:
    question: str
    standalone: str
    mode: str = "concept"
    subquestions: list[str] = field(default_factory=list)
    probe: str = ""
    variants: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    case: dict = field(default_factory=dict)


@dataclass
class Graded:
    passage: Passage
    score: int        # 1-3 from the grader
    rank: float       # retrieval score, for ties


@dataclass
class Consultation:
    """What one question produced; `answer` is what gets stored."""
    plan: Plan | None = None
    candidates: int = 0
    kept: int = 0
    missing: list[str] = field(default_factory=list)
    retried: list[str] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)
    repaired: bool = False
    answer: str = ""
    suggestions: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)      # totals, for the message
    records: list[dict] = field(default_factory=list)  # per call, for analytics


LEVELS = {
    "preclinical": "a pre-clinical MBBS student (years 1-2): build from basic "
                   "science, define terms, favour mechanisms over management detail",
    "clinical": "a clinical MBBS student (years 3-5): connect mechanism to "
                "presentation and management, highlight what is examinable",
    "intern": "an intern or junior resident: practical, decision-focused, doses "
              "and monitoring first",
    "pg": "a postgraduate resident: specialist depth, nuance and controversies",
    "phd": "a PhD or research student: mechanistic depth, the strength and "
           "limits of the evidence as the sources state them, precise terminology",
}
LENGTHS = {
    "concise": "Keep it tight: the essentials only.",
    "detailed": "Go deep: cover every relevant point the sources give.",
}


def profile_text(settings) -> str:
    """The student's study profile, as a line or two for the writer."""
    parts = []
    level = LEVELS.get(getattr(settings, "learner_level", "") or "")
    if level:
        parts.append(f"The student is {level}.")
    exam = (getattr(settings, "exam_target", "") or "").strip()
    if exam:
        parts.append(f"They are preparing for {exam}: make High-yield what that exam rewards.")
    length = LENGTHS.get(getattr(settings, "answer_length", "") or "")
    if length:
        parts.append(length)
    return ("Student: " + " ".join(parts) + "\n\n") if parts else ""


class Meter:
    """Token use for one question, across every model and embedding call.

    Calls report from several threads (the checker and the suggester run side
    by side), so additions are locked.
    """

    def __init__(self) -> None:
        self.records: list[dict] = []
        self._lock = threading.Lock()

    def add(self, record: dict) -> None:
        with self._lock:
            self.records.append(dict(record))

    def summary(self) -> dict:
        with self._lock:
            records = list(self.records)
        total = {"input": 0, "output": 0, "cached": 0, "cache_write": 0, "calls": 0}
        steps: dict[str, dict] = {}
        for r in records:
            step = steps.setdefault(r["step"], {"input": 0, "output": 0, "cached": 0, "calls": 0})
            for key in ("input", "output", "cached", "calls"):
                step[key] += r.get(key, 0)
            for key in total:
                total[key] += r.get(key, 0)
        models = sorted({f"{r['provider']}:{r['model']}" for r in records if r.get("model")})
        return {**total, "steps": steps, "models": models}


def _strings(values, limit: int) -> list[str]:
    """A clean list of non-empty strings from whatever the model returned."""
    if not isinstance(values, list):
        return []
    return [str(v).strip() for v in values if str(v or "").strip()][:limit]


# --------------------------------------------------------------------------- #
# numbers: the cheapest accuracy check there is
# --------------------------------------------------------------------------- #

NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?![\w])|\d+(?:[.,]\d+)?(?=\s?(?:mg|g|mcg|µg|ug|ml|mL|%|kg|mmol|IU|units?)\b)")
SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def _norm_number(value: str) -> str:
    value = value.replace(",", ".")
    return value.rstrip("0").rstrip(".") if "." in value else value


def unsupported_figures(answer: str, cited: dict[int, Passage]) -> list[dict]:
    """Figures in a cited sentence that appear in none of the sources it cites.

    No model involved: a dose that is not in its source is wrong or invented,
    and catching it costs nothing. Small integers (1-9) are skipped - they are
    mostly counts and list numbers, and too common to tell anything.
    """
    flagged = []
    for sentence in SENTENCE.split(answer):
        numbers = [int(n) for n in CITATION.findall(sentence)]
        if not numbers:
            continue
        sources = " ".join(cited[n].text for n in numbers if n in cited)
        if not sources:
            continue
        available = {_norm_number(v) for v in NUMBER.findall(sources)}
        body = CITATION.sub(" ", sentence)
        missing = [
            v for v in dict.fromkeys(NUMBER.findall(body))
            if (len(v) > 1 or "." in v or "," in v) and _norm_number(v) not in available
        ]
        if missing:
            flagged.append({"sentence": sentence.strip()[:300], "figures": missing})
    return flagged


# --------------------------------------------------------------------------- #
# the tutor
# --------------------------------------------------------------------------- #


class Team:
    def __init__(self, settings, chat: ChatModel | None = None,
                 retriever: Retriever | None = None,
                 helper: ChatModel | None = None) -> None:
        self.settings = settings
        self.chat = chat or ChatModel(settings)
        # Planning, grading and checking need judgement, not prose: they run on
        # the helper model when one is configured.
        self.helper = helper or (ChatModel(settings, fast=True)
                                 if settings.fast_model else self.chat)
        self.retriever = retriever or Retriever(settings)
        self.sources = Sources()
        self.meter = Meter()
        for model in {id(self.chat): self.chat, id(self.helper): self.helper}.values():
            model.meter = self.meter.add
        embedder = getattr(self.retriever, "embedder", None)
        if embedder is not None:
            embedder.meter = self.meter.add

    # -- plan ---------------------------------------------------------------- #

    def plan(self, question: str, history: list[dict], summary: str = "") -> Plan:
        conversation = "\n".join(
            f"{m['role']}: {m['content'][:500]}" for m in history[-4:]
        )
        memory = f"Earlier in this conversation (summary):\n{summary}\n\n" if summary else ""
        data, ok = self.helper.structured(
            PLAN.replace("{variants}", str(self.settings.query_variants)),
            f"{memory}Recent messages:\n{conversation or '(none)'}\n\nQuestion: {question}",
            fallback={}, step="plan",
        )
        mode = str(data.get("mode") or "").strip().lower()
        standalone = str(data.get("standalone") or question).strip() or question
        plan = Plan(
            question=question,
            standalone=standalone,
            mode=mode if mode in MODES else "concept",
            subquestions=_strings(data.get("subquestions"), 4) or [standalone],
            probe=str(data.get("probe") or "").strip() if self.settings.use_hypothesis else "",
            variants=_strings(data.get("variants"), self.settings.query_variants),
            terms=_strings(data.get("terms"), 8),
        )
        if not ok:
            log.info("plan unparseable; searching the question as asked")
        case = data.get("case") if isinstance(data.get("case"), dict) else {}
        differential = [
            {
                "diagnosis": str(d.get("diagnosis") or "").strip(),
                "for": str(d.get("for") or "").strip(),
                "against": str(d.get("against") or "").strip(),
            }
            for d in (case.get("differential") or [])
            if isinstance(d, dict) and str(d.get("diagnosis") or "").strip()
        ][:5]
        if plan.mode in ("vignette", "mcq") and differential:
            plan.case = {
                "findings": str(case.get("findings") or "").strip(),
                "differential": differential,
                "leading": str(case.get("leading") or "").strip(),
            }
            # A case never names its diagnosis; the differential is what makes
            # the documents searchable at all.
            plan.terms += [d["diagnosis"] for d in differential]
        return plan

    @staticmethod
    def case_text(case: dict) -> str:
        if not case:
            return ""
        lines = []
        if case.get("findings"):
            lines.append(f"Key findings: {case['findings']}")
        for entry in case.get("differential", []):
            line = f"- {entry['diagnosis']}"
            if entry["for"]:
                line += f" - for: {entry['for']}"
            if entry["against"]:
                line += f"; against: {entry['against']}"
            lines.append(line)
        if case.get("leading"):
            lines.append(f"Leading: {case['leading']}")
        return "\n".join(lines)

    # -- search and grade ---------------------------------------------------- #

    def search(self, plan: Plan, queries: list[str] | None = None,
               exclude: set[int] | None = None) -> list[Passage]:
        if queries:  # a targeted second search for what the first missed
            return self.retriever.search(
                queries[0], variants=queries[1:], top_k=self.settings.shortlist // 2,
                exclude=exclude,
            )
        return self.retriever.search(
            plan.standalone,
            variants=plan.subquestions + plan.variants,
            hypothetical=plan.probe or None,
            terms=plan.terms,
            top_k=self.settings.shortlist,
            exclude=exclude,
        )

    def grade(self, plan: Plan, candidates: list[Passage]) -> tuple[list[Graded], list[dict]]:
        """Score candidates in one call and report uncovered sub-questions.

        One listwise call over compact snippets replaces a reranker model and
        the old per-specialist sifting calls, and it catches gaps before any
        writing is done rather than after.
        """
        if not candidates:
            return [], [{"subquestion": i + 1, "search": q}
                        for i, q in enumerate(plan.subquestions)]
        listing = "\n\n".join(
            f"[{i}] {p.citation}" + (f" - {p.section}" if p.section else "")
            + f"\n{p.text[:900]}"
            for i, p in enumerate(candidates, start=1)
        )
        subquestions = "\n".join(f"{i}. {q}" for i, q in enumerate(plan.subquestions, start=1))
        data, ok = self.helper.structured(
            GRADE.replace("{subquestions}", subquestions),
            f"Question: {plan.standalone}\n\nPASSAGES\n{listing}",
            fallback={}, step="grade",
        )
        if not ok:
            # No grades: trust retrieval order, keep the top of it.
            return [Graded(p, 2, p.score) for p in candidates[: self.settings.passages]], []

        graded = []
        scores = data.get("scores") if isinstance(data.get("scores"), dict) else {}
        for key, value in scores.items():
            try:
                index, score = int(str(key).strip("[] ")), int(value)
            except (TypeError, ValueError):
                continue
            if 1 <= index <= len(candidates) and score >= 1:
                passage = candidates[index - 1]
                graded.append(Graded(passage, min(score, 3), passage.score))
        missing = [
            m for m in (data.get("missing") or [])
            if isinstance(m, dict) and str(m.get("search") or "").strip()
        ][:3]
        return graded, missing

    # -- pack ---------------------------------------------------------------- #

    def pack(self, graded: list[Graded]) -> list[Passage]:
        """Choose, widen and merge evidence within the token budget.

        Strongest passages go in first. The best ones are then widened with the
        passages either side of them, which restores the sentence that
        qualifies a dose without sending the whole section. Passages that touch
        are merged into one source, so overlapping text is sent once.
        """
        budget = self.settings.evidence_tokens
        ranked = sorted(graded, key=lambda g: (g.score, g.rank), reverse=True)
        ranked = [g for g in ranked if g.score >= 2] or ranked[:3]
        ranked = ranked[: self.settings.passages]

        chosen: dict[int, Passage] = {}
        strength: dict[int, float] = {}
        used = 0
        for g in ranked:
            tokens = count_tokens(g.passage.text)
            if chosen and used + tokens > budget:
                continue
            chosen[g.passage.id] = g.passage
            strength[g.passage.id] = g.score + g.rank
            used += tokens

        # Widen the strongest hits while the budget allows.
        best = [g.passage.id for g in ranked if g.score == 3 and g.passage.id in chosen][:6]
        if best and used < budget:
            around = self.retriever.neighbours(best)
            for hit in best:
                for side in (hit - 1, hit + 1):
                    neighbour = around.get(side)
                    if (neighbour is None or side in chosen
                            or neighbour.document_id != chosen[hit].document_id):
                        continue
                    tokens = count_tokens(neighbour.text)
                    if used + tokens > budget:
                        continue
                    chosen[side] = neighbour
                    strength[side] = strength[hit] - 0.5
                    used += tokens

        spans = _merge_runs(chosen, strength)
        log.info("evidence: %d passages -> %d sources, ~%d tokens", len(chosen), len(spans), used)
        return spans

    # -- write, check, correct ------------------------------------------------ #

    def evidence_block(self, numbered: list[tuple[int, Passage]]) -> str:
        return ("SOURCES - excerpts from the student's own documents\n\n"
                + Sources.format(numbered, compact=True))

    def brief(self, plan: Plan, missing: list[str], summary: str = "") -> str:
        """The writer's user message: the question plus what it must know."""
        parts = []
        if summary:
            parts.append(f"Earlier in this conversation (summary, not evidence):\n{summary}")
        parts.append(f"Question: {plan.question}")
        if plan.standalone != plan.question:
            parts.append(f"(In full: {plan.standalone})")
        case = self.case_text(plan.case)
        if case:
            parts.append(
                "Bedside reading of the case (yours, not evidence - every "
                f"clinical claim still needs a source):\n{case}")
        if missing:
            parts.append("Searched for but not found in the documents:\n- "
                         + "\n- ".join(missing))
        return "\n\n".join(parts)

    def write_system(self, mode: str) -> str:
        return (WRITE.replace("{profile}", profile_text(self.settings))
                .replace("{mode_guidance}", MODE_GUIDANCE.get(mode, MODE_GUIDANCE["concept"])))

    def verify(self, answer: str, evidence: str) -> list[dict]:
        """Issues with the answer: figures first (free), then one model check."""
        figures = unsupported_figures(answer, self.sources.passages())
        hints = ""
        if figures:
            hints = "\n\nAn automatic check found figures absent from the sources " \
                    "their sentence cites (check each; some may be legitimate):\n" + \
                "\n".join(f"- {', '.join(f['figures'])} in: {f['sentence']}" for f in figures)
        data, ok = self.helper.structured(
            VERIFY, f"ANSWER TO CHECK\n{answer}{hints}", fallback={}, context=evidence,
            step="verify",
        )
        issues = [
            {k: str(i.get(k) or "").strip() for k in ("quote", "problem", "fix")}
            for i in (data.get("issues") or []) if isinstance(i, dict)
        ] if ok else []
        return [i for i in issues if i["quote"] or i["problem"]][:8]

    def repair(self, answer: str, issues: list[dict], evidence: str) -> str:
        listing = "\n".join(
            f"- \"{i['quote']}\": {i['problem']}" + (f" -> {i['fix']}" if i["fix"] else "")
            for i in issues
        )
        return self.chat.complete(
            REPAIR, f"ANSWER\n{answer}\n\nISSUES\n{listing}", context=evidence,
            step="repair",
        )

    def suggest(self, question: str, answer: str) -> list[str]:
        data, ok = self.helper.structured(
            SUGGEST, f"Question: {question}\n\nAnswer:\n{answer[:3000]}", fallback={},
            step="suggest",
        )
        return _strings(data.get("questions"), 3) if ok else []


def update_memory(chat: ChatModel, summary: str, turns: list[dict]) -> str:
    """Fold older turns into a chat's running summary (one cheap call)."""
    transcript = "\n\n".join(
        f"{t['role']}: {CITATION.sub('', t['content'])[:1800]}" for t in turns
    )
    return chat.complete(
        MEMORY,
        f"CURRENT SUMMARY\n{summary or '(none yet)'}\n\nNEW TURNS\n{transcript}",
        effort="low", step="memory",
    )


def _merge_runs(chosen: dict[int, Passage], strength: dict[int, float]) -> list[Passage]:
    """Merge consecutive passages of one document into single sources.

    Each stored passage repeats the last few words of the one before it (the
    overlap that keeps sentences whole at boundaries); that repeat is dropped
    when the two are joined.
    """
    by_document: dict[int, list[Passage]] = {}
    for passage in chosen.values():
        by_document.setdefault(passage.document_id, []).append(passage)

    spans: list[tuple[float, Passage]] = []
    for passages in by_document.values():
        passages.sort(key=lambda p: p.id)
        run: list[Passage] = []
        for passage in passages + [None]:
            if passage is not None and run and passage.id == run[-1].id + 1:
                run.append(passage)
                continue
            if run:
                spans.append((max(strength[p.id] for p in run), _join(run)))
            run = [passage] if passage is not None else []
    spans.sort(key=lambda s: s[0], reverse=True)
    return [span for _, span in spans]


def _join(run: list[Passage]) -> Passage:
    text = run[0].text
    for passage in run[1:]:
        piece = passage.text
        head, _, rest = piece.partition("\n\n")
        head_words = head.split()
        if rest and head_words and text.split()[-len(head_words):] == head_words:
            piece = rest
        text = f"{text}\n\n{piece}"
    first = run[0]
    return Passage(
        id=first.id, text=text, parent_text=text, section=first.section,
        page_start=min(p.page_start for p in run), page_end=max(p.page_end for p in run),
        filename=first.filename, year=first.year, score=first.score,
        signals=first.signals, document_id=first.document_id,
        line_start=first.line_start, line_end=run[-1].line_end,
        title=first.title, author=first.author,
    )
