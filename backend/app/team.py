"""A panel of specialists working a question the way a senior team would.

    triage        reads the question, classifies it, and briefs each specialist
    diagnostician for case vignettes: findings, differential, leading diagnosis
    specialists   investigate their own area in parallel, each retrieving separately
    registrar     drafts an answer from the collected findings
    consultant    reviews the draft against the cited passages and sends back gaps
    (second pass runs only for what the consultant asked for)
    attending     writes the final answer, teaching as a senior doctor would

The point is the review step. A single pass produces whatever the first search
returned; a panel that criticises its own draft and goes back for what is
missing produces something closer to a considered answer.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from .providers import ChatModel
from .retrieve import Passage, Retriever, Sources

log = logging.getLogger(__name__)

# Roles are framed by what they look for in the documents, not by seniority.
SPECIALISTS: dict[str, tuple[str, str]] = {
    "mechanism": (
        "Pathophysiology",
        "disease mechanism, immunology, histopathology, pathogenesis, natural history",
    ),
    "clinical": (
        "Clinical assessment",
        "presentation, morphology, distribution, variants, differential diagnosis, "
        "diagnostic criteria and investigations",
    ),
    "therapeutics": (
        "Therapeutics",
        "treatment options, drug selection, dosing and regimens, sequencing, "
        "expected response and duration",
    ),
    "safety": (
        "Safety and monitoring",
        "adverse effects, contraindications, drug interactions, monitoring "
        "requirements, use in pregnancy, children, renal or hepatic impairment",
    ),
    "evidence": (
        "Evidence and guidelines",
        "trial data, comparative effectiveness, guideline positions, strength and "
        "limitations of the evidence, areas of disagreement",
    ),
}

TRIAGE = """\
You are triaging a medical student's question for a specialist panel that \
answers strictly from a fixed library of medical textbooks.

Available specialists:
{roster}

First classify the question, then decide who is needed. Straightforward factual \
lookups need one specialist; managing a condition usually needs several.

Modes:
  lookup      a single fact: a dose, a value, a definition, a classification
  concept     how or why something happens: mechanism, physiology, pharmacology
  management  how to treat, monitor or work up a condition
  comparison  weighing two or more options against each other
  vignette    a clinical case description to reason through

Return JSON:
{
  "reading"      : one sentence on what is actually being asked
  "mode"         : one of lookup | concept | management | comparison | vignette
  "depth"        : "single" for a direct lookup, "panel" for anything needing \
several perspectives
  "assignments"  : [{"specialist": "<key>", "brief": "the specific question this \
specialist should answer, written as a full standalone question"}]
  "search_terms" : 5-10 terms a textbook index would use for this topic, with \
every abbreviation expanded
}

Each brief must stand entirely on its own: name the condition, the drug, the \
patient group explicitly. The specialist sees nothing else, so resolve every \
"it", "this" and "what about..." from the conversation into the thing it refers \
to.

Between one and four assignments. Use the keys exactly as given."""

SEARCH_PLAN = """\
You are planning how to search a library of medical textbooks for the question \
below.

Return JSON:
{
  "probe"    : the paragraph you would expect the textbook itself to contain in \
answer. Confident textbook prose, technical vocabulary, four sentences at most. \
This is a search probe, never shown to anyone; sounding like the passage being \
looked for matters more than being right.
  "variants" : up to {variants} rephrasings of the question as different authors \
would write it: expand every abbreviation, swap in synonyms, use both generic \
and brand drug names where they exist.
  "terms"    : 3-6 short terms or phrases a textbook index would list for this \
topic.
}"""

VIGNETTE = """\
You are the diagnostician on a panel of doctors. A medical student has presented \
a clinical case. Reason through it the way you would at the bedside, before \
anything is looked up.

Return JSON:
{
  "findings"     : one or two sentences naming the salient positives and negatives \
in the case
  "differential" : 3-5 entries, most likely first: [{"diagnosis": "...", \
"for": "what in this case supports it", "against": "what argues against it"}]
  "leading"      : the single most likely diagnosis
  "lookups"      : 2-4 short search terms naming the candidate diagnoses and key \
findings, for searching the library
}

Work from the case's own details. Never invent a finding the stem does not \
contain."""

SIFT = """\
You are selecting which retrieved passages are worth reading closely.

Given a question and numbered passages, return the numbers that genuinely help \
answer it, most useful first. Discard passages that only mention the topic in \
passing, reference lists, and figure captions with no substance.

Return JSON: {"keep": [numbers]}. Keep at most {keep}. Keep fewer if fewer are \
useful; an empty list is a valid answer."""

# The consultant's follow-up questions can fall in any area, so they are
# worked by a generalist rather than forced through one specialist's lens.
FOLLOW_UP = "follow_up"
FOLLOW_UP_ROLE = ("Follow-up", "whatever the question asks, in any area")

SCOPE_SPECIALIST = """\
Report what the passages say **on your area only**. Another specialist covers \
the rest; do not stray into theirs."""

SCOPE_FOLLOW_UP = """\
This question is being asked because the panel's first pass left it open. \
Report everything the passages say that bears on it, whatever the area."""

INVESTIGATE = """\
You are the {title} member of a panel answering from a fixed medical library.
Your area: {focus}

{scope}

Rules:
  - Every statement carries its passage number, like [3].
  - Quote figures, doses and thresholds exactly. Never round or convert.
  - Say plainly when the passages do not cover part of your brief. A gap stated \
is more useful than a gap filled from memory.
  - No preamble. Findings only, as short paragraphs or bullets.

If the passages contain nothing relevant to your area, reply with exactly: \
NOTHING RELEVANT"""

DRAFT = """\
You are the registrar drafting the panel's answer from the specialists' findings.

Write the answer to the question asked. Not a summary of the findings, and not a \
section per specialist: one coherent answer that happens to draw on all of them.

Rules:
  - Keep every citation number from the findings, like [3]. Never invent one.
  - Lead with the direct answer, then the detail that supports it.
  - Quote doses, figures and thresholds exactly as the findings give them.
  - Where the findings disagree, say so and attribute each position.
  - Where the findings record a gap, say what is not covered.
  - A CLINICAL REASONING section, when present, is the panel's own bedside \
reading of the case, not library evidence: use it to structure the answer, but \
every clinical claim still needs a passage citation.
  - No preamble, no restating the question."""

REVIEW = """\
You are the senior consultant reviewing this draft before it reaches the \
student. The numbered PASSAGES below are the only admissible evidence. You are \
looking for what is wrong with the draft, not what is good.

Check, in order:
  1. Does every claim trace to a cited passage - and does that passage actually \
say it? Compare against the PASSAGES provided, not your own knowledge.
  2. Are figures, doses and thresholds reproduced exactly as the passages give \
them?
  3. Does it answer the question that was asked, or something adjacent?
  4. What important aspect is missing that the library probably covers?

Return JSON:
{
  "sound"      : true if this can go out as it stands
  "problems"   : specific faults, each quoting the phrase at fault
  "follow_up"  : up to 3 standalone questions worth searching the library for to \
fix what is missing. Empty when nothing is missing.
}

Be exacting but not obstructive. A complete, well-supported answer is sound even \
if more could theoretically be said."""

FINAL = """\
You are an attending physician sitting beside a medical student, answering from \
the numbered passages of a textbook library. Everything you teach must come from \
those passages; your own knowledge may organise the material, never add to it.

Voice: a sharp, warm attending. Direct and concrete, never padded. Answer first, \
reasons second, so the student always knows where they stand.

Rules:
  - Open with the direct answer to the question, its key point in **bold**.
  - Every clinical statement carries its passage number, like [3]. A sentence \
you cannot cite is a sentence you delete.
  - Quote doses, figures and thresholds exactly as the PASSAGES text gives them, \
not as the findings paraphrase them.
  - Address the consultant's problems by fixing them. Remove anything \
unsupported rather than softening it.
  - Where sources disagree, present both and attribute them.
  - Close with **Pearls**: one to three short takeaways worth carrying out of \
the room, each cited. If the library left real gaps, add one plain sentence \
naming them. Skip either when there is nothing worth saying.

{mode_guidance}

Markdown: short ### headings where they help, **bold** for the terms that \
matter, a table only for a genuine comparison, prose over bullets unless the \
content really is a list. No preamble; never restate the question."""

# How the answer should be shaped, by the kind of question asked.
MODE_GUIDANCE = {
    "lookup": (
        "Shape: this is a direct lookup. Give the fact in the first sentence "
        "and the qualifying details right after. A few tight sentences, no "
        "headings."
    ),
    "concept": (
        "Shape: teach the mechanism as a chain of causes - what happens, why, "
        "and what that produces clinically. Make each causal link explicit; "
        "that chain is what the student is trying to build."
    ),
    "management": (
        "Shape: lay out the plan in the order a clinician acts - first-line "
        "with exact doses, then alternatives and when to switch, then "
        "monitoring and safety checks."
    ),
    "comparison": (
        "Shape: compare directly rather than describing each option in turn - "
        "a markdown table for the head-to-head facts, then a plain "
        "recommendation of when to choose which."
    ),
    "vignette": (
        "Shape: walk the case the way a clinician thinks - key findings, the "
        "differential and how the findings sort it, the leading diagnosis, "
        "what confirms it, then management. Teach the reasoning at each step, "
        "citing as you go."
    ),
}

SUGGEST = """\
A medical student has just read the answer below. Suggest the follow-up \
questions a good teacher would nudge them toward next: the natural next step in \
understanding this topic, answerable from the same textbook library.

Return JSON: {"questions": ["...", "...", "..."]}

Exactly 3. Each short, standalone and specific - it will be searched as \
written. No numbering inside the strings."""


@dataclass
class Finding:
    specialist: str
    title: str
    brief: str
    text: str
    passages: int = 0


@dataclass
class Consultation:
    reading: str = ""
    mode: str = "concept"
    depth: str = "panel"
    assignments: list[dict] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    reasoning: dict = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    draft: str = ""
    problems: list[str] = field(default_factory=list)
    follow_up: list[str] = field(default_factory=list)
    revisited: bool = False
    answer: str = ""
    suggestions: list[str] = field(default_factory=list)


def roster() -> str:
    return "\n".join(f"  {key}: {title} - {focus}" for key, (title, focus) in SPECIALISTS.items())


def _strings(values, limit: int) -> list[str]:
    """A clean list of non-empty strings from whatever the model returned."""
    if not isinstance(values, list):
        return []
    return [str(v).strip() for v in values if str(v or "").strip()][:limit]


class Team:
    def __init__(self, settings, chat: ChatModel | None = None,
                 retriever: Retriever | None = None) -> None:
        self.settings = settings
        self.chat = chat or ChatModel(settings)
        self.retriever = retriever or Retriever(settings)
        self.sources = Sources()

    # -- triage ------------------------------------------------------------- #

    def triage(self, question: str, history: list[dict]) -> Consultation:
        conversation = "\n".join(
            f"{m['role']}: {m['content'][:400]}" for m in history[-4:]
        )
        data, ok = self.chat.structured(
            TRIAGE.replace("{roster}", roster()),
            f"Conversation so far:\n{conversation or '(none)'}\n\nQuestion: {question}",
            fallback={"depth": "single", "assignments": [
                {"specialist": "clinical", "brief": question}]},
        )
        assignments = [
            {
                "specialist": a.get("specialist"),
                "brief": str(a.get("brief") or question).strip(),
            }
            for a in (data.get("assignments") or [])
            if a.get("specialist") in SPECIALISTS
        ][: self.settings.max_specialists]

        if not assignments:
            assignments = [{"specialist": "clinical", "brief": question}]

        mode = str(data.get("mode") or "").strip().lower()
        return Consultation(
            reading=str(data.get("reading") or question),
            mode=mode if mode in MODE_GUIDANCE else "concept",
            depth=str(data.get("depth") or ("single" if not ok else "panel")),
            assignments=assignments,
            terms=_strings(data.get("search_terms"), 10),
        )

    # -- the diagnostician, for case vignettes ------------------------------ #

    def analyze_vignette(self, question: str) -> dict:
        """Bedside reasoning before anything is looked up.

        The pay-off is retrieval as much as pedagogy: a vignette never names
        its diagnosis, so until someone commits to a differential there is
        nothing worth searching the library for.
        """
        data, ok = self.chat.structured(VIGNETTE, f"The case:\n{question}", fallback={})
        differential = [
            {
                "diagnosis": str(d.get("diagnosis") or "").strip(),
                "for": str(d.get("for") or "").strip(),
                "against": str(d.get("against") or "").strip(),
            }
            for d in (data.get("differential") or [])
            if isinstance(d, dict) and str(d.get("diagnosis") or "").strip()
        ][:5]
        if not ok or not differential:
            return {}
        return {
            "findings": str(data.get("findings") or "").strip(),
            "differential": differential,
            "leading": str(data.get("leading") or "").strip(),
            "lookups": _strings(data.get("lookups"), 4),
        }

    @staticmethod
    def reasoning_text(reasoning: dict) -> str:
        if not reasoning:
            return ""
        lines = []
        if reasoning.get("findings"):
            lines.append(f"Key findings: {reasoning['findings']}")
        for entry in reasoning.get("differential", []):
            line = f"- {entry['diagnosis']}"
            if entry["for"]:
                line += f" - for: {entry['for']}"
            if entry["against"]:
                line += f"; against: {entry['against']}"
            lines.append(line)
        if reasoning.get("leading"):
            lines.append(f"Leading diagnosis: {reasoning['leading']}")
        return "\n".join(lines)

    # -- one specialist ----------------------------------------------------- #

    def investigate(self, specialist: str, brief: str,
                    terms: list[str] | None = None) -> Finding:
        if specialist == FOLLOW_UP:
            title, focus = FOLLOW_UP_ROLE
            scope = SCOPE_FOLLOW_UP
        else:
            title, focus = SPECIALISTS[specialist]
            scope = SCOPE_SPECIALIST

        # One structured call plans the whole search: a hypothetical passage to
        # embed (HyDE), rephrasings that expand abbreviations and swap synonyms,
        # and index-style terms for keyword search. Same latency as the old
        # hypothesis-only call, three more angles on the library.
        probe, variants, planned = "", [], []
        if self.settings.use_hypothesis:
            try:
                plan, ok = self.chat.structured(
                    SEARCH_PLAN.replace("{variants}", str(self.settings.query_variants)),
                    brief, fallback={},
                )
                if ok:
                    probe = str(plan.get("probe") or "").strip()
                    variants = _strings(plan.get("variants"), self.settings.query_variants)
                    planned = _strings(plan.get("terms"), 6)
            except Exception as exc:
                log.info("search planning failed for %s: %s", specialist, exc)

        candidates = self.retriever.search(
            brief,
            variants=variants,
            hypothetical=probe or None,
            terms=(terms or []) + planned,
            top_k=self.settings.shortlist,
        )
        if not candidates:
            return Finding(specialist, title, brief, "NOTHING RELEVANT", 0)

        chosen = self._sift(brief, candidates)
        if not chosen:
            return Finding(specialist, title, brief, "NOTHING RELEVANT", 0)

        numbered = self.sources.register(chosen)
        report = self.chat.complete(
            INVESTIGATE.format(title=title, focus=focus, scope=scope),
            f"Question for you: {brief}\n\nPASSAGES\n{Sources.format(numbered)}",
        )
        return Finding(specialist, title, brief, report, len(numbered))

    def _sift(self, brief: str, candidates: list[Passage]) -> list[Passage]:
        """Have the model discard the near-misses that retrieval always returns.

        Replaces a cross-encoder reranker, which would drag PyTorch into the
        image for a job one cheap call does adequately.
        """
        if len(candidates) <= self.settings.passages:
            return candidates
        listing = "\n\n".join(
            f"[{i}] {p.citation}\n{p.text[:600]}" for i, p in enumerate(candidates, start=1)
        )
        data, ok = self.chat.structured(
            SIFT.replace("{keep}", str(self.settings.passages)),
            f"Question: {brief}\n\nPASSAGES\n{listing}",
            fallback={},
        )
        if not ok:
            return candidates[: self.settings.passages]
        keep = [
            candidates[int(n) - 1]
            for n in (data.get("keep") or [])
            if str(n).isdigit() and 1 <= int(n) <= len(candidates)
        ]
        return keep[: self.settings.passages]

    # -- review ------------------------------------------------------------- #

    def review(self, question: str, draft: str) -> tuple[bool, list[str], list[str]]:
        # The reviewer gets the passages the draft cites; without them, "does
        # every claim trace to its passage" would be a question it answers from
        # its own general knowledge, which is exactly what it must not do.
        evidence = self.evidence(draft)
        data, ok = self.chat.structured(
            REVIEW,
            f"Question asked: {question}\n\nPASSAGES\n{evidence or '(none cited)'}"
            f"\n\nDraft answer:\n{draft}",
            fallback={"sound": True},
        )
        if not ok:
            return True, [], []
        problems = [str(p) for p in (data.get("problems") or [])][:5]
        follow_up = _strings(data.get("follow_up"), 3)
        return bool(data.get("sound", True)), problems, follow_up

    def evidence(self, *texts: str) -> str:
        """The passages the given texts cite, formatted for exact quoting."""
        referenced = self.sources.referenced(*texts)[: self.settings.evidence_limit]
        return Sources.format(referenced, compact=True)

    # -- after the answer ---------------------------------------------------- #

    def suggest(self, question: str, answer: str) -> list[str]:
        """Three follow-up questions a good teacher would prompt next."""
        data, ok = self.chat.structured(
            SUGGEST,
            f"Question: {question}\n\nAnswer:\n{answer[:4000]}",
            fallback={},
        )
        if not ok:
            return []
        return _strings(data.get("questions"), 3)

    def follow_up_search(self, questions: list[str]) -> list[Finding]:
        findings = []
        with ThreadPoolExecutor(max_workers=min(3, len(questions))) as pool:
            futures = {
                pool.submit(self.investigate, FOLLOW_UP, question): question
                for question in questions
            }
            for future in as_completed(futures):
                try:
                    finding = future.result()
                except Exception as exc:
                    log.warning("follow-up failed: %s", exc)
                    continue
                if finding.text.strip() != "NOTHING RELEVANT":
                    findings.append(finding)
        return findings

    # -- assembly ----------------------------------------------------------- #

    @staticmethod
    def collate(findings: list[Finding]) -> str:
        blocks = [
            f"### {f.title}\nBrief: {f.brief}\n\n{f.text}"
            for f in findings
            if f.text.strip() and f.text.strip() != "NOTHING RELEVANT"
        ]
        return "\n\n".join(blocks)
