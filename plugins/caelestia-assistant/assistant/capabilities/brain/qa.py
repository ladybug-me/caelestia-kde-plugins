"""brain.qa — extractive QA over the repo docs and user notes (D17,
Group D).

A no-LLM core cannot write an answer, and should not pretend to. What
it CAN do honestly is FIND the passage that answers the question and
hand it back verbatim, with the provenance to check it against. That
is this module: the pre-LLM discipline of extractive QA.

Three stages, each honest about what it contributes:

  1. ANSWER-TYPE DETECTION — "why" wants a cause, "how" wants a
     procedure, "when" wants a time, "how many" wants a number, "does"
     wants a yes/no. The type decides which sentences COUNT as answers
     (type-specific cue bonuses) and how many are assembled (procedures
     read in document order; single facts take the single best).
  2. PASSAGE RETRIEVAL — plain BM25 (k1=1.5, b=0.75) over paragraph /
     table-row / list-item units, NOT whole documents: a symptom|cause|fix
     table row is the natural answer carrier in this corpus, and ranking
     whole pages would bury it. Tokenization is the retrieval layer's own
     `indexer.tokenize` so QA and `caelestia-assist search` agree on what
     a word is.
  3. SENTENCE SELECTION — inside the top passages, sentences are ranked
     by idf-weighted query overlap plus the type's cue bonuses; the
     extracted span is returned VERBATIM with doc id, section and line
     number. If the evidence is thin the verdict says THIN; if there is
     no evidence at all it says NOT_FOUND — an extractive QA that
     fabricates would just be a search box that lies.

Temporal questions ("when did the layout reset logic change?", "what
did I note last night?") reuse the EXISTING time-expression parser
(cortex.nlhistory) to resolve the window; the span is echoed in the
answer so the reader can see exactly what was understood.

Sources are two, kept strictly apart and nameable:
  - the committed repo-docs corpus (assistant/retrieval/corpus/*.md —
    the same text `search` ranks);
  - caller-supplied user notes ({"name", "text"} records; the CLI reads
    note files, the bridge takes their text — the pure function never
    opens anything itself).

Deterministic end to end: sorted files, stable tie-breaks, a fixed
anchor clock for relative windows. Same question, same answer.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from assistant.capabilities.retrieval.indexer import tokenize

__all__ = ["answer", "render_lines", "detect_type", "corpus_units"]

CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"

_K1 = 1.5
_B = 0.75
_TOP_PASSAGES = 8      # passages to sentence-select from
_MAX_SENTENCES = 3     # longest assembled answer
# verdict CAPS, pinned on the committed corpus (see test_qa.py): below
# LOW there is no real evidence; HIGH with a real margin over the
# runner-up is what "the corpus answers this" honestly means. They are
# CAPS, not floors: idf mass scales with log(N docs), so a two-line
# user note must not be held to the 1200-unit corpus's absolute bar —
# the effective floors are min(cap, share of the query's available
# idf mass), computed per question.
_SCORE_LOW = 2.0
_SCORE_HIGH = 3.5
_LOW_SHARE = 0.5
_HIGH_SHARE = 0.55
_MARGIN = 0.10
_CUE_BONUS = 0.6       # per type-specific cue matched in a sentence
_TIME_BONUS = 0.4      # sentence carries an absolute date/clock and type==time
_UNIT_INHERIT = 0.05   # a sentence inherits a share of its passage's score

# the question's own VERBS are never subject terms: 'what causes X' asks
# about X, and a gate or coverage test that demands the word 'causes' in
# the answer span downgrades every correct reply (the question-verb set
# is used ONLY for gate/coverage exclusion, never for cue bonuses)
_QUESTION_VERBS = frozenset((
    "cause", "causes", "caused", "causing", "fix", "fixes", "fixed",
    "fixing", "mean", "means", "meant", "say", "says", "said",
    "tell", "show", "shows", "explain", "happen", "happens"))

# hard answer-shape requirements: a when without a date, a count without
# a number and a yes/no without polarity are not answers to the question
# asked — they are passages NEAR the answer, and returning them as one
# would be the search-box-that-lies failure this module exists to avoid
_TYPE_HARD: Dict[str, Any] = {
    "time": (lambda t: bool(_TIME_IN_TEXT_RE.search(t)),
             "no dated evidence in the corpus for this when-question"),
    "count": (lambda t: bool(_DIGIT_RE.search(t)),
              "no numbered evidence in the corpus for this count-question"),
    "boolean": (lambda t: bool(_POLARITY_RE.search(t)),
                "no explicit yes/no statement in the corpus for this "
                "boolean-question"),
}


# ---------------------------------------------------------------------------
# units: paragraphs, table rows, list items — the answer-sized granularity
# ---------------------------------------------------------------------------

_TABLE_SEP_RE = re.compile(r"^\|?[\s:|-]+\|[\s:|-]*$")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+")
_FENCE_RE = re.compile(r"^\s*```")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`*#\[])")
_TIME_IN_TEXT_RE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}|\d{4})\b")
_DIGIT_RE = re.compile(r"\d")
# a yes/no answer must STATE polarity; 'Thankyou for the active support'
# is neither yes nor no, and extractive QA must be able to tell
_POLARITY_RE = re.compile(
    r"\b(?:not|no|never|cannot|can't|cannot|doesn't|does not|won't|"
    r"shouldn't|must not|unsupported|supported|yes|works|working|"
    r"possible|impossible|enabled|disabled)\b", re.IGNORECASE)


def _split_units(text: str, doc_id: str, section: str,
                 units: List[Dict[str, Any]], line_start: int = 1) -> None:
    """One markdown body -> passage units. A unit is a paragraph, a
    table row (this corpus's fixes live in tables — a row is one
    self-contained symptom/cause/fix triple), or one list item."""
    para: List[str] = []
    para_line = 0
    in_fence = False

    def flush() -> None:
        if para:
            units.append({"doc_id": doc_id, "section": section,
                          "line": para_line, "text": " ".join(para)})
            para.clear()

    for i, raw in enumerate(text.split("\n")):
        line_no = line_start + i
        stripped = raw.strip()
        if _FENCE_RE.match(stripped):
            # fenced code: formatting and commands, not prose answers —
            # the fix rows in tables carry the commands extractively
            in_fence = not in_fence
            flush()
            continue
        if in_fence:
            continue
        if stripped.startswith("#"):
            flush()  # headers end paragraphs; section tracked by caller
            continue
        if not stripped:
            flush()
            continue
        if stripped.startswith("|"):
            flush()
            if _TABLE_SEP_RE.match(stripped):
                continue
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            cells = [c for c in cells if c]
            if cells:
                units.append({"doc_id": doc_id, "section": section,
                              "line": line_no,
                              "text": " | ".join(cells)})
            continue
        if _LIST_ITEM_RE.match(stripped):
            flush()
            units.append({"doc_id": doc_id, "section": section,
                          "line": line_no,
                          "text": _LIST_ITEM_RE.sub("", stripped, count=1)})
            continue
        if not para:
            para_line = line_no
        para.append(stripped)
    flush()


def corpus_units() -> List[Dict[str, Any]]:
    """The committed repo-docs corpus as passage units (sorted paths;
    the nearest preceding markdown header is the section name)."""
    units: List[Dict[str, Any]] = []
    if not CORPUS_DIR.is_dir():
        return units
    for path in sorted(CORPUS_DIR.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        doc_id = path.stem
        lines = text.split("\n")
        # the retrieval corpus uses a strict 'key: value' header block;
        # everything after the first blank line is body. The header's
        # title/tags/synonyms lines are the doc's OWN index terms — the
        # body of ISS-418 never says 'quickshell crashed dialog' in one
        # place, its header does — so they rank too, each as its own unit.
        body_start = 0
        for i, raw in enumerate(lines):
            if not raw.strip():
                body_start = i + 1
                break
        header_keys = ("title", "tags", "synonyms")
        for i, raw in enumerate(lines[:body_start]):
            key, _, value = raw.partition(":")
            if key.strip().lower() in header_keys and value.strip():
                units.append({"doc_id": doc_id, "section": doc_id,
                              "line": i + 1, "text": value.strip(),
                              "meta": True})
        # units carry the section AS OF their position: rebuild in one
        # pass, splitting per current header
        chunks: List[Tuple[str, List[str]]] = []
        cur_section = doc_id
        for raw in lines[body_start:]:
            if raw.startswith("#"):
                cur_section = raw.lstrip("#").strip() or doc_id
                chunks.append((cur_section, []))
            else:
                if not chunks:
                    chunks.append((cur_section, []))
                chunks[-1][1].append(raw)
        for sec, chunk_lines in chunks:
            _split_units("\n".join(chunk_lines), doc_id, sec, units)
    return units


def _note_units(notes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    for n in notes:
        name = str(n.get("name") or n.get("path") or "note")
        text = str(n.get("text") or "")
        if not text.strip():
            continue
        _split_units(text, name, name, units)
    return units


# ---------------------------------------------------------------------------
# answer-type detection: what would an answer even look like?
# ---------------------------------------------------------------------------

_TYPE_CUES: Dict[str, Tuple[str, ...]] = {
    "cause": ("because", "cause", "causes", "due", "since", "root",
              "fix", "broken", "missing", "fails", "failure", "regression",
              "stale", "outdated", "conflict"),
    "how": ("run", "install", "execute", "add", "set", "edit", "enable",
            "disable", "remove", "rebuild", "update", "first", "then",
            "command", "steps", "sudo"),
    "time": ("when", "date", "version", "release", "since", "until",
             "window", "night", "morning", "yesterday", "week",
             "opened", "reported", "closed", "fixed"),
    "boolean": ("not", "no", "never", "cannot", "can't", "doesn't",
                "does not", "unsupported", "supported", "yes", "enabled",
                "disabled"),
    "count": ("total", "number", "count", "how many"),
    "list": ("include", "includes", "supported", "available", "options",
             "kinds", "types", "profiles", "presets"),
}


def detect_type(question: str) -> str:
    """One of: cause, how, time, who, where, count, list, boolean, what.

    Ordered rules, most specific first; the default is "what" because a
    bare noun phrase ("bar scale?") asks for a definition. Deliberately
    mechanical — the detector should be auditable in one glance.
    """
    q = (question or "").strip().lower()
    if not q:
        return "what"
    if re.search(r"\bwhy\b", q):
        return "cause"
    if re.search(r"\bhow (?:do|to|can|does|did|would|i)\b", q) or \
            re.search(r"\b(?:steps|instructions|procedure)\b", q):
        return "how"
    if re.search(r"\bwhen\b", q):
        return "time"
    if re.search(r"\bwho\b|\bwhose\b", q):
        return "who"
    if re.search(r"\bwhere\b", q):
        return "where"
    if re.search(r"\bhow (?:many|much)\b|\bcount\b", q):
        return "count"
    if re.search(r"\blist\b|\bwhich\b", q):
        return "list"
    if re.match(r"^(?:does|do|is|are|can|could|should|will|has|have)\b", q):
        return "boolean"
    return "what"


# ---------------------------------------------------------------------------
# mini BM25 over units
# ---------------------------------------------------------------------------

class _BM25:
    """In-memory BM25 over passage units. Small because the corpus is
    small; deterministic by construction (dicts iterate in insertion
    order and every loop sorts before scoring)."""

    def __init__(self, units: List[Dict[str, Any]]):
        self.units = units
        self.tf: List[Dict[str, int]] = []
        self.df: Dict[str, int] = {}
        total = 0
        for u in units:
            counts: Dict[str, int] = {}
            for tok in tokenize(u["text"]):
                counts[tok] = counts.get(tok, 0) + 1
            for term in counts:
                self.df[term] = self.df.get(term, 0) + 1
            self.tf.append(counts)
            total += len(counts)
        self.n = max(len(units), 1)
        self.avgdl = (total / self.n) or 1.0
        self._idf: Dict[str, float] = {}

    def idf(self, term: str) -> float:
        v = self._idf.get(term)
        if v is None:
            df = self.df.get(term, 0)
            v = math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))
            self._idf[term] = v
        return v

    def unit_scores(self, query_terms: Sequence[str]) -> List[float]:
        scores = [0.0] * len(self.units)
        for i, counts in enumerate(self.tf):
            dl = sum(counts.values()) or 1
            norm = 1.0 - _B + _B * (dl / self.avgdl)
            s = 0.0
            for term in set(query_terms):
                tf = counts.get(term)
                if tf:
                    s += self.idf(term) * tf * (_K1 + 1.0) / (
                        tf + _K1 * norm)
            scores[i] = s
        return scores


# ---------------------------------------------------------------------------
# sentence selection
# ---------------------------------------------------------------------------

def _sentences_of(unit: Dict[str, Any]) -> List[Dict[str, Any]]:
    """A unit -> sentence records (a table row IS one sentence)."""
    text = unit["text"]
    parts = _SENT_SPLIT_RE.split(text) if not text.startswith("|") \
        else [text]
    out = []
    for j, part in enumerate(parts):
        part = part.strip()
        if part:
            out.append({"unit": unit, "idx": j, "text": part,
                        "tokens": set(tokenize(part))})
    return out


def _sentence_score(sent: Dict[str, Any], q_terms: List[str],
                    idf: Any, qtype: str) -> float:
    """Idf-weighted query overlap + type cues - a mild verbosity
    penalty. Not BM25 (sentences are short); the idf weights keep
    'the' from ever beating 'qml'."""
    toks = sent["tokens"]
    s = 0.0
    for term in set(q_terms):
        if term in toks:
            s += idf(term)
    cues = _TYPE_CUES.get(qtype, ())
    lowered = sent["text"].lower()
    for cue in cues:
        if re.search(r"\b" + re.escape(cue) + r"\b", lowered):
            s += _CUE_BONUS
    if qtype == "time" and _TIME_IN_TEXT_RE.search(sent["text"]):
        s += _TIME_BONUS
    s -= 0.02 * len(sent["tokens"])
    return s


# ---------------------------------------------------------------------------
# the answer
# ---------------------------------------------------------------------------

def answer(question: str,
           notes: Optional[Sequence[Dict[str, Any]]] = None,
           sources: str = "both",
           now=None) -> Dict[str, Any]:
    """One question -> one extractive answer (or an honest verdict).

    notes: [{"name", "text"}] user-note records; sources: "both" |
    "corpus" | "notes". now anchors relative time windows (the module
    never reads a clock itself). The returned dict is JSON-shaped and
    deterministic; answer spans are verbatim substrings of their unit
    (pinned by test) and carry doc/section/line provenance.
    """
    qtype = detect_type(question)
    q_terms = tokenize(question)
    units: List[Dict[str, Any]] = []
    if sources in ("both", "corpus"):
        units.extend(corpus_units())
    if sources in ("both", "notes"):
        units.extend(_note_units(notes or []))

    base: Dict[str, Any] = {"question": question, "type": qtype,
                            "sources": sources}
    if not q_terms or not units:
        return {**base, "verdict": "NOT_FOUND", "answer": [],
                "reason": "no usable terms" if not q_terms
                else "no sources"}

    bm25 = _BM25(units)
    unit_scores = bm25.unit_scores(q_terms)
    # effective floors: the caps, or a share of the idf mass the
    # question's vocabulary actually has in these sources — whichever
    # is lower (a 3-line note cannot carry corpus-sized evidence)
    mass = sum(bm25.idf(t) for t in set(q_terms) if t in bm25.df)
    low_floor = min(_SCORE_LOW, _LOW_SHARE * mass)
    high_floor = min(_SCORE_HIGH, _HIGH_SHARE * mass)
    order = sorted(range(len(units)),
                   key=lambda i: (-unit_scores[i], units[i]["doc_id"],
                                  units[i]["line"]))
    top = order[:_TOP_PASSAGES]

    # temporal window, when the question carries one (existing parser)
    window = None
    if qtype in ("time", "what"):
        try:
            from assistant.core.nlhistory import parse_query
            parsed = parse_query(question, now=now)
            if parsed.window:
                lo, hi = parsed.window
                window = {"from": lo.isoformat(), "to": hi.isoformat()}
        except Exception:  # noqa: BLE001 — the window is a bonus, never
            window = None  # a dependency of a correct non-temporal answer

    cands: List[Dict[str, Any]] = []
    hard = _TYPE_HARD.get(qtype)
    # the sentence pool: every sentence of the top-3 DOCS (a when-answer
    # lives on the 'opened 2026-08-05' state line, a unit with almost no
    # query lexicon — passage rank must buy the whole document, not eight
    # lucky paragraphs), plus the top-8 units of every other doc. A TIME
    # question is stricter still: its answer is a fact ABOUT the winning
    # subject (when was IT reported), so only the top doc's dated
    # sentences qualify — a date on some other issue's comment is a
    # coincidence, not an answer.
    doc_best: Dict[str, float] = {}
    for i in order:
        d = units[i]["doc_id"]
        doc_best[d] = max(doc_best.get(d, 0.0), unit_scores[i])
    top_docs = sorted(doc_best, key=lambda d: (-doc_best[d], d))[:3]
    scope_docs = top_docs[:1] if qtype == "time" else top_docs
    pool_idx = [i for i in order
                if units[i]["doc_id"] in scope_docs][:400]
    if qtype != "time":
        pool_idx += [i for i in order[:_TOP_PASSAGES]
                     if i not in set(pool_idx)]
    for ui in pool_idx:
        unit = units[ui]
        if unit.get("meta"):
            # title/tags/synonyms rank the doc but are not prose — an
            # answer span quoted from a tags line would be the index
            # talking to itself
            continue
        for sent in _sentences_of(unit):
            if hard and not hard[0](sent["text"]):
                continue
            score = _sentence_score(sent, q_terms, bm25.idf, qtype)
            # a sentence inherits a share of its passage's rank: the
            # best passage's sentences matter more than its words alone
            score += _UNIT_INHERIT * unit_scores[ui]
            cands.append({"score": score, "doc_id": unit["doc_id"],
                          "section": unit["section"],
                          "line": unit["line"], "text": sent["text"],
                          "unit_order": ui, "sent_idx": sent["idx"]})
    if not cands:
        return {**base, "verdict": "NOT_FOUND", "answer": [],
                "reason": hard[1] if hard and qtype in _TYPE_HARD
                else "no candidate sentences"}
    cands.sort(key=lambda c: (-c["score"], c["doc_id"], c["line"],
                              c["sent_idx"]))
    best = cands[0]
    second = cands[1]["score"] if len(cands) > 1 else 0.0

    # query-term coverage of the assembled answer: a span that names the
    # tool but dodges the DISCRIMINATING term (dock badges offered for an
    # autohide question) is near-miss evidence — downgrade to THIN
    def _coverage(spans: List[Dict[str, Any]]) -> float:
        ranked = sorted(set(q_terms), key=lambda t: -bm25.idf(t))
        # coverage judges CONTENT terms: for 'what causes X', the word
        # 'causes' is the question's verb, not its subject — demanding
        # it in the answer would downgrade every correct reply
        cues = set(_TYPE_CUES.get(qtype, ())) | _QUESTION_VERBS
        ranked = [t for t in ranked if t not in cues][:3]
        if not ranked:
            return 1.0
        text = " ".join(s["text"] for s in spans).lower()
        hit = sum(1 for t in ranked if re.search(r"\b" + re.escape(t) +
                                                 r"\b", text))
        return hit / len(ranked)

    def _verdict_for(chosen_spans: List[Dict[str, Any]]) -> str:
        if best["score"] < low_floor:
            return "NOT_FOUND"
        conf_ok = best["score"] >= high_floor and (
            (best["score"] - second) / best["score"] >= _MARGIN
            or best["score"] >= 2.0 * high_floor)
        if conf_ok and qtype not in _TYPE_HARD and \
                _coverage(chosen_spans) < 0.5:
            return "THIN"
        return "CONFIDENT" if conf_ok else (
            "THIN" if best["score"] >= low_floor else "NOT_FOUND")

    # assembly: procedures read in document order (steps make no sense
    # ranked by score); every other type takes the single best sentence,
    # except list/cause which may carry one runner-up from the same
    # passage when the margin is wide.
    chosen: List[Dict[str, Any]] = []
    verdict = "NOT_FOUND"
    gate_blocked = False
    if best["score"] >= low_floor:
        # the assembly gate for open types: a span must carry one of the
        # two most discriminative CONTENT query terms, or it is adjacency,
        # not evidence ('the Fix: header' does not answer a gamescope
        # question). Cue words are excluded — 'causes' is the question's
        # verb; demanding it in the span downgrades correct answers.
        cues = set(_TYPE_CUES.get(qtype, ())) | _QUESTION_VERBS
        gate = [t for t in sorted(set(q_terms), key=lambda t: -bm25.idf(t))
                if t not in cues][:2]
        gate_ok = (lambda c: True) if (hard or not gate) else \
            (lambda c: any(re.search(r"\b" + re.escape(t) + r"\b",
                                     c["text"].lower()) for t in gate))
        if qtype == "how":
            # secondary steps come only from strong-contender docs (best
            # unit within 2x of the winner): a fringe doc sharing the
            # word 'launching' is not part of this procedure
            leader = max(doc_best.values())
            docs_ok = {d for d, s in doc_best.items() if s >= 0.5 * leader}
            pool = [c for c in cands
                    if c["score"] >= max(low_floor, 0.5 * best["score"])
                    and gate_ok(c) and c["doc_id"] in docs_ok]
            pool.sort(key=lambda c: (c["unit_order"], c["sent_idx"]))
            chosen = pool[:_MAX_SENTENCES]
        else:
            chosen = [best] if gate_ok(best) else []
            for c in cands[1:]:
                if len(chosen) >= _MAX_SENTENCES:
                    break
                if c["doc_id"] == best["doc_id"] and \
                        qtype in ("list", "cause", "what") and \
                        c["score"] >= low_floor and gate_ok(c):
                    chosen.append(c)
                else:
                    break
        if not chosen:
            verdict = "NOT_FOUND"
            gate_blocked = True
        else:
            gate_blocked = False
            verdict = _verdict_for(chosen)

    answer_spans = [{"text": c["text"], "doc_id": c["doc_id"],
                     "section": c["section"], "line": c["line"]}
                    for c in chosen]
    out = {**base, "verdict": verdict,
           "confidence": round(best["score"], 4),
           "runner_up": round(second, 4),
           "answer": answer_spans}
    if window:
        out["window"] = window
    if verdict == "NOT_FOUND":
        out["reason"] = ("evidence existed but never named the subject "
                         "terms" if gate_blocked
                         else "no sentence reached the evidence floor")
    return out


def render_lines(r: Dict[str, Any]) -> List[str]:
    """Human surface: the verdict first, the spans verbatim, the
    provenance inline — an extractive answer you can click-check."""
    head = f"[{r['verdict']}] {r['type']}-question: {r['question']}"
    lines = [head]
    if r.get("window"):
        w = r["window"]
        lines.append(f"  time window understood: {w['from']} .. {w['to']}")
    if not r["answer"]:
        lines.append(f"  no extractive answer ({r.get('reason', '')})")
        lines.append("  say: the honest fallback is `search`, or ask a "
                     "question the corpus can quote.")
        return lines
    for span in r["answer"]:
        lines.append(f"  {span['text']}")
        lines.append(f"    — {span['doc_id']} § {span['section']}:"
                     f"{span['line']}")
    if r["verdict"] == "THIN":
        lines.append("  (thin evidence: verify the citation before "
                     "acting on this)")
    return lines
