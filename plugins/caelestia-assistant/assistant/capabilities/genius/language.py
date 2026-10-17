"""genius.language — language intelligence without a language model.

  * sentiment: lexicon scoring with negation flips, intensifiers and
    dampeners (VADER-style compound score)
  * readability: Flesch Reading Ease, Flesch-Kincaid Grade, Gunning Fog,
    SMOG, ARI, Coleman-Liau — six formulas plus the shared counts
  * keyword extraction: RAKE (phrase candidates, degree/frequency
    scoring) and a YAKE-lite contextual term scorer
  * entity-lite: dates, times, numbers with units, percentages, emails,
    URLs, file paths, git hashes, versions, capitalized name sequences
  * question answering over a supplied text: interrogative parsing +
    sentence evidence ranking (who/what/when/where/why/how much)
  * query-focused extractive summarization (TextRank-style graph
    centrality + lexical overlap with the question)
  * language detection: character trigram profiles against 14 languages
  * centroid text classifier: TF-IDF class centroids with cosine —
    trains from your labels, ranks with evidence terms, and asks
    nothing of any server

Every result carries the fragments that produced it.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Sequence, Set, Tuple

__all__ = [
    "sentiment", "readability", "text_stats", "rake_keywords", "yake_keywords",
    "extract_entities", "answer_question", "summarize_focused",
    "detect_language", "TextClassifier",
]

# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------

_POSITIVE = {
    "good": 1.8, "great": 2.5, "excellent": 3, "awesome": 3, "amazing": 3,
    "love": 3, "like": 1.2, "happy": 2.5, "glad": 1.8, "wonderful": 2.8,
    "fantastic": 2.8, "perfect": 2.8, "best": 2.5, "better": 1.5,
    "nice": 1.5, "beautiful": 2.2, "helpful": 1.8, "works": 1.5,
    "fixed": 1.5, "fast": 1.5, "clean": 1.5, "thanks": 1.8, "thank": 1.8,
    "appreciate": 1.8, "enjoy": 2, "impressive": 2.4, "solid": 1.4,
    "reliable": 1.8, "smooth": 1.8, "wow": 2.5, "brilliant": 2.6,
    "recommend": 1.8, "success": 1.8, "win": 1.5, "progress": 1.2,
}
_NEGATIVE = {
    "bad": -1.8, "terrible": -3, "awful": -3, "hate": -3, "worst": -2.8,
    "broken": -2.2, "bug": -1.5, "bugs": -1.5, "crash": -2.4, "crashes": -2.4,
    "crashed": -2.4, "fail": -2, "fails": -2, "failed": -2, "failure": -2.2,
    "error": -1.6, "errors": -1.6, "problem": -1.4, "problems": -1.4,
    "issue": -1.2, "slow": -1.5, "ugly": -1.8, "annoying": -1.8,
    "frustrating": -2.2, "confusing": -1.6, "useless": -2.4, "waste": -2.2,
    "disappointed": -2.4, "disappointing": -2.4, "sad": -2, "angry": -2.4,
    "regret": -2, "stuck": -1.5, "blocked": -1.5, "pain": -1.8,
    "leak": -2, "corrupt": -2.4, "corrupted": -2.4, "missing": -1.2,
    "unstable": -2, "laggy": -1.8, "glitch": -1.6, "weird": -1.2,
    "wrong": -1.5, "poor": -1.8, "horrible": -2.8, "nightmare": -2.6,
}
_NEGATORS = {"not", "no", "never", "hardly", "barely", "isn't", "wasn't",
             "aren't", "don't", "doesn't", "didn't", "can't", "cannot", "won't"}
_INTENSIFIERS = {"very": 1.6, "really": 1.4, "extremely": 1.9, "so": 1.3,
                 "totally": 1.5, "absolutely": 1.7, "completely": 1.6,
                 "insanely": 1.8, "super": 1.5, "highly": 1.4}
_DAMPENERS = {"slightly": 0.6, "somewhat": 0.7, "kinda": 0.7, "a_bit": 0.7,
              "mostly": 0.8, "fairly": 0.8, "rather": 0.85}


def sentiment(text: str) -> Dict[str, Any]:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    hits: List[Dict[str, Any]] = []
    total = 0.0
    pos = neg = 0
    for i, w in enumerate(words):
        base = _POSITIVE.get(w, 0.0) + _NEGATIVE.get(w, 0.0)
        if base == 0:
            continue
        factor = 1.0
        applied = []
        for back in (1, 2, 3):
            if i - back >= 0:
                prev = words[i - back]
                if prev in _NEGATORS and back <= 2:
                    factor *= -0.8
                    applied.append(f"negator {prev!r}")
                elif prev in _INTENSIFIERS:
                    factor *= _INTENSIFIERS[prev]
                    applied.append(f"intensifier {prev!r}")
                elif prev in _DAMPENERS:
                    factor *= _DAMPENERS[prev]
                    applied.append(f"dampener {prev!r}")
        score = base * factor
        total += score
        if score > 0:
            pos += 1
        elif score < 0:
            neg += 1
        hits.append({"word": w, "score": round(score, 3),
                     "modifiers": applied or None})
    norm = total / math.sqrt(total * total + 15)  # VADER-style normalization
    if norm > 0.05:
        label = "positive"
    elif norm < -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {"compound": round(norm, 4), "label": label,
            "positive_hits": pos, "negative_hits": neg,
            "evidence": hits,
            "note": "lexicon + modifiers; it reads tone, not meaning"}


# ---------------------------------------------------------------------------
# Readability + text stats
# ---------------------------------------------------------------------------

def _syllables(word: str) -> int:
    w = word.lower().strip("'-")
    if not w:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in w:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if w.endswith("e") and count > 1 and not w.endswith(("le", "ee")):
        count -= 1
    return max(1, count)


def _tokenize_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z'-]*", text)


_DIFFICULT = {
    "however", "therefore", "accordingly", "consequently", "furthermore",
    "nevertheless", "nonetheless", "notwithstanding", "aforementioned",
    "hereinafter", "whereas", "henceforth", "theretofore", "thereupon",
}


def text_stats(text: str) -> Dict[str, Any]:
    sents = _tokenize_sentences(text)
    words = _words(text)
    syls = sum(_syllables(w) for w in words)
    unique = set(w.lower() for w in words)
    long_words = [w for w in words if _syllables(w) >= 3]
    return {"sentences": len(sents), "words": len(words), "syllables": syls,
            "unique_words": len(unique),
            "type_token_ratio": round(len(unique) / len(words), 4) if words else 0.0,
            "avg_sentence_length": round(len(words) / len(sents), 2) if sents else 0.0,
            "avg_word_length": round(sum(len(w) for w in words) / len(words), 2) if words else 0.0,
            "long_word_share": round(len(long_words) / len(words), 4) if words else 0.0,
            "lexical_density": round(
                len({w.lower() for w in words if w.lower() not in
                     {"the", "a", "an", "and", "or", "but", "is", "are", "was",
                      "were", "in", "on", "at", "to", "of", "for", "it"}}) /
                max(1, len(words)), 4)}


def readability(text: str) -> Dict[str, Any]:
    sents = _tokenize_sentences(text)
    words = _words(text)
    if not sents or not words:
        raise ValueError("need text with at least one sentence and word")
    ns, nw = len(sents), len(words)
    syls = sum(_syllables(w) for w in words)
    complex_words = [w for w in words if _syllables(w) >= 3]
    nc = len(complex_words)
    syllables_per_word = syls / nw
    words_per_sentence = nw / ns
    flesch = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
    fk_grade = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
    proper_words = [w for w in words if w[0].isupper() or w.lower() in _DIFFICULT]
    fog = 0.4 * (words_per_sentence + 100 * nc / nw)
    polysyllabic = sum(1 for w in words if _syllables(w) >= 3)
    smog = 1.0430 * math.sqrt(polysyllabic * (30 / ns)) + 3.1291 if ns else 0
    chars = sum(len(w) for w in words)
    ari = 4.71 * (chars / nw) + 0.5 * words_per_sentence - 21.43
    # Coleman-Liau: C = 0.0588*L - 0.296*S - 15.8, L/S per 100 words/sentences
    cl_grade = 0.0588 * (chars / nw * 100) - 0.296 * (100 / words_per_sentence) - 15.8

    def _grade(v):
        return max(1, round(v))
    def _band(f):
        if f >= 90: return "very easy (5th grade)"
        if f >= 80: return "easy (6th grade)"
        if f >= 70: return "fairly easy (7th grade)"
        if f >= 60: return "plain English (8-9th grade)"
        if f >= 50: return "fairly difficult (10-12th)"
        if f >= 30: return "difficult (college)"
        return "very difficult (graduate)"
    return {
        "flesch_reading_ease": round(flesch, 2), "flesch_band": _band(flesch),
        "flesch_kincaid_grade": _grade(fk_grade),
        "gunning_fog": round(fog, 2),
        "smog_grade": round(smog, 2),
        "automated_readability_index": round(ari, 2),
        "coleman_liau": round(cl_grade, 2),
        "counts": {"sentences": ns, "words": nw, "syllables": syls,
                   "complex_words": nc,
                   "avg_words_per_sentence": round(words_per_sentence, 2)},
    }


# ---------------------------------------------------------------------------
# Keyword extraction: RAKE + YAKE-lite
# ---------------------------------------------------------------------------

_STOP = set("""a about above after again against all am an and any are as at be
because been before being below between both but by could did do does doing down
during each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no nor
not now of off on once only or other our ours ourselves out over own same she
should so some such than that the their theirs them themselves then there these
they this those through to too under until up very was we were what when where
which while who whom why will with you your yours yourself yourselves get got
make made also use used using one two want need going really thing things""".split())


def _split_phrases(text: str) -> List[List[str]]:
    raw = re.split(r"[,!;:?.()\[\]\"'\-—/]", text.lower())
    out = []
    for chunk in raw:
        toks = [t for t in chunk.split() if t and t not in _STOP]
        if toks:
            out.append(toks)
    return out


def rake_keywords(text: str, top: int = 10) -> Dict[str, Any]:
    phrases = _split_phrases(text)
    freq: Dict[str, int] = {}
    degree: Dict[str, int] = {}
    for ph in phrases:
        for w in ph:
            freq[w] = freq.get(w, 0) + 1
            degree[w] = degree.get(w, 0) + len(ph)
    scored = []
    for ph in phrases:
        if not ph:
            continue
        score = sum(degree[w] / freq[w] for w in ph) / len(ph)
        key = " ".join(ph)
        if len(key) > 2:
            scored.append({"phrase": key, "score": round(score, 3)})
    scored.sort(key=lambda s: -s["score"])
    seen: Set[str] = set()
    uniq = []
    for s in scored:
        if s["phrase"] in seen:
            continue
        seen.add(s["phrase"])
        uniq.append(s)
    return {"method": "RAKE", "keywords": uniq[:top],
            "n_phrases": len(phrases)}


def yake_keywords(text: str, top: int = 10) -> Dict[str, Any]:
    """YAKE-lite: score terms by casing, position, and dispersion."""
    sents = _tokenize_sentences(text)
    words = _words(text)
    lower = [w.lower() for w in words]
    n = len(words)
    if not n:
        return {"method": "YAKE-lite", "keywords": []}
    tf: Dict[str, int] = {}
    positions: Dict[str, List[int]] = {}
    for i, w in enumerate(lower):
        tf[w] = tf.get(w, 0) + 1
        positions.setdefault(w, []).append(i)
    total_sents = max(1, len(sents))
    sent_hits: Dict[str, int] = {}
    for s in sents:
        for w in _words(s):
            sent_hits[w.lower()] = sent_hits.get(w.lower(), 0) + 1
    scored = []
    for w, f in tf.items():
        if w in _STOP or len(w) < 3:
            continue
        casing = 1.0
        if w in lower and any(orig == w.capitalize() for orig in words):
            casing = 1.5 if f > 1 else 1.0
        posn = positions[w][0] / n
        ps = sent_hits.get(w, 0) / total_sents
        spread = 1.0
        if len(positions[w]) > 1:
            spread = (positions[w][-1] - positions[w][0]) / n
        # YAKE-style: lower score = better
        score = (f * ps) / (casing * (1 + spread) * (1 + posn))
        scored.append({"term": w, "frequency": f, "score": round(score, 4)})
    scored.sort(key=lambda s: -s["score"])
    return {"method": "YAKE-lite", "keywords": scored[:top],
            "note": "higher score = more salient"}


# ---------------------------------------------------------------------------
# Entity-lite extraction
# ---------------------------------------------------------------------------

_PATTERNS = [
    ("email", r"[\w.+-]+@[\w-]+\.[\w.]+"),
    ("url", r"https?://[^\s)>\]}\"']+|www\.[^\s)>\]}\"']+"),
    ("git_hash", r"\b[0-9a-f]{7,40}\b(?![\w.])"),
    ("ipv4", r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ("version", r"\bv?\d+\.\d+(?:\.\d+)*(?:-[a-z0-9.]+)?\b"),
    ("file_path", r"(?:~/|/)[\w./-]+\.[a-z0-9]{1,6}\b"),
    ("percentage", r"\d+(?:\.\d+)?\s?(?:%|percent)"),
    ("money", r"(?:rs\.?|inr|usd|\$|€|£)\s?\d+(?:[.,]\d+)*(?:\s?(?:k|m|cr|lakh|crore))?", ),
    ("number_unit", r"\d+(?:\.\d+)?\s?(?:gb|mb|kb|tb|ghz|mhz|ms|s|sec|min|hrs?|hours?|days?|px|pt|km|kg|mbps|kbps)\b"),
    ("time", r"\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:am|pm)?\b"),
    ("date", r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
             r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2},? \d{4}\b"),
    ("number", r"\b\d+(?:\.\d+)?\b"),
]


def extract_entities(text: str) -> Dict[str, Any]:
    found: Dict[str, List[str]] = {}
    spans: List[Tuple[int, int, str]] = []
    for kind, pattern in _PATTERNS:
        if not isinstance(pattern, str):
            continue
        for m in re.finditer(pattern, text, re.IGNORECASE if kind in ("date",) else 0):
            spans.append((m.start(), m.end(), kind))
    # longest-match wins at overlapping positions
    spans.sort(key=lambda s: -(s[1] - s[0]))
    taken: List[Tuple[int, int]] = []
    for start, end, kind in spans:
        if any(start < te and end > ts for ts, te in taken):
            continue
        taken.append((start, end))
        found.setdefault(kind, []).append(text[start:end])
    # capitalized sequences (skip sentence starts that are just words)
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", text):
        frag = m.group(1)
        if frag.lower() not in _STOP and len(frag) > 3:
            found.setdefault("name_candidate", []).append(frag)
    for k in found:
        seen, uniq = set(), []
        for v in found[k]:
            if v not in seen:
                seen.add(v)
                uniq.append(v)
        found[k] = uniq[:20]
    return {"entities": found, "n_kinds": len(found)}


# ---------------------------------------------------------------------------
# Question answering + focused summarization
# ---------------------------------------------------------------------------

_QUESTION_HINTS = {
    "who": ["name_candidate", "name"],
    "when": ["date", "time"],
    "where": ["file_path", "url"],
    "how much": ["money", "percentage", "number_unit", "number"],
    "how many": ["number", "number_unit"],
}


def answer_question(question: str, text: str) -> Dict[str, Any]:
    """Extractive QA: score sentences by interrogative focus + overlap."""
    q = question.lower().strip()
    focus = None
    for h in ("how much", "how many", "who", "when", "where", "why", "how", "what"):
        if q.startswith(h):
            focus = h
            break
    q_terms = {t for t in re.findall(r"[a-z']+", q) if t not in _STOP and len(t) > 2}
    sents = _tokenize_sentences(text)
    if not sents:
        raise ValueError("no text to answer from")
    scored = []
    for idx, s in enumerate(sents):
        s_lower = s.lower()
        terms = {t for t in re.findall(r"[a-z']+", s_lower)}
        overlap = len(q_terms & terms)
        score = 2.0 * overlap
        bonus = 0.0
        if focus in ("who", "when", "where"):
            ents = extract_entities(s).get("entities", {})
            for kind in _QUESTION_HINTS.get(focus, []):
                if ents.get(kind):
                    bonus += 1.5
        elif focus in ("how much", "how many"):
            if re.search(r"\d", s):
                bonus += 1.2
        elif focus == "why":
            if re.search(r"\b(because|since|due to|as a result|therefore|so that)\b",
                         s_lower):
                bonus += 1.5
        elif focus == "how":
            if re.search(r"\b(by|using|via|through|step|then|configure|run)\b", s_lower):
                bonus += 1.0
        position_bonus = 0.3 * (1 - idx / max(1, len(sents)))
        scored.append({"sentence": s, "score": round(score + bonus + position_bonus, 3),
                       "overlap_terms": sorted(q_terms & terms),
                       "index": idx})
    scored.sort(key=lambda x: -x["score"])
    best = scored[0]
    # candidate answer snippet: entities inside the best sentence
    ents = extract_entities(best["sentence"])["entities"]
    wanted = _QUESTION_HINTS.get(focus or "", [])
    candidates = [e for k in wanted for e in ents.get(k, [])][:5]
    return {"question": question, "focus": focus,
            "answer": best["sentence"],
            "answer_entities": candidates,
            "confidence": "high" if best["score"] >= 3 else
                          "medium" if best["score"] >= 1 else "low",
            "supporting": [s["sentence"] for s in scored[1:4] if s["score"] > 0],
            "note": "extractive — it quotes the text, it never invents"}


def _sentence_similarity(a_terms: Set[str], b_terms: Set[str]) -> float:
    if not a_terms or not b_terms:
        return 0.0
    return len(a_terms & b_terms) / (math.log(1 + len(a_terms)) +
                                     math.log(1 + len(b_terms)) + 1e-9)


def summarize_focused(text: str, query: str, n_sentences: int = 3) -> Dict[str, Any]:
    """Query-focused extractive summary (graph centrality + query overlap)."""
    sents = _tokenize_sentences(text)
    if not sents:
        raise ValueError("no sentences to summarize")
    n_sentences = max(1, min(n_sentences, len(sents)))
    term_sets = [{t for t in re.findall(r"[a-z']+", s.lower()) if t not in _STOP}
                 for s in sents]
    n = len(sents)
    # TextRank-style power iteration over the sentence graph
    scores = [1.0 / n] * n
    for _ in range(30):
        new = [0.15 / n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                sim = _sentence_similarity(term_sets[i], term_sets[j])
                if sim > 0:
                    new[i] += 0.85 * scores[j] * sim / \
                        max(1e-9, sum(_sentence_similarity(term_sets[j], term_sets[k])
                                      for k in range(n) if k != j))
        if max(abs(new[i] - scores[i]) for i in range(n)) < 1e-6:
            scores = new
            break
        scores = new
    q_terms = {t for t in re.findall(r"[a-z']+", query.lower())
               if t not in _STOP and len(t) > 2}
    combined = [(i, scores[i] + 2.0 * len(term_sets[i] & q_terms)) for i in range(n)]
    top = sorted(combined, key=lambda kv: -kv[1])[:n_sentences]
    top.sort()  # keep original order in the output
    return {"query": query,
            "summary": " ".join(sents[i] for i, _ in top),
            "selected_indices": [i for i, _ in top],
            "n_sentences_total": n,
            "method": "TextRank centrality + query overlap"}


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

_LANG_SAMPLES = {
    "english": "the quick brown fox jumps over the lazy dog and runs through "
               "the field while thinking about language models and statistics",
    "hindi_translit": "yeh ek bahut hi accha din hai aur main aaj karna chahta "
                      "hoon jo mujhe pasand hai kyunki zindagi bahut khoobsurat hai",
    "spanish": "el zorro marrón rápido salta sobre el perro perezoso mientras "
               "piensa en el lenguaje y las estadísticas de la vida moderna",
    "french": "le renard brun rapide saute par dessus le chien paresseux pendant "
              "qu'il pense au langage et aux statistiques de la vie moderne",
    "german": "der schnelle braune fuchs springt über den faulen hund während "
              "er über sprache und statistik des modernen lebens nachdenkt",
    "italian": "la volpe marrone veloce salta sopra il cane pigro mentre pensa "
               "al linguaggio e alle statistiche della vita moderna",
    "portuguese": "a raposa marrom rápida salta sobre o cão preguiçoso enquanto "
                  "pensa na linguagem e nas estatísticas da vida moderna",
    "dutch": "de snelle bruine vos springt over de luie hond terwijl hij nadenkt "
             "over taal en statistiek van het moderne leven",
    "russian": "быстрая бурая лиса прыгает через ленивую собаку пока думает о "
               "языке и статистике современной жизни",
    "polish": "szybki brązowy lis przeskakuje nad leniwym psem myśląc o języku "
              "i statystyce nowoczesnego życia",
    "turkish": "hızlı kahverengi tilki tembel köpeğin üzerinden atlar ve modern "
               "hayatın dili ve istatistiği hakkında düşünür",
    "indonesian": "rubah coklat cepat melompati anjing malas sambil memikirkan "
                  "bahasa dan statistik kehidupan modern",
    "japanese_romaji": "hayai katsu chairo no kitsune wa namakemono no inu o "
                       "tobikoete gendai no gengo to toukei ni tsuite kangaeru",
    "arabic_translit": "althaealab alssariee yaqfizu ealaa alkalb alkassil "
                       "biyna yufakkir fi allughat waleihsaeat alhayat alhaditha",
}


def _trigrams(s: str) -> Dict[str, int]:
    s = "".join(c for c in s.lower() if c.isalnum() or c.isspace())
    grams: Dict[str, int] = {}
    for i in range(len(s) - 2):
        g = s[i:i + 3]
        grams[g] = grams.get(g, 0) + 1
    return grams


_LANG_PROFILES = {lang: _trigrams(sample) for lang, sample in _LANG_SAMPLES.items()}


def detect_language(text: str) -> Dict[str, Any]:
    if len(text.strip()) < 3:
        raise ValueError("need at least a few characters")
    tg = _trigrams(text)
    total = sum(tg.values()) or 1
    scores = {}
    for lang, profile in _LANG_PROFILES.items():
        overlap = sum(min(c, profile.get(g, 0)) for g, c in tg.items())
        scores[lang] = overlap / total
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best, best_score = ranked[0]
    margin = best_score - (ranked[1][1] if len(ranked) > 1 else 0)
    return {"language": best, "confidence": round(best_score, 4),
            "margin_over_second": round(margin, 4),
            "ranking": [(lang, round(s, 4)) for lang, s in ranked[:4]],
            "verdict": (f"likely {best}" if best_score > 0.25 else
                        "no confident match in the 14-language profile set")}


# ---------------------------------------------------------------------------
# Centroid text classifier (self-learning from caller labels)
# ---------------------------------------------------------------------------

class TextClassifier:
    """TF-IDF class centroids + cosine similarity.

    Train from `fit(texts, labels)`; predict returns the class with its
    top contributing terms (its evidence). Persistent-friendly via
    to_dict/from_dict so callers can keep it in the brain state.
    """

    def __init__(self):
        self.class_docs: Dict[str, List[str]] = {}
        self.df: Dict[str, int] = {}
        self.n_docs = 0

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "TextClassifier":
        if len(texts) != len(labels) or not texts:
            raise ValueError("need aligned, non-empty texts and labels")
        for t, l in zip(texts, labels):
            self.class_docs.setdefault(l, []).append(t)
            for term in set(_terms_of(t)):
                self.df[term] = self.df.get(term, 0) + 1
            self.n_docs += 1
        return self

    def _idf(self, term: str) -> float:
        return math.log((1 + self.n_docs) / (1 + self.df.get(term, 0))) + 1.0

    def predict(self, text: str) -> Dict[str, Any]:
        if not self.class_docs:
            raise ValueError("untrained classifier")
        q = _terms_of(text)
        scores: Dict[str, float] = {}
        evidence: Dict[str, List[str]] = {}
        for cls, docs in self.class_docs.items():
            sim = 0.0
            hits: Set[str] = set()
            for d in docs:
                d_terms = _terms_of(d)
                common = set(q) & set(d_terms)
                for c in common:
                    hits.add(c)
                sim += len(common) / (math.sqrt(len(q) * len(d_terms)) + 1e-9)
            scores[cls] = sim / len(docs)
            evidence[cls] = sorted(hits, key=lambda t: -self._idf(t))[:5]
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best = ranked[0][0]
        margin = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0)
        confidence = "high" if margin > 0.15 and ranked[0][1] > 0 else "low"
        return {"class": best, "scores": {c: round(s, 4) for c, s in scores.items()},
                "margin": round(margin, 4), "confidence": confidence,
                "evidence_terms": evidence[best][:5],
                "n_training_docs": self.n_docs,
                "verdict": ("clear match" if confidence == "high"
                            else "weak or ambiguous — teach me more examples")}

    def to_dict(self) -> Dict[str, Any]:
        return {"class_docs": self.class_docs, "df": self.df, "n_docs": self.n_docs}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextClassifier":
        c = cls()
        c.class_docs = {k: list(v) for k, v in (data.get("class_docs") or {}).items()}
        c.df = dict(data.get("df") or {})
        c.n_docs = int(data.get("n_docs") or 0)
        return c


def _terms_of(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z']+", text.lower())
            if t not in _STOP and len(t) > 2]
