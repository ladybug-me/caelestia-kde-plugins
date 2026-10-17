"""Keyword extraction (TF-IDF) and extractive summarization (TextRank).

Both work with no external model: TF-IDF ranks a note's own distinctive words
against a corpus of other notes, and TextRank scores each sentence by how
much vocabulary it shares with the note's other sentences, run through the
same power-iteration idea as graph.Graph.pagerank but over a weighted
similarity matrix instead of wiki-links.
"""
import math
import re
from collections import Counter

from .nlp import tokens

SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _tf(doc_tokens):
    c = Counter(doc_tokens)
    total = sum(c.values()) or 1
    return {w: n / total for w, n in c.items()}


def keywords(text, corpus_texts=(), top=8):
    """Rank `text`'s own words by TF-IDF against corpus_texts (other notes,
    NOT including `text` itself). With no corpus, falls back to raw frequency.
    """
    doc_tokens = tokens(text)
    if not doc_tokens:
        return []
    tf = _tf(doc_tokens)
    other_docs = [tokens(t) for t in corpus_texts]
    n_docs = len(other_docs) + 1
    scored = []
    for w, f in tf.items():
        df = 1 + sum(1 for d in other_docs if w in d)
        idf = math.log(n_docs / df) + 1.0
        scored.append((w, f * idf))
    scored.sort(key=lambda x: -x[1])
    return [w for w, _ in scored[:top]]


def tfidf_vectors(texts):
    """TF-IDF vectors for a list of texts, one {word: weight} per text.

    The same TF-IDF family keywords() uses (term frequency times a
    smoothed inverse document frequency, Salton & Buckley 1988's
    standard weighting), computed over the given corpus: df counts the
    texts containing the word, n_docs = len(texts). Deterministic; the
    returned dicts are plain and comparable with cosine() below.
    """
    doc_tokens = [tokens(t) for t in texts]
    n_docs = len(doc_tokens)
    if n_docs == 0:
        return []
    dfs = Counter()
    for d in doc_tokens:
        dfs.update(set(d))
    vectors = []
    for d in doc_tokens:
        tf = _tf(d)
        vectors.append({w: f * (math.log(n_docs / dfs[w]) + 1.0)
                        for w, f in tf.items()} if d else {})
    return vectors


def cosine(a, b):
    """Cosine similarity between two TF-IDF vectors (tfidf_vectors'
    output). 0.0 when either side is empty — no vocabulary is not a
    similarity."""
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[w] * b[w] for w in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _sentences(text):
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in SENTENCE.split(text) if s.strip()]


def _similarity(a_tokens, b_tokens):
    a, b = set(a_tokens), set(b_tokens)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    norm = math.log(len(a) + 1) + math.log(len(b) + 1)
    return inter / norm if norm else 0.0


def summarize(text, sentences_out=3, damping=0.85, iters=40):
    """Extractive summary: the `sentences_out` highest-PageRank sentences, in
    their original order. Returns fewer sentences than that if the note is
    already shorter than the target.
    """
    sents = _sentences(text)
    n = len(sents)
    if n <= sentences_out:
        return sents
    tok = [tokens(s) for s in sents]
    weights = [[_similarity(tok[i], tok[j]) if i != j else 0.0 for j in range(n)]
               for i in range(n)]
    out_sum = [sum(row) or 1.0 for row in weights]
    rank = [1.0 / n] * n
    for _ in range(iters):
        new = [(1 - damping) / n] * n
        for j in range(n):
            for i in range(n):
                if weights[i][j]:
                    new[j] += damping * rank[i] * weights[i][j] / out_sum[i]
        rank = new
    top_idx = sorted(range(n), key=lambda i: -rank[i])[:sentences_out]
    return [sents[i] for i in sorted(top_idx)]
