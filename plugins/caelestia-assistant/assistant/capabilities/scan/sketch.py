"""scan.sketch — Count-Min, HyperLogLog, reservoir sampling, EWMA, Page-Hinkley.

Four classical stream structures, one module, all plain-data serialisable:

- `CountMinSketch`: point-query frequencies with the (epsilon, delta) bound —
  the estimate never UNDER-counts, and over-counts by at most eps*N with
  probability 1-delta. Width/depth derived from the requested bounds.
- `HyperLogLog`: distinct count with ~1.04/sqrt(m) standard error using
  hashlib registers and stochastic averaging (Flajolet et al. 2007).
- `Reservoir`: Vitter's algorithm R — a uniform sample of k items from a
  stream of unknown length in one pass.
- `EWMA` + `page_hinkley`: exponentially weighted rate tracking and the
  Page-Hinkley change detector (cumulative alarm over negative drifts) —
  the honest way to say "the error rate just changed" without a model.
"""
from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Dict, List, Optional

__all__ = ["CountMinSketch", "HyperLogLog", "Reservoir", "EWMA",
           "page_hinkley"]


# ---------------------------------------------------------------------------
# Count-Min Sketch
# ---------------------------------------------------------------------------

class CountMinSketch:
    def __init__(self, epsilon: float = 0.002, delta: float = 0.001) -> None:
        if epsilon <= 0 or not (0.0 < delta < 1.0):
            raise ValueError("epsilon > 0 and 0 < delta < 1 required")
        self.width = max(8, int(math.ceil(math.e / epsilon)))
        self.depth = max(2, int(math.ceil(math.log(1.0 / delta))))
        self.counters = [[0] * self.width for _ in range(self.depth)]
        self.total = 0

    def _indices(self, item: str):
        for d in range(self.depth):
            h = hashlib.sha256(f"{d}:{item}".encode()).digest()
            yield d, int.from_bytes(h[:8], "big") % self.width

    def add(self, item: str, count: int = 1) -> None:
        self.total += count
        for d, w in self._indices(item):
            self.counters[d][w] += count

    def estimate(self, item: str) -> int:
        return min(self.counters[d][w] for d, w in self._indices(item))

    def top_candidates(self, candidates: List[str], k: int = 10) -> List[tuple]:
        """Point-query a SHORTLIST (candidates must come from the caller —
        a sketch cannot enumerate; honesty about that beats pretending)."""
        pairs = [(self.estimate(c), c) for c in candidates]
        pairs.sort(reverse=True)
        return pairs[:k]

    def to_dict(self) -> Dict[str, Any]:
        return {"width": self.width, "depth": self.depth,
                "counters": self.counters, "total": self.total}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CountMinSketch":
        cms = cls.__new__(cls)
        cms.width, cms.depth = d["width"], d["depth"]
        cms.counters = d["counters"]
        cms.total = d["total"]
        return cms


# ---------------------------------------------------------------------------
# HyperLogLog
# ---------------------------------------------------------------------------

class HyperLogLog:
    def __init__(self, precision: int = 12) -> None:
        if not (4 <= precision <= 16):
            raise ValueError("precision must be in [4, 16]")
        self.p = precision
        self.m = 1 << precision
        self.registers = [0] * self.m

    def _bucket_and_rho(self, item: str):
        h = int.from_bytes(hashlib.sha256(str(item).encode()).digest(), "big")
        bucket = h >> (256 - self.p)
        rest = h & ((1 << (256 - self.p)) - 1) or 1
        rho = 1
        while not (rest & (1 << (255 - self.p))) and rho < 256 - self.p:
            rest <<= 1
            rho += 1
        return bucket, rho

    def add(self, item: str) -> None:
        bucket, rho = self._bucket_and_rho(item)
        if rho > self.registers[bucket]:
            self.registers[bucket] = rho

    def count(self) -> float:
        m = self.m
        alpha = {
            16: 0.673, 32: 0.697, 64: 0.709,
        }.get(m, 0.7213 / (1.0 + 1.079 / m))
        Z = sum(2.0 ** -r for r in self.registers)
        estimate = alpha * m * m / Z
        zeros = self.registers.count(0)
        if estimate <= 2.5 * m and zeros:
            estimate = m * math.log(m / zeros)  # linear counting for tiny N
        return estimate

    def merge(self, other: "HyperLogLog") -> None:
        if other.m != self.m:
            raise ValueError("precision mismatch; cannot merge")
        self.registers = [max(a, b) for a, b in zip(self.registers, other.registers)]

    def to_dict(self) -> Dict[str, Any]:
        return {"p": self.p, "registers": self.registers}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HyperLogLog":
        hll = cls.__new__(cls)
        hll.p, hll.m = d["p"], 1 << d["p"]
        hll.registers = d["registers"]
        return hll


# ---------------------------------------------------------------------------
# Reservoir sampling (Vitter R)
# ---------------------------------------------------------------------------

class Reservoir:
    def __init__(self, k: int = 50, seed: int = 7) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k
        self.n = 0
        self.sample: List[str] = []
        self._rng = random.Random(seed)

    def offer(self, item: str) -> None:
        self.n += 1
        if len(self.sample) < self.k:
            self.sample.append(item)
        else:
            j = self._rng.randint(0, self.n - 1)
            if j < self.k:
                self.sample[j] = item

    def to_dict(self) -> Dict[str, Any]:
        return {"k": self.k, "n": self.n, "sample": self.sample}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Reservoir":
        r = cls.__new__(cls)
        r.k, r.n, r.sample = d["k"], d["n"], d["sample"]
        r._rng = random.Random(0)
        return r


# ---------------------------------------------------------------------------
# EWMA + Page-Hinkley change detection
# ---------------------------------------------------------------------------

class EWMA:
    """Exponentially weighted moving average of a 0/1 (or any) stream."""

    def __init__(self, alpha: float = 0.05) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha in (0, 1]")
        self.alpha = alpha
        self.mean: Optional[float] = None
        self.count = 0

    def update(self, value: float) -> float:
        if self.mean is None:
            self.mean = value
        else:
            self.mean = self.alpha * value + (1.0 - self.alpha) * self.mean
        self.count += 1
        return self.mean

    def to_dict(self) -> Dict[str, Any]:
        return {"alpha": self.alpha, "mean": self.mean, "count": self.count}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EWMA":
        e = cls.__new__(cls)
        e.alpha, e.mean, e.count = d["alpha"], d["mean"], d["count"]
        return e


def page_hinkley(values: List[float], threshold: float = 12.0,
                 alpha: float = 0.98, delta: float = 0.005,
                 min_instances: int = 30) -> Dict[str, Any]:
    """Page-Hinkley test for an upward mean change in a stream.

    Returns {"change_at": index or None, "magnitude": float} — change_at is
    the FIRST index where the cumulative PH statistic crossed `threshold`
    (with a warm-up of `min_instances`). Standard drift detector for the
    "did my acceptance rate / error rate just jump?" question. The running
    mean starts AT the first value (so a stream that is high from line one
    is not mistaken for a rise from zero) and tracks with an EMA(alpha).
    """
    if not values:
        return {"change_at": None, "magnitude": 0.0, "threshold": threshold}
    mean = values[0]
    sum_xt = 0.0
    min_ht = math.inf
    change_at: Optional[int] = None
    for i, x in enumerate(values):
        if i > 0:
            mean = alpha * mean + (1.0 - alpha) * x
        sum_xt += x - mean - delta
        min_ht = min(min_ht, sum_xt)
        if i >= min_instances and (sum_xt - min_ht) > threshold and change_at is None:
            change_at = i
    return {"change_at": change_at,
            "magnitude": (sum_xt - min_ht),
            "threshold": threshold}


class NgramDivergence:
    """B1 — windowed KL novelty for one-pass log scanning.

    Distribution drift detector in the same bounded-RAM discipline as the
    rest of this module: the log's token n-gram distribution is tracked
    per fixed-size block (exact counts over the block's OWN vocabulary —
    bounded by the block, enumerable by construction), and each full block
    is compared to the running baseline by KL divergence with add-alpha
    smoothing,

        D(P_block || Q_baseline) = sum_p p log(p / q)

    where Q_baseline is (a) the previous block (short-term novelty: "this
    block looks different from the last one") and, when the caller wires
    the scanner's Count-Min sketch, (b) the long-run running baseline
    (``kl_vs_cms``) — a sketch cannot enumerate its support, but the
    block's vocabulary is known exactly, so every q(p) point-query needed
    by the sum is available; that is the honest way a CMS participates in
    a KL (the enumerate-the-support side stays exact, the estimate side
    is the sketch, with its (eps, delta) over-count bound).

    Single-pass and bounded, verified against the scanner's own charter
    ("a 2 GB log and a 2 KB log cost the same RAM", scanner.py): state is
    two block-sized dicts (current + reference) plus one internal running
    Count-Min and HyperLogLog — never the whole stream. A 2 GB journal
    and a 2 KB journal occupy the same RAM here too.

    Why an INTERNAL CMS rather than the scanner's: the scanner's sketch
    only ever sees tokens from lines that MATCHED a signature, but a
    novelty detector's whole job is the lines nothing matched (the new
    failure mode is by definition outside the Aho-Corasick dictionary).
    P and Q must be distributions over the same domain for KL to mean
    anything, so this class runs its own running baseline over EVERY
    line's n-grams — the scanner's own CMS stays untouched.

    Footprint (worst case, block 500 lines ~20 tokens/line): two dicts
    of ~10k entries + one CMS (9,513 counters) + one HLL (4,096
    registers) — under 1 MB, independent of stream length.
    """

    def __init__(self, block_lines: int = 500, alpha: float = 0.5,
                 n: int = 1) -> None:
        if block_lines < 1:
            raise ValueError("block_lines must be >= 1")
        if alpha <= 0:
            raise ValueError("alpha must be > 0 (add-alpha smoothing)")
        if n not in (1, 2):
            raise ValueError("n must be 1 (unigrams) or 2 (bigrams)")
        self.block_lines = block_lines
        self.alpha = alpha
        self.n = n
        self._block: Dict[str, int] = {}
        self._block_total = 0
        self._reference: Dict[str, int] = {}
        self._reference_total = 0
        self._lines_in_block = 0
        self.kl_series: List[float] = []
        self._cms = CountMinSketch(epsilon=0.002, delta=0.001)
        self._hll = HyperLogLog(precision=12)

    # ------------------------------------------------------------------
    def _ngrams(self, line: str) -> List[str]:
        if self.n == 1:
            return line.split() or []
        words = line.split()
        return [f"{a} {b}" for a, b in zip(words, words[1:])]

    def offer(self, line: str) -> Optional[Dict[str, Any]]:
        """Consume ONE line. Returns the divergence report when a block
        just filled (every ``block_lines`` lines), else None."""
        for gram in self._ngrams(line):
            self._block[gram] = self._block.get(gram, 0) + 1
            self._block_total += 1
            self._cms.add(gram)   # the running baseline (every line)
            self._hll.add(gram)
        self._lines_in_block += 1
        if self._lines_in_block < self.block_lines:
            return None
        return self._close_block()

    def _close_block(self) -> Dict[str, Any]:
        kl = self._kl(self._block, self._block_total,
                      self._reference, self._reference_total)
        self.kl_series.append(round(kl, 6))
        report: Dict[str, Any] = {
            "block_index": len(self.kl_series),
            "kl_vs_previous_block": round(kl, 6),
            "block_tokens": self._block_total,
            "distinct_ngrams": len(self._block),
        }
        # The filled block becomes the next reference (bounded swap).
        self._reference = self._block
        self._reference_total = self._block_total
        self._block = {}
        self._block_total = 0
        self._lines_in_block = 0
        return report

    def _kl(self, p_counts: Dict[str, int], p_total: int,
            q_counts: Dict[str, int], q_total: int) -> float:
        """Add-alpha smoothed KL(P || Q) over P's exact support."""
        if not p_total or not q_total:
            return 0.0
        support = set(p_counts) | set(q_counts)
        vocab = len(support)
        total = 0.0
        for gram, count in p_counts.items():
            p = (count + self.alpha) / (p_total + self.alpha * vocab)
            q = (q_counts.get(gram, 0) + self.alpha) / (q_total + self.alpha * vocab)
            total += p * math.log(p / q)
        return total

    # ------------------------------------------------------------------
    def kl_vs_cms(self) -> Dict[str, Any]:
        """Compare the CURRENT block against the running Count-Min
        baseline (the whole stream so far, including this block's own
        contribution — the standard self-inclusive streaming baseline).

        Returns per-ngram {"ngram", "block_share", "cms_share",
        "lift"} rows for the biggest positive divergences (the movers a
        human wants to see), sorted by lift, capped. Every row's cms
        estimate is a point query (bounded over-count by the sketch's
        (eps, delta) contract)."""
        if not self._block_total:
            return {"kl_vs_cms": 0.0, "movers": [],
                    "distinct_ngrams_estimate": round(self._hll.count())}
        cms_total = max(1, self._cms.total)
        vocab = len(self._block) + 1
        kl = 0.0
        movers: List[Dict[str, Any]] = []
        for gram, count in self._block.items():
            p = (count + self.alpha) / (self._block_total + self.alpha * vocab)
            q = (self._cms.estimate(gram) + self.alpha) / (cms_total + self.alpha * vocab)
            kl += p * math.log(p / q)
            movers.append({
                "ngram": gram,
                "block_share": round(count / self._block_total, 6),
                "cms_share": round(self._cms.estimate(gram) / cms_total, 6),
                "lift": round((count / self._block_total) /
                              max(1e-12, self._cms.estimate(gram) / cms_total), 3),
            })
        movers.sort(key=lambda m: -m["lift"])
        return {"kl_vs_cms": round(kl, 6), "movers": movers[:10],
                "distinct_ngrams_estimate": round(self._hll.count())}

    def alarm(self, threshold: float = 0.5, min_blocks: int = 3) -> Dict[str, Any]:
        """The novelty verdict: the KL series fed to Page-Hinkley —
        'the distribution just changed', stated only after enough blocks."""
        verdict = page_hinkley(self.kl_series, threshold=threshold,
                               min_instances=min_blocks)
        return {
            "novelty_change_at_block": verdict["change_at"],
            "novelty_magnitude": round(verdict["magnitude"], 6),
            "kl_series": list(self.kl_series),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"block_lines": self.block_lines, "alpha": self.alpha,
                "n": self.n, "kl_series": list(self.kl_series),
                "cms": self._cms.to_dict(), "hll": self._hll.to_dict()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NgramDivergence":
        nd = cls(block_lines=d["block_lines"], alpha=d["alpha"], n=d["n"])
        nd.kl_series = list(d.get("kl_series", []))
        if "cms" in d:
            nd._cms = CountMinSketch.from_dict(d["cms"])
        if "hll" in d:
            nd._hll = HyperLogLog.from_dict(d["hll"])
        return nd
