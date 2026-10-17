"""scan.simhash — SimHash fingerprints for near-duplicate text detection.

A 64-bit fingerprint per text; two texts are NEAR DUPLICATES when their
fingerprints differ in at most ``threshold`` bit positions (Hamming
distance). This is the classical web-scale near-duplicate construction:
SimHash itself is Charikar's rounding technique ("Similarity Estimation
Techniques from Rounding Algorithms", STOC 2002), and its use for
near-duplicate detection at scale — 64-bit fingerprints, Hamming distance
<= 3 — is Manku, Jain & Das Sarma ("Detecting Near-Duplicates for Web
Crawling", WWW 2007).

WHY THIS MODULE (and not an existing one) — the non-duplication note:

- ``scan.bloom.BloomFilter`` answers EXACT-seen ("have I seen this exact
  line?"), with no notion of similarity; a repeated line with a new PID
  or timestamp is a new line to it.
- ``brain/minhash.py`` (MinHash + LSH banding) estimates Jaccard over
  shingle sets for the PERSONAL-NOTES module, storing a 64-permutation
  signature per document. Different layer, different data, heavier
  per-document state.
- SimHash here is the stream-side answer: ONE machine word per text —
  64x smaller than a MinHash signature — cheap enough to keep a
  fingerprint for every line of a bounded window, which is exactly what
  a bounded-memory log/clipboard scan needs. scan/ had no near-duplicate
  structure at all before this module.

Construction (deterministic, stdlib only): tokens are lowercase word
runs with their counts as weights (a text with fewer than two word
tokens falls back to character 3-grams, so short/garbled lines still
fingerprint stably); each token is hashed with blake2b (64-bit digest)
and its weight is added to / subtracted from per-bit accumulators; the
fingerprint bit is the sign of each accumulator. Order-independent in
the tokens (a set-similarity signal, like Jaccard over token multisets),
sensitive to token-mass shifts, and stable across rewording of
connective text.

HONEST OPERATING-POINT NOTE (measured, 2026-09-26): Hamming <= 3 is
Manku et al.'s operating point for LONG documents, where a one-word
edit barely moves the fingerprint. A log line has too little feature
mass for that: swapping "pid 1234, attempt 1" for "pid 9183, attempt 2"
measures ~15 bits apart on the raw text. The domain-correct fix is the
one journal dedup tools use — mask the volatile fields first:
``mask_digits=True`` replaces every digit run with ``#`` before
tokenizing, and those same two lines then fingerprint IDENTICALLY
(distance 0). ``NearDupTracker`` masks digits by default (its charter
is log/clipboard streams); ``simhash()`` defaults to the raw text and
takes the flag explicitly. Very short texts (< ~4 word tokens) carry
too little mass for ANY small threshold — mask or accept exact-match
semantics only.

Deferred with a measured reason: perceptual-hash-for-images. The cost
is NOT the blocker — measured on this machine (2026-09-26), a stdlib
pHash of a 1920x1080 PNG costs ~0.02 s (zlib IDAT decode ~0.01 s,
32x32 luminance sample ~0.000 s, naive 32x32 DCT ~0.013 s). The blocker
is format coverage: wallpapers in the wild are heavily JPEG, and the
stdlib-only dependency set provides no JPEG
decoder, so a pHash here would silently cover only PNG — a partial
feature pretending to be a general one. If image near-dup matters
later, revisit with the coverage question answered first.
"""

from __future__ import annotations

import hashlib
import re
from collections import deque
from typing import Any, Dict, List, Optional

__all__ = ["simhash", "hamming", "near_duplicate", "NearDupTracker",
           "mask_digits"]

_WORD_RE = re.compile(r"\w+", re.UNICODE)
# Volatile-field normalization for log-style text: every digit run
# (timestamps, PIDs, counters) collapses to '#' — the same idea journal
# dedup tools use, applied BEFORE tokenization.
_DIGIT_RE = re.compile(r"\d+")

DEFAULT_BITS = 64
DEFAULT_THRESHOLD = 3  # Manku et al.'s operating point for 64-bit fingerprints


def mask_digits(text: str) -> str:
    """Replace every digit run with '#' (volatile-field normalization)."""
    return _DIGIT_RE.sub("#", text)


def _tokens(text: str, masked: bool) -> List[str]:
    """Lowercase word tokens; character 3-gram fallback for texts with
    fewer than two word tokens (short or non-word lines still fingerprint)."""
    if masked:
        text = mask_digits(text)
    words = _WORD_RE.findall(text.lower())
    if len(words) >= 2:
        return words
    compact = re.sub(r"\s+", "", text.lower())
    if len(compact) < 3:
        return [compact] if compact else []
    return [compact[i:i + 3] for i in range(len(compact) - 2)]


def simhash(text: str, bits: int = DEFAULT_BITS,
            mask_digits: bool = False) -> int:
    """The SimHash fingerprint of ``text`` as an int with ``bits`` bits
    (64 by default — one machine word). ``mask_digits=True`` normalizes
    volatile numeric fields away first (see the module's operating-point
    note). Deterministic: a pure function of the text, no seeding, no
    state."""
    if bits < 8 or bits > 64:
        raise ValueError("bits must be in [8, 64] (one word's worth)")
    counts: Dict[str, int] = {}
    for token in _tokens(text, mask_digits):
        counts[token] = counts.get(token, 0) + 1
    if not counts:
        return 0
    digest_size = 8  # 64 bits of blake2b per token, exactly one block
    accum = [0] * bits
    for token, weight in counts.items():
        h = int.from_bytes(
            hashlib.blake2b(token.encode("utf-8", "replace"),
                            digest_size=digest_size).digest(), "big")
        for i in range(bits):
            accum[i] += weight if (h >> i) & 1 else -weight
    out = 0
    for i in range(bits):
        if accum[i] > 0:
            out |= 1 << i
    return out


def hamming(a: int, b: int) -> int:
    """Bit positions in which the two fingerprints differ."""
    return bin(a ^ b).count("1")


def near_duplicate(a: int, b: int, threshold: int = DEFAULT_THRESHOLD) -> bool:
    """Manku et al.'s operating rule: near-duplicate iff Hamming distance
    is at most ``threshold`` (default 3 of 64 bits)."""
    return hamming(a, b) <= threshold


class NearDupTracker:
    """A bounded-memory near-duplicate detector over a stream of texts.

    Keeps the last ``window`` fingerprints (one machine word each) and
    reports, for each observed text, whether it near-duplicates something
    already in the window — the question the Bloom filter cannot answer
    ("have I seen approximately this line before?") at the same order of
    memory cost. JSON round-trippable like every other scan structure, so
    a scan can be paused, persisted and resumed.

    A text that exactly repeats something in the window counts as a
    near-duplicate too (distance 0).
    """

    def __init__(self, window: int = 500, threshold: int = DEFAULT_THRESHOLD,
                 bits: int = DEFAULT_BITS, mask_digits: bool = True) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        if threshold < 0 or threshold > bits:
            raise ValueError("threshold must be in [0, bits]")
        self.window = int(window)
        self.threshold = int(threshold)
        self.bits = int(bits)
        self.mask_digits = bool(mask_digits)
        self._ring: deque = deque(maxlen=self.window)
        self.seen = 0
        self.near_dups = 0

    def observe(self, text: str) -> Optional[int]:
        """Fingerprint one text; returns the Hamming distance to its
        nearest neighbor in the window if it is a near duplicate, else
        None. O(window) bit-popcounts per line — 500 popcounts of one
        machine word is the budgeted bound, no full re-comparisons.
        Digit masking is ON by default: log lines whose only difference
        is a PID/timestamp/count are exactly the stream case this
        exists for."""
        fp = simhash(text, self.bits, self.mask_digits)
        self.seen += 1
        best: Optional[int] = None
        for other in self._ring:
            d = hamming(fp, other)
            if d <= self.threshold and (best is None or d < best):
                best = d
        self._ring.append(fp)
        if best is not None:
            self.near_dups += 1
        return best

    def to_dict(self) -> Dict[str, Any]:
        return {"window": self.window, "threshold": self.threshold,
                "bits": self.bits, "mask_digits": self.mask_digits,
                "seen": self.seen, "near_dups": self.near_dups,
                "fingerprints": list(self._ring)}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "NearDupTracker":
        tracker = NearDupTracker(int(data.get("window", 500)),
                                int(data.get("threshold", DEFAULT_THRESHOLD)),
                                int(data.get("bits", DEFAULT_BITS)),
                                bool(data.get("mask_digits", True)))
        tracker._ring.extend(int(fp) for fp in data.get("fingerprints", []))
        tracker.seen = int(data.get("seen", 0))
        tracker.near_dups = int(data.get("near_dups", 0))
        return tracker
