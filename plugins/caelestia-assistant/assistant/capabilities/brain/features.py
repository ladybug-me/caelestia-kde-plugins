"""brain.features — the shared feature-hashing scheme (phase 2.4).

The hashing trick (Weinberger, Daslambert, Smola & Atreya 2009, "Feature
hashing for large scale multitask learning", ICML): tokens are mapped
into a fixed d-dimensional space by a seeded hash, with signed
collision resolution — the projection that lets many learners share ONE
feature function without sharing state.

Why this module exists: the router (cortex), the genius meta router,
and the preset/plan learners each keep SEPARATELY AUDITABLE state
files — that boundary is deliberate (the package rationale's per-learner
reviewability) — but they can still TRANSFER signal on structurally
similar inputs by agreeing on the same feature mapping. Share the
projection, never the state.

Guarantees:
- PURE: no I/O, no randomness at call time (the hash is deterministic
  and seeded); identical input yields identical features, always.
- stdlib only (hashlib); d is bounded (callers use 8..64).
- Signed hashing: the sign comes from a second hash bit, halving the
  effective collision bias (the 2009 paper's own trick).
"""

from __future__ import annotations

import hashlib
from typing import Iterable, List, Tuple

__all__ = ["hash_features", "DEFAULT_DIMENSIONS", "DEFAULT_SEED",
           "text_features", "context_features"]

DEFAULT_DIMENSIONS = 16
DEFAULT_SEED = 120


def _digest(seed: int, token: str) -> Tuple[int, int]:
    """(bucket, sign) for one token: a stable 64-bit hash split into a
    bucket index bit-source and one sign bit."""
    payload = f"{seed}:{token}".encode("utf-8", "replace")
    full = int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return full >> 1, 1 if (full & 1) else -1


def hash_features(tokens: Iterable[str], d: int = DEFAULT_DIMENSIONS,
                  seed: int = DEFAULT_SEED) -> List[float]:
    """Project tokens into a dense d-vector by signed feature hashing.

    Each token adds +sign to its bucket; the result is L2-normalized so
    every consumer sees the same scale regardless of text length.
    Deterministic, pure, O(len(tokens)).
    """
    if d < 1:
        raise ValueError("d must be >= 1")
    vec = [0.0] * d
    for token in tokens:
        if not token:
            continue
        value, sign = _digest(seed, token)
        vec[value % d] += sign
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [round(v / norm, 6) for v in vec]
    return vec


def text_features(text: str, d: int = DEFAULT_DIMENSIONS,
                  seed: int = DEFAULT_SEED) -> List[float]:
    """Word tokens of a request hashed into the shared space (lowercased,
    alphanumeric runs — the tokenizer every layer already agrees on)."""
    import re

    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return hash_features(tokens, d=d, seed=seed)


def context_features(text: str, hour: int = -1,
                     d: int = DEFAULT_DIMENSIONS,
                     seed: int = DEFAULT_SEED) -> List[float]:
    """Request text + coarse context atoms (hour bucket) in the shared
    space — the contextual bandit's input shape: structure transfers
    across learners while each learner keeps its own weights."""
    import re

    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if 0 <= hour <= 23:
        tokens.append(f"hour={hour // 3}")  # 8 buckets of 3h
    return hash_features(tokens, d=d, seed=seed)
