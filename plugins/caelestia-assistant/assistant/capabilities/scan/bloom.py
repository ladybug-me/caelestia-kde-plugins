"""scan.bloom — a Bloom filter over hashlib (double hashing, Kirsch-Mitzenmacher).

"No false negatives, bounded false positives": if `__contains__` says a line
was never seen, that is CERTAIN; if it says maybe-seen, the false-positive
rate is at most (1 - e^(-kn/m))^k — for the defaults here (1% target) that is
the expected order. Used by the scanner to deduplicate repeated error lines
in a journal without storing the lines themselves.

Memory is m bits in a bytearray — for 1M entries at 1% FPR that is ~1.2 MB,
constant regardless of how long the entries are.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict

__all__ = ["BloomFilter"]


class BloomFilter:
    def __init__(self, capacity: int = 100_000, error_rate: float = 0.01) -> None:
        if capacity <= 0 or not (0.0 < error_rate < 1.0):
            raise ValueError("capacity > 0 and 0 < error_rate < 1 required")
        m = -capacity * math.log(error_rate) / (math.log(2.0) ** 2)
        self.m = max(8, int(math.ceil(m)))
        self.k = max(1, int(round((self.m / capacity) * math.log(2.0))))
        self.bits = bytearray((self.m + 7) // 8)
        self.n = 0

    # ------------------------------------------------------------------
    def _hashes(self, item: str):
        h1 = int.from_bytes(hashlib.sha256(item.encode("utf-8")).digest()[:8], "big")
        h2 = int.from_bytes(hashlib.md5(item.encode("utf-8")).digest()[:8], "big")
        return ((h1 + i * h2) % self.m for i in range(1, self.k + 1))

    def _setbit(self, idx: int) -> None:
        self.bits[idx >> 3] |= 1 << (idx & 7)

    def _getbit(self, idx: int) -> int:
        return (self.bits[idx >> 3] >> (idx & 7)) & 1

    # ------------------------------------------------------------------
    def add(self, item: str) -> None:
        for h in self._hashes(item):
            self._setbit(h)
        self.n += 1

    def __contains__(self, item: str) -> bool:
        return all(self._getbit(h) for h in self._hashes(item))

    def add_if_absent(self, item: str) -> bool:
        """Add and report whether it was (probably) new — the dedup fast path."""
        if item in self:
            return False
        self.add(item)
        return True

    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {"m": self.m, "k": self.k, "n": self.n,
                "bits": self.bits.hex()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BloomFilter":
        bf = cls.__new__(cls)
        bf.m, bf.k, bf.n = d["m"], d["k"], d["n"]
        bf.bits = bytearray.fromhex(d["bits"])
        return bf
