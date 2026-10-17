"""scan.ac — Aho-Corasick multi-pattern matcher.

Builds the goto/fail/output automaton ONCE over every known signature
(keyword anchors) and then scans any amount of text in a single pass:
O(len(text) + matches) INDEPENDENT of the number of patterns. This is the
algorithmic difference between "scan my 2 GB journal against the 500 known
caelestia signatures" being impossible and being trivial.

The implementation is the standard dictionary-automaton:

  1. goto:    trie over the patterns
  2. fail:    BFS-computed failure links (longest proper suffix that is also
              a pattern prefix)
  3. output:  per-state pattern ids, plus dictionary-output chaining so a
              single position reports every pattern ending there

Deterministic; matches reported as (end_index, pattern_id). Also exposes
`first_match` for hot loops.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple

__all__ = ["Automaton"]

_FAILURE = "\0__fail__"  # sentinel key outside the byte/alphabet space


class Automaton:
    """Aho-Corasick automaton over literal patterns (bytes/str, casefolded
    if built with ignore_case=True)."""

    def __init__(self, patterns: Iterable[str], ignore_case: bool = True) -> None:
        self.ignore_case = ignore_case
        self._patterns: List[str] = []
        self._goto: List[Dict[str, int]] = [{}]
        self._fail: List[int] = [0]
        self._out: List[List[int]] = [[]]
        for pat in patterns:
            self.add(pat)
        self._build()

    # ------------------------------------------------------------------
    def add(self, pattern: str) -> int:
        """Add one pattern (call before `finalize`/matching; the automaton
        rebuilds lazily). Returns the pattern id."""
        if self.ignore_case:
            pattern = pattern.casefold()
        pid = len(self._patterns)
        self._patterns.append(pattern)
        state = 0
        for ch in pattern:
            nxt = self._goto[state].get(ch)
            if nxt is None:
                self._goto.append({})
                self._fail.append(0)
                self._out.append([])
                nxt = len(self._goto) - 1
                self._goto[state][ch] = nxt
            state = nxt
        self._out[state].append(pid)
        self._dirty = True
        return pid

    # ------------------------------------------------------------------
    def _build(self) -> None:
        """BFS the trie, wiring failure links and merging dictionary outputs."""
        queue: deque = deque()
        for ch, s in self._goto[0].items():
            self._fail[s] = 0
            queue.append(s)
        while queue:
            r = queue.popleft()
            for ch, s in self._goto[r].items():
                queue.append(s)
                f = self._fail[r]
                while f and ch not in self._goto[f]:
                    f = self._fail[f]
                self._fail[s] = self._goto[f].get(ch, 0)
                if self._fail[s] == s:
                    self._fail[s] = 0
                self._out[s] = self._out[s] + self._out[self._fail[s]]
        self._dirty = False

    def _ensure_built(self) -> None:
        if getattr(self, "_dirty", False):
            self._build()

    # ------------------------------------------------------------------
    def scan(self, text: str) -> List[Tuple[int, int]]:
        """All matches as (end_index_exclusive, pattern_id), non-overlapping
        per position but exhaustive across the dictionary-output chain."""
        self._ensure_built()
        if self.ignore_case:
            text = text.casefold()
        hits: List[Tuple[int, int]] = []
        state = 0
        for i, ch in enumerate(text, 1):
            while state and ch not in self._goto[state]:
                state = self._fail[state]
            state = self._goto[state].get(ch, 0)
            for pid in self._out[state]:
                hits.append((i, pid))
        return hits

    def first_match(self, text: str) -> Optional[Tuple[int, int]]:
        self._ensure_built()
        if self.ignore_case:
            text = text.casefold()
        state = 0
        for i, ch in enumerate(text, 1):
            while state and ch not in self._goto[state]:
                state = self._fail[state]
            state = self._goto[state].get(ch, 0)
            if self._out[state]:
                return (i, self._out[state][0])
        return None

    def pattern(self, pid: int) -> str:
        return self._patterns[pid]

    def count(self) -> int:
        return len(self._patterns)
