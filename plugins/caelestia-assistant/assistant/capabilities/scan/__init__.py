"""assistant.capabilities.scan — bounded-memory stream intelligence for huge logs.

The low-end-device story: a 2 GB journalctl dump cannot be regex-scanned
rule-by-rule (O(rules x lines) and a pile of Python re machinery) and cannot
be loaded into RAM. This package answers the stream questions with the
classical probabilistic data structures — all stdlib, all with proven error
bounds, all JSON-serialisable so a scan can be paused, persisted, resumed:

| Module | Structures | Question it answers |
| --- | --- | --- |
| `ac` | Aho-Corasick automaton (goto/fail/output) | Which of my 500 known error signatures appear anywhere in this stream, in ONE pass, at O(lines) regardless of rule count? |
| `bloom` | Bloom filter (double hashing over hashlib) | Have I seen this exact error line before? (no false negatives, tunable false-positive rate) |
| `sketch` | Count-Min Sketch | Approximate token frequencies under a fixed memory cap, with the epsilon bound |
| `sketch` | HyperLogLog (flajolet) | How many DISTINCT error lines/hosts/units in the stream, ~1-2% typical error? |
| `sketch` | Reservoir sampler | A uniform random sample of k matching lines from a stream of unknown length (Vitter R) |
| `sketch` | EWMA + Page-Hinkley | Is the error rate stable, or has something just changed? (change-point alarm) |

`scanner.scan_stream` composes all of them: one pass over the lines, a
summary dict out. Nothing here executes anything, reads anything but the
caller-provided line iterable, or needs the whole file in memory.
"""
from .ac import Automaton
from .bloom import BloomFilter
from .simhash import NearDupTracker, hamming, near_duplicate, simhash
from .sketch import (CountMinSketch, HyperLogLog, Reservoir, EWMA,
                     page_hinkley)
from .scanner import scan_stream, scan_text, render

__all__ = [
    "Automaton", "BloomFilter", "CountMinSketch", "HyperLogLog", "Reservoir",
    "EWMA", "page_hinkley", "scan_stream", "scan_text", "render",
    "simhash", "hamming", "near_duplicate", "NearDupTracker",
]
