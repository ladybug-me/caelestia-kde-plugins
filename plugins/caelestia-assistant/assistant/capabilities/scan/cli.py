"""scan.cli — command surface for the stream scanner.

    python3 -m assistant.capabilities.scan FILE [--pattern "quickshell crashed"] [--json]
    python3 -m assistant.capabilities.scan --demo                 # synthetic stream, no file
    cat journal.log | python3 -m assistant.capabilities.scan

Read-only over the caller-provided stream; nothing executes, nothing leaves
the machine. The default pattern set ships the well-known caelestia/Quickshell
anchors so `caelestia-assist scan journal.log` is useful with zero setup.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from .ac import Automaton
from .scanner import render, scan_stream

DEFAULT_PATTERNS: List[str] = [
    "quickshell has crashed",
    "quickshell",
    "pragma",
    "signal handoff",
    "sigsegv",
    "segfault",
    "cannot open",
    "no such file or directory",
    "permission denied",
    "failed to",
    "error",
    "critical",
    "journal",
    "vesktop",
    "gamescope",
    "pacman",
    "wayland",
    "kwin",
    "dbus",
    "timeout",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="assistant.capabilities.scan",
        description="One-pass bounded-memory log scanner (Aho-Corasick + "
                    "Bloom + Count-Min + HyperLogLog + reservoir + Page-Hinkley)")
    p.add_argument("file", nargs="?", help="log file to scan ('-' or omitted = stdin)")
    p.add_argument("--pattern", action="append", default=[],
                   help="extra literal pattern to track (repeatable)")
    p.add_argument("--chunk", type=int, default=500, help="lines per rate chunk")
    p.add_argument("--json", action="store_true", help="machine output")
    p.add_argument("--demo", action="store_true",
                   help="scan a synthetic demo stream instead of a file")
    return p


def _demo_lines() -> List[str]:
    lines: List[str] = []
    for i in range(2000):
        if i % 97 == 0:
            lines.append(f"sep 25 10:{i % 60:02d} quickshell[666]: error: QML Quickshell: something (demo)")
        elif i % 211 == 0:
            lines.append("kwin_wayland: dbus call failed: timeout")
        else:
            lines.append(f"systemd[1]: normal demo line {i} doing nothing interesting")
    # a mid-stream burst so Page-Hinkley has something honest to find
    for i in range(300):
        lines.insert(1200, "quickshell has crashed: signal handoff (demo burst)")
    return lines


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    patterns = list(DEFAULT_PATTERNS) + list(args.pattern)
    auto = Automaton(patterns)

    if args.demo:
        lines: List[str] = _demo_lines()
    else:
        if args.file and args.file != "-":
            with open(args.file, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        else:
            lines = sys.stdin.readlines()

    summary = scan_stream(lines, auto, chunk_size=max(10, args.chunk))
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(render(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
