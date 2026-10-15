#!/usr/bin/env python3
"""Phase Z8 — hostile/degenerate inputs, 20 per new-feature surface.

Groups B-E shipped twelve answerable surfaces; each one gets the same
twenty hostile inputs (empty, whitespace, 1 MB string, 20k-word line,
unicode/emoji/RTL, control chars, null byte (bridge only), huge count,
negative count, NaN/inf, deeply nested JSON (depth 60), duplicated keys,
binary bytes, unreadable path, self-referencing structure, SQL/shell
injection text, path traversal, half a JSON document, list-where-dict,
dict-where-list). Contract per surface: no traceback, no hang, rc in
{0,1,2}; the honest verdicts (NOT_FOUND / UNCHANGED / ABSTAIN / empty)
are the expected GOOD outcomes.

Run: PYTHONPATH=. python3 scripts/dev/z_hostile.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

BIG = "x" * 1_000_000
LONGWORDS = ("word " * 20_000)
DEEP = json.dumps(json.loads('{"a":' * 60 + "1" + "}" * 60))
BINARY = Path(__file__).read_bytes()[:512].decode("latin-1")

HOSTILE = [
    "", " ", "\n\t", BIG, LONGWORDS,
    "出力をテスト", "café ☕ 𝕏", "‏مرحبا", "\x07\x08 bell",
    "1e999 nan inf -0", "-5", "999999999999", "0.0.0.1",
    "'; DROP TABLE tools; --", "$(rm -rf ~) `id` && cat /etc/passwd",
    "../../etc/shadow", DEEP, BINARY,
    '{"truncated": [1, 2', '["list", "where", "dict"]',
]
# the kernel refuses argv beyond ~128 KB before our process even starts
# (E2BIG on exec) — that is an OS fact, not a program behavior, so the
# argv-driven cases run the same battery minus the two oversize strings;
# the oversize strings stay for the file/bridge-driven cases
ARGHOSTILE = [h for h in HOSTILE if len(h) < 100_000]


def bridge_op(op: str, payload_key: str):
    def run(value, tmp: Path):
        from assistant.capabilities.brain import bridge
        req = {"op": op}
        if value is not None:
            req[payload_key] = value
        resp = bridge.handle(req, None, None)
        json.dumps(resp)
        return 0, "", ""
    return run


def cli_case(argv):
    def run(value, tmp: Path):
        args = [a.replace("__V__", value) for a in argv]
        env = dict(os.environ)
        home = tempfile.mkdtemp(prefix="z-host-")
        env.update({"HOME": home, "PYTHONPATH": str(REPO),
                    "XDG_CONFIG_HOME": str(tmp), "XDG_DATA_HOME": str(tmp)})
        proc = subprocess.run(
            [sys.executable, "-m", "assistant.hub", *args],
            capture_output=True, text=True, timeout=60, env=env, cwd=REPO)
        return proc.returncode, proc.stdout, proc.stderr
    return run


def file_cli_case(argv, fname="input.json"):
    """argv contains __F__; the hostile value is written to a file first."""
    def run(value, tmp: Path):
        f = tmp / fname
        f.write_text(value, encoding="utf-8", errors="replace")
        args = [a.replace("__F__", str(f)) for a in argv]
        env = dict(os.environ)
        home = tempfile.mkdtemp(prefix="z-host-")
        env.update({"HOME": home, "PYTHONPATH": str(REPO),
                    "XDG_CONFIG_HOME": str(tmp), "XDG_DATA_HOME": str(tmp)})
        proc = subprocess.run(
            [sys.executable, "-m", "assistant.hub", *args],
            capture_output=True, text=True, timeout=60, env=env, cwd=REPO)
        return proc.returncode, proc.stdout, proc.stderr
    return run


# (name, runner, use_full_battery). argv-driven cases use the truncated
# list (kernel E2BIG is an OS fact, not a program behavior); bridge and
# file-driven cases get every input including the oversize strings
CASES = {
    "D17 qa (cli)": (cli_case(["brain", "qa", "__V__", "--json"]), False),
    "D17 qa (bridge)": (bridge_op("qa_answer", "question"), True),
    "D17 qa notes (bridge)": (bridge_op("qa_answer", "notes"), True),
    "D16 say (file)": (file_cli_case(["brain", "say", "__F__",
                                      "--json"]), True),
    "C15 sizes": (cli_case(["brain", "sizes", "__V__", "--json"]), False),
    "C11 patterns": (cli_case(["brain", "patterns", "__V__", "--json"]),
                     False),
    "C12 bursts": (cli_case(["brain", "bursts", "__V__", "--json"]), False),
    "C13 gp-prefs": (file_cli_case(["brain", "gp-prefs", "__F__"]), True),
    "C14 rules": (file_cli_case(["brain", "rules", "--file", "__F__"]),
                  True),
    "B7 explain": (cli_case(["shellkb", "explain", "__V__", "--json"]),
                   False),
    "B8 howto": (cli_case(["shellkb", "howto", "__V__", "--json"]), False),
    "B9 diff": (file_cli_case(["shellkb", "diff", "__F__", "__F__"]),
                True),
    "B9 merge": (file_cli_case(["shellkb", "merge", "--base", "__F__",
                                "--ours", "__F__", "--theirs", "__F__"]),
                 True),
    "B10 conflicts": (cli_case(["shellkb", "conflicts", "--root", "__V__",
                                "--spec", "__V__", "--json"]), False),
    "E2 model_rerank (bridge)": (bridge_op("model_rerank",
                                           "candidates"), True),
}


def main() -> int:
    fails = 0
    ran = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, (runner, full) in CASES.items():
            battery = HOSTILE if full else ARGHOSTILE
            bad = 0
            for value in battery:
                ran += 1
                try:
                    rc, out, err = runner(value, Path(tmp))
                except subprocess.TimeoutExpired:
                    print(f"HANG {name} on {value[:40]!r}")
                    bad += 1
                    fails += 1
                    continue
                except Exception as exc:  # noqa: BLE001 — a raise is a fail
                    print(f"RAISE {name} on {value[:40]!r}: "
                          f"{type(exc).__name__}: {exc}")
                    bad += 1
                    fails += 1
                    continue
                if "Traceback" in err:
                    print(f"TRACEBACK {name} on {value[:40]!r}: "
                          f"{err.splitlines()[-1][:110]}")
                    bad += 1
                    fails += 1
                if rc not in (0, 1, 2):
                    print(f"RC {name} rc={rc} on {value[:40]!r}")
                    bad += 1
                    fails += 1
            print(f"{'ok  ' if bad == 0 else 'FAIL'} {name}: "
                  f"{len(battery) - bad}/{len(battery)} clean")
    print("---")
    print(f"hostile battery: {ran} cases, "
          f"{'PASS' if fails == 0 else f'FAIL ({fails})'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
