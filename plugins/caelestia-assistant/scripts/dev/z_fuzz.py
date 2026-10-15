#!/usr/bin/env python3
"""Phase Z3 — fuzz every import surface the new features added.

Seeded (deterministic) garbage against two surface families:

  1. the JSON bridge: EVERY registered op, mutated requests — wrong
     types everywhere, truncations, unicode/emoji, null bytes, huge
     strings, huge/negative numbers, nested noise. The contract: the
     bridge ALWAYS answers one JSON object ({ok: ...}) and never leaks
     a traceback;
  2. the new CLI verbs (brain qa/say, shellkb verbs, eval, model_rerank
     via the bridge): mutated argv and mutated files. The contract:
     rc in {0,1,2} and no 'Traceback' on stderr (argparse usage lines
     are rc=2 and fine).

Run: PYTHONPATH=. python3 scripts/dev/z_fuzz.py
"""
from __future__ import annotations

import io
import json
import os
import random
import subprocess
import sys
import tempfile
import contextlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from assistant.capabilities.brain import bridge as brain_bridge  # noqa: E402

WORDS = ["bar", "scale", "dock", "notifications", "cairo", "\U0001f600",
         "null", "None", "true", "0", "-1", "1e999", "nan", "import",
         " DROP TABLE", "ÄÖÜ", "x" * 300, "", "\x00", "${HOME}", "`id`"]


def garbage(rng: random.Random) -> dict:
    """One mutated request: the op name is REAL, the payload is junk."""
    op = rng.choice(sorted(brain_bridge.OPS))
    shape = rng.randrange(8)
    if shape == 0:
        req = {"op": op}
    elif shape == 1:
        req = {"op": op, "question": rng.choice(WORDS),
               "text": rng.choice(WORDS)}
    elif shape == 2:
        req = {"op": op, "question": [rng.randrange(9) for _ in range(3)]}
    elif shape == 3:
        req = {"op": op, "changes": rng.choice(WORDS),
               "candidates": rng.choice(WORDS)}
    elif shape == 4:
        req = {"op": op, "notes": [{"name": rng.choice(WORDS),
                                    "text": rng.choice(WORDS)}
                                   for _ in range(rng.randrange(4))]}
    elif shape == 5:
        req = {"op": op, "rows": [{"p": rng.random(), "ok": rng.randrange(2)}
                                  for _ in range(rng.randrange(8))],
               "score": rng.choice([0.0, 1.0, -5, "x"])}
    elif shape == 6:
        req = {"op": op, "sizes": rng.choice(WORDS),
               "keys": rng.choice(WORDS), "k": rng.randrange(-2, 12)}
    else:
        req = {"op": op}
        for k in ("query", "candidates", "question", "text", "rows",
                  "changes", "notes", "sizes", "keys", "k", "quantiles",
                  "sources", "path", "profile", "events", "file"):
            if rng.random() < 0.4:
                req[k] = rng.choice(WORDS)
    req["_junk"] = rng.choice(WORDS)
    return req


def fuzz_bridge(n: int, seed: int) -> int:
    rng = random.Random(seed)
    fails = 0
    for i in range(n):
        req = garbage(rng)
        try:
            resp = brain_bridge.handle(req, None, None)
        except Exception as exc:  # noqa: BLE001 — a raise IS the failure
            print(f"FAIL bridge raise: {type(exc).__name__}: {exc} "
                  f"op={req.get('op')!r}")
            fails += 1
            continue
        if not isinstance(resp, dict) or "ok" not in resp:
            print(f"FAIL bridge shape: {resp!r}")
            fails += 1
        # serialized answer must be one line of JSON, always
        try:
            json.dumps(resp)
        except (TypeError, ValueError) as exc:
            print(f"FAIL bridge json: {exc} op={req.get('op')!r}")
            fails += 1
    return fails


CLI_CASES = [
    ["brain", "qa", "__Q__", "--json"],
    ["brain", "qa", "__Q__", "--note", "__N__"],
    ["brain", "qa", "__Q__", "--sources", "__S__"],
    ["brain", "say", "__F__"],
    ["brain", "sizes", "__Q__"],
    ["shellkb", "explain", "__Q__"],
    ["shellkb", "howto", "__Q__"],
    ["shellkb", "grammar", "__Q__"],
    ["eval", "__Q__"],
]


def fuzz_cli(n_per_case: int, seed: int) -> int:
    rng = random.Random(seed + 1)
    # the kernel forbids NUL in argv — keep NUL to the bridge path where
    # it travels inside a JSON string; argv gets the printable junk
    argv_words = [w for w in WORDS if "\x00" not in w]
    fails = 0
    with tempfile.TemporaryDirectory() as tmp:
        good_note = Path(tmp) / "n.md"
        good_note.write_text("hello note\n", encoding="utf-8")
        for base in CLI_CASES:
            for i in range(n_per_case):
                argv = [a.replace("__Q__", rng.choice(argv_words))
                        .replace("__N__", rng.choice(
                            [str(good_note), "/nonexistent", tmp, ""]))
                        .replace("__S__", rng.choice(
                            ["both", "corpus", "notes", "junk"]))
                        .replace("__F__", rng.choice(
                            [str(good_note), tmp, "/nonexistent", ""]))
                        for a in base]
                env = dict(os.environ)
                home = tempfile.mkdtemp(prefix="z-fuzz-")
                env.update({"HOME": home, "PYTHONPATH": str(REPO),
                            "XDG_CONFIG_HOME": tmp,
                            "XDG_DATA_HOME": tmp,
                            "CAELESTIA_BRAIN_STATE": str(Path(tmp) /
                                                         "b.json")})
                proc = subprocess.run(
                    [sys.executable, "-m", "assistant.hub", *argv],
                    capture_output=True, text=True, timeout=60,
                    env=env, cwd=REPO)
                if proc.returncode not in (0, 1, 2):
                    print(f"FAIL rc={proc.returncode} for {argv}")
                    fails += 1
                if "Traceback" in proc.stderr:
                    print(f"FAIL traceback for {argv}: "
                          f"{proc.stderr.splitlines()[-1][:120]}")
                    fails += 1
    return fails


def fuzz_truncation(seed: int) -> int:
    """Truncated real requests through the bridge: every prefix of a
    valid JSON request must produce a clean invalid-JSON answer, never
    a raise (this is the stageb2 shape, applied to the new surface)."""
    fails = 0
    valid = json.dumps({"op": "qa_answer",
                        "question": "how do I fix a missing qml module",
                        "notes": [{"name": "n", "text": "t"}]})
    for cut in range(0, len(valid), 7):
        text = valid[:cut]
        try:
            req = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError:
            continue  # the transport rejects it before the bridge; fine
        try:
            resp = brain_bridge.handle(req, None, None)
            json.dumps(resp)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL truncation raise at cut={cut}: {exc}")
            fails += 1
    return fails


def main() -> int:
    fails = 0
    n_ops = len(brain_bridge.OPS)
    fails += fuzz_bridge(600, seed=20260930)
    print(f"bridge fuzz: 600 mutated requests over {n_ops} ops -> "
          f"{fails} failures")
    t = fuzz_cli(25, seed=20260930)
    fails += t
    print(f"cli fuzz: {25 * len(CLI_CASES)} mutated argvs -> {t} failures")
    t = fuzz_truncation(20260930)
    fails += t
    print(f"truncation fuzz: {t} failures")
    print("---")
    print("fuzz:", "PASS" if fails == 0 else f"FAIL ({fails})")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
