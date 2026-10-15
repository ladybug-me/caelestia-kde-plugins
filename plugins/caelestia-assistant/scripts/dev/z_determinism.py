#!/usr/bin/env python3
"""Phase Z2 — determinism: 3 hash seeds x 2 interpreters x 7 surfaces.

The behavior lock pins goldens under ONE seed and ONE interpreter; this
sweep pins the seeds-and-interpreters dimension the lock cannot cover.
Every surface runs 6 times (3 PYTHONHASHSEED values x 2 interpreters)
in a throwaway HOME; all 6 stdout bytes must be identical.

Run: python3 scripts/dev/z_determinism.py
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

PLAN_PATH = Path(tempfile.gettempdir()) / "z-sweep-plan.json"
PLAN_PATH.write_text(json.dumps(
    [{"tool": "setSpacingScale", "value": 1.1, "field": "spacing"},
     {"tool": "setAccentColor", "value": "red", "field": "color"}]))

SURFACES = [
    ("route", ["route", "make the bar slightly thicker", "--json"]),
    ("settings", ["settings", "make the bar slightly thicker", "--json"]),
    ("search", ["search", "bar scale", "--json"]),
    ("brain-say", ["brain", "say", str(PLAN_PATH), "--json"]),
    ("brain-qa", ["brain", "qa",
                  "how do I fix a missing qml module metadata", "--json"]),
    ("shellkb", ["shellkb", "explain", "ls -la /tmp", "--json"]),
    ("eval", ["eval", "routing", "--json"]),
]

SEEDS = ["0", "1", "20260930"]


def interpreters():
    seen = [sys.executable]
    for alt in ("/usr/bin/python3.13", "/usr/bin/python3.12",
                "/usr/bin/python3.11"):
        if os.path.exists(alt) and os.path.realpath(alt) != \
                os.path.realpath(sys.executable):
            seen.append(alt)
    return seen


def run(py, argv, seed):
    env = dict(os.environ)
    env.update({
        "PYTHONHASHSEED": seed,
        "PYTHONPATH": str(REPO),
        "XDG_CONFIG_HOME": "/tmp/z-sweep-config",
        "XDG_DATA_HOME": "/tmp/z-sweep-data",
        "CAELESTIA_BRAIN_STATE": "/tmp/z-sweep-brain.json",
    })
    env.pop("CAELESTIA_ASSIST_CAPABILITIES", None)
    home = tempfile.mkdtemp(prefix="z-det-")
    env["HOME"] = home
    proc = subprocess.run([py, "-m", "assistant.hub", *argv],
                          capture_output=True, text=True, timeout=120,
                          env=env, cwd=REPO)
    return proc.returncode, proc.stdout.replace(home, "<HOME>")


def main():
    pys = interpreters()
    fails = 0
    for name, argv in SURFACES:
        outs = []
        for py in pys:
            for seed in SEEDS:
                rc, out = run(py, argv, seed)
                if rc != 0:
                    print(f"FAIL {name} {py} seed={seed} rc={rc}")
                    fails += 1
                else:
                    outs.append(out)
        if outs and len(set(outs)) == 1:
            print(f"ok  {name} ({len(outs)} runs byte-identical, "
                  f"{len(outs[0])} bytes)")
        elif outs:
            print(f"FAIL {name} drift across {len(outs)} runs")
            fails += 1
    print("---")
    print("determinism sweep:",
          "PASS" if fails == 0 else f"FAIL ({fails})")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
