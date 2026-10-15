#!/usr/bin/env python3
"""Stage C1 behavior lock — the byte-level checker.

Re-runs every golden case in a fresh subprocess (isolated HOME, pinned
PYTHONHASHSEED=0), normalizes the sandbox home path away, and compares
rc + sha256(stdout) against tests/goldens/manifest.json.

This is a scripts/ dev tool ON PURPOSE: the import policy forbids
subprocess inside assistant/ Python, so the byte lock cannot live in
the unittest tree; the unittest tree holds the structural checks
(manifest integrity, sample consistency, minimum coverage) and the
bash suite invokes this checker (tests/test_assistant.sh).

Exit 0 = all goldens hold. Exit 1 = drift (or count below minimum) —
regenerate via scripts/regen_goldens.py ONLY with a reviewed commit
justifying the behavior change.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "tests" / "goldens" / "manifest.json"


def run_case(argv):
    home = tempfile.mkdtemp(prefix="golden-check-")
    env = dict(os.environ)
    env.update({
        "HOME": home,
        "XDG_DATA_HOME": os.path.join(home, "data"),
        "XDG_CONFIG_HOME": os.path.join(home, "config"),
        "CAELESTIA_BRAIN_STATE": os.path.join(home, "brain-state.json"),
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": str(REPO),
    })
    env.pop("CAELESTIA_ASSISTANT_PLAIN", None)
    proc = subprocess.run(
        [sys.executable, "-m", "assistant.hub", *argv],
        capture_output=True, text=True, timeout=60, env=env, cwd=REPO)
    norm = lambda s: s.replace(home, "<HOME>")  # noqa: E731
    return proc.returncode, norm(proc.stdout), norm(proc.stderr)


def main():
    if not MANIFEST.exists():
        print("golden lock: manifest missing (run scripts/regen_goldens.py)")
        return 1
    data = json.loads(MANIFEST.read_text())
    cases = data["cases"]
    if len(cases) < data["min_cases"]:
        print(f"golden lock: only {len(cases)} cases < minimum "
              f"{data['min_cases']}")
        return 1
    failures = []
    for n, case in enumerate(cases, 1):
        if n % 50 == 0:
            print(f"  ...{n}/{len(cases)}", flush=True)
        try:
            rc, out, _err = run_case(case["argv"])
        except subprocess.TimeoutExpired:
            failures.append((case["id"], "timeout"))
            continue
        if rc != case["rc"]:
            failures.append((case["id"], f"rc {rc} != {case['rc']}"))
            continue
        if hashlib.sha256(out.encode()).hexdigest() != case["sha"]:
            failures.append((case["id"], "stdout drift"))
    if failures:
        print(f"golden lock: {len(failures)} FAILURE(S)")
        for cid, why in failures[:10]:
            print(f"  DRIFT {cid}: {why}")
        print("if the behavior change is intentional and reviewed, "
              "regenerate via scripts/regen_goldens.py and say why in "
              "the commit message")
        return 1
    print(f"golden lock: all {len(cases)} outputs byte-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
