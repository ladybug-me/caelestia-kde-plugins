#!/usr/bin/env python3
"""Cold-CLI latency measurer (the script the footprint suite's note
names; referenced by assistant/eval/engine.py).

Measures the true COLD first route: a fresh subprocess, fresh HOME,
fresh state, one `route` call, wall-clocked. This is the number the
250 ms budget gate is about — the in-process footprint probe cannot
see interpreter startup or module import, and the import policy
rightly forbids subprocess inside assistant/.

Run: PYTHONPATH=. python3 scripts/dev/measure_cli.py [N]
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def one_cold_route() -> float:
    home = tempfile.mkdtemp(prefix="cold-cli-")
    env = dict(os.environ)
    env.update({
        "HOME": home,
        "XDG_CONFIG_HOME": os.path.join(home, "config"),
        "XDG_DATA_HOME": os.path.join(home, "data"),
        "CAELESTIA_BRAIN_STATE": os.path.join(home, "brain.json"),
        "PYTHONPATH": str(REPO),
    })
    env.pop("CAELESTIA_ASSIST_CAPABILITIES", None)
    t0 = time.perf_counter()
    subprocess.run(
        [sys.executable, "-m", "assistant.hub", "route",
         "make the bar slightly thicker", "--json"],
        capture_output=True, text=True, timeout=60, env=env, cwd=REPO,
        check=True)
    return (time.perf_counter() - t0) * 1000.0


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    times = [one_cold_route() for _ in range(n)]
    times.sort()
    p50 = times[len(times) // 2]
    print(json.dumps({
        "n": n,
        "cold_cli_ms": [round(t, 1) for t in times],
        "p50_ms": round(p50, 1),
        "budget_ms": 250.0,
        "within_budget": p50 <= 250.0,
        "note": "sandbox numbers; the gate is judged on the reference "
                "1-vCPU machine (this sandbox runs ~2.2-2.5x slower)",
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
