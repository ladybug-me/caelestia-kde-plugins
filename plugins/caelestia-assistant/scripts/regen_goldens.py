#!/usr/bin/env python3
"""Behavior-lock goldens (Stage C1).

Generates tests/goldens/manifest.json: >= 300 read-only CLI
cases, each run TWICE in-process from an isolated environment; a case
is kept only when both runs return rc 0 and byte-identical stdout
(self-filtering — timing-dependent surfaces drop out on their own).
Every kept case records sha256(stdout); a reviewable sample of full
outputs is stored alongside as samples/*.txt.

Update protocol: goldens change ONLY via
    python3 scripts/regen_goldens.py
followed by a reviewed commit whose message states the behavior change
that justified the update. The lock test fails loudly on any silent
drift, including case-count loss (minimum 300).
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from assistant.hub import main as hub_main  # noqa: E402

GOLD_DIR = REPO / "tests" / "goldens"
SAMPLE_COUNT = 60


def _items(suite_file):
    data = json.loads((REPO / "assistant" / "core" / "eval" / "sets" /
                       suite_file).read_text())
    return data["items"]


def build_cases():
    cases = []  # (id, argv)

    def add(cid, *argv):
        cases.append((cid, [str(a) for a in argv]))

    # 1. route over every dev routing item
    for it in _items("routing_dev.json"):
        add(f"route-{it['id']}", "route", it["text"], "--json")
    # 2. route over the dev abstention negatives (honest refusals)
    for it in _items("abstention_dev.json"):
        add(f"route-{it['id']}", "route", it["text"], "--json")
    # 3. route over nlplan dev items (tool-exact expectations)
    for it in _items("nlplan_dev.json"):
        add(f"route-{it['id']}", "route", it["text"], "--json")
    # 4. the SAME nlplan phrasings through the settings planner surface
    for it in _items("nlplan_dev.json"):
        add(f"settings-{it['id']}", "settings", it["text"], "--json")
    # 5. hand-written settings plan-only requests
    for i, req in enumerate([
        "make the bar slightly thicker", "hide the dock icons",
        "move the dock to the left edge", "set the accent colour to red",
        "turn off blur", "make notifications quieter",
        "change the wallpaper to a dark one", "enable night light",
        "make the terminal font bigger", "disable the system tray",
        "add a widget to the bar", "set the bar to auto-hide",
        "increase the corner rounding", "disable desktop icons",
        "make the launcher open faster",
    ], 1):
        add(f"settings-manual-{i:02d}", "settings", req, "--json")
    # 6. genius arithmetic / stats / matrix / solve
    calcs = [
        "2 + 3 * 7", "(2+3)*7", "2**10", "17 % 5", "7!/3!",
        "sqrt(2)", "sqrt(2)^2", "sin(pi/6)", "cos(0)", "ln(e)",
        "log(1000)", "1/3 + 1/6", "0.1 + 0.2", "abs(-4.2)",
        "floor(3.7)", "ceil(3.2)", "round(3.5)", "max(3, 7, 5)",
        "min(3, 7, 5)", "gcd(12, 18)", "lcm(4, 6)", "2^0.5",
        "e^2", "10 - 4 - 3", "100 / 8", "2 * pi",
        "ncr(5, 2)", "hypot(3, 4)", "sqrt(16) + 2^3",
        "-5 + 12", "3 * (4 + 5) - 2", "pi * 2", "pi^2", "1/7",
        "9.75 * 4", "12.5 % 3", "2^(-2)", "0.5!",
        "tanh(0)", "atan(1)", "exp(ln(7))", "floor(5 / 2)", "0 - (-3)",
        "1e3 + 25", "10 + 15",
    ]
    for i, expr in enumerate(calcs, 1):
        add(f"genius-math-{i:02d}", "genius", "math", expr, "--json")
    stats_rows = [
        "3 5 7 9 11", "1 1 2 3 5 8 13", "10 20 30 40 50",
        "2.5 2.5 2.5", "42", "1 2", "5 5 5 5 5 5", "100 200 300",
    ]
    for i, row in enumerate(stats_rows, 1):
        add(f"genius-stats-{i:02d}", "genius", "stats", row, "--json")
    matrices = ["[[2,1],[1,3]]", "[[1,0],[0,1]]", "[[4,7],[2,6]]",
                "[[3]]", "[[5,2],[7,3]]", "[[0,1],[1,0]]"]
    for i, m in enumerate(matrices, 1):
        add(f"genius-matrix-{i:02d}", "genius", "matrix", m, "--json")
    for i, (expr, lo, hi) in enumerate([
        ("x**2 - 4", 0, 5), ("x - 3", 0, 10), ("sin(x) - 0.5", 0, 2),
        ("x**3 - 8", 0, 4),
    ], 1):
        add(f"genius-solve-{i:02d}", "genius", "solve", expr,
            "--lo", lo, "--hi", hi, "--json")
    calcs_opt = [
        (["genius", "calc", "--derivative", "x**2", "--at", "3", "--json"]),
        (["genius", "calc", "--derivative", "sin(x)", "--at", "0",
          "--json"]),
        (["genius", "calc", "--integral", "x**2", "--a", "0", "--b", "1",
          "--json"]),
        (["genius", "calc", "--taylor", "exp(x)", "--at", "0",
          "--order", "5", "--json"]),
        (["genius", "calc", "--derivative", "x**3 - 2*x", "--at", "1",
          "--json"]),
        (["genius", "calc", "--integral", "sin(x)", "--a", "0", "--b",
          "3.141592653589793", "--json"]),
    ]
    for i, argv in enumerate(calcs_opt, 1):
        add(f"genius-calcopt-{i:02d}", *argv)
    dos = [
        "what is 15% of 240", "convert 5 km to m",
        "convert 2 hours to minutes", "what is 3^4",
        "20 percent of 150", "convert 1 GB to MB",
        "convert -40 C to F", "what is sqrt(144)",
        "convert 3 days to hours", "what is 7 + 8 * 2",
    ]
    for i, q in enumerate(dos, 1):
        add(f"do-{i:02d}", "do", q, "--json")
    # 7. the arena itself (dev split)
    for suite in ("routing", "nlplan", "abstention", "calibration"):
        add(f"eval-{suite}", "eval", suite, "--json")
    # 8. accessibility contrast checks
    for i, (fg, bg) in enumerate([
        ("#000000", "#ffffff"), ("#ffffff", "#000000"),
        ("#777777", "#ffffff"), ("#333333", "#e0e0e0"),
        ("#ffcc00", "#333366"), ("#aaaaaa", "#dddddd"),
    ], 1):
        add(f"a11y-{i:02d}", "settings", "--accessibility-check",
            fg, bg, "--json")
    # 8b. extra volume: math, a11y, settings, graph
    for i, expr in enumerate([
        "erf(0.5)", "gamma(5)", "sinh(0)", "cosh(0)", "atan(1) * 4",
        "sqrt(2) * sqrt(8)", "3! + 4!", "fact(6) / fact(4)",
        "hypot(3, 4) * 2", "ncr(10, 3)",
    ], 1):
        add(f"genius-math2-{i:02d}", "genius", "math", expr, "--json")
    for i, (fg, bg) in enumerate([
        ("#123456", "#fedcba"), ("#ff0000", "#00ff00"),
        ("#202020", "#f0f0f0"), ("#abcdef", "#123456"),
    ], 1):
        add(f"a11y2-{i:02d}", "settings", "--accessibility-check",
            fg, bg, "--json")
    for i, req in enumerate([
        "make the text smaller in the bar", "show the battery percentage",
        "hide the workspace indicator", "set volume to fifty percent",
        "mute the microphone", "make the icons round",
        "enable the floating dock", "set the clock to 24 hour",
        "show seconds on the clock", "disable notifications at night",
    ], 1):
        add(f"settings2-{i:02d}", "settings", req, "--json")
    for i, (sub, arg) in enumerate([
        ("breaks", "setBarScale"), ("related", "setPosition"),
        ("affects", "notifications.doNotDisturb"),
        ("affects", "blur.enabled"), ("related", "setWallpaper"),
        ("breaks", "setPosition"),
    ], 1):
        add(f"graph2-{i:02d}", "graph", sub, arg)
    # 8c. extractive QA over the committed corpus (D17): read-only,
    # deterministic spans + pinned verdicts — the answers are quotes
    # with citations, so the lock pins the QUOTING behavior itself
    for i, q in enumerate([
        "how do I fix a missing qml module metadata",
        "what causes the vesktop freeze on screenshare",
        "when was the quickshell crash dialog issue reported",
        "why does the installer build fail",
        "how do I fix gamescope launching in half the screen",
        "why is my bluetooth turning on and off",
        "zzqq blorptastic frumious",
    ], 1):
        add(f"qa-{i:02d}", "brain", "qa", q, "--json")

    # 9. read-only dashboards and cards on the cold-start state
    add("misc-selfcheck", "selfcheck")
    add("misc-capabilities", "capabilities")
    add("misc-graph-rank", "graph", "rank")
    add("misc-graph-related", "graph", "related", "setBarScale")
    add("misc-graph-affects", "graph", "affects", "bar.scale")
    add("misc-graph-path", "graph", "path", "setBarScale", "setBlur")
    add("misc-inbox-list", "inbox", "list")
    add("misc-cortex-report", "cortex", "report")
    add("misc-search-1", "search", "bar scale", "--json")
    add("misc-search-2", "search", "notification position", "--json")
    add("misc-search-3", "search", "how to reset layout", "--json")
    add("misc-usage", "--help")
    return cases


def run_once(argv):
    """One case in a fresh subprocess (timeout-guarded): full env
    isolation, no global mutation, a hung case fails the case instead
    of the whole generation run. scripts/ is a dev tool — the
    assistant/-only import policy does not apply here."""
    home = tempfile.mkdtemp(prefix="golden-home-")
    env = dict(os.environ)
    env.update({
        "HOME": home,
        "XDG_DATA_HOME": os.path.join(home, "data"),
        "XDG_CONFIG_HOME": os.path.join(home, "config"),
        "CAELESTIA_BRAIN_STATE": os.path.join(home, "brain-state.json"),
        "PYTHONPATH": str(REPO),
    })
    env.pop("CAELESTIA_ASSISTANT_PLAIN", None)
    proc = subprocess.run(
        [sys.executable, "-m", "assistant.hub", *argv],
        capture_output=True, text=True, timeout=20, env=env, cwd=REPO)
    # normalize the sandbox home away: the lock is on BEHAVIOR, not on
    # the temporary path this run happened to use
    return (proc.returncode,
            proc.stdout.replace(home, "<HOME>"),
            proc.stderr.replace(home, "<HOME>"))


def main():
    cases = build_cases()
    print(f"probing {len(cases)} cases (2 runs each)...", flush=True)
    manifest = []
    dropped = []
    for n, (cid, argv) in enumerate(cases, 1):
        if n % 25 == 0:
            print(f"  ...{n}/{len(cases)}", flush=True)
        try:
            rc1, out1, err1 = run_once(argv)
            rc2, out2, err2 = run_once(argv)
        except Exception as exc:  # noqa: BLE001 — self-filter, documented
            dropped.append((cid, f"exception/timeout: {exc!r}"))
            continue
        if rc1 != rc2 or rc1 not in (0, 1):
            # rc 1 with byte-identical output locks an honest refusal;
            # rc 2 (argparse usage) and rc mismatches are dropped
            dropped.append((cid, f"rc {rc1}/{rc2}: {err1.strip()[:90]}"))
            continue
        if out1 != out2 or err1 != err2:
            dropped.append((cid, "nondeterministic across runs"))
            continue
        manifest.append({
            "id": cid, "argv": argv, "rc": rc1,
            "sha": hashlib.sha256(out1.encode()).hexdigest(),
            "bytes": len(out1),
        })
        case_out = {"id": cid, "argv": argv, "rc": rc1,
                    "stdout": out1, "stderr": err1}
        (GOLD_DIR / "_raw").mkdir(parents=True, exist_ok=True)
        (GOLD_DIR / "_raw" / f"{cid}.json").write_text(
            json.dumps(case_out, indent=1))
    print(f"kept {len(manifest)} deterministic cases; "
          f"dropped {len(dropped)}")
    for cid, why in dropped[:12]:
        print(f"  drop {cid}: {why}")
    if len(dropped) > 12:
        print(f"  ... and {len(dropped) - 12} more drops")
    if len(manifest) < 300:
        print(f"FAIL: only {len(manifest)} cases locked (need >= 300)")
        return 1
    # sample of human-reviewable full outputs
    samples = GOLD_DIR / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    for old in samples.glob("*.txt"):
        old.unlink()
    for entry in manifest[:SAMPLE_COUNT]:
        raw = json.loads((GOLD_DIR / "_raw" / f"{entry['id']}.json")
                         .read_text())
        header = (f"# golden sample: {entry['id']}\n"
                  f"# argv: {entry['argv']}\n"
                  f"# rc: {entry['rc']}\n")
        (samples / f"{entry['id']}.txt").write_text(
            header + "\n--- stdout ---\n" + raw["stdout"] +
            ("\n--- stderr ---\n" + raw["stderr"] if raw["stderr"] else ""))
    import shutil
    shutil.rmtree(GOLD_DIR / "_raw")
    body = {
        "protocol": __doc__.strip(),
        "min_cases": 300,
        "cases": manifest,
    }
    (GOLD_DIR / "manifest.json").write_text(json.dumps(body, indent=1))
    print(f"wrote {GOLD_DIR / 'manifest.json'} "
          f"({len(manifest)} cases, {SAMPLE_COUNT} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
