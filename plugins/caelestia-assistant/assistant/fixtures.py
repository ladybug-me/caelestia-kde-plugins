"""assistant.fixtures — the synthetic-fixture test framework
(exponential-build-3 F2).

Every builder in this module fabricates, on demand and determin-
istically, the artifacts the assistant normally reads from a real
caelestia-kde installation: shell-config JSON trees, /proc + /sys
telemetry snapshot trees, markdown vaults, and the brain state /
ledger files. The goal is stated plainly: a contributor with NO
caelestia-kde checkout, no installed shell, and no real battery can
run the FULL test suite, because nothing in the suite depends on any
of those things existing — and this module is the shared vocabulary
that keeps it that way (new tests build fixtures with these helpers
instead of hand-rolling temp files, so the no-installment guarantee
stays true by construction instead of by discipline).

Design rules (the same ones the package's tests already follow):

- every builder takes an explicit target PATH plus fully explicit
  content parameters — defaults exist only where the default IS the
  interesting fixture (e.g. the canonical valid config), and every
  default is documented inline;
- no RNG anywhere: same arguments -> byte-identical trees;
- builders return plain dicts of the paths they created, so tests
  can pass exactly the pieces they need to the injectable readers
  (telemetry.read_battery(pattern=...), vault.scan(root), ...);
- nothing here reads or writes outside the caller-supplied directory.

Import surface stays on the allow-list: json, os, pathlib, tempfile.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

__all__ = ["shell_config", "shell_tree", "telemetry_tree", "vault",
           "state_file", "ledger_file"]

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# shell.json fixtures (the settings layer's world)
# ---------------------------------------------------------------------------

# The canonical VALID config: a fresh install's plausible state —
# every value in range, every key known to the core registry.
CANONICAL_CONFIG: Dict[str, Any] = {
    "bar": {"scale": 1.0, "persistent": True, "position": "bottom"},
    "unrelated": {"keep": [1, 2]},
}


def shell_config(path: PathLike, config: Optional[Dict[str, Any]] = None,
                 raw: Optional[str] = None) -> Path:
    """Write one shell.json-shaped file and return its path.

    ``config`` (default: the canonical valid config above) is dumped
    with the SAME formatting the settings applier writes (4-space
    indent + trailing newline), so byte-level tests see realistic
    bytes. ``raw`` overrides: the exact bytes written, for
    malformed-JSON and non-object fixtures."""
    target = Path(path)
    if raw is not None:
        target.write_text(raw, encoding="utf-8")
        return target
    doc = CANONICAL_CONFIG if config is None else config
    target.write_text(json.dumps(doc, indent=4) + "\n", encoding="utf-8")
    return target


def shell_tree(directory: PathLike) -> Dict[str, Path]:
    """A spread of shell-config VARIANTS — the fixture vocabulary for
    settings tests: the canonical valid file, an out-of-range value,
    an unknown key, a type mismatch, malformed JSON, and a non-object
    document. Returns {name: path} with stable names (valid.json,
    out-of-range.json, unknown-key.json, type-mismatch.json,
    malformed.json, non-object.json)."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    made: Dict[str, Path] = {}
    made["valid"] = shell_config(root / "valid.json")
    made["out-of-range"] = shell_config(
        root / "out-of-range.json", {"bar": {"scale": 99.0}})
    made["unknown-key"] = shell_config(
        root / "unknown-key.json", {"bar": {"totally_unknown": 1}})
    made["type-mismatch"] = shell_config(
        root / "type-mismatch.json", {"bar": {"scale": "big"}})
    made["malformed"] = shell_config(
        root / "malformed.json", raw='{"bar": {"scale": ')
    made["non-object"] = shell_config(
        root / "non-object.json", raw="[1, 2, 3]\n")
    return made


# ---------------------------------------------------------------------------
# /proc + /sys telemetry fixtures (the diagnostics layer's world)
# ---------------------------------------------------------------------------

def telemetry_tree(directory: PathLike,
                   load1: float = 0.40, load5: float = 0.35,
                   load15: float = 0.30,
                   mem_total_kb: int = 16_000_000,
                   mem_available_kb: int = 8_000_000,
                   batteries: Optional[Dict[str, int]] = None,
                   thermal_zones: Optional[Dict[str, int]] = None,
                   ) -> Dict[str, Any]:
    """A fake /proc + /sys snapshot tree: exactly the file shapes the
    read-only probes in diagnostics/telemetry.py parse. Default
    values describe a calm, on-battery, mildly-warm laptop. Returns
    {"proc": <dir>, "sys": <dir>, "loadavg": path, "meminfo": path,
    "battery_globs": ["<dir>/BAT*/capacity", ...], "thermal_globs":
    [...]} — the glob patterns the injectable readers expect."""
    root = Path(directory)
    proc = root / "proc"
    sysdir = root / "sys"
    proc.mkdir(parents=True, exist_ok=True)
    (sysdir / "class" / "power_supply").mkdir(parents=True, exist_ok=True)
    (sysdir / "class" / "thermal").mkdir(parents=True, exist_ok=True)

    loadavg = proc / "loadavg"
    loadavg.write_text(f"{load1:.2f} {load5:.2f} {load15:.2f} "
                       f"1/800 12345\n", encoding="utf-8")
    used_kb = mem_total_kb - mem_available_kb
    meminfo = proc / "meminfo"
    meminfo.write_text(
        f"MemTotal:       {mem_total_kb} kB\n"
        f"MemFree:         {used_kb // 4} kB\n"
        f"MemAvailable:    {mem_available_kb} kB\n", encoding="utf-8")

    battery_globs: List[str] = []
    supply = sysdir / "class" / "power_supply"
    for name, pct in (batteries if batteries is not None
                      else {"BAT0": 74}).items():
        bat = supply / name
        bat.mkdir(parents=True, exist_ok=True)
        (bat / "capacity").write_text(f"{pct}\n", encoding="utf-8")
    battery_globs.append(str(supply / "*" / "capacity"))
    thermal_globs: List[str] = []
    thermal = sysdir / "class" / "thermal"
    for name, mc in (thermal_zones if thermal_zones is not None
                     else {"thermal_zone0": 62_000}).items():
        zone = thermal / name
        zone.mkdir(parents=True, exist_ok=True)
        (zone / "temp").write_text(f"{mc}\n", encoding="utf-8")
    thermal_globs.append(str(thermal / "*" / "temp"))
    return {"proc": str(proc), "sys": str(sysdir),
            "loadavg": str(loadavg), "meminfo": str(meminfo),
            "battery_globs": battery_globs,
            "thermal_globs": thermal_globs}


# ---------------------------------------------------------------------------
# markdown vault fixtures (the personal brain's world)
# ---------------------------------------------------------------------------

VAULT_NOTES: Dict[str, str] = {
    "cat.md": "cat cat cat purr whisker cat purr\nsee [[dog-notes]]",
    "dog.md": "dog dog bark fetch dog bone\nsee [[cat-notes]]",
    "mix.md": "cat dog purr fetch",
    "orphan.md": "an island, linked from nowhere",
}


def vault(directory: PathLike,
          notes: Optional[Dict[str, str]] = None,
          mtimes: Optional[Dict[str, float]] = None) -> Path:
    """A fake markdown vault (default: the cat/dog/mix/orphan quartet
    the graph/topic tests think in). ``mtimes`` optionally pins file
    ages (epoch seconds, via os.utime) for the time-decay fixtures.
    Returns the vault root path."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    for name, text in (notes if notes is not None
                       else VAULT_NOTES).items():
        p = root / name
        p.write_text(text, encoding="utf-8")
        if mtimes and name in mtimes:
            os.utime(p, (mtimes[name], mtimes[name]))
    return root


# ---------------------------------------------------------------------------
# brain state / ledger fixtures (the learning layer's world)
# ---------------------------------------------------------------------------

def state_file(path: PathLike, state: Optional[Dict[str, Any]] = None
               ) -> Path:
    """A brain-state JSON file (default: a plausible mid-use state —
    a few learned examples, an SRS card, a comparison row)."""
    doc = state if state is not None else {
        "version": 1,
        "preset_comparisons": [
            {"winner": "minimal", "loser": "gaming", "at": "2026-09-01T10:00:00"},
        ],
        "srs": {"card-1": {"due": 1.0, "interval": 1.0, "ease": 2.5}},
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return target


def ledger_file(path: PathLike,
                entries: Optional[List[Dict[str, Any]]] = None) -> Path:
    """A proposal-ledger JSON file in the ledger's OWN schema
    (brain/ledger.py: {"proposals": [...]}, each item id/kind/target/
    diff/reason/confidence/status/decided_at). The default holds one
    pending and one approved proposal."""
    items = entries if entries is not None else [
        {"id": 1, "kind": "launch_pattern",
         "target": "editor -> terminal",
         "diff": {"sequence": ["editor", "terminal"]},
         "reason": "frequent launch sequence (2/3 days)",
         "confidence": 0.667, "status": "pending",
         "decided_at": None},
        {"id": 2, "kind": "tag", "target": "orphan.md",
         "diff": {"add_tags": ["notes"]},
         "reason": "naive-bayes tag suggestion", "confidence": 0.4,
         "status": "approved",
         "decided_at": "2026-09-01T10:00:00+00:00"},
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"proposals": items}, indent=2, sort_keys=True)
        + "\n", encoding="utf-8")
    return target
