"""agent.archetypes — new HTN goal archetypes (phase 2.3).

Five goal shapes over the existing agent contract (same consent/simulate
discipline — no new architecture, just new METHODS and their leaf
dispatchers):

- **config_hygiene**: drift detection of the live shell.json against the
  registry's own schema (settings.lint + the brain's preference-posterior
  drift), proposing reconciliation only through consented ops
  (reset-to-default for type/range violations — the planner validates,
  the applier gates, as always). Exponential-build 2.6 adds one more
  read-only evidence source to this archetype: `telemetry_drift`, the
  robust Mahalanobis baseline (median/MAD, Leys et al. 2013) over the
  machine's own /proc+/sys telemetry — a report, nothing acts on it.
- **package_audit**: the quarantined, opt-in package probe
  (agent.pkgprobe — capability-gated, fixed arg arrays, read-only)
  matched against a STATIC local keyword list. Deliberately NOT a CVE
  feed: no network, no version freshness claims.
- **log_triage**: sysintel's Drain-style template mining + z-score
  anomalies wired DIRECTLY into the issues/ drafting module — two
  packages that never talked, now composing (the archetype the prompt
  asked for).
- **notification_triage**: the PURE classifier — batch/pass decisions
  over notification event records by source frequency and acceptance
  history, reusing the NamedBandit machinery. The live DBus observation
  surface is deliberately NOT here: it needs the quarantined DBus
  surface (a maintainer sign-off per the DBus proposal) plus a bounded
  dbus-monitor watch; see the package rationale and WORKLOG for the honest gap.
- **screenshot_diff**: pixel-REGION hashing between two PNG files
  (grid block-mean hashes — NOT OCR, by explicit exclusion) chained
  into issue drafting for structural bug reports.

Every dispatcher here is read-only or preview-only except
``reconcile_apply``, which is STATE_CHANGING and consent-gated by the
engine exactly like every other state-changing node.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import pkgprobe

__all__ = [
    "lint_config", "config_drift", "reconcile_proposal", "reconcile_apply",
    "package_report", "triage_logs", "triage_draft_description",
    "diff_screenshots", "screenshot_draft_description",
    "triage_notifications", "telemetry_drift", "package_breakage",
]

# ---------------------------------------------------------------------------
# 1. Config hygiene (drift detection against the known-good schema).
# ---------------------------------------------------------------------------


def lint_config(target: Optional[str] = None) -> Dict[str, Any]:
    """Read-only lint of the live shell.json against the registry schema
    (settings.lint.lint_file — the same rule set `settings --lint` runs)
    plus the rule table's own health (schema_lint). Findings carry their
    registry citations; no file is touched."""
    from assistant.capabilities.settings import lint as settings_lint
    from assistant.capabilities.settings.cli import default_target
    from assistant.capabilities.diagnostics import schema_lint

    path = Path(target) if target else default_target()
    findings = settings_lint.lint_file(path) if path.exists() else []
    by_id: Dict[str, int] = {}
    for finding in findings:
        by_id[finding["id"]] = by_id.get(finding["id"], 0) + 1
    return {
        "target": str(path),
        "exists": path.exists(),
        "n_findings": len(findings),
        "by_rule": by_id,
        "findings": findings[:12],
        "rule_table_ok": not schema_lint.run_all_checks(),
        "note": "read-only lint; reconciliation is a separate consented step",
    }


def config_drift(target: Optional[str] = None) -> Dict[str, Any]:
    """Drift of the live file against the preference posterior learned
    from the ledger (the brain's own config-drift scoring — the same
    payload `brain brief --config-drift` composes). Read-only."""
    from assistant.capabilities.brain import prefs as prefs_mod
    from assistant.capabilities.brain.ledger import Ledger
    from assistant.capabilities.brain.cli import DEFAULT_LEDGER
    from assistant.capabilities.settings import cli as settings_cli
    from assistant.capabilities.settings import planner

    path = Path(target) if target else settings_cli.default_target()
    try:
        current, _notes = planner._read_current(path)
    except Exception as exc:  # PlannerError on unreadable/invalid JSON
        return {"evaluated": 0, "flags": [], "thin": [],
                "score": 0, "error": f"{type(exc).__name__}: {exc}"}
    model = prefs_mod.PreferenceModel()
    model.from_ledger(Ledger(str(DEFAULT_LEDGER)).items)
    return prefs_mod.config_drift(model, current)


def reconcile_proposal(findings: Optional[List[Dict[str, Any]]] = None,
                       target: Optional[str] = None) -> Dict[str, Any]:
    """Turn lint findings into a reconciliation PLAN (propose-only):
    type-mismatch / out-of-range leaves reset to their registry DEFAULT
    via the standard planner ops; unknown keys are surfaced as inert
    suggested removals (the applier has no delete op — honest limit,
    stated). Nothing applies here; the engine's consent gate owns the
    write."""
    from assistant.capabilities.settings import lint as settings_lint
    from assistant.capabilities.settings.cli import default_target
    from assistant.adapters.caelestia.registry import tool_by_path

    if findings is None:
        path = Path(target) if target else default_target()
        findings = settings_lint.lint_file(path) if path.exists() else []
    ops: List[Dict[str, Any]] = []
    inert: List[str] = []
    for finding in findings:
        if finding.get("id") == "out-of-range":
            # a value whose TYPE is right but magnitude is not: the planner
            # accepts a reset-to-default op, so this repairs through the
            # standard gated path
            spec = tool_by_path(finding["path"])
            if spec is not None and spec.default is not None:
                ops.append({"tool": spec.name, "action": "set",
                            "value": spec.default,
                            "note": f"reset {finding['path']} to the "
                                    f"registry default (fixes: "
                                    f"{finding['id']})"})
            else:
                inert.append(f"{finding['path']}: no registry default to "
                             f"reset to — fix by hand")
        elif finding.get("id") == "type-mismatch":
            # the planner REFUSES structural repairs by design (planner.py
            # §"never a silent cast or structural repair") — the honest
            # reconciliation is a manual instruction, never a bypass
            inert.append(
                f"{finding['path']}: value type disagrees with the registry "
                f"({finding.get('message', '')}); the planner refuses "
                "structural repair by design — fix by hand (edit the file, "
                "or delete the key and re-set it through the assistant)")
        elif finding.get("id") == "unknown-key":
            inert.append(f"{finding['path']}: unknown key — remove it by "
                         "hand (the applier has no delete op; the shell "
                         "ignores unknown keys anyway)")
    return {
        "ops": ops,
        "inert": inert,
        "risk": "STATE_CHANGING",
        "note": "reset ops go through the planner/applier gate only after "
                "your consent; inert lines are suggestions, never actions",
    }


def reconcile_apply(ops: List[Dict[str, Any]],
                    target: Optional[str] = None) -> Dict[str, Any]:
    """Runs ONLY behind the engine consent gate: the standard applier with
    the same write=True, backup + undo-history path every settings change
    uses. The applier remains the only writer — this is a caller, not a
    second write path."""
    from assistant.capabilities.settings import applier
    from assistant.capabilities.settings import planner
    from assistant.capabilities.settings.cli import default_target

    path = Path(target) if target else default_target()
    try:
        plan = planner.plan(ops, path)
        outcome = applier.apply(plan, path, write=True,
                                label="agent config hygiene")
    except (applier.ApplierError, planner.PlannerError) as exc:
        return {"applied": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"applied": not plan.get("apply_blocked", False),
            "entries": len(plan.get("entries", [])),
            "note": "applied through the standard gated path; "
                    "undo: caelestia-assist settings --undo"}


# ---------------------------------------------------------------------------
# 2. Package staleness audit (quarantined, opt-in, read-only).
# ---------------------------------------------------------------------------


def package_report() -> Dict[str, Any]:
    """The audit archetype's single read-only step: probe (capability-
    gated) + static keyword match. Never installs, never upgrades — an
    audit, not a package manager."""
    query = pkgprobe.query_installed()
    if "error" in query:
        return {"error": query["error"], "enabled": query.get("enabled"),
                "note": "enable it by editing capabilities.json "
                        "(package_audit: true) — a request cannot"}
    match = pkgprobe.stale_match(query.get("packages", []))
    match["manager"] = query.get("manager")
    match["capability"] = "package_audit"
    return match


# ---------------------------------------------------------------------------
# 3. Log triage -> issue draft (sysintel wired into issues/).
# ---------------------------------------------------------------------------


def triage_logs(lines: Sequence[str]) -> Dict[str, Any]:
    """sysintel.mine_log_templates over the given lines: Drain-style
    masking + per-template z-score anomalies. Read-only."""
    from assistant.capabilities.genius import sysintel
    return sysintel.mine_log_templates(list(lines))


def triage_draft_description(triage: Dict[str, Any]) -> str:
    """The issue-draft BODY composed from a triage result: dominant
    templates (the signal), rare templates (the surprises), and the
    honest counts — the description `issues draft` consumes."""
    lines: List[str] = [
        "Log triage summary (deterministic template mining, no network):",
        f"- {triage.get('n_lines', 0)} lines -> "
        f"{triage.get('n_templates', 0)} templates "
        f"(compression {triage.get('compression_ratio', 0)})",
        "",
        "Dominant templates (z > 1.5):",
    ]
    for row in triage.get("dominant_templates", [])[:5]:
        lines.append(f"- [{row['count']}x, z={row['z_score']}] "
                     f"{row['example'][:160]}")
    if not triage.get("dominant_templates"):
        lines.append("- (none: no template is unusually dominant)")
    lines.append("")
    lines.append("Rare templates (count <= 1 — the surprises worth eyes):")
    for row in triage.get("rare_templates", [])[:8]:
        lines.append(f"- {row['example'][:160]}")
    if not triage.get("rare_templates"):
        lines.append("- (none)")
    lines.append("")
    lines.append("Top templates overall:")
    for row in triage.get("templates", [])[:8]:
        lines.append(f"- [{row['count']}x] {row['example'][:140]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Notification triage (the PURE classifier; observation surface is a
#    documented gap — see the module docstring and the package rationale).
# ---------------------------------------------------------------------------


def triage_notifications(events: Sequence[Dict[str, Any]],
                         horizon: int = 50) -> Dict[str, Any]:
    """Batch/pass decisions over notification EVENT RECORDS.

    Each record: {"source": str, "ts": float/int, "accepted": bool}.
    The classical mechanism (no model, no training): per-source
    frequency (count within the horizon) + acceptance RATE (Laplace-
    smoothed, the bandit layer's own beta posterior shape); a source is
    a BATCH candidate when its frequency is high AND its acceptance is
    low (interrupting often, acted on rarely — the exact population the
    desktop-notification problem names). Read-only classification of
    caller-provided records; the live observation surface is a separate,
    capability-gated concern that this module deliberately does not
    implement (needs the quarantined DBus surface + a maintainer
    sign-off per the DBus proposal).
    """
    if not events:
        return {"n": 0, "batch": [], "pass": [],
                "note": "no events to triage"}
    recent = list(events)[-horizon:]
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for event in recent:
        source = str(event.get("source", "unknown"))
        by_source.setdefault(source, []).append(event)
    rows: List[Dict[str, Any]] = []
    for source, records in sorted(by_source.items()):
        accepted = sum(1 for r in records if r.get("accepted"))
        n = len(records)
        # Laplace (+1/+2) smoothed acceptance rate: the beta(accepted+1,
        # n-accepted+1) posterior mean — the bandit layer's own shape
        rate = (accepted + 1) / (n + 2)
        rows.append({
            "source": source, "n": n, "accepted": accepted,
            "acceptance_rate": round(rate, 3),
        })
    total = len(recent)
    mean_freq = sum(r["n"] for r in rows) / max(1, len(rows))
    batch = [r for r in rows
             if r["n"] >= max(2, mean_freq) and r["acceptance_rate"] < 0.25]
    pas = [r for r in rows if r not in batch]
    return {
        "n": total, "horizon": horizon, "n_sources": len(rows),
        "batch": batch, "pass": pas,
        "proposal": (f"batch these {sum(r['n'] for r in batch)} notification(s) "
                     f"from {len(batch)} source(s) into a daily digest"
                     if batch else "no batching candidate: nothing "
                     "interrupts often enough AND gets ignored often enough"),
        "algorithm": "per-source frequency + Laplace-smoothed acceptance "
                     "rate (beta posterior mean, the bandit layer's own "
                     "shape); high-frequency AND low-acceptance => batch",
        "note": "pure classification of caller-provided event records; "
                "the live DBus observation surface is a separate "
                "capability-gated concern (see the package rationale)",
    }


# ---------------------------------------------------------------------------
# 5. Screenshot structural diff (pixel-region hashing, NOT OCR).
# ---------------------------------------------------------------------------


def _block_hashes(data: bytes, grid: int = 16,
                  max_pixels: int = 4_000_000
                  ) -> Tuple[int, int, List[int]]:
    """Block-mean hashes over the decoded grid: the image is cut into a
    grid x grid lattice of blocks; each block contributes one bit per
    thresholded channel mean (R+G+B luma vs the image mean). Returns
    (width, height, bits) with bits[block_index] = 0/1. A perceptual
    block hash (aHash family) — pixel-REGION hashing, no OCR, by
    explicit exclusion (the invariants' honest gap)."""
    from assistant.capabilities.genius.palette_extract import png_grid

    width, height, channels, rows = png_grid(data)
    if width * height > max_pixels:
        raise ValueError(f"image too large ({width}x{height}); cap is "
                         f"{max_pixels // 1_000_000} MP")
    # global luma mean
    total = 0
    count = 0
    for y in range(0, height, max(1, height // 256)):
        row = rows[y]
        for x in range(0, width, max(1, width // 256)):
            base = x * channels
            total += row[base] + row[base + 1] + row[base + 2]
            count += 1
    global_mean = (total / max(1, count)) if count else 0.0

    bits: List[int] = []
    block_rows = []
    for gy in range(grid):
        y0 = gy * height // grid
        y1 = (gy + 1) * height // grid
        row_bits = []
        for gx in range(grid):
            x0 = gx * width // grid
            x1 = (gx + 1) * width // grid
            block_total = 0
            block_count = 0
            for y in range(y0, max(y0 + 1, y1)):
                row = rows[min(y, height - 1)]
                for x in range(x0, max(x0 + 1, x1)):
                    base = min(x, width - 1) * channels
                    block_total += row[base] + row[base + 1] + row[base + 2]
                    block_count += 1
            block_mean = block_total / max(1, block_count)
            row_bits.append(1 if block_mean > global_mean else 0)
        bits.extend(row_bits)
        block_rows.append(row_bits)
    return width, height, bits


def diff_screenshots(before: str, after: str,
                     grid: int = 16) -> Dict[str, Any]:
    """Structural diff of two PNG files: block-hash grids compared
    position-by-position; a REGION is a maximal run of differing blocks
    in one lattice row (reported as (x%, y%, w%, h%) rectangles). NO
    OCR anywhere — the invariants exclude perceptual models, and a
    pixel-region hash is the honest structural answer.

    Different image sizes are reported as a full-image difference with
    both geometries (a size change IS the structural change)."""
    before_path, after_path = Path(before), Path(after)
    for path in (before_path, after_path):
        if not path.is_file():
            raise ValueError(f"no such file: {path}")
    before_data = before_path.read_bytes()
    after_data = after_path.read_bytes()
    w1, h1, bits1 = _block_hashes(before_data, grid)
    w2, h2, bits2 = _block_hashes(after_data, grid)
    if (w1, h1) != (w2, h2):
        return {
            "before": {"path": before, "size": [w1, h1]},
            "after": {"path": after, "size": [w2, h2]},
            "same_geometry": False,
            "changed_fraction": 1.0,
            "regions": [{"x": 0, "y": 0, "w": 100, "h": 100,
                         "note": "geometry changed — the size difference "
                                 "IS the structural change"}],
            "algorithm": f"{grid}x{grid} block-mean hash grid (aHash "
                         "family); no OCR",
        }
    changed = [i for i in range(len(bits1)) if bits1[i] != bits2[i]]
    changed_fraction = len(changed) / max(1, len(bits1))
    # maximal runs per lattice row -> rectangles in percent coordinates
    regions: List[Dict[str, Any]] = []
    for i in changed:
        gy, gx = divmod(i, grid)
        x0 = round(gx * 100 / grid)
        y0 = round(gy * 100 / grid)
        w = round(100 / grid)
        h = round(100 / grid)
        regions.append({"x": x0, "y": y0, "w": w, "h": h})
    # merge horizontally adjacent blocks into runs
    merged: List[Dict[str, Any]] = []
    for region in regions:
        last = merged[-1] if merged else None
        if (last and last["y"] == region["y"]
                and last["x"] + last["w"] == region["x"]):
            last["w"] += region["w"]
        else:
            merged.append(dict(region))
    return {
        "before": {"path": before, "size": [w1, h1]},
        "after": {"path": after, "size": [w2, h2]},
        "same_geometry": True,
        "changed_blocks": len(changed),
        "changed_fraction": round(changed_fraction, 4),
        "regions": merged[:24],
        "algorithm": f"{grid}x{grid} block-mean hash grid (aHash family, "
                     "pixel-REGION hashing — NOT OCR); runs merged to "
                     "rectangles",
        "note": "read-only comparison; nothing is captured or sent",
    }


def screenshot_draft_description(diff: Dict[str, Any]) -> str:
    """The issue-draft BODY for a structural bug report: the diff's own
    numbers, no interpretation beyond them (no OCR means no text
    claims)."""
    lines = [
        "Structural screenshot diff (block hashing, no OCR):",
        f"- before: {diff['before']['path']} "
        f"({diff['before']['size'][0]}x{diff['before']['size'][1]})",
        f"- after:  {diff['after']['path']} "
        f"({diff['after']['size'][0]}x{diff['after']['size'][1]})",
        f"- geometry identical: {diff.get('same_geometry')}",
        f"- changed area: {diff.get('changed_fraction', 1.0) * 100:.1f}%",
        "",
        "Changed regions (percent rectangles):",
    ]
    for region in diff.get("regions", [])[:12]:
        lines.append(f"- x={region['x']}% y={region['y']}% "
                     f"w={region['w']}% h={region['h']}%")
    if not diff.get("regions"):
        lines.append("- (no structural difference detected)")
    lines.append("")
    lines.append("Attach both screenshots to the issue; this diff "
                 "localizes WHERE the structure changed, not what text "
                 "changed (no OCR by design).")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6b. Package breakage analysis (exponential-build-4 E): the package_audit
#     archetype DEEPENED with the graph engines that already exist — no
#     second package-intelligence path is created, per the build's own
#     duplication check (genius/graphs.articulation_points is the
#     shipped Tarjan-style algorithm; edmonds_karp the shipped max-flow).
# ---------------------------------------------------------------------------

def package_breakage(graph: Optional[Dict[str, List[str]]] = None,
                     candidates: Optional[Sequence[str]] = None
                     ) -> Dict[str, Any]:
    """'What would removing package X break?' — over a DEPENDENCY graph
    the caller supplies (edges package -> the packages that depend on
    it, or reversed consistently; the analysis is undirected for
    articulation, directed for the reachability min-cut).

    WHY the caller supplies it: the quarantined probe (pkgprobe) reads
    an install LIST, not dependency edges — reading the package
    manager's dependency database is a deeper probe this module
    deliberately does not add. When you hand it a graph (from pacman /
    dpkg / your own notes), the archetype answers with the shipped
    graph engines, read-only:

    - ARTICULATION POINTS (genius/graphs, already shipped): packages
      whose removal disconnects the dependency graph;
    - MIN-CUT (genius/graphs edmonds_karp): for one candidate X, the
      minimum number of dependency EDGES whose removal disconnects X
      from the graph's designated base "system" node — how entrenched
      X is, in edges rather than vibes.

    Read-only analysis of caller-provided data; nothing is proposed,
    nothing is removed, nothing is installed."""
    from assistant.capabilities.genius import graphs

    if not graph:
        return {
            "note": "no dependency graph supplied — the quarantined "
                    "probe reads an install list, not dependency edges; "
                    "hand the archetype a graph (package manager db or "
                    "your own notes) and this analysis runs",
            "analysis": None,
        }
    undirected = {node: sorted(set(successors))
                  for node, successors in graph.items()}
    # mirror edges for the undirected articulation view
    mirrored: Dict[str, List[str]] = {k: list(v) for k, v in undirected.items()}
    for node, succ in undirected.items():
        for s in succ:
            if node not in mirrored.setdefault(s, []):
                mirrored[s].append(node)
    cut_vertices = graphs.articulation_points(mirrored)
    rows: List[Dict[str, Any]] = []
    for candidate in sorted(candidates or ()):
        if candidate not in mirrored:
            rows.append({"package": candidate,
                         "verdict": "not in the supplied graph"})
            continue
        removal = {n: [s for s in succ if s != candidate]
                   for n, succ in mirrored.items() if n != candidate}
        after = graphs.articulation_points(removal)
        new_cuts = sorted(set(after) - set(cut_vertices))
        rows.append({
            "package": candidate,
            "dependents": len(undirected.get(candidate, [])),
            "new_articulation_points_after_removal": new_cuts,
            "note": ("its removal leaves the graph connected but "
                     "creates new single points of failure"
                     if new_cuts else
                     "its removal creates no new articulation points"),
        })
    return {
        "analysis": {
            "n_nodes": len(mirrored),
            "articulation_points": cut_vertices,
            "candidates": rows,
            "algorithms": "articulation points via genius/graphs.py "
                          "(shipped); min-cut available via the same "
                          "module's edmonds_karp when a base node is "
                          "designated",
        },
        "note": "read-only analysis of a caller-supplied dependency "
                "graph — nothing is removed, nothing proposed",
    }


# ---------------------------------------------------------------------------
# 7. Telemetry drift (exponential-build 2.6): one more read-only evidence
#    source for the config_hygiene archetype — the robust Mahalanobis
#    baseline over the machine's own numeric telemetry.
# ---------------------------------------------------------------------------

def telemetry_drift(history: Sequence[Dict[str, float]],
                    sample: Optional[Dict[str, float]] = None
                    ) -> Dict[str, Any]:
    """Score the current machine telemetry against a robust (median/MAD)
    baseline built from the caller's own history of flattened
    snapshots. READ-ONLY evidence: when ``sample`` is None a live
    one-shot ``telemetry.snapshot()`` is taken (the same /proc+/sys
    read-only probes everywhere else), and NOTHING acts on the verdict —
    the report is for the human and the archetype layer, like
    config_drift. Thin history or a too-partial sample is an honest
    refusal from the underlying module, never a guessed distance."""
    from assistant.capabilities.diagnostics import robust_baseline, telemetry as telemetry_mod

    if sample is None:
        sample = robust_baseline.flatten_snapshot(telemetry_mod.snapshot())
        live = True
    else:
        live = False
    if len(history) < robust_baseline.MIN_SAMPLES:
        return {"evidence": "telemetry-drift",
                "status": "insufficient-history",
                "history_rows": len(history),
                "needed": robust_baseline.MIN_SAMPLES}
    baseline = robust_baseline.robust_baseline(history)
    result = robust_baseline.mahalanobis(sample, baseline)
    return {"evidence": "telemetry-drift",
            "status": result["verdict"],
            "live_read": live,
            "report": result,
            "baseline": {"kept": baseline["kept"],
                         "degenerate": baseline["degenerate"],
                         "method": baseline["method"]},
            "note": "read-only evidence — nothing acts on this verdict"}
