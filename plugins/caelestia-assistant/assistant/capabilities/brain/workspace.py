"""Session clustering -> workspace profiles (issue #120 Phase 3 roadmap:
"personalized desktop presets" / "workspace profiles"), Phase 2.1.

k-means over (app, workspace, monitor, hour-of-day) session vectors;
clusters that co-occur consistently become NAMED PROFILE PROPOSALS in the
existing ledger. Nothing is ever auto-applied: the ledger's
approve/reject flow is the whole interaction, exactly like every other
brain proposal.

DATA SOURCE — verified, not assumed (2026-09-26, upstream caelestia-kde):
the shell persists NO per-session (app, workspace, monitor, hour) log.
The workspace-tracker KWin effect broadcasts live state over a local
socket for the bar widget and writes nothing to disk
(shell/kwin-effects/workspace-tracker/workspace_tracker.cpp: a socket
write, no file); the launcher's apps.sqlite stores aggregate launch
counts, not sessions. This module therefore consumes EXPLICITLY
SUPPLIED session records — the same caller-supplied-series pattern as
brain/rhythm.py, brain/forecast.py and brain/dreamtime.py — and says so
plainly. If upstream ever persists a real session log, only the caller
changes; the schema here stays {"app": str, "workspace": int,
"monitor": str, "hour": 0-23}.

Pipeline (every step a named, citable technique):

1. Feature encoding — one-hot app (rare apps below `min_count` fold into
   "__other__"), one-hot monitor, min-max normalized workspace index, and
   circular hour encoding (sin/cos of 2*pi*h/24, the standard time-of-day
   embedding, so hour 23 neighbors hour 0).
2. k-means with k-means++ seeding (reuses genius/data.py::kmeans — the
   same deterministic implementation the genius layer ships and tests;
   k defaults to the min(8, max(2, round(sqrt(n/2)))) heuristic and is
   caller-overridable).
3. Cluster purity filter (standard cluster-purity metric): a cluster is
   proposed only when its modal (app, monitor, workspace) triple covers
   >= `purity` of its members AND it has >= `min_support` sessions —
   consistency is the point; noise is not a profile.
4. Proposal via ledger.propose(kind="workspace_profile", ...,
   confidence=purity) — the existing vocabulary, no fourth mechanism.

Read-only except the ledger proposal it is given a Ledger to make.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from assistant.capabilities.genius.data import kmeans

STATE_KEY = "workspace_profiles"
OTHER = "__other__"


def encode(records: List[Dict[str, Any]], min_count: int = 2) -> Dict[str, Any]:
    """Session records -> (feature_matrix, vocabulary) deterministically.

    Returns {"points", "apps", "monitors", "max_workspace"} where
    points[i] encodes records[i] in the order:
    [app one-hot..., monitor one-hot..., workspace, sin(h), cos(h)].
    """
    if not records:
        raise ValueError("no session records to cluster")
    app_counts: Dict[str, int] = {}
    for r in records:
        app_counts[str(r["app"])] = app_counts.get(str(r["app"]), 0) + 1
    apps = sorted(a for a, c in app_counts.items() if c >= min_count)
    known = set(apps)
    monitors = sorted({str(r["monitor"]) for r in records})
    max_workspace = max(int(r["workspace"]) for r in records)
    max_workspace = max(max_workspace, 1)

    points: List[List[float]] = []
    for r in records:
        app = str(r["app"]) if str(r["app"]) in known else OTHER
        app_vec = [1.0 if a == app else 0.0 for a in apps + ([OTHER] if OTHER not in known else [])]
        mon_vec = [1.0 if m == str(r["monitor"]) else 0.0 for m in monitors]
        workspace = (int(r["workspace"]) - 1) / max_workspace
        hour = int(r["hour"]) % 24
        angle = 2.0 * math.pi * hour / 24.0
        points.append(app_vec + mon_vec + [workspace, math.sin(angle), math.cos(angle)])
    return {"points": points, "apps": apps + ([OTHER] if OTHER not in known else []),
            "monitors": monitors, "max_workspace": max_workspace}


def _modal(rows: List[Dict[str, Any]]) -> tuple:
    triple = {}
    for r in rows:
        key = (str(r["app"]), str(r["monitor"]), int(r["workspace"]))
        triple[key] = triple.get(key, 0) + 1
    best = max(triple.items(), key=lambda kv: (kv[1], kv[0]))
    return best[0], best[1]


def profile_name(app: str, monitor: str, workspace: int) -> str:
    """Deterministic, human-readable profile name from the modal triple."""
    slug = f"{app}-{monitor}-ws{workspace}".lower()
    return "".join(c if (c.isalnum() or c in "-_.") else "-" for c in slug)[:60]


def cluster(records: List[Dict[str, Any]], k: Optional[int] = None,
            min_support: int = 3, purity: float = 0.6,
            min_count: int = 2, seed: int = 42) -> Dict[str, Any]:
    """Cluster sessions and return the consistent ones as candidate profiles.

    Returns {"n_sessions", "k", "profiles": [{"name", "app", "monitor",
    "workspace", "hours_mean", "support", "purity", "members"}],
    "rejected_clusters": int}. Pure: no I/O, no ledger, no state.
    """
    enc = encode(records, min_count=min_count)
    n = len(records)
    if k is None:
        k = min(8, max(2, round(math.sqrt(n / 2))))
    k = max(1, min(k, n))
    if n < 2 or k < 2:
        return {"n_sessions": n, "k": k, "profiles": [], "rejected_clusters": 0}

    result = kmeans(enc["points"], k, seed=seed)
    assignments = result["labels"]

    profiles: List[Dict[str, Any]] = []
    rejected = 0
    for cid in sorted(set(assignments)):
        members = [r for r, a in zip(records, assignments) if a == cid]
        if len(members) < min_support:
            rejected += 1
            continue
        (app, monitor, workspace), hits = _modal(members)
        p = hits / len(members)
        if p < purity:
            rejected += 1
            continue
        hours = [int(r["hour"]) for r in members]
        profiles.append({
            "name": profile_name(app, monitor, workspace),
            "app": app, "monitor": monitor, "workspace": workspace,
            "hours_mean": round(sum(hours) / len(hours), 2),
            "support": len(members), "purity": round(p, 3),
        })
    profiles.sort(key=lambda pr: (-pr["support"], pr["name"]))
    return {"n_sessions": n, "k": k, "profiles": profiles,
            "rejected_clusters": rejected}


def propose_profiles(records: List[Dict[str, Any]], ledger,
                     k: Optional[int] = None, min_support: int = 3,
                     purity: float = 0.6, reason: Optional[str] = None,
                     seed: int = 42) -> Dict[str, Any]:
    """Cluster + emit one ledger proposal per consistent profile.

    The proposal diff carries the observed facts only — which is what a
    workspace profile IS at this layer: a named, evidenced co-occurrence
    the user can approve and act on with the shell's own per-monitor and
    workspace settings. Returns {"proposals": [pid...], "summary"}.
    """
    summary = cluster(records, k=k, min_support=min_support, purity=purity,
                      seed=seed)
    pids: List[int] = []
    for pr in summary["profiles"]:
        why = reason or (f"app '{pr['app']}' co-occurs with monitor "
                         f"{pr['monitor']} workspace {pr['workspace']} "
                         f"({pr['support']} sessions, purity {pr['purity']}, "
                         f"mean hour {pr['hours_mean']:g})")
        pids.append(ledger.propose(
            "workspace_profile", f"profile:{pr['name']}",
            {"profile": pr}, why, pr["purity"]))
    return {"proposals": pids, "summary": summary}
