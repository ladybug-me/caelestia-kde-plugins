"""agent.pkgprobe — the quarantined, opt-in package-list probe.

THE EXECUTOR SURFACE (the DBus proposal's exact pattern, applied here):
this module spawns nothing itself — every probe goes through the
executor (``assistant/executor``), the single argv-arrays-only runner.
The quarantine registry (``diagnostics/schema_lint.py``) names exactly
two subprocess users: ``executor/runner.py`` and ``dbus_surface.py``
(pending its own migration), and tests pin that set.

What it does (§2's discipline, line by line):

- READ-ONLY: the only executed commands are list/query operations —
  ``pacman -Q``, ``dpkg-query -W``, ``flatpak list``. Nothing installs,
  upgrades, removes, or writes anything, ever;
- TYPED ACTION PLANS: every call is an executor ActionPlan over argv
  tuples — never ``shell=True``, never string interpolation, the
  upstream CONTRIBUTING.md "pass arguments as a list" rule;
- OFF BY DEFAULT: the probe refuses to run unless the capability
  manifest enables ``package_audit`` (a file edit — never an NL
  request), the same kill-switch posture as the DBus proposal;
- SHORT-LIVED: one bounded ``run`` per invocation (~15-30 ms fork+exec),
  no resident process, zero steady-state memory (§5 compliant);
- HONEST DEGRADATION: missing package managers, timeouts, or the
  disabled capability return error dicts, never exceptions.

The result feeds a STATIC, LOCAL keyword list (``_PACKAGE_NOTES``) —
grouping and naming flags only. This is deliberately NOT a CVE feed:
no network, no freshness claims about versions, nothing external.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from assistant.core import features as capabilities
from assistant.executor import ActionPlan, Step, run as executor_run

__all__ = ["query_installed", "stale_match", "PROBES"]

# The allow-listed probes: (binary, fixed args, output format).
# Adding an entry here is a reviewable diff with a the package rationale line —
# the list is pinned by tests.
PROBES: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("pacman", ("-Q",), "name version"),          # Arch-family local DB
    ("dpkg-query", ("-W", "-f=${Package} ${Version}\\n"), "name version"),
    ("flatpak", ("list", "--app", "--columns=application,version"),
     "name version"),
)

_TIMEOUT_SECONDS = 10


def query_installed(managers: Optional[List[str]] = None) -> Dict[str, Any]:
    """Read-only list of installed packages from the FIRST available
    package manager (or the named ones). Returns
    ``{"manager", "packages": [(name, version)], "n"}`` or an honest
    error dict. Refuses when the capability is disabled."""
    if not capabilities.enabled("package_audit"):
        return {"error": "package_audit is disabled in the capability "
                         "manifest (edit capabilities.json to enable; "
                         "it cannot be enabled by a request)",
                "capability": "package_audit",
                "enabled": False}
    wanted = set(managers) if managers else {p[0] for p in PROBES}
    for binary, args, _fmt in PROBES:
        if binary not in wanted:
            continue
        if not os.path.isfile(f"/usr/bin/{binary}") and \
                not os.path.isfile(f"/bin/{binary}"):
            continue  # not installed on this machine: try the next
        try:
            result = executor_run(ActionPlan(
                steps=(Step((binary, *args), timeout_s=_TIMEOUT_SECONDS),),
                reversibility="read_only", blast_radius="none",
                description=f"package probe: {binary}"))
        except Exception as exc:  # pragma: no cover - executor is honest
            return {"error": f"{binary} probe failed: "
                             f"{type(exc).__name__}: {exc}"}
        first = result["steps"][0]
        if first.get("error") and "timeout" in first["error"]:
            return {"error": f"{binary} probe failed: {first['error']}"}
        if first.get("error"):
            continue  # binary vanished between probe and run: try next
        if not first["ok"]:
            return {"error": f"{binary} exited {first['returncode']}",
                    "stderr": first["stderr"][:400]}
        packages: List[Tuple[str, str]] = []
        for line in first["stdout"].splitlines():
            parts = line.split()
            if len(parts) >= 2:
                packages.append((parts[0], parts[1]))
            elif parts:
                packages.append((parts[0], ""))
        return {"manager": binary, "packages": packages,
                "n": len(packages), "enabled": True}
    return {"error": "no known package manager found "
                     "(pacman / dpkg-query / flatpak)",
            "enabled": True}


# Static, LOCAL keyword notes — grouping and naming flags only.
# Deliberately NOT a vulnerability feed: no versions, no CVEs, no
# network, no freshness claims. Entries are reviewable lines.
_PACKAGE_NOTES: Dict[str, str] = {
    "-git": "a -git build: version numbers are commit hashes, not "
            "releases — comparing them to release notes misleads",
    "-svn": "a source-control build: same caveat as -git",
    "-beta": "pre-release channel",
    "-gitdebug": "debug symbols build; large, safe to remove if unused",
    "plasma": "KDE Plasma family: group for shell-side regression rules",
    "quickshell": "the caelestia shell host: pin its version when "
                  "reporting issues",
    "caelestia": "caelestia-family package",
    "linux": "kernel package: reboot required after upgrade",
}


def stale_match(packages: List[Tuple[str, str]],
                top: int = 15) -> Dict[str, Any]:
    """Match installed package names against the static local keyword
    list: group counts + per-group flag notes. Read-only, offline, no
    version freshness claims (the list is naming knowledge, not a feed).
    """
    groups: Dict[str, List[str]] = {}
    for name, _version in packages:
        lowered = name.lower()
        for keyword, note in _PACKAGE_NOTES.items():
            if keyword in lowered:
                groups.setdefault(keyword, []).append(name)
                break
    flagged = [{"keyword": k, "note": _PACKAGE_NOTES[k],
                "n": len(v), "packages": sorted(v)[:10]}
               for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))]
    return {
        "n_packages": len(packages),
        "n_groups": len(groups),
        "groups": flagged[:top],
        "algorithm": "static local keyword list match (no network, no "
                     "CVE feed, no version freshness claims)",
    }
