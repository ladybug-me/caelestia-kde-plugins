"""settings.dbus_surface — the quarantined, opt-in DBus surface
(the former proposals/2026-09-26-c-dbus-surface.md, pruned in
exponential-build-3 Phase 2 once implemented — this docstring and
git history carry the design).

THE QUARANTINE (the pkgprobe pattern, second module): this is one of
exactly TWO modules in ``assistant/`` permitted to import
``subprocess`` — the import-policy lint
(``diagnostics/schema_lint.py``) carries a per-module carve-out for
``dbus_surface.py`` alone, and a test pins the exemption set to
exactly {pkgprobe.py, dbus_surface.py}. Every other module stays
under the zero-tolerance rule.

What it covers (the four surfaces the watched shell.json CANNOT
reach — the proposal's own table):

- kwinrc companion writes (``kwriteconfig6``) — the target repo's own
  precedent (lock-screen section, CONTRIBUTING.md);
- display topology (``kscreen-doctor``) — read via ``--outputs``,
  write via a machine-derived ``--output <name> --mode <mode>`` (the
  values come FROM the tool's own observed output, never free text);
- power profiles (``powerprofilesctl``) — read the table, set one of
  the OBSERVED profile names;
- KWin scripting (``dbus-send`` to org.kde.KWin) — the only scripting
  entry point that survives across KDE releases; irreversible by
  catalog declaration, never auto-run.

The discipline, line by line (the proposal's safety contract):

- FIXED ARGUMENT ARRAYS: every spawned command is
  ``subprocess.run([binary, *fixed_args])`` — never ``shell=True``,
  never free-text interpolation. Values fill typed slots
  (bool/int/enum) or are machine-derived from a prior READ;
- OFF BY DEFAULT: nothing runs unless the capability manifest enables
  ``dbus_surface`` (a file edit — never an NL request), the same
  kill-switch posture as package_audit;
- RISK-CLASSIFIED BEFORE EXECUTION: every catalog command carries its
  static risk class; DESTRUCTIVE classifications are withheld
  entirely (none are cataloged — adding one is a reviewable diff that
  must also change the pinning test);
- SHORT-LIVED: one bounded ``run`` per invocation, no resident
  process, zero steady-state memory;
- UNDO WHERE THE TOOL SUPPORTS IT: kwriteconfig6 writes record the
  previous value (read via the same fixed argv shape through
  kreadconfig6) and undo writes it back; kscreen mode changes record
  the observed prior mode; power profile sets record the prior active
  profile; KWin script loads are declared irreversible and refuse to
  run without an explicit ``confirm_irreversible=True`` — the second
  confirmation the proposal demands;
- HONEST DEGRADATION: missing binaries, timeouts, the disabled
  capability, and unknown commands return error dicts, never
  exceptions.

This is a LIBRARY surface, not a conversational one: no natural
language request reaches it, no CLI route applies a write. The caller
(a QML service or the agent layer) renders ``plan_write`` (the dry-run
artifact), obtains consent, then calls ``run_write``; the returned
undo record is the caller's to persist and later hand to
``apply_undo``.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from assistant.core import features as capabilities

__all__ = [
    "CATALOG", "catalog_lines", "plan_write", "run_write", "apply_undo",
    "read_kscreen_outputs", "read_power_profiles", "run_read",
]

_TIMEOUT_SECONDS = 10

# ---------------------------------------------------------------------------
# The command catalog: every entry is a complete fixed command shape.
# Adding an entry (or a risk class above STATE_CHANGING) is a reviewable
# diff: the pinning test asserts the exact ids and classes, and
# the package rationale (see module docstrings) §16 must grow with it.
# ---------------------------------------------------------------------------

CATALOG: Tuple[Dict[str, Any], ...] = (
    {
        "id": "kwinrc.nightcolor.active",
        "binary": "kwriteconfig6",
        "group": "NightColor",
        "key": "Active",
        "value_kind": "bool",
        "risk": "STATE_CHANGING",
        "reversible": True,
        "note": "Night Color master switch (the kwinrc companion the "
                "shell.json watched file cannot reach)",
    },
    {
        "id": "kwinrc.lockscreen.timeout",
        "binary": "kwriteconfig6",
        "group": "ScreenSaver",
        "key": "Timeout",
        "value_kind": "int",
        "value_min": 0,
        "value_max": 7200,
        "risk": "STATE_CHANGING",
        "reversible": True,
        "note": "the lock-screen section precedent from the target "
                "repo's own CONTRIBUTING.md",
    },
    {
        "id": "kscreen.mode.set",
        "binary": "kscreen-doctor",
        "argv": ("output", "{output}", "mode", "{mode}"),
        "value_kind": "observed-pair",  # {output, mode} from the read
        "risk": "STATE_CHANGING",
        "reversible": True,
        "note": "display mode change; the output name and mode come from "
                "the tool's own observed topology",
    },
    {
        "id": "powerprofiles.set",
        "binary": "powerprofilesctl",
        "argv": ("set", "{profile}"),
        "value_kind": "observed",  # from read_power_profiles
        "risk": "STATE_CHANGING",
        "reversible": True,
        "note": "sets one of the OBSERVED profile names",
    },
    {
        "id": "kwin.script.unload",
        "binary": "dbus-send",
        "argv": ("--session", "--dest=org.kde.KWin", "/Scripting",
                 "org.kde.kwin.Scripting.unloadScript", "string:{name}"),
        "value_kind": "name",
        "risk": "STATE_CHANGING",
        "reversible": False,
        "note": "unloads a loaded KWin script by name; IRREVERSIBLE per "
                "the proposal (no undo record exists) — requires "
                "confirm_irreversible=True",
    },
)

_CATALOG_IDS = frozenset(entry["id"] for entry in CATALOG)


# ---------------------------------------------------------------------------
# Spawning (the ONLY place subprocess appears).
# ---------------------------------------------------------------------------


def _spawn(argv: List[str]) -> Dict[str, Any]:
    """One short-lived, bounded, fixed-array run. Returns the completed
    result as a dict (stdout/stderr/rc) or an honest error dict."""
    try:
        completed = subprocess.run(  # noqa: S603 — fixed array, caller-built
            argv, capture_output=True, text=True,
            timeout=_TIMEOUT_SECONDS, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"{argv[0]} timed out after {_TIMEOUT_SECONDS}s"}
    except OSError as exc:
        return {"error": f"{argv[0]} failed to start: "
                         f"{type(exc).__name__}: {exc}"}
    return {"rc": completed.returncode, "stdout": completed.stdout,
            "stderr": completed.stderr}


def _guard() -> Optional[Dict[str, Any]]:
    """The kill-switch: the capability manifest must enable the surface
    (a file edit, never an NL request)."""
    if not capabilities.enabled("dbus_surface"):
        return {"error": "dbus_surface is disabled in the capability "
                         "manifest (edit capabilities.json to enable; "
                         "it cannot be enabled by a request)",
                "capability": "dbus_surface",
                "enabled": False}
    return None


def _spec(command_id: str) -> Optional[Dict[str, Any]]:
    return next((e for e in CATALOG if e["id"] == command_id), None)


def _validate_value(spec: Dict[str, Any], value: Any) -> Optional[str]:
    kind = spec.get("value_kind")
    if kind == "bool":
        if not isinstance(value, bool):
            return "this command takes true or false"
    elif kind == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            return "this command takes an integer"
        lo, hi = spec.get("value_min"), spec.get("value_max")
        if lo is not None and value < lo:
            return f"below the catalog minimum {lo}"
        if hi is not None and value > hi:
            return f"above the catalog maximum {hi}"
    elif kind == "observed":
        if not isinstance(value, str) or not value.strip() or len(value) > 80:
            return ("this command takes a value observed from the "
                    "matching read probe (run the read first)")
    elif kind == "observed-pair":
        if not isinstance(value, dict) \
                or not isinstance(value.get("output"), str) \
                or not isinstance(value.get("mode"), str) \
                or not value["output"].strip() or not value["mode"].strip() \
                or len(value["output"]) > 40 or len(value["mode"]) > 40:
            return ("this command takes {output, mode} from the observed "
                    "topology (run read_kscreen_outputs first)")
    elif kind == "name":
        if not isinstance(value, str) or not value.strip() or len(value) > 64:
            return "this command takes a short script name"
    return None


# ---------------------------------------------------------------------------
# kwriteconfig6 / kreadconfig6 helpers (the fixed argv shapes).
# ---------------------------------------------------------------------------


def _kwrite_argv(spec: Dict[str, Any], value: Any) -> List[str]:
    return [spec["binary"], "--file", "kwinrc", "--group",
            spec["group"], "--key", spec["key"], str(value).lower()
            if isinstance(value, bool) else str(value)]


def _kread_argv(spec: Dict[str, Any]) -> List[str]:
    return ["kreadconfig6", "--file", "kwinrc", "--group",
            spec["group"], "--key", spec["key"]]


# ---------------------------------------------------------------------------
# The read probes (read-only, still behind the kill-switch).
# ---------------------------------------------------------------------------


def read_kscreen_outputs() -> Dict[str, Any]:
    """``kscreen-doctor --outputs`` parsed into a topology dict: per
    output, its name, modes and current mode. Read-only."""
    off = _guard()
    if off:
        return off
    out = _spawn(["kscreen-doctor", "--outputs"])
    if "error" in out:
        return out
    if out["rc"] != 0:
        return {"error": "kscreen-doctor exited "
                         f"{out['rc']}", "stderr": out["stderr"][:400]}
    outputs: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}
    for line in out["stdout"].splitlines():
        stripped = line.strip()
        if stripped.startswith("Output:"):
            current = {"raw": stripped, "modes": []}
            outputs.append(current)
        elif current is not None and "Modes:" in stripped:
            continue
        elif current is not None and stripped.startswith("Mode:"):
            current.setdefault("modes", []).append(stripped)
            if "*" in stripped:  # the active mode is starred
                # "Mode: 1920x1080@60*" -> "1920x1080@60"
                fields = stripped.split()
                if len(fields) >= 2:
                    current["current_mode"] = fields[1].rstrip("*")
                    current["current_mode_line"] = stripped
        elif current is not None and stripped.startswith("Name:"):
            fields = stripped.split(":", 1)
            if len(fields) == 2:
                current["name"] = fields[1].strip()
    return {"outputs": outputs, "n": len(outputs),
            "note": "machine-derived from kscreen-doctor's own output; "
                    "the write surface only accepts values seen here"}


def read_power_profiles() -> Dict[str, Any]:
    """``powerprofilesctl`` parsed into the profile table (which are
    available, which is active). Read-only."""
    off = _guard()
    if off:
        return off
    out = _spawn(["powerprofilesctl"])
    if "error" in out:
        return out
    if out["rc"] != 0:
        return {"error": f"powerprofilesctl exited {out['rc']}",
                "stderr": out["stderr"][:400]}
    profiles: List[Dict[str, Any]] = []
    for line in out["stdout"].splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # rows look like "  performance:", "* balanced:", or
        # "  power-saver:" (the starred one is active)
        active = stripped.startswith("*")
        if active:
            stripped = stripped[1:].strip()
        name = stripped.split(":")[0].strip()
        if name:
            profiles.append({"name": name, "active": active})
    return {"profiles": profiles,
            "active": next((p["name"] for p in profiles if p["active"]),
                           None),
            "note": "machine-derived from powerprofilesctl's own output"}


def run_read(command_id: str) -> Dict[str, Any]:
    """The read probes by catalog intent (dispatch helper): 'kscreen'
    reads the topology, 'power' reads the profile table. Read-only."""
    if command_id in ("kscreen", "kscreen.mode.set"):
        return read_kscreen_outputs()
    if command_id in ("power", "powerprofiles.set"):
        return read_power_profiles()
    return {"error": f"no read probe for {command_id!r} "
                     f"(have: kscreen, power)"}


# ---------------------------------------------------------------------------
# The write spine: plan (dry-run render) -> run (gated) -> undo.
# ---------------------------------------------------------------------------


def plan_write(command_id: str, value: Any) -> Dict[str, Any]:
    """The DRY-RUN artifact: the exact argv that WOULD run, the static
    risk class, the reversibility statement, and the undo strategy.
    Spawns nothing, writes nothing, works even with the capability
    disabled (you can always LOOK)."""
    spec = _spec(command_id)
    if spec is None:
        return {"error": f"unknown command {command_id!r} "
                         f"(catalog: {sorted(_CATALOG_IDS)})"}
    bad = _validate_value(spec, value)
    if bad:
        return {"error": f"invalid value for {command_id}: {bad}"}
    argv = _render_argv(spec, value)
    out: Dict[str, Any] = {
        "command": command_id,
        "argv": argv,
        "risk": spec["risk"],
        "reversible": spec["reversible"],
        "note": spec["note"],
        "dry_run": True,
    }
    if not spec["reversible"]:
        out["requires"] = "run_write(..., confirm_irreversible=True) — " \
                          "no undo record exists for this command"
    elif spec["binary"] == "kwriteconfig6":
        out["undo_strategy"] = "read the previous value (kreadconfig6, " \
                               "same fixed shape) then write it back"
    elif command_id == "kscreen.mode.set":
        out["undo_strategy"] = "re-apply the mode from the pre-change " \
                               "topology snapshot"
    elif command_id == "powerprofiles.set":
        out["undo_strategy"] = "set the previously active profile"
    return out


def _scalar_str(value: Any) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)


def _render_argv(spec: Dict[str, Any], value: Any) -> List[str]:
    if spec["binary"] == "kwriteconfig6":
        return _kwrite_argv(spec, value)
    argv = [spec["binary"]]
    pair = value if isinstance(value, dict) else {}
    for piece in spec.get("argv", ()):
        argv.append(piece.replace("{output}", str(pair.get("output", "")))
                    .replace("{mode}", str(pair.get("mode", "")))
                    .replace("{profile}", _scalar_str(value))
                    .replace("{name}", _scalar_str(value))
                    .replace("{value}", _scalar_str(value)))
    return argv


def run_write(command_id: str, value: Any,
              confirm_irreversible: bool = False,
              _prev: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute ONE catalog write. The caller must have shown
    plan_write's artifact and obtained consent — this function is the
    apply gate, and nothing conversational ever calls it.

    Returns the run report plus, for reversible commands, the UNDO
    RECORD (a plain dict): hand it to ``apply_undo`` to revert. An
    undo record is only produced when the write actually succeeded.
    """
    off = _guard()
    if off:
        return off
    spec = _spec(command_id)
    if spec is None:
        return {"error": f"unknown command {command_id!r}"}
    bad = _validate_value(spec, value)
    if bad:
        return {"error": f"invalid value: {bad}"}
    if not spec["reversible"] and not confirm_irreversible:
        return {"error": f"{command_id} is irreversible per the catalog; "
                         "pass confirm_irreversible=True after the second "
                         "confirmation the proposal demands"}
    # DESTRUCTIVE risk classes are structurally absent from the catalog;
    # the guard stays so a future edit cannot silently bypass review.
    if spec["risk"] in ("PRIVILEGED", "DESTRUCTIVE"):
        return {"error": f"{command_id} is classified {spec['risk']} — "
                         "withheld from execution entirely"}

    # capture the pre-change state for the undo record
    undo: Optional[Dict[str, Any]] = None
    if spec["reversible"] and spec["binary"] == "kwriteconfig6":
        old = _spawn(_kread_argv(spec))
        if "error" not in old and old.get("rc") == 0:
            undo = {"command": command_id, "kind": "kwriteconfig6",
                    "old_value": _coerce(spec, old["stdout"].strip()),
                    "at": datetime.now().isoformat()}
    elif command_id == "kscreen.mode.set":
        snap = read_kscreen_outputs()
        if "error" not in snap:
            prior = _prev if isinstance(_prev, dict) else snap
            # the inverse pair: the output being changed, its CURRENT
            # (pre-change) observed mode
            target_output = value.get("output") \
                if isinstance(value, dict) else ""
            old_pair = next(
                ({"output": o.get("name", ""),
                  "mode": o.get("current_mode", "")}
                 for o in prior.get("outputs", [])
                 if o.get("name") == target_output and o.get("current_mode")),
                None)
            if old_pair is not None:
                undo = {"command": command_id, "kind": "kscreen",
                        "old_value": old_pair,
                        "at": datetime.now().isoformat()}
    elif command_id == "powerprofiles.set":
        snap = read_power_profiles()
        if "error" not in snap and snap.get("active"):
            undo = {"command": command_id, "kind": "powerprofiles",
                    "old_value": snap["active"],
                    "at": datetime.now().isoformat()}

    argv = _render_argv(spec, value)
    out = _spawn(argv)
    if "error" in out:
        return {"error": out["error"], "argv": argv}
    if out["rc"] != 0:
        return {"error": f"{command_id} exited {out['rc']}",
                "argv": argv, "stderr": out["stderr"][:400]}
    report: Dict[str, Any] = {"command": command_id, "argv": argv,
                              "applied": True, "risk": spec["risk"],
                              "reversible": spec["reversible"]}
    if undo is not None:
        report["undo_record"] = undo
    return report


def _coerce(spec: Dict[str, Any], raw: str) -> Any:
    """Coerce a value READ back from the tool into the spec's own type
    (kreadconfig6 prints 'true'/'false'/digits; the write side wants
    bool/int) — so an undo record round-trips through validation."""
    if spec.get("value_kind") == "bool":
        if raw.lower() in ("true", "1", "yes", "on"):
            return True
        if raw.lower() in ("false", "0", "no", "off", ""):
            return False
    if spec.get("value_kind") == "int":
        try:
            return int(raw)
        except ValueError:
            return raw
    return raw


def apply_undo(record: Dict[str, Any],
               confirm_irreversible: bool = False) -> Dict[str, Any]:
    """Revert ONE write using its undo record (bounded: one record, one
    inverse write — never a replay loop)."""
    off = _guard()
    if off:
        return off
    kind = record.get("kind")
    if kind == "kwriteconfig6":
        return run_write(record["command"], record.get("old_value", ""))
    if kind == "powerprofiles":
        return run_write("powerprofiles.set", record.get("old_value", ""))
    if kind == "kscreen":
        old_pair = record.get("old_value")
        if not (isinstance(old_pair, dict) and old_pair.get("output")
                and old_pair.get("mode")):
            return {"error": "the undo record carries no observed "
                             "prior mode; nothing to revert to"}
        return run_write("kscreen.mode.set", old_pair)
    return {"error": f"unknown undo record kind {kind!r}"}


# ---------------------------------------------------------------------------
# Introspection.
# ---------------------------------------------------------------------------


def catalog_lines() -> List[str]:
    """The reviewable one-line-per-command listing (ids, binaries,
    risk, reversibility)."""
    lines = [f"dbus surface catalog ({len(CATALOG)} commands, "
             f"kill-switch: capabilities.json dbus_surface="
             f"{capabilities.enabled('dbus_surface')}):"]
    for spec in CATALOG:
        lines.append(f"  {spec['id']}  [{spec['binary']}, "
                     f"{spec['risk']}, "
                     f"{'reversible' if spec['reversible'] else 'IRREVERSIBLE'}]"
                     f"  {spec['note']}")
    return lines
