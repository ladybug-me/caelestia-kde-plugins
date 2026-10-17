"""assistant.core.features — the per-install capability manifest.

A static, user-editable config that lists which goal archetypes / surfaces
are enabled on THIS install, so a community deployment can turn off
package-auditing or the DBus surface without a code change. Every new
capability added in this codebase registers itself here.

Posture (the generative layer's own pattern, applied system-wide):
a capability defaults to ON only when it is read-only and zero-risk —
no process spawning, no write path, no new import surface. Everything
that shells out, writes, or reaches beyond the offline core defaults
OFF and must be flipped by editing this file (never by a natural
language request — the kill-switch stays a file edit, exactly like the
DBus proposal's settings.dbus.enabled rule).

File: $CAELESTIA_ASSIST_CAPABILITIES if set, else
~/.config/caelestia-assistant/capabilities.json — a JSON object mapping
capability name -> true/false. Unknown keys are ignored (forward
compat); a broken file degrades to defaults with a note, never a crash.

Read-only module: nothing here writes the manifest; the user owns it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

__all__ = ["DEFAULTS", "capabilities_path", "manifest", "enabled", "render"]

# The shipped posture. ON = read-only, zero-risk, offline.
# OFF = any new import surface, any process, any write.
DEFAULTS: Dict[str, bool] = {
    # -- read-only, zero-risk: on by default ------------------------------
    "fsbrain": True,              # filesystem analyses (genius/fsbrain)
    "config_hygiene": True,       # lint + drift report + consented reconcile
    "log_triage": True,           # sysintel template mining -> issue drafts
    "screenshot_diff": True,      # pixel-region hashing between two files
    "notification_triage": True,  # PURE classification of event records
    "lexicon_sharing": True,      # cortex lexicon export/import/forget (CLI,
                                  # offline, no network; import is an explicit
                                  # user command with rollback — phase 2.6)
    "eval_arena": True,           # the measurement arena incl. the
                                  # metamorphic suite (read-only, in-process,
                                  # dev sets only unless --sealed is asked for)
    "confusable_clarifier": True,  # upgrades the generic ambiguity ask to
                                   # the mined pair's own question (read-only
                                   # text swap over the committed artifact)
    "label_fusion": True,          # Dawid-Skene vote fusion may demote a
                                   # contested confident route to an ask
                                   # (never promotes; read-only)
    "shell_cli_grammar": True,     # shellkb: the shell CLIs' typed option
                                   # grammar, induced from upstream file text
                                   # (read-only; nothing executed)
    "shell_explain": True,         # shellkb: token-by-token command-line
                                   # explanation, read-only glob previews
                                   # (os.scandir), destructive-pattern flags;
                                   # the verdict is always
                                   # SUGGESTED_NOT_EXECUTED
    "shell_howto": True,           # shellkb: offline how-to answers over a
                                   # self-authored CC0 corpus (BM25 + the
                                   # existing time-expression parser); every
                                   # step SUGGESTED_NOT_EXECUTED
    "shell_json_merge": True,      # shellkb: structured diff (Myers paths +
                                   # ordered tree-edit distance) and
                                   # three-way merge PROPOSALS for config
                                   # JSON; pure math over caller-supplied
                                   # documents, never a write
    "shell_conflicts": True,       # shellkb: PubGrub-style dependency
                                   # explanations over a bounded DPLL-style
                                   # search; any installed-version facts
                                   # still come only from the read-only
                                   # package_audit probe (or the caller)
    "engine_telemetry": True,     # cortex engine coverage/accuracy report +
                                  # opt-in Laplace-DP export (read-only,
                                  # stdout only — exponential-build-3 F3)
    "seasonal_patterns": True,    # brain: matrix profile + SAX motifs/
                                  # discords, PELT changepoints, Holt-
                                  # Winters seasonality over caller-passed
                                  # series (pure math, read-only)
    "burst_detection": True,      # brain: Hawkes self-excitation fit and
                                  # burst detection over caller-passed event
                                  # times; reports correlation, never
                                  # causation (pure math, read-only)
    "gp_preferences": True,       # brain: Gaussian-process preference
                                  # learning over pairwise answers (Cholesky,
                                  # bounded at 50 items); emits ONE
                                  # registry-validated setter proposal,
                                  # SUGGESTED_NOT_EXECUTED
    "user_rules": True,           # brain: user-authored event-condition-
                                  # action rules on a Rete network; the
                                  # rules FILE is the opt-in and every
                                  # fired action is a ledger proposal,
                                  # never an applied change
    "size_sketches": True,        # brain: Space-Saving heavy hitters and
                                  # t-digest quantiles over caller-passed
                                  # sizes; counts are labeled estimates
                                  # with reported max error (pure math)
    "nlg_templates": True,        # brain: template NLG with discourse
                                  # planning and aggregation over plan
                                  # changes (deterministic; no model)
    "extractive_qa": True,        # brain: extractive QA over the repo docs
                                  # and caller-supplied notes — BM25 passage
                                  # retrieval, answer-type detection, VERBATIM
                                  # spans with doc/section/line provenance;
                                  # thin evidence says THIN, none says
                                  # NOT_FOUND (no generation, ever)
    # -- optional model tier (Group E): OFF, and never on the offline core
    "model_reranker": False,      # generative: sub-1B loopback reranker over
                                  # the router's OWN candidate list; output
                                  # constrained to candidate names, marked
                                  # MODEL_SUGGESTED, plan/apply gates untouched
    # -- surfaces that shell out or extend the import surface: OFF ------
    "package_audit": False,       # read-only package-manager query (quarantined)
    "dbus_surface": False,        # kwriteconfig6/kscreen/powerdevil/KWin scripts
    "notification_observation": False,  # live DBus notification listening
}


def capabilities_path() -> Path:
    env = os.environ.get("CAELESTIA_ASSIST_CAPABILITIES")
    if env:
        return Path(env)
    config_home = os.environ.get("XDG_CONFIG_HOME") or \
        str(Path.home() / ".config")
    return Path(config_home) / "caelestia-assistant" / "capabilities.json"


def manifest() -> Dict[str, Any]:
    """The effective manifest: defaults, overridden by the user's file."""
    out: Dict[str, Any] = dict(DEFAULTS)
    out["_path"] = str(capabilities_path())
    out["_source"] = "defaults"
    try:
        user = json.loads(capabilities_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out  # missing or broken file: defaults, honestly
    if not isinstance(user, dict):
        out["_note"] = "user manifest is not an object; defaults stand"
        return out
    for key, value in user.items():
        if key.startswith("_"):
            continue
        if isinstance(value, bool):
            out[key] = value
    out["_source"] = "defaults+user"
    return out


def enabled(name: str) -> bool:
    """Is this capability enabled on this install? (Default posture when
    the user has said nothing — the safe direction.)"""
    return bool(manifest().get(name, DEFAULTS.get(name, False)))


def render() -> str:
    """The `caelestia-assist capabilities` card."""
    current = manifest()
    lines = [
        "caelestia-assist — capability manifest",
        f"  file: {current['_path']} (edit it to flip a switch; NL requests "
        "cannot)",
        "",
    ]
    for name in sorted(DEFAULTS):
        state = "on " if current.get(name) else "off"
        lines.append(f"  [{state}]  {name}")
    # The DBus surface's command catalog: the reviewable one-line-per-
    # command listing, shown when the capability is on (the operator
    # can see exactly what can spawn without reading the module).
    if current.get("dbus_surface"):
        try:
            from assistant.capabilities.settings.dbus_surface import catalog_lines
            lines.append("")
            lines.extend("  " + line for line in catalog_lines())
        except Exception:
            pass  # the catalog listing is best-effort, never fatal
    if current.get("_source") != "defaults+user":
        lines.append("")
        lines.append("  (defaults in effect — the user file has not "
                     "overridden anything)")
    return "\n".join(lines)
