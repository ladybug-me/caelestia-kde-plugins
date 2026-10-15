"""Pre-write sanity simulator for the settings applier (issue #120 Phase 1.2).

Deterministic, named-algorithm checks over the POST-apply state, run by
applier.apply() before anything touches the disk — including single-setting
applies, which are exactly the ones that skip the multi-change preview.

Checks:

1. Minimum tap-target size (WCAG 2.5.8 "Target Size (Minimum)", 24x24 px):
   the projected bar.dock.iconSize below 24 px is reported. The shipped
   control allows 16 px, so this is a WARNING with the citation, not a
   refusal — refusing would break presets the shipped controls themselves
   define (the registry's own `minimal` preset sets 20 px). The assistant
   flags the accessibility cost; the shipped range stays the authority.

2. Projected WCAG contrast (WCAG 2.x relative-luminance contrast ratio, via
   assistant/genius/creative.py::contrast_ratio — the same code the
   caelestia_genius_palette bridge tool uses): when the caller supplies a
   scheme (a {"role": "#hex"} mapping, e.g. exported from the shell's scheme
   system), every role pair is contrast-checked on the projected state and a
   ratio below 4.5:1 (WCAG 1.4.3 AA, normal text) between a text role and a
   background role is a REFUSAL: the apply exits nonzero with nothing
   written. When no scheme is supplied the check is honestly skipped —
   shell.json itself carries no color leaves (colors live in the scheme
   system, outside this write path), and pretending otherwise would be a
   fabricated check.

Refusal semantics: applier.apply raises ApplierError("sanity ... nothing was
written") before the backup file is touched, so a refused apply leaves the
target, its backup, and the undo history byte-identical. Dry-runs (write=False)
never run the gate at all — the caller sees the same dry-run result as before.

The scheme parameter is optional in every signature; existing callers are
unaffected.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# WCAG 2.5.8 minimum target size (CSS px). Shipped control range for
# bar.dock.iconSize is 16-96 (tools.json); default 32.
TAP_TARGET_MIN_PX = 24
TAP_TARGET_PATH = "bar.dock.iconSize"

# WCAG 1.4.3 AA minimum contrast for normal text.
CONTRAST_MIN_AA = 4.5

# Text roles must be checked against background roles when both are present.
_TEXT_ROLES = ("foreground", "text", "accent")
_BG_ROLES = ("background", "base", "surface")

_SKIPPED_CONTRAST_NOTE = (
    "contrast check skipped: no scheme colors were supplied to this write "
    "path (shell.json carries no color leaves; the scheme system is outside "
    "this file's scope)"
)


def _dotted_get(config: Dict[str, Any], dotted: str) -> Any:
    node: Any = config
    for segment in dotted.split("."):
        if not isinstance(node, dict) or segment not in node:
            return None
        node = node[segment]
    return node


def project(current: Dict[str, Any], entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Deep-copy `current` with the plan's applicable entries merged in —
    the exact state the applier would serialize if it wrote. The deep copy
    is a JSON round-trip (the config is JSON by definition, and json is the
    already-allow-listed module — no import-policy expansion needed)."""
    import json as _json

    projected = _json.loads(_json.dumps(current))
    for entry in entries:
        path = str(entry.get("path", ""))
        if not path or entry.get("error") or entry.get("no_op"):
            continue
        if entry.get("unset"):
            segments = path.split(".")
            node: Any = projected
            for segment in segments[:-1]:
                if not isinstance(node, dict) or segment not in node:
                    node = None
                    break
                node = node[segment]
            if isinstance(node, dict):
                node.pop(segments[-1], None)
            continue
        value = _json.loads(_json.dumps(entry.get("new")))  # JSON-safe copy
        segments = path.split(".")
        node = projected
        for segment in segments[:-1]:
            nxt = node.get(segment)
            if not isinstance(nxt, dict):
                nxt = {}
                node[segment] = nxt
            node = nxt
        node[segments[-1]] = value
    return projected


def _contrast_verdicts(scheme: Dict[str, str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """WCAG contrast for every (text role, background role) pair present in
    the scheme. Returns (verdicts, unparseable_roles)."""
    from assistant.capabilities.genius.creative import contrast_ratio

    verdicts: List[Dict[str, Any]] = []
    bad: List[str] = []
    for text_role in _TEXT_ROLES:
        fg = scheme.get(text_role)
        if fg is None:
            continue
        for bg_role in _BG_ROLES:
            bg = scheme.get(bg_role)
            if bg is None:
                continue
            try:
                result = contrast_ratio(str(fg), str(bg))
            except (ValueError, TypeError):
                bad.append(f"{text_role}/{bg_role}")
                continue
            verdicts.append({
                "text_role": text_role,
                "background_role": bg_role,
                "ratio": result["ratio"],
                "aa_normal_text": result["aa_normal_text"],
            })
    return verdicts, bad


def check(projected: Dict[str, Any],
          scheme: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Run the invariant checks over the POST-apply state.

    Returns {"refusals": [...], "warnings": [...], "notes": [...]}. An empty
    refusals list means the write may proceed; applier.apply raises before
    writing when refusals is non-empty.
    """
    refusals: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    notes: List[str] = []

    # 1. Tap-target size (WCAG 2.5.8) — warning, not refusal (see docstring).
    icon = _dotted_get(projected, TAP_TARGET_PATH)
    if isinstance(icon, (int, float)) and not isinstance(icon, bool):
        if icon < TAP_TARGET_MIN_PX:
            warnings.append({
                "id": "tap-target-below-wcag",
                "path": TAP_TARGET_PATH,
                "message": (f"projected {TAP_TARGET_PATH}={icon:g}px is below "
                            f"the WCAG 2.5.8 minimum target size "
                            f"({TAP_TARGET_MIN_PX}px); the shipped control "
                            "allows it, so the apply proceeds"),
                "cite": (f"WCAG 2.5.8 Target Size (Minimum); "
                         f"assistant/settings/tools.json {TAP_TARGET_PATH} "
                         "range 16-96"),
            })

    # 2. Projected WCAG contrast — refusal, but only with scheme context.
    if scheme:
        verdicts, bad = _contrast_verdicts(scheme)
        for v in verdicts:
            if v["ratio"] is not None and v["ratio"] < CONTRAST_MIN_AA and \
                    v["text_role"] in ("foreground", "text"):
                refusals.append({
                    "id": "contrast-below-aa",
                    "path": f"scheme:{v['text_role']}/{v['background_role']}",
                    "message": (f"projected contrast {v['text_role']} on "
                                f"{v['background_role']} is {v['ratio']:.2f}:1, "
                                f"below the WCAG 1.4.3 AA minimum "
                                f"({CONTRAST_MIN_AA}:1)"),
                    "cite": ("WCAG 1.4.3 Contrast (Minimum); "
                             "assistant/genius/creative.py::contrast_ratio"),
                })
        if bad:
            notes.append("scheme roles with unparseable colors ignored: "
                         + ", ".join(sorted(bad)))
    else:
        notes.append(_SKIPPED_CONTRAST_NOTE)

    return {"refusals": refusals, "warnings": warnings, "notes": notes}
