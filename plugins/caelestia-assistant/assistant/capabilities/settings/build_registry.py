#!/usr/bin/env python3
"""Build assistant/settings/tools.json from the repo's real headers.

The generated-registry pipeline. The ONLY settings-layer piece that reads the
caelestia checkout: runs the stdlib header walker (enumerate.py), applies
the safety classification (curations.py) plus the transcribed shipped
control table (ui_ranges_data.py), and emits the frozen tool table that
registry.py loads at runtime. Deterministic; regenerating against a moved
upstream and diffing is the drift guard (a unittest does that).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import curations
from .enumerate import enumerate_leaves
from .ui_ranges_data import UI_RANGES

REPO_COMMIT = "70ee7da115fde9012e28e393c73d274f55bd2dd7"
_ENUM_NOTE = re.compile(r"enum ([A-Za-z0-9_]+)\s*:\s*([A-Za-z0-9_|]+)")


def parse_default(raw: str, cpp_type: str) -> Tuple[Optional[Any], Optional[str]]:
    """C++ default literal -> JSON value, or (None, reason)."""
    raw = raw.strip()
    if raw == "{}":
        if cpp_type == "QString":
            return "", None
        return None, f"untyped brace default {raw!r}"
    m = re.fullmatch(r'u"([^"]*)"_s', raw)
    if m:
        return m.group(1), None
    m = re.fullmatch(r'QStringLiteral\("([^"]*)"\)', raw)
    if m:
        return m.group(1), None
    if raw == "QString()":
        return "", None
    if raw == "true":
        return True, None
    if raw == "false":
        return False, None
    if re.fullmatch(r"-?\d+", raw):
        return (float(raw) if cpp_type in ("qreal", "double", "float") else int(raw)), None
    if re.fullmatch(r"-?\d+\.\d*", raw) or re.fullmatch(r"\.\d+", raw):
        return float(raw), None
    # Enum-typed default: ClockFormat::Auto -> "Auto" (the metaenum key IS
    # the serialized form; EnumCodec::encode writes the key, codecs.cpp:238-240)
    m = re.fullmatch(r"([A-Za-z0-9_]+)::([A-Za-z0-9_]+)", raw)
    if m and "::" in cpp_type:
        return m.group(2), None
    return None, f"non-representable C++ default literal {raw!r}"


def enum_members_from_note(note: str) -> Optional[List[str]]:
    """'enum ClockFormat: Auto|TwelveHour|TwentyFourHour' -> members."""
    if not note:
        return None
    m = _ENUM_NOTE.search(note)
    return m.group(2).split("|") if m else None


def _preset_value_ok(t: Dict[str, Any], value: Any) -> Tuple[bool, str]:
    """Validate one preset call against its tool spec (build-time gate)."""
    kind = t["kind"]
    if kind == "bool":
        return (True, "") if isinstance(value, bool) else (False, "bool tool needs true/false")
    if kind == "enum":
        return (True, "") if value in (t["enum"] or []) else (False, f"not one of {t['enum']}")
    if kind == "string":
        ok = isinstance(value, str) and 0 < len(value) <= int(t.get("string_max_len") or 64)
        return (True, "") if ok else (False, "non-empty string within max length required")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False, "numeric tool needs a number"
    if t["minimum"] is not None and value < t["minimum"]:
        return False, f"below minimum {t['minimum']}"
    if t["maximum"] is not None and value > t["maximum"]:
        return False, f"above maximum {t['maximum']}"
    if kind == "int" and int(value) != value:
        return False, "int tool needs an integer"
    return True, ""


def build(repo_root: str) -> Dict[str, Any]:
    leaves = enumerate_leaves(repo_root)
    by_path = {r["path"]: r for r in leaves}
    tools: List[Dict[str, Any]] = []
    not_exposed: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {"ui_rows_not_leaves": [], "excluded_by_rule": {}}

    def exclude(path: str, tag: str, reason: str) -> None:
        not_exposed.append({"path": path, "reason": f"[{tag}] {reason}"})
        stats["excluded_by_rule"][tag] = stats["excluded_by_rule"].get(tag, 0) + 1

    scalar_rows = [r for r in leaves if r["kind"] == "scalar"]
    for row in sorted(scalar_rows, key=lambda r: r["path"]):
        path = row["path"]
        excl = curations.exclusion_for(row)
        if excl is not None:
            name, reason = excl
            exclude(path, name, reason)
            continue
        ui = UI_RANGES.get(path)
        if ui is None:
            exclude(path, "no-shipped-control",
                    "no Nexus control touches this leaf; not exposed without "
                    "a UI-grounded citation")
            continue
        control = str(ui.get("control", ""))
        if not curations.control_is_scalar(control):
            exclude(path, "complex-control",
                    f"the shipped control is {control!r}, not a scalar setter")
            continue
        cpp_type = row["type"]
        default, bad = parse_default(row["default"], cpp_type)
        if bad is not None:
            exclude(path, "default", bad)
            continue

        enum_vals: Optional[List[str]] = None
        int_enum: Optional[List[int]] = None
        int_range = None
        extra_fixup_cite: List[List[str]] = []
        cpp_members = enum_members_from_note(row.get("notes", ""))
        is_enum_typed = cpp_type != "QString" and cpp_members is not None

        if path in curations.ENUM_FIXUPS:
            vals, cite = curations.ENUM_FIXUPS[path]
            enum_vals = vals
            kind = "enum"
            extra_fixup_cite = [list(cite)]
        elif is_enum_typed:
            # C++ enum declaration wins over any UI annotation: the
            # metaenum keys ARE the serialized form (codecs.cpp:238-240).
            enum_vals = cpp_members
            kind = "enum"
            extra_fixup_cite = []
        elif control.startswith("SelectRow"):
            ui_list = ui.get("enum")
            vals = [str(v) for v in ui_list] if isinstance(ui_list, list) else []
            # strip parenthetical annotations: "Stretch (Image...)" -> "Stretch"
            vals = [v.split(" (")[0].strip() for v in vals]
            # numeric-coded rows: "0=KDE Grid" / "-1 Random" / "5 attempts"
            num_prefix = [re.match(r"^(-?\d+)", v) for v in vals]
            if all(m is not None for m in num_prefix) and vals:
                ints = [int(m.group(1)) for m in num_prefix]  # type: ignore[union-attr]
                if ints == list(range(min(ints), max(ints) + 1)):
                    kind = "int"
                    int_range = (min(ints), max(ints))
                    enum_vals = None
                else:
                    kind = "enum"
                    int_enum = sorted(set(ints))
                    enum_vals = [str(i) for i in int_enum]
                extra_fixup_cite = []
            elif vals and all(v and " " not in v and "+" not in v for v in vals):
                enum_vals = vals
                kind = "enum"
                extra_fixup_cite = []
            else:
                exclude(path, "dynamic-enum",
                        "the SelectRow's value list is dynamic, numeric-coded "
                        "without per-value ids, or not a frozen enumeration")
                continue
        elif control.startswith("FontCard"):
            kind = "string"
            extra_fixup_cite = []
        else:
            kind = {"bool": "bool", "int": "int", "qreal": "float",
                    "double": "float", "float": "float"}.get(cpp_type)
            if cpp_type == "QString" and kind is None:
                kind = "string"
            extra_fixup_cite = []
        if kind is None:
            exclude(path, "type", f"unhandled C++ type {cpp_type!r}")
            continue
        if cpp_type == "QString" and control.startswith(("StepperRow", "DoubleStepperRow")):
            exclude(path, "string-stepper",
                    "the shipped stepper stores a transformed string "
                    "(e.g. HH:MM); not a passthrough scalar")
            continue
        if cpp_type == "QString" and kind == "string" and not control.startswith("FontCard"):
            exclude(path, "free-string",
                    "free-form string with no frozen enumeration")
            continue

        spec: Dict[str, Any] = {
            "name": curations.tool_name_for(path),
            "path": path,
            "group": curations.group_for(path),
            "kind": kind,
            "default": default,
            "minimum": None, "maximum": None, "step": None, "enum": None,
            "string_max_len": None,
            # enumerate_leaves returns a bool; the TSV form spelled it
            # "yes"/"no" — accept both so neither shape can silently
            # flatten the flag again (2-a-2 audit: it did, for every tool then shipped).
            "global_only": row["global_only"] is True or row["global_only"] == "yes",
            "nouns": [],
            "citations": [],
        }
        if kind == "enum":
            if int_enum is not None:
                spec["enum"] = int_enum  # enum of ints (numeric-coded select)
            else:
                spec["enum"] = enum_vals
        elif kind == "string":
            spec["string_max_len"] = 64
        else:
            ov = curations.CORE_OVERRIDES.get(path)
            if ov is not None:
                spec["minimum"] = ov.get("minimum")
                spec["maximum"] = ov.get("maximum")
                spec["step"] = ov.get("step")
            elif path in curations.RANGE_FIXUPS:
                # Transformed stepper: adopt the range in the STORED unit
                # (see curations.RANGE_FIXUPS for the per-row evidence).
                fix = curations.RANGE_FIXUPS[path]
                spec["minimum"] = fix.get("minimum")
                spec["maximum"] = fix.get("maximum")
                spec["step"] = fix.get("step")
                if "cite" in fix:
                    extra_fixup_cite.append(list(fix["cite"]))  # type: ignore[arg-type]
            elif kind == "int" and int_range is not None:
                spec["minimum"], spec["maximum"] = int_range
                spec["step"] = 1
            else:
                spec["minimum"] = ui.get("min")
                spec["maximum"] = ui.get("max")
                spec["step"] = ui.get("step")
            if kind == "int":
                for k in ("minimum", "maximum", "step"):
                    if spec[k] is not None:
                        spec[k] = int(spec[k])
        core = curations.CORE_TOOLS.get(path)
        if core is not None:
            spec["name"] = core["name"]
            spec["nouns"] = list(core.get("nouns", []))
            for k in ("minimum", "maximum", "step", "enum"):
                if k in core:
                    spec[k] = list(core[k]) if k == "enum" else core[k]

        header_rel = f"shell/plugin/src/Caelestia/Config/{row['header']}"
        cites = [
            [f"{header_rel}:{row['line']}",
             f"CONFIG declaration: {cpp_type} {path.rsplit('.', 1)[-1]} = {row['default']}"],
            [f"{ui['file']}:{ui['line']}",
             f"shipped Nexus control ({control.split('(')[0].strip()})"],
        ]
        cites.extend([list(c) for c in curations.extra_citations_for(path)])
        cites.extend(extra_fixup_cite)
        spec["citations"] = cites
        tools.append(spec)

    leaf_paths = set(by_path)
    stats["ui_rows_not_leaves"] = sorted(k for k in UI_RANGES if k not in leaf_paths)
    # Two-pass naming: short names first; colliding basenames get the full
    # dotted path camel-cased (area prefix included), e.g. setSessionVimKeybinds.
    names = [t["name"] for t in tools]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        for t in tools:
            if t["name"] in dupes and t["path"] not in curations.CORE_TOOLS:
                parts = [p for p in t["path"].split(".") if p]
                t["name"] = "set" + "".join(p[:1].upper() + p[1:] for p in parts)
        names = [t["name"] for t in tools]
        dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise SystemExit(f"tool name collisions: {sorted(dupes)}")
    group_order = {g: i for i, g in enumerate(curations.GROUPS)}
    tools.sort(key=lambda t: (group_order.get(t["group"], 99), t["name"]))
    counts: Dict[str, int] = {}
    for t in tools:
        counts[t["group"]] = counts.get(t["group"], 0) + 1

    # ---- presets: bundles of validated calls (fail the build if invalid) --
    by_name = {t["name"]: t for t in tools}
    presets_out: List[Dict[str, Any]] = []
    for p in curations.PRESETS:
        calls_out = []
        for tool_name, value in p["calls"]:  # type: ignore[index]
            t = by_name.get(str(tool_name))
            if t is None:
                raise SystemExit(f"preset {p['name']!r} references unknown tool {tool_name!r}")
            ok, why = _preset_value_ok(t, value)
            if not ok:
                raise SystemExit(f"preset {p['name']!r} value {value!r} for {tool_name}: {why}")
            calls_out.append({"tool": str(tool_name), "value": value})
        presets_out.append({
            "name": p["name"], "label": p["label"],
            "description": p["description"], "calls": calls_out,
        })

    explain_out = [
        {
            "path": r["path"], "when": r["when"], "answer": r["answer"],
            "cites": list(r["cites"]),  # type: ignore[arg-type]
        }
        for r in curations.EXPLAIN_RULES
    ]

    return {
        "meta": {
            "generator": "assistant.capabilities.settings.build_registry",
            "repo_commit": REPO_COMMIT,
            "notes": (
                "Validation ranges/enums adopted exactly from the shipped "
                "Nexus controls (ui_ranges_data.py) and the C++ enum "
                "declarations; upstream declares no numeric bounds in the "
                "settings schema (type-only hook). not_exposed records every "
                "scalar leaf without a tool, with its reason."
            ),
            "tool_count": len(tools),
            "group_counts": dict(sorted(counts.items(),
                                        key=lambda kv: group_order.get(kv[0], 99))),
            "not_exposed_count": len(not_exposed),
            "excluded_by_rule": stats["excluded_by_rule"],
        },
        "tools": tools,
        "not_exposed": sorted(not_exposed, key=lambda e: e["path"]),
        "presets": presets_out,
        "explain_rules": explain_out,
    }


def find_repo_root(start: Path) -> Optional[Path]:
    for cand in [start, *start.parents]:
        if (cand / "shell" / "plugin" / "src" / "Caelestia" / "Config").is_dir():
            return cand
    return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="build tools.json from a caelestia checkout")
    ap.add_argument("--repo-root", default=None, help="caelestia checkout root")
    ap.add_argument("--output", default=None, help="output path (default: tools.json beside this module)")
    args = ap.parse_args(argv)
    root = Path(args.repo_root) if args.repo_root else find_repo_root(Path(__file__).resolve().parent)
    if root is None or not root.exists():
        print("error: cannot locate a caelestia checkout "
              "(need shell/plugin/src/Caelestia/Config under the repo root)", file=sys.stderr)
        return 2
    data = build(str(root))
    out = Path(args.output) if args.output else Path(__file__).resolve().parent / "tools.json"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    m = data["meta"]
    print(f"wrote {out}")
    print(f"  tools: {m['tool_count']} by group: {m['group_counts']}")
    print(f"  not exposed: {m['not_exposed_count']} ({m['excluded_by_rule']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
