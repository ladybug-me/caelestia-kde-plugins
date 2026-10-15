"""Registry tests for the settings layer (the settings design notes (see module docstrings) §1, §9-§10).

Three independent guards:

- a GOLDEN table: the 18 core tools must keep their exact names,
  paths, kinds, defaults, ranges, enums, global_only flags, steps and nouns
  (the natural-language surface is frozen), and the generated registry is
  pinned by counts DERIVED from tools.json meta (tool_count, per-group tallies);
- a STRUCTURAL guard: every tool — the full registry — has a unique name and path, a
  known kind, a coherent validation payload, a feature-area group, and at
  least two citations whose cited files exist in this checkout;
- a CITATION drift guard: every tool's cited C++ declaration line still
  contains the property name (a sample of five NEW tools is line-pinned
  needle-style, and the whole table is property-name-checked), plus the
  regeneration guard: rebuilding tools.json from the checkout must be
  byte-identical to the committed artifact.

If the shell/ tree is absent (module copied into a packaged install), the
citation and regeneration guards SKIP loudly instead of silently passing.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Optional

from assistant.adapters.caelestia.registry import (
    GROUPS,
    NOT_EXPOSED,
    TOOL_COUNT,
    TOOL_SPECS,
    CORE_TOOL_NAMES,
    describe,
    list_tools_lines,
    registry_paths,
    tools_by_group,
    tool_by_name,
    tool_by_path,
)

# The golden table (the settings design notes (see module docstrings) §1, rows 1-18): (name, path, kind, default,
# minimum, maximum, enum, global_only, step, nouns)
GOLDEN_CORE: tuple = (
    ("setBarScale", "bar.scale", "float", 1.0, 0.6, 1.6, None, False, 0.1,
     ("bar|taskbar|panel",)),
    ("setBarPosition", "bar.position", "enum", "bottom", None, None,
     ("top", "bottom", "left", "right"), False, 0, ("bar|taskbar|panel",)),
    ("setDockIconSize", "bar.dock.iconSize", "int", 32, 16, 96, None, False, 4,
     ("dock icons|taskbar dock|dock",)),
    ("setBarPersistent", "bar.persistent", "bool", True, None, None, None, False, 0,
     ("bar|taskbar", "persistent|always visible|auto.?hide")),
    ("setLivePreviews", "bar.livePreviews", "bool", True, None, None, None, False, 0,
     ("live previews|window previews|thumbnails|previews",)),
    ("setBlurEnabled", "appearance.blur", "bool", True, None, None, None, True, 0,
     ("background blur|frosted glass|blur",)),
    ("setTransparencyBase", "appearance.transparency.base", "float", 0.85, 0.0, 1.0,
     None, True, 0.05, ("base opacity|transparency|opacity",)),
    ("setRoundingScale", "appearance.rounding.scale", "float", 1, 0.5, 2.0, None,
     False, 0.1, ("corner rounding|corner radius|rounding|corners",)),
    ("setSpacingScale", "appearance.spacing.scale", "float", 1, 0.5, 2.0, None,
     False, 0.1, ("gaps|spacing|gap",)),
    ("setPaddingScale", "appearance.padding.scale", "float", 1, 0.5, 2.0, None,
     False, 0.1, ("padding|insets",)),
    ("setFontScale", "appearance.font.scale", "float", 1, 0.5, 2.0, None, False, 0.1,
     ("font size|text size|fonts|font|text",)),
    ("setAnimationSpeed", "appearance.anim.durations.scale", "float", 1, 0.25, 4.0,
     None, True, 0.25,
     ("animation speed|animations?|transitions|motion|effects speed",)),
    ("setBorderThickness", "border.thickness", "int", 10, 0, 50, None, False, 2,
     ("shell border|border|outline",)),
    ("setLauncherMaxShown", "launcher.maxShown", "int", 7, 1, 20, None, False, 1,
     ("launcher results|search results|launcher",)),
    ("setPitchBlack", "appearance.pitchBlack", "bool", False, None, None, None, True, 0,
     ("pitch black|bezel mode|bezel",)),
    ("setNotifsMaxPopups", "notifs.maxPopups", "int", 8, 0, 30, None, True, 1,
     ("notification popups|popup notifications",)),
    ("setNotifsMaxNotifs", "notifs.maxNotifs", "int", 50, 20, 2000, None, True, 50,
     ("stored notifications|max stored notifications",)),
    ("setDockBadges", "bar.dock.showBadges", "bool", True, None, None, None, False, 0,
     ("app badges|badges|badge",)),
)

# Generated-registry pins: counts are DERIVED from the committed tools.json
# (single source of truth — the loader must agree with the file, and neither
# is re-pinned by a magic number here). Regenerating tools.json therefore
# cannot desync this test; the loader-vs-file agreement it asserts is the
# real invariant.
import json as _json

_TOOLS_JSON = Path(__file__).resolve().parents[2] / "assistant" / "adapters" / "caelestia" / "tools.json"
_META_COUNTS = _json.loads(_TOOLS_JSON.read_text(encoding="utf-8")).get(
    "meta", {})
EXPANSION_COUNT = _META_COUNTS.get("tool_count")
GROUP_COUNTS = dict(_META_COUNTS.get("group_counts") or {})
NOT_EXPOSED_COUNT = _META_COUNTS.get("not_exposed_count")

# Five NEW tools whose C++ citation lines are needle-pinned (drift guard).
NEW_TOOL_PINS = (
    ("setWorkspacesDisplayType", "bar.workspaces.displayType",
     "shell/plugin/src/Caelestia/Config/barconfig.hpp"),
    ("setWeatherUnits", "services.weatherUnits",
     "shell/plugin/src/Caelestia/Config/serviceconfig.hpp"),
    ("setWallpaperFillMode", "background.wallpaperFillMode",
     "shell/plugin/src/Caelestia/Config/backgroundconfig.hpp"),
    ("setEasingType", "overview.easingType",
     "shell/plugin/src/Caelestia/Config/overviewconfig.hpp"),
    ("setFullscreenExpireTimeout", "notifs.fullscreenExpireTimeout",
     "shell/plugin/src/Caelestia/Config/notifsconfig.hpp"),
)


def _find_repo_root(start: Path) -> Optional[Path]:
    for cand in [start, *start.parents]:
        if (cand / "shell" / "plugin" / "src" / "Caelestia" / "Config").is_dir():
            return cand
    return None


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)


class GoldenTableTests(unittest.TestCase):
    """The 18-tool core surface is frozen; the expansion is count-pinned."""

    def test_core_tools_lead_and_match_golden(self) -> None:
        self.assertEqual(len(TOOL_SPECS), TOOL_COUNT)
        self.assertGreaterEqual(TOOL_COUNT, len(GOLDEN_CORE))
        for idx, expected in enumerate(GOLDEN_CORE):
            spec = TOOL_SPECS[idx]
            name, path, kind, default, minimum, maximum, enum, gonly, step, nouns = expected
            self.assertEqual(spec.name, name, f"row {idx + 1} name")
            self.assertEqual(spec.path, path, f"row {idx + 1} path")
            self.assertEqual(spec.kind, kind, f"row {idx + 1} kind")
            self.assertEqual(spec.default, default, f"row {idx + 1} default")
            self.assertEqual(spec.minimum, minimum, f"row {idx + 1} minimum")
            self.assertEqual(spec.maximum, maximum, f"row {idx + 1} maximum")
            if enum is None:
                self.assertIsNone(spec.enum, f"row {idx + 1} enum")
            else:
                self.assertEqual(spec.enum, tuple(enum), f"row {idx + 1} enum")
            self.assertEqual(spec.global_only, gonly, f"row {idx + 1} global_only")
            self.assertEqual(spec.step, step, f"row {idx + 1} step")
            self.assertEqual(spec.nouns, nouns, f"row {idx + 1} nouns")

    def test_expansion_counts(self) -> None:
        self.assertEqual(TOOL_COUNT, EXPANSION_COUNT)
        counts = {g: len(specs) for g, specs in tools_by_group().items()}
        self.assertEqual(counts, GROUP_COUNTS)
        self.assertEqual(len(NOT_EXPOSED), NOT_EXPOSED_COUNT)

    def test_core_names_are_the_only_noun_tools(self) -> None:
        noun_tools = [s.name for s in TOOL_SPECS if s.nouns]
        self.assertEqual(sorted(noun_tools), sorted(CORE_TOOL_NAMES))

    def test_lookup_helpers_and_path_order(self) -> None:
        self.assertEqual(len(registry_paths()), TOOL_COUNT)
        self.assertEqual(len(set(registry_paths())), TOOL_COUNT)
        for spec in TOOL_SPECS:
            self.assertIs(tool_by_name(spec.name), spec)
            self.assertIs(tool_by_path(spec.path), spec)
        self.assertIsNone(tool_by_name("setNotARealTool"))
        self.assertIsNone(tool_by_path("not.a.real.path"))

    def test_list_tools_lines_renders(self) -> None:
        lines = list_tools_lines()
        joined = "\n".join(lines)
        self.assertIn(f"{TOOL_COUNT} tools", joined)
        for slug in GROUPS:
            self.assertIn(slug, joined)
        self.assertIn("setBarScale", joined)
        # filtered view: one group only
        eff = "\n".join(list_tools_lines("effects"))
        self.assertIn("setBlurEnabled", eff)
        self.assertNotIn("bar (", eff)
        # unknown group -> one-line error
        err = list_tools_lines("nope")
        self.assertEqual(len(err), 1)
        self.assertIn("unknown group", err[0])

    def test_describe_renders_validation_for_each_kind(self) -> None:
        for spec in TOOL_SPECS:
            text = describe(spec)
            self.assertIn(spec.name, text)
            self.assertIn(spec.path, text)
            if spec.kind == "enum":
                self.assertIn("|".join(str(v) for v in (spec.enum or ())), text)
            elif spec.kind == "bool":
                self.assertIn("on/off", text)
            elif spec.kind == "string":
                self.assertIn("string", text)
            else:
                self.assertIn(f"{spec.minimum}-{spec.maximum}", text)


class StructuralGuardTests(unittest.TestCase):
    """Every tool row is coherent, unique, and cited."""

    def test_unique_names_and_paths(self) -> None:
        names = [s.name for s in TOOL_SPECS]
        paths = [s.path for s in TOOL_SPECS]
        self.assertEqual(len(set(names)), len(names))
        self.assertEqual(len(set(paths)), len(paths))

    def test_kinds_and_validation_payloads(self) -> None:
        for spec in TOOL_SPECS:
            self.assertIn(spec.kind, ("bool", "int", "float", "enum", "string"),
                          f"{spec.name} kind {spec.kind!r}")
            if spec.kind == "enum":
                self.assertTrue(spec.enum, f"{spec.name} empty enum")
                types = {type(v) for v in spec.enum}
                self.assertEqual(len(types), 1,
                                 f"{spec.name} mixed enum types {types}")
                self.assertTrue(types <= {str, int})
            elif spec.kind in ("int", "float"):
                self.assertIsNotNone(spec.minimum, f"{spec.name} no minimum")
                self.assertIsNotNone(spec.maximum, f"{spec.name} no maximum")
                self.assertLessEqual(spec.minimum, spec.maximum)
            else:  # bool / string
                self.assertIsNone(spec.minimum, f"{spec.name} unexpected minimum")
                self.assertIsNone(spec.maximum, f"{spec.name} unexpected maximum")
            if spec.kind == "string":
                self.assertIsNotNone(spec.string_max_len)
            self.assertIn(spec.group, GROUPS, f"{spec.name} group {spec.group!r}")

    def test_every_tool_has_two_citations(self) -> None:
        for spec in TOOL_SPECS:
            self.assertGreaterEqual(
                len(spec.citations), 2,
                f"{spec.name} has {len(spec.citations)} citations")
            files = [c[0].split(":")[0] for c in spec.citations]
            self.assertTrue(
                any(f.startswith("shell/plugin/src/Caelestia/Config/") for f in files),
                f"{spec.name} lacks a C++ declaration citation")
            self.assertTrue(
                any(f.startswith("shell/") and "/Config/" not in f for f in files),
                f"{spec.name} lacks a UI/consumer citation")

    def test_not_exposed_has_reasons_and_no_overlap(self) -> None:
        tool_paths = set(registry_paths())
        for path, reason in NOT_EXPOSED:
            self.assertTrue(reason, f"{path} empty reason")
            self.assertNotIn(path, tool_paths,
                             f"{path} is both a tool and not_exposed")

    def test_default_types_match_kinds(self) -> None:
        for spec in TOOL_SPECS:
            d = spec.default
            if spec.kind == "bool":
                self.assertIsInstance(d, bool)
            elif spec.kind == "int":
                self.assertIsInstance(d, int)
                self.assertNotIsInstance(d, bool)
            elif spec.kind == "float":
                self.assertIsInstance(d, (int, float))
                self.assertNotIsInstance(d, bool)
            elif spec.kind == "enum":
                if not isinstance(d, int):
                    self.assertIsInstance(d, str)
            elif spec.kind == "string":
                self.assertIsInstance(d, str)


@unittest.skipIf(REPO_ROOT is None, "shell/ tree absent — citation guards skip")
class CitationDriftGuardTests(unittest.TestCase):
    """Cited lines still hold the cited declarations."""

    def _line(self, rel: str, line: int) -> str:
        path = REPO_ROOT / rel  # type: ignore[operator]
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[line - 1]

    def test_every_cited_cpp_line_contains_the_property(self) -> None:
        for spec in TOOL_SPECS:
            cpp = next(c for c in spec.citations
                       if c[0].startswith("shell/plugin/src/Caelestia/Config/"))
            rel, _, lineno = cpp[0].rpartition(":")
            line = self._line(rel, int(lineno))
            prop = spec.path.rsplit(".", 1)[-1]
            self.assertIn(
                prop, line,
                f"{spec.name}: {rel}:{lineno} no longer declares {prop}: {line!r}")

    def test_new_tool_pins(self) -> None:
        for name, path, header in NEW_TOOL_PINS:
            spec = tool_by_name(name)
            self.assertIsNotNone(spec, f"pin tool {name} vanished from the registry")
            assert spec is not None
            self.assertEqual(spec.path, path)
            cpp = next(c for c in spec.citations
                       if c[0].startswith("shell/plugin/src/Caelestia/Config/"))
            rel, _, lineno = cpp[0].rpartition(":")
            self.assertEqual(rel, header, f"{name}: C++ citation moved file")
            line = self._line(rel, int(lineno))
            self.assertIn(path.rsplit(".", 1)[-1], line)

    def test_regenerated_registry_is_byte_identical(self) -> None:
        from assistant.capabilities.settings import build_registry
        data = build_registry.build(str(REPO_ROOT))
        rendered = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        committed = (REPO_ROOT / "assistant" / "adapters" / "caelestia" / "tools.json").read_text(
            encoding="utf-8")
        self.assertEqual(
            rendered, committed,
            "tools.json drifted from the checkout: regenerate with "
            "python3 -m assistant.capabilities.settings.build_registry and re-pin the counts")

    def test_range_fixup_citations_resolve(self) -> None:
        from assistant.capabilities.settings import curations
        for path, fix in curations.RANGE_FIXUPS.items():
            cite = fix.get("cite")
            if not cite:
                continue
            rel, _, lineno = cite[0].rpartition(":")
            line = self._line(rel, int(lineno))
            self.assertTrue(
                any(tok in line for tok in ("/ 1000", "* 1000", "/ 60", "* 60", "* 100", "/ 100")),
                f"{path}: fixup citation {cite[0]} shows no unit transform: {line!r}")


if __name__ == "__main__":
    unittest.main()
