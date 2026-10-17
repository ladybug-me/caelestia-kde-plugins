"""Safety classification and grouping rules for the settings tool registry.

The generated-registry pipeline's curated data. Pure data + pure functions: no I/O, no
environment, no execution, no network. Every rule carries the reason that
lands in tools.json's ``not_exposed`` table, so the archive's manifest can
cite exactly why each leaf is (or is not) a tool.

THE MECHANICAL EXPOSURE RULE (build_registry.py implements it):
a leaf becomes a tool IFF ALL hold:
  (a) it is a serialized scalar leaf of the ConfigRoot tree (kind=scalar;
      type bool/int/qreal/QString/enum) — lists, maps, unions, subobjects
      are never tools;
  (b) a shipped Nexus control touches it (ui_ranges_data.UI_RANGES) — the
      conservative line: no UI row, no tool. UI-less bools with
      live QML consumers stay unexposed rather than guessed
      (recorded in not_exposed with this reason); every core tool also has a
      UI row, so nothing regresses;
  (c) it fails none of the EXCLUSION rules below.

VALIDATION DERIVATION (exactly as the shipped QML declares it):
- Stepper/DoubleStepper/Slider rows: adopt min/max(/step) verbatim
  (sliders are normalized 0..1 by StyledSlider.qml:152,173);
- SelectRow rows: adopt the verbatim enum list;
- ToggleRow rows: bool;
- FontCard rows: kind "string" (font family), max 64 chars, non-empty;
- any other control type (list editors, chips, free-text fields, ...):
  not a scalar tool — excluded with the control type in the reason.

Upstream's C++ declares type + default only (type-only validation hook,
Settings/macros.hpp SETTINGS_PROPERTY_IMPL) — the ranges here are the
shipped UI's own, which is what issue #120 asks tools to enforce
("If a value falls outside the supported range, it simply isn't applied").
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .ui_ranges_data import UI_RANGES

# ---------------------------------------------------------------------------
# Group slugs, in display order. A tool's group is derived from its dotted
# path by GROUP_RULES (first match wins).
# ---------------------------------------------------------------------------

GROUPS: Tuple[str, ...] = (
    "bar",
    "dock",
    "appearance",
    "effects",
    "animations",
    "notifications",
    "launcher",
    "lockscreen",
    "wallpaper-scheme",
    "overview",
    "osd",
    "dashboard",
    "sidebar",
    "nexus",
    "border",
    "general",
    "services",
    "utilities",
    "audio",
    "ai",
)

_EFFECTS_PATHS = frozenset((
    "appearance.blur",
    "appearance.blurMask",
    "appearance.pitchBlack",
    "appearance.islands",
    "appearance.ambientColor",
    "appearance.ambientOpacity",
    "appearance.transparency.enabled",
    "appearance.transparency.base",
    "appearance.transparency.layers",
))

GROUP_RULES: Tuple[Tuple[str, str], ...] = (
    ("dock", "bar.dock."),
    ("animations", "appearance.anim."),
    ("effects", "\x00effects-exact\x00"),  # marker resolved in group_for()
    ("appearance", "appearance."),
    ("notifications", "notifs."),
    ("launcher", "launcher."),
    ("lockscreen", "lock."),
    ("wallpaper-scheme", "background."),
    ("overview", "overview."),
    ("osd", "osd."),
    ("dashboard", "dashboard."),
    ("sidebar", "sidebar."),
    ("nexus", "nexus."),
    ("border", "border."),
    ("general", "general."),
    ("services", "services."),
    ("utilities", "utilities."),
    ("audio", "audio."),
    ("ai", "ai."),
    ("bar", "bar."),
)


def group_for(path: str) -> str:
    """Feature-area slug for a dotted config path (first rule wins)."""
    if path in _EFFECTS_PATHS:
        return "effects"
    for slug, prefix in GROUP_RULES:
        if path.startswith(prefix):
            return slug
    return "general"


GROUP_TITLES: Dict[str, str] = {
    "bar": "Bar / taskbar",
    "dock": "Dock",
    "appearance": "Appearance (scales, fonts)",
    "effects": "Blur, transparency & effects",
    "animations": "Animations",
    "notifications": "Notifications",
    "launcher": "Launcher",
    "lockscreen": "Lock screen",
    "wallpaper-scheme": "Wallpaper & colour scheme",
    "overview": "Overview",
    "osd": "On-screen display",
    "dashboard": "Dashboard",
    "sidebar": "Sidebar",
    "nexus": "Nexus (settings app)",
    "border": "Shell border",
    "general": "General",
    "services": "Services",
    "utilities": "Utilities",
    "audio": "Audio",
    "ai": "AI assistant",
}

# ---------------------------------------------------------------------------
# Exclusion rules. Each: (name, predicate over the walker's leaf row,
# reason). Applied in order; the FIRST matching rule's reason is recorded.
# A leaf may match none (then it is a tool candidate) — the exposure rule
# (UI row + scalar) is applied by build_registry.py separately.
# ---------------------------------------------------------------------------

LeafRow = Dict[str, str]


def _prefix(*prefixes: str) -> Callable[[LeafRow], bool]:
    def check(row: LeafRow) -> bool:
        return any(row["path"].startswith(p) for p in prefixes)
    return check


def _exact(*paths: str) -> Callable[[LeafRow], bool]:
    paths_ = frozenset(paths)

    def check(row: LeafRow) -> bool:
        return row["path"] in paths_
    return check


EXCLUSION_RULES: Tuple[Tuple[str, Callable[[LeafRow], bool], str], ...] = (
    (
        "non-scalar",
        lambda r: r["kind"] != "scalar",
        "non-scalar structured key (list/map/union/subobject); the settings "
        "tool surface is scalar leaves only",
    ),
    (
        "master-switch",
        _exact("enabled"),
        "ConfigRoot master switch — a bad write disables the entire shell",
    ),
    (
        "filesystem-paths",
        _prefix("paths."),
        "filesystem path redirection; the wallpaper/CLI surface owns this",
    ),
    (
        "session-commands",
        _prefix("session.commands."),
        "command lists the session menu executes — never a settings tool",
    ),
    (
        "session-icons",
        _prefix("session.icons."),
        "icon-name strings with no shipped range/format control",
    ),
    (
        "general-command-lists",
        _exact(
            "general.terminal", "general.audio", "general.playback",
            "general.explorer",
        ),
        "command lists the shell executes to open apps — never a settings tool",
    ),
    (
        "variant-unions",
        _exact("general.idleAction", "general.returnAction"),
        "QVariant union key; the type checker allows two metatypes and the "
        "value shape is not a scalar",
    ),
    (
        "ai-credentials",
        _exact(
            "ai.anthropicApiKey", "ai.openaiApiKey", "ai.geminiApiKey",
            "ai.openrouterApiKey", "ai.opencodeApiKey",
        ),
        "plaintext credential — never exposed as a tool",
    ),
    (
        "ai-endpoints",
        _exact(
            "ai.ollamaUrl", "ai.anthropicUrl", "ai.openaiUrl", "ai.geminiUrl",
            "ai.openrouterUrl", "ai.opencodeUrl", "ai.opencodeGoUrl",
        ),
        "network endpoint the shell sends data to — outside the settings "
        "tool surface",
    ),
    (
        "ai-executables",
        _exact("ai.claudeCodeBin", "ai.loginTerminal"),
        "binary/path the shell executes — never a settings tool",
    ),
    (
        "ai-internal-state",
        _exact("ai.ollamaHistoryJson", "ai.claudeAccountsJson"),
        "internally-managed state blob",
    ),
    (
        "ai-provider-selfconfig",
        _prefix("ai."),
        "AI-provider self-configuration (models/providers/accounts): the "
        "assistant must not reconfigure its own plumbing; the Nexus "
        "AiSettingsPage owns these",
    ),
    (
        "kwinrc-companion-writes",
        _exact(
            "general.magicLampEnabled", "general.krohnkiteEnabled",
            "general.krohnkiteLastLayout",
            "tabSwitch.enabled", "tabSwitch.orderTree",
            "tabSwitch.highlightWindows", "tabSwitch.minMode",
            "tabSwitch.multiScreenMode",
        ),
        "the shipped Nexus control writes companion kwinrc keys (or "
        "installs a kwin script) alongside this key; a shell.json-only "
        "write would desync the two stores",
    ),
    (
        "deprecated",
        lambda r: bool(r.get("notes") and "deprecat" in r["notes"].lower()),
        "deprecated migration key kept only for ConfigMigrations",
    ),
)

# Survey-transcription fixups: keys whose UI-survey enum annotation
# was imprecise, re-verified at source. Path -> (values, citation).
# Values may be strs (serialized enum keys) or ints (numeric-coded selects).
ENUM_FIXUPS: Dict[str, Tuple[List[Any], Tuple[str, str]]] = {
    # ServicesPage.qml:47 — the value list is a FIXED literal (only the
    # labels vary); the survey's "(dynamic gpuValues list)" note was wrong.
    "services.gpuType": (
        ["", "NVIDIA", "GENERIC", "None"],
        ("shell/modules/nexus/pages/ServicesPage.qml:47",
         'gpuValues: ["", "NVIDIA", "GENERIC", "None"] (fixed literal)'),
    ),
    # WallpaperSettingsPage.qml:26/56-58 — the SelectRow's menu items are
    # LABELS (Crop|Fit|Stretch); the values it writes are the scalingValues
    # ints [Image.PreserveAspectCrop, PreserveAspectFit, Image.Stretch] =
    # [2, 1, 0]. The survey recorded the labels; the C++ leaf is int
    # (default 2), so the enum must be the ints the UI actually stores.
    "background.wallpaperFillMode": (
        [0, 1, 2],
        ("shell/modules/nexus/pages/wallandstyle/WallpaperSettingsPage.qml:26",
         "scalingValues: [Image.PreserveAspectCrop, Image.PreserveAspectFit, "
         "Image.Stretch] = [2, 1, 0]; the SelectRow writes these ints"),
    ),
}

# Numeric leaves whose ONLY range source is not a stepper/slider row but a
# documented dynamic bound; the 18-tool core already handles bar.dock.iconSize
# with a conservative fixed cap (96) — preserved verbatim below.
CORE_OVERRIDES: Dict[str, Dict[str, object]] = {
    "bar.dock.iconSize": {
        "minimum": 16, "maximum": 96, "step": 4,
        "reason": (
            "a conservative fixed cap; upstream's bound is dynamic "
            "(Tokens.sizes.bar.innerWidth, tokens.hpp:115)"
        ),
    },
}

# Transformed steppers, re-verified at source (found by the
# 2-a-2 default-outside-range audit): the shipped StepperRow DISPLAYS a
# transformed value — seconds while the stored leaf is milliseconds,
# percent while the stored leaf is a fraction, minutes while the stored
# leaf is seconds — so the row's from/to/stepSize describe the DISPLAYED
# unit, not the stored one. tools.json must record the range in the STORED
# unit (the only unit a shell.json write can carry), so each entry carries
# the converted minimum/maximum/step plus a citation of the transforming
# row (the conversion arithmetic is on those very lines).
RANGE_FIXUPS: Dict[str, Dict[str, object]] = {
    # NotificationPreferencesPage.qml:117-124 — UI 1-60 seconds.
    "notifs.defaultExpireTimeout": {
        "minimum": 1000, "maximum": 60000, "step": 1000,
        "cite": ("shell/modules/nexus/pages/services/NotificationPreferencesPage.qml:120",
                 "stepper displays seconds (value: x/1000; onMoved: "
                 "Math.round(v*1000)); stored unit is ms — range converted x1000"),
    },
    # NotificationPreferencesPage.qml:180-187 — UI 1-30 seconds.
    "notifs.fullscreenExpireTimeout": {
        "minimum": 1000, "maximum": 30000, "step": 1000,
        "cite": ("shell/modules/nexus/pages/services/NotificationPreferencesPage.qml:183",
                 "stepper displays seconds (value: x/1000; onMoved: "
                 "Math.round(value*1000)); stored unit is ms — range converted x1000"),
    },
    # utilities/OsdPage.qml:83-93 — UI 1-10 seconds.
    "osd.hideDelay": {
        "minimum": 1000, "maximum": 10000, "step": 1000,
        "cite": ("shell/modules/nexus/pages/utilities/OsdPage.qml:88",
                 "stepper displays seconds (value: x/1000; onMoved: "
                 "Math.round(value*1000)); stored unit is ms — range converted x1000"),
    },
    # ServicesPage.qml:110-118 — UI 5-120 seconds.
    "nexus.networkRescanInterval": {
        "minimum": 5000, "maximum": 120000, "step": 5000,
        "cite": ("shell/modules/nexus/pages/ServicesPage.qml:114",
                 "stepper displays seconds (value: x/1000; onMoved: "
                 "Math.round(v*1000)); stored unit is ms — range converted x1000"),
    },
    # ServicesPage.qml:100-108 — UI 0.5-10 seconds.
    "dashboard.resourceUpdateInterval": {
        "minimum": 500, "maximum": 10000, "step": 500,
        "cite": ("shell/modules/nexus/pages/ServicesPage.qml:103",
                 "stepper displays seconds (value: x/1000; onMoved: "
                 "Math.round(v*1000)); stored unit is ms — range converted x1000"),
    },
    # services/ArpcPage.qml:83-94 — UI 0-60 minutes (0 = never hide).
    "services.arpcIdleTimeout": {
        "minimum": 0, "maximum": 3600, "step": 60,
        "cite": ("shell/modules/nexus/pages/services/ArpcPage.qml:89",
                 "stepper displays minutes (value: Math.round(x/60); onMoved: "
                 "Math.round(v*60)); stored unit is seconds — range converted x60"),
    },
    # ServicesPage.qml:151-160 — UI 1-50 percent.
    "services.audioIncrement": {
        "minimum": 0.01, "maximum": 0.5, "step": 0.01,
        "cite": ("shell/modules/nexus/pages/ServicesPage.qml:155",
                 "stepper displays percent (value: Math.round(x*100); onMoved: "
                 "v/100); stored unit is a fraction — range converted /100"),
    },
    # ServicesPage.qml:162-170 — UI 1-50 percent.
    "services.brightnessIncrement": {
        "minimum": 0.01, "maximum": 0.5, "step": 0.01,
        "cite": ("shell/modules/nexus/pages/ServicesPage.qml:165",
                 "stepper displays percent (value: Math.round(x*100); onMoved: "
                 "v/100); stored unit is a fraction — range converted /100"),
    },
    # ServicesPage.qml:172-181 — UI 50-200 percent.
    "services.maxVolume": {
        "minimum": 0.5, "maximum": 2.0, "step": 0.05,
        "cite": ("shell/modules/nexus/pages/ServicesPage.qml:176",
                 "stepper displays percent (value: Math.round(x*100); onMoved: "
                 "v/100); stored unit is a fraction — range converted /100"),
    },
}

# The 18 core tools, preserved byte-for-byte (name/path/kind/default/min/max/
# enum/global_only/step/nouns). Citations are re-derived from the walker +
# UI rows, plus these hand-carried citation notes for consumer evidence.
CORE_TOOLS: Dict[str, Dict[str, object]] = {
    "bar.scale": {"name": "setBarScale", "minimum": 0.6, "maximum": 1.6, "step": 0.1,
                  "nouns": ["bar|taskbar|panel"],
                  "extra_citations": [
                      ("shell/modules/bar/BarWrapper.qml:24", "contentWidth = innerWidth * barScale; floor 0.6"),
                  ]},
    "bar.position": {"name": "setBarPosition", "enum": ["top", "bottom", "left", "right"],
                     "nouns": ["bar|taskbar|panel"],
                     "extra_citations": [
                         ("shell/modules/bar/BarWrapper.qml:23", "position-driven anchoring"),
                     ]},
    "bar.dock.iconSize": {"name": "setDockIconSize", "minimum": 16, "maximum": 96, "step": 4,
                          "nouns": ["dock icons|taskbar dock|dock"],
                          "extra_citations": [
                              ("shell/modules/bar/components/Dock.qml:38", "QML floor 16"),
                          ]},
    "bar.persistent": {"name": "setBarPersistent", "nouns": ["bar|taskbar", "persistent|always visible|auto.?hide"],
                       "extra_citations": [
                           ("shell/modules/bar/BarWrapper.qml:27", "persistent reader"),
                       ]},
    "bar.livePreviews": {"name": "setLivePreviews", "nouns": ["live previews|window previews|thumbnails|previews"],
                         "extra_citations": [
                             # Path re-verified at 70ee7da: the file lives under
                             # shell/components/images/ (an earlier the settings design notes (see module docstrings) path
                             # shell/modules/bar/components/ no longer exists).
                             ("shell/components/images/WindowPreview.qml:42", "live reader"),
                         ]},
    "appearance.blur": {"name": "setBlurEnabled", "nouns": ["background blur|frosted glass|blur"],
                        "extra_citations": [
                            ("shell/modules/drawers/blur/BlurOffsets.qml:17", "isActive = transparency.enabled && blur"),
                        ]},
    "appearance.transparency.base": {"name": "setTransparencyBase", "minimum": 0.0, "maximum": 1.0, "step": 0.05,
                                     "nouns": ["base opacity|transparency|opacity"],
                                     "extra_citations": [
                                         # Path re-verified at 70ee7da: ContentWindow.qml lives under
                                         # shell/modules/drawers/ (an earlier the settings design notes (see module docstrings) path
                                         # shell/modules/components/containers/ no longer exists).
                                         ("shell/modules/drawers/ContentWindow.qml:383", "surface opacity reader"),
                                     ]},
    "appearance.rounding.scale": {"name": "setRoundingScale", "minimum": 0.5, "maximum": 2.0, "step": 0.1,
                                  "nouns": ["corner rounding|corner radius|rounding|corners"]},
    "appearance.spacing.scale": {"name": "setSpacingScale", "minimum": 0.5, "maximum": 2.0, "step": 0.1,
                                 "nouns": ["gaps|spacing|gap"]},
    "appearance.padding.scale": {"name": "setPaddingScale", "minimum": 0.5, "maximum": 2.0, "step": 0.1,
                                 "nouns": ["padding|insets"]},
    "appearance.font.scale": {"name": "setFontScale", "minimum": 0.5, "maximum": 2.0, "step": 0.1,
                              "nouns": ["font size|text size|fonts|font|text"]},
    "appearance.anim.durations.scale": {"name": "setAnimationSpeed", "minimum": 0.25, "maximum": 4.0, "step": 0.25,
                                        "nouns": ["animation speed|animations?|transitions|motion|effects speed"],
                                        "extra_citations": [
                                            ("shell/plugin/src/Caelestia/Config/appearanceconfig.cpp:207", "durations multiplied by scale"),
                                        ]},
    "border.thickness": {"name": "setBorderThickness", "minimum": 0, "maximum": 50, "step": 2,
                         "nouns": ["shell border|border|outline"]},
    "launcher.maxShown": {"name": "setLauncherMaxShown", "minimum": 1, "maximum": 20, "step": 1,
                          "nouns": ["launcher results|search results|launcher"]},
    "appearance.pitchBlack": {"name": "setPitchBlack", "nouns": ["pitch black|bezel mode|bezel"],
                              "extra_citations": [
                                  # Path re-verified at 70ee7da (see setTransparencyBase above).
                                  ("shell/modules/drawers/ContentWindow.qml:394", "#000000 surface when pitchBlack"),
                              ]},
    "notifs.maxPopups": {"name": "setNotifsMaxPopups", "minimum": 0, "maximum": 30, "step": 1,
                         "nouns": ["notification popups|popup notifications"]},
    "notifs.maxNotifs": {"name": "setNotifsMaxNotifs", "minimum": 20, "maximum": 2000, "step": 50,
                         "nouns": ["stored notifications|max stored notifications"]},
    "bar.dock.showBadges": {"name": "setDockBadges", "nouns": ["app badges|badges|badge"],
                            "extra_citations": [
                                ("shell/modules/bar/components/Dock.qml:630", "badge over LauncherEntry.forApp, ?? true"),
                            ]},
}

# UI control types that are scalar-tool capable. Anything else on a UI row
# excludes the key (reason carries the control type).
_SCALAR_CONTROLS = re.compile(
    r"^(ToggleRow|StepperRow|DoubleStepperRow|SliderRow|SelectRow|FontCard)"
)

# ToggleRow variants annotated in parentheses remain ToggleRows (the note
# describes enabled/visible gating, not the control shape).


def control_is_scalar(control: str) -> bool:
    """True when the shipped control is a scalar setter (toggle/stepper/
    slider/select/font card). Parenthesized annotations are ignored."""
    return bool(_SCALAR_CONTROLS.match(control))


def exclusion_for(row: LeafRow) -> Optional[Tuple[str, str]]:
    """First matching exclusion rule for a walker leaf row, if any."""
    for name, predicate, reason in EXCLUSION_RULES:
        try:
            if predicate(row):
                return name, reason
        except Exception:  # noqa: BLE001 - defensive: rules are pure data
            continue
    return None


def tool_name_for(path: str) -> str:
    """Derive the tool name from the dotted path (core names preserved via
    CORE_TOOLS). 'appearance.transparency.base' -> 'setTransparencyBase';
    the leading area segment is dropped when the remainder is distinctive."""
    if path in CORE_TOOLS:
        return str(CORE_TOOLS[path]["name"])
    parts = path.split(".")
    # Drop a pure area prefix that matches the group's own root segment.
    if len(parts) > 1:
        parts = parts[1:]
    camel = "".join(p[:1].upper() + p[1:] for p in parts if p)
    return "set" + (camel if camel else "Value")


def extra_citations_for(path: str) -> List[Tuple[str, str]]:
    core = CORE_TOOLS.get(path)
    if not core:
        return []
    return list(core.get("extra_citations", []))  # type: ignore[arg-type]

# ---------------------------------------------------------------------------
# Explainability rules. Each rule is pure data the Python CLI and
# the QML service BOTH render; every citation below was re-verified by eye
# in this checkout.
# Fields: path (the key the question is about), when (a predicate over a
# state dict {path: value} — expressed as a small DSL string evaluated by
# both layers, see build_registry/explain.py for the evaluator), answer
# (a {fmt} template over state values), citations.
# ---------------------------------------------------------------------------

EXPLAIN_RULES: List[Dict[str, object]] = [
    {
        "path": "appearance.blur",
        "when": "appearance.blur == true and appearance.transparency.enabled == true",
        "answer": "Blur is enabled because transparency is currently active "
                  "and appearance.blur is on — the blur regions gate on both "
                  "({cite}).",
        "cites": ["shell/modules/drawers/blur/BlurOffsets.qml:17"],
    },
    {
        "path": "appearance.blur",
        "when": "appearance.blur == true and appearance.transparency.enabled == false",
        "answer": "Blur is ON but has no effect, because transparency is "
                  "currently disabled — the blur regions gate on "
                  "transparency.enabled && blur together ({cite}).",
        "cites": ["shell/modules/drawers/blur/BlurOffsets.qml:17"],
    },
    {
        "path": "appearance.transparency.enabled",
        "when": "appearance.transparency.enabled == true and "
                "utilities.gameMode.disableShellTransparency == true and "
                "_gamemode == true",
        "answer": "Transparency is configured ON but is currently suppressed, "
                  "because Game Mode is active with 'disable shell "
                  "transparency' set — the colour service gates on that pair "
                  "({cite}).",
        "cites": ["shell/services/Colours.qml:418"],
    },
    {
        "path": "appearance.transparency.base",
        "when": "appearance.transparency.base > 0 and _light == true",
        "answer": "Surfaces look less transparent than the configured base "
                  "opacity {appearance.transparency.base}, because light mode "
                  "trims the base by 0.1 ({cite}).",
        "cites": ["shell/services/Colours.qml:419"],
    },
    {
        "path": "appearance.pitchBlack",
        "when": "appearance.pitchBlack == true",
        "answer": "Surfaces are fully opaque and pure black because Pitch "
                  "Black (bezel mode) is active — it forces opacity 1 "
                  "({cite1}) and #000000 surfaces ({cite2}).",
        "cites": ["shell/modules/drawers/ContentWindow.qml:383",
                  "shell/modules/drawers/ContentWindow.qml:394"],
    },
    {
        "path": "overview.enableOverviewBlur",
        "when": "overview.enableOverviewBlur == true and appearance.blur == false",
        "answer": "The overview blur setting is ON but has no effect, because "
                  "the global appearance blur is off — the overview effect "
                  "gates on both ({cite}).",
        "cites": ["shell/modules/drawers/ContentWindow.qml:374"],
    },
    {
        "path": "appearance.anim.durations.scale",
        "when": "appearance.anim.durations.scale != 1",
        "answer": "Animations feel {slower|faster} because the animation "
                  "duration scale is currently set to "
                  "{appearance.anim.durations.scale} — every duration token "
                  "is multiplied by it, so lower means faster ({cite}).",
        "cites": ["shell/plugin/src/Caelestia/Config/appearanceconfig.cpp:209"],
    },
    {
        "path": "appearance.ambientColor",
        "when": "appearance.ambientColor == true and _light == true",
        "answer": "The ambient glow is configured ON but is currently hidden, "
                  "because ambient glow only renders in dark mode — it gates "
                  "on ambientColor && !light ({cite}).",
        "cites": ["shell/components/effects/AmbientGlow.qml:34"],
    },
    {
        "path": "bar.persistent",
        "when": "bar.persistent == false",
        "answer": "The bar hides because bar persistence is currently off — "
                  "a non-persistent bar dodges or auto-hides instead of "
                  "staying put ({cite}).",
        "cites": ["shell/modules/bar/BarWrapper.qml:62"],
    },
    {
        "path": "bar.dodgeWindows",
        "when": "bar.dodgeWindows == true and bar.persistent == false",
        "answer": "Window dodging has no effect, because dodge mode requires "
                  "a persistent bar — the dodge gate is dodgeWindows && "
                  "persistent ({cite}).",
        "cites": ["shell/modules/bar/BarWrapper.qml:27"],
    },
    {
        "path": "bar.scale",
        "when": "bar.scale < 0.6",
        "answer": "The bar is not as small as asked: the effective bar scale "
                  "is floored at 0.6 ({cite}); the configured value is "
                  "{bar.scale}.",
        "cites": ["shell/modules/bar/BarWrapper.qml:24"],
    },
]

# ---------------------------------------------------------------------------
# Named presets — bundles of validated tool calls ONLY, never
# bespoke code paths. Every (tool, value) must pass that tool's validation
# (a test asserts it). Values are in the STORED unit (see RANGE_FIXUPS).
# ---------------------------------------------------------------------------

PRESETS: List[Dict[str, object]] = [
    {
        "name": "compact",
        "label": "Compact",
        "description": "Smaller dock icons and tighter spacing/padding/rounding "
                       "— the same keys the Nexus steppers expose.",
        "calls": [
            ["setDockIconSize", 24],
            ["setSpacingScale", 0.8],
            ["setPaddingScale", 0.8],
            ["setRoundingScale", 0.8],
        ],
    },
    {
        "name": "minimal",
        "label": "Minimal",
        "description": "Issue #120's own example direction: smaller bar and "
                       "dock, lower rounding, faster animations, badges and "
                       "previews off.",
        "calls": [
            ["setBarScale", 0.7],
            ["setDockIconSize", 20],
            ["setRoundingScale", 0.5],
            ["setAnimationSpeed", 0.5],
            ["setSpacingScale", 0.7],
            ["setDockBadges", False],
            ["setLivePreviews", False],
        ],
    },
    {
        "name": "gaming",
        "label": "Gaming",
        "description": "Snappier shell, no blur/transparency compositing, no "
                       "popup notifications. (Game Mode itself is a service "
                       "toggle, not a shell.json scalar — this bundle is the "
                       "config-side reduction only.)",
        "calls": [
            ["setBlurEnabled", False],
            ["setTransparencyEnabled", False],
            ["setAnimationSpeed", 0.25],
            ["setNotifsMaxPopups", 0],
        ],
    },
    {
        "name": "battery-saver",
        "label": "Battery saver",
        "description": "Reduce compositing and background work: blur, ambient "
                       "glow and wallpaper recolour off, animations fast.",
        "calls": [
            ["setBlurEnabled", False],
            ["setAnimationSpeed", 0.25],
            ["setAmbientColor", False],
            ["setWallpaperRecolor", False],
        ],
    },
    {
        "name": "macos-like",
        "label": "More like macOS",
        "description": "Always-visible bottom dock with larger icons, rounder "
                       "corners, roomier spacing and badges on — the keys the "
                       "shipped controls already expose.",
        "calls": [
            ["setBarPosition", "bottom"],
            ["setBarPersistent", True],
            ["setDockIconSize", 48],
            ["setRoundingScale", 1.3],
            ["setSpacingScale", 1.1],
            ["setPaddingScale", 1.1],
            ["setAnimationSpeed", 1.0],
            ["setDockBadges", True],
        ],
    },
]
