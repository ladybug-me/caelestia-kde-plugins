"""Static guards for the sidebar's command execution gate (T2).

shell/services/CommandGate.qml is the validated gate every state-changing
tool call the sidebar's model can request must pass through — the same
allow-list / preview / explicit-confirm shape SettingsTools.qml gives the
settings tools. The sidebar (shell/modules/sidebar/AiAssistant.qml) wires
`caelestia_command`, `open_app` and `set_timer` through it and renders an
approval card; nothing state-changing may run from raw model output.

No Qt toolchain is required (or assumed) — these tests statically pin the
shipped QML, exactly like the settings suite pins SettingsTools.qml:

1. GATE SHAPE: CommandGate.qml exists, is a singleton, exposes the pinned
   request/confirm/cancel API, and itself executes nothing.
2. ALLOW-LIST: exactly the verified read-only `caelestia` subcommands
   (upstream src/bin/caelestia @ dev, read 2026-09-26): `version`, `help`,
   and the bare read forms `scheme list` / `scheme get`. Every other
   subcommand needs the card — including `shell` (which always runs
   `caelestia-shell-ipc start`), `install`, `update`, `wallpaper`,
   `screenshot`, `record` and `scheme set`.
3. WIRING: the three gated tools' dispatch branches route through the gate;
   the read-only fast path is guarded by needsConfirm; `set_timer` is
   counted as async (its result now arrives at card-Apply time).
4. CARD: the chat delegate renders the command card and its Apply/Cancel
   resolve through resolveCommandCard; every chatHistory.append carries
   the command roles.
5. HONEST SYSPROMPT: the caelestia_command description lists exactly the
   verified subcommands (the pre-gate text advertised `toggle`, `search`,
   `clipboard`, `emoji`, `resizer`, none of which exist upstream).
6. BALANCE: braces/parens/brackets balance outside comments and strings.

If the QML files are absent (packaged installs without the shell tree),
all tests SKIP loudly.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import List

QML_PATH = (Path(__file__).resolve().parents[2] / "assistant" / "adapters" / "caelestia" /
            "shell" / "modules" / "sidebar" / "AiAssistant.qml")
GATE_PATH = (Path(__file__).resolve().parents[2] / "assistant" / "adapters" / "caelestia" /
            "shell" / "services" / "CommandGate.qml")

# The verified upstream `caelestia` subcommand list (src/bin/caelestia @ dev,
# fetched and read in full 2026-09-26): the usage block declares exactly these.
VERIFIED_SUBCOMMANDS = "shell, install, update, wallpaper, scheme (list/get/set), screenshot, record, version, help"
STALE_SUBCOMMANDS = ("toggle", "search", "clipboard", "emoji", "resizer")

GATED_TOOLS = ("open_app", "set_timer", "caelestia_command")


def _strip_strings_and_comments(text: str) -> str:
    """Remove // and /* */ comments plus ' " ` string contents so bracket
    counting only sees code (single quotes included: the sidebar's JS uses
    all three quote styles). Regex literals are detected by the standard
    last-significant-char heuristic (a '/' directly after one of
    ``( , = : [ ! & | ? { ;`` opens a regex, never a division), because the
    file contains quote-carrying regexes like ``/'/g`` that would otherwise
    poison the string tracker."""
    out: List[str] = []
    i, n = 0, len(text)
    mode = "code"  # code | line | block | str | tstr | chr | regex
    prev_sig = ""   # last significant code character emitted
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if mode == "code":
            if c == "/" and nxt == "/":
                mode = "line"
                i += 2
                continue
            if c == "/" and nxt == "*":
                mode = "block"
                i += 2
                continue
            if c == "/" and (prev_sig == "" or prev_sig in "(,=:[!&|?{};\n"):
                mode = "regex"  # regex literal, not division
                i += 1
                continue
            if c == '"':
                mode = "str"
                i += 1
                continue
            if c == "'":
                mode = "chr"
                i += 1
                continue
            if c == "`":
                mode = "tstr"
                i += 1
                continue
            out.append(c)
            if not c.isspace():
                prev_sig = c
            i += 1
            continue
        if mode == "line":
            if c == "\n":
                mode = "code"
                out.append(c)
            i += 1
            continue
        if mode == "block":
            if c == "*" and nxt == "/":
                mode = "code"
                i += 2
                continue
            i += 1
            continue
        if mode == "regex":
            # skip to the closing unescaped '/'; flags after it land as code
            if c == "\\":
                i += 2
                continue
            if c == "/":
                mode = "code"
            i += 1
            continue
        # string modes: honor backslash escapes, close on the opening quote
        if c == "\\":
            i += 2
            continue
        if (mode == "str" and c == '"') or (mode == "chr" and c == "'") \
                or (mode == "tstr" and c == "`"):
            mode = "code"
        i += 1
    return "".join(out)


def _dispatch_branches(qml: str) -> dict:
    """{'toolName': 'body of its else-if branch'} in the tool dispatch chain,
    split at each '} else if (toolName === ' boundary."""
    parts = qml.split('} else if (toolName === "')
    branches: dict = {}
    for part in parts[1:]:
        name = part.split('")', 1)[0]
        branches.setdefault(name, part.split('} else if (toolName === "', 1)[0])
    return branches


def _append_blocks(qml: str) -> List[str]:
    """Every chatHistory.append({...}) block, brace-matched on the raw text."""
    blocks: List[str] = []
    needle = "chatHistory.append({"
    pos = 0
    while True:
        start = qml.find(needle, pos)
        if start == -1:
            return blocks
        i = start + len(needle) - 1  # at the '{'
        depth, j = 0, i
        while j < len(qml):
            if qml[j] == "{":
                depth += 1
            elif qml[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        blocks.append(qml[i:j + 1])
        pos = j + 1


@unittest.skipIf(not (QML_PATH.exists() and GATE_PATH.exists()),
                 "AiAssistant.qml / CommandGate.qml absent — QML guards skip")
class CommandGateGuardTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.qml = QML_PATH.read_text(encoding="utf-8")
        cls.gate = GATE_PATH.read_text(encoding="utf-8")
        cls.branches = _dispatch_branches(cls.qml)

    # ------------------------------------------------------------------ gate

    def test_gate_is_singleton_with_pinned_api(self) -> None:
        self.assertIn("pragma Singleton", self.gate)
        for fn in ("function isReadOnlyCaelestia(", "function requestCaelestia(",
                   "function requestOpenApp(", "function requestTimer(",
                   "function confirm(", "function cancel("):
            self.assertIn(fn, self.gate, f"CommandGate.qml lost {fn.rstrip('(')}")

    def test_gate_allowlist_is_the_verified_read_only_set(self) -> None:
        # `version` / `help` — verified print-and-exit subcommands.
        self.assertIn('caelestiaReadOnly: ["version", "help"]', self.gate)
        # bare read forms of scheme only (`scheme list`, `scheme get` — argv
        # length 3, no extra args); `scheme set` is not allow-listed.
        self.assertIn('if (sub === "scheme" && argv.length === 3)', self.gate)
        self.assertIn('return mode === "list" || mode === "get";', self.gate)
        # `shell` is deliberately NOT allow-listed: cmd_shell ends with
        # `helper caelestia-shell-ipc start` on EVERY invocation.
        self.assertIn("`caelestia-shell-ipc start` on EVERY invocation",
                      self.gate)

    def test_gate_itself_executes_nothing(self) -> None:
        for banned in ("Process", "runAgentCommand", "Qt.createQmlObject",
                       "XMLHttpRequest"):
            self.assertNotIn(banned, self.gate,
                              f"the gate must never execute; found {banned}")

    def test_gate_brackets_balance(self) -> None:
        code = _strip_strings_and_comments(self.gate)
        for op, cl in (("{", "}"), ("(", ")"), ("[", "]")):
            self.assertEqual(code.count(op), code.count(cl),
                             f"unbalanced {op}{cl} in CommandGate.qml")

    # --------------------------------------------------------------- wiring

    def test_gated_tools_route_through_the_gate(self) -> None:
        for tool in GATED_TOOLS:
            self.assertIn(tool, self.branches, f"no dispatch branch for {tool}")
            body = self.branches[tool]
            self.assertIn("CommandGate.request", body,
                          f"{tool} must go through the gate, not run directly")

    def test_open_app_and_set_timer_have_no_direct_run(self) -> None:
        # Only caelestia_command keeps a direct run — the verified read-only
        # fast path, guarded by needsConfirm.
        for tool in ("open_app", "set_timer"):
            self.assertNotIn("runAgentCommand(", self.branches[tool],
                             f"{tool} must not execute from its dispatch branch")
        cc = self.branches["caelestia_command"]
        self.assertIn("cmdGate.needsConfirm", cc,
                      "caelestia_command's direct run must sit behind the "
                      "allow-list's needsConfirm guard")
        self.assertLess(cc.index("cmdGate.needsConfirm"),
                       cc.index("runAgentCommand("),
                       "the direct run must come after the gate check")

    def test_set_timer_counted_as_async(self) -> None:
        # set_timer's result now arrives at card-Apply time, so it must be in
        # the async counting condition or the tool loop would finish early.
        cond = next(l for l in self.qml.splitlines()
                    if 'toolName === "take_screenshot"' in l
                    and 'toolName === "caelestia_command"' in l)
        self.assertIn('toolName === "set_timer"', cond)

    # ----------------------------------------------------------------- card

    def test_command_card_plumbing(self) -> None:
        for fn in ("function showCommandConfirmCard(", "function resolveCommandCard("):
            self.assertIn(fn, self.qml, f"AiAssistant.qml lost {fn.rstrip('(')}")
        expected_props = {
            "required property bool isCommandPlan",
            "required property string cmdLabel",
            "required property var cmdLines",
            "required property string cmdResolved",
            "required property string cmdResult",
        }
        for prop in expected_props:
            self.assertIn(prop, self.qml,
                          f"the chat delegate lost '{prop}' — the card roles "
                          "must be required properties on the delegate")
        self.assertIn("onClicked: root.resolveCommandCard(true)", self.qml)
        self.assertIn("onClicked: root.resolveCommandCard(false)", self.qml)
        self.assertIn("visible: delegateItem.isCommandPlan", self.qml)
        self.assertIn("visible: !delegateItem.isSettingsPlan && !delegateItem.isCommandPlan",
                      self.qml)

    def test_every_append_carries_command_roles(self) -> None:
        blocks = _append_blocks(self.qml)
        self.assertGreaterEqual(len(blocks), 7, "fewer chatHistory.append blocks "
                                "than expected — did the chat plumbing change?")
        for block in blocks:
            self.assertIn('"isCommandPlan"', block,
                          "a chatHistory.append block is missing the "
                          "isCommandPlan role (ListModel roles must be set "
                          "on EVERY appended row)")

    def test_open_app_launch_pipeline_only_from_gated_path(self) -> None:
        # one definition + exactly one call site, inside resolveCommandCard
        self.assertEqual(self.qml.count("openAppCommand("), 2,
                         "openAppCommand must be defined once and called "
                         "only from the card's Apply resolution")

    # ------------------------------------------------------------ sysprompt

    def test_sysprompt_subcommands_match_verified_cli(self) -> None:
        desc = next((l for l in self.qml.splitlines()
                     if "Valid subcommands:" in l), "")
        self.assertIn(VERIFIED_SUBCOMMANDS, desc,
                      "the caelestia_command tool description drifted from "
                      "the verified upstream subcommand list")
        for stale in STALE_SUBCOMMANDS:
            self.assertNotIn(f", {stale}," , desc,
                             f"stale subcommand {stale!r} reappeared")
        self.assertIn("approval card", desc,
                      "the description must tell the model about the gate")

    def test_sidebar_brackets_balance(self) -> None:
        code = _strip_strings_and_comments(self.qml)
        for op, cl in (("{", "}"), ("(", ")"), ("[", "]")):
            self.assertEqual(code.count(op), code.count(cl),
                             f"unbalanced {op}{cl} in AiAssistant.qml")


if __name__ == "__main__":
    unittest.main()
