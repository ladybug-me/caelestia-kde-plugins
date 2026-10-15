"""QML service cross-checks: shell/services/SettingsTools.qml.

No Qt toolchain is required (or assumed) — these tests statically guard the
shipped QML against the Python registry:

1. TABLE IDENTITY: the tool/preset/explain tables embedded in the QML are
   re-rendered from tools.json by an INDEPENDENT mini-renderer inside this
   test and compared line-for-line, so the QML service and the Python layer
   cannot drift apart without failing the suite.
2. BALANCE: braces/parens/brackets balance outside comments and string
   literals (a cheap syntax sanity net; qmllint is the real check upstream).
3. API SURFACE: every function the AiAssistant.qml patch depends on exists.
4. BOUNDS: the undo history bound (>= the required 10).

If the QML file is absent (packaged installs without the shell tree), all
tests SKIP loudly.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import List

from assistant.adapters.caelestia.registry import TOOL_COUNT

QML_PATH = (Path(__file__).resolve().parents[2] / "assistant" / "adapters" / "caelestia" /
            "shell" / "services" / "SettingsTools.qml")
TOOLS_JSON = Path(__file__).resolve().parents[2] / "assistant" / "adapters" / "caelestia" / "tools.json"

TABLE_START = "    readonly property var toolTable: ["
TABLE_END = "    ]"


def _qm(v) -> str:
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    return json.dumps(v)


def _render_expected(data) -> List[str]:
    """Independent re-render of the three generated tables."""
    out: List[str] = []
    out.append("    readonly property var toolTable: [")
    for t in data["tools"]:
        cites = ", ".join(f"[{_qm(c[0])}, {_qm(c[1])}]" for c in t["citations"])
        out.append(
            f'        {{ n: {_qm(t["name"])}, p: {_qm(t["path"])}, '
            f'g: {_qm(t["group"])}, k: {_qm(t["kind"])}, d: {_qm(t["default"])}, '
            f'lo: {_qm(t.get("minimum"))}, hi: {_qm(t.get("maximum"))}, '
            f'st: {_qm(t.get("step"))}, en: {_qm(t.get("enum"))}, '
            f'sl: {_qm(t.get("string_max_len"))}, '
            f'go: {"true" if t.get("global_only") else "false"}, c: [{cites}] }},'
        )
    out.append("    ]")
    out.append("")
    out.append("    // Named presets: bundles of validated tool calls only (see")
    out.append("    // curations.PRESETS; build_registry validates every call).")
    out.append("    readonly property var presetTable: [")
    for p in data["presets"]:
        calls = ", ".join(f'[{_qm(c["tool"])}, {_qm(c["value"])}]' for c in p["calls"])
        out.append(
            f'        {{ name: {_qm(p["name"])}, label: {_qm(p["label"])}, '
            f'description: {_qm(p["description"])}, calls: [{calls}] }},'
        )
    out.append("    ]")
    out.append("")
    out.append("    // Explainability rules: state predicate + answer template + citations.")
    out.append("    // Rendered identically by the Python CLI (assistant/settings/explain.py).")
    out.append("    readonly property var explainTable: [")
    for r in data["explain_rules"]:
        cites = ", ".join(_qm(c) for c in r["cites"])
        out.append(
            f'        {{ path: {_qm(r["path"])}, when: {_qm(r["when"])}, '
            f'answer: {_qm(r["answer"])}, cites: [{cites}] }},'
        )
    out.append("    ]")
    return out


def _strip_strings_and_comments(text: str) -> str:
    """Remove // comments, /* */ comments, and string/template contents so
    bracket counting only sees code. Conservative and small on purpose."""
    out: List[str] = []
    i, n = 0, len(text)
    mode = "code"  # code | line | block | str | tstr
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
            if c == '"':
                mode = "str"
                i += 1
                continue
            if c == "`":
                mode = "tstr"
                i += 1
                continue
            out.append(c)
            i += 1
        elif mode == "line":
            if c == "\n":
                mode = "code"
                out.append(c)
            i += 1
        elif mode == "block":
            if c == "*" and nxt == "/":
                mode = "code"
                i += 2
                continue
            if c == "\n":
                out.append(c)
            i += 1
        elif mode == "str":
            if c == "\\":
                i += 2
                continue
            if c == '"':
                mode = "code"
            i += 1
        else:  # tstr
            if c == "\\":
                i += 2
                continue
            if c == "`":
                mode = "code"
            i += 1
    return "".join(out)


@unittest.skipIf(not QML_PATH.exists(), "SettingsTools.qml absent — QML guards skip")
class SettingsToolsQmlTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.qml = QML_PATH.read_text(encoding="utf-8")
        cls.data = json.loads(TOOLS_JSON.read_text(encoding="utf-8"))

    def _extract_block(self, start_marker: str) -> List[str]:
        lines = self.qml.splitlines()
        for idx, line in enumerate(lines):
            if line == start_marker:
                end = idx
                while lines[end] != TABLE_END:
                    end += 1
                return lines[idx:end + 1]
        self.fail(f"marker {start_marker!r} not found in the QML")

    def test_tool_table_matches_tools_json(self) -> None:
        expected = _render_expected(self.data)
        # The whole rendered region: toolTable .. explainTable
        got = self._extract_block(TABLE_START)
        # expected spans three tables; extract the full region instead
        lines = self.qml.splitlines()
        start = next(i for i, l in enumerate(lines) if l == TABLE_START)
        end = next(i for i, l in enumerate(lines) if l == "    readonly property var explainTable: [")
        # find the closing ] of explainTable
        close = end
        while lines[close] != TABLE_END:
            close += 1
        got = lines[start:close + 1]
        self.assertEqual(got, expected,
                         "SettingsTools.qml tables drifted from tools.json: "
                         "re-run scripts/assemble_settings_tools.py")

    def test_tool_count_in_table(self) -> None:
        rows = [l for l in self.qml.splitlines() if l.strip().startswith("{ n: ")]
        self.assertEqual(len(rows), TOOL_COUNT)

    def test_brackets_balance(self) -> None:
        code = _strip_strings_and_comments(self.qml)
        for op, cl in (("{", "}"), ("(", ")"), ("[", "]")):
            self.assertEqual(code.count(op), code.count(cl),
                             f"unbalanced {op}{cl} in SettingsTools.qml")

    def test_api_surface(self) -> None:
        for fn in ("function toolInfo(", "function listTools(", "function get(",
                   "function validate(", "function readPath(", "function writePath(",
                   "function buildPlan(", "function request(", "function previewOf(",
                   "function confirmPlan(", "function cancelPlan(", "function applyOps(",
                   "function pushHistory(", "function historyEntries(",
                   "function undo(", "function undoById(", "function explain(",
                   "function presetList(", "function requestPreset("):
            self.assertIn(fn, self.qml, f"SettingsTools.qml lost {fn.rstrip('(')}")

    def test_history_bound_at_least_ten(self) -> None:
        self.assertIn("maxHistory: 12", self.qml)
        self.assertIn("oldest evicted", self.qml)

    def test_confirm_flow_policy_present(self) -> None:
        # The issue-#120 policy: multi-op -> pendingPlan, nothing written
        # until confirmPlan(); single-op applies directly.
        self.assertIn("needsConfirm", self.qml)
        self.assertIn("pendingPlan", self.qml)
        self.assertIn("nothing is written until", self.qml)


if __name__ == "__main__":
    unittest.main()
