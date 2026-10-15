#!/usr/bin/env python3
"""Regenerate the three tables embedded in shell/services/SettingsTools.qml
from assistant/settings/tools.json.

This restores the documented workflow (CONTRIBUTING.md and the drift-failure
message in test_qml_service.py): after rebuilding tools.json, run

    python3 scripts/assemble_settings_tools.py          # rewrite tables
    python3 scripts/assemble_settings_tools.py --check  # verify only, rc=1 on drift

The renderer is imported from the byte-identity test itself so the table
format has exactly one source of truth — the independent mini-renderer the
test compares against.

Read-only outside the repo; writes only SettingsTools.qml. Never touches
tools.json (that is gen_adapter's job).
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tests.settings.test_qml_service import (  # noqa: E402
    QML_PATH,
    TOOLS_JSON,
    TABLE_START,
    _render_expected,
)


def splice(check_only: bool) -> int:
    qml = QML_PATH.read_text(encoding="utf-8")
    data = None
    import json

    data = json.loads(TOOLS_JSON.read_text(encoding="utf-8"))
    expected = _render_expected(data)

    lines = qml.splitlines()
    try:
        start = lines.index(TABLE_START)
    except ValueError:
        print("FAIL: toolTable marker not found in", QML_PATH)
        return 2
    # region ends at the closing "    ]" of explainTable: find the
    # explainTable declaration, then its terminator
    try:
        exp_decl = next(
            i for i, l in enumerate(lines)
            if l.startswith("    readonly property var explainTable: [")
        )
    except StopIteration:
        print("FAIL: explainTable marker not found in", QML_PATH)
        return 2
    close = exp_decl
    while lines[close] != "    ]":
        close += 1
        if close - exp_decl > 100000:
            print("FAIL: explainTable terminator not found")
            return 2

    current = lines[start:close + 1]
    if current == expected:
        print("OK: SettingsTools.qml tables match tools.json "
              f"({len(data['tools'])} tools); nothing to do")
        return 0

    if check_only:
        print("DRIFT: SettingsTools.qml tables differ from tools.json "
              f"(tools.json has {len(data['tools'])} tools); "
              "re-run without --check to rewrite")
        return 1

    new_lines = lines[:start] + expected + lines[close + 1:]
    QML_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"wrote {QML_PATH}: replaced {len(current)} lines with "
          f"{len(expected)} ({len(data['tools'])} tools, "
          f"{len(data['presets'])} presets, {len(data['explain_rules'])} explain rules)")
    return 0


if __name__ == "__main__":
    sys.exit(splice(check_only="--check" in sys.argv[1:]))
