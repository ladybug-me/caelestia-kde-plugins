"""Parity guards: the sidebar's caelestia_genius_* tools vs the bridge ops.

The sidebar's system-prompt registry (shell/modules/sidebar/AiAssistant.qml)
describes five deterministic compute tools backed by the assistant JSON
bridge's genius ops (assistant/brain/bridge.py). These tests pin the two
halves together the same way the settings suite pins tools.json to
SettingsTools.qml:

  * the bridge op lambdas must read exactly the parameters the test pins
    (introspected from the lambda sources, so a drift in bridge.py fails);
  * each tool's registry entry must appear in the QML BYTE-EXACTLY as the
    expected rendering built from the pinned spec (so a drift in the QML
    fails);
  * every registered tool must have a dispatch handler and async-count
    wiring, and there must be no undocumented caelestia_genius_* entries.

Note on escaping: the registry lives inside a JS double-quoted string, so
the file holds backslash-n for newlines, backslash-quote for quotes and
literal tool_call tags around the Example JSON. Escapes/tags below are
built with chr() so nothing in the transport can eat them.
"""

import json
import re
import unittest
from pathlib import Path

from assistant.capabilities.brain import bridge

REPO_ROOT = Path(__file__).resolve().parents[2]
QML_PATH = (REPO_ROOT / "assistant" / "adapters" / "caelestia" / "shell" /
            "modules" / "sidebar" / "AiAssistant.qml")

NL = chr(92) + "n"      # literal backslash-n, as the file holds it
EQ = chr(92) + '"'      # literal backslash-quote
TAG_OPEN = chr(60) + "tool_call" + chr(62)
TAG_CLOSE = chr(60) + "/tool_call" + chr(62)
DQ = chr(34)            # plain double quote, keeps patterns readable


def _js(obj) -> str:
    """JSON in the registry's house style (escaped quotes, spaced separators)."""
    return json.dumps(obj, separators=(", ", ": ")).replace('"', EQ)


def _entry(name: str, desc: str, args: str, example_args: dict) -> str:
    """One AVAILABLE TOOLS entry in the exact house format."""
    return (f"- {name}: {desc} Args: {args}." + NL + "  Example: " + TAG_OPEN
            + NL + _js({"name": name, "args": example_args}) + NL + TAG_CLOSE)


def _q(text: str) -> str:
    """Wrap a hint token in escaped quotes (as the file holds them)."""
    return EQ + text + EQ


# The pinned surface. (name, description, args text, example args,
#                      required params, optional params)
GENIUS_TOOLS = (
    ("caelestia_genius_math",
     "Evaluates a math expression deterministically on-device — no model in"
     " the loop (arithmetic, ^, sqrt, sin/cos/tan, log, factorial, pi and"
     " e).",
     "expr (string, required — e.g. " + _q("840*0.15") + " or "
     + _q("2^10 + sqrt(144)") + ")",
     {"expr": "840*0.15"},
     ("expr",), ()),
    ("caelestia_genius_stats",
     "Computes full descriptive statistics for a list of numbers on-device"
     " (mean, median, quartiles, spread, outliers, normality when n >= 8).",
     "numbers (array of numbers, required — e.g. [3, 9, 12, 1, 44])",
     {"numbers": [3, 9, 12, 1, 44]},
     ("numbers",), ()),
    ("caelestia_genius_logic",
     "Classifies a propositional logic formula as tautology, contradiction"
     " or contingency and shows the truth table — deterministic, no model.",
     "formula (string, required — e.g. " + _q("(p and q) -> p") + ")",
     {"formula": "(p and q) -> p"},
     ("formula",), ()),
    ("caelestia_genius_decide",
     "Ranks options against weighted criteria with a classical decision"
     " method (WSM/TOPSIS/...) and returns a winner with the full ranking —"
     " deterministic and auditable, no model judgement. Use it when the"
     " user asks which of several options to pick.",
     "matrix (array of number rows, required — one row per option, e.g."
     " [[8,256],[6,512]]), labels (array of strings, required — one per"
     " option, e.g. [" + _q("air") + "," + _q("pro") + "]), criteria (array"
     " of strings, required — one per column, e.g. [" + _q("battery") + ","
     + _q("storage") + "]), weights (array of numbers, optional — one per"
     " criterion, defaults to equal), benefits (array of booleans, optional"
     " — true = more is better; mark cost criteria like storage-in-GB as"
     " false), method (string, optional — wsm (default), wpm, topsis,"
     " pareto, regret)",
     {"matrix": [[8, 256], [6, 512]], "labels": ["air", "pro"],
      "criteria": ["battery", "storage"], "weights": [0.5, 0.5],
      "benefits": [True, False], "method": "topsis"},
     ("matrix", "labels", "criteria"),
     ("weights", "benefits", "method")),
    ("caelestia_genius_palette",
     "Builds an OKLch harmony palette from a hex color and checks WCAG"
     " contrast — run it BEFORE proposing an accent or scheme change so the"
     " suggestion is grounded (scheme changes themselves stay inert"
     " suggestions the user applies).",
     "hex (string, required — e.g. " + _q("#3b7dd8") + "), harmony (string,"
     " optional — analogous (default), complementary, triadic, tetradic,"
     " split_complementary), n (number, optional — palette size, default 5)",
     {"hex": "#3b7dd8", "harmony": "triadic"},
     ("hex",), ("harmony", "n")),
    ("caelestia_genius_graphs",
     "Runs classical graph algorithms on-device (Dijkstra shortest paths,"
     " Kruskal minimum spanning tree, Hungarian assignment) — exact and"
     " deterministic, no model.",
     "method (string, optional — dijkstra (default), mst, assign),"
     " graph (object, for dijkstra — {node: {neighbor: weight}}),"
     " source (string, for dijkstra), target (string, optional —"
     " dijkstra), nodes (array of strings, for mst), edges (array of"
     " [u, v, weight], for mst), cost (array of number rows, for assign)",
     {"method": "dijkstra", "graph": {"a": {"b": 2.0}}, "source": "a"},
     (), ("method", "graph", "source", "target", "nodes", "edges",
          "cost")),
    ("caelestia_genius_units",
     "Converts units and does dimension-checked arithmetic on-device"
     " \u2014 mismatched dimensions are rejected, never coerced (SI"
     " prefixes, compound units, affine temperature conversion).",
     "text (string, required \u2014 e.g. " + _q("convert 5 m to cm") + ", "
     + _q("3 km in m") + " or " + _q("3 km + 200 m") + ")",
     {"text": "convert 5 m to cm"},
     ("text",), ()),
    ("caelestia_genius_optimize",
     "Runs the read-only settings optimizer preview on-device \u2014"
     " Pareto fronts and AC-3 constraint propagation rank which preset"
     " profile fits the stated priority; PREVIEW ONLY, it changes"
     " nothing (apply stays a separate, confirmed action).",
     "profile (string, required \u2014 one of compact, minimal, gaming,"
     " battery-saver, macos-like), k (number, optional \u2014 how many"
     " recommendations to return, default 6)",
     {"profile": "battery-saver", "k": 3},
     (), ("profile", "k")),
    ("caelestia_genius_fsbrain_summary",
     "Reads the filesystem second-brain staleness REPORT for a folder"
     " on-device (frecency-ranked stale files, read-only). This is the"
     " report only \u2014 it never moves, renames or deletes anything"
     " (the tidy mover is a separate, confirmed tool).",
     "path (string, required \u2014 the directory to summarize), top"
     " (number, optional \u2014 rows kept, default 10), half_life_days"
     " (number, optional \u2014 frecency half-life, default 30)",
     {"path": "~/documents", "top": 5},
     ("path",), ("top", "half_life_days")),
)

OP_OF = {
    "caelestia_genius_math": "genius_math",
    "caelestia_genius_stats": "genius_stats",
    "caelestia_genius_logic": "genius_logic",
    "caelestia_genius_decide": "genius_decide",
    "caelestia_genius_palette": "genius_palette",
    "caelestia_genius_graphs": "genius_graphs",
    "caelestia_genius_units": "genius_units",
    "caelestia_genius_optimize": "optimize_recommend",
    "caelestia_genius_fsbrain_summary": "genius_fsbrain_summary",
}

# q["key"]  /  q.get("key"  — as regex, quotes built from DQ
_REQUIRED_RE = "q" + chr(92) + "[" + DQ + "(" + chr(92) + "w+)" + DQ + "]"
_OPTIONAL_RE = ("q" + chr(92) + ".get" + chr(92) + "(" + DQ + "("
                + chr(92) + "w+)" + DQ)


BRIDGE_SOURCE = Path(bridge.__file__).read_text(encoding="utf-8")


def _op_source(op: str) -> str:
    """The op's registration slice, read from bridge.py itself.

    Reading the shipped source text (not runtime introspection) mirrors
    the tools.json <-> SettingsTools.qml pattern and keeps this test's
    imports on the ALLOWED_IMPORTS allow-list.
    """
    start = BRIDGE_SOURCE.index(chr(34) + op + chr(34) + ": lambda")
    end = BRIDGE_SOURCE.index(chr(10) + "    " + chr(34), start)
    return BRIDGE_SOURCE[start:end]


def _op_params(op: str):
    """(required, optional) query keys the bridge op actually reads."""
    src = _op_source(op)
    required = tuple(dict.fromkeys(re.findall(_REQUIRED_RE, src)))
    optional = tuple(dict.fromkeys(re.findall(_OPTIONAL_RE, src)))
    return required, optional


@unittest.skipIf(not QML_PATH.exists(),
                 "AiAssistant.qml absent — QML guards skip")
class GeniusQmlParityTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.qml = QML_PATH.read_text(encoding="utf-8")

    def _registry_region(self) -> str:
        start = self.qml.find("AVAILABLE TOOLS:")
        end = self.qml.find("CRITICAL RULES:")
        self.assertGreater(start, -1)
        self.assertGreater(end, start)
        return self.qml[start:end]

    def test_bridge_ops_read_exactly_the_pinned_params(self) -> None:
        # Introspection guard: if bridge.py's lambdas start reading other
        # or fewer query keys, the pinned registry text below is stale.
        for name, _d, _a, _ex, required, optional in GENIUS_TOOLS:
            got_req, got_opt = _op_params(OP_OF[name])
            self.assertEqual(got_req, required, f"{OP_OF[name]} required")
            self.assertEqual(got_opt, optional, f"{OP_OF[name]} optional")

    def test_registry_entries_are_byte_identical(self) -> None:
        region = self._registry_region()
        for name, desc, args, example, _req, _opt in GENIUS_TOOLS:
            expected = _entry(name, desc, args, example)
            self.assertIn(expected, region,
                          f"{name} registry entry drifted from the pinned "
                          f"format/args — update QML and this guard together")

    def test_no_undocumented_genius_entries(self) -> None:
        found = set(re.findall(r"- (caelestia_genius_\w+):", self.qml))
        self.assertEqual(found, {t[0] for t in GENIUS_TOOLS})

    def test_every_tool_is_wired_in_the_dispatch_chain(self) -> None:
        for name, _d, _a, _ex, _req, _opt in GENIUS_TOOLS:
            self.assertIn(f'toolName === "{name}")', self.qml,
                          f"{name} has no dispatch handler")
        # async tool accounting: every genius tool must be in the counting
        # condition (the one that also lists caelestia_command), or the
        # model would hang waiting for a synchronous result
        cond = next(l for l in self.qml.splitlines()
                    if 'toolName === "take_screenshot"' in l
                    and 'toolName === "caelestia_command"' in l)
        counted = set(re.findall(r'toolName === "([a-z_]+)"', cond))
        for name, _d, _a, _ex, _req, _opt in GENIUS_TOOLS:
            self.assertIn(name, counted,
                          f"{name} missing from the async counting condition")

    def test_example_json_covers_required_params(self) -> None:
        for name, _d, _a, example, required, _opt in GENIUS_TOOLS:
            for key in required:
                self.assertIn(key, example, f"{name} example misses {key}")


if __name__ == "__main__":
    unittest.main()
