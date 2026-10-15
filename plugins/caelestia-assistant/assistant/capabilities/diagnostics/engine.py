"""Layer 1 diagnostic engine: load rules, match signatures, render a plan.

Guarantees:
- Pure stdlib; no subprocess/os.system anywhere — this module has no way to
  execute the commands it suggests. Suggested commands are inert strings.
- Deterministic: identical input always produces an identical report. No
  fuzzy matching, no first-match-wins, no silent tie-breaking.
- Ambiguity is surfaced, not hidden: when two rules match within the margin,
  the report lists the top candidates and asks a clarifying question.

Matcher semantics:
- {"allOf": [leaves]} — gate: every required leaf must hold.
- {"anyOf": [leaves]} — met when at least one *required* member holds;
  optional members add evidence only.
- Leaves: text_regex, text_substring, file_exists, cmd_output.
  file_exists is the only probe the engine performs itself (os.path, read
  only). cmd_output is satisfied only by output the user pasted.

Scoring (fixed weights, no ML):
    required text match            +2
    required file_exists satisfied +3
    required cmd_output confirmed  +3
    each optional leaf matched     +1
    deterministic rule passing gate +2

Ranking is the stable tuple
    (-score, deterministic-first, id).
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import argumentation
from . import rete
from .settings_join import settings_tools_for_rule

RULES_DIR = Path(__file__).resolve().parent / "rules.d"

CONFIDENCE_RANK = {"deterministic": 0, "probable": 1}

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Only these variables are expanded inside file_exists paths; everything
# else is left literal so a typo can never point somewhere unintended.
ENV_ALLOWLIST = ["HOME", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_RUNTIME_DIR"]

COMMAND_PREFIX = "SUGGESTED_NOT_EXECUTED:"

VALID_CATEGORIES = [
    "install",
    "runtime-shell",
    "runtime-lockscreen",
    "config",
    "network",
    "kde",
    "post-install",
    "uninstall",
    "update",
    "folder-cleanup",
]

VALID_SEVERITIES = ["critical", "high", "medium", "low", "info"]
VALID_RISKS = ["READ_ONLY", "STATE_CHANGING", "PRIVILEGED", "DESTRUCTIVE"]


class RuleError(ValueError):
    """Raised when a rule file violates the schema."""


# ---------------------------------------------------------------------------
# Rule validation (structural; used at load time)
# ---------------------------------------------------------------------------


def _walk_matchers(node: Dict[str, Any], depth: int = 1) -> List[Dict[str, Any]]:
    """Collect leaves of a matcher tree; combinators nest at most one level."""
    if depth > 2:
        raise ValueError("matcher tree nests deeper than one combinator level")
    if "allOf" in node or "anyOf" in node:
        members = node.get("allOf") or node.get("anyOf") or []
        leaves: List[Dict[str, Any]] = []
        for member in members:
            if "allOf" in member or "anyOf" in member:
                leaves.extend(_walk_matchers(member, depth + 1))
            else:
                leaves.append(member)
        return leaves
    return [node]


def validate_rule(rule: Dict[str, Any], source: str = "<memory>") -> List[str]:
    """Structural validation of one rule; returns a list of failures."""
    from . import risk as risk_mod

    failures: List[str] = []
    rule_id = rule.get("id", "<no-id>")

    for field in ("id", "title", "category", "severity", "confidence", "matchers", "fix"):
        if field not in rule:
            failures.append(f"{source} {rule_id}: missing required field {field!r}")
    if failures:
        return failures

    if rule["category"] not in VALID_CATEGORIES:
        failures.append(f"{source} {rule_id}: bad category {rule['category']!r}")
    if rule["severity"] not in VALID_SEVERITIES:
        failures.append(f"{source} {rule_id}: bad severity {rule['severity']!r}")
    if rule["confidence"] not in CONFIDENCE_RANK:
        failures.append(f"{source} {rule_id}: bad confidence {rule['confidence']!r}")

    try:
        leaves = _walk_matchers(rule["matchers"])
    except ValueError as exc:
        failures.append(f"{source} {rule_id}: {exc}")
        return failures

    for leaf in leaves:
        kind = leaf.get("type")
        if kind == "text_regex":
            try:
                re.compile(leaf.get("pattern", ""))
            except re.error as exc:
                failures.append(f"{source} {rule_id}: invalid regex {leaf.get('pattern')!r}: {exc}")
        elif kind == "text_substring":
            if not isinstance(leaf.get("value"), str):
                failures.append(f"{source} {rule_id}: text_substring missing value")
        elif kind == "file_exists":
            if "path" not in leaf:
                failures.append(f"{source} {rule_id}: file_exists missing path")
            elif leaf.get("expected", "present") not in ("present", "absent"):
                failures.append(f"{source} {rule_id}: file_exists bad expected")
        elif kind == "cmd_output":
            if "command" not in leaf or "pattern" not in leaf:
                failures.append(f"{source} {rule_id}: cmd_output needs command+pattern")
        else:
            failures.append(f"{source} {rule_id}: unknown matcher type {kind!r}")

    if not isinstance(rule["fix"], list) or not rule["fix"]:
        failures.append(f"{source} {rule_id}: fix must be a non-empty list")
        return failures

    for step in rule["fix"]:
        if not isinstance(step, dict):
            failures.append(f"{source} {rule_id}: fix steps must be objects")
            continue
        if "command" in step and step.get("command_risk") not in risk_mod.RISK_ORDER:
            failures.append(
                f"{source} {rule_id}: command step missing/invalid command_risk {step.get('command_risk')!r}"
            )
    return failures


# ---------------------------------------------------------------------------
# Rule loading
# ---------------------------------------------------------------------------


def load_rules(rules_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load and validate every rules.d/*.json file, sorted by filename."""
    directory = Path(rules_dir) if rules_dir else RULES_DIR
    rule_files = sorted(glob.glob(str(directory / "*.json")))
    if not rule_files:
        raise RuleError(f"no rule files found in {directory}")
    rules: List[Dict[str, Any]] = []
    seen_ids: set = set()
    for path in rule_files:
        with open(path, "r", encoding="utf-8") as handle:
            envelope = json.load(handle)
        if envelope.get("schema_version") != 1:
            raise RuleError(f"{path}: unsupported schema_version")
        for rule in envelope.get("rules", []):
            validate_rule(rule, source=str(path))
            if rule["id"] in seen_ids:
                raise RuleError(f"{path}: duplicate rule id {rule['id']}")
            seen_ids.add(rule["id"])
            rules.append(rule)
    return rules


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------


def normalize_input(text: str) -> str:
    """Strip terminal noise so signatures match what the user meant."""
    text = ANSI_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ---------------------------------------------------------------------------
# Matcher evaluation
# ---------------------------------------------------------------------------


def _expand_path(path: str) -> str:
    """Expand ~ and the allow-listed XDG vars in a file probe path."""
    out = os.path.expanduser(path)
    for var in ENV_ALLOWLIST:
        token = f"${var}"
        if token in out:
            out = out.replace(token, os.environ.get(var, ""))
        token = f"${{{var}}}"
        if token in out:
            out = out.replace(token, os.environ.get(var, ""))
    return out


def _matched_lines(pattern: str, case_insensitive: bool, text: str) -> List[str]:
    """Return the input lines a pattern matched, for evidence display."""
    flags = re.IGNORECASE if case_insensitive else 0
    lines: List[str] = []
    for line in text.split("\n"):
        if re.search(pattern, line, flags):
            lines.append(line.strip())
    return lines


TEXT_LEAF_TYPES = ("text_regex", "text_substring")


def eval_leaf(leaf: Dict[str, Any], text: str) -> Tuple[bool, Optional[str]]:
    """Evaluate one matcher leaf.

    Returns (satisfied, evidence). For cmd_output leaves that the user has
    not corroborated, returns (False, None) — an unresolved probe is treated
    as unmet evidence, never as a hard gate failure (see module docstring).
    """
    kind = leaf.get("type")
    ci = bool(leaf.get("case_insensitive", False))

    if kind == "text_regex":
        pattern = leaf["pattern"]
        flags = re.IGNORECASE if ci else 0
        hit = re.search(pattern, text, flags) is not None
        evidence = _matched_lines(pattern, ci, text)
        return hit, (evidence[0] if evidence else None)

    if kind == "text_substring":
        value = leaf["value"]
        if ci:
            hit = value.lower() in text.lower()
            line = next((ln for ln in text.split("\n") if value.lower() in ln.lower()), None)
        else:
            hit = value in text
            line = next((ln for ln in text.split("\n") if value in ln), None)
        return hit, line

    if kind == "file_exists":
        expanded = _expand_path(leaf["path"])
        exists = os.path.exists(expanded)
        expected = leaf.get("expected", "present")
        satisfied = exists if expected == "present" else not exists
        shown = "present" if exists else "absent"
        return satisfied, f"{leaf['path']} -> {shown}"

    if kind == "cmd_output":
        # Never executed. Satisfied only when the user's paste contains the
        # command itself and its pattern matches the pasted output.
        command = leaf["command"]
        pattern = leaf["pattern"]
        flags = re.IGNORECASE if bool(leaf.get("case_insensitive", True)) else 0
        if command in text and re.search(pattern, text, flags):
            line = _matched_lines(re.escape(command), False, text)
            return True, line[0] if line else command
        return False, None

    raise RuleError(f"unknown matcher type: {kind!r}")


def _is_combinator(node: Dict[str, Any]) -> bool:
    return isinstance(node, dict) and ("allOf" in node or "anyOf" in node)


def eval_matchers(node: Dict[str, Any], text: str) -> Tuple[bool, List[str], int]:
    """Evaluate a matcher tree (combinators nested at most one level).

    Returns (gate_passed, evidence_lines, score_contribution). Scoring
    follows the fixed weight table in the module docstring; required leaves
    only contribute when satisfied, optional leaves contribute +1.
    """
    evidence: List[str] = []
    score_contrib = 0

    if _is_combinator(node):
        kind = "allOf" if "allOf" in node else "anyOf"
        members = node[kind]
        passed = True
        any_required = [m for m in members if bool(m.get("required", True))]
        any_required_met = False
        for member in members:
            ok, ev, contrib = eval_matchers(member, text)
            evidence.extend(ev)
            score_contrib += contrib
            required = bool(member.get("required", True))
            if required:
                if not ok:
                    passed = False
                else:
                    any_required_met = True
        if kind == "anyOf":
            # anyOf is met when at least one required member holds; a
            # group of only optional members is met if any member matched.
            passed = any_required_met if any_required else any(
                bool(m.get("required", True)) is False and _leaf_ok(m, text) for m in members
            )
            if not members:
                passed = False
        return passed, evidence, score_contrib

    # Plain leaf.
    ok, ev = eval_leaf(node, text)
    if ev:
        evidence.append(ev)
    required = bool(node.get("required", True))
    if required:
        if ok:
            if node.get("type") in TEXT_LEAF_TYPES:
                score_contrib += 2
            else:
                score_contrib += 3  # engine-verified or user-pasted evidence
    elif ok:
        score_contrib += 1
    return ok, evidence, score_contrib


def _leaf_ok(node: Dict[str, Any], text: str) -> bool:
    ok, _ev, _score = eval_matchers(node, text)
    return ok


# ---------------------------------------------------------------------------
# Scoring and decision
# ---------------------------------------------------------------------------


def score_rule(rule: Dict[str, Any], text: str) -> Optional[Dict[str, Any]]:
    """Score one rule against normalized text.

    Returns None when the required gate fails, otherwise a result dict.
    """
    passed, evidence, score = eval_matchers(rule["matchers"], text)
    if not passed:
        return None

    if rule.get("confidence") == "deterministic":
        score += 2
    return {
        "rule": rule,
        "score": score,
        "evidence": evidence,
    }


def rank_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stable total ordering: score desc, deterministic first, id asc."""

    def key(result: Dict[str, Any]) -> Tuple[Any, ...]:
        rule = result["rule"]
        return (
            -result["score"],
            CONFIDENCE_RANK.get(rule.get("confidence", "probable"), 1),
            rule["id"],
        )

    return sorted(results, key=key)


def diagnose(text: str, rules: Optional[List[Dict[str, Any]]] = None, top: int = 3) -> Dict[str, Any]:
    """Run the full deterministic pipeline over one input blob.

    Verdicts:
      MATCH     — exactly one gate-passing rule, or a clear margin (>=2
                  points) over the runner-up.
      AMBIGUOUS — several candidates within the margin; report top N with
                  clarify probes. Never silently pick one.
      NO_MATCH  — nothing passed its gate; the honest answer is "I don't
                  know", plus pointers to diagnostic commands and Layer 2.
    """
    normalized = normalize_input(text)
    rules = rules if rules is not None else load_rules()
    results = []
    for rule in rules:
        result = score_rule(rule, normalized)
        if result is not None:
            results.append(result)
    ranked = rank_results(results)

    # A4 — reverse lookup: attach the settings tools (if any) that address
    # each matched rule's root cause, via the citation registry join. A
    # pure, additive key: rules with no settings surface carry [].
    for result in ranked:
        result["settings_tools"] = settings_tools_for_rule(result["rule"])["tools"]

    if not ranked:
        result = {"verdict": "NO_MATCH", "candidates": [], "top": None}
        # exponential-build-4 A: lines no signature matched are TEMPLATED
        # (Drain, He et al. 2017) instead of silently discarded — shapes,
        # not causes, for a human to consider turning into a real rule.
        from . import drain
        return drain.augment_diagnosis(result, normalized)

    if len(ranked) == 1:
        margin: Optional[int] = None
    else:
        margin = ranked[0]["score"] - ranked[1]["score"]

    single = len(ranked) == 1 or (margin is not None and margin >= 2)
    if single:
        return {
            "verdict": "MATCH",
            "candidates": ranked[:top],
            "top": ranked[0],
            "margin": margin,
        }
    return {
        "verdict": "AMBIGUOUS",
        "candidates": ranked[:top],
        "top": ranked[0],
        "margin": margin,
    }


def _drain_lines(diagnosis: Dict[str, Any]) -> List[str]:
    """exponential-build-4 A: the Drain template block for the report,
    rendered only when the augment step actually attached templates."""
    from . import drain
    unmatched = diagnosis.get("unmatched_templates")
    if not unmatched:
        return []
    return drain.render_unmatched_lines(unmatched)


# ---------------------------------------------------------------------------
# Rendering (plain text, human readable, nothing executable)
# ---------------------------------------------------------------------------


def _format_refs(rule: Dict[str, Any]) -> str:
    parts: List[str] = []
    for ref in rule.get("references", []):
        if ref.get("kind") == "doc":
            parts.append(f"docs/TROUBLESHOOTING.md {ref.get('anchor', '')}".rstrip())
        elif ref.get("kind") == "issue":
            closing = f", closing PR #{ref['closing_pr']}" if ref.get("closing_pr") else ""
            parts.append(f"issue #{ref.get('number')} ({ref.get('state', '?')}{closing})")
    return "; ".join(parts) if parts else "-"


def render_result(result: Dict[str, Any]) -> str:
    rule = result["rule"]
    conf = rule.get("confidence", "probable")
    out: List[str] = []
    out.append(f"Rule {rule['id']} [{conf}/{rule.get('severity', '?')}]")
    out.append(f"  {rule.get('title', '')}")
    evidence = result.get("evidence") or []
    if evidence:
        out.append("  Evidence:")
        for ev in evidence[:6]:
            out.append(f"    - {ev}")
    fix_steps = rule.get("fix", [])
    if fix_steps:
        out.append("  Suggested fix (review every step; nothing is run for you):")
        for idx, step in enumerate(fix_steps, start=1):
            out.append(f"    {idx}. {step.get('text', '')}")
            command = step.get("command")
            if command:
                out.append(f"       {COMMAND_PREFIX} {command}")
                risk_tier = step.get("command_risk", "STATE_CHANGING")
                out.append(f"       [risk: {risk_tier}]")
            if step.get("warning"):
                out.append(f"       WARNING: {step['warning']}")
            if step.get("undo"):
                out.append(f"       undo: {step['undo']}")
    refs = _format_refs(rule)
    out.append(f"  References: {refs}")
    settings_tools = result.get("settings_tools") or settings_tools_for_rule(rule)["tools"]
    if settings_tools:
        listing = ", ".join(
            f"{entry['tool']} ({entry['path']})" for entry in settings_tools
        )
        out.append(f"  Settings tools addressing this root cause: {listing}")
    if rule.get("clarify_probe"):
        out.append(f"  Clarify: {rule['clarify_probe']}")
    return "\n".join(out)


def render_report(diagnosis: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("caelestia assistant — layer 1 diagnostics (deterministic rules)")
    lines.append("Nothing was executed, nothing was sent anywhere; this is a printed plan only.")
    lines.append("")
    verdict = diagnosis["verdict"]
    if verdict == "NO_MATCH":
        lines.append("Verdict: NO known signature matched.")
        lines.append("")
        lines.extend(_drain_lines(diagnosis))
        lines.append("")
        lines.append("Honest next steps instead of a guess:")
        lines.append("  1. Enable Debug Mode (Nexus -> About -> Advanced), reproduce, then run:")
        lines.append("       SUGGESTED_NOT_EXECUTED: journalctl --user -u caelestia-shell --no-pager -n 200")
        lines.append("  2. Skim the catalog: docs/TROUBLESHOOTING.md (section 11 lists diagnostic commands).")
        lines.append("  3. If nothing fits, draft an issue report:")
        lines.append("       SUGGESTED_NOT_EXECUTED: python3 -m assistant.capabilities.issues.cli draft --help")
        return "\n".join(lines)

    if verdict == "MATCH":
        lines.append(f"Verdict: MATCH (single strong candidate, margin {diagnosis.get('margin')} pts).")
    else:
        lines.append("Verdict: AMBIGUOUS — several rules fit; pick one or paste the probe output.")
    lines.append("")
    for idx, result in enumerate(diagnosis["candidates"], start=1):
        lines.append(f"[{idx}] {render_result(result)}")
        lines.append("")

    # Read-only enrichment (exponential-build-3 A1/A2): derived composite
    # evidence (Rete, rete.py) and conflict resolution (Dung,
    # argumentation.py) computed from THIS diagnosis without touching the
    # matching semantics above — rendered only when they fire, nothing
    # when they do not.
    derived = rete.derived_evidence(diagnosis)
    if derived is not None:
        lines.extend(rete.render_derived_lines(derived))
        lines.append("")
    conflicts = argumentation.conflict_resolution(diagnosis)
    if conflicts is not None:
        lines.extend(argumentation.render_conflict_lines(conflicts))
        lines.append("")

    lines.append("Commands above are suggestions to copy-paste after review. The assistant")
    lines.append("cannot execute them, cannot reach the network, and never edits your config.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry (python -m assistant.capabilities.diagnostics.cli)
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="caelestia-assist diagnose",
        description="Deterministic troubleshooting for caelestia-kde (reads text, prints a plan, runs nothing).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    diag = sub.add_parser("diagnose", help="match pasted text/logs against known signatures")
    diag.add_argument("input", nargs="?", default="-", help="text/log file to read, or - for stdin")
    diag.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    diag.add_argument("--top", type=int, default=3, help="max candidates shown when ambiguous")

    sub.add_parser("selfcheck", help="validate rule files and the no-executor safety pins")

    rp = sub.add_parser("rulepack", help="the signed rule-pack format: "
                                         "export/import/list/render "
                                         "(signatures live in YOUR "
                                         "external tool)")
    rp.add_argument("action", choices=["export", "import", "list",
                                       "render", "rules-d", "forget"])
    rp.add_argument("target", nargs="?", default="",
                    help="export: output path; import: pack path (or - "
                         "for stdin); render/rules-d/forget: pack id")
    rp.add_argument("--rulesets", default="",
                    help="export: comma-separated rules.d stems")
    rp.add_argument("--id", default="",
                    help="export: the pack id (required)")
    rp.add_argument("--signer", default="",
                    help="import: the identity YOU verified with your own "
                         "tool (recorded as a claim, never auto-trusted)")

    args = parser.parse_args(argv)

    if args.cmd == "rulepack":
        # exponential-build-3 F1: the signed rule-pack surface. The
        # module (diagnostics/rulepack.py) carries the full contract;
        # this CLI is a thin router over it.
        from . import rulepack
        from assistant.capabilities.brain import state as brain_state
        if args.action == "export":
            if not args.rulesets or not args.id:
                print("error: rulepack export needs --rulesets A,B and "
                      "--id NAME", file=sys.stderr)
                return 2
            try:
                pack = rulepack.export_pack(
                    Path(__file__).resolve().parent / "rules.d",
                    [s.strip() for s in args.rulesets.split(",")
                     if s.strip()], args.id)
            except rulepack.RulePackError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            text = json.dumps(pack, indent=2) + "\n"
            if args.target in ("", "-"):
                sys.stdout.write(text)
            else:
                Path(args.target).write_text(text, encoding="utf-8")
                print(f"wrote pack {args.id!r} to {args.target} "
                      "(sign it with YOUR external tool before sharing)")
            return 0
        if args.action == "import":
            if not args.target:
                print("error: rulepack import needs a pack path (or - "
                      "for stdin)", file=sys.stderr)
                return 2
            raw = (sys.stdin.read() if args.target == "-"
                   else Path(args.target).read_text(encoding="utf-8"))
            state = brain_state.load()
            existing = [rec.get("id") for rec in
                        state.get(rulepack.STATE_KEY, []) or []]
            try:
                record = rulepack.import_pack(raw, existing,
                                              signer=args.signer)
            except rulepack.RulePackError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            state.setdefault(rulepack.STATE_KEY, []).append(record)
            brain_state.save(state)
            print(f"imported pack {record['id']!r}: "
                  f"{record['rule_count']} rule(s), integrity + safety "
                  "verified; signer recorded as a CLAIM")
            print("imported rules are review data, NOT live — see: "
                  "caelestia-assist rulepack list / render")
            return 0
        if args.action == "list":
            packs = rulepack.list_packs(brain_state.load())
            if not packs:
                print("no rule packs imported; import one with "
                      "rulepack import FILE")
                return 0
            for rec in packs:
                signer = rec.get("signer") or "(none recorded)"
                print(f"  {rec['id']}: {rec['rule_count']} rule(s), "
                      f"signer {signer!r}, payload sha256 "
                      f"{str(rec.get('payload_sha256'))[:12]}")
            return 0
        if args.action in ("render", "rules-d"):
            state = brain_state.load()
            record = next(
                (rec for rec in state.get(rulepack.STATE_KEY, []) or []
                 if rec.get("id") == args.target), None)
            if record is None:
                print(f"error: no imported pack named {args.target!r}",
                      file=sys.stderr)
                return 1
            if args.action == "render":
                print(rulepack.render_pack(record))
            else:
                sys.stdout.write(rulepack.render_rules_d(record))
            return 0
        # forget
        state = brain_state.load()
        packs = state.get(rulepack.STATE_KEY, []) or []
        kept = [rec for rec in packs if rec.get("id") != args.target]
        if len(kept) == len(packs):
            print(f"error: no imported pack named {args.target!r}",
                  file=sys.stderr)
            return 1
        state[rulepack.STATE_KEY] = kept
        brain_state.save(state)
        print(f"forgot pack {args.target!r}")
        return 0

    if args.cmd == "selfcheck":
        from . import schema_lint

        failures = schema_lint.run_all_checks()
        for failure in failures:
            print(f"LINT FAIL: {failure}")
        if failures:
            return 1
        # Settings config-lint rule table (issue #120 Phase 1.1): the lint
        # rule table itself must be structurally sound. If a real shell.json
        # exists it is linted INFORMATIONALLY — findings about the user's
        # config never fail the selfcheck (the assistant's health is what
        # selfcheck verifies, not the user's config).
        from assistant.capabilities.settings import lint as config_lint
        rule_failures = config_lint.lint_rules_ok()
        for failure in rule_failures:
            print(f"LINT FAIL: settings lint rule table: {failure}")
        if rule_failures:
            return 1
        note = ""
        try:
            from assistant.capabilities.settings.cli import default_target
            target = default_target()
            if Path(target).exists():
                findings = config_lint.lint_file(target)
                if findings:
                    note = (f"; config lint of {target}: {len(findings)} "
                            "finding(s) (informational — see: caelestia-assist "
                            "settings --lint)")
        except (OSError, ValueError, ImportError):
            pass
        print("selfcheck OK: rule schema valid, risk tiers consistent, "
              "safety pins clean, settings lint rules valid" + note)
        return 0

    if args.input == "-":
        text = sys.stdin.read()
    else:
        with open(args.input, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()

    diagnosis = diagnose(text, top=args.top)
    if args.json:
        payload = {
            "verdict": diagnosis["verdict"],
            "candidates": [
                {
                    "id": r["rule"]["id"],
                    "title": r["rule"].get("title"),
                    "score": r["score"],
                    "confidence": r["rule"].get("confidence"),
                    "evidence": r["evidence"][:6],
                }
                for r in diagnosis["candidates"]
            ],
        }
        # Read-only enrichment, same rule as the text renderer: keys are
        # added ONLY when the enrichment actually fires.
        derived = rete.derived_evidence(diagnosis)
        if derived is not None:
            payload["derived_evidence"] = derived
        conflicts = argumentation.conflict_resolution(diagnosis)
        if conflicts is not None:
            payload["conflict_resolution"] = conflicts
        print(json.dumps(payload, indent=2))
    else:
        print(render_report(diagnosis))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
