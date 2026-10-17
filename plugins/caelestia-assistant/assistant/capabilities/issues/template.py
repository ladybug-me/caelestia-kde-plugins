"""Issue-template loading and draft rendering for Layer 4 (drafting only).

Guarantees:
- Pure stdlib, offline, deterministic: identical inputs always render an
  identical draft. No parsing library is used — the two GitHub issue
  templates (.github/ISSUE_TEMPLATE/1-issue.yml, 2-feature_request.yml) are
  extracted with a hand-rolled, line-based reader that only understands the
  shapes actually present in those two files (top-level name/description;
  a body list of markdown / checkboxes / textarea / input items with label,
  description, placeholder, value, options and validations.required).
- Clear failure mode: if the template files are missing, unreadable, or the
  line-based reader finds nothing it recognises, load_template falls back to
  hardcoded mirrors of the two templates (FALLBACK_* below) so a draft can
  always be rendered.
- This module RENDERS TEXT and nothing else: no network, no subprocess, no
  filesystem writes (writing is the CLI's job, behind --confirm).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import BUG_TEMPLATE_FILE, FEATURE_TEMPLATE_FILE

# Exact banner that starts every rendered draft and every written file.
BANNER = (
    "DRAFT — NOT SUBMITTED TO ANYWHERE. Review, then copy-paste into GitHub's "
    "new-issue form yourself. This file was created by the caelestia assistant (offline)."
)

# Advisory dedup threshold: if the top retrieval hit scores above this, the
# draft header gains a "possible duplicate" warning line. Advisory only —
# drafting is never blocked.
DUPLICATE_WARN_SCORE = 8.0

# Commands the assistant would otherwise have to run to fill in logs. They are
# NEVER executed here; they are rendered as inert SUGGESTED_NOT_EXECUTED lines.
SUGGESTED_COMMANDS = [
    "journalctl --user -u caelestia-shell --no-pager -n 100",
    "plasmashell --version",
    "quickshell --version",
]

VERSION_FILE_SUGGESTION = "cat ~/.config/caelestia/version"

TEMPLATE_FILES = {"bug": BUG_TEMPLATE_FILE, "feature": FEATURE_TEMPLATE_FILE}

GITHUB_ISSUES_URL = "https://github.com/ladybug-me/caelestia-kde/issues"

# ---------------------------------------------------------------------------
# Hardcoded mirrors (fallback when the yml files cannot be parsed)
# ---------------------------------------------------------------------------

FALLBACK_BUG_META = {"name": "Bug report", "description": "Something's not working right"}
FALLBACK_BUG_FIELDS: List[Dict[str, Any]] = [
    {
        "kind": "markdown",
        "value": (
            "Thanks for taking the time to report a bug!\n\nA quick search of [existing issues]"
            f"({GITHUB_ISSUES_URL}?q=is%3Aissue) helps avoid duplicates - if you find one,"
            " give it a thumbs up instead."
        ),
    },
    {
        "kind": "checkboxes",
        "label": "Issue type",
        "description": "Pick one label that best describes this issue.",
        "options": [
            "Bug - something is broken or not working correctly",
            "Help wanted - I need assistance with configuration, setup, or usage",
        ],
        "required": True,
    },
    {
        "kind": "textarea",
        "label": "What happened?",
        "description": "What went wrong? What did you expect instead?",
        "placeholder": (
            'The launcher froze when I typed "terminal" and pressed Enter. '
            "I expected it to open Konsole."
        ),
        "required": True,
    },
    {
        "kind": "textarea",
        "label": "Steps to reproduce",
        "description": "How can someone else trigger this?",
        "placeholder": '1. Open the launcher (Super+Space)\n2. Type "terminal"\n3. Press Enter',
        "required": False,
    },
    {
        "kind": "input",
        "label": "Caelestia version",
        "description": (
            "What version of Caelestia are you running? Check with `cat ~/.config/caelestia/version`"
            " or look for the version string in the shell."
        ),
        "placeholder": "e.g. v2.2.2",
        "required": True,
    },
    {
        "kind": "input",
        "label": "Distro",
        "description": "Which Linux distribution are you on?",
        "placeholder": "e.g. CachyOS, Arch, Fedora",
        "required": True,
    },
    {
        "kind": "textarea",
        "label": "Logs or screenshots",
        "description": (
            "If relevant, paste logs or drag in screenshots. The `caelestia-shell-ipc log`"
            " command shows live shell logs."
        ),
        "placeholder": "Drag and drop images here, or paste log output.",
        "required": False,
    },
]

FALLBACK_FEATURE_META = {"name": "Feature request", "description": "Got an idea? We'd love to hear it"}
FALLBACK_FEATURE_FIELDS: List[Dict[str, Any]] = [
    {
        "kind": "markdown",
        "value": (
            "Thanks for the idea!\n\nA quick search of [existing requests]("
            f"{GITHUB_ISSUES_URL}?q=is%3Aissue+label%3Afeature) avoids duplicates"
            " - if you find one, give it a thumbs up instead."
        ),
    },
    {
        "kind": "textarea",
        "label": "What's the idea?",
        "description": (
            "Describe what you'd like to see. Mockups, screenshots, or examples from other projects all help."
        ),
        "placeholder": "A widget that shows the current weather in the top bar.",
        "required": True,
    },
    {
        "kind": "textarea",
        "label": "Why would it be useful?",
        "description": "What problem does it solve, or what workflow does it improve?",
        "placeholder": "I check the weather frequently and hate pulling out my phone.",
        "required": False,
    },
]

# ---------------------------------------------------------------------------
# Line-based template reader (minimal YAML awareness, shape-specific)
# ---------------------------------------------------------------------------

_BLOCK_SCALAR_KEYS = ("value", "placeholder", "description")
_ATTRIBUTE_KEYS = ("label", "description", "placeholder", "value")
_QUOTED_RE = re.compile(r"^\"(.*)\"$|^'(.*)'$")


def _strip_quotes(value: str) -> str:
    match = _QUOTED_RE.match(value)
    if match:
        return match.group(1) or match.group(2) or ""
    return value


def parse_issue_template(path: Path) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """Extract (meta, fields) from one GitHub issue-template yml file.

    Understands ONLY the shapes present in 1-issue.yml / 2-feature_request.yml.
    Returns empty items when nothing recognizable is found — the caller is
    responsible for falling back to the hardcoded mirrors.
    """
    template_path = Path(path)
    if not template_path.is_file():
        return {}, []
    meta: Dict[str, str] = {}
    fields: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    pending_key: Optional[str] = None
    pending_indent = 0
    in_body = False
    in_options = False
    in_validations = False
    for raw in template_path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            if pending_key is not None and current is not None:
                current[pending_key].append("")  # keep blank lines inside block scalars
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if pending_key is not None and current is not None and indent >= pending_indent:
            current[pending_key].append(stripped)
            continue
        pending_key = None
        if indent == 0:
            key, _, val = stripped.partition(":")
            if key == "body" and not val.strip():
                in_body = True
            elif key in ("name", "description") and val.strip():
                meta[key] = _strip_quotes(val.strip())
            continue
        if in_body and stripped.startswith("- type:"):
            current = {"kind": stripped.split(":", 1)[1].strip()}
            fields.append(current)
            in_options = False
            in_validations = False
            continue
        if current is None:
            continue
        if in_options and stripped.startswith("- label:"):
            current.setdefault("options", []).append(_strip_quotes(stripped.split(":", 1)[1].strip()))
            continue
        if stripped == "options:":
            in_options, in_validations = True, False
            continue
        if stripped == "validations:":
            in_validations, in_options = True, False
            continue
        if stripped.startswith("required:") and in_validations:
            current["required"] = stripped.split(":", 1)[1].strip() == "true"
            continue
        for key in _ATTRIBUTE_KEYS:
            if stripped.startswith(key + ":"):
                val = stripped[len(key) + 1:].strip()
                if val in ("|", "|-", "|+"):
                    pending_key = key
                    pending_indent = indent + 1
                    current[key] = []
                elif val:
                    current[key] = _strip_quotes(val)
                break
    # Join collected block-scalar lines back into strings.
    for field in fields:
        for key in _BLOCK_SCALAR_KEYS:
            if isinstance(field.get(key), list):
                field[key] = "\n".join(field[key]).strip("\n")
    return meta, fields


def load_template(issue_type: str) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """Load one template; falls back to hardcoded mirrors on any failure.

    The lookup of parse_issue_template is a module-global lookup on purpose so
    tests can monkeypatch it to return nothing and exercise the fallback.
    """
    if issue_type not in TEMPLATE_FILES:
        raise ValueError(f"unknown issue type {issue_type!r} (expected bug|feature)")
    try:
        meta, fields = parse_issue_template(TEMPLATE_FILES[issue_type])
    except Exception:  # noqa: BLE001 — any reader failure must fall back, never crash drafting
        meta, fields = {}, []
    if not fields:
        if issue_type == "bug":
            return dict(FALLBACK_BUG_META), [dict(f) for f in FALLBACK_BUG_FIELDS]
        return dict(FALLBACK_FEATURE_META), [dict(f) for f in FALLBACK_FEATURE_FIELDS]
    if "name" not in meta:
        meta["name"] = "Bug report" if issue_type == "bug" else "Feature request"
    return meta, fields


def top_hit_is_strong(similar: List[Dict[str, Any]]) -> bool:
    """True when the top retrieval hit is strong enough to warn about."""
    return bool(similar) and float(similar[0].get("score", 0.0)) > DUPLICATE_WARN_SCORE


def _field_suffix(field: Dict[str, Any]) -> str:
    required = "required" if field.get("required") else "optional"
    return f"[{field['kind']}] ({required})"


def _fill_in_hint(field: Dict[str, Any]) -> str:
    placeholder = str(field.get("placeholder", "")).replace("\n", " / ")
    return f"(fill in yourself — template example: {placeholder})" if placeholder else "(fill in yourself)"


def _env_or_suggestion(field: Dict[str, Any], env_block: Dict[str, str]) -> List[str]:
    """Fill an input/textarea field from the env block or a suggested command."""
    label = field.get("label", "")
    if label == "Caelestia version":
        value = env_block.get("caelestia_version", "").strip()
        if value:
            return [value, "(read locally from ~/.config/caelestia/version by the assistant)"]
        return [
            f"SUGGESTED_NOT_EXECUTED: {VERSION_FILE_SUGGESTION}",
            "(run this yourself and paste the output)",
        ]
    if label == "Distro":
        value = env_block.get("distro", "").strip()
        return [value] if value else ["(fill in — run: cat /etc/os-release)"]
    if label == "Logs or screenshots":
        out = ["Commands only you can run (run these yourself and paste output here if relevant):"]
        out.extend(f"  SUGGESTED_NOT_EXECUTED: {cmd}" for cmd in SUGGESTED_COMMANDS)
        return out
    return [_fill_in_hint(field)]


def body_label(fields: List[Dict[str, Any]]) -> str:
    """Label of the template's first textarea — the field receiving the body."""
    for field in fields:
        if field.get("kind") == "textarea":
            return str(field.get("label", ""))
    return ""


def _render_similar(similar: List[Dict[str, Any]]) -> List[str]:
    lines = [
        "=" * 78,
        "Before filing — similar existing material (advisory only, from the local offline index)",
        "=" * 78,
        "The template asks you to search existing issues first and give an existing one a",
        "thumbs up instead of filing a duplicate. Local index hits for this problem:",
    ]
    if not similar:
        lines.append("  (no similar material found in the local index — searching manually is still worth it)")
        return lines
    for hit in similar:
        lines.append(
            f"  - {hit.get('doc_id', '?')}  (score {hit.get('score', 0)})  {hit.get('title', '')}"
        )
        if hit.get("source"):
            lines.append(f"      source: {hit['source']}")
        if hit.get("snippet"):
            lines.append(f"      snippet: {hit['snippet']}")
    lines.append("(dedup is advisory only — the assistant never blocks filing and never files anything)")
    return lines


def _render_environment(env_block: Dict[str, str]) -> List[str]:
    lines = [
        "=" * 78,
        "Environment (assembled locally by the assistant — pure file reads, nothing sent anywhere)",
        "=" * 78,
    ]
    labels = (
        ("Distro", "distro"),
        ("Caelestia version", "caelestia_version"),
        ("Branch", "branch"),
        ("Commit", "commit"),
    )
    for label, key in labels:
        value = env_block.get(key, "").strip()
        lines.append(f"- {label}: {value if value else '(unknown — fill in yourself)'}")
    lines.append("Commands only you can run (run these yourself and paste output):")
    lines.extend(f"  SUGGESTED_NOT_EXECUTED: {cmd}" for cmd in SUGGESTED_COMMANDS)
    return lines


def render_draft(
    issue_type: str,
    title: str,
    body: str,
    env_block: Dict[str, str],
    similar: List[Dict[str, Any]],
    commit_info: Optional[str],
    generated_at: str,
) -> str:
    """Render a paste-ready issue draft as markdown text. Writes nothing.

    Mirrors the real repo template (sections, labels, descriptions, options,
    required-ness) field by field so each rendered section can be copied into
    the matching box of GitHub's new-issue form.
    """
    meta, fields = load_template(issue_type)
    template_file = TEMPLATE_FILES[issue_type].name
    lines: List[str] = [BANNER, f"Generated: {generated_at} (local time)"]
    if commit_info:
        lines.append(f"Source repo commit: {commit_info}")
    desc = meta.get("description", "")
    lines.append(f"Template: {meta.get('name', issue_type)} ({desc}) — mirrors .github/ISSUE_TEMPLATE/{template_file}")
    lines.append(f"Proposed title: {title}")
    if top_hit_is_strong(similar):
        lines.append(f"possible duplicate — check {similar[0].get('doc_id', '?')} first (see similar section below)")
    lines.append("")

    for field in fields:
        if field.get("kind") == "markdown":
            lines.append("Template notice (from the template itself):")
            for md_line in str(field.get("value", "")).splitlines():
                lines.append(f"  {md_line}" if md_line else "")
            lines.append("")
            continue
        lines.append("-" * 78)
        number = sum(1 for ln in lines if ln.startswith("### "))
        lines.append(f"### {number + 1}. {field.get('label', '?')}  {_field_suffix(field)}")
        if field.get("description"):
            lines.append(f"    <!-- {field['description']} -->")
        if field.get("kind") == "checkboxes":
            for option in field.get("options", []):
                tick = "[x]" if issue_type == "bug" and option.startswith("Bug") else "[ ]"
                lines.append(f"  {tick} {option}")
        elif field.get("kind") == "textarea" and field.get("label") == body_label(fields):
            lines.append(str(body).strip() or "(empty — describe the problem)")
        else:
            lines.extend(_env_or_suggestion(field, env_block))
        lines.append("")

    lines.extend(_render_similar(similar))
    lines.append("")
    lines.extend(_render_environment(env_block))
    lines.append("")
    lines.append("-" * 78)
    lines.append(
        "Reminder: this draft exists only as a local file on this machine. Nothing was posted, sent,"
    )
    lines.append(
        "or transmitted anywhere — the assistant has no way to file issues. You file it yourself"
    )
    lines.append("by copy-pasting the fields above into GitHub's new-issue form.")
    return "\n".join(lines).rstrip() + "\n"
