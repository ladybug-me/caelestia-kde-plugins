"""Layer 3 (OPTIONAL, OFF by default): retrieval-grounded generative suggestions.

Hard guarantees of this layer:

- OFF by default: suggest(enable=False) — the default, and the CLI without
  --generative — never opens a connection of any kind. The result states that
  the generative layer is disabled and carries the unchanged Layer 2 hits.
- Opt-in only: with enable=True the layer still requires a loopback Ollama
  server (generative.client validates the host BEFORE connecting; the only
  allowed hosts are localhost / 127.0.0.1 / [::1], port configurable). If the
  server is missing or errors: one attempt, ~10s timeout, graceful
  "unavailable" reason, retrieval hits unchanged. Never a crash, never a
  retry storm.
- Grounded only: the prompt is built ONLY from Layer 2 retrieval hits
  (assistant.capabilities.retrieval.search) plus the user's problem text, under a system
  preamble that forbids inventing commands and forbids claiming execution.
  With zero retrieval hits the model is not contacted at all.
- Suggestion-only: every line of the model output that looks like a command
  is prefixed with SUGGESTED_NOT_EXECUTED: and labeled with a risk tier via
  assistant.capabilities.diagnostics.risk.classify. Suggestions classified DESTRUCTIVE are
  withheld entirely and replaced by a blocked note pointing at
  docs/TROUBLESHOOTING.md. Nothing is ever executed, auto-run, or trained on.
- No side effects: no files written, no subprocess/socket imports; the only
  network surface is the single loopback http.client POST in
  assistant/generative/client.py.

Usage:
    python3 -m assistant.capabilities.generative "problem text"                # disabled notice + hits
    python3 -m assistant.capabilities.generative "problem text" --generative   # opt-in (needs local Ollama)
    python3 -m assistant.capabilities.generative "problem text" --generative -k 6 --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Dict, List, Optional

try:  # normal path: `assistant` is a package under the repo root
    from assistant.capabilities.diagnostics import risk
    from assistant.capabilities.retrieval import search
except ImportError:  # discovery from inside assistant/ makes it top-level
    from assistant.capabilities.diagnostics import risk  # type: ignore
    from assistant.capabilities.retrieval import search  # type: ignore
from . import client

GENERATIVE_BANNER = (
    "Generative suggestion (local model, may be wrong). "
    "Review before acting; nothing was or will be executed."
)

NOT_EXECUTED_PREFIX = "SUGGESTED_NOT_EXECUTED:"

BLOCKED_STEP_NOTE = (
    "[blocked by safety review] destructive command suggested by the model was withheld; "
    "consult docs/TROUBLESHOOTING.md instead."
)

# The system preamble embedded at the top of every prompt. It is the grounding
# contract: context-only, no invented commands, no execution claims.
SAFETY_PREAMBLE = (
    "You are the caelestia-kde on-device assistant. Suggest troubleshooting steps grounded "
    "ONLY in the provided context excerpts; do not invent commands that are not present in "
    "or clearly derivable from that context; never claim to execute anything; output plain "
    "numbered steps and nothing else."
)

DISABLED_REASON = (
    "generative layer is disabled (OFF by default); pass --generative to opt in. "
    "Layer 2 retrieval hits are returned unchanged."
)

UNGROUNDED_REASON = (
    "generative layer skipped: Layer 2 retrieval found no grounding excerpts, and the "
    "generative layer refuses to prompt the model without grounded context. "
    "Nothing was sent to any server."
)

PROBLEM_MAX_CHARS = 4000

# Shell-ish verbs that make a line "command-looking". Over-labeling is the
# safe direction: a prose line that merely mentions a verb gets a harmless
# SUGGESTED_NOT_EXECUTED label, while an unlabeled command line would be a
# safety failure.
_COMMAND_VERBS = (
    "systemctl", "journalctl", "loginctl", "systemd-run", "pacman-key", "pacman", "paru",
    "yay", "pamac", "apt-get", "apt", "dnf", "flatpak", "rm", "rmdir", "mv", "cp", "mkdir",
    "chmod", "chown", "ln", "dd", "mkfs", "shred", "wipefs", "mount", "umount", "sudo",
    "pkexec", "doas", "kill", "killall", "pkill", "pgrep", "kwriteconfig5", "kwriteconfig6",
    "kreadconfig5", "kreadconfig6", "kbuildsycoca5", "kbuildsycoca6", "kquitapp5", "kvantummanager",
    "plasma-apply-colorscheme", "plasma-apply-lookandfeel", "plasma-apply-desktoptheme",
    "plasma-apply-cursortheme", "lookandfeeltool", "kscreen-doctor", "qdbus", "qdbus6",
    "dbus-send", "gdbus", "sed", "bash", "sh", "zsh", "source", "export", "env", "echo",
    "cat", "tee", "truncate", "curl", "wget", "git", "dmesg", "modprobe", "lsmod", "dkms",
    "mkinitcpio", "dracut", "grub-mkconfig", "update-grub", "nmcli", "bluetoothctl", "rfkill",
    "wpctl", "pactl", "hyprctl", "xdg-mime", "xdg-open", "python3",
)

_VERB_RE = re.compile(
    rf"\b({'|'.join(sorted(_COMMAND_VERBS, key=len, reverse=True))})\b", re.IGNORECASE
)

# Shell prompt shapes: "$ ", "% ", "[user@host dir]$ ", "user@host:~$ ".
# A bare "# " is NOT treated as a prompt so markdown headings stay unlabeled
# (a root-prompt line like "# rm -rf /" is still caught by the verb scan).
_PROMPT_RE = re.compile(r"^(?:\[[^\]]+\]\s*[$#]\s*|[\w.@-]+@[\w.-]+:\S*\s*[$#]\s*|[$%>]\s+)")

_BULLET_RE = re.compile(r"^(?:[-*•]|\d+[.)])\s+")

# Terminal escape sequences are stripped from model output before anything
# else: they can both hide a verb from the classifiers ("rm \x1b[31m-rf\x1b[0m /")
# and inject arbitrary escape codes into the user's terminal. Same pattern
# the Layer 1 engine uses to normalize pasted input.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _ends_with_continuation(line: str) -> bool:
    """True when a line ends with an odd number of backslashes (shell
    line-continuation). An even count is escaped literal backslashes."""
    stripped = line.rstrip()
    trailing = len(stripped) - len(stripped.rstrip("\\"))
    return trailing % 2 == 1


def _logical_lines(text: str) -> List[str]:
    r"""Merge backslash-continued physical lines into single logical lines.

    Without this, model output like "rm \<newline>-rf /" is classified per
    physical line: "rm \" alone is non-recursive (STATE_CHANGING) and the
    "-rf /" half passes through unlabeled — so copying the labeled step
    plus its visible continuation executes the withheld destructive command.
    Merging first makes the whole shape classify (and be withheld) as one.
    """
    physical = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    logical: List[str] = []
    for line in physical:
        if logical and _ends_with_continuation(logical[-1]):
            head = logical[-1].rstrip()[:-1].rstrip()
            logical[-1] = f"{head} {line.strip()}".strip() if head else line.strip()
        else:
            logical.append(line)
    return logical


def build_rag_prompt(problem_text: str, retrieval_hits: List[Dict[str, Any]]) -> str:
    """One prompt string: safety preamble + problem text + retrieval excerpts.

    The excerpts are the ONLY permitted grounding; if there are none the
    prompt says so and the caller (suggest) refuses to contact the model.
    """
    problem = problem_text.strip()
    if len(problem) > PROBLEM_MAX_CHARS:
        problem = problem[:PROBLEM_MAX_CHARS] + " …[truncated]"

    lines: List[str] = [SAFETY_PREAMBLE, ""]
    lines.append("If the context excerpts below do not cover the problem, say so plainly instead of guessing.")
    lines.append("")
    lines.append("Problem reported by the user:")
    lines.append(problem if problem else "(empty problem text)")
    lines.append("")
    lines.append("Context excerpts — the ONLY permitted grounding (from this repo's docs and resolved issues):")
    if retrieval_hits:
        for i, hit in enumerate(retrieval_hits, start=1):
            lines.append(f"[{i}] {hit.get('doc_id', '?')} — {hit.get('title', '')} (source: {hit.get('source', '')})")
            snippet = (hit.get("snippet") or "").strip()
            if snippet:
                lines.append(f"    {snippet}")
    else:
        lines.append("(none)")
    lines.append("")
    lines.append("Output plain numbered steps grounded in those excerpts only. Never claim to execute anything.")
    return "\n".join(lines)


def _command_core(line: str) -> str:
    """Strip one leading bullet and/or one shell prompt from a line."""
    core = _BULLET_RE.sub("", line.strip(), count=1)
    core = _PROMPT_RE.sub("", core, count=1)
    return core.strip()


def _looks_like_command(line: str) -> bool:
    """True for shell prompts and for lines containing a known shell verb."""
    stripped = line.strip()
    if not stripped or stripped.startswith("```"):
        return False  # code-fence markers are formatting, not commands
    if _PROMPT_RE.match(stripped):
        return True
    return _VERB_RE.search(stripped) is not None


def sanitize_suggestion(text: str) -> str:
    """Post-process raw model output into a labeled, suggestion-only text.

    - Banner is prefixed (nothing was or will be executed).
    - Every command-looking line gets "SUGGESTED_NOT_EXECUTED:" plus a risk
      tier line from assistant.capabilities.diagnostics.risk.classify.
    - A DESTRUCTIVE suggestion is withheld: the whole step is replaced by
      BLOCKED_STEP_NOTE.
    """
    out: List[str] = []
    for raw_line in _logical_lines(_ANSI_RE.sub("", text)):
        if not _looks_like_command(raw_line):
            out.append(raw_line.rstrip())
            continue
        core = _command_core(raw_line)
        tier = risk.classify(core) if core else "STATE_CHANGING"
        if tier == "DESTRUCTIVE":
            out.append(BLOCKED_STEP_NOTE)
        else:
            shown = core if core else raw_line.strip()
            out.append(f"{NOT_EXECUTED_PREFIX} {shown}")
            out.append(f"[risk: {tier}]")
    body = "\n".join(out).strip("\n")
    return f"{GENERATIVE_BANNER}\n\n{body}" if body else GENERATIVE_BANNER


def suggest(
    problem_text: str,
    k: int = 4,
    enable: bool = False,
    model: Optional[str] = None,
    conn_factory: Optional[client.ConnectionFactory] = None,
    url: Optional[str] = None,
) -> Dict[str, Any]:
    """Layer 3 entry point; see the module docstring for the guarantees.

    Returns {"enabled", "available", "suggestion", "retrieval_hits", "reason"}.
    retrieval_hits are present and unchanged on EVERY path, including the
    disabled and unavailable ones. conn_factory/url exist for injection in
    tests; production callers pass neither.
    """
    hits = search.search(problem_text, k=k)
    result: Dict[str, Any] = {
        "enabled": bool(enable),
        "available": False,
        "suggestion": None,
        "retrieval_hits": hits,
        "reason": "",
    }

    if not enable:
        result["reason"] = DISABLED_REASON
        return result

    if not hits:
        result["reason"] = UNGROUNDED_REASON
        return result

    try:
        host, port = client.parse_loopback(client.resolve_url(url))
    except client.LoopbackViolation as exc:
        result["reason"] = f"generative layer unavailable: {exc}"
        return result

    if not client.is_available(url=url, conn_factory=conn_factory):
        result["reason"] = (
            f"generative layer unavailable: no Ollama server answered on {host}:{port} "
            f"(single attempt, {client.DEFAULT_TIMEOUT_S:g}s timeout); "
            "Layer 2 retrieval hits are returned unchanged."
        )
        return result

    prompt = build_rag_prompt(problem_text, hits)
    try:
        raw = client.generate(prompt, model=model, url=url, conn_factory=conn_factory)
    except (client.GenerativeError, client.LoopbackViolation, OSError) as exc:
        result["reason"] = f"generative layer unavailable: {type(exc).__name__}: {exc}"
        return result

    if not raw.strip():
        result["reason"] = "generative layer unavailable: the local model returned empty output."
        return result

    result["available"] = True
    result["suggestion"] = sanitize_suggestion(raw)
    result["reason"] = (
        f"generative layer active (local model {client.resolve_model(model)} on {host}:{port}); "
        "suggestion only — nothing was or will be executed."
    )
    return result


def render_result(result: Dict[str, Any]) -> str:
    """Human-readable rendering: state notice, reason, hits, then suggestion."""
    if result["suggestion"]:
        heading = "caelestia assistant — Layer 3 generative suggestion (local model)"
    elif result["enabled"]:
        heading = "caelestia assistant — Layer 3 generative (UNAVAILABLE)"
    else:
        heading = "caelestia assistant — Layer 3 generative (DISABLED)"
    lines: List[str] = [heading, result["reason"], "", "Layer 2 retrieval hits:"]
    hits = result["retrieval_hits"]
    if not hits:
        lines.append("  (no hits)")
    for i, hit in enumerate(hits, start=1):
        lines.append(f"  [{i}] {hit['doc_id']} — {hit['title']}  (score {hit['score']})")
        lines.append(f"      source: {hit['source']}")
        lines.append(f"      snippet: {hit['snippet']}")
    if result["suggestion"]:
        lines.append("")
        lines.append(result["suggestion"])
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI: python3 -m assistant.capabilities.generative "problem" [--generative] [-k N]."""
    parser = argparse.ArgumentParser(
        prog="python3 -m assistant.capabilities.generative",
        description=(
            "caelestia assistant Layer 3 (OPTIONAL, OFF by default): retrieval-grounded "
            "suggestions from a local Ollama model. Suggestion-only: nothing is ever "
            "executed; needs --generative plus a loopback Ollama server."
        ),
    )
    parser.add_argument("text", nargs="?", default="-", help="problem text or pasted log; '-' reads stdin")
    parser.add_argument(
        "--generative",
        action="store_true",
        help="opt in to the generative layer (requires a local Ollama server on loopback)",
    )
    parser.add_argument("-k", type=int, default=4, help="max retrieval hits used as grounding (default 4)")
    parser.add_argument(
        "--model", default=None, help="Ollama model tag (default: CAELESTIA_ASSISTANT_OLLAMA_MODEL or llama3)"
    )
    parser.add_argument("--json", action="store_true", help="emit the raw result dict as JSON")
    args = parser.parse_args(argv)

    text = sys.stdin.read() if args.text == "-" else args.text
    result = suggest(text, k=args.k, enable=args.generative, model=args.model)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(render_result(result))
    return 0
