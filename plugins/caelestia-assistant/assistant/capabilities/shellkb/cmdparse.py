"""shellkb.cmdparse — one shell command line, explained token by token (B7).

The shell employee's second job: LOOK at a command the user is about
to run (or asks about) and say what each piece is and does — without
running anything. The parser is a PEG (packrat: ordered choices,
memoized rule applications over a token stream), so the grammar is a
set of small honest rules rather than a pile of nested ifs:

    segment    <- assignment* command
    command    <- BIN subcommand? (option / value / positional / redirect)*
    option     <- ('-' char | '--' word) value?

Each token gets a ROLE and, where the B6 grammar knows the binary, a
READING (which subcommand, which option, expected value). Tokens with
glob characters get a read-only ``os.scandir`` PREVIEW of what they
would match (capped, sorted, never touched). Tokens matching a
destructive pattern get a reason. The overall verdict is always
SUGGESTED_NOT_EXECUTED — the assistant explains; it does not do.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import cligrammar

__all__ = ["explain_line", "explain_segment", "tokenize", "render_explanation"]

# --- tokenizer (quote-aware; shlex is deliberately not imported —
# the allow-list stays untouched and the rule is 20 honest lines) -------

_TOKEN_RE = re.compile(r"""
    (?P<redirect>>>{1,2}|>&\d|>&|>\||>|<)   # redirection operators
  | (?P<control>&&|\|\||\||;|&)             # segment separators
  | (?P<ws>\s+)
  | (?P<word>"(?:[^"\\]|\\.)*"              # double-quoted
           | '[^']*'                        # single-quoted
           | (?:[^\\'"\s><|;&]|\\.)+)       # bare word (backslash escapes)
""", re.X)


def tokenize(line: str) -> List[Dict[str, Any]]:
    """One command line -> ordered tokens {kind, text}. Quotes are kept
    on the token so the explanation can say 'single-quoted' (no glob,
    no expansion) — the reading IS the point."""
    tokens: List[Dict[str, Any]] = []
    pos = 0
    while pos < len(line):
        m = _TOKEN_RE.match(line, pos)
        if not m:
            tokens.append({"kind": "raw", "text": line[pos:]})
            break
        kind = m.lastgroup
        text = m.group()
        if kind == "ws":
            pos = m.end()
            continue
        if kind == "word":
            quoted = text.startswith("'") or text.startswith('"')
            tokens.append({"kind": "quoted" if quoted else "word",
                           "text": text})
        else:
            tokens.append({"kind": kind, "text": text})
        pos = m.end()
    return tokens


# --- destructive patterns (evidence-cited, sorted for determinism) ------

# (regex over the joined segment words, severity, reason). Adding a
# pattern is a reviewable diff — the list is pinned by tests.
DESTRUCTIVE: Tuple[Tuple[str, str, str], ...] = (
    (r"^rm\b", "destructive", "removes files; with -r it takes trees"),
    (r"^rmdir\b", "destructive", "removes directories"),
    (r"^dd\b", "destructive", "raw device write; a typo overwrites data"),
    (r"^mkfs", "destructive", "formats a filesystem"),
    (r"^shutdown\b|^poweroff\b|^halt\b|^reboot\b", "destructive",
     "stops or reboots the machine"),
    (r"^chmod\s+(-[a-zA-Z]*R[a-zA-Z]*\s+)?", "destructive",
     "recursive permission change"),
    (r"^chown\s+-[a-zA-Z]*R", "destructive", "recursive ownership change"),
    (r"^git\s+push\b.*(--force|-f)\b", "destructive",
     "force-push rewrites remote history"),
    (r"^kwriteconfig\d?\b", "config-write",
     "writes KDE config files directly (use the settings layer's "
     "gated apply instead)"),
    (r"^caelestia-update\b|^caelestia\s+update\b", "system",
     "fetches and re-installs the shell from the network"),
    (r"^caelestia-check-updates\b", "system",
     "network query driven by a timer"),
    (r"^curl\b|^wget\b.*\|\s*(sh|bash|zsh)\b", "destructive",
     "piping a network download into a shell executes unreviewed code"),
)

# shell-specific destructive options, checked against the B6 grammar
SHELLKB_DESTRUCTIVE_OPTS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "caelestia": (("--kill", "stops the running shell"),
                  ("-k", "stops the running shell")),
    "caelestia-shell-ipc": (("quit", "gracefully stops the shell"),
                            ("restart", "stops and restarts the shell")),
    "caelestia-record": (("--stop", "ends the active recording"),
                         ("stop", "ends the active recording")),
}

_GLOB_RE = re.compile(r"[*?\[]")
_GLOB_PREVIEW_CAP = 20


def _glob_preview(token: str) -> Optional[Dict[str, Any]]:
    """What would this glob token match? Read-only scandir, capped,
    sorted. ``None`` when the token has no glob characters. QUOTED
    tokens never expand — that is the reading, reported not applied."""
    if token[:1] in ("'", '"'):
        if _GLOB_RE.search(token):
            return {"note": "quoted: glob characters are literal, "
                            "nothing expands"}
        return None
    unquoted = token
    if not _GLOB_RE.search(unquoted):
        return None
    p = Path(unquoted)
    directory = p.parent if str(p.parent) else Path(".")
    try:
        names = sorted(e.name for e in os.scandir(
            str(directory) if str(directory) != "" else "."))
    except OSError as exc:
        return {"dir": str(directory),
                "error": f"{type(exc).__name__}: cannot preview "
                         f"(read-only scandir failed)"}
    import fnmatch
    matches = [n for n in names if fnmatch.fnmatchcase(n, p.name)]
    capped = matches[:_GLOB_PREVIEW_CAP]
    return {"dir": str(directory) if str(directory) != "." else ".",
            "pattern": p.name,
            "matches": capped,
            "n_matches": len(matches),
            "capped": len(matches) > _GLOB_PREVIEW_CAP}


class _Packrat:
    """Minimal packrat PEG over the token stream: memoized positions per
    rule, ordered choices, bounded repetition. The grammar above maps
    1:1 onto these methods."""

    def __init__(self, tokens: List[Dict[str, Any]]) -> None:
        self.tokens = tokens
        self.pos = 0
        self.memo: Dict[Tuple[str, int], Any] = {}

    def peek(self) -> Optional[Dict[str, Any]]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def rule(self, name: str, fn):
        key = (name, self.pos)
        if key in self.memo:
            result, saved = self.memo[key]
            self.pos = saved
            return result
        result = fn()
        self.memo[key] = (result, self.pos)
        return result

    def many(self, fn) -> List[Any]:
        out = []
        while True:
            saved = self.pos
            item = fn()
            if item is None:
                self.pos = saved
                return out
            out.append(item)

    def _word(self) -> Optional[Dict[str, Any]]:
        tok = self.peek()
        if tok and tok["kind"] in ("word", "quoted"):
            self.pos += 1
            return tok
        return None


def explain_line(line: str) -> Dict[str, Any]:
    """Split on control operators, explain each segment, keep the
    SUGGESTED_NOT_EXECUTED verdict global."""
    tokens = tokenize(line)
    segments: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for tok in tokens:
        if tok["kind"] == "control" and tok["text"] in ("&&", "||", ";", "|"):
            if current:
                segments.append(current)
            current = []
            # the separator itself is explained too
            segments.append([tok])
        else:
            current.append(tok)
    if current:
        segments.append(current)
    explained = [explain_segment(seg) for seg in segments]
    return {
        "verdict": "SUGGESTED_NOT_EXECUTED",
        "note": "the shellkb explains command lines; it never runs them",
        "line": line,
        "n_segments": len([e for e in explained
                           if e.get("segment_kind") != "separator"]),
        "segments": explained,
    }


def explain_segment(tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One segment's tokens, each with a role and reading."""
    if len(tokens) == 1 and tokens[0]["kind"] == "control":
        text = tokens[0]["text"]
        reading = {"&&": "run the next segment only if this one succeeded",
                   "||": "run the next segment only if this one failed",
                   "|": "pipe this segment's output into the next",
                   ";": "run the next segment regardless"}[text]
        return {"segment_kind": "separator", "text": text, "reading": reading}

    grammar = cligrammar.load_grammar().get("bins", {})
    parser = _Packrat(tokens)
    words: List[Dict[str, Any]] = []

    def take_word() -> Optional[Dict[str, Any]]:
        tok = parser.rule("word", parser._word)
        return tok

    assignments: List[Dict[str, Any]] = []
    while True:
        tok = take_word()
        if tok and re.fullmatch(r"[A-Za-z_]\w*(=(\"[^\"]*\"|'[^']*'|\S*)?)?",
                                tok["text"]) and "=" in tok["text"]:
            var, _, val = tok["text"].partition("=")
            assignments.append({"token": tok["text"], "var": var,
                                "value": val.strip("'\"") or "(empty)",
                                "role": "env-assignment",
                                "reading": f"sets {var} for this command only"})
            continue
        if tok:
            parser.pos -= 1  # not an assignment: rewind, it's the bin
        break

    bin_tok = take_word()
    bin_name = bin_tok["text"].strip("'\"") if bin_tok else None
    bin_grammar = grammar.get(bin_name or "", {})
    if bin_tok:
        if bin_grammar:
            reading = f"the shell CLI '{bin_name}' (known to the grammar)"
        else:
            reading = "not in the induced grammar: reading it generically"
        words.append({"token": bin_tok["text"], "role": "binary",
                      "reading": reading})

    # the remaining tokens: walk with the grammar when we have one.
    # The SUBCOMMAND is identified first (the first bare token that the
    # bin routes), because option tables are per-subcommand — a flat
    # merge would let 'scheme get's flag -f shadow wallpaper's value
    # -f. Options are then read against THAT subcommand's table plus
    # the bin-level table.
    subs = bin_grammar.get("subcommands", {})
    sub_taken: Optional[str] = None
    for tok in tokens:
        bare = tok["text"].strip("'\"")
        if tok is bin_tok:
            continue
        if not bare.startswith("-") and bare in subs:
            sub_taken = bare
            break
    sub_opts: Dict[str, Dict[str, Any]] = {}
    if sub_taken:
        for opt in subs[sub_taken].get("options", []):
            if opt.get("long"):
                sub_opts.setdefault(opt["long"], opt)
            if opt.get("short"):
                sub_opts.setdefault(opt["short"], opt)
    known_opts: Dict[str, Dict[str, Any]] = dict(sub_opts)
    for opt in bin_grammar.get("options", []):
        if opt.get("long"):
            known_opts.setdefault(opt["long"], opt)
        if opt.get("short"):
            known_opts.setdefault(opt["short"], opt)

    positional_idx = 0
    pending_opt: Optional[Dict[str, Any]] = None
    while True:
        tok = parser.rule("arg", take_word)
        if tok is None:
            break
        text = tok["text"]
        bare = text.strip("'\"")
        role, reading = "positional", "argument passed through to the command"
        if pending_opt is not None:
            role = "option-value"
            reading = (f"value for {pending_opt['token']}"
                       + (f" ({pending_opt['expected']})" if
                          pending_opt.get("expected") else ""))
            pending_opt["value_tokens"] = pending_opt.get("value_tokens", 0) + 1
            pending_opt = None
        elif text.startswith("--") and len(bare) > 2 and bare != "--":
            opt = known_opts.get(bare)
            role = "option"
            if opt is not None:
                takes = opt.get("takes_value")
                reading = (f"known option of {bin_name} {sub_taken or ''}: "
                           f"{opt.get('desc') or 'documented in its usage'}"
                           ).strip()
                if takes:
                    pending_opt = {"token": text, "expected": "a value"}
                    reading += "; expects a value"
            else:
                reading = ("NOT in the induced grammar for this command — "
                           "check the spelling against "
                           "`shellkb grammar`")
        elif re.fullmatch(r"-\w", bare) or (bare.startswith("-") and
                                            len(bare) == 2):
            opt = known_opts.get(bare)
            if opt is not None:
                takes = opt.get("takes_value")
                reading = (f"known option of {bin_name} {sub_taken or ''}"
                           ).strip()
                if opt.get("long"):
                    reading += f" (long form: {opt['long']})"
                if takes:
                    pending_opt = {"token": text, "expected": "a value"}
                    reading += "; expects a value"
                role = "option"
            elif not bin_grammar:
                role = "option(s)"
                reading = ("option cluster of an unknown binary "
                           "(generic reading; not in the grammar)")
            else:
                role = "option"
                reading = ("NOT in the induced grammar for this command — "
                           "check the spelling against `shellkb grammar`")
        elif bare.startswith("-") and len(bare) > 2 and \
                not bare.startswith("--") and not bin_grammar:
            role = "option(s)"
            reading = (f"combined short options of an unknown binary "
                       f"({bare[1]!r} first; generic reading)")
        elif sub_taken is not None and bare == sub_taken:
            role = "subcommand"
            reading = (f"'{bin_name} {bare}': "
                       f"{subs[bare].get('desc') or 'a known subcommand'}"
                       ).strip()
        else:
            positional_idx += 1
            reading = f"positional #{positional_idx} for {bin_name or 'bin'}"

        entry = {"token": text, "role": role, "reading": reading}
        if tok["kind"] == "quoted":
            entry["quoted"] = True
        glob = _glob_preview(text)
        if glob is not None:
            entry["glob_preview"] = glob
        words.append(entry)

    # destructive classification (words joined, redirect-aware)
    joined = " ".join(t.strip("'\"") for t in
                      [w["token"] for w in words])
    flags: List[Dict[str, str]] = []
    for pattern, severity, reason in DESTRUCTIVE:
        if re.search(pattern, joined):
            flags.append({"severity": severity, "reason": reason})
    for marker, reason in SHELLKB_DESTRUCTIVE_OPTS.get(bin_name or "", ()):
        if any(w["token"] == marker or w["token"].strip("'\"") == marker
               for w in words):
            flags.append({"severity": "destructive", "reason": reason})
    for w in words:
        if w["role"] == "option" and w["token"] in ("-k", "--kill") \
                and bin_name == "caelestia" and sub_taken == "shell":
            pass  # already flagged above via the table
    redirects = [t for t in tokens if t["kind"].startswith("redirect")]
    for tok in redirects:
        flags.append({"severity": "config-write" if tok["text"] == ">" else
                      "write",
                      "reason": f"redirect {tok['text']} overwrites its "
                                f"target file"})
    return {"segment_kind": "command",
            "assignments": assignments,
            "tokens": words,
            "redirects": [t["text"] for t in redirects],
            "flags": flags,
            "verdict": ("SUGGESTED_NOT_EXECUTED"
                        + (" — flagged: " + "; ".join(
                            f["reason"] for f in flags) if flags else ""))}


def render_explanation(data: Dict[str, Any]) -> List[str]:
    lines = [f"{data['verdict']}: {data['line']}",
             f"  ({data['note']})"]
    n = 0
    for seg in data["segments"]:
        if seg.get("segment_kind") == "separator":
            lines.append(f"  separator {seg['text']!r}: {seg['reading']}")
            continue
        n += 1
        lines.append(f"  segment {n}:")
        for a in seg.get("assignments", []):
            lines.append(f"    {a['token']:24s} [{a['role']}] {a['reading']}")
        for w in seg.get("tokens", []):
            suffix = " <glob>" if w.get("glob_preview") else ""
            lines.append(f"    {w['token']:24s} [{w['role']}{suffix}] "
                         f"{w['reading']}")
            g = w.get("glob_preview")
            if g:
                if "error" in g:
                    lines.append(f"      glob: {g['error']}")
                elif "note" in g:
                    lines.append(f"      glob: {g['note']}")
                else:
                    shown = ", ".join(g["matches"]) or "(no matches)"
                    cap = f" (first {len(g['matches'])})" if g["capped"] \
                        else ""
                    lines.append(f"      glob would match {g['n_matches']} "
                                 f"entries in {g['dir']}{cap}: {shown}")
        for tok in seg.get("redirects", []):
            lines.append(f"    redirect {tok}: overwrites its target file")
        for f in seg.get("flags", []):
            lines.append(f"    FLAG [{f['severity']}]: {f['reason']}")
    return lines


def main(argv=None) -> int:  # pragma: no cover - thin CLI
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: explain -- <command line>")
        return 2
    line = " ".join(argv)
    data = explain_line(line)
    for out in render_explanation(data):
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
