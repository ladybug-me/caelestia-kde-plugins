"""shellkb.cligrammar — CLI syntax induction from the upstream FILES (B6).

The shell employee's first job: KNOW the shell's own command line.
Not by executing anything — by reading it. The upstream checkout ships
bash CLIs under ``src/bin/`` whose syntax is written out in prose (the
``usage()`` heredocs and echo'd option tables), repeated in one-line
``printf``/``echo`` usage strings, and enforced again in the ``case``
option-dispatch loops. All of it lives in plain text files. This
module parses those files and produces a TYPED option grammar: which
subcommands exist, which options take values (as far as the file text
itself evidences), which words are positional.

Why parse instead of running ``caelestia --help``?  Executing the
binary would (a) violate the no-subprocess rule everywhere except the
pkgprobe quarantine, (b) make the knowledge depend on an installed
shell, and (c) hide the SOURCE from review. Parsing the files keeps
every claim citable: an option is in the grammar because line N of
file F says so. The mined artifact (``shell_cli_grammar.json``) is
committed — rebuilding it is a reviewed act like re-pinning the
registry — and the runtime only ever READS it, so the knowledge works
on machines with no upstream checkout at all.

Evidence classes, in review order:
  - ``usage-table row``      — a heredoc/echo option table line
  - ``usage line (brackets)``— an inline ``[-s|--sound]`` group
  - ``case branch``          — the dispatch loop (what is ENFORCED);
    ``takes_value`` is typed from ``shift 2`` / ``$2`` reads
  - ``commands table``       — the bin's Commands section
  - ``dispatch``             — a ``sub) cmd_sub`` routing row

Verdicts: every example command line this module (or cmdparse, its
consumer) produces is labeled SUGGESTED_NOT_EXECUTED. The shellkb
knows; it does not do.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["GRAMMAR_PATH", "mine", "load_grammar", "render_lines",
           "example_lines"]

GRAMMAR_PATH = Path(__file__).resolve().parent / "shell_cli_grammar.json"

# The upstream files this miner reads (relative to the checkout root).
# Adding a file here is a reviewable diff; the artifact records what
# was actually read.
BIN_FILES: Tuple[str, ...] = (
    "src/bin/caelestia",
    "src/bin/caelestia-shell-ipc",
    "src/bin/caelestia-color",
    "src/bin/caelestia-screenshot",
    "src/bin/caelestia-record",
    "src/bin/caelestia-update",
    "src/bin/caelestia-check-updates",
)

# --- file-text patterns (all against bash SOURCE, never execution) -------

# "  -d, --daemon    start the shell detached" / "  -r, --region  text"
_OPT_ROW_RE = re.compile(r"^\s{1,6}(-\w)?,?\s?(--[\w-]+|-\w)\s{2,}(.+)$")
# "  shell [OPTS] [MESSAGE...]    start the shell, or message the running one"
_SUB_ROW_RE = re.compile(r"^\s{1,6}([a-z][\w-]*(?:\s+\S+)*?)\s{2,}(\S.*?)$")
_USAGE_LINE_RE = re.compile(r"Usage:\s*(.+)")
# option branches inside a case dispatch:
#   "-d|--daemon)"  /  "--flat)"  /  "-f|--file) file=\"${2:-}\"; ... shift 2 ;;"
_CASE_BRANCH_RE = re.compile(
    r"^\s*((?:-\w|--[\w-]+)(?:\|(?:-\w|--[\w-]+))*)\)")
# bare-word case branches ("start)", "pause)", "wallpaper|scheme)",
# "version|--version|-v)") — subcommand evidence. The FIRST
# alternative must be a bare word, so option branches (-d|--daemon)
# never match.
_WORD_BRANCH_RE = re.compile(
    r"^\s+([a-z][\w-]*(?:\|-?[\w-]+)*)\)", re.M)
_VALUE_EVIDENCE_RE = re.compile(r"shift\s+2|\$2|\$\{2")
# top-level command dispatch: "shell) cmd_shell \"$@\" ;;"
_DISPATCH_RE = re.compile(r"^\s*([a-z][\w-]*)\)\s+cmd_\w+", re.M)
# a quoted-string statement line (echo / printf, anywhere on the line)
_QUOTED_LINE_RE = re.compile(r"(?:echo|printf)\s+(\"|')")
# inline bracket groups in a one-line usage string: [-s|--sound] [--gif]
_BRACKET_OPT_RE = re.compile(r"\[(?:(-\w)\|)?(--[\w-]+)\]")


def _quoted_literal(line: str) -> Optional[str]:
    """The string literal of one echo/printf statement (even when the
    statement sits mid-line after a case branch: `-h|--help) printf
    'Usage: ...'`). Returns the '...' or \"...\" contents."""
    m = _QUOTED_LINE_RE.search(line)
    if not m:
        return None
    quote = m.group(1)
    start = m.end(1)
    end = line.find(quote, start)
    if end < 0:
        return None
    return line[start:end].replace("\\n", "\n").replace("\\t", "\t")


def _string_blocks(text: str) -> List[str]:
    """Runs of consecutive echo/printf lines joined into one block —
    the echo'd usage tables are heredocs written vertically."""
    blocks: List[str] = []
    current: List[str] = []
    for line in text.splitlines():
        literal = _quoted_literal(line)
        if literal is not None:
            current.append(literal)
        elif current:
            blocks.append("\n".join(current))
            current = []
    if current:
        blocks.append("\n".join(current))
    return [b for b in blocks if "Usage:" in b or _OPT_ROW_RE.search(b)]


def _case_options(body: str) -> Dict[str, Dict[str, Any]]:
    """Options evidenced by a case-dispatch loop: long/short names plus
    takes_value, from the loop's own value-handling (``shift 2`` or a
    ``$2`` read). This is the ENFORCEMENT site — what the script
    actually accepts, as opposed to what the help text advertises."""
    found: Dict[str, Dict[str, Any]] = {}
    current: Optional[Dict[str, Any]] = None
    for line in body.splitlines():
        branch = _CASE_BRANCH_RE.match(line)
        if branch:
            names = branch.group(1).split("|")
            short = next((n for n in names if len(n) == 2), None)
            long = next((n for n in names if n.startswith("--")), None)
            # value evidence can sit ON the branch line:
            #   -f|--file) file="${2:-}"; ... shift 2 ;;
            rest = line[branch.end():]
            current = {"short": short, "long": long, "desc": "",
                       "takes_value": bool(_VALUE_EVIDENCE_RE.search(rest)),
                       "evidence": "case branch", "section": ""}
            key = long or short
            found[key] = current
            continue
        if current is not None and _VALUE_EVIDENCE_RE.search(line):
            current["takes_value"] = True
        if line.strip().startswith(";;"):
            current = None
    return found


def _word_branches(text: str) -> set:
    """Bare-word case branches across the file (each alternative in
    ``wallpaper|scheme)`` counts) — the cross-check a Commands-table
    row must pass before it counts as a subcommand."""
    out = set()
    for m in _WORD_BRANCH_RE.finditer(text):
        for word in m.group(1).split("|"):
            if re.fullmatch(r"[a-z][\w-]*", word):
                out.add(word)
    return out


def _dispatch_subs(text: str) -> set:
    """Subcommand words routed by an exact ``sub) cmd_sub`` row."""
    out = set()
    for m in _DISPATCH_RE.finditer(text):
        sub = m.group(1)
        fn = re.split(r"[\s\"']", m.group(0).split("cmd_", 1)[1])[0]
        if fn == sub:
            out.add(sub)
    return out


def _function_bodies(text: str) -> List[Tuple[str, str]]:
    """(function name, body) for the bash functions whose closing brace
    sits at column 0 — the layout every upstream bin script uses."""
    return re.findall(r"(\w+)\s*\(\)\s*\{(.*?)\n\}", text, re.S)


def _usage_line(text: str) -> Optional[str]:
    m = _USAGE_LINE_RE.search(text)
    return m.group(1).strip() if m else None


def _sub_path(bin_name: str, usage: str) -> Optional[str]:
    """The subcommand path a one-line usage names: leading lowercase
    words after the binary ('scheme list'), None when the usage is the
    bin's own ('[start]', '<command>')."""
    parts = usage.split()
    if not parts or parts[0] != bin_name:
        return None
    path: List[str] = []
    for tok in parts[1:]:
        if re.fullmatch(r"[a-z][\w-]*", tok):
            path.append(tok)
        else:
            break
    return " ".join(path) if path else None


def _inline_options(usage: str) -> List[Dict[str, Any]]:
    """[-s|--sound] / [--gif] groups inside a one-line usage string."""
    return [{"short": short or None, "long": long, "desc": "",
             "takes_value": False,
             "evidence": "usage line (brackets)", "section": ""}
            for short, long in _BRACKET_OPT_RE.findall(usage)]


class _Grammar:
    """One bin's grammar under construction."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.bin_entry: Dict[str, Any] = {"options": []}
        self.subs: Dict[str, Dict[str, Any]] = {}

    def entry(self, sub: Optional[str]) -> Dict[str, Any]:
        if sub is None:
            return self.bin_entry
        return self.subs.setdefault(
            sub, {"synopsis": f"{self.name} {sub}", "desc": "",
                  "options": []})

    def merge_opt(self, sub: Optional[str], opt: Dict[str, Any]) -> None:
        pool = self.entry(sub)["options"]
        merged = next((o for o in pool
                       if o["long"] == opt["long"]
                       and o["short"] == opt["short"]), None)
        if merged is None:
            pool.append(dict(opt))
        else:
            if opt.get("desc"):
                merged["desc"] = opt["desc"]
            if opt.get("takes_value") is not None:
                merged["takes_value"] = opt["takes_value"]
            merged["evidence"] = opt.get("evidence", merged["evidence"])

    def to_dict(self) -> Dict[str, Any]:
        out = dict(self.bin_entry)
        subs = {}
        for sub in sorted(self.subs):
            entry = dict(self.subs[sub])
            entry.setdefault("options", [])
            subs[sub] = entry
        out["subcommands"] = subs
        return out


def parse_bin(path: Path) -> Dict[str, Any]:
    """One upstream CLI file -> its typed grammar. Pure text parsing:
    usage prose gives the ADVERTISED surface; case-dispatch bodies give
    the ENFORCED surface (typed takes_value evidence); the two are
    merged per subcommand."""
    text = path.read_text(encoding="utf-8", errors="replace")
    name = path.name
    g = _Grammar(name)
    words = _word_branches(text)
    routed = words | _dispatch_subs(text)

    # Every usage surface, as (block, base-sub): heredocs attach to
    # their function's subcommand; echo/printf blocks to the sub their
    # Usage line names.
    blocks: List[Tuple[str, Optional[str]]] = []
    for fn_name, body in _function_bodies(text):
        heredoc = re.search(r"cat\s+<<'EOF'\n(.*?)\nEOF", body, re.S)
        if heredoc:
            blocks.append((heredoc.group(1), _fn_sub(fn_name, g)))
    for block in _string_blocks(text):
        usage = _usage_line(block)
        base = _sub_path(name, usage) if usage else None
        if base is not None and base not in routed and \
                base.split()[0] not in routed:
            base = None  # an unroutable word after the bin: bin-level
        blocks.append((block, base))

    def section_sub(section: str) -> Optional[str]:
        head = section.split()[0].lower().strip(":") if section else ""
        return head if head and head in routed else None

    def _is_header(line: str) -> bool:
        """A usage-block section header ('Shell options', 'Scheme',
        'Commands'): uppercase-initial, at most a few words, and not
        itself a row — so wrapped description lines ('Defaults to the
        wallpaper already set', '...on their own.') never clobber the
        section."""
        s = line.strip()
        return bool(s) and s[0].isupper() and len(s) <= 40 \
            and len(s.split()) <= 3 and not _OPT_ROW_RE.match(line)

    def absorb_subrows(block: str, base: Optional[str]) -> None:
        """Pass A: usage synopsis + Commands rows (so every sub exists
        before any option is placed)."""
        usage = _usage_line(block)
        if usage:
            g.entry(base).setdefault("synopsis", usage)
            for opt in _inline_options(usage):
                g.merge_opt(base, opt)
        section = ""
        for line in block.splitlines():
            if not line.strip() or _USAGE_LINE_RE.search(line):
                continue  # blank or the Usage line itself: not a section
            if _OPT_ROW_RE.match(line):
                continue
            row = _SUB_ROW_RE.match(line)
            if row:
                head = row.group(1).split()[0]
                if head in routed:
                    target = head
                    sec = section_sub(section)
                    if sec and sec != head:
                        target = f"{sec} {head}"
                    entry = g.entry(target)
                    entry["desc"] = row.group(2).strip()
                    entry["evidence"] = "commands table"
                    if len(row.group(1).split()) > 1:
                        entry.setdefault("synopsis", f"{name} {target}")
            elif _is_header(line):
                section = line.strip()

    def absorb_optrows(block: str, base: Optional[str]) -> None:
        """Pass B: option rows, resolved to (a) the last subcommand row
        of the current section (the 'set [OPTS]' rows own their indented
        options), (b) the section's sub ('Shell options' -> shell), or
        (c) the block's base sub. A subcommand row only captures
        following options inside a NAMED section or a sub-block — a
        positional row like record's 'start (default)' under a bare
        'Options' header must not steal the bin's own options."""
        section = ""
        last_sub: Optional[str] = None
        for line in block.splitlines():
            if line.strip() and _USAGE_LINE_RE.search(line):
                continue
            row = _OPT_ROW_RE.match(line)
            if row:
                opt = {"short": row.group(1), "long": row.group(2),
                       "desc": row.group(3).strip(), "takes_value": None,
                       "evidence": "usage-table row"}
                long = opt["long"]
                if long and not long.startswith("--"):
                    opt["short"], opt["long"] = \
                        (long if long.startswith("-") else None), None
                target = last_sub
                if target is None:
                    target = section_sub(section) or base
                g.merge_opt(target, opt)
            elif _SUB_ROW_RE.match(line):
                head = line.strip().split()[0]
                sec = section_sub(section)
                if head in routed and (sec or base):
                    last_sub = (f"{sec} {head}" if sec and sec != head
                                else head)
            elif _is_header(line):
                section = line.strip()
                last_sub = None  # a new section: sub rows start over

    for block, base in blocks:
        absorb_subrows(block, base)
    for block, base in blocks:
        absorb_optrows(block, base)

    # case dispatch bodies: takes_value typed from shift 2 / $2
    for fn_name, body in _function_bodies(text):
        if not re.search(r"case\s+", body):
            continue
        opts = _case_options(body)
        if not opts:
            continue
        sub = _fn_sub(fn_name, g)
        for opt in opts.values():
            g.merge_opt(sub, opt)

    out = g.to_dict()
    out["file"] = f"src/bin/{name}"
    return out


def _fn_sub(fn_name: str, g: "_Grammar") -> Optional[str]:
    """cmd_shell -> shell; shell_usage -> shell; cmd_scheme_list ->
    'scheme list' when that sub exists; usage -> None (the bin)."""
    if fn_name.startswith("cmd_"):
        raw = fn_name[4:]
        spaced = raw.replace("_", " ")
        return spaced if spaced in g.subs else raw
    if fn_name.endswith("_usage"):
        prefix = fn_name[:-len("_usage")]
        return None if prefix == "usage" else prefix
    return None


def find_upstream() -> Optional[Path]:
    """The pinned checkout to mine from, by the doctor's own convention
    (walk-up, then siblings), plus the CAELESTIA_KDE_ROOT override for
    non-standard layouts. None when absent — mining is a maintainer
    act, runtime reads the committed artifact."""
    import os
    env = os.environ.get("CAELESTIA_KDE_ROOT")
    if env and (Path(env) / "src" / "bin").is_dir():
        return Path(env)
    repo = Path(__file__).resolve().parents[2]
    for cand in [repo, *repo.parents]:
        if (cand / "shell" / "plugin" / "src" / "Caelestia"
                / "Config").is_dir():
            return cand
    for sibling in sorted(repo.parent.iterdir()):
        if (sibling / "shell" / "plugin" / "src" / "Caelestia"
                / "Config").is_dir() and (sibling / "src" / "bin").is_dir():
            return sibling
    return None


def mine(root: Optional[Path] = None) -> Dict[str, Any]:
    """Mine the upstream bin scripts into the typed grammar artifact.
    Deterministic: files in the fixed BIN_FILES order, keys sorted on
    write."""
    root = root or find_upstream()
    if root is None or not (root / "src" / "bin").is_dir():
        return {"error": "no caelestia-kde checkout found beside this repo "
                         "(mining needs the source files; runtime reads the "
                         "committed artifact)"}
    bins: Dict[str, Any] = {}
    for rel in BIN_FILES:
        path = root / rel
        if path.is_file():
            bins[path.name] = parse_bin(path)
    if not bins:
        return {"error": f"no src/bin CLI files found under {root}"}
    commit = _head_commit(root)
    return {
        "generator": "assistant.capabilities.shellkb.cligrammar.mine",
        "note": "induced from upstream FILE TEXT (usage heredocs, echo "
                "usage tables, one-line usage strings, case option "
                "dispatch). No binary was executed. Every example is "
                "SUGGESTED_NOT_EXECUTED.",
        "upstream": {"checkout": root.name, "commit": commit},
        "bins": bins,
    }


def _head_commit(root: Path) -> str:
    head = root / ".git" / "HEAD"
    if not head.is_file():
        return ""
    ref = head.read_text().strip()
    if ref.startswith("ref:"):
        ref_path = root / ".git" / ref.split(" ", 1)[-1]
        return ref_path.read_text().strip()[:12] if ref_path.is_file() else ""
    return ref[:12]


def load_grammar() -> Dict[str, Any]:
    """The committed artifact (runtime view). Never mines live."""
    try:
        return json.loads(GRAMMAR_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"generator": "missing", "bins": {}}


def _opt_label(opt: Dict[str, Any]) -> str:
    parts = [p for p in (opt.get("short"), opt.get("long")) if p]
    return "/".join(parts) if parts else "?"


def example_lines(bins: Optional[Dict[str, Any]] = None) -> List[str]:
    """One SUGGESTED_NOT_EXECUTED example per subcommand, generated
    from the grammar itself (synopsis when present, else the option
    list) — the honest floor of 'suggest': valid syntax, nothing run."""
    bins = bins if bins is not None else load_grammar().get("bins", {})
    lines: List[str] = []
    for bin_name in sorted(bins):
        bin_g = bins[bin_name]
        for sub_name in sorted(bin_g.get("subcommands", {})):
            sub = bin_g["subcommands"][sub_name]
            synopsis = sub.get("synopsis") or f"{bin_name} {sub_name}"
            lines.append(f"SUGGESTED_NOT_EXECUTED: {synopsis}")
        for opt in bin_g.get("options", [])[:4]:
            lines.append(f"SUGGESTED_NOT_EXECUTED: {bin_name} "
                         f"{_opt_label(opt)}")
    return lines


def render_lines() -> List[str]:
    data = load_grammar()
    bins = data.get("bins", {})
    lines = [
        "shell CLI grammar — induced from upstream file text "
        f"({data.get('upstream', {}).get('checkout', '?')}@"
        f"{data.get('upstream', {}).get('commit', '?') or 'pinned'}); "
        f"{len(bins)} binaries. Nothing is ever executed.",
    ]
    for bin_name in sorted(bins):
        bin_g = bins[bin_name]
        lines.append(f"  {bin_name}  ({bin_g.get('file', '?')})")
        synopsis = bin_g.get("synopsis")
        if synopsis:
            lines.append(f"    synopsis: {synopsis}")
        for opt in bin_g.get("options", []):
            tv = "value" if opt.get("takes_value") else "flag"
            lines.append(f"    {_opt_label(opt):24s} [{tv}] "
                         f"{opt.get('desc', '')[:70]}")
        for sub_name in sorted(bin_g.get("subcommands", {})):
            sub = bin_g["subcommands"][sub_name]
            lines.append(f"    {sub_name}: "
                         f"{(sub.get('synopsis') or sub_name)[:78]}")
            for opt in sub.get("options", []):
                tv = "value" if opt.get("takes_value") else "flag"
                lines.append(f"      {_opt_label(opt):22s} [{tv}] "
                             f"{opt.get('desc', '')[:60]}")
    return lines


def main(argv=None) -> int:  # pragma: no cover - thin CLI
    """`python3 -m assistant.capabilities.shellkb.cligrammar [--mine]`: print the
    committed grammar; --mine rebuilds it from an upstream checkout (a
    reviewed, committed change, like re-pinning the registry)."""
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--mine":
        data = mine()
        if "error" in data:
            print(data["error"])
            return 1
        GRAMMAR_PATH.write_text(
            json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False)
            + "\n", encoding="utf-8")
        n_subs = sum(len(b.get("subcommands", {}))
                     for b in data["bins"].values())
        print(f"wrote {GRAMMAR_PATH.name}: {len(data['bins'])} binaries, "
              f"{n_subs} subcommands")
        return 0
    for line in render_lines():
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
