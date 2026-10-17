"""CLI for Layer 4 issue drafting. Drafts ONLY — nothing is ever filed.

Usage:
    python3 -m assistant.capabilities.issues.cli draft --title "..." [--type bug|feature]
            [--from-file PATH | stdin] [--out PATH] [--confirm] [--list-similar]
    python3 -m assistant.capabilities.issues.cli list-similar --title "..." [--from-file PATH | stdin] [-k N]

Gate mechanics:
- WITHOUT --confirm the command is PREVIEW-ONLY: the draft is printed to
  stdout and no file is written anywhere.
- WITH --confirm the draft is written to a LOCAL file only: --out PATH, or
  $XDG_DATA_HOME/caelestia/assistant/drafts/issue-draft-YYYYmmdd-HHMMSS.md
  (default ~/.local/share/...). Nothing is posted, sent, or transmitted.
- --list-similar just prints the retrieval hits for the described problem:
  no file write, no --confirm needed.

Guarantees: pure stdlib, no network code, no subprocess — the only commands
that appear in output are inert SUGGESTED_NOT_EXECUTED strings the user runs
themselves. Environment facts are assembled from pure file reads only
(/etc/os-release, .git/HEAD + .git/refs/heads/<branch>, ~/.config/caelestia/version).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import template
from assistant.capabilities.retrieval import search as retrieval_search

DEFAULT_TOP_K = 3
LIST_SIMILAR_TOP_K = 5

DRAFTS_SUBDIR = Path("caelestia") / "assistant" / "drafts"

REPO_ROOT = Path(__file__).resolve().parents[2]

EPILOG = (
    "This tool drafts issues only: it never posts, submits, opens, or transmits anything. "
    "You copy-paste the draft into GitHub's new-issue form yourself."
)


def default_drafts_dir() -> Path:
    """$XDG_DATA_HOME/caelestia/assistant/drafts (default ~/.local/share/...)."""
    data_home = os.environ.get("XDG_DATA_HOME")
    base = Path(data_home) if data_home else Path.home() / ".local" / "share"
    return base / DRAFTS_SUBDIR


def default_draft_path(now: Optional[datetime] = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return default_drafts_dir() / f"issue-draft-{stamp}.md"


def read_os_release() -> str:
    """PRETTY_NAME (or NAME) from /etc/os-release — pure file read, '' if absent."""
    for key in ("PRETTY_NAME", "NAME"):
        value = _os_release_field(key)
        if value:
            return value
    return ""


def _os_release_field(key: str) -> str:
    try:
        text = Path("/etc/os-release").read_text(encoding="utf-8")
    except OSError:
        return ""
    for line in text.splitlines():
        if line.startswith(key + "="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def read_commit_info(repo_root: Path = REPO_ROOT) -> str:
    """Branch + commit from .git files (pure reads); '' when unavailable.

    Never shells out: reads .git/HEAD, then .git/refs/heads/<branch> directly.
    The refs/heads/ prefix is stripped for display (feat/assistant, not refs/...).
    """
    git_dir = repo_root / ".git"
    head_path = git_dir / "HEAD"
    try:
        head = head_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if head.startswith("ref:"):
        branch = head[len("ref:"):].strip()
        branch = branch.split("refs/heads/", 1)[-1]  # display form: feat/assistant
        ref_path = git_dir / "refs" / "heads" / branch
        try:
            commit = ref_path.read_text(encoding="utf-8").strip()
        except OSError:
            return branch
        return f"{commit} (branch {branch})"
    return head  # detached HEAD: the sha itself


def read_shell_version() -> str:
    """Best-effort pure read of ~/.config/caelestia/version; '' if absent."""
    try:
        return Path.home().joinpath(".config", "caelestia", "version").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def gather_description(args: argparse.Namespace, stdin_text: Optional[str]) -> str:
    """Description from --from-file, else stdin text; '' when neither given."""
    if getattr(args, "from_file", None):
        return Path(args.from_file).read_text(encoding="utf-8")
    return (stdin_text or "").strip()


def fetch_similar(query: str, k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """Top-k retrieval hits for the described problem; [] if retrieval fails."""
    if not query.strip():
        return []
    try:
        return retrieval_search.search(query, k=k)
    except Exception:  # noqa: BLE001 — retrieval must never break drafting
        return []


def build_env_block(commit_info: str = "") -> Dict[str, str]:
    """Environment facts assembled from pure local file reads only."""
    commit_info = commit_info or read_commit_info()
    branch = ""
    if "(branch " in commit_info:
        branch = commit_info.split("(branch ", 1)[1].rstrip(")")
    return {
        "distro": read_os_release(),
        "caelestia_version": read_shell_version(),
        "branch": branch,
        "commit": commit_info.split(" ", 1)[0] if commit_info else "",
    }


def _print_similar(hits: List[Dict[str, Any]]) -> None:
    print("Similar existing material (local offline index — advisory only; nothing was sent anywhere):")
    if not hits:
        print("  (no hits)")
        return
    for i, hit in enumerate(hits, 1):
        print(f"  {i}. {hit.get('doc_id', '?')}  score {hit.get('score', 0)}  {hit.get('title', '')}")
        if hit.get("source"):
            print(f"     source: {hit['source']}")
        if hit.get("snippet"):
            print(f"     snippet: {hit['snippet']}")
    print("The template asks you to search existing issues first;")
    print("give an existing one a thumbs up instead of filing a duplicate.")


def run_list_similar(args: argparse.Namespace, stdin_text: Optional[str]) -> int:
    query = " ".join(part for part in [args.title or "", gather_description(args, stdin_text)] if part)
    hits = fetch_similar(query, k=getattr(args, "k", LIST_SIMILAR_TOP_K))
    _print_similar(hits)
    return 0


def run_draft(args: argparse.Namespace, stdin_text: Optional[str]) -> int:
    description = gather_description(args, stdin_text)
    query_text = " ".join(part for part in [args.title, description] if part)

    if args.list_similar:
        _print_similar(fetch_similar(query_text, k=DEFAULT_TOP_K))
        print("(--list-similar mode: no draft composed, no file written)")
        return 0

    if not description:
        print("error: empty description — pass --from-file PATH or pipe the description on stdin", file=sys.stderr)
        return 2
    similar = fetch_similar(query_text, k=DEFAULT_TOP_K)

    commit_info = read_commit_info()
    env_block = build_env_block(commit_info)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    draft_text = template.render_draft(
        issue_type=args.type,
        title=args.title,
        body=description,
        env_block=env_block,
        similar=similar,
        commit_info=commit_info or None,
        generated_at=generated_at,
    )

    if not args.confirm:
        print(draft_text, end="")
        print("(preview only — no file written; pass --confirm to write the draft to a local file)")
        return 0

    out_path = Path(args.out) if args.out else default_draft_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(draft_text, encoding="utf-8")
    print(f"Draft written: {out_path}")
    print("Reminder: nothing was submitted anywhere — the assistant cannot file issues.")
    print("Review the draft, then copy-paste it into GitHub's new-issue form yourself.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m assistant.capabilities.issues.cli",
        description="Issue-DRAFTING helper: composes paste-ready issue text. Drafts only, local files only.",
        epilog=EPILOG,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    draft = sub.add_parser("draft", help="compose an issue draft (preview by default)")
    draft.add_argument("--title", required=True, help="proposed issue title")
    draft.add_argument("--type", choices=["bug", "feature"], default="bug", help="issue template (default: bug)")
    draft.add_argument("--from-file", default=None, help="read the problem description from PATH (else stdin)")
    draft.add_argument("--out", default=None, help="write the draft to PATH (default under XDG_DATA_HOME)")
    draft.add_argument("--confirm", action="store_true", help="write the draft file (without it: preview only)")
    draft.add_argument("--list-similar", action="store_true", help="print retrieval hits only; no draft, no file")
    draft.set_defaults(func=run_draft)

    similar = sub.add_parser("list-similar", help="print retrieval hits for a described problem (no file write)")
    similar.add_argument("--title", default="", help="short description of the problem")
    similar.add_argument("--from-file", default=None, help="read a longer description from PATH (else stdin)")
    similar.add_argument("-k", type=int, default=LIST_SIMILAR_TOP_K, help="number of hits (default 5)")
    similar.set_defaults(func=run_list_similar)
    return parser


def main(argv: Optional[List[str]] = None, stdin_text: Optional[str] = None) -> int:
    """Entry point. stdin_text overrides sys.stdin (tests pass text directly)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    stdin_payload = stdin_text
    if stdin_payload is None and not getattr(args, "from_file", None) and not sys.stdin.isatty():
        stdin_payload = sys.stdin.read()
    return int(args.func(args, stdin_payload))


if __name__ == "__main__":
    raise SystemExit(main())
