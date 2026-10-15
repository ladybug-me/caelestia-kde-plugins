#!/usr/bin/env python3
"""Parse captured GitHub issue HTML into retrieval corpus docs (ISS-*.md).

Usage:
    python3 assistant/retrieval/tools/parse_issues.py <html_dir> [--out DIR] [--all]

The HTML files are offline captures of github.com issue pages. Each embeds its
data in a ``<script type="application/json" data-target="react-app.embeddedData">``
blob. This script walks that JSON for:
- the issue object  (dict with title + number + body)
- comment objects   (``__typename == "IssueComment"`` with body/author/createdAt)

and writes one markdown corpus doc per issue with a strict ``key: value``
header block (id, title, source, tags[, synonyms]), the original problem
statement (condensed), a Resolution section assembled ONLY from what the
captured comments actually say (verbatim quotes, closing PR when the HTML
shows one), and a full comment log. If resolution detail is thin, the doc says
so instead of inventing anything.

Stdlib only; reads local files; writes only to the output directory.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "assistant" / "retrieval" / "corpus"
DEFAULT_HTML_DIR = REPO_ROOT.parent / "research" / "corpus-html"

# Issues this corpus is built from. 6..763 are CLOSED resolutions; 120 and 802
# are OPEN feature threads that define what the on-device assistant may claim
# to do (they get a Scope section instead of a Resolution).
DEFAULT_ISSUES = (6, 14, 333, 402, 409, 416, 418, 461, 528, 616, 641, 763, 120, 802)
OPEN_ISSUES = (120, 802)

REPO_URL = "https://github.com/ladybug-me/caelestia-kde"

# Curated retrieval tags per issue (honest keywords from the thread content).
TAGS: Dict[int, List[str]] = {
    6: ["bluetooth", "bluez", "bluez-utils", "pairing", "bluetooth.service", "turns on then off", "cachyos"],
    14: ["kde-material-you-colors", "material you dark", "color scheme", "setup.sh", "systemd user service", "logout"],
    333: ["shortcuts not working", "kglobalaccel", "unrecognized pragma", "defaultenv", "qs_no_reload_popup",
          "dev branch update"],
    402: ["vesktop", "discord", "screenshare", "screen share", "camera", "freeze", "zkde_screencast_unstable_v1",
          "pipewire", "nvidia"],
    409: ["video wallpaper", "live wallpaper", "pause video wallpaper", "pipewire", "audio endpoint",
          "wallpaper flashing"],
    416: ["launcher wrong monitor", "alt-tab", "window switcher", "fullscreen video", "multi-monitor", "input region"],
    418: ["quickshell has crashed", "kwinactivewindowbridge", "window switcher empty", "workspace icons",
          "shell restart", "clean reset"],
    461: ["random apps crash", "chromium", "spotify", "discord", "steam", "memory usage", "live previews",
          "systemd-cgtop"],
    528: ["quickshell does not open", "error after update", "reinstall quickshell-git", "paru", "pacman", "garuda"],
    616: ["gpu widget", "gpu usage averaged", "gpu_busy_percent", "/sys/class/drm", "multi-gpu", "hybrid graphics",
          "temperature"],
    641: ["gamescope", "game half screen", "fullscreen", "dodge windows", "meta+shift+e", "taskbar",
          "games launch options"],
    763: ["color variant reverts", "tonal spot", "after reboot", "after login", "automatic color scheme",
          "color engine"],
    120: ["ai assistant", "natural language", "settings assistant", "scope", "ollama", "caelestia-cli tools"],
    802: ["ai assistant", "narrow ai", "scope", "troubleshooting", "cleaning", "issue reporting", "training data"],
}

# Curated synonym lines (alternate phrasings a user might type).
SYNONYMS: Dict[int, List[str]] = {
    6: ["bluetooth widget does nothing", "bt pairing fails", "bluetooth toggle flips back"],
    14: ["colors change on their own", "scheme resets to material you", "screen flash stutters"],
    333: ["keyboard shortcuts dead", "keybinds do nothing", "global shortcuts broken"],
    402: ["vesktop freezes when i screenshare", "discord crash camera", "stream with audio freezes"],
    409: ["video wallpapers not appearing", "live wallpaper duplicated", "wallpaper does not play"],
    416: ["launcher opens on wrong screen", "cant alt tab when fullscreened", "window switcher wrong monitor"],
    418: ["quickshell crashed dialog", "bar shows no window icons", "restart shell after crash"],
    461: ["apps randomly crash", "browsers crash after caelestia", "memory leak quickshell"],
    528: ["quickshell wont start after update", "shell broken after caelestia update", "reinstall quickshell"],
    616: ["gpu percent wrong", "gpu shows half usage", "two gpus averaged"],
    641: ["game launches in half the screen", "gamescope bottom of desktop visible", "dodge windows toggle"],
    763: ["colors revert after reboot", "variant changes back to tonal spot", "scheme resets after login"],
    120: ["what can the assistant do", "assistant scope", "nl settings assistant"],
    802: ["what can the assistant do", "narrow assistant", "assistant limitations"],
}

# Scope sections for the two OPEN issues: these define what the on-device
# assistant itself may claim to do. Every claim quotes the captured thread.
SCOPE_NOTES: Dict[int, List[str]] = {
    120: [
        'Proposal (verbatim): "The assistant would interpret the request, convert it into structured '
        'configuration changes, and apply them through Caelestia\'s existing configuration system."',
        'Proposal (verbatim): "The assistant should never modify configuration files directly. Instead, it '
        'should only request changes through a controlled API."',
        'Maintainer (ladybug-me): the AI assistant "already supports handling the shell through '
        '`caelestia-cli` but is not working at the moment"; working tools are "screenshot (read screen), '
        'get_weather, web_search and read_wbpg"; "Many more are there but not working yet."',
        "Corpus note (curated from this thread): the on-device assistant in this repo stays on the narrow "
        "side of this proposal — it retrieves documentation and past resolutions and prints inert "
        "suggestions; it never parses natural-language requests into config writes.",
    ],
    802: [
        'Proposal (verbatim): "it has to be narrow ai, not like an LLM ... it will be quite good at solving '
        'easy to intermediate prblms on linux shell, any kde specific prblms, folder organisations, '
        'cleaning, issue reportings to repo directly etc."',
        'Maintainer (ladybug-me): "Why don\'t you create & train a small model for #120 ?" — i.e. any '
        "assistant work is scoped against the #120 feature thread.",
        "Corpus note (curated from this thread): the on-device assistant claims only narrow troubleshooting "
        "help (docs + resolved issues), folder-size reporting guidance, and issue drafting. It does not "
        "claim general LLM abilities and does not act on its own.",
    ],
}

IMG_RE = re.compile(r"<img[^>]*>", re.IGNORECASE)
ATTACH_URL_RE = re.compile(r"https?://\S*(?:user-attachments|github\.com/user-attachments)\S*", re.IGNORECASE)
DISCORD_CDN_RE = re.compile(r"https?://\S*cdn\.discordapp\.com\S*", re.IGNORECASE)
HEADING_RE = re.compile(r"^#{1,6}\s")
BOLD_ONLY_RE = re.compile(r"^\*\*[^*]+\*\*:?$")
CHECKBOX_RE = re.compile(r"^- \[[ xX]\]")
NO_RESPONSE_RE = re.compile(r"^_No response_$", re.IGNORECASE)
EMBEDDED_JSON_RE = re.compile(
    r'<script type="application/json" data-target="react-app\.embeddedData">(.*?)</script>', re.DOTALL
)


def find_issue(payload: Any) -> Optional[Dict[str, Any]]:
    """Depth-first search for the issue object: a dict with title+number+body."""
    if isinstance(payload, dict):
        if (
            isinstance(payload.get("title"), str)
            and isinstance(payload.get("body"), str)
            and isinstance(payload.get("number"), int)
        ):
            return payload
        for value in payload.values():
            found = find_issue(value)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = find_issue(value)
            if found is not None:
                return found
    return None


def iter_comment_nodes(node: Any) -> Iterator[Dict[str, Any]]:
    """Yield every IssueComment dict in document order (no dedup here)."""
    if isinstance(node, dict):
        if node.get("__typename") == "IssueComment" and isinstance(node.get("body"), str):
            yield node
        for value in node.values():
            yield from iter_comment_nodes(value)
    elif isinstance(node, list):
        for value in node:
            yield from iter_comment_nodes(value)


def author_login(obj: Any) -> str:
    if isinstance(obj, dict):
        return str(obj.get("login", "unknown"))
    return str(obj) if obj else "unknown"


def condense(text: str, limit: int, drop_headings: bool = True) -> str:
    """Light deterministic condensing: drop template noise, images, blank runs."""
    if not text:
        return ""
    out: List[str] = []
    for raw in text.split("\n"):
        line = IMG_RE.sub("[image]", raw)
        line = ATTACH_URL_RE.sub("[attachment]", line)
        line = DISCORD_CDN_RE.sub("[attachment]", line)
        stripped = line.strip()
        if NO_RESPONSE_RE.match(stripped):
            continue
        if drop_headings and HEADING_RE.match(stripped):
            continue
        if BOLD_ONLY_RE.match(stripped):
            continue
        if CHECKBOX_RE.match(stripped):
            continue
        if stripped in ("[image]", "[attachment]"):
            if out and out[-1].strip() in ("[image]", "[attachment]"):
                continue
        out.append(line.rstrip())
    collapsed = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    return cap(collapsed, limit)


def cap(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip()
    return f"{cut} …[truncated]"


def load_issue(path: Path) -> Dict[str, Any]:
    html = path.read_text(encoding="utf-8", errors="replace")
    match = EMBEDDED_JSON_RE.search(html)
    if not match:
        raise ValueError(f"{path.name}: no react-app.embeddedData JSON blob found")
    payload = json.loads(match.group(1))
    issue = find_issue(payload)
    if issue is None:
        raise ValueError(f"{path.name}: no issue object (title+number+body) in embedded JSON")

    comments: List[Dict[str, Any]] = []
    seen_ids = set()
    for node in iter_comment_nodes(issue):
        key = node.get("databaseId", id(node))
        if key in seen_ids:
            continue
        seen_ids.add(key)
        comments.append(
            {
                "author": author_login(node.get("author")),
                "association": str(node.get("authorAssociation", "NONE")),
                "created": str(node.get("createdAt", "")),
                "body": node["body"],
            }
        )
    # front/back timeline pagination can overlap; sort for a stable order.
    comments.sort(key=lambda c: (c["created"], str(c.get("author"))))
    closing_prs = [
        {"number": n.get("number"), "state": n.get("state")}
        for n in (issue.get("closedByPullRequestsReferences") or {}).get("nodes", [])
        if isinstance(n, dict) and n.get("number")
    ]
    return {
        "number": issue["number"],
        "title": issue["title"],
        "state": issue.get("state", "?"),
        "state_reason": issue.get("stateReason"),
        "author": author_login(issue.get("author")),
        "created": str(issue.get("createdAt", "")),
        "body": issue["body"],
        "comments": comments,
        "closing_prs": closing_prs,
    }


def find_html_file(html_dir: Path, number: int) -> Path:
    for pattern in (f"issue_{number}.html", f"i{number}.html", f"issue{number}.html"):
        candidate = html_dir / pattern
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no HTML capture for issue #{number} in {html_dir}")


def render_resolution(issue: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    pr_bits = ", ".join(f"#{pr['number']} ({pr['state']})" for pr in issue["closing_prs"])
    if pr_bits:
        lines.append(f"Closed by PR {pr_bits}.")
    keepers = [
        c
        for c in issue["comments"]
        if c["association"] in ("OWNER", "MEMBER", "COLLABORATOR")
    ]
    if keepers:
        lines.append("What maintainers/collaborators actually said (verbatim, lightly trimmed):")
        lines.append("")
        for c in keepers:
            date = c["created"][:10]
            lines.append(f"> **{c['author']}** ({c['association']}, {date}):")
            for qline in condense(c["body"], 900, drop_headings=False).split("\n"):
                lines.append(f"> {qline}" if qline.strip() else ">")
            lines.append("")
    else:
        lines.append(
            "No maintainer/collaborator explanation was captured in the HTML for this issue; "
            "the comment log below is all the resolution detail that exists. Do not invent one."
        )
        lines.append("")
    return lines


def render_scope(issue: Dict[str, Any]) -> List[str]:
    lines = ["## Scope", ""]
    lines.extend(SCOPE_NOTES.get(issue["number"], ["(no curated scope note for this open issue)"]))
    lines.append("")
    return lines


def render_doc(issue: Dict[str, Any]) -> str:
    number = issue["number"]
    doc_id = f"ISS-{number:03d}"
    state = issue["state"]
    reason = f", stateReason {issue['state_reason']}" if issue["state_reason"] else ""
    tags = ", ".join(TAGS.get(number, [])) or f"issue, {state.lower()}"
    syn = ", ".join(SYNONYMS.get(number, []))

    lines: List[str] = []
    lines.append(f"id: {doc_id}")
    lines.append(f"title: {issue['title'].replace(chr(10), ' ')}")
    lines.append(f"source: issue #{number} ({state}{reason}) {REPO_URL}/issues/{number}")
    lines.append(f"tags: {tags}")
    if syn:
        lines.append(f"synonyms: {syn}")
    lines.append("")
    lines.append(f"# {doc_id} — {issue['title']}")
    lines.append("")
    opener = f"State: {state}{reason} · opened {issue['created'][:10]} by {issue['author']}."
    lines.append(opener)
    lines.append("")

    body = condense(issue["body"], 1400)
    if number in OPEN_ISSUES:
        lines.append(
            "This thread is OPEN: there is no resolution. It is part of the corpus because it "
            "defines what the on-device assistant itself may claim to do — no more, no less."
        )
        lines.append("")
        lines.append("## Proposal (original, condensed)")
        lines.append("")
        lines.append(body if body else "(body empty in capture)")
        lines.append("")
        lines.extend(render_scope(issue))
    else:
        lines.append("## Problem (original report, condensed)")
        lines.append("")
        lines.append(body if body else "(body empty in capture)")
        lines.append("")
        lines.append("## Resolution")
        lines.append("")
        lines.extend(render_resolution(issue))

    lines.append("## Comment log")
    lines.append("")
    if not issue["comments"]:
        lines.append("(no comments captured)")
        lines.append("")
    for c in issue["comments"]:
        date = c["created"][:16].replace("T", " ")
        lines.append(f"- **{c['author']}** ({c['association']}, {date}):")
        body = condense(c["body"], 900, drop_headings=False)
        for cline in body.split("\n"):
            lines.append(f"  {cline}" if cline.strip() else "")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build ISS-*.md corpus docs from captured issue HTML.")
    parser.add_argument(
        "html_dir", nargs="?", default=str(DEFAULT_HTML_DIR), help="directory with issue_N.html captures"
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="corpus output directory")
    parser.add_argument("--all", action="store_true", help="parse every captured issue file, not just the curated list")
    args = parser.parse_args(argv)

    html_dir = Path(args.html_dir)
    out_dir = Path(args.out)
    if not html_dir.is_dir():
        print(f"error: {html_dir} is not a directory", file=sys.stderr)
        return 2

    numbers: List[int]
    if args.all:
        numbers = sorted(
            int(m.group(1))
            for p in html_dir.glob("*.html")
            for m in [re.search(r"(?:issue_?|i)(\d+)\.html$", p.name)]
            if m
        )
    else:
        numbers = sorted(DEFAULT_ISSUES)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for number in numbers:
        try:
            issue = load_issue(find_html_file(html_dir, number))
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            print(f"skip #{number}: {exc}", file=sys.stderr)
            continue
        target = out_dir / f"ISS-{number:03d}.md"
        target.write_text(render_doc(issue), encoding="utf-8")
        written.append(target.name)
    print(f"wrote {len(written)} issue corpus docs to {out_dir}")
    for name in written:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
