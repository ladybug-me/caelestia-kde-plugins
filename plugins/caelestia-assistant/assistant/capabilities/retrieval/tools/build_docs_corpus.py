#!/usr/bin/env python3
"""Build the DOC-*.md half of the retrieval corpus from the repo's own docs.

Usage:
    python3 assistant/retrieval/tools/build_docs_corpus.py [--repo ROOT] [--out DIR]

Splits docs/TROUBLESHOOTING.md at its top-level "## N." sections (one corpus
doc per section, internal headings and Symptom|Cause|Fix tables preserved
verbatim), and wraps docs/architecture/{kwin_port,lockscreen,shortcut}_architecture.md
plus README.md as one doc each. Every doc gets a strict `key: value` header
(id, title, source, tags[, synonyms]) followed by a blank line and the body.

Curated tags/synonyms are hand-picked keywords from the actual section content
(retrieval hooks, not new claims). Stdlib only, fully offline.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "assistant" / "retrieval" / "corpus"

SECTION_SPLIT_RE = re.compile(r"^## (\d+)\. (.+?)\s*$", re.MULTILINE)

# Curated retrieval tags per TROUBLESHOOTING section (keywords from content).
TS_TAGS: Dict[str, List[str]] = {
    "1": ["build", "compile", "cmake", "g++", "gcc", "make", "installer tui", "c++20", "qt6", "qml module metadata",
          "qmldir", "ccache", "sigsegv", "exit 139", "exit 127"],
    "2": ["packages", "pacman", "yay", "aur", "fedora", "copr", "dnf", "rpm fusion", "matugen", "dos2unix", "crlf",
          "failed_packages.txt", "quickshell-git", "libcava"],
    "3": ["runtime", "shell", "quickshell", "caelestia-shell.service", "systemd user unit", "journalctl",
          "environment variables", "qml2_import_path", "window thumbnails", "screencast", "screen share", "vesktop",
          "camera freeze", "colors not applying", "scheme", "screen recording", "screenshots", "screen flashes",
          "kde-material-you-colors", "workspace tracker", "workspace pills", "kde update", "tonal spot"],
    "4": ["lock screen", "lockscreen", "greeter", "breeze", "kscreenlockerrc", "plasmashellrc", "shell package",
          "kreadconfig6", "kwriteconfig6"],
    "5": ["configuration", "darkly", "theme", "window decoration", "osd", "virtual desktops", "kglobalaccel",
          "shortcut conflicts", "stolen shortcuts", "terminal", "garbled output"],
    "6": ["network", "proxy", "mirror ranking", "pacman mirrors", "git submodules", "src/dots empty",
          "aur behind proxy"],
    "7": ["kde", "plasma", "kwin", "qs-kwin-bridge", "xdg-desktop-portal", "portals", "ydotoold", "krohnkite",
          "kwin script injection", "login screen", "sddm", "window rules", "installer window rules"],
    "8": ["post-install", "shell not visible", "installer exited prematurely", "confirm_arg", "stale lock files",
          "update.sh fails", "partial upgrade", "packages replaced in memory", "prebuilt shell", "checksum"],
    "9": ["uninstall", "backups", "konsave", "restore", "rc files", "dependencies", "input group", "failed patches"],
    "10": ["update", "updater", "update.sh", "caelestia updater", "re-clone", "cache", "install latest version"],
    "11": ["diagnostics", "diagnostic commands", "reference", "journalctl", "systemctl status", "debug mode",
           "system state checks", "kde cache refresh", "kbuildsycoca6", "logs"],
}

TS_SYNONYMS: Dict[str, List[str]] = {
    "1": ["install fails cmake missing", "g++ command not found", "build errors"],
    "3": ["shell does not start", "blank screen at login", "vesktop freezes when i screenshare",
          "colors revert after reboot", "workspace pills not loading", "scheme resets"],
    "4": ["breeze lock screen shows instead of caelestia"],
    "5": ["shortcuts stopped working", "osd keeps showing", "wrong number of desktops"],
    "6": ["git clone fails", "submodule empty"],
    "8": ["disk space", "ccache size", "free disk space used by the shell", "stale lock"],
}

ARCH_DOCS: List[Tuple[str, str, str, List[str]]] = [
    ("kwin_port_architecture.md", "DOC-ARCH-KWIN", "KWin port architecture (native backend)",
     ["kwin", "bridge", "kwinactivewindowbridge", "kwinworkspacestate", "plasma-window-management", "wayland",
      "globalshortcut", "kglobalaccel", "hyprctl", "workspace tracker", "qlocalsocket", "window list", "api"]),
    ("lockscreen_architecture.md", "DOC-ARCH-LOCKSCREEN", "Lock screen architecture",
     ["lockscreen", "lock screen", "greeter", "kscreenlocker_greet", "plasma shell package", "pam", "fingerprint",
      "kpackage", "caelestia.desktop"]),
    ("shortcut_architecture.md", "DOC-ARCH-SHORTCUT", "Shortcut system architecture",
     ["shortcuts", "globalshortcut", "keybinds", "keybindsmodel", "globalshortcutdispatcher", "stolen shortcuts",
      "kglobalaccel", "conflicts", "rebind"]),
]

ARCH_SYNONYMS: Dict[str, List[str]] = {
    "DOC-ARCH-KWIN": ["kwin bridge", "what is the kwin bridge", "hyprctl replacement native plugins",
                      "kwinactivewindowbridge kwinworkspacestate globalshortcut", "window tracking backend"],
    "DOC-ARCH-LOCKSCREEN": ["how does the lock screen work", "greeter architecture"],
    "DOC-ARCH-SHORTCUT": ["how shortcuts are registered", "shortcut stealing recovery"],
}

README_TAGS = ["readme", "install", "installation", "update", "updating", "uninstall", "keybinds", "configuring",
               "nexus", "troubleshooting", "repository layout", "requirements", "caelestia scheme set",
               "widgets not appearing"]
README_SYNONYMS = ["how do i install", "how do i update the shell", "how to uninstall", "what are the keybinds"]


def split_troubleshooting(text: str) -> List[Tuple[str, str, str]]:
    """Return [(number, heading, body)] for each '## N.' section, body verbatim."""
    matches = list(SECTION_SPLIT_RE.finditer(text))
    sections: List[Tuple[str, str, str]] = []
    for idx, match in enumerate(matches):
        number, heading = match.group(1), match.group(2).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end]
        body = body.lstrip("\n")
        body = re.sub(r"\n---\s*$", "", body).rstrip()  # drop the trailing section divider only
        sections.append((number, heading, body))
    return sections


def render_doc(doc_id: str, title: str, source: str, tags: List[str], body: str,
               synonyms: Optional[List[str]] = None) -> str:
    header = [f"id: {doc_id}", f"title: {title}", f"source: {source}", f"tags: {', '.join(tags)}"]
    if synonyms:
        header.append(f"synonyms: {', '.join(synonyms)}")
    return "\n".join(header) + "\n\n" + body.rstrip() + "\n"


def build(repo_root: Path) -> List[Tuple[str, str]]:
    """Return [(filename, content)] for every DOC-* corpus doc."""
    docs: List[Tuple[str, str]] = []
    ts_path = repo_root / "docs" / "TROUBLESHOOTING.md"
    for number, heading, body in split_troubleshooting(ts_path.read_text(encoding="utf-8")):
        doc_id = f"DOC-TS-{int(number):02d}"
        docs.append(
            (
                f"{doc_id}.md",
                render_doc(
                    doc_id,
                    f"{number}. {heading}",
                    f"docs/TROUBLESHOOTING.md — section {number} ({heading})",
                    TS_TAGS.get(number, ["troubleshooting"]),
                    body,
                    TS_SYNONYMS.get(number),
                ),
            )
        )

    for rel, doc_id, title, tags in ARCH_DOCS:
        path = repo_root / "docs" / "architecture" / rel
        body = path.read_text(encoding="utf-8")
        body = re.sub(r"^# .+?\n", "", body, count=1).strip()  # drop the duplicated H1, keep the rest verbatim
        body = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", body)  # screenshot embeds add no retrievable text
        docs.append(
            (
                f"{doc_id}.md",
                render_doc(doc_id, title, f"docs/architecture/{rel}", tags, body, ARCH_SYNONYMS.get(doc_id)),
            )
        )

    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    docs.append(("DOC-README.md", render_doc("DOC-README", "caelestia-kde README", "README.md", README_TAGS, readme,
                                             README_SYNONYMS)))
    return docs


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build DOC-*.md corpus docs from the repo's own docs.")
    parser.add_argument("--repo", default=str(REPO_ROOT), help="repository root")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="corpus output directory")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo)
    out_dir = Path(args.out)
    ts_path = repo_root / "docs" / "TROUBLESHOOTING.md"
    if not ts_path.is_file():
        print(f"error: {ts_path} not found", file=sys.stderr)
        return 2

    docs = build(repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in docs:
        (out_dir / filename).write_text(content, encoding="utf-8")
    print(f"wrote {len(docs)} doc corpus files to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
