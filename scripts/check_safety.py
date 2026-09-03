#!/usr/bin/env python3
"""Source-content and safety audit for the Caelestia plugin store.

Complements ``scripts/validate.py``, which checks metadata and folder
structure. This script scans the *content* of every plugin for behaviours the
store's source-only, no-surprise policy forbids, plus low-severity naming
heuristics:

* external downloads / network fetches at build or runtime   (error)
* privilege escalation (pkexec / sudo / doas / ...)           (error)
* generic shelling out                                        (warning)
* identifier and file/directory-name typos                    (warning)
* functions that look like unimplemented stubs                (warning)
* numbered variants with gaps (e.g. icon1, icon3)             (warning)

Errors fail the build unless the offending line carries an opt-out comment on
the same line:

    file(DOWNLOAD https://... ...)  # caelestia-audit: allow-network
    pkexec cmake --install          # caelestia-audit: allow-privilege

Usage:
    python scripts/check_safety.py
    python scripts/check_safety.py --plugin plugins/web-cursor

Exit code 0 means no hard violations (warnings do not fail the build).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from plugin_audit import (
    COMMON_WORDS,
    closest_word,
    is_text_file,
    split_identifier_tokens,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGINS_DIR = REPO_ROOT / "plugins"

ALLOW_NETWORK = "caelestia-audit: allow-network"
ALLOW_PRIVILEGE = "caelestia-audit: allow-privilege"

# Extensions we scan for behaviour. README/LICENSE/docs are excluded so prose
# like "you need curl to build this" does not trip the download rule.
CODE_EXTENSIONS = {
    ".c", ".cc", ".cmake", ".cpp", ".css", ".desktop", ".h", ".hpp",
    ".html", ".js", ".json", ".qml", ".py", ".sh", ".svg",
}

# (regex, category, human description). Category is "net" or "priv".
HARD_PATTERNS = [
    (r"file\s*\(\s*DOWNLOAD", "net", "CMake file(DOWNLOAD) fetches a remote file at build time"),
    (r"\bFetchContent_\w+", "net", "CMake FetchContent pulls an external dependency"),
    (r"\bExternalProject_\w+", "net", "CMake ExternalProject pulls an external dependency"),
    (r"\binclude\s*\(\s*FetchContent\s*\)", "net", "CMake FetchContent pulls an external dependency"),
    (r"\bcurl\b", "net", "curl downloads a remote resource"),
    (r"\bwget\b", "net", "wget downloads a remote resource"),
    (r"\bgit\s+(clone|ls-remote|fetch)\b", "net", "git fetches a remote repository"),
    (r"\bXMLHttpRequest\b", "net", "XMLHttpRequest performs a network request"),
    (r"\bfetch\s*\(\s*['\"]https?:", "net", "fetch() performs a network request"),
    (r"\bQNetworkAccessManager\b", "net", "QNetworkAccessManager performs a network request"),
    (r"\bQNetworkRequest\b", "net", "QNetworkRequest performs a network request"),
    (r"\bcurl_easy_\w+", "net", "libcurl performs a network request"),
    (r"\burlopen\s*\(", "net", "urllib downloads a remote resource"),
    (r"\brequests\.(get|post|put|delete|head|patch)\s*\(", "net", "requests downloads a remote resource"),
    (r"\bhttpx\.", "net", "httpx performs a network request"),
    (r"\baiohttp\b", "net", "aiohttp performs a network request"),
    (r"\bpkexec\b", "priv", "pkexec elevates to root"),
    (r"\bsudo\b", "priv", "sudo elevates to root"),
    (r"\bdoas\b", "priv", "doas elevates to root"),
    (r"\bgksu\b", "priv", "gksu elevates to root"),
    (r"\bkdesu\b", "priv", "kdesu elevates to root"),
]

# (regex, human description). Warning-only: QML plugins legitimately spawn
# helpers, but the store wants these surfaced for maintainer review.
SOFT_PATTERNS = [
    (r"\bsh\s+-c\b", "shells out via sh -c"),
    (r"\bbash\s+-c\b", "shells out via bash -c"),
    (r"\bProcess\.create\b", "spawns a child process (Quickshell Process)"),
    (r"\bQProcess\b", "spawns a child process (QProcess)"),
    (r"\bsubprocess\.(run|call|Popen|check_output)\b", "spawns a child process (subprocess)"),
    (r"\bos\.system\b", "shells out via os.system"),
    (r"\bbusctl\b", "shells out via busctl"),
    (r"\bsystemctl\b", "shells out via systemctl"),
]

# Identifier tokens we never flag as typos, beyond the domain words themselves.
STOP_WORDS = {
    "about", "after", "again", "also", "always", "between", "bool", "both",
    "break", "case", "catch", "class", "close", "column", "const",
    "constexpr", "continue", "delegate", "double", "else", "enum", "false",
    "final", "float", "for", "friend", "function", "height", "include",
    "inline", "int", "long", "name", "namespace", "new", "noexcept", "null",
    "number", "object", "open", "override", "private", "protected", "public",
    "readonly", "return", "row", "select", "short", "signal", "slot", "static",
    "string", "struct", "switch", "template", "this", "throw", "true", "try",
    "typename", "undefined", "union", "unsigned", "using", "var", "virtual",
    "void", "while", "width",
}

# Path-component names we never flag (structural / conventional).
IGNORED_NAMES = {
    "assets", "components", "config", "contents", "header", "lib", "license",
    "main.qml", "metadata.desktop", "metadata.json", "qmldir", "readme.md",
    "index.html", "utils", "views",
}

STUB_RE = re.compile(
    r"\b\w+(?:::\w+)*\s*\([^;{}]*\)\s*(?:const)?\s*"
    r"\{\s*(?:return\s+(?:true|false|nullptr|0|-?[0-9]+|"
    r"\{\s*\}\s*;|[A-Za-z_][\w:<>]*\s*\(\s*\)\s*;|;\s*)?)?\s*\}"
)

NUMBERED_RE = re.compile(r"^(?P<prefix>[A-Za-z_][A-Za-z0-9_]*?)(?P<num>\d+)$")


def is_code_file(path: Path) -> bool:
    return path.name == "CMakeLists.txt" or path.suffix.lower() in CODE_EXTENSIONS


def iter_code_files(plugin_dir: Path):
    for path in sorted(plugin_dir.rglob("*")):
        if path.is_file() and is_code_file(path):
            yield path


def scan_hard_patterns(path: Path, problems):
    text = path.read_text(encoding="utf-8", errors="ignore")
    rel = path.relative_to(REPO_ROOT).as_posix()
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
            continue
        if stripped.startswith("<!--"):
            continue
        if stripped.startswith("#"):
            # C/C++ preprocessor lines are not comments; #include <curl/...>
            # is a real network signal, everything else (CMake/Python/shell
            # comments) is prose we ignore.
            if re.search(r"#\s*include\s*[<\"]curl/", line):
                problems.append(f"{rel}:{lineno}: network download/fetch (libcurl include)")
            continue
        for pattern, category, desc in HARD_PATTERNS:
            if not re.search(pattern, line):
                continue
            marker = ALLOW_NETWORK if category == "net" else ALLOW_PRIVILEGE
            if marker in line:
                continue
            problems.append(f"{rel}:{lineno}: {desc}")
            break


def scan_soft_patterns(path: Path, warnings):
    text = path.read_text(encoding="utf-8", errors="ignore")
    rel = path.relative_to(REPO_ROOT).as_posix()
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
            continue
        if stripped.startswith("#") or stripped.startswith("<!--"):
            continue
        for pattern, desc in SOFT_PATTERNS:
            if re.search(pattern, line):
                warnings.append(f"{rel}:{lineno}: {desc}")


def _flag_token(rel, token, warnings):
    token = token.strip("_")
    if len(token) < 5:
        return
    word = token.rstrip("0123456789")
    if not word or word in COMMON_WORDS or word in STOP_WORDS:
        return
    if word.endswith("s") and word[:-1] in COMMON_WORDS:
        return
    if word.startswith("i") and word[1:] in COMMON_WORDS:
        return  # interface prefix (e.g. IMouse) is a common, intentional form
    suggestion = closest_word(word, COMMON_WORDS)
    if suggestion:
        warnings.append(f"{rel}: possible typo {token!r} - did you mean {suggestion!r}?")


def scan_name_typos(plugin_dir, warnings):
    for path in sorted(plugin_dir.rglob("*")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        parts = path.relative_to(plugin_dir).parts
        last = len(parts) - 1
        for index, part in enumerate(parts):
            is_file_part = path.is_file() and index == last
            name = Path(part).stem if is_file_part else part
            if not name or name.lower() in IGNORED_NAMES:
                continue
            for token in split_identifier_tokens(name):
                _flag_token(rel, token, warnings)


TYPE_RE = re.compile(r"\b(?:class|struct|enum(?:\s+class)?)\s+([A-Za-z_][A-Za-z0-9_]*)")


def scan_type_typos(path: Path, warnings):
    """Check declared class/struct/enum names for near-miss typos."""
    if path.suffix.lower() not in {".cpp", ".hpp", ".h", ".c"}:
        return
    rel = path.relative_to(REPO_ROOT).as_posix()
    text = path.read_text(encoding="utf-8", errors="ignore")
    seen = set()
    for name in TYPE_RE.findall(text):
        if name in seen:
            continue
        seen.add(name)
        for token in split_identifier_tokens(name):
            _flag_token(rel, token, warnings)


def scan_stubs(path: Path, warnings):
    if path.suffix.lower() not in {".cpp", ".hpp", ".h", ".c"}:
        return
    rel = path.relative_to(REPO_ROOT).as_posix()
    text = path.read_text(encoding="utf-8", errors="ignore")
    for lineno, line in enumerate(text.splitlines(), 1):
        if "= default" in line or "= delete" in line or "= 0" in line:
            continue
        if STUB_RE.search(line):
            warnings.append(f"{rel}:{lineno}: possible stub / unimplemented function")


def scan_sequence_gaps(plugin_dir, warnings):
    groups: dict[str, set[int]] = {}
    for path in plugin_dir.rglob("*"):
        name = path.stem if path.is_file() else path.name
        match = NUMBERED_RE.match(name)
        if not match:
            continue
        groups.setdefault(match.group("prefix"), set()).add(int(match.group("num")))
    for prefix, nums in groups.items():
        if len(nums) < 3:
            continue
        missing = [n for n in range(min(nums), max(nums) + 1) if n not in nums]
        if missing:
            warnings.append(
                f"{plugin_dir.name}: '{prefix}<N>' items are missing "
                f"{missing} (e.g. {prefix}{missing[0]})"
            )


def scan_plugin(plugin_dir: Path, problems, warnings):
    for path in iter_code_files(plugin_dir):
        scan_hard_patterns(path, problems)
        scan_soft_patterns(path, warnings)
        scan_type_typos(path, warnings)
        scan_stubs(path, warnings)
    scan_name_typos(plugin_dir, warnings)
    scan_sequence_gaps(plugin_dir, warnings)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plugin",
        help="audit only one plugin directory, relative to the repo root",
    )
    args = parser.parse_args(argv)

    if args.plugin:
        target = Path(args.plugin)
        plugin_dirs = [target if target.is_absolute() else REPO_ROOT / target]
    else:
        plugin_dirs = [p for p in sorted(PLUGINS_DIR.iterdir()) if p.is_dir()]

    problems = []
    warnings = []
    for plugin_dir in plugin_dirs:
        if plugin_dir.is_dir():
            scan_plugin(plugin_dir, problems, warnings)

    for warning in warnings:
        print(f"warning: {warning}")
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)

    if problems:
        print(f"\n{len(problems)} error(s) found.", file=sys.stderr)
        print(
            "hint: add '# caelestia-audit: allow-network' or "
            "'# caelestia-audit: allow-privilege' on the offending line to "
            "opt out after maintainer review.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {len(plugin_dirs)} plugin(s) audited ({len(warnings)} warning(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
