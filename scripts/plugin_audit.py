"""Shared helpers for the Caelestia plugin store audits.

Imported by ``validate.py`` (manifest / structure rules) and ``check_safety.py``
(source-content / safety rules). Kept dependency-free so both scripts run on a
plain Python 3.10+ install.
"""

from __future__ import annotations

import re
from pathlib import Path

# Extensions we read as text when scanning plugin source.
TEXT_EXTENSIONS = {
    ".c", ".cc", ".cmake", ".cpp", ".css", ".desktop", ".h", ".hpp",
    ".html", ".js", ".json", ".md", ".qml", ".py", ".sh", ".svg",
    ".txt", ".xml", ".yaml", ".yml",
}

# Key combinations (e.g. "Meta+Shift+W") mentioned in free-form text such as a
# manifest description. Used to check a description never advertises a binding
# the plugin never registers.
SHORTCUT_RE = re.compile(
    r"\b(?:Ctrl|Control|Meta|Super|Alt|Shift)"
    r"(?:\+(?:Ctrl|Control|Meta|Super|Alt|Shift|[A-Za-z0-9]))+\b"
)

COMMON_WORDS = [
    "asset", "assets", "blacklist", "button", "color", "config", "constant",
    "create", "cursor", "default", "delete", "desktop", "disabled", "download",
    "effect", "enabled", "file", "folder", "global", "icon", "install", "item",
    "layout", "license", "list", "manager", "margin", "media", "metadata",
    "model", "mouse", "opacity", "padding", "path", "plugin", "position",
    "property", "radius", "remove", "resource", "restart", "save", "screen",
    "screenshot", "screenshots", "selected", "settings", "shell", "shortcut",
    "size", "theme", "themes", "title", "uninstall", "update", "upload",
    "variant", "version", "visible", "wallpaper", "window", "write",
]


def is_text_file(path: Path) -> bool:
    """True for files we are willing to read as text."""
    return path.suffix.lower() in TEXT_EXTENSIONS


def iter_text_files(plugin_dir: Path):
    """Yield every text file under ``plugin_dir`` in a stable order."""
    for path in sorted(plugin_dir.rglob("*")):
        if path.is_file() and is_text_file(path):
            yield path


def read_text(path: Path) -> str:
    """Read a file as UTF-8 text, tolerating stray bytes."""
    return path.read_text(encoding="utf-8", errors="ignore")


def plugin_text_blob(plugin_dir: Path, exclude: set[str] | None = None) -> str:
    """Concatenate all text files in a plugin folder, minus any named files."""
    exclude = exclude or set()
    chunks = []
    for path in iter_text_files(plugin_dir):
        if path.name in exclude:
            continue
        chunks.append(read_text(path))
    return "\n".join(chunks)


def split_identifier_tokens(name: str) -> list[str]:
    """Split a camelCase/PascalCase/snake_case/kebab-case name into lowercase
    word tokens, e.g. 'GlobalConstas' -> ['global', 'constas']."""
    words = []
    for part in re.split(r"[^0-9A-Za-z]+", name):
        if not part:
            continue
        spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", part)
        words.extend(spaced.lower().split())
    return words


def osa_distance(a: str, b: str) -> int:
    """Damerau-Levenshtein (optimal string alignment) edit distance.

    Counts single-character insertions, deletions, substitutions and adjacent
    transpositions, so it catches typos like 'asstes' vs 'assets'.
    """
    la, lb = len(a), len(b)
    if a == b:
        return 0
    if la == 0:
        return lb
    if lb == 0:
        return la
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,          # deletion
                d[i][j - 1] + 1,          # insertion
                d[i - 1][j - 1] + cost,   # substitution
            )
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)  # transposition
    return d[la][lb]


def closest_word(word: str, candidates, max_dist: int = 1) -> str | None:
    """Return the closest candidate to ``word`` within ``max_dist``, else None."""
    best = None
    best_dist = max_dist + 1
    for cand in candidates:
        dist = osa_distance(word, cand)
        if dist < best_dist:
            best, best_dist = cand, dist
            if dist == 0:
                break
    return best if best is not None and best_dist <= max_dist else None
