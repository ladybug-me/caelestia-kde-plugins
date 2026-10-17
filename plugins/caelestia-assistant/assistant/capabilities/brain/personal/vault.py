"""Read a markdown vault: tags, wiki-links, and text per note."""
import pathlib
import re

FRONT_TAGS = re.compile(r"^tags:\s*\[(.*?)\]", re.MULTILINE)
INLINE_TAG = re.compile(r"(?<![\w&])#([A-Za-z][\w-]*)")


def parse_note(text):
    tags = set()
    m = FRONT_TAGS.search(text)
    if m:
        tags.update(t.strip().strip("'\"").lower() for t in m.group(1).split(",") if t.strip())
    tags.update(t.lower() for t in INLINE_TAG.findall(text))
    return tags


def scan(root):
    """Return {relative_path: {"text", "tags", "path"}} for every .md file."""
    root = pathlib.Path(root)
    notes = {}
    for p in sorted(root.rglob("*.md")):
        if any(part.startswith(".") for part in p.relative_to(root).parts):
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        rel = str(p.relative_to(root))
        notes[rel] = {"text": text, "tags": parse_note(text), "path": str(p)}
    return notes
