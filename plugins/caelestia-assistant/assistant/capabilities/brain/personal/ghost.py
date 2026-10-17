"""Ghost tasks: actionable-sounding lines in notes that never became a
tracked task.

Heuristic, not NLP: an unchecked markdown checkbox, or a line opening with a
small set of action phrases, is a candidate. A candidate is only reported if
it doesn't already overlap strongly with a known task title — otherwise
every note that mentions an existing task would flag as a ghost.
"""
import re

CHECKBOX = re.compile(r"^\s*-\s*\[ \]\s*(.+)$")
ACTION_START = re.compile(
    r"^\s*(?:todo|to do|need to|should|must|have to|remember to|don'?t forget to)\b[:,]?\s*(.+)$",
    re.IGNORECASE,
)


def _candidate_lines(text):
    out = []
    for line in text.splitlines():
        m = CHECKBOX.match(line)
        if m:
            out.append(m.group(1).strip())
            continue
        m = ACTION_START.match(line)
        if m:
            out.append(m.group(1).strip())
    return out


def _overlap(a, b):
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def find_ghosts(notes, known_task_titles, overlap_threshold=0.4):
    """notes: {note_id: text}. known_task_titles: iterable of tracked titles.

    Returns [{"note", "line"}] for candidates not already tracked.
    """
    known = list(known_task_titles)
    out = []
    for note_id, text in notes.items():
        for line in _candidate_lines(text):
            if len(line) < 4:
                continue
            if any(_overlap(line, k) >= overlap_threshold for k in known):
                continue
            out.append({"note": note_id, "line": line})
    return out
