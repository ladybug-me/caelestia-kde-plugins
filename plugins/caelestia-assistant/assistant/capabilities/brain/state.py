"""Persistent model state (learned weights, counts, bandit arms) as one JSON file.

Path resolution: an explicit ``path`` argument always wins; else the
``CAELESTIA_BRAIN_STATE`` environment variable (tests and sandboxes
isolate their state without touching the user's); else the default
below. The same override pattern as the capability manifest
(``CAELESTIA_ASSIST_CAPABILITIES``) — the user's runtime state is
never silently redirected.
"""
import json
import os
import pathlib

DEFAULT_STATE = pathlib.Path(os.path.expanduser("~/.local/state/caelestia-brain/state.json"))


def resolve_path(path=None) -> pathlib.Path:
    """The effective state path (explicit arg > env var > default)."""
    if path is not None:
        return pathlib.Path(path)
    env = os.environ.get("CAELESTIA_BRAIN_STATE")
    if env:
        return pathlib.Path(env)
    return DEFAULT_STATE


def load(path=None):
    p = resolve_path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def save(state, path=None):
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)
