#!/usr/bin/env python3
"""Validate the Caelestia community plugin store.

Every folder under ``plugins/`` is one plugin. This script checks, for each
plugin folder:

* the folder name is a valid, unique plugin id (lowercase kebab-case),
* ``metadata.json`` exists, parses, and conforms to
  ``schemas/plugin.schema.json``,
* ``metadata.id`` equals the folder name,
* ``version`` is valid semver (``MAJOR.MINOR.PATCH`` with optional
  ``-prerelease`` / ``+build``),
* a ``LICENSE`` file is present,
* referenced ``icon`` / ``screenshots`` exist inside the folder (no path
  traversal),
* the folder contains no compiled binaries or archives (source-only policy),
* type-specific requirements are met:
    * ``quickshell`` - at least one ``.qml`` source file,
    * ``kwineffect``  - a KPackage layout with ``metadata.desktop``,
    * ``theme``       - at least one theme source file,
* ``kwineffect`` metadata is coherent with the declared type (only allowed
  on ``kwineffect`` plugins, and its ``kpluginId`` matches the KWin package),
* the description does not advertise a shortcut the plugin never registers.

It also checks the store as a whole (unique ids, dependency references).

Exit code 0 means the store is valid; any non-zero exit means it is not.

Usage:
    python scripts/validate.py

Requires the ``jsonschema`` package (``pip install jsonschema``).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

try:
    from jsonschema import FormatChecker, ValidationError, validate
except ImportError:  # pragma: no cover
    print(
        "error: the 'jsonschema' package is required.\n"
        "       install it with: pip install jsonschema",
        file=sys.stderr,
    )
    sys.exit(2)

from plugin_audit import SHORTCUT_RE, closest_word, plugin_text_blob

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGINS_DIR = REPO_ROOT / "plugins"
SCHEMA_PATH = REPO_ROOT / "schemas" / "plugin.schema.json"

# A plugin id is the folder name: lowercase letters, digits, single hyphens.
ID_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

# Semver: 1.2.3, 1.2.3-alpha.1, 1.2.3+build.5, 1.2.3-alpha.1+build.5
SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9]\d*|[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*))*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)

# Names that can never be used as a plugin id.
RESERVED_NAMES = {
    ".git", ".github", "docs", "schemas", "scripts", "template-plugin",
    "plugins", "index.json", "README.md", "LICENSE", "assets",
}

# Extensions we never accept inside a plugin folder (source-only store).
FORBIDDEN_EXTENSIONS = {
    ".7z", ".a", ".ar", ".bin", ".class", ".dll", ".dylib", ".exe", ".gz",
    ".jar", ".o", ".obj", ".plasmoid", ".pyc", ".pyo", ".rar", ".rpm", ".so",
    ".tar", ".tgz", ".whl", ".xz", ".zip",
}

# Images are fine (icons / screenshots).
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".avif"}

# Directory names conventionally used for icon / screenshot assets. An icon or
# screenshot path whose first component is a near-miss of one of these (e.g.
# 'asstes' for 'assets') is almost certainly a typo.
CONVENTIONAL_ASSET_DIRS = {
    "assets", "data", "icon", "icons", "img", "images", "media",
    "resources", "res", "screenshots",
}


def iter_plugin_dirs():
    """Yield each plugin folder under ``plugins/`` in sorted order."""
    if not PLUGINS_DIR.is_dir():
        return
    for child in sorted(PLUGINS_DIR.iterdir()):
        if child.is_dir():
            yield child


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def check_referenced_file(plugin_dir, plugin_id, field, rel):
    """Ensure a referenced icon/screenshot is a real file inside the folder."""
    path = Path(rel)
    if path.is_absolute() or ".." in path.parts:
        return [
            f"plugin '{plugin_id}': {field} path {rel!r} must be relative and "
            "stay inside the plugin folder"
        ]
    resolved = (plugin_dir / path).resolve()
    if not resolved.is_file():
        return [f"plugin '{plugin_id}': {field} references missing file {rel!r}"]
    if not resolved.is_relative_to(plugin_dir.resolve()):
        return [f"plugin '{plugin_id}': {field} path {rel!r} escapes the plugin folder"]
    return []


def read_desktop_field(plugin_dir, key):
    """Read a `Key=Value` entry from a KPackage `metadata.desktop` file."""
    desktop = plugin_dir / "metadata.desktop"
    if not desktop.is_file():
        return None
    try:
        text = desktop.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(f"{key}="):
            return line[len(key) + 1:].strip()
    return None


def shortcut_claims(text):
    """Key combinations (e.g. 'Meta+Shift+W') mentioned in free-form text."""
    return {m.group(0) for m in SHORTCUT_RE.finditer(text)}


def check_asset_dir(plugin_id, field, rel, warnings):
    """Flag icon/screenshot paths that start with an unconventional directory
    name that is a near-miss of a conventional one (a probable typo)."""
    parts = rel.replace("\\", "/").split("/")
    if len(parts) < 2:
        return
    first = parts[0].lower()
    if first in CONVENTIONAL_ASSET_DIRS:
        return
    suggestion = closest_word(first, CONVENTIONAL_ASSET_DIRS)
    if suggestion:
        warnings.append(
            f"plugin '{plugin_id}': {field} path {rel!r} starts with "
            f"directory {parts[0]!r} - did you mean {suggestion!r}?"
        )


def _check_kwineffect_coherence(plugin_dir, plugin_id, meta, problems):
    """Cross-field rules the JSON schema cannot express on its own."""
    plugin_type = meta.get("type")
    kwineffect = meta.get("kwineffect")
    has_desktop = (plugin_dir / "metadata.desktop").is_file()

    if plugin_type == "kwineffect":
        if not kwineffect or not kwineffect.get("kpluginId"):
            problems.append(
                f"plugin '{plugin_id}': type 'kwineffect' requires "
                "'kwineffect.kpluginId' in metadata.json"
            )
        elif has_desktop:
            registered = read_desktop_field(plugin_dir, "X-KDE-PluginInfo-Name")
            if registered and registered != kwineffect["kpluginId"]:
                problems.append(
                    f"plugin '{plugin_id}': kwineffect.kpluginId "
                    f"({kwineffect['kpluginId']!r}) does not match "
                    f"metadata.desktop X-KDE-PluginInfo-Name ({registered!r})"
                )
    else:
        if kwineffect:
            problems.append(
                f"plugin '{plugin_id}': 'kwineffect' metadata is only valid for "
                f"type 'kwineffect', but type is {plugin_type!r}"
            )
        if has_desktop:
            problems.append(
                f"plugin '{plugin_id}': a KWin 'metadata.desktop' is present but "
                f"type is {plugin_type!r}; KWin packages must declare "
                "type 'kwineffect'"
            )


def _check_shortcut_description(plugin_dir, plugin_id, meta, problems):
    """The description must not advertise a key binding the plugin never binds."""
    claims = shortcut_claims(meta.get("description", ""))
    if not claims:
        return
    blob = plugin_text_blob(plugin_dir, exclude={"metadata.json"})
    missing = sorted(claim for claim in claims if claim not in blob)
    if missing:
        problems.append(
            f"plugin '{plugin_id}': description mentions shortcut(s) {missing} "
            "that do not appear anywhere in the plugin source"
        )


def find_forbidden_files(plugin_dir):
    """Return relative paths of any compiled binary / archive in the folder."""
    bad = []
    for path in plugin_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in FORBIDDEN_EXTENSIONS:
            bad.append(path.relative_to(plugin_dir).as_posix())
    return sorted(bad)


def has_theme_content(plugin_dir):
    """True if the folder has any non-metadata, non-image theme file."""
    for path in plugin_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(plugin_dir).as_posix()
        if rel in {"metadata.json", "LICENSE", "README.md"}:
            continue
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            continue
        return True
    return False


def validate_plugin(plugin_dir, schema, warnings):
    """Validate a single plugin folder.

    Appends hard failures to the returned `problems` list and softer
    judgement-call findings to `warnings`.
    """
    problems = []
    plugin_id = plugin_dir.name
    metadata_path = plugin_dir / "metadata.json"

    if ID_RE.fullmatch(plugin_id) is None:
        problems.append(
            f"plugin '{plugin_id}': folder name is not a valid plugin id "
            "(lowercase letters, digits and single hyphens only, e.g. my-plugin)"
        )
    if plugin_id in RESERVED_NAMES:
        problems.append(f"plugin '{plugin_id}': '{plugin_id}' is a reserved name")

    if not metadata_path.is_file():
        problems.append(f"plugin '{plugin_id}': missing required file metadata.json")
        return problems

    try:
        meta = load_json(metadata_path)
    except json.JSONDecodeError as exc:
        problems.append(f"plugin '{plugin_id}': metadata.json is not valid JSON: {exc}")
        return problems
    except OSError as exc:
        problems.append(f"plugin '{plugin_id}': could not read metadata.json: {exc}")
        return problems

    try:
        validate(instance=meta, schema=schema, format_checker=FormatChecker())
    except ValidationError as exc:
        location = "/".join(str(part) for part in exc.absolute_path) or "/"
        problems.append(
            f"plugin '{plugin_id}': metadata.json violates the schema at {location}: "
            f"{exc.message}"
        )

    if meta.get("id") != plugin_id:
        problems.append(
            f"plugin '{plugin_id}': metadata.json 'id' ({meta.get('id')!r}) "
            "must equal the folder name"
        )

    version = meta.get("version")
    if isinstance(version, str) and SEMVER_RE.fullmatch(version) is None:
        problems.append(
            f"plugin '{plugin_id}': version {version!r} is not valid semver "
            "(expected MAJOR.MINOR.PATCH, optionally with -prerelease and/or +build)"
        )

    if not (plugin_dir / "LICENSE").is_file():
        problems.append(f"plugin '{plugin_id}': missing required LICENSE file")

    icon = meta.get("icon")
    if icon:
        check_asset_dir(plugin_id, "icon", icon, warnings)
        if icon != "default":
            icon_path = Path(icon)
            if icon_path.is_absolute() or ".." in icon_path.parts:
                problems.append(
                    f"plugin '{plugin_id}': icon path {icon!r} must be relative and "
                    "stay inside the plugin folder unless it is the special 'default' value"
                )
            elif "/" in icon or "\\" in icon or icon_path.suffix:
                problems.extend(check_referenced_file(plugin_dir, plugin_id, "icon", icon))
    for value in meta.get("screenshots", []) or []:
        check_asset_dir(plugin_id, "screenshots", value, warnings)
        problems.extend(check_referenced_file(plugin_dir, plugin_id, "screenshots", value))

    for rel in find_forbidden_files(plugin_dir):
        problems.append(
            f"plugin '{plugin_id}': compiled binary/archive not allowed "
            f"(source-only policy): {rel}"
        )

    plugin_type = meta.get("type")
    if plugin_type == "quickshell":
        if not any(p.suffix == ".qml" for p in plugin_dir.rglob("*") if p.is_file()):
            problems.append(
                f"plugin '{plugin_id}': type 'quickshell' requires at least one .qml file"
            )
    elif plugin_type == "kwineffect":
        if not (plugin_dir / "metadata.desktop").is_file():
            problems.append(
                f"plugin '{plugin_id}': type 'kwineffect' requires a KPackage "
                "metadata.desktop file"
            )
    elif plugin_type == "theme":
        if not has_theme_content(plugin_dir):
            problems.append(
                f"plugin '{plugin_id}': type 'theme' requires at least one theme file"
            )

    _check_kwineffect_coherence(plugin_dir, plugin_id, meta, problems)
    _check_shortcut_description(plugin_dir, plugin_id, meta, problems)

    return problems


def main():
    if not PLUGINS_DIR.is_dir():
        print("No plugins/ directory found; the store is empty.")
        return 0

    schema = load_json(SCHEMA_PATH)

    plugin_dirs = list(iter_plugin_dirs())
    seen_ids = set()
    metadata_by_id = {}
    problems = []
    warnings = []

    for plugin_dir in plugin_dirs:
        plugin_id = plugin_dir.name
        if plugin_id in seen_ids:
            problems.append(f"duplicate plugin id: {plugin_id}")
            continue
        seen_ids.add(plugin_id)
        problems.extend(validate_plugin(plugin_dir, schema, warnings))
        metadata_path = plugin_dir / "metadata.json"
        if metadata_path.is_file():
            try:
                metadata_by_id[plugin_id] = load_json(metadata_path)
            except (json.JSONDecodeError, OSError):
                metadata_by_id[plugin_id] = {}

    for plugin_id, meta in metadata_by_id.items():
        for dependency in meta.get("dependencies", []) or []:
            if dependency not in seen_ids:
                warnings.append(
                    f"plugin '{plugin_id}' depends on '{dependency}' "
                    "which is not in the store"
                )

    for warning in warnings:
        print(f"warning: {warning}")
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)

    if problems:
        print(f"\n{len(problems)} error(s) found.", file=sys.stderr)
        return 1

    print(f"OK: {len(plugin_dirs)} plugin(s) validated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
