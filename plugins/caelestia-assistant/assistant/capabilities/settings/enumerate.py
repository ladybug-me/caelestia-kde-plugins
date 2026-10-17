"""Enumerate every settings leaf of the Caelestia config tree (registry input).

Walks the C++ config headers into a leaf-key table, verified against the
dev tree @ 70ee7da.

  * ``enumerate_leaves(repo_root)`` — importable, returns the leaf rows as a
    list of dicts with the same fields the survey's TSV carried (root, path,
    kind, type, default, global_only, header, line, class, macro,
    element_type, derived, notes, order).
  * ``main()`` — prints the TSV to stdout (``--repo-root`` argument, default
    the current directory) instead of writing survey files.

Walks the ACTUAL C++ headers in ``<repo>/shell/plugin/src/Caelestia/Config/``:

  * rootnodes.hpp  -> class ConfigRoot (shell.json via the GlobalConfig/Config
                      singletons/attached objects) and class TokensRoot
                      (shell-tokens.json via TokenConfig).
  * Every class mounted through CONFIG_SUBOBJECT / SETTINGS_SUBOBJECT is
                      recursively parsed for CONFIG_PROPERTY /
                      CONFIG_GLOBAL_PROPERTY / CONFIG_ENUM_PROPERTY /
                      CONFIG_GLOBAL_ENUM_PROPERTY / CONFIG_LIST /
                      CONFIG_GLOBAL_LIST / SETTINGS_* equivalents.

Semantics note (verified against the C++ implementation):
  settings::Schema::build(meta, Base::staticMetaObject.propertyCount())
  (Settings/schema.cpp:89-131) builds each class's schema from the base's
  property count onwards, i.e. a class's schema contains ONLY the properties
  declared in its own body. ObjectNode::toJson/loadFromJson iterate
  schema().descriptors(). Therefore a mounted instance of a subclass (the one
  case in the tree: IconFontStyleConfig : FontStyleConfig at appearance.font.icon)
  exposes only its OWN declarations (extraLarge); the inherited family/large/
  medium/small are live QObject properties on the instance but are outside the
  settings schema (never serialised/loaded/reset). This walker follows that
  "own body only" rule.

Multi-line macro invocations are parsed by parenthesis balancing that skips
comment/string contents (defaults contain parens inside strings, e.g.
ICON_RULE_REGEX("steam(_app_(default|[0-9]+))?", ...)).

Read-only with respect to the repository. Stdlib only.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROPERTY_MACROS = {
    "CONFIG_PROPERTY", "CONFIG_GLOBAL_PROPERTY",
    "SETTINGS_PROPERTY", "SETTINGS_GLOBAL_PROPERTY",
}
ENUM_PROPERTY_MACROS = {
    "CONFIG_ENUM_PROPERTY", "CONFIG_GLOBAL_ENUM_PROPERTY",
}
SUBOBJECT_MACROS = {
    "CONFIG_SUBOBJECT", "CONFIG_GLOBAL_SUBOBJECT",
    "SETTINGS_SUBOBJECT", "SETTINGS_GLOBAL_SUBOBJECT",
}
LIST_MACROS = {
    "CONFIG_LIST", "CONFIG_GLOBAL_LIST",
    "SETTINGS_LIST", "SETTINGS_GLOBAL_LIST",
}
NODE_MACROS = {
    "CONFIG_NODE", "CONFIG_NODE_NO_CTOR",
    "SETTINGS_NODE", "SETTINGS_NODE_NO_CTOR",
}
LIST_TYPE_MACROS = {"CONFIG_LIST_TYPE", "SETTINGS_LIST_TYPE"}
ALL_MACROS = (PROPERTY_MACROS | ENUM_PROPERTY_MACROS | SUBOBJECT_MACROS
              | LIST_MACROS | NODE_MACROS | LIST_TYPE_MACROS)

MACRO_RE = re.compile(
    r"(?<![A-Za-z0-9_])(" + "|".join(sorted(ALL_MACROS, key=len, reverse=True)) + r")\b"
)
CLASS_RE = re.compile(r"(?<![A-Za-z0-9_])(?<!enum )class\s+([A-Za-z_]\w*)\s*(:\s*[^{;]+?)?\s*\{")

FIELDS = ("root", "path", "kind", "type", "default", "global_only", "header",
          "line", "class", "macro", "element_type", "derived", "notes", "order")


# ---------------------------------------------------------------- masking

def mask_comments_and_strings(text: str) -> str:
    """Blank out comment bodies and string/char literal contents (keep newlines)."""
    out = list(text)
    i, n = 0, len(text)
    NORMAL, LINE_C, BLOCK_C, STR, CHR = range(5)
    state = NORMAL
    while i < n:
        c = text[i]
        if state == NORMAL:
            if c == "/" and i + 1 < n and text[i + 1] == "/":
                out[i] = out[i + 1] = " "
                state = LINE_C
                i += 2
                continue
            if c == "/" and i + 1 < n and text[i + 1] == "*":
                out[i] = out[i + 1] = " "
                state = BLOCK_C
                i += 2
                continue
            if c == '"':
                out[i] = " "
                state = STR
                i += 1
                continue
            if c == "'":
                out[i] = " "
                state = CHR
                i += 1
                continue
            i += 1
            continue
        if state == LINE_C:
            if c == "\n":
                state = NORMAL
            else:
                out[i] = " "
            i += 1
            continue
        if state == BLOCK_C:
            if c == "*" and i + 1 < n and text[i + 1] == "/":
                out[i] = out[i + 1] = " "
                state = NORMAL
                i += 2
                continue
            if c != "\n":
                out[i] = " "
            i += 1
            continue
        # STR or CHR
        if c == "\\" and i + 1 < n:
            out[i] = out[i + 1] = " "
            i += 2
            continue
        if (state == STR and c == '"') or (state == CHR and c == "'"):
            out[i] = " "
            state = NORMAL
            i += 1
            continue
        if c != "\n":
            out[i] = " "
        i += 1
    return "".join(out)


def mask_preprocessor(masked: str) -> str:
    """Blank out preprocessor logical lines (handles backslash continuations)."""
    lines = masked.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("#"):
            while True:
                out.append(re.sub(r"[^\n]", " ", lines[i]))
                if not lines[i].rstrip().endswith("\\"):
                    i += 1
                    break
                i += 1
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def collapse_ws(s: str) -> str:
    """Collapse whitespace runs outside string/char literals; keep literals verbatim."""
    out: list[str] = []
    in_lit = False
    quote = ""
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if in_lit:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(s[i + 1])
                i += 2
                continue
            if c == quote:
                in_lit = False
            i += 1
            continue
        if c in "\"'":
            in_lit = True
            quote = c
            out.append(c)
            i += 1
            continue
        if c.isspace():
            out.append(" ")
            while i < n and s[i].isspace():
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out).strip()


def unwrap_default_arg(s: str) -> str:
    """Unwrap a single DEFAULT_ARG(...) wrapper if it spans the whole string."""
    t = s.strip()
    if t.startswith("DEFAULT_ARG(") and t.endswith(")"):
        depth = 0
        for i, c in enumerate(t):
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0 and i != len(t) - 1:
                    return t  # closing paren is not the last char: not a pure wrapper
        return t[len("DEFAULT_ARG("):-1]
    return t


# ---------------------------------------------------------------- parsing

def split_args(masked_region: str, orig_region: str) -> list[str]:
    """Split a macro argument region on top-level commas.

    Depth tracking covers (), {} and <> (templates); string contents are
    already blanked in `masked_region`.
    """
    parts: list[str] = []
    depth_p = depth_b = depth_a = 0
    start = 0
    for i, c in enumerate(masked_region):
        if c == "(":
            depth_p += 1
        elif c == ")":
            depth_p -= 1
        elif c == "{":
            depth_b += 1
        elif c == "}":
            depth_b -= 1
        elif c == "<":
            depth_a += 1
        elif c == ">":
            if depth_a > 0:
                depth_a -= 1
        elif c == "," and depth_p == 0 and depth_b == 0 and depth_a == 0:
            parts.append(orig_region[start:i])
            start = i + 1
    parts.append(orig_region[start:])
    return [collapse_ws(p) for p in parts]


def balanced_paren(masked: str, open_idx: int) -> int:
    """Index of the ')' matching the '(' at open_idx (strings already blanked)."""
    depth = 0
    for i in range(open_idx, len(masked)):
        c = masked[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"unbalanced parens from offset {open_idx}")


def balanced_brace(masked: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(masked)):
        c = masked[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"unbalanced braces from offset {open_idx}")


class Invocation:
    __slots__ = ("name", "args", "line", "start")

    def __init__(self, name: str, args: list[str], line: int, start: int):
        self.name = name
        self.args = args
        self.line = line
        self.start = start


class ClassInfo:
    __slots__ = ("name", "base", "header", "line", "start", "end", "macros", "is_list_wrapper")

    def __init__(self, name, base, header, line, start, end):
        self.name = name
        self.base = base
        self.header = header
        self.line = line
        self.start = start
        self.end = end
        self.macros: list[Invocation] = []
        self.is_list_wrapper = False


def parse_header(path: Path):
    """Return (classes, list_types, free_macros) for one header."""
    text = path.read_text(encoding="utf-8")
    masked = mask_preprocessor(mask_comments_and_strings(text))

    invocations: list[Invocation] = []
    for m in MACRO_RE.finditer(masked):
        name = m.group(1)
        # next non-space char must be '('
        j = m.end()
        while j < len(masked) and masked[j].isspace():
            j += 1
        if j >= len(masked) or masked[j] != "(":
            continue
        close = balanced_paren(masked, j)
        args = split_args(masked[j + 1:close], text[j + 1:close])
        invocations.append(Invocation(name, args, line_of(text, m.start()), m.start()))

    classes: dict[str, ClassInfo] = {}
    for m in CLASS_RE.finditer(masked):
        name = m.group(1)
        base_txt = (m.group(2) or "").strip()
        bases = [b for b in re.findall(r"[A-Za-z_][A-Za-z0-9_:]*", base_txt) if b != "public"]
        base = bases[0] if bases else ""
        end = balanced_brace(masked, m.end() - 1)
        classes[name] = ClassInfo(name, base, path.name, line_of(text, m.start()), m.start(), end)

    # attribute macro invocations to the innermost containing class body
    free: list[Invocation] = []
    for inv in invocations:
        best: ClassInfo | None = None
        for ci in classes.values():
            if ci.start <= inv.start <= ci.end:
                if best is None or ci.start >= best.start:
                    best = ci
        if best is not None:
            best.macros.append(inv)
        else:
            free.append(inv)

    list_types: dict[str, str] = {}  # wrapper class -> element class
    for inv in free:
        if inv.name in LIST_TYPE_MACROS and len(inv.args) >= 2:
            list_types[inv.args[1]] = inv.args[0]

    # Cross-check CONFIG_NODE base vs the class declaration base
    for ci in classes.values():
        for inv in ci.macros:
            if inv.name in NODE_MACROS and len(inv.args) >= 2:
                if ci.base and ci.base != inv.args[1]:
                    # keep the CONFIG_NODE-declared base (authoritative for the macros)
                    ci.base = inv.args[1]
                else:
                    ci.base = inv.args[1]
    return classes, list_types, free


def parse_enums(path: Path) -> dict[str, list[str]]:
    """enums.hpp: ENUM(Name, Value, ...) namespace enums."""
    text = path.read_text(encoding="utf-8")
    masked = mask_preprocessor(mask_comments_and_strings(text))
    enums: dict[str, list[str]] = {}
    for m in re.finditer(r"(?<![A-Za-z0-9_])ENUM\s*\(", masked):
        j = m.end() - 1
        close = balanced_paren(masked, j)
        args = split_args(masked[j + 1:close], text[j + 1:close])
        if args:
            enums[args[0]] = args[1:]
    return enums


# ---------------------------------------------------------------- walking

class Row:
    __slots__ = ("root", "path", "kind", "type", "default", "global_only", "header",
                 "line", "cls", "macro", "element", "derived", "notes", "order")

    def __init__(self, root, path, kind, type_, default, global_only, header, line,
                 cls, macro, element, derived, notes, order):
        self.root = root
        self.path = path
        self.kind = kind
        self.type = type_
        self.default = default
        self.global_only = global_only
        self.header = header
        self.line = line
        self.cls = cls
        self.macro = macro
        self.element = element
        self.derived = derived
        self.notes = notes
        self.order = order

    def as_dict(self) -> dict:
        return {
            "root": self.root,
            "path": self.path,
            "kind": self.kind,
            "type": self.type,
            "default": self.default,
            "global_only": self.global_only,
            "header": self.header,
            "line": self.line,
            "class": self.cls,
            "macro": self.macro,
            "element_type": self.element,
            "derived": self.derived,
            "notes": self.notes,
            "order": self.order,
        }


class Enumerator:
    def __init__(self, classes, list_types, enums):
        self.classes = classes
        self.list_types = list_types
        self.enums = enums
        self.rows: list[Row] = []
        self.anomalies: list[str] = []
        self.mounted_headers: dict[str, set] = {}
        self._order = 0
        self._seen: set[tuple[str, str]] = set()

    def _next(self):
        self._order += 1
        return self._order

    def _class(self, name: str, where: str) -> ClassInfo | None:
        ci = self.classes.get(name)
        if ci is None:
            self.anomalies.append(f"class {name} not found in any Config header (referenced from {where})")
        return ci

    def walk(self, cls_name: str, prefix: str, global_parent: bool, root_label: str,
             derived: bool):
        key = (root_label, prefix or "(root)")
        if key in self._seen:
            self.anomalies.append(f"cycle/重复 mount avoided at {key}")
            return
        self._seen.add(key)
        ci = self._class(cls_name, f"mount {prefix or '(root)'}")
        if ci is None:
            return
        self.mounted_headers.setdefault(root_label, set()).add(ci.header)

        for inv in ci.macros:
            n = inv.name
            if n in NODE_MACROS or n in LIST_TYPE_MACROS:
                continue
            args = inv.args
            if n in SUBOBJECT_MACROS:
                if len(args) < 2:
                    self.anomalies.append(f"{n} with {len(args)} args at {ci.header}:{inv.line}")
                    continue
                typ, name = args[0], args[1]
                g = global_parent or n.endswith("GLOBAL_SUBOBJECT")
                path = f"{prefix}.{name}" if prefix else name
                self.rows.append(Row(root_label, path, "subobject", typ, "", g,
                                     ci.header, inv.line, cls_name, n, "", derived,
                                     "", self._next()))
                self.walk(typ, path, g, root_label, derived)
            elif n in LIST_MACROS:
                if len(args) < 2:
                    self.anomalies.append(f"{n} with {len(args)} args at {ci.header}:{inv.line}")
                    continue
                typ, name = args[0], args[1]
                default = unwrap_default_arg(args[2]) if len(args) > 2 else ""
                g = global_parent or n.endswith("GLOBAL_LIST")
                path = f"{prefix}.{name}" if prefix else name
                elem = typ
                note = ""
                eci = self.classes.get(self.list_types.get(typ, ""))
                if eci is not None:
                    eprops = sum(1 for x in eci.macros if x.name in (PROPERTY_MACROS | ENUM_PROPERTY_MACROS))
                    note = f"element class {eci.name} ({eci.header}:{eci.line}, {eprops} props, not enumerated as tree leaves)"
                else:
                    note = f"element list type {typ} (no element class record)"
                self.rows.append(Row(root_label, path, "list", typ, default, g,
                                     ci.header, inv.line, cls_name, n, elem, derived,
                                     note, self._next()))
            elif n in PROPERTY_MACROS or n in ENUM_PROPERTY_MACROS:
                if len(args) < 2:
                    self.anomalies.append(f"{n} with {len(args)} args at {ci.header}:{inv.line}")
                    continue
                typ, name = args[0], args[1]
                default = unwrap_default_arg(args[2]) if len(args) > 2 else ""
                # CONFIG_GLOBAL_ENUM_PROPERTY also ends "...ENUM_PROPERTY",
                # not "...GLOBAL_PROPERTY" — both spellings mean global-only.
                g = global_parent or n.endswith("GLOBAL_PROPERTY") \
                    or n.endswith("GLOBAL_ENUM_PROPERTY")
                path = f"{prefix}.{name}" if prefix else name
                notes = ""
                if n in ENUM_PROPERTY_MACROS:
                    typ = f"caelestia::config::{typ}::Enum"
                    vals = self.enums.get(args[0], [])
                    notes = f"enum {args[0]}: " + "|".join(vals) if vals else f"enum {args[0]} (values not found)"
                if len(args) > 3:
                    extra = ", ".join(args[3:])
                    notes = (notes + "; " if notes else "") + f"extra: {extra}"
                kind = "map" if typ == "QVariantMap" else "scalar"
                self.rows.append(Row(root_label, path, kind, typ, default, g,
                                     ci.header, inv.line, cls_name, n, "", derived,
                                     notes, self._next()))
            else:
                self.anomalies.append(f"unhandled macro {n} at {ci.header}:{inv.line}")


# ---------------------------------------------------------------- API

def enumerate_leaves(repo_root: str) -> list[dict]:
    """Walk the Config headers under ``repo_root`` and return every node row.

    Rows are dicts with the survey TSV's fields (see FIELDS). The config tree
    (root == "config", persisted as shell.json) and the derived tokens tree
    (root == "tokens", persisted as shell-tokens.json) are both included;
    callers filter by ``root``. Deterministic order: (root, path segments,
    declaration order).
    """
    repo = Path(repo_root)
    config_dir = repo / "shell" / "plugin" / "src" / "Caelestia" / "Config"
    if not config_dir.is_dir():
        raise FileNotFoundError(
            f"{config_dir} does not exist — run from a caelestia checkout")

    headers = sorted(config_dir.glob("*.hpp"))
    classes: dict[str, ClassInfo] = {}
    list_types: dict[str, str] = {}
    for h in headers:
        cls, lt, _free = parse_header(h)
        for name, ci in cls.items():
            classes[name] = ci
        for w, e in lt.items():
            list_types[w] = e

    enums = parse_enums(config_dir / "enums.hpp")

    en = Enumerator(classes, list_types, enums)

    # Root node rows + traversal
    en.rows.append(Row("config", "(root)", "subobject", "ConfigRoot", "", False,
                       "rootnodes.hpp", 34, "ConfigRoot", "CONFIG_NODE_NO_CTOR", "",
                       False, "root of the shell.json tree (GlobalConfig singleton / Config attached)", en._next()))
    en.walk("ConfigRoot", "", False, "config", derived=False)

    en.rows.append(Row("tokens", "(root)", "subobject", "TokensRoot", "", False,
                       "rootnodes.hpp", 70, "TokensRoot", "CONFIG_NODE_NO_CTOR", "",
                       True, "root of the shell-tokens.json tree (TokenConfig singleton); DERIVED/computed base values", en._next()))
    en.walk("TokensRoot", "", False, "tokens", derived=True)

    rows = sorted(en.rows, key=lambda r: (r.root, r.path.split("."), r.order))
    return [r.as_dict() for r in rows]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print the Caelestia config-leaf enumeration as TSV.")
    parser.add_argument("--repo-root", default=".",
                        help="caelestia-kde checkout root (default: current directory)")
    args = parser.parse_args()

    try:
        rows = enumerate_leaves(args.repo_root)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out = sys.stdout
    out.write("root\tpath\tkind\ttype\tdefault\tglobal_only\theader\tline\tclass\tmacro\telement_type\tderived\tnotes\n")
    for r in rows:
        fields = [r["root"], r["path"], r["kind"], r["type"], r["default"],
                  "yes" if r["global_only"] else "no", r["header"], str(r["line"]),
                  r["class"], r["macro"], r["element_type"],
                  "yes" if r["derived"] else "no", r["notes"]]
        fields = [x.replace("\t", " ").replace("\n", " ") for x in fields]
        out.write("\t".join(fields) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
