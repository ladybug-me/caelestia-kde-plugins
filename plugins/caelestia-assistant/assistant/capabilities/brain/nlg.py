"""brain.nlg — template natural-language generation with discourse
planning and aggregation (D16, Group D).

The assistant already says things; this module is about saying LIST
things well. A plan with nine changes reads terribly as nine
sentences in a row ("the bar scale will change. the bar position will
change. the ..."). Human writers aggregate:

  - COARSE-TO-FINE discourse plan: group the changes by tool GROUP
    (bar, dock, notifications, ...), order the groups deterministically
    (by first appearance, then alphabetically), put a summary sentence
    first, details after;
  - AGGREGATION with counting cowards: "2 changes to the bar
    (scale, position)" instead of naming the same group twice;
  - SYNTACTIC fusion: same-field messages merge ("3 spacing changes
    and 2 color changes"), hedged honestly when counts are one
    ("1 spacing change" — never "1 changes");

The output is deterministic: same plan in, same sentences out. No
model, no randomness — templates with real morphology (plural/singular)
and a document plan, which is what "NLG" meant before LLMs and still
is what most production systems need.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

__all__ = ["plan_document", "aggregate", "render_plan_paragraph",
           "render_changes_summary"]

# morphological helper ------------------------------------------------------


def _plural(count: int, noun: str, plural_form: Optional[str] = None) \
        -> str:
    """'1 change' / '3 changes'; irregular forms via plural_form."""
    if count == 1:
        return f"1 {noun}"
    return f"{count} {plural_form or noun + 's'}"


def _verb(count: int, singular: str, plural: str) -> str:
    return singular if count == 1 else plural


# ---------------------------------------------------------------------------
# discourse planning: group -> order -> summarize
# ---------------------------------------------------------------------------

def plan_document(changes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Coarse-to-fine plan: group by the tool's group (or an explicit
    'group' key), keep a deterministic order, count per group.

    Each change is a dict with at least 'tool' (a registry tool name or
    dotted path); 'group' overrides; 'field' names the human aspect
    (spacing, color, ...) when the caller has one. A bare string is
    tolerated and treated as a tool name — hostile input becomes an
    honest group, never an AttributeError.
    """
    changes = [c if isinstance(c, dict) else {"tool": str(c)}
               for c in changes]
    groups: Dict[str, List[Dict[str, Any]]] = {}
    group_order: List[str] = []
    for change in changes:
        group = str(change.get("group") or _group_of(change))
        if group not in groups:
            groups[group] = []
            group_order.append(group)
        groups[group].append(change)
    # deterministic: sort within a group; groups keep first-appearance
    # order, ties impossible after that
    planned = []
    for group in group_order:
        members = sorted(groups[group],
                         key=lambda c: (str(c.get("tool", "")),
                                        str(c.get("field", "")),
                                        json_key(c)))
        planned.append({"group": group, "n": len(members),
                        "changes": members})
    return {"n_changes": len(changes), "n_groups": len(planned),
            "groups": planned}


def json_key(change: Dict[str, Any]) -> str:
    import json
    return json.dumps(change, sort_keys=True)


def _group_of(change: Dict[str, Any]) -> str:
    """The tool's group from a registry name (setBarScale -> bar); a
    dotted path uses its first segment. Unknown names group as 'other'
    — honest grouping beats a guessed one."""
    tool = str(change.get("tool", ""))
    if not tool:
        return "other"
    if "." in tool:
        return tool.split(".", 1)[0]
    from assistant.core.lexicon import camel_split
    words = [w.lower() for w in camel_split(tool)
             if w.lower() not in ("set", "enable", "disabled", "toggle")]
    return words[0] if words else "other"


# ---------------------------------------------------------------------------
# aggregation: count and fuse
# ---------------------------------------------------------------------------

def aggregate(planned: Dict[str, Any]) -> List[str]:
    """Per-group aggregated sentences: '2 spacing changes and 1 color
    change in the bar'. Fields fuse into counting clauses; the noun
    carried is 'change', so mass-noun fields (spacing, color) never
    get an ugly plural."""
    sentences: List[str] = []
    for group in planned["groups"]:
        fields: Dict[str, int] = {}
        for change in group["changes"]:
            field = change.get("field")
            if field:
                fields[str(field)] = fields.get(str(field), 0) + 1
        if fields:
            parts = [_plural(n, f"{field} change") for field, n in
                     sorted(fields.items(), key=lambda kv: (-kv[1], kv[0]))]
            body = ", ".join(parts[:-1])
            if len(parts) > 1:
                body += f" and {parts[-1]}"
            else:
                body = parts[0]
            sentences.append(f"{_plural(group['n'], 'change')} in the "
                             f"{group['group']}: {body}")
        else:
            names = [str(c.get("tool", "?")) for c in group["changes"]]
            shown = ", ".join(names[:3])
            if len(names) > 3:
                shown += f" and {_plural(len(names) - 3, 'more')}"
            sentences.append(f"{_plural(group['n'], 'change')} in the "
                             f"{group['group']} ({shown})")
    return sentences


def render_changes_summary(changes: Sequence[Dict[str, Any]]) -> List[str]:
    """The whole summary: one lead sentence, then aggregated lines."""
    if not changes:
        return ["no changes to make."]
    planned = plan_document(changes)
    n = planned["n_changes"]
    lead = (f"this plan makes {_plural(n, 'change')} across "
            f"{_plural(planned['n_groups'], 'area')}:")
    return [lead] + [f"  {line}" for line in aggregate(planned)]


# ---------------------------------------------------------------------------
# the paragraph surface: full document rendering
# ---------------------------------------------------------------------------

def render_plan_paragraph(changes: Sequence[Dict[str, Any]],
                          action: str = "This plan will") -> str:
    """One paragraph: lead + fused clause enumeration. Deterministic;
    every sentence ends; no run-ons. A single change collapses to one
    clause with the tool named — no '1 change: 1 change' echo."""
    if not changes:
        return "Nothing to change."
    planned = plan_document(changes)
    n = planned["n_changes"]
    if n == 1:
        only = planned["groups"][0]["changes"][0]
        group = planned["groups"][0]["group"]
        tool = str(only.get("tool", "unknown"))
        return f"{action} make 1 change to the {group} " \
            f"({tool})."
    clauses: List[str] = []
    for group in planned["groups"]:
        clauses.append(f"{_plural(group['n'], 'change')} to the "
                       f"{group['group']}")
    body = ", ".join(clauses[:-1])
    if len(clauses) > 1:
        body += f" and {clauses[-1]}"
    else:
        body = clauses[0]
    return f"{action} make {_plural(n, 'change')}: {body}."
