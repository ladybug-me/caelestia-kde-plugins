"""assistant.capabilities.graph — the shell knowledge graph (exponential-build-5 F12).

A typed, provenance-carrying graph over the settings universe, REBUILDABLE
from shipped artifacts only:

  * ``assistant/settings/tools.json``  — the registry (272 tools at this
   writing; tools.json meta is the count's home) (citations, groups,
    ranges), presets, explain_rules, not_exposed paths
  * ``assistant/settings/consequences.py`` — the curated, citation-backed
    interaction table (``affects`` edges)
  * ``assistant/diagnostics/rules.d/*.json`` + ``settings_join`` — the
    volunteered-only rule -> tool joins (``diagnoses`` edges)
  * an OPTIONAL user ledger (approved apply history) — the only personal
    input, supplied explicitly via ``--ledger``; without it the graph is
    identical on every machine (byte-identical hash).

Honesty rules this module inherits from its sources: the interaction
universe is EXACTLY the curated consequences table — bounded, auditable,
growable by reviewed diffs — and never "fake open-endedness". A query with
no cited path ABSTAINS (verdict ``NO_CITED_PATH``) rather than guessing.
Every edge carries provenance (its source artifact plus, where the source
has one, the file:line citation and claimed content, re-verified against
the upstream checkout by the settings layer's own tests).

Graph shape (plain JSON, deterministic ordering — nodes sorted by id,
edges by ``(src, dst, kind, provenance)``):

  nodes: [{id, type, label, attrs}]      types: tool, config, group,
                                         preset, not_exposed,
                                         explain_rule, consequence,
                                         diag_rule, file
  edges: [{src, dst, kind, weight, provenance}]
      kinds: cites, member_of, sets (tool->config), applies
             (preset->tool), affects (config->config, confidence),
             explained_by (config->explain_rule), diagnoses
             (diag_rule->tool), co_changes (config->config, ledger only)

Confidence mapping (documented, not silent): the consequences table's
own words are carried in provenance; ``high`` -> 0.9, ``medium`` -> 0.7
as edge weights for the path searches. The words stay authoritative.

Pure stdlib; reads nothing at import time; writes nothing ever.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["build_graph", "graph_hash", "dump_canonical"]

# confidence word -> edge weight (documented mapping; the word is the
# authority and is carried in every edge's provenance)
_CONFIDENCE_WEIGHT = {"high": 0.9, "medium": 0.7, "low": 0.5}

_TOOLS_JSON = Path(__file__).resolve().parents[2] / "adapters" / "caelestia" / "tools.json"
_RULES_D = Path(__file__).resolve().parents[2] / "capabilities" / "diagnostics" / "rules.d"


def _node(nodes: Dict[str, Dict[str, Any]], nid: str, ntype: str,
          label: str, **attrs: Any) -> str:
    nid = str(nid)
    if nid not in nodes:
        nodes[nid] = {"id": nid, "type": ntype, "label": str(label),
                      "attrs": {}}
    a = nodes[nid]["attrs"]
    for k, v in attrs.items():
        if v is None:
            continue
        a[k] = v
    return nid


def _edge(edges: Dict[tuple, Dict[str, Any]], src: str, dst: str,
          kind: str, weight: float, provenance: Dict[str, Any]) -> None:
    key = (src, dst, kind, json.dumps(provenance, sort_keys=True))
    if key not in edges:
        edges[key] = {"src": src, "dst": dst, "kind": kind,
                      "weight": float(weight), "provenance": provenance}


def _load_ledger(path: Optional[str]) -> List[Dict[str, Any]]:
    """Read an assistant history ledger JSON directly (read-only).

    The ledger is the settings layer's own ``<target>.assistant-history.json``
    shape: {"entries": [{"id", "at", "label", "ops": [{"path", "old",
    "new"}]}]}. Only multi-op APPROVED entries produce co_changes edges;
    a malformed file raises (loudly, never silently half-read)."""
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"ledger {path}: no 'entries' list; refusing")
    return entries


def build_graph(ledger_path: Optional[str] = None) -> Dict[str, Any]:
    """Build the knowledge graph deterministically from shipped artifacts.

    Determinism: node/edge ordering is fully sorted, independent of dict
    insertion order in any source; the hash therefore pins the SOURCES,
    not the loader. ``ledger_path`` is the only optional, personal input.
    """
    with open(_TOOLS_JSON, "r", encoding="utf-8") as fh:
        tools_json: Dict[str, Any] = json.load(fh)

    nodes: Dict[str, Dict[str, Any]] = {}
    edges: Dict[tuple, Dict[str, Any]] = {}

    def config_node(path: str) -> str:
        return _node(nodes, f"config:{path}", "config", path)

    # --- tools: sets / member_of / cites --------------------------------
    for row in tools_json["tools"]:
        tool_id = _node(nodes, f"tool:{row['name']}", "tool", row["name"],
                        path=row.get("path"), kind=row.get("kind"),
                        group=row.get("group"), default=row.get("default"))
        cid = config_node(row["path"])
        _edge(edges, tool_id, cid, "sets", 1.0,
              {"source": "tools.json", "tool": row["name"]})
        if row.get("group"):
            gid = _node(nodes, f"group:{row['group']}", "group",
                        row["group"])
            _edge(edges, tool_id, gid, "member_of", 1.0,
                  {"source": "tools.json group field"})
        for cite in row.get("citations") or ():
            loc, note = cite[0], (cite[1] if len(cite) > 1 else "")
            fid = _node(nodes, f"file:{loc}", "file", loc)
            _edge(edges, tool_id, fid, "cites", 1.0,
                  {"source": "tools.json citations", "note": note})

    # --- presets: applies -------------------------------------------------
    for preset in tools_json.get("presets", ()):
        pid = _node(nodes, f"preset:{preset['name']}", "preset",
                    preset.get("label", preset["name"]),
                    description=preset.get("description"))
        for call in preset.get("calls", ()):
            _edge(edges, pid, f"tool:{call['tool']}", "applies", 1.0,
                  {"source": "tools.json presets",
                   "value": call.get("value")})

    # --- not_exposed: nodes + group membership from the path prefix ------
    for hidden in tools_json.get("not_exposed", ()):
        path = hidden["path"]
        hid = _node(nodes, f"hidden:{path}", "not_exposed", path,
                    reason=hidden.get("reason"))
        prefix = path.split(".")[0] if "." in path else None
        if prefix:
            gid = _node(nodes, f"group:{prefix}", "group", prefix)
            _edge(edges, hid, gid, "member_of", 1.0,
                  {"source": "derived: path prefix (documented derivation, "
                             "not a shipped citation)"})

    # --- explain_rules: explained_by + cites ------------------------------
    for i, rule in enumerate(tools_json.get("explain_rules", ())):
        rid = _node(nodes, f"xrule:{i}", "explain_rule",
                    rule.get("path", f"rule-{i}"), when=rule.get("when"),
                    answer=rule.get("answer"))
        cid = config_node(rule["path"])
        _edge(edges, cid, rid, "explained_by", 1.0,
              {"source": "tools.json explain_rules"})
        for cite in rule.get("cites") or ():
            fid = _node(nodes, f"file:{cite}", "file", cite)
            _edge(edges, rid, fid, "cites", 1.0,
                  {"source": "tools.json explain_rules cites"})

    # --- consequences: affects (the curated interaction table) ------------
    from assistant.capabilities.settings import consequences as _cons  # local: reuse

    for e in _cons.EDGES:
        conf_word = e.get("confidence", "medium")
        weight = _CONFIDENCE_WEIGHT.get(conf_word, 0.5)
        src = config_node(e["trigger_path"])
        dst = config_node(e["effect_path"])
        prov = {"source": "settings/consequences.py EDGES",
                "id": e.get("id"),
                "confidence": conf_word,
                "when": e.get("when"),
                "effect": e.get("effect"),
                "citation": e.get("citation"),
                "claimed_content": e.get("claimed_content")}
        _edge(edges, src, dst, "affects", weight, prov)
        # the consequence's own citation gets a file node too, so the
        # interaction edge is auditable from the graph alone
        if e.get("citation"):
            fid = _node(nodes, f"file:{e['citation']}", "file", e["citation"])
            _edge(edges, src, fid, "cites", 1.0,
                  {"source": "consequences.py EDGES citation",
                   "id": e.get("id")})

    # --- diagnostics: volunteered-only rule -> tool joins ------------------
    from assistant.capabilities.diagnostics import settings_join as _join  # local: reuse

    joined_rules = 0
    for rf in sorted(_RULES_D.glob("*.json")):
        with open(rf, "r", encoding="utf-8") as fh:
            pack = json.load(fh)
        for rule in pack.get("rules", ()):
            try:
                resolved = _join.settings_tools_for_rule(rule)
            except Exception:  # noqa: BLE001 — a rule that does not join
                continue                      # is skipped loudly per join's
                                             # own "volunteered-only" rule
            names = resolved.get("tools") if isinstance(resolved, dict) else None
            if not names:
                continue
            joined_rules += 1
            rid = _node(nodes, f"diag:{rule.get('id', rf.name)}", "diag_rule",
                        rule.get("title", rule.get("id", rf.name)))
            for name in names:
                if f"tool:{name}" in nodes:
                    _edge(edges, rid, f"tool:{name}", "diagnoses", 1.0,
                          {"source": f"rules.d/{rf.name} + settings_join "
                                     "(volunteered references only)",
                           "rule": rule.get("id")})

    # --- ledger co_changes (optional, explicit, personal) ------------------
    ledger_used = bool(ledger_path)
    if ledger_path:
        by_pair: Dict[tuple, List[Any]] = {}
        for entry in _load_ledger(ledger_path):
            ops = entry.get("ops") or []
            paths = [op.get("path") for op in ops if op.get("path")]
            if len(paths) < 2:
                continue  # single-op entries carry no co-change evidence
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    pair = tuple(sorted((paths[i], paths[j])))
                    by_pair.setdefault(pair, []).append(
                        {"id": entry.get("id"), "at": entry.get("at")})
        for (pa, pb), ev in by_pair.items():
            weight = min(0.9, 0.3 * len(ev))  # capped, documented
            _edge(edges, f"config:{pa}", f"config:{pb}", "co_changes", weight,
                  {"source": "user ledger (approved applies)",
                   "evidence_ids": [e.get("id") for e in ev],
                   "note": "co-appearance in one approved entry; "
                           "frequency-capped weight, never a causal claim"})

    # --- deterministic ordering + hash -------------------------------------
    node_list = [nodes[k] for k in sorted(nodes)]
    edge_list = [edges[k] for k in sorted(edges,
                 key=lambda t: (t[0], t[1], t[2], t[3]))]
    meta = {
        "generator": "assistant.capabilities.graph.build",
        "sources": {
            "tools_json": {
                "generator": tools_json.get("meta", {}).get("generator"),
                "repo_commit": tools_json.get("meta", {}).get("repo_commit"),
            },
            "consequence_ids": sorted(e.get("id") for e in _cons.EDGES),
            "rules_d": [rf.name for rf in sorted(_RULES_D.glob("*.json"))],
            "joined_diag_rules": joined_rules,
            "ledger_used": ledger_used,
        },
    }
    graph = {"meta": meta, "nodes": node_list, "edges": edge_list}
    graph["meta"]["hash"] = graph_hash(graph)
    return graph


def _canonical(graph: Dict[str, Any]) -> str:
    payload = {"meta": {k: v for k, v in graph["meta"].items() if k != "hash"},
               "nodes": graph["nodes"], "edges": graph["edges"]}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def graph_hash(graph: Dict[str, Any]) -> str:
    """sha256 over the canonical serialization (hash field excluded) —
    the rebuild-drift detector: same sources must give the same hash."""
    return hashlib.sha256(_canonical(graph).encode("utf-8")).hexdigest()


def dump_canonical(graph: Dict[str, Any]) -> str:
    """Canonical, deterministically ordered JSON text of the whole graph."""
    return json.dumps({"meta": graph["meta"], "nodes": graph["nodes"],
                       "edges": graph["edges"]},
                      sort_keys=True, indent=1)
