"""agent.archetype_pack — the goal-archetype marketplace format
(exponential-build-4 G), extending the signed-rule-pack distribution
model (diagnostics/rulepack.py) to the agent's HTN goal archetypes.

TREATED AS MATERIALLY HIGHER RISK THAN A RULE PACK, and the module is
built like it: a rule pack shares FACTS (symptom -> fix text, inert);
an archetype describes ACTIONS — HTN nodes bound to the agent's
dispatchers. The import gate is therefore structural, not advisory:

  1. INTEGRITY — the manifest's payload_sha256 must equal the sha256
     of the canonical serialization of the embedded payload (the
     rulepack contract verbatim);
  2. ACTION WHITELIST — every node's ``action`` must resolve to an
     ALREADY-DECLARED dispatcher in this installation
     (agent.engine.default_dispatchers()). Nothing freeform: an
     archetype cannot name a dispatcher that does not exist, and
     cannot bring its own executors;
  3. NO CODE CAN ENTER — an archetype document is a JSON data
     document with a CLOSED field set (id, title, description,
     nodes; node fields: id, action, title, risk, consent, params,
     depends, required_capability). Unknown fields are rejected, so
     there is no field a payload of code could hide in; the repo-wide
     AST import lint continues to guard the Python side;
  4. RISK + CONSENT — node risk tiers must be valid
     (diagnostics.risk.RISK_ORDER), and every node at
     STATE_CHANGING or above must carry consent=True, mirroring the
     engine's own discipline;
  5. CAPABILITY — a node may declare ``required_capability``; the
     name must be one of the manifest's known capabilities
     (capabilities.DEFAULTS). The import NEVER grants it: the record
     says "required, NOT granted", and the engine's own
     capabilities.enabled check applies at execution exactly as it
     does for built-in archetypes. No imported archetype silently
     inherits any grant an existing archetype already has;
  6. REVIEW-ONLY — imported archetypes land in the learned-state
     JSON under archetype_pack.STATE_KEY with status "review-only".
     They are NOT registered into the engine's dispatch table by
     import; registration/promotion is a human decision this module
     deliberately does not implement (see OPEN_QUESTIONS).

SIGNING HAPPENS OUTSIDE, identically to the rulepack and lexicon-diff
contracts: the exporter writes canonical bytes; the sharer detaches a
signature with their own tool; the recipient verifies BEFORE import.
The assistant's part is integrity + structure, not cryptography.

OPEN_QUESTIONS (underspecified for a human to decide — recorded, not
guessed at): see the OPEN_QUESTIONS constant below.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Union

from assistant.capabilities.diagnostics import risk as risk_mod
from assistant.capabilities.diagnostics.rulepack import payload_sha256
from assistant.capabilities.diagnostics.schema_lint import validate_rule_safety  # noqa: F401
from .engine import default_dispatchers

__all__ = ["STATE_KEY", "ArchetypePackError", "export_pack",
           "import_pack", "list_imported", "render_archetype",
           "OPEN_QUESTIONS"]

STATE_KEY = "imported_archetype_packs"

#: fields an archetype document may carry (CLOSED set)
_ARCHETYPE_FIELDS = {"id", "title", "description", "nodes"}
#: fields an archetype NODE may carry (CLOSED set)
_NODE_FIELDS = {"id", "action", "title", "risk", "consent", "params",
                "depends", "required_capability"}


class ArchetypePackError(ValueError):
    """Any pack-format, integrity, or structural-safety refusal."""


OPEN_QUESTIONS = [
    "What does PROMOTION from review-only to registered look like? "
    "Options: (a) a reviewed archetype file committed to the repo, "
    "(b) an interactive per-archetype consent flow, (c) a separate "
    "explicit CLI verb with typed confirmation. Import deliberately "
    "implements none of them — a human must pick, because promotion "
    "is the step where an archetype gains the right to run.",
    "Should imported archetypes be scored/rated by the community "
    "infrastructure (reputation-weighted lexicon trust exists for "
    "phrases), or does reputation-transfer across action vocabularies "
    "import trust that does not generalize? Not decided here.",
    "Min-cut/bounded-model-check auditing of an imported archetype's "
    "node graph is available (genius/sat.graph_audit) and cheap — "
    "should a CLEAN audit be REQUIRED for import, or reported only? "
    "Currently reported only (an audit failure blocks nothing, which "
    "may be too soft; that call belongs to a human).",
]


def _validate_archetype(doc: Any, index: int,
                        dispatchers: Dict[str, Any]) -> List[str]:
    """The structural gate on one archetype document. Returns the list
    of refusal reasons (empty = accepted)."""
    failures: List[str] = []
    where = f"archetype[{index}]"
    if not isinstance(doc, dict):
        return [f"{where}: not a JSON object"]
    unknown = set(doc) - _ARCHETYPE_FIELDS
    if unknown:
        failures.append(
            f"{where}: unknown field(s) {sorted(unknown)} — the field "
            "set is CLOSED so no code can hide in an unspecified key")
    if not isinstance(doc.get("id"), str) or not doc["id"]:
        failures.append(f"{where}: 'id' must be a non-empty string")
    nodes = doc.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        failures.append(f"{where}: 'nodes' must be a non-empty list")
        return failures
    for j, node in enumerate(nodes):
        nwhere = f"{where}.node[{j}]"
        if not isinstance(node, dict):
            failures.append(f"{nwhere}: not a JSON object")
            continue
        unknown_node = set(node) - _NODE_FIELDS
        if unknown_node:
            failures.append(
                f"{nwhere}: unknown field(s) {sorted(unknown_node)}")
        action = node.get("action")
        if not isinstance(action, str) or action not in dispatchers:
            failures.append(
                f"{nwhere}: action {action!r} does not resolve to an "
                "already-declared dispatcher in this installation — "
                "an archetype cannot name an executor that does not "
                "exist here, and cannot bring its own")
        risk = node.get("risk")
        if risk not in risk_mod.RISK_ORDER:
            failures.append(
                f"{nwhere}: risk {risk!r} is not a valid tier "
                f"(known: {sorted(risk_mod.RISK_ORDER)})")
        elif risk_mod.RISK_ORDER[risk] >= risk_mod.RISK_ORDER["STATE_CHANGING"] \
                and node.get("consent") is not True:
            failures.append(
                f"{nwhere}: risk {risk} requires consent=true — "
                "imported archetypes follow the engine's own consent "
                "discipline, they do not relax it")
        cap = node.get("required_capability")
        if cap is not None:
            from assistant.core.features import DEFAULTS
            if cap not in DEFAULTS:
                failures.append(
                    f"{nwhere}: required_capability {cap!r} is not a "
                    "known capability — import never grants it, and an "
                    "unknown capability cannot even be checked")
        params = node.get("params", {})
        if not isinstance(params, dict):
            failures.append(f"{nwhere}: params must be a JSON object")
        depends = node.get("depends", [])
        if not isinstance(depends, list) or not all(
                isinstance(d, str) for d in depends):
            failures.append(f"{nwhere}: depends must be a list of ids")
    return failures


def export_pack(archetypes: Sequence[Dict[str, Any]], pack_id: str
                ) -> Dict[str, Any]:
    """Canonical pack bytes for the sharer to sign OUTSIDE the
    assistant. The payload is validated by the same structural gate an
    importer will run (an export that would not import is refused
    here, with the same reasons)."""
    if not isinstance(pack_id, str) or not pack_id:
        raise ArchetypePackError("pack_id must be a non-empty string")
    dispatchers = default_dispatchers()
    failures: List[str] = []
    for i, doc in enumerate(archetypes):
        failures.extend(_validate_archetype(doc, i, dispatchers))
    if failures:
        raise ArchetypePackError(
            "export refused: the pack would not pass its own import "
            "gate — " + "; ".join(failures))
    payload = [dict(doc) for doc in archetypes]
    return {
        "schema_version": 1,
        "pack": {"id": pack_id,
                 "archetype_count": len(payload),
                 "payload_sha256": payload_sha256(payload)},
        "payload": payload,
    }


def import_pack(raw: Union[str, bytes], existing_ids: Sequence[str],
                signer: str = "") -> Dict[str, Any]:
    """Import one pack FILE (canonical JSON text): integrity, then the
    structural gate, then record as REVIEW-ONLY data. Returns the
    record for the caller's learned-state JSON. Raises
    ArchetypePackError with every failure listed — partial imports are
    impossible (all-or-nothing, like the rulepack)."""
    import json as _json

    try:
        pack = _json.loads(raw)
    except _json.JSONDecodeError as exc:
        raise ArchetypePackError(f"not valid JSON: {exc}") from exc
    if not isinstance(pack, dict) or pack.get("schema_version") != 1:
        raise ArchetypePackError("schema_version must be 1")
    manifest = pack.get("pack") or {}
    payload = pack.get("payload")
    if not isinstance(payload, list):
        raise ArchetypePackError("payload must be a list of archetypes")
    declared = manifest.get("payload_sha256")
    if not isinstance(declared, str) or payload_sha256(payload) != declared:
        raise ArchetypePackError(
            "integrity failure: payload_sha256 does not match the "
            "embedded payload — the pack was edited after its manifest "
            "was written; importing is refused")
    pack_id = manifest.get("id")
    if not isinstance(pack_id, str) or not pack_id:
        raise ArchetypePackError("pack.id must be a non-empty string")
    if pack_id in set(existing_ids):
        raise ArchetypePackError(
            f"pack id {pack_id!r} already imported — re-importing the "
            "same id would silently replace reviewed data")
    dispatchers = default_dispatchers()
    failures: List[str] = []
    for i, doc in enumerate(payload):
        failures.extend(_validate_archetype(doc, i, dispatchers))
    if failures:
        raise ArchetypePackError(
            "import refused — " + "; ".join(failures))
    # the bounded model-check audit is REPORTED, never a gate (per
    # OPEN_QUESTIONS item 3 — whether a CLEAN audit should be required
    # for import is a human decision this module does not make)
    audits: List[Dict[str, Any]] = []
    try:
        from assistant.capabilities.genius import sat as sat_mod
        for doc in payload:
            nodes = doc.get("nodes", [])
            graph = {n["id"]: list(n.get("depends", [])) for n in nodes
                     if isinstance(n, dict) and "id" in n}
            root = graph and next(iter(sorted(graph)))
            if root and len(graph) > 1:
                audit = sat_mod.graph_audit(
                    graph, root, k=max(2, len(graph)), node_budget=20_000)
                audits.append({"archetype": doc.get("id"),
                               "status": audit["status"],
                               "unreachable_within_k":
                                   audit["unreachable_within_k"]})
    except Exception as exc:  # the audit must never block an import
        audits.append({"status": "ERROR",
                       "reason": f"{type(exc).__name__}: {exc}"})
    return {
        "id": pack_id,
        "status": "review-only",
        "signer": signer,
        "signer_is_claim": True,
        "archetype_count": len(payload),
        "archetypes": payload,
        "capabilities_granted": [],
        "graph_audits": audits,
        "note": "review data, NOT registered into the dispatcher table; "
                "every node re-checks capabilities.enabled at execution "
                "exactly like the built-ins; see OPEN_QUESTIONS for the "
                "promotion decisions a human still owes",
    }


def list_imported(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The imported packs (review data) from the learned-state JSON."""
    return [dict(rec) for rec in state.get(STATE_KEY, [])]


def render_archetype(doc: Dict[str, Any]) -> List[str]:
    """A human-readable review card for one archetype — what a user
    reads BEFORE deciding anything. Plain text, nothing executable."""
    lines: List[str] = [
        f"archetype {doc.get('id', '?')}: {doc.get('title', '')}",
        f"  {doc.get('description', '')}",
    ]
    for node in doc.get("nodes", []):
        cap = node.get("required_capability")
        cap_note = f" [capability required, NOT granted: {cap}]" if cap else ""
        consent = "consent-gated" if node.get("consent") else "no consent"
        lines.append(f"  - {node.get('id', '?')}: {node.get('action')} "
                     f"[{node.get('risk', '?')}, {consent}]{cap_note}")
    return lines
