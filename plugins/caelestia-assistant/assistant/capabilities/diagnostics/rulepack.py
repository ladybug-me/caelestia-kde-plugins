"""diagnostics.rulepack — the signed rule-pack format (exponential-
build-3 F1): share troubleshooting rules WITHOUT giving anyone a way
to smuggle auto-execution past the fences.

THE FORMAT. A pack is one JSON document:

    {"schema_version": 1,
     "pack": {"id": "...", "rulesets": [...], "rule_count": N,
              "payload_sha256": "<sha256 hex of the canonical payload>"},
     "payload": [<rules.d documents, verbatim>]}

The payload is the SAME document shape assistant/diagnostics/rules.d
ships (schema_version / ruleset / provenance / rules), so a pack is
literally a bundle of existing ruleset files plus a manifest.

SIGNING HAPPENS OUTSIDE — deliberately, and identically to the
lexicon-diff contract (cortex/lexicon_diff.py): the assistant does no
cryptography. The exporter writes canonical, byte-stable bytes; the
SHARER detaches an Ed25519 signature with a tool they already trust
(minisign / sq / gpg) over the pack FILE; the RECIPIENT verifies that
signature with their own tool BEFORE importing. The assistant's part
of the contract is two independent checks it CAN do honestly:

- INTEGRITY (in-module): the manifest's payload_sha256 must equal the
  sha256 of the canonical serialization of the embedded payload, so a
  pack whose rules were edited after the manifest was written cannot
  import even if nobody checks the signature;
- SAFETY (in-module): every rule in the payload passes the SAME
  structural validation the built-in rulesets pass at load time
  (engine.validate_rule, including the schema checks) plus the safety
  lint the selfcheck runs (schema_lint.validate_rule_safety, including
  the forbidden-key scan that makes auto-execution inexpressible).
  A pack that fails ANY check is refused whole — no partial imports.

IDENTITY IS A CLAIM, NOT A FACT: ``--signer`` records the name the
user verified with their external tool, exactly like lexicon import;
trust in that name stays a human judgment the assistant never makes.

CONSENT: importing a pack stores it in the brain state as REVIEWABLE
DATA under "imported_rule_packs" (the existing state write path —
no new file, no new mechanism). Imported rules are deliberately NOT
live: the engine loads rules.d only, and wiring community rules into
live matching is a maintainer policy decision, not an import side
effect. ``render`` prints a pack's rules as a rules.d-ready document;
placing that file into rules.d is a human action and stays one.

Pure module: no execution, no network, no RNG, no I/O beyond the
explicit paths its callers pass — canonical bytes are deterministic
for fixed input (pinned by test).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

__all__ = ["RulePackError", "canonical_bytes", "payload_sha256",
           "export_pack", "import_pack", "list_packs", "render_pack",
           "render_rules_d"]

PACK_SCHEMA_VERSION = 1
STATE_KEY = "imported_rule_packs"


class RulePackError(ValueError):
    """Every refusal (bad pack shape, hash mismatch, safety failure,
    duplicate id, unknown ruleset) — the caller renders the reason;
    nothing is guessed and nothing is partially imported."""


# ---------------------------------------------------------------------------
# canonical bytes + integrity
# ---------------------------------------------------------------------------

def canonical_bytes(payload: Any) -> bytes:
    """The byte-stable serialization a signature can rest on:
    sorted keys, no insignificant whitespace, ASCII-escaped, UTF-8.
    Same input -> same bytes, always (pinned by test)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


# ---------------------------------------------------------------------------
# export (build a shareable pack from shipped rulesets)
# ---------------------------------------------------------------------------

def export_pack(rules_dir: Union[str, Path], rulesets: Sequence[str],
                pack_id: str, source_note: str = ""
                ) -> Dict[str, Any]:
    """Bundle the named rulesets (rules.d file STEMS, e.g.
    ``"config_kde"``) into a pack document. Every bundled ruleset is
    read from ``rules_dir`` and every rule validated through the
    ordinary structural + safety gates FIRST — an exporter ships only
    rules that would survive import themselves. Unknown stems are
    refused with the directory's actual listing."""
    if not pack_id or not isinstance(pack_id, str):
        raise RulePackError("pack_id must be a non-empty string")
    directory = Path(rules_dir)
    available = sorted(p.stem for p in directory.glob("*.json"))
    missing = [name for name in rulesets if name not in available]
    if missing:
        raise RulePackError(
            f"unknown ruleset stem(s) {missing}; available in "
            f"{directory}: {available}")
    from .engine import validate_rule
    from . import schema_lint
    payload: List[Dict[str, Any]] = []
    rule_count = 0
    for stem in rulesets:
        path = directory / f"{stem}.json"
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RulePackError(
                f"ruleset {stem!r} cannot be read ({exc})") from exc
        if doc.get("schema_version") != 1:
            raise RulePackError(
                f"ruleset {stem!r}: unsupported schema_version "
                f"{doc.get('schema_version')!r}")
        failures: List[str] = []
        for rule in doc.get("rules", []):
            failures.extend(validate_rule(rule, source=str(path)))
            failures.extend(
                schema_lint.validate_rule_safety(rule, str(path)))
        if failures:
            raise RulePackError(
                f"ruleset {stem!r} failed validation and cannot be "
                f"exported: {'; '.join(failures[:3])}")
        payload.append(doc)
        rule_count += len(doc.get("rules", []))
    manifest = {
        "id": pack_id,
        "rulesets": list(rulesets),
        "rule_count": rule_count,
        "payload_sha256": payload_sha256(payload),
        "source_note": str(source_note or ""),
    }
    return {"schema_version": PACK_SCHEMA_VERSION, "pack": manifest,
            "payload": payload}


# ---------------------------------------------------------------------------
# import (verify + store as reviewable data)
# ---------------------------------------------------------------------------

def import_pack(pack: Union[str, Path, Dict[str, Any]],
                existing_ids: Optional[Sequence[str]] = None,
                signer: str = "") -> Dict[str, Any]:
    """Verify one pack document and return the RECORD to store (the
    caller persists it in the brain state — this module writes
    nothing). Accepts a path, the pack's JSON text, or the parsed
    dict. Checks, in order: document shape; payload hash INTEGRITY
    (manifest sha256 == sha256 of canonical payload bytes); every
    ruleset's schema_version; every rule's STRUCTURAL validity and
    SAFETY lint (the same gates the built-ins pass); duplicate pack
    id (refused — delete first, never a silent overwrite). Returns
    the record: the manifest plus "payload", the claimed ``signer``
    (identity is a claim verified outside), and the validation stamp
    (sha256 of the whole canonical pack — the audit trail)."""
    if isinstance(pack, (str, Path)):
        text = str(pack)
        looks_like_json = text.lstrip().startswith("{")
        path = Path(text)
        if looks_like_json or not path.exists():
            try:
                doc = json.loads(text)
            except ValueError as exc:
                raise RulePackError(
                    f"the pack is not valid JSON and not a readable "
                    f"file ({exc})") from exc
        else:
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise RulePackError(
                    f"the pack file cannot be read ({exc})") from exc
    else:
        doc = pack
    if not isinstance(doc, dict) \
            or doc.get("schema_version") != PACK_SCHEMA_VERSION \
            or not isinstance(doc.get("pack"), dict) \
            or not isinstance(doc.get("payload"), list):
        raise RulePackError(
            "not a rule pack: expected a document with schema_version "
            f"{PACK_SCHEMA_VERSION}, a 'pack' manifest object, and a "
            "'payload' list")
    manifest = doc["pack"]
    payload = doc["payload"]
    pack_id = manifest.get("id")
    if not isinstance(pack_id, str) or not pack_id:
        raise RulePackError("the pack manifest has no id")
    if existing_ids and pack_id in existing_ids:
        raise RulePackError(
            f"a pack named {pack_id!r} is already imported; remove it "
            "first (rulepack forget) — imports never silently overwrite")
    # integrity: the manifest hash is the whole point of the format
    stated = manifest.get("payload_sha256")
    actual = payload_sha256(payload)
    if stated != actual:
        raise RulePackError(
            f"payload hash mismatch: manifest says {stated!r}, payload "
            f"hashes to {actual!r} — the payload was modified after the "
            "manifest was written; refusing")
    if manifest.get("rule_count") != sum(
            len(d.get("rules", [])) for d in payload):
        raise RulePackError(
            "manifest rule_count does not match the payload's rules")
    # safety: the same gates the built-ins pass, nothing weaker
    from .engine import validate_rule
    from . import schema_lint
    seen_rule_ids: set = set()
    for doc_i, ruleset in enumerate(payload):
        source = f"pack {pack_id!r} ruleset #{doc_i}"
        if ruleset.get("schema_version") != 1:
            raise RulePackError(
                f"{source}: unsupported schema_version "
                f"{ruleset.get('schema_version')!r}")
        for rule in ruleset.get("rules", []):
            failures = list(validate_rule(rule, source=source))
            failures.extend(schema_lint.validate_rule_safety(rule, source))
            if failures:
                raise RulePackError(
                    f"{source}: rule refused — {'; '.join(failures)}; "
                    "the pack imports whole or not at all")
            if rule["id"] in seen_rule_ids:
                raise RulePackError(
                    f"{source}: duplicate rule id {rule['id']}")
            seen_rule_ids.add(rule["id"])
    record = {
        "id": pack_id,
        "rulesets": manifest.get("rulesets", []),
        "rule_count": manifest.get("rule_count", 0),
        "payload_sha256": actual,
        "pack_sha256": hashlib.sha256(canonical_bytes(doc)).hexdigest(),
        "signer": str(signer or ""),
        "payload": payload,
    }
    return record


def list_packs(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The stored pack records, oldest import first (copies, without
    the bulky payload — the review view)."""
    out = []
    for rec in state.get(STATE_KEY, []) or []:
        out.append({k: v for k, v in rec.items() if k != "payload"})
    return out


# ---------------------------------------------------------------------------
# render (the review + human-placement surfaces)
# ---------------------------------------------------------------------------

def render_pack(record: Dict[str, Any]) -> str:
    """The human REVIEW view of one imported pack: id, claimed signer,
    hashes, rulesets, and every rule id + title + severity."""
    lines = [f"rule pack {record.get('id')!r}",
             f"  claimed signer: {record.get('signer') or '(none recorded)'}"
             "  (identity is a claim — verify signatures with YOUR "
             "external tool)",
             f"  payload sha256: {record.get('payload_sha256')}",
             f"  pack sha256   : {record.get('pack_sha256')}"]
    for ruleset in record.get("payload", []):
        name = ruleset.get("ruleset", "?")
        lines.append(f"  ruleset {name} "
                     f"({len(ruleset.get('rules', []))} rules):")
        for rule in ruleset.get("rules", []):
            lines.append(f"    [{rule.get('severity', '?')}] "
                         f"{rule.get('id')}: {rule.get('title', '')}")
    lines.append("imported rules are REVIEW DATA, not live: the engine "
                 "loads rules.d only; 'rulepack render' prints a "
                 "rules.d-ready file and PLACING it is a human action")
    return "\n".join(lines)


def render_rules_d(record: Dict[str, Any]) -> str:
    """The pack's rules as ONE rules.d-ready JSON document (stdout
    text — this module never writes files). The human who decides to
    trust a pack writes this to rules.d themselves; that decision is
    the consent step and it stays human."""
    merged: Dict[str, Any] = {
        "schema_version": 1,
        "ruleset": f"imported-pack:{record.get('id')}",
        "provenance": {
            "imported_from_pack": record.get("id"),
            "pack_payload_sha256": record.get("payload_sha256"),
            "claimed_signer": record.get("signer") or "(none)",
        },
        "rules": [rule for ruleset in record.get("payload", [])
                  for rule in ruleset.get("rules", [])],
    }
    return json.dumps(merged, indent=2) + "\n"
