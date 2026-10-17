"""Detect what changed between two labelled snapshots of text — a vault
yesterday vs today, or an upstream doc set vs your fork. Generic: works for
notes, config docs, or any other {id: text} collection.

Uses minhash signatures so a reworded-but-not-rewritten item counts as
"unchanged" rather than showing up as noisy add+remove pairs.
"""
from .minhash import jaccard, signature


def diff(old, new, similarity_threshold=0.85):
    """old, new: {id: text}.

    Returns {"added", "removed", "changed"}: added/removed are id lists;
    changed is [{"id", "similarity"}] for ids present in both snapshots
    whose text drifted below similarity_threshold, most-changed first.
    """
    old_ids, new_ids = set(old), set(new)
    added = sorted(new_ids - old_ids)
    removed = sorted(old_ids - new_ids)
    changed = []
    for shared_id in sorted(old_ids & new_ids):
        sig_a, sig_b = signature(old[shared_id]), signature(new[shared_id])
        if not sig_a or not sig_b:
            continue
        sim = jaccard(sig_a, sig_b)
        if sim < similarity_threshold:
            changed.append({"id": shared_id, "similarity": round(sim, 3)})
    return {"added": added, "removed": removed,
            "changed": sorted(changed, key=lambda r: r["similarity"])}
