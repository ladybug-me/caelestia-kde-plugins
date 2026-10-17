"""cortex.tree — a CART decision tree beside the logistic router (§6.3).

The learner's online logistic regression (learn.py) is accurate but its
verdict is a weight vector: "why did it route here" is five signed numbers.
This module grows the SAME decision on the SAME five features
(lex / sem / fuzz / noun / cue-kind agreement) as a CART classification
tree (Breiman, Friedman, Olshen & Stone 1984, "Classification and
Regression Trees", Wadsworth: binary recursive partitioning, Gini
impurity, midpoint thresholds) — a verdict path you can READ:

    lex >= 0.45
      noun < 0.20 -> reject   (0.92 impure, n=41)
      noun >= 0.20 -> accept  (0.05, n=118)

Placement — complement, not replacement: the tree never replaces the
logistic router anywhere in the pipeline. It trains from the learner's
own bounded example log (the audit trail the learner already keeps),
answers the same accept/reject question, and reports its agreement with
the fitted logistic model BEFORE anyone decides whether to promote it to
a real verdict path (``agreement_report`` below; the numbers it produced
on the shipped corpus are in the CHANGELOG entry, not in a promise).

Deterministic: ties broken by feature order then threshold value, no
randomness anywhere. Bounded: depth and leaf-size caps are hard stops.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .learn import _FEATURES

__all__ = ["gini", "build_tree", "tree_classify", "tree_from_examples",
           "tree_to_rules", "agreement_report", "MAX_DEPTH", "MIN_LEAF"]

# Readability-first caps: a tree deeper than this stops being an
# explanation and starts being a model.
MAX_DEPTH = 3
MIN_LEAF = 5


def gini(labels: Sequence[int]) -> float:
    """Gini impurity of a label multiset: 1 - sum(p_k^2)."""
    n = len(labels)
    if n == 0:
        return 0.0
    counts: Dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


def _best_split(rows: List[Sequence[float]], labels: List[int]
                ) -> Optional[Tuple[int, float, float]]:
    """The (feature index, threshold, weighted impurity) minimizing the
    children's impurity, or None when no split helps. Candidate thresholds
    are midpoints between consecutive distinct sorted values — CART's own
    convention; ties break on feature order then threshold."""
    n = len(labels)
    parent = gini(labels)
    n_features = len(rows[0]) if rows else 0
    best: Optional[Tuple[int, float, float]] = None
    for feature in range(n_features):
        pairs = sorted(set(row[feature] for row in rows))
        for a, b in zip(pairs, pairs[1:]):
            threshold = (a + b) / 2.0
            left = [labels[i] for i in range(n) if rows[i][feature] < threshold]
            right = [labels[i] for i in range(n) if rows[i][feature] >= threshold]
            if not left or not right:
                continue
            weighted = (len(left) * gini(left) + len(right) * gini(right)) / n
            if best is None or weighted < best[2] - 1e-12:
                best = (feature, threshold, weighted)
    if best is not None and best[2] >= parent - 1e-12:
        return None  # no split improves impurity
    return best


def build_tree(rows: List[Sequence[float]], labels: List[int],
               depth: int = 0, max_depth: int = MAX_DEPTH,
               min_leaf: int = MIN_LEAF) -> Dict[str, Any]:
    """Grow the CART tree over (features, labels); returns a nested dict
    {leaf: True, prediction, distribution, n} or
    {leaf: False, feature, name, threshold, left, right, n}."""
    n = len(labels)
    counts: Dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    majority = max(sorted(counts), key=lambda k: counts[k])
    leaf = {"leaf": True, "prediction": majority,
            "distribution": {str(k): v for k, v in sorted(counts.items())},
            "impurity": round(gini(labels), 4), "n": n}
    if depth >= max_depth or n < 2 * min_leaf or gini(labels) == 0.0:
        return leaf
    split = _best_split(rows, labels)
    if split is None:
        return leaf
    feature, threshold, _weighted = split
    left_rows = [rows[i] for i in range(n) if rows[i][feature] < threshold]
    left_labels = [labels[i] for i in range(n) if rows[i][feature] < threshold]
    right_rows = [rows[i] for i in range(n) if rows[i][feature] >= threshold]
    right_labels = [labels[i] for i in range(n) if rows[i][feature] >= threshold]
    if len(left_labels) < min_leaf or len(right_labels) < min_leaf:
        return leaf
    return {
        "leaf": False,
        "feature": feature,
        "name": _FEATURES[feature] if feature < len(_FEATURES) else f"f{feature}",
        "threshold": round(threshold, 6),
        "left": build_tree(left_rows, left_labels, depth + 1, max_depth, min_leaf),
        "right": build_tree(right_rows, right_labels, depth + 1, max_depth, min_leaf),
        "n": n,
    }


def tree_classify(tree: Dict[str, Any],
                  features: Dict[str, float]) -> Tuple[int, List[str]]:
    """Walk the tree; returns (prediction, the readable path taken)."""
    path: List[str] = []
    node = tree
    while not node.get("leaf"):
        name = node["name"]
        value = float(features.get(name, 0.0))
        go_left = value < node["threshold"]
        path.append(f"{name} {'<' if go_left else '>='} {node['threshold']}")
        node = node["left"] if go_left else node["right"]
    path.append(f"-> {'accept' if node['prediction'] == 1 else 'reject'} "
                f"(impurity {node['impurity']}, n={node['n']})")
    return int(node["prediction"]), path


def tree_from_examples(examples: Sequence[Dict[str, Any]],
                       max_depth: int = MAX_DEPTH,
                       min_leaf: int = MIN_LEAF) -> Optional[Dict[str, Any]]:
    """Train the tree on the learner's example log rows ({features,
    label}); None when there are too few examples to split honestly
    (the tree says nothing rather than guessing from thin air)."""
    rows = [[float(e.get("features", {}).get(name, 0.0))
             for name in _FEATURES]
            for e in examples if isinstance(e, dict) and "label" in e]
    labels = [int(e["label"]) for e in examples
              if isinstance(e, dict) and "label" in e]
    if len(labels) < 2 * min_leaf:
        return None
    return build_tree(rows, labels, max_depth=max_depth, min_leaf=min_leaf)


def tree_to_rules(tree: Dict[str, Any], prefix: str = "") -> List[str]:
    """Render every root-to-leaf path as one readable rule line."""
    if tree.get("leaf"):
        prediction = "accept" if tree["prediction"] == 1 else "reject"
        return [f"{prefix or 'always'} -> {prediction} "
                f"(impurity {tree['impurity']}, n={tree['n']})"]
    out: List[str] = []
    out.extend(tree_to_rules(tree["left"],
                             f"{prefix} and {tree['name']} < {tree['threshold']}"
                             if prefix else f"{tree['name']} < {tree['threshold']}"))
    out.extend(tree_to_rules(tree["right"],
                              f"{prefix} and {tree['name']} >= {tree['threshold']}"
                              if prefix else f"{tree['name']} >= {tree['threshold']}"))
    return [line.lstrip(" and ").replace("always and ", "")
            for line in out]


def agreement_report(examples: Sequence[Dict[str, Any]], *,
                     test_fraction: float = 0.3,
                     max_depth: int = MAX_DEPTH,
                     min_leaf: int = MIN_LEAF) -> Optional[Dict[str, Any]]:
    """Train the CART tree AND the logistic model on a training split of
    the labeled examples, then measure how often they agree on the held-out
    rest — the number that decides whether the tree is worth promoting to
    a real verdict path. Deterministic split (every 3rd example held out
    by position, no RNG). Returns None with too few examples to say
    anything honest."""
    usable = [e for e in examples if isinstance(e, dict) and "label" in e]
    if len(usable) < 4 * MIN_LEAF:
        return None
    stride = max(2, round(1.0 / test_fraction))
    train = [e for i, e in enumerate(usable) if i % stride != 0]
    test = [e for i, e in enumerate(usable) if i % stride == 0]
    if not train or not test:
        return None

    tree = tree_from_examples(train, max_depth=max_depth, min_leaf=min_leaf)
    from .learn import OnlineLogistic
    model = OnlineLogistic()
    # a fair fit, not a single undertrained pass: the same example log the
    # learner itself replays; five epochs is where the online AdaGrad
    # model stabilizes on corpora of this size (the learner itself sees
    # each example exactly once per session and sessions repeat)
    for _epoch in range(5):
        for e in train:
            model.update(e.get("features", {}), int(e["label"]))
    if tree is None:
        return None

    def _agree(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
        agree = 0
        for e in rows:
            tree_label, _path = tree_classify(tree, e.get("features", {}))
            logi_label = 1 if model.predict(e.get("features", {})) >= 0.5 else 0
            if tree_label == logi_label:
                agree += 1
        return agree, len(rows)

    train_agree, train_n = _agree(train)
    test_agree, test_n = _agree(test)
    tree_right = sum(1 for e in test
                     if tree_classify(tree, e.get("features", {}))[0] == int(e["label"]))
    logi_right = sum(1 for e in test
                     if (1 if model.predict(e.get("features", {})) >= 0.5 else 0)
                     == int(e["label"]))
    return {
        "n_train": train_n, "n_test": test_n,
        "train_agreement": round(train_agree / max(1, train_n), 3),
        "test_agreement": round(test_agree / max(1, test_n), 3),
        "tree_test_accuracy": round(tree_right / max(1, test_n), 3),
        "logistic_test_accuracy": round(logi_right / max(1, test_n), 3),
        "rules": tree_to_rules(tree),
    }
