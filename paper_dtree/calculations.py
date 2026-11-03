"""Program-evaluated nodes.

A calculation node is the only place a tree answer can be produced without a
model call: it counts answers already recorded in the same tree and compares
the count with a threshold.  Every kind declares its own key set and knows how
to recompute its boolean branch answer from prior answers, so a bundle cannot
smuggle executable logic into free text.
"""
from __future__ import annotations

from typing import Any

ALLOWED_KEYS = {
    # Original ADHD domain gate: age-dependent threshold (5 for adults, 6 otherwise).
    "adhd_domain_threshold_v1": {"kind", "age_node", "symptom_nodes"},
    # Standard count gate: "at least N of the listed questions were answered yes".
    "count_threshold_v1": {"kind", "symptom_nodes", "threshold"},
}


def _node_ids(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list of node ids")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{name} must contain nonempty node ids")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} contains duplicates")
    return tuple(value)


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate(tree_id: str, spec: Any) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("calculation must be an object")
    kind = spec.get("kind")
    if kind not in ALLOWED_KEYS:
        raise ValueError(f"Unknown calculation kind: {kind!r}")
    unexpected = set(spec) - ALLOWED_KEYS[kind]
    if unexpected:
        raise ValueError(f"calculation: unexpected keys {sorted(unexpected)}")
    symptoms = _node_ids(spec.get("symptom_nodes"), "calculation.symptom_nodes")
    if kind == "adhd_domain_threshold_v1":
        if len(symptoms) != 9:
            raise ValueError("adhd_domain_threshold_v1 requires nine distinct symptom nodes")
        age_node = spec.get("age_node")
        if not isinstance(age_node, str) or not age_node.strip():
            raise ValueError("calculation.age_node must be nonempty text")
        if age_node in symptoms:
            raise ValueError("calculation.age_node must not also be a symptom node")
    else:
        threshold = _positive_int(spec.get("threshold"), "calculation.threshold")
        if threshold > len(symptoms):
            raise ValueError("calculation.threshold cannot exceed the number of symptom nodes")
    return spec


def dependencies(spec: dict[str, Any]) -> set[str]:
    """Node ids whose answers this calculation consumes."""
    if "age_node" in spec:
        return {spec["age_node"], *spec["symptom_nodes"]}
    return set(spec["symptom_nodes"])


def compute(spec: dict[str, Any], prior: dict[str, dict]) -> tuple[bool, dict[str, Any]]:
    """Recompute a calculation from answers already recorded in the same tree.

    ``prior`` maps node ids to trace records carrying ``answer`` in {"yes", "no"}.
    A missing dependency is an error, never an implicit "no".
    """
    missing = sorted(key for key in dependencies(spec) if key not in prior)
    if missing:
        raise ValueError(f"Calculation ran before answers existed: {missing}")
    inputs = spec["symptom_nodes"]
    count = sum(prior[key]["answer"] == "yes" for key in inputs)
    block: dict[str, Any] = {
        "kind": spec["kind"], "symptom_count": count,
        "inputs": [{"node_id": key, "answer": prior[key]["answer"]} for key in inputs],
    }
    if spec["kind"] == "adhd_domain_threshold_v1":
        age_17_or_over = prior[spec["age_node"]]["answer"] == "yes"
        threshold = 5 if age_17_or_over else 6
        block.update(threshold=threshold, age_17_or_over=age_17_or_over, age_node=spec["age_node"])
    else:
        threshold = spec["threshold"]
        block.update(threshold=threshold)
    return count >= threshold, block