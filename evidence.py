"""沿用 CLINICR 的规则生成证据白名单，不信任输入中的同名元数据。"""
from typing import Any


def known_evidence_refs(profile: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    scale_keys = {
        "PHQ9", "GAD7", "PCL5", "Y_BOCS", "ISI", "MDQ",
        "ADNM", "RRS", "AAQ_II", "PHQ15",
    }
    for key in scale_keys:
        if key in profile and profile[key] is not None:
            refs.add(f"{key}={profile[key]}")
        items = profile.get(f"{key}_items")
        if isinstance(items, dict):
            for item, score in items.items():
                refs.add(f"{key}_{item}={score}")

    for collection in (
        "key_symptoms_extracted",
        "emotions",
        "cognitive_patterns",
        "key_narratives",
        "functional_impairments",
    ):
        for item in profile.get(collection) or []:
            if isinstance(item, dict):
                refs.update(str(ref) for ref in item.get("sources") or [] if ref)

    for node in profile.get("evidence_nodes") or []:
        if isinstance(node, dict) and node.get("evidence_id"):
            refs.add(str(node["evidence_id"]))
    return refs
