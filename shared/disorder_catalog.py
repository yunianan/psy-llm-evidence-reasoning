"""Project labels and presentation metadata, separate from diagnostic content.

Display groups and Chinese names are presentation metadata, not DSM rules.
Insomnia is the existing project extension and is not a MentalKG node.
"""
from __future__ import annotations

from typing import Any
from functools import lru_cache

from .knowledge_sources import SUPPLEMENT_REF, get_insomnia_source, get_mental_kg_source


DISORDERS = get_mental_kg_source().disorders
SUPPORTED_DISORDERS = (*sorted(DISORDERS), "Insomnia")
DISORDER_COUNT = len(SUPPORTED_DISORDERS)
SUPPLEMENT_DISORDERS = get_insomnia_source().disorders
@lru_cache
def _symptom_sources():
    """Display labels and CandidateGeneration imports must not read symptoms."""
    mental = get_mental_kg_source().symptoms
    supplement = get_insomnia_source().symptoms
    if mental.keys() & supplement.keys():
        raise ValueError("Supplement symptom IDs must not overwrite MentalKG")
    return mental, supplement, {**mental, **supplement}


def __getattr__(name):
    # Preserve existing explicit imports; resolve only when symptoms are requested.
    names = ("MENTALKG_SYMPTOMS", "SUPPLEMENT_SYMPTOMS", "SYMPTOMS")
    if name in names:
        return _symptom_sources()[names.index(name)]
    raise AttributeError(name)

DISPLAY_GROUPS = {
    "attention": "注意缺陷与多动相关",
    "psychotic": "精神病性及相关障碍",
    "bipolar": "双相相关",
    "anxiety": "焦虑相关",
    "sleep": "睡眠相关",
    "depressive": "抑郁相关",
    "obsessive": "强迫及相关障碍",
    "stress": "创伤与应激相关",
    "eating": "进食相关",
}
_GROUP_NODES = {
    "attention": ("D001", "D002", "D003"),
    "psychotic": ("D004", "D005", "D006", "D007"),
    "bipolar": ("D008", "D009", "D010"),
    "anxiety": ("D011", "D012"),
    "sleep": ("Insomnia",),
    "depressive": ("D013", "D014", "D015"),
    "obsessive": ("D016", "D017"),
    "stress": ("D018", "D019", "D020"),
    "eating": ("D021", "D022", "D023"),
}
_GROUP_BY_NODE = {node: group for group, nodes in _GROUP_NODES.items() for node in nodes}
_ZH_NAMES = dict(zip(sorted(DISORDERS), (
    "注意缺陷/多动障碍（混合表现）", "注意缺陷/多动障碍（注意缺陷为主）",
    "注意缺陷/多动障碍（多动冲动为主）", "妄想障碍", "精神分裂症",
    "分裂情感性障碍（双相型）", "分裂情感性障碍（抑郁型）",
    "双相Ⅰ型障碍", "双相Ⅱ型障碍", "双相Ⅰ型障碍伴精神病性特征",
    "广泛性焦虑障碍", "特定恐怖症", "重性抑郁障碍", "持续性抑郁障碍",
    "重性抑郁障碍伴精神病性特征", "强迫症", "躯体变形障碍",
    "创伤后应激障碍", "急性应激障碍", "适应障碍", "神经性厌食",
    "神经性贪食", "暴食障碍",
)))
_ZH_NAMES["Insomnia"] = "失眠障碍"

# Only unambiguous old codes are accepted for historical report display.
# The former generic Bipolar label deliberately has no single-node mapping.
LEGACY_ALIASES = {"MDD": "D013", "GAD": "D011", "PTSD": "D018",
                  "OCD": "D016", "AdjustmentDisorder": "D020"}


def canonical_disorder(value: str) -> str:
    if value in SUPPORTED_DISORDERS:
        return value
    if value in LEGACY_ALIASES:
        return LEGACY_ALIASES[value]
    for node, record in DISORDERS.items():
        if value.casefold() == record["name"].casefold():
            return node
    if value.casefold() in {"insomnia disorder", "insomnia"}:
        return "Insomnia"
    return value


def disorder_metadata(value: str) -> dict[str, str]:
    node = canonical_disorder(value)
    if node not in SUPPORTED_DISORDERS:
        return {"disorder": value, "label": value, "display_name": value}
    group = _GROUP_BY_NODE[node]
    return {
        "disorder": node,
        "name": (DISORDERS if node in DISORDERS else SUPPLEMENT_DISORDERS)[node]["name"],
        "label": _ZH_NAMES[node],
        "display_group": group,
        "display_group_label": DISPLAY_GROUPS[group],
        "display_name": f"{DISPLAY_GROUPS[group]} · {_ZH_NAMES[node]}",
        "knowledge_source": ("MentalKG/DSM-5" if node in DISORDERS
                             else f"{SUPPLEMENT_REF}/diagnostic_criteria.json#/{node}"),
    }


def grouped_catalog() -> list[dict[str, Any]]:
    return [{"group": group, "label": label,
             "disorders": [disorder_metadata(node) for node in _GROUP_NODES[group]]}
            for group, label in DISPLAY_GROUPS.items()]


def symptom_catalog() -> dict[str, str]:
    return {sid: record["description"] for sid, record in _symptom_sources()[2].items()}


# Preserve the existing six treatment routes only. Other diagnoses have no
# validated project treatment mapping and are routed to manual review.
TREATMENT_KEY_BY_NODE = {"D008": "Bipolar", "D009": "Bipolar", "D011": "GAD",
                         "D013": "MDD", "D016": "OCD", "D018": "PTSD",
                         "D020": "AdjustmentDisorder", "Insomnia": "Insomnia"}


def treatment_key(node: str) -> str:
    return TREATMENT_KEY_BY_NODE.get(canonical_disorder(node), node)
