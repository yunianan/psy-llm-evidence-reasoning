"""Research-only three-valued criteria arithmetic; clinical facts remain sourced.

Missing symptom facts are unknown, never negative. Natural-language temporal
and exclusion rules that this module cannot represent remain LLM questions.
"""
from __future__ import annotations

import re
import math
from typing import Any

from layers.layer3.diagnosis.schemas import TreeNodeJudgment

VERSION = "computable-criteria-v1"
DURATION = re.compile(r"(\d+(?:\.\d+)?)\s+(days?|weeks?|months?|years?)", re.I)


def rule_for(node: dict) -> dict | None:
    path = node["node_id"].split(".", 2)[2]
    part = path.rsplit("/", 1)[-1]
    condition = node["source_condition"]
    if part in {"min_count", "must_include_one_of", "must_include_all"}:
        pool = node.get("symptom_pool", [])
        if not pool or (part == "min_count" and (type(condition) is not int or not 1 <= condition <= len(pool))):
            return None
        if part != "min_count" and (not isinstance(condition, list) or not condition or not set(condition) <= set(pool)):
            return None
        group = node["node_id"].rsplit("/", 1)[0]
        return {"kind": part, "facts": {f"{group}::{sid}": {"kind": "symptom", "group": group,
            "disorder": node["disorders"][0], "symptom_id": sid} for sid in pool}}
    if part in {"min_duration", "max_duration"} and isinstance(condition, str):
        match = DURATION.fullmatch(condition.strip())
        if match:
            number, unit = float(match[1]), match[2].lower().rstrip("s")
            return {"kind": part, "threshold": number, "unit": unit,
                    "facts": {node["node_id"] + "::duration": {"kind": "duration", "node_id": node["node_id"]}}}
    # Interpreting clinical significance/event exposure is still a semantic task.
    # Program logic applies the positive requirement to the extracted predicate.
    if (part == "functional_impairment_required" and condition is True) or (
        part in {"traumatic_stressor_required", "psychosocial_stressor_required"} and isinstance(condition, str)
    ):
        return {"kind": "required_predicate", "facts": {node["node_id"] + "::predicate": {
            "kind": "predicate", "node_id": node["node_id"]}}}
    return None


def validate_fact(value: dict, spec: dict, allowed_refs: set[str]) -> dict:
    common = {"fact_id", "evidence_refs", "confidence"}
    expected = common | ({"lower", "upper", "unit"} if spec["kind"] == "duration" else {"state"})
    if set(value) != expected:
        raise ValueError("Unexpected fact fields")
    refs = value["evidence_refs"]
    if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs) or not set(refs) <= set(allowed_refs):
        raise ValueError("Unlisted fact evidence")
    confidence = value["confidence"]
    if type(confidence) not in {int, float} or not 0 <= confidence <= 1:
        raise ValueError("Invalid fact confidence")
    if spec["kind"] == "duration":
        lo, hi, unit = value["lower"], value["upper"], value["unit"]
        if lo is None and hi is None:
            if unit is not None:
                raise ValueError("Unknown duration needs null unit")
        else:
            if not isinstance(unit, str) or unit not in {"day", "week", "month", "year"} or not refs:
                raise ValueError("Known duration needs unit and evidence")
            if any(x is not None and (type(x) not in {int, float} or not math.isfinite(x) or x < 0) for x in [lo, hi]):
                raise ValueError("Invalid duration bound")
            if lo is not None and hi is not None and lo > hi:
                raise ValueError("Inverted duration range")
    else:
        if not isinstance(value["state"], str) or value["state"] not in {"present", "absent", "unknown"}:
            raise ValueError("Invalid presence state")
        if value["state"] != "unknown" and not refs:
            raise ValueError("Positive/negative fact needs patient evidence")
    return value


def _scale(from_unit, to_unit):
    if from_unit == to_unit:
        return 1
    # Do not invent a 30-day month at a clinical boundary.
    return {("week", "day"): 7, ("day", "week"): 1 / 7,
            ("year", "month"): 12, ("month", "year"): 1 / 12}.get((from_unit, to_unit))


def calculate(node: dict, rule: dict, facts: dict[str, dict]) -> TreeNodeJudgment:
    specs = rule["facts"]
    selected = {key: facts[key] for key in specs}
    kind = rule["kind"]
    used, mapped, status = [], [], "pending"
    if kind in {"min_count", "must_include_one_of", "must_include_all"}:
        present = {specs[key]["symptom_id"]: value for key, value in selected.items() if value["state"] == "present"}
        absent = {specs[key]["symptom_id"]: value for key, value in selected.items() if value["state"] == "absent"}
        unknown = {specs[key]["symptom_id"] for key, value in selected.items() if value["state"] == "unknown"}
        condition = node["source_condition"]
        if kind == "min_count":
            if len(present) >= condition:
                status, mapped, used = "met", sorted(present), list(present.values())
            elif len(present) + len(unknown) < condition:
                status, used = "not_met", list(absent.values())
        else:
            required = set(condition)
            positive = required & set(present)
            negative = required & set(absent)
            if (kind == "must_include_one_of" and positive) or (kind == "must_include_all" and required <= set(present)):
                status, mapped = "met", sorted(positive)
                used = [present[sid] for sid in mapped]
            elif (kind == "must_include_one_of" and required <= set(absent)) or (kind == "must_include_all" and negative):
                status, used = "not_met", [absent[sid] for sid in sorted(negative)]
        rationale = f"{VERSION}: {kind}; supported={len(present)}, denied={len(absent)}, unknown={len(unknown)}; source_condition={condition}"
    elif kind == "required_predicate":
        value = next(iter(selected.values()))
        status = {"present": "met", "absent": "not_met", "unknown": "pending"}[value["state"]]
        used = [value]
        rationale = f"{VERSION}: sourced required predicate={value['state']}"
    else:
        value = next(iter(selected.values()))
        factor = _scale(value["unit"], rule["unit"]) if value["unit"] is not None else None
        if factor is not None:
            lo, hi = [value[key] * factor if value[key] is not None else None for key in ["lower", "upper"]]
            threshold = rule["threshold"]
            if kind == "min_duration":
                status = "met" if lo is not None and lo >= threshold else "not_met" if hi is not None and hi < threshold else "pending"
            else:
                status = "met" if hi is not None and hi <= threshold else "not_met" if lo is not None and lo > threshold else "pending"
        used = [value]
        rationale = f"{VERSION}: {kind}, source={node['source_condition']}, patient_range={value['lower']}..{value['upper']} {value['unit']}; incompatible/unknown bounds remain pending"
    if not used:
        used = list(selected.values())
    refs = list(dict.fromkeys(ref for value in used for ref in value["evidence_refs"]))
    if status != "pending" and not refs:
        raise ValueError("Computed known judgment must have patient evidence")
    return TreeNodeJudgment(node_id=node["node_id"], status=status, rationale=rationale,
        evidence_refs=refs, symptom_ids=mapped if status == "met" else [],
        confidence=min((value["confidence"] for value in used), default=.5))
