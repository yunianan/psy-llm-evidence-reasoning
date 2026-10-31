"""L3病程判定只能使用绑定原文的共享区间；输出可回写L2。"""
from __future__ import annotations

from shared.clinical_contract import DurationEvidence, TemporalDecision
from shared.temporal import parse_duration, parse_duration_records, compare_duration

GENERIC_SYMPTOMS = {
    "depressed_mood": {"S024"}, "anhedonia": {"S025"}, "insomnia": {"S027", "S085", "S086", "S087"},
    "fatigue": {"S029"}, "concentration_difficulty": {"S031"},
    "anxiety": {"S038", "S039"},
}


def duration_node(definition):
    return definition["node_id"].rsplit("/", 1)[-1].rsplit(".", 1)[-1] in {"min_duration", "max_duration"}


def evaluate_temporal_node(profile, definition, *, evidence_refs=()):
    node_id = definition["node_id"]
    nodes = {n.get("evidence_id"): n for n in profile.get("evidence_nodes", [])}
    pool = set(definition.get("duration_symptom_pool", []))
    records = [DurationEvidence.model_validate(d) for d in profile.get("duration_evidence", [])]
    selected = []
    for index, record in enumerate(records):
        mapped = GENERIC_SYMPTOMS.get(record.symptom_id, {record.symptom_id})
        valid_refs = bool(record.evidence_refs) and all(
            ref in nodes and record.raw_span in nodes[ref].get("quote", "") for ref in record.evidence_refs)
        reparsed = parse_duration(record.raw_span, symptom_id=record.symptom_id, scope=record.scope,
            evidence_refs=record.evidence_refs, source=record.source,
            interview_date=profile.get("interview_date") or record.interview_date)
        parsed_fields = ("min_days", "max_days", "parse_status", "pattern", "duration_kind", "approx", "approximation", "unit", "quantity")
        if any(getattr(record, key) != getattr(reparsed, key) for key in parsed_fields):
            profile.setdefault("temporal_audit", []).append({"node_id": node_id,
                "detected_errors": ["structured_interval_disagrees_with_raw_span"], "raw_span": record.raw_span,
                "remaining_errors": [], "corrected_interval": reparsed.model_dump(mode="json")})
            record = reparsed
            profile["duration_evidence"][index] = reparsed.model_dump(mode="json")
        if valid_refs and (record.scope == node_id or (record.symptom_id and mapped.intersection(pool))):
            selected.append(record)
    # 只有L3明确引用的原始患者证据可回写；不读取参考诊断或无引用叙事。
    if not selected:
        blocked_scope = False
        for ref in evidence_refs:
            node = nodes.get(ref)
            if not node or not node.get("quote"):
                continue
            bound = [r for r in records if ref in r.evidence_refs and (r.symptom_id or r.scope.startswith("criterion."))]
            if bound and not any(r.scope == node_id or (r.symptom_id and GENERIC_SYMPTOMS.get(r.symptom_id, {r.symptom_id}).intersection(pool)) for r in bound):
                blocked_scope = True
                continue
            for record in parse_duration_records(node["quote"], scope=node_id, evidence_refs=[ref], source="L3_writeback",
                                                 interview_date=profile.get("interview_date")):
                selected.append(record)
                value = record.model_dump(mode="json")
                if value not in profile.setdefault("duration_evidence", []):
                    profile["duration_evidence"].append(value)
    if not selected:
        reason = "时长已绑定其他症状群，当前病程归属需澄清" if blocked_scope else "没有绑定到该病程条件的患者原文或结构化区间"
        return TemporalDecision(status="unknown", reason=reason)
    is_episode = "episode/" in node_id
    matching_kind = [r for r in selected if r.duration_kind == ("episode" if is_episode else "overall")]
    if matching_kind:
        selected = matching_kind
    # 判定采用的L2区间也生成节点级回写快照，保留原L2记录。
    for record in selected:
        snapshot = record.model_copy(update={"scope": node_id, "source": "L3_writeback"}).model_dump(mode="json")
        if snapshot not in profile.setdefault("duration_evidence", []):
            profile["duration_evidence"].append(snapshot)
    unique = {(r.min_days, r.max_days, r.pattern, r.duration_kind, r.parse_status): r for r in selected}
    selected = list(unique.values())
    for a in selected:
        for b in selected:
            if a.max_days is not None and b.min_days is not None and a.max_days < b.min_days:
                return TemporalDecision(status="unknown", reason="同一病程条件存在不相容区间，需澄清冲突",
                    evidence_refs=list(dict.fromkeys(ref for r in selected for ref in r.evidence_refs)))
    condition = definition.get("source_condition")
    if any(token in str(condition).lower() for token in ("after", "following", "termination")):
        if not all(r.anchor.startswith("event:") for r in selected):
            return TemporalDecision(status="unknown", reason="来源门槛要求事件结束等锚点，当前仅有整体病程",
                evidence_refs=list(dict.fromkeys(ref for r in selected for ref in r.evidence_refs)))
    required_pattern = "continuous" if "continuous" in str(condition).lower() else "unknown"
    decisions = []
    for record in selected:
        if record.pattern == "episodic" and record.duration_kind == "overall" and is_episode:
            decisions.append(TemporalDecision(status="unknown", reason="多年反复病程不能证明单次发作持续时间", evidence_refs=record.evidence_refs))
        else:
            decisions.append(compare_duration(record, condition,
                operator="max" if "max_duration" in node_id else "min", required_pattern=required_pattern,
                required_kind="episode" if is_episode and record.duration_kind == "episode" else "overall"))
    statuses = {d.status for d in decisions}
    if len(statuses) != 1:
        return TemporalDecision(status="unknown", reason="多个病程证据的门槛比较不一致，需澄清",
            evidence_refs=list(dict.fromkeys(ref for d in decisions for ref in d.evidence_refs)))
    answer = decisions[0].model_copy(deep=True)
    answer.evidence_refs = list(dict.fromkeys(ref for d in decisions for ref in d.evidence_refs))
    return answer


def apply_temporal_contract(generation, profile, *, knowledge):
    """在汇总和奖励终验之前统一改写所有病程节点，其他节点不变。"""
    from layers.layer3.diagnosis.decision_tree import criterion_nodes
    from layers.layer3.diagnosis.schemas import MissingDimension

    definitions = {}
    for candidate in generation.candidates:
        definitions.update({nid: {"node_id": nid, **d} for nid, d in criterion_nodes(candidate.disorder, knowledge=knowledge).items()})
    records = {}
    for record in generation.tree_trace:
        records.setdefault(record.node_id, []).append(record)
    for candidate in generation.candidates:
        for path in candidate.decision_paths:
            for record in path.decisions:
                records.setdefault(record.node_id, []).append(record)
    for nid, copies in records.items():
        definition = definitions.get(nid)
        if not definition or not duration_node(definition):
            continue
        original = copies[0].status
        had_structured = any(d.get("scope") == nid for d in profile.get("duration_evidence", []))
        decision = evaluate_temporal_node(profile, definition, evidence_refs=copies[0].evidence_refs)
        status = decision.status if decision.status in ("met", "not_met") else "pending"
        errors = []
        if original in ("met", "not_met") and not had_structured:
            errors.append("duration_judgment_without_structured_evidence")
        if original != status:
            errors.append("duration_judgment_conflicts_with_interval")
        profile.setdefault("temporal_audit", []).append({"node_id": nid, "original_status": original,
            "effective_status": status, "decision": decision.model_dump(mode="json"),
            "detected_errors": errors, "remaining_errors": [] if decision.evidence_refs or status == "pending" else ["missing_duration_evidence"]})
        for record in copies:
            record.status, record.rationale = status, decision.reason
            record.evidence_refs, record.temporal_decision = list(decision.evidence_refs), decision
            if status != "met":
                record.symptom_ids = []
        missing_field = f"decision_tree.{nid}"
        generation.missing_dimensions = [m for m in generation.missing_dimensions if m.field != missing_field]
        if status == "pending":
            generation.missing_dimensions.append(MissingDimension(field=missing_field,
                purpose=definition["question"], suggested_question=decision.suggested_question))
    for candidate in generation.candidates:
        for path in candidate.decision_paths:
            mandatory = criterion_nodes(candidate.disorder, knowledge=knowledge)
            known = {r.node_id: r for r in path.decisions}
            path.provisional = any(nid not in known or known[nid].status != "met" for nid, definition in mandatory.items() if definition["mandatory"]) or any(
                r.status == "pending" for r in path.decisions if r.stage == "exclusion")
        generation.candidate_coverage[candidate.disorder] = "provisional_candidate" if any(p.provisional for p in candidate.decision_paths) else "candidate"
    return generation
