from __future__ import annotations

from typing import Any

from layers.layer2.scales.catalog import SCALE_CATALOG
from layers.layer2.schemas import ClinicalIntake
from shared.l2_l3_contract import L2_L3_REQUIRED_DIAGNOSTIC_FIELDS, requirement_value


class L2ExitGuardError(ValueError):
    pass


class L2ExitGuard:
    """L2→L3 硬守卫；不判断诊断，只验证画像质量和证据完整性。"""

    def verify(self, intake: ClinicalIntake) -> dict[str, Any]:
        violations: list[str] = []
        evidence_by_id = {
            node.evidence_id: node for node in intake.evidence_nodes
        }
        evidence_ids = set(evidence_by_id)

        if intake.information_sufficiency != "sufficient":
            violations.append("患者画像信息不足")
        if intake.patient_confirmation != "confirmed":
            violations.append("患者尚未确认画像")
        if intake.missing_dimensions:
            violations.append("仍有未完成的信息维度")

        for requirement in L2_L3_REQUIRED_DIAGNOSTIC_FIELDS:
            value = requirement_value(intake.diagnostic_context, requirement.field)
            if value is None:
                violations.append(f"缺少L3必需字段：{requirement.missing_label}")
                continue
            sources = intake.diagnostic_context.sources.get(requirement.field, [])
            if not sources and not (requirement.field == "symptom_duration_days" and intake.diagnostic_context.duration_evidence):
                violations.append(
                    f"L3必需字段 {requirement.field} 缺少证据"
                )

        for duration in intake.diagnostic_context.duration_evidence:
            if not duration.evidence_refs or any(ref not in evidence_by_id or duration.raw_span not in evidence_by_id[ref].quote
                                                 for ref in duration.evidence_refs):
                violations.append("病程区间缺少匹配的患者原文证据")
        for fact in intake.clinical_facts:
            if not fact.evidence_refs or any(ref not in evidence_by_id or fact.raw_span not in evidence_by_id[ref].quote
                                             for ref in fact.evidence_refs):
                violations.append("症状属性缺少匹配的患者原文证据")

        for symptom in intake.key_symptoms_extracted:
            if not set(symptom.sources).issubset(evidence_ids):
                violations.append(f"症状 {symptom.symptom} 的证据节点不存在")
            if not symptom.patient_quotes:
                violations.append(f"症状 {symptom.symptom} 缺少患者原话")
            for source in symptom.sources:
                node = evidence_by_id.get(source)
                if node is None:
                    continue
                if node.target_path != f"symptoms.{symptom.symptom}":
                    violations.append(
                        f"症状 {symptom.symptom} 的证据指向其他画像字段"
                    )
                if node.quote not in symptom.patient_quotes:
                    violations.append(
                        f"症状 {symptom.symptom} 的证据原话与画像不一致"
                    )

        fact_groups = (
            ("emotions", intake.emotions),
            ("cognitive_patterns", intake.cognitive_patterns),
            ("patient_goals", intake.patient_goal_facts),
            ("key_narratives", intake.key_narratives),
            ("functional_impairments", intake.functional_impairments),
            ("focus_topics", intake.focus_topic_facts),
            ("change_talk", intake.change_talk),
            ("sustain_talk", intake.sustain_talk),
        )
        for prefix, facts in fact_groups:
            for index, fact in enumerate(facts):
                if not set(fact.sources).issubset(evidence_ids):
                    violations.append(f"{prefix} 字段引用了不存在的证据")
                for source in fact.sources:
                    node = evidence_by_id.get(source)
                    if node is None:
                        continue
                    if node.target_path != f"{prefix}.{index}":
                        violations.append(f"{prefix} 的证据指向其他画像字段")
                    if node.quote not in fact.patient_quotes:
                        violations.append(f"{prefix} 的证据原话与画像不一致")

        for name, result in intake.scale_results.items():
            if result.status != "completed":
                continue
            definition = SCALE_CATALOG.get(name)
            if definition is None:
                violations.append(f"未知量表：{name}")
                continue
            expected_items = set(range(1, len(definition.items) + 1))
            if set(result.items) != expected_items:
                violations.append(f"{name} 量表题目不完整")
                continue
            for number, score in result.items.items():
                item = definition.items[number - 1]
                if not item.min_score <= score <= item.max_score:
                    violations.append(f"{name} 第{number}题分值越界")
            expected_score = definition.total_score(result.items)
            if result.score != expected_score:
                violations.append(f"{name} 总分与逐题分值不一致")
            if not set(result.evidence_ids).issubset(evidence_ids):
                violations.append(f"{name} 引用了不存在的证据节点")
            expected_targets = {
                f"scale_results.{name}.items.{number}" for number in result.items
            }
            actual_targets = {
                evidence_by_id[source].target_path
                for source in result.evidence_ids
                if source in evidence_by_id
            }
            if expected_targets != actual_targets:
                violations.append(f"{name} 逐题证据与题号不一致")

        context_values = {
            "symptom_duration_days": intake.diagnostic_context.symptom_duration_days,
            "functional_impairment": intake.diagnostic_context.functional_impairment,
            "mania_or_hypomania_history": intake.diagnostic_context.mania_or_hypomania_history,
            "substance_or_medical_cause": intake.diagnostic_context.substance_or_medical_cause,
            "differential_diagnoses_assessed": intake.diagnostic_context.differential_diagnoses_assessed,
            "evidence_conflicts": intake.diagnostic_context.evidence_conflicts,
            "relapse_count": intake.diagnostic_context.relapse_count,
            "rhythm_disturbance": intake.diagnostic_context.rhythm_disturbance,
            "exposure_avoidant": intake.diagnostic_context.exposure_avoidant,
            "ritual_behaviors": intake.diagnostic_context.ritual_behaviors,
            "insomnia_chief_complaint": intake.diagnostic_context.insomnia_chief_complaint,
            "chronic_months": intake.diagnostic_context.chronic_months,
            "interpersonal_conflicts": intake.diagnostic_context.interpersonal_conflicts,
            "interpersonal_impairment": intake.diagnostic_context.interpersonal_impairment,
            "single_focus": intake.diagnostic_context.single_focus,
            "psychotic_symptoms": intake.diagnostic_context.psychotic_symptoms,
            "substance_active": intake.diagnostic_context.substance_active,
            "severe_dissociation": intake.diagnostic_context.severe_dissociation,
        }
        for field, value in context_values.items():
            if value is None:
                continue
            sources = intake.diagnostic_context.sources.get(field, [])
            if not sources:
                violations.append(f"结构化字段 {field} 缺少证据")
            for source in sources:
                node = evidence_by_id.get(source)
                if node is None:
                    violations.append(f"结构化字段 {field} 引用了不存在的证据")
                elif node.target_path != f"diagnostic_context.{field}":
                    violations.append(f"结构化字段 {field} 的证据指向错误")

        motivation_values = (
            intake.motivation.importance,
            intake.motivation.confidence,
            intake.motivation.readiness,
        )
        if any(value is not None for value in motivation_values):
            if not intake.motivation.sources:
                violations.append("改变动机评分缺少证据")
            elif not set(intake.motivation.sources).issubset(evidence_ids):
                violations.append("改变动机评分引用了不存在的证据")

        conflicts = intake.diagnostic_context.evidence_conflicts or []
        if any(item.get("resolved") is not True for item in conflicts):
            violations.append("存在未解决的证据冲突")

        return {
            "status": "pass" if not violations else "fail",
            "route": "allow_l3" if not violations else "return_to_l2",
            "violations": violations,
            "evidence_node_count": len(evidence_ids),
        }

    def assert_valid(self, intake: ClinicalIntake) -> dict[str, Any]:
        result = self.verify(intake)
        if result["status"] != "pass":
            raise L2ExitGuardError("；".join(result["violations"]))
        return result
