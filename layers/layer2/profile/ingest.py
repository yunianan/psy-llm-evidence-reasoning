from __future__ import annotations

from typing import Any

from layers.layer2.evidence import EvidenceRegistry
from layers.layer2.schemas import (
    CorrectionCandidate,
    L2SessionState,
    ProfileFact,
    SemanticAnalysis,
    SemanticCandidate,
    SymptomEvidence,
)


from shared.disorder_catalog import SYMPTOMS
from shared.clinical_contract import ClinicalFact, STRUCTURED_FIELD_NAMES
from shared.temporal import parse_duration_records

ALLOWED_SYMPTOMS = {
    "depressed_mood",
    "anhedonia",
    "appetite_or_weight_change",
    "insomnia",
    "fatigue",
    "psychomotor_change",
    "worthlessness",
    "concentration_difficulty",
    "anxiety",
    "trauma_related",
    "obsession_compulsion",
    "elevated_mood",
    "passive_si",
}
ALLOWED_SYMPTOMS.update(SYMPTOMS)

ALLOWED_STRUCTURED_FIELDS = set(STRUCTURED_FIELD_NAMES) - {"scale_consent", "scale_answer_score", "confirmation_decision"}


class ProfileIngestor:
    CORRECTABLE_LISTS = {
        "emotions",
        "cognitive_patterns",
        "patient_goals",
        "key_narratives",
        "functional_impairments",
        "focus_topics",
    }

    def apply_corrections(
        self,
        state: L2SessionState,
        *,
        analysis: SemanticAnalysis,
        patient_text: str,
        message_id: str,
    ) -> list[str]:
        evidence_ids: list[str] = []
        for correction in analysis.corrections:
            if correction.evidence_quote not in patient_text:
                continue
            parent_ids = self._remove_target(state, correction)
            if parent_ids is None:
                continue
            node = EvidenceRegistry.add_patient_evidence(
                state,
                message_id=message_id,
                quote=correction.evidence_quote,
                target_path=f"corrections.{correction.target_path}",
                parent_ids=parent_ids,
            )
            evidence_ids.append(node.evidence_id)
        return evidence_ids

    def ingest(
        self,
        state: L2SessionState,
        *,
        analysis: SemanticAnalysis,
        patient_text: str,
        message_id: str,
    ) -> list[str]:
        evidence_ids: list[str] = []
        for candidate in analysis.symptoms:
            if candidate.value not in ALLOWED_SYMPTOMS:
                continue
            node = self._evidence(
                state, candidate, patient_text, message_id,
                f"symptoms.{candidate.value}",
            )
            if node is None:
                continue
            evidence_ids.append(node.evidence_id)
            existing = state.symptoms.get(candidate.value)
            if existing:
                if node.evidence_id not in existing.sources:
                    existing.sources.append(node.evidence_id)
                if candidate.evidence_quote not in existing.patient_quotes:
                    existing.patient_quotes.append(candidate.evidence_quote)
                for attribute, value in candidate.attributes.items():
                    previous = existing.attributes.get(attribute)
                    if previous is not None and previous != value:
                        self._conflict(state, f"symptoms.{candidate.value}.{attribute}", previous, value,
                                       list(existing.sources), node.evidence_id)
                existing.attributes.update(candidate.attributes)
            else:
                state.symptoms[candidate.value] = SymptomEvidence(
                    symptom=candidate.value,
                    sources=[node.evidence_id],
                    patient_quotes=[candidate.evidence_quote],
                    attributes=candidate.attributes,
                )
            symptom = state.symptoms[candidate.value]
            for attribute, value in candidate.attributes.items():
                slot = {"duration_days": "duration", "functional_impact": "impact", "present": "existence"}.get(attribute, attribute)
                if slot not in {"existence", "onset", "duration", "frequency", "severity", "impact", "trigger", "denial"}:
                    continue
                fact = ClinicalFact(fact_id=f"{candidate.value}:{slot}:{node.evidence_id}", symptom_id=candidate.value,
                    attribute=slot, state="absent" if value is False else "present", value=value,
                    raw_span=candidate.evidence_quote, evidence_refs=[node.evidence_id])
                if slot == "duration":
                    fact.duration = self._capture_duration(state, candidate.evidence_quote, node.evidence_id,
                                                           symptom_id=candidate.value, scope=f"symptom:{candidate.value}")
                    fact.state = "present" if fact.duration.parse_status == "ok" else "unknown"
                    fact.value = fact.duration.model_dump(mode="json")
                if any(node.evidence_id in c.get("new_evidence_refs", []) and
                       (c["target_path"].endswith(f".{attribute}") or (slot == "duration" and c["target_path"].startswith("duration_evidence.")))
                       for c in state.diagnostic_context.evidence_conflicts or []):
                    fact.state = "conflict"
                symptom.attribute_facts.append(fact)
                state.clinical_facts.append(fact)

        collections = (
            ("emotions", analysis.emotions),
            ("cognitive_patterns", analysis.cognitive_patterns),
            ("patient_goals", analysis.patient_goals),
            ("key_narratives", analysis.key_narratives),
            ("functional_impairments", analysis.functional_impairments),
            ("focus_topics", analysis.focus_topics),
            ("change_talk", analysis.change_talk),
            ("sustain_talk", analysis.sustain_talk),
        )
        for target, candidates in collections:
            destination: list[ProfileFact] = getattr(state, target)
            for candidate in candidates:
                existing_index = next(
                    (index for index, item in enumerate(destination)
                     if item.value == candidate.value),
                    None,
                )
                target_index = (
                    existing_index if existing_index is not None else len(destination)
                )
                node = self._evidence(
                    state, candidate, patient_text, message_id,
                    f"{target}.{target_index}",
                )
                if node is None:
                    continue
                evidence_ids.append(node.evidence_id)
                if existing_index is not None:
                    existing = destination[existing_index]
                    if node.evidence_id not in existing.sources:
                        existing.sources.append(node.evidence_id)
                    if candidate.evidence_quote not in existing.patient_quotes:
                        existing.patient_quotes.append(candidate.evidence_quote)
                else:
                    destination.append(ProfileFact(
                        value=candidate.value,
                        sources=[node.evidence_id],
                        patient_quotes=[candidate.evidence_quote],
                    ))

        for candidate in analysis.structured_fields:
            if candidate.field not in ALLOWED_STRUCTURED_FIELDS:
                continue
            if candidate.evidence_quote not in patient_text:
                continue
            if candidate.field == "symptom_duration_days":
                node = EvidenceRegistry.add_patient_evidence(state, message_id=message_id,
                    quote=candidate.evidence_quote, target_path="diagnostic_context.symptom_duration_days" if type(candidate.value) is int else "diagnostic_context.duration_evidence")
                evidence_ids.append(node.evidence_id)
                task = next((t for t in state.follow_up_tasks if t.status == "pending" and t.attribute == "duration"), None)
                sid = task.symptom_id if task else analysis.symptoms[0].value if len(analysis.symptoms) == 1 else ""
                record = self._capture_duration(state, candidate.evidence_quote, node.evidence_id, symptom_id=sid,
                    scope=task.node_id if task else f"symptom:{sid}" if sid else "unspecified")
                # 老整数只作兼容视图；区间始终保留，L3按作用域读取。
                if type(candidate.value) is int and record.parse_status == "ok" and record.min_days == record.max_days == candidate.value:
                    self._write_structured(state, candidate.field, candidate.value, node.evidence_id)
                continue
            if not self._valid_structured_value(candidate.field, candidate.value):
                continue
            node = EvidenceRegistry.add_patient_evidence(
                state,
                message_id=message_id,
                quote=candidate.evidence_quote,
                target_path=f"diagnostic_context.{candidate.field}",
            )
            evidence_ids.append(node.evidence_id)
            self._write_structured(state, candidate.field, candidate.value, node.evidence_id)
        # 省略回答也须按补采任务定向绑定；不要求LLM重新输出症状名。
        task = next((t for t in state.follow_up_tasks if t.status == "pending" and t.attribute == "duration"), None)
        if task:
            node = EvidenceRegistry.add_patient_evidence(state, message_id=message_id, quote=patient_text,
                                                        target_path=f"follow_up_tasks.{task.task_id}")
            evidence_ids.append(node.evidence_id)
            self._capture_duration(state, patient_text, node.evidence_id, symptom_id=task.symptom_id, scope=task.node_id)
        return evidence_ids

    @staticmethod
    def _conflict(state, target, old, new, old_refs, new_ref):
        state.diagnostic_context.evidence_conflicts = state.diagnostic_context.evidence_conflicts or []
        state.diagnostic_context.evidence_conflicts.append({"target_path": target, "previous": old, "new": new,
            "previous_evidence_refs": old_refs, "new_evidence_refs": [new_ref], "status": "unresolved"})
        for fact in state.clinical_facts:
            relevant = True
            if target.startswith("symptoms."):
                _, sid, attribute = target.split(".", 2)
                attribute = {"duration_days": "duration", "functional_impact": "impact", "present": "existence"}.get(attribute, attribute)
                relevant = fact.symptom_id == sid and fact.attribute == attribute
            elif target.startswith("duration_evidence."):
                relevant = fact.attribute == "duration"
            else:
                relevant = False
            if relevant and fact.evidence_refs and set(fact.evidence_refs).intersection(old_refs):
                fact.state = "conflict"

    def _capture_duration(self, state, quote, ref, *, symptom_id="", scope="unspecified"):
        records = parse_duration_records(quote, symptom_id=symptom_id, scope=scope, evidence_refs=[ref], interview_date=state.interview_date)
        for record in records:
            previous = [d for d in state.diagnostic_context.duration_evidence
                        if d.scope == scope and d.symptom_id == symptom_id and d.duration_kind == record.duration_kind]
            for old in previous:
                if old.parse_status == record.parse_status == "ok" and old.min_days is not None and record.min_days is not None:
                    if (old.max_days is not None and old.max_days < record.min_days) or (record.max_days is not None and record.max_days < old.min_days):
                        self._conflict(state, f"duration_evidence.{scope}", old.model_dump(mode="json"),
                                       record.model_dump(mode="json"), old.evidence_refs, ref)
            state.diagnostic_context.duration_evidence.append(record)
        for task in state.follow_up_tasks:
            if task.attribute == "duration" and task.status == "pending" and (not symptom_id or task.symptom_id in ("", symptom_id)):
                task.status = "answered" if all(r.parse_status == "ok" for r in records) else "unresolved"
                task.answer_evidence_refs.append(ref)
        return records[0]

    @staticmethod
    def _evidence(
        state: L2SessionState,
        candidate: SemanticCandidate,
        patient_text: str,
        message_id: str,
        target_path: str,
    ):
        if candidate.evidence_quote not in patient_text:
            return None
        return EvidenceRegistry.add_patient_evidence(
            state,
            message_id=message_id,
            quote=candidate.evidence_quote,
            target_path=target_path,
        )

    @staticmethod
    def _valid_structured_value(field: str, value: Any) -> bool:
        if field in {
            "symptom_duration_days",
            "relapse_count",
            "chronic_months",
            "interpersonal_conflicts",
        }:
            return (
                isinstance(value, int) and not isinstance(value, bool)
                and 0 <= value <= 36500
            )
        if field in {
            "functional_impairment",
            "mania_or_hypomania_history",
            "substance_or_medical_cause",
            "rhythm_disturbance",
            "exposure_avoidant",
            "ritual_behaviors",
            "insomnia_chief_complaint",
            "interpersonal_impairment",
            "single_focus",
            "psychotic_symptoms",
            "substance_active",
            "severe_dissociation",
        }:
            return isinstance(value, bool)
        if field == "differential_diagnoses_assessed":
            return isinstance(value, list) and all(isinstance(item, str) for item in value)
        if field == "diagnostic_details":
            return isinstance(value, dict) and all(isinstance(key, str) for key in value)
        if field == "evidence_conflicts":
            return isinstance(value, list) and all(isinstance(item, dict) for item in value)
        if field.startswith("motivation_"):
            return (
                isinstance(value, int) and not isinstance(value, bool)
                and 0 <= value <= 10
            )
        return False

    @staticmethod
    def _write_structured(
        state: L2SessionState, field: str, value: Any, evidence_id: str
    ) -> None:
        if field.startswith("motivation_"):
            name = field.removeprefix("motivation_")
            setattr(state.motivation, name, value)
            if evidence_id not in state.motivation.sources:
                state.motivation.sources.append(evidence_id)
            return
        if field == "diagnostic_details":
            state.diagnostic_context.diagnostic_details.update(value)
        else:
            old = getattr(state.diagnostic_context, field)
            if old is not None and old != value and field != "evidence_conflicts":
                ProfileIngestor._conflict(state, f"diagnostic_context.{field}", old, value,
                    list(state.diagnostic_context.sources.get(field, [])), evidence_id)
            setattr(state.diagnostic_context, field, value)
        state.diagnostic_context.sources.setdefault(field, []).append(evidence_id)

    def _remove_target(
        self, state: L2SessionState, correction: CorrectionCandidate
    ) -> list[str] | None:
        target = correction.target_path
        if target.startswith("symptoms."):
            symptom = target.split(".", 1)[1]
            if symptom not in ALLOWED_SYMPTOMS:
                return None
            existing = state.symptoms.pop(symptom, None)
            return list(existing.sources) if existing else []
        if target in self.CORRECTABLE_LISTS:
            existing = list(getattr(state, target))
            setattr(state, target, [])
            return [source for fact in existing for source in fact.sources]
        if target.startswith("motivation."):
            field = target.split(".", 1)[1]
            if field not in {"importance", "confidence", "readiness"}:
                return None
            parent_ids = list(state.motivation.sources)
            setattr(state.motivation, field, None)
            return parent_ids
        if target.startswith("diagnostic_context."):
            field = target.split(".", 1)[1]
            if field not in ALLOWED_STRUCTURED_FIELDS or field.startswith("motivation_"):
                return None
            existing_sources = state.diagnostic_context.sources.pop(field, [])
            setattr(state.diagnostic_context, field, None)
            return existing_sources
        if target.startswith("scale_results."):
            scale = target.split(".", 1)[1]
            existing = state.scale_results.get(scale)
            if existing is None:
                return []
            parent_ids = list(existing.evidence_ids)
            try:
                state.scale_results[scale] = type(existing)(scale_name=scale)
            except ValueError:
                return None
            if state.pending_scale_name == scale:
                state.pending_scale_name = None
            return parent_ids
        return None
