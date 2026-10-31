from __future__ import annotations

from typing import Any

from layers.layer2.guards import L2ExitGuard
from layers.layer2.schemas import ClinicalIntake
from shared.clinical_profile import ClinicalProfile


SCALE_TO_L3 = {
    "PHQ-9": "PHQ9",
    "GAD-7": "GAD7",
    "PCL-5": "PCL5",
    "Y-BOCS": "Y_BOCS",
    "ISI": "ISI",
    "MDQ": "MDQ",
}


class L2ToL3Adapter:
    def __init__(self) -> None:
        self.guard = L2ExitGuard()

    def assert_valid(self, intake: ClinicalIntake) -> dict[str, Any]:
        return self.guard.assert_valid(intake)

    def to_profile(self, intake: ClinicalIntake) -> dict[str, Any]:
        self.assert_valid(intake)
        return self.serialize_profile(intake)

    @staticmethod
    def serialize_profile(intake: ClinicalIntake) -> dict[str, Any]:
        """Serialize the shared shape; this alone does not grant L2 exit permission.

        Offline benchmark preparation may serialize an explicitly incomplete
        intake. Production callers must continue using guarded ``to_profile``.
        """
        profile: dict[str, Any] = {
            "contract_version": intake.contract_version,
            "interview_date": intake.interview_date,
            "clinical_facts": [f.model_dump(mode="json") for f in intake.clinical_facts],
            "duration_evidence": [d.model_dump(mode="json") for d in intake.diagnostic_context.duration_evidence],
            "follow_up_tasks": [t.model_dump(mode="json") for t in intake.follow_up_tasks],
            "key_symptoms_extracted": [
                item.model_dump(mode="json") for item in intake.key_symptoms_extracted
            ],
            "symptom_duration_days": intake.diagnostic_context.symptom_duration_days,
            "functional_impairment": intake.diagnostic_context.functional_impairment,
            "mania_or_hypomania_history": intake.diagnostic_context.mania_or_hypomania_history,
            "substance_or_medical_cause": intake.diagnostic_context.substance_or_medical_cause,
            "differential_diagnoses_assessed": intake.diagnostic_context.differential_diagnoses_assessed,
            "evidence_conflicts": intake.diagnostic_context.evidence_conflicts,
            "diagnostic_details": intake.diagnostic_context.diagnostic_details,
            "patient_goals": intake.patient_goals,
            "emotions": [
                item.model_dump(mode="json") for item in intake.emotions
            ],
            "cognitive_patterns": [
                item.model_dump(mode="json") for item in intake.cognitive_patterns
            ],
            "key_narratives": [
                item.model_dump(mode="json") for item in intake.key_narratives
            ],
            "functional_impairments": [
                item.model_dump(mode="json")
                for item in intake.functional_impairments
            ],
            "motivation": intake.motivation.model_dump(mode="json"),
            "evidence_nodes": [
                node.model_dump(mode="json") for node in intake.evidence_nodes
            ],
        }
        for field in (
            "relapse_count",
            "rhythm_disturbance",
            "exposure_avoidant",
            "ritual_behaviors",
            "insomnia_chief_complaint",
            "chronic_months",
            "interpersonal_conflicts",
            "interpersonal_impairment",
            "single_focus",
            "psychotic_symptoms",
            "substance_active",
            "severe_dissociation",
        ):
            value = getattr(intake.diagnostic_context, field)
            if value is not None:
                profile[field] = value
        for scale_name, result in intake.scale_results.items():
            if result.status != "completed" or result.score is None:
                continue
            key = SCALE_TO_L3[scale_name]
            profile[key] = result.score
            profile[f"{key}_items"] = {
                f"item{number}": score for number, score in result.items.items()
            }
        ClinicalProfile.model_validate(profile)
        return profile
