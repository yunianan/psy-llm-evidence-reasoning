from __future__ import annotations

from layers.layer2.guards import L2ExitGuard
from layers.layer2.profile.sufficiency import missing_dimensions
from layers.layer2.schemas import ClinicalIntake, L2SessionState


def build_clinical_intake(state: L2SessionState) -> ClinicalIntake:
    missing = missing_dimensions(state)
    intake = ClinicalIntake(
        session_id=state.session_id,
        contract_version=state.contract_version, interview_date=state.interview_date,
        clinical_facts=state.clinical_facts, follow_up_tasks=state.follow_up_tasks,
        current_stage=state.current_stage,
        key_symptoms_extracted=list(state.symptoms.values()),
        emotions=state.emotions,
        cognitive_patterns=state.cognitive_patterns,
        patient_goals=[item.value for item in state.patient_goals],
        patient_goal_facts=state.patient_goals,
        key_narratives=state.key_narratives,
        functional_impairments=state.functional_impairments,
        focus_topics=[item.value for item in state.focus_topics],
        focus_topic_facts=state.focus_topics,
        change_talk=state.change_talk,
        sustain_talk=state.sustain_talk,
        motivation=state.motivation,
        scale_results=state.scale_results,
        diagnostic_context=state.diagnostic_context,
        evidence_nodes=state.evidence_nodes,
        information_sufficiency="sufficient" if not missing else "insufficient",
        missing_dimensions=missing,
        patient_confirmation=state.confirmation_status,
        safety_controls=state.active_safety_controls,
        flags=list(dict.fromkeys(state.flags + state.active_safety_controls)),
    )
    intake.l2_validation = L2ExitGuard().verify(intake)
    return intake
