import json

import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from shared.disorder_catalog import (
    DISORDERS, MENTALKG_SYMPTOMS, SUPPORTED_DISORDERS, SYMPTOMS,
    TREATMENT_KEY_BY_NODE, grouped_catalog,
)
from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider, candidate_knowledge
from shared.knowledge_sources import MENTALKG_ROOT
from layers.layer3.api import app, diagnose
from layers.layer3.diagnosis.clinicr import CLINICR
from layers.layer3.diagnosis.schemas import CandidateGeneration, EliminativeAssessment
from layers.layer3.diagnosis.verifier import IndependentVerifier
from layers.layer4.schemas.l3_report import DiagnosticReport
from layers.layer4.validation.l3_to_l4 import L3ToL4Gate
from layers.layer2.schemas.models import L2SessionState, SemanticAnalysis
from layers.layer2.profile.ingest import ProfileIngestor
from tests.l3_test_double import ControlledRewardModelVerifier
from tests.test_layer3_clinicr import _profile


def test_catalog_uses_exact_upstream_mapping_and_nine_display_groups():
    assert DISORDERS == json.loads((MENTALKG_ROOT / "disorder.json").read_text(encoding="utf-8"))
    assert SUPPORTED_DISORDERS == (*tuple(f"D{i:03d}" for i in range(1, 24)), "Insomnia")
    groups = grouped_catalog()
    assert len(groups) == 9
    by_group = {group["group"]: group for group in groups}
    assert [item["disorder"] for item in by_group["anxiety"]["disorders"]] == ["D011", "D012"]
    assert [item["disorder"] for item in by_group["sleep"]["disorders"]] == ["Insomnia"]
    assert by_group["anxiety"]["label"] == "焦虑相关"
    assert by_group["sleep"]["label"] == "睡眠相关"
    assert [item["disorder"] for group in groups for item in group["disorders"]].count("Insomnia") == 1
    assert {item["disorder"] for group in groups for item in group["disorders"]} == set(SUPPORTED_DISORDERS)
    assert len(MENTALKG_SYMPTOMS) == 84
    assert len(SYMPTOMS) == 87


def test_catalog_endpoint_and_raw_knowledge():
    data = TestClient(app).get("/diagnostic-catalog").json()
    assert len(data["supported_disorders"]) == 24
    assert len(data["diagnostic_catalog"]) == 9
    by_group = {group["group"]: group for group in data["diagnostic_catalog"]}
    assert {item["display_group"] for item in by_group["anxiety"]["disorders"]} == {"anxiety"}
    assert by_group["sleep"]["disorders"][0]["display_name"] == "睡眠相关 · 失眠障碍"
    assert data["diagnostic_standard"] == "DSM-5"
    original = json.loads((MENTALKG_ROOT / "diagnostic_criteria.json").read_text(encoding="utf-8"))
    for node in DISORDERS:
        assert candidate_knowledge(node)["criteria"] == original[node]
    insomnia = candidate_knowledge("Insomnia")
    assert insomnia["criteria"]["name"] == "Insomnia Disorder"
    assert insomnia["source_ref"].endswith("diagnostic_criteria.json#/Insomnia")


class NodeReasoner:
    def __init__(self, node):
        self.node = node

    def generate_candidates(self, profile):
        from tests.tree_test_support import controlled_generation
        return controlled_generation(
            [self.node], ref="patient_utterance_L008", complete_labels=[self.node],
            confidences={self.node: .9},
        )


@pytest.mark.parametrize("node", SUPPORTED_DISORDERS)
def test_all_twenty_four_nodes_complete_diagnostic_pipeline(node):
    engine = CLINICR(reasoner=NodeReasoner(node),
                     verifier=IndependentVerifier(model=ControlledRewardModelVerifier()),
                     knowledge=JsonDiagnosticKnowledgeProvider())
    report = diagnose(_profile(), clinicr=engine)
    parsed = DiagnosticReport.model_validate(report)
    assert parsed.diagnosis.primary.disorder == node
    assert parsed.diagnosis.primary.name
    assert parsed.diagnosis.primary.display_name
    assert len(parsed.considered_disorders) == 24
    assert parsed.candidates[0]["disorder"] == node
    assert parsed.checked[0]["disorder"] == node
    assert parsed.checked[0]["source"] == "step1_tree_records"
    assert parsed.verifier_decision == "confirmed"
    if node in TREATMENT_KEY_BY_NODE:
        assert report["l3_validation"]["passed"]
        assert L3ToL4Gate().validate(parsed)["passed"]
        assert parsed.to_anchor().primary == node
    else:
        assert parsed.verifier_routing == "manual_review"
        assert "treatment_mapping_unavailable" in parsed.flags
        assert not parsed.recommended_therapy.primary
        assert not L3ToL4Gate().validate(parsed)["passed"]


@pytest.mark.parametrize("removed", ["MDD", "Bipolar", "PanicDisorder", "SocialAnxietyDisorder", "SomaticSymptomDisorder"])
def test_old_labels_are_not_active_candidates(removed):
    with pytest.raises(ValueError):
        NodeReasoner(removed).generate_candidates({})


def test_new_symptoms_and_episode_context_are_not_discarded_by_l2():
    state = L2SessionState(session_id="catalog-test", safety_passport={}, turn_index=1)
    analysis = SemanticAnalysis.model_validate({
        "input_type": "free_narrative",
        "symptoms": [{"value": "S001", "evidence_quote": "我经常漏看题目"}],
        "structured_fields": [{"field": "diagnostic_details", "value": {"onset_age": 10},
                               "evidence_quote": "十岁时就这样"}],
    })
    ProfileIngestor().ingest(state, analysis=analysis, patient_text="我经常漏看题目，十岁时就这样",
                            message_id="m1")
    assert "S001" in state.symptoms
    assert state.symptoms["S001"].patient_quotes == ["我经常漏看题目"]
    assert state.diagnostic_context.diagnostic_details == {"onset_age": 10}
    assert state.diagnostic_context.sources["diagnostic_details"]
