"""Knowledge-source isolation and provider propagation through the L3 pipeline."""
import copy
import json

import pytest

from shared.diagnostic_knowledge import (
    DiagnosticKnowledgeNotConnected, JsonDiagnosticKnowledgeProvider,
    UnconnectedDiagnosticKnowledgeProvider, get_default_knowledge_provider,
)
from shared.knowledge_sources import InsomniaSupplementSource, MentalKGSource
from layers.layer3.diagnosis import clinicr, decision_tree
from layers.layer3.llm import client
from tests.tree_test_support import node_payload


@pytest.mark.parametrize("source_type,label", [
    (MentalKGSource, "D013"), (InsomniaSupplementSource, "Insomnia"),
])
def test_each_source_reads_its_own_root_without_the_other_source(tmp_path, source_type, label):
    disorders = ({f"D{i:03d}": {"name": f"fixture-{i}"} for i in range(1, 24)}
                 if label == "D013" else {label: {"name": "fixture-insomnia"}})
    symptom_id = "S001" if label == "D013" else "S085"
    criteria = {"required_criteria": {"group": {"symptom_pool": [symptom_id]}}}
    symptoms = {symptom_id: {"description": "isolated source fixture"}}
    records = {
        "disorder.json": disorders,
        "diagnostic_criteria.json": {label: criteria},
        "symptom/fixture.json" if label == "D013" else "symptom/10insomnia.json": symptoms,
        f"differential_diagnosis/{label}.json": {label: ["fixture differential"]},
        "event.json": {"fixture_event": "isolated source"},
    }
    for relative, value in records.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    source = source_type(root=tmp_path)
    knowledge = source.get_candidate(label)
    assert source.disorders == disorders
    assert knowledge["criteria"] == criteria
    assert knowledge["symptoms"] == symptoms
    assert knowledge["differential"] == records[f"differential_diagnosis/{label}.json"]
    assert knowledge["events"] == records["event.json"]
    assert knowledge["source_ref"].endswith(f"diagnostic_criteria.json#/{label}")
    assert "metadata" not in knowledge


class InjectedKnowledge:
    """Changed test-only criteria make an accidental default lookup observable."""

    def __init__(self, label):
        self.base = JsonDiagnosticKnowledgeProvider()
        self.label = label
        self.calls = []

    def get_candidate(self, node):
        self.calls.append(node)
        result = copy.deepcopy(self.base.get_candidate(node))
        result["source_ref"] = f"injected:diagnostic_criteria.json#/{node}"
        if node == self.label:
            result["criteria"]["required_criteria"]["provider_fixture"] = "injected requirement"
        return result

    def get_symptoms(self):
        return self.base.get_symptoms()


@pytest.mark.parametrize("label", ["D013", "Insomnia"])
def test_injected_provider_reaches_generation_aggregation_verifier_and_report(monkeypatch, label):
    provider = InjectedKnowledge(label)
    prompts = []

    def forbid_default():
        pytest.fail("Injected Layer3 must not switch back to the default knowledge provider")

    for module in (clinicr, decision_tree, client):
        monkeypatch.setattr(module, "get_default_knowledge_provider", forbid_default)

    def controlled_api(prompt, **kwargs):
        prompts.append(prompt)
        if "当前节点 JSON：" in prompt:
            payload = node_payload(prompt, [label], "ev-fixture", complete_labels=[label])
        else:
            assert "Reward Model Verifier" in prompt
            payload = {
                "score": .9,
                "dimensions": [{"dimension": dimension, "score": .9,
                                "reason": "controlled fixture", "evidence_refs": ["ev-fixture"]}
                               for dimension in ["evidence_sufficiency", "logic_coherence",
                                                 "differential_coverage", "no_reasoning_leaps"]],
                "fail_reasons": [], "missing_dimensions": [],
            }
        return json.dumps(payload)

    monkeypatch.setattr(client, "ask_llm_stream", controlled_api)
    engine = clinicr.CLINICR(knowledge=provider)
    assert engine.reasoner.knowledge is provider
    assert engine.verifier.model.knowledge is provider
    report = engine.run({"evidence_nodes": [{"evidence_id": "ev-fixture"}]})

    assert report["diagnosis"]["primary"]["disorder"] == label
    code = f"criterion.{label}.provider_fixture"
    assert any(item["code"] == code for item in report["checked"][0]["criteria_detail"])
    source = next(item["source_doc"] for item in report["evidence_refs"]
                  if item.get("criterion_code") == code)
    assert source == f"injected:diagnostic_criteria.json#/{label}/required_criteria/provider_fixture"
    assert "injected requirement" in prompts[-1]
    assert f"injected:diagnostic_criteria.json#/{label}" in prompts[-1]
    assert label in provider.calls


def test_report_source_fallback_uses_the_same_injected_provider():
    provider = InjectedKnowledge("D013")
    # Exercise the report fallback for callers supplying records without a tree trace.
    from tests.tree_test_support import controlled_generation
    from layers.layer3.diagnosis.assessment import assessments_from_step1

    generation = controlled_generation(["D013"], complete_labels=["D013"])
    assessments = assessments_from_step1(
        generation, generation.candidates, knowledge=JsonDiagnosticKnowledgeProvider(),
    )
    report = clinicr.CLINICR(knowledge=provider)._result(
        candidates=generation.candidates, assessments=assessments,
        primary=None, provisional_primary=None, secondary=[], under_observation=[],
        ruled_out=[], severity={}, severity_level=1,
        verifier={"score": 0, "decision": "rejected", "routing": "return_to_l2",
                  "checks": [], "fail_reasons": [], "missing_dimensions": []},
        safety_context=None,
    )
    refs = [item["source_doc"] for item in report["evidence_refs"] if item["type"] == "DSM-5-criterion"]
    assert refs
    assert all(ref == "injected:diagnostic_criteria.json#/D013" for ref in refs)


def test_default_provider_has_no_automatic_source_connection():
    provider = get_default_knowledge_provider()
    assert isinstance(provider, UnconnectedDiagnosticKnowledgeProvider)
    with pytest.raises(DiagnosticKnowledgeNotConnected, match="尚未接入"):
        provider.get_candidate("D013")
    with pytest.raises(DiagnosticKnowledgeNotConnected, match="尚未接入"):
        provider.get_symptoms()


def test_default_layer3_stops_before_reading_knowledge_or_calling_llm(monkeypatch):
    from fastapi.testclient import TestClient
    from layers.layer3.api import app
    from tests.test_layer3_clinicr import _profile

    def forbidden(*args, **kwargs):
        pytest.fail("未显式接入知识时，不得读取知识内容或请求模型")

    monkeypatch.setattr(MentalKGSource, "get_candidate", forbidden)
    monkeypatch.setattr(InsomniaSupplementSource, "get_candidate", forbidden)
    monkeypatch.setattr(client, "ask_llm_stream", forbidden)
    with pytest.raises(client.CLINICRModelUnavailable, match="诊断知识尚未接入"):
        clinicr.CLINICR().run(_profile())

    response = TestClient(app).post("/diagnose", json=_profile())
    assert response.status_code == 503
    assert "诊断知识尚未接入" in response.json()["detail"]


def test_default_verifier_does_not_reconnect_knowledge(monkeypatch):
    from tests.test_layer3_llm_integration import _score_with_test_inputs

    monkeypatch.setattr(client, "ask_llm_stream", lambda *_a, **_k: pytest.fail("不得请求模型"))
    with pytest.raises(client.CLINICRModelUnavailable, match="诊断知识尚未接入"):
        _score_with_test_inputs(client.APIRewardModelVerifier())
