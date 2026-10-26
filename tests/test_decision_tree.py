from concurrent.futures import ThreadPoolExecutor
import copy
import json

import pytest

from layers.layer3.diagnosis.decision_tree import DecisionTreeCandidateGenerator, load_candidate_graph
from layers.layer3.diagnosis.schemas import TreeNodeJudgment
from layers.layer3.diagnosis.clinicr import CLINICR
from layers.layer3.llm import client
from shared.disorder_catalog import SUPPORTED_DISORDERS
from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider
from tests.tree_test_support import node_payload


def evaluator(labels=("D013",), overrides=None):
    def evaluate(nodes):
        prompt = "当前节点 JSON：" + json.dumps(nodes)
        return [TreeNodeJudgment.model_validate(item) for item in node_payload(
            prompt, labels, overrides=overrides
        )["nodes"]]
    return evaluate


def generator(evaluate):
    return DecisionTreeCandidateGenerator(evaluate, knowledge=JsonDiagnosticKnowledgeProvider())


@pytest.mark.parametrize("label", SUPPORTED_DISORDERS)
def test_each_label_reachable_as_provisional_candidate(label):
    result = generator(evaluator([label])).generate()
    assert [item.disorder for item in result.candidates] == [label]
    candidate = result.candidates[0]
    assert candidate.decision_paths[0].provisional
    assert candidate.symptom_match.matched == 1
    assert len(result.candidate_coverage) == 24
    assert result.missing_dimensions
    assert all(record.source_refs for record in result.tree_trace)


def test_confirmed_mania_excludes_unipolar_candidates_but_unknown_preserves():
    labels = ["D008", "D009", "D013", "D014", "D015"]
    pending = generator(evaluator(labels)).generate()
    assert len(pending.candidates) == 5
    confirmed = generator(evaluator(labels, {"history.mania": "met"})).generate()
    assert [item.disorder for item in confirmed.candidates] == ["D008"]
    assert confirmed.candidate_coverage["D013"] == "excluded:history.mania"


def test_insomnia_comorbidity_does_not_automatically_exclude():
    result = generator(evaluator(["D013", "Insomnia"])).generate()
    assert {item.disorder for item in result.candidates} == {"D013", "Insomnia"}
    result = generator(evaluator(["Insomnia"], {"sleep.explained": "met"})).generate()
    assert not result.candidates


def test_unknown_screen_can_reach_positive_anchor():
    result = generator(evaluator(["D013"], {"screen.mood": "pending"})).generate()
    assert [item.disorder for item in result.candidates] == ["D013"]


def test_no_supported_symptom_does_not_force_a_candidate():
    result = generator(evaluator([])).generate()
    assert result.candidates == []


def test_requests_do_not_mutate_cached_graph_even_on_failure():
    original = copy.deepcopy(load_candidate_graph())
    def fail_on_criteria(nodes):
        if any(node["stage"] == "criterion" for node in nodes):
            raise ValueError("provider failed")
        return evaluator()(nodes)
    with pytest.raises(ValueError):
        generator(fail_on_criteria).generate()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda label: generator(evaluator([label])).generate(), ["D013", "Insomnia"]))
    assert [[c.disorder for c in result.candidates] for result in results] == [["D013"], ["Insomnia"]]
    assert load_candidate_graph() == original


def test_empty_candidate_routes_to_l2_without_step2_or_reward():
    class EmptyReasoner:
        def generate_candidates(self, profile):
            return generator(evaluator([])).generate()
        def evaluate_candidate(self, *args):
            pytest.fail("no candidate should reach Step 2")
    result = CLINICR(reasoner=EmptyReasoner()).run({"evidence_nodes": [{"evidence_id": "ev-mood"}]})
    assert result["verifier_routing"] == "return_to_l2"
    assert result["candidates"] == []
    assert result["tree_trace"]
    assert result["missing_dimensions"]


@pytest.mark.parametrize("bad", ["foreign_ref", "extra_node", "foreign_symptom", "missing_evidence"])
def test_api_rejects_invalid_nodes_after_one_repair(monkeypatch, bad):
    calls = []
    def respond(prompt, **kwargs):
        calls.append(prompt)
        payload = node_payload(prompt, ref="ev-mood")
        node = payload["nodes"][0]
        if bad == "foreign_ref": node["evidence_refs"] = ["made-up"]
        if bad == "extra_node": payload["nodes"].append(dict(node, node_id="invented"))
        if bad == "foreign_symptom": node["symptom_ids"] = ["S999"]
        if bad == "missing_evidence": node["evidence_refs"] = []
        return json.dumps(payload)
    monkeypatch.setattr(client, "ask_llm_stream", respond)
    with pytest.raises(client.CLINICRModelUnavailable):
        client.APICLINICRReasoner(
            "offline", knowledge=JsonDiagnosticKnowledgeProvider(),
        ).generate_candidates({client.ALLOWED_EVIDENCE_REFS_KEY: ["ev-mood"]})
    assert len(calls) == 2


def test_failed_full_criterion_kept_visible_in_step1_evidence():
    result = generator(evaluator(["D013"], {"criterion.D013.depressive_episode/min_duration": "not_met"})).generate()
    assert result.candidates
    assert any(r.node_id == "criterion.D013.depressive_episode/min_duration" and r.status == "not_met" for r in result.tree_trace)


def test_claimed_symptom_count_cannot_exceed_supported_ids(monkeypatch):
    def respond(prompt, **kwargs):
        return json.dumps({"nodes": [{"node_id": "criterion.D013.depressive_episode/min_count",
                          "status": "met", "rationale": "claims five but returns one",
                          "evidence_refs": ["ev-mood"], "symptom_ids": ["S024"], "confidence": 0.9}]})
    monkeypatch.setattr(client, "ask_llm_stream", respond)
    with pytest.raises(ValueError, match="below source count"):
        client.APICLINICRReasoner("offline", knowledge=JsonDiagnosticKnowledgeProvider())._evaluate_tree_nodes(
            {client.ALLOWED_EVIDENCE_REFS_KEY: ["ev-mood"]}, [{
                "node_id": "criterion.D013.depressive_episode/min_count", "stage": "criterion",
                "disorders": ["D013"], "symptom_pool": ["S024", "S025", "S026", "S027", "S028"],
                "source_condition": 5, "question": "是否满足来源计数？",
            }])
