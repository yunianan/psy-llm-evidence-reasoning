import json

import pytest

from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider
from layers.layer3.diagnosis.decision_tree import criterion_nodes
from layers.layer3.llm import client
from tools.l3_computable_criteria import calculate, rule_for, validate_fact
from tools.l3_hybrid_criteria import ComputableCriteriaReasoner, pack_sources
from layers.layer3.diagnosis.decision_tree import load_candidate_graph
from layers.layer3.diagnosis.schemas import TreeNodeJudgment
from tests.tree_test_support import node_payload


def node(part="min_count", condition=2):
    return {"node_id": "criterion.fixture.group/" + part, "stage": "criterion", "source_condition": condition,
        "source_group": {}, "symptom_pool": ["S1", "S2", "S3"], "disorders": ["fixture"]}


def facts(rule, states):
    return {key: {"fact_id": key, "state": state, "evidence_refs": ["p1"] if state != "unknown" else [], "confidence": .8}
            for key, state in zip(rule["facts"], states)}


@pytest.mark.parametrize("states,expected", [
    (["present", "unknown", "unknown"], "pending"),
    (["present", "absent", "absent"], "not_met"),
    (["present", "present", "unknown"], "met"),
])
def test_count_uses_upper_bound_not_absence_of_anchor_ids(states, expected):
    definition = node()
    rule = rule_for(definition)
    result = calculate(definition, rule, facts(rule, states))
    assert result.status == expected
    assert bool(result.symptom_ids) == (expected == "met")


@pytest.mark.parametrize("part,states,expected", [
    ("must_include_one_of", ["unknown", "unknown", "present"], "pending"),
    ("must_include_one_of", ["absent", "absent", "unknown"], "not_met"),
    ("must_include_all", ["present", "unknown", "absent"], "pending"),
    ("must_include_all", ["present", "absent", "present"], "not_met"),
    ("must_include_all", ["present", "present", "unknown"], "met"),
])
def test_core_symptom_three_valued_logic(part, states, expected):
    definition = node(part, ["S1", "S2"])
    rule = rule_for(definition)
    assert calculate(definition, rule, facts(rule, states)).status == expected


@pytest.mark.parametrize("lower,upper,unit,expected", [
    (2, 2, "week", "met"), (1, 1, "week", "not_met"),
    (1, 3, "week", "pending"), (None, None, None, "pending"),
    (1, 1, "month", "pending"), (14, None, "day", "met"),
])
def test_duration_ranges_and_no_approximate_month_conversion(lower, upper, unit, expected):
    definition = node("min_duration", "2 weeks")
    rule = rule_for(definition)
    key = next(iter(rule["facts"]))
    value = {"fact_id": key, "lower": lower, "upper": upper, "unit": unit, "evidence_refs": ["p1"], "confidence": .8}
    validate_fact(value, rule["facts"][key], {"p1"})
    assert calculate(definition, rule, {key: value}).status == expected


def test_nonrequired_impairment_and_complex_temporal_conditions_stay_semantic():
    assert rule_for(node("functional_impairment_required", False)) is None
    assert rule_for(node("max_duration", "6 months after stressor ends")) is None


def test_known_negative_fact_needs_evidence_and_invalid_bounds_rejected():
    spec = {"kind": "symptom"}
    with pytest.raises(ValueError):
        validate_fact({"fact_id": "f", "state": "absent", "evidence_refs": [], "confidence": .5}, spec, {"p1"})
    with pytest.raises(ValueError):
        validate_fact({"fact_id": "d", "lower": 3, "upper": 1, "unit": "day", "evidence_refs": ["p1"], "confidence": .5}, {"kind": "duration"}, {"p1"})


def test_lossless_full_knowledge_pack_keeps_conflicting_insomnia_event():
    provider = JsonDiagnosticKnowledgeProvider()
    original = [provider.get_candidate(label) for label in ["D001", "D013", "Insomnia"]]
    packed = pack_sources(original)
    restored = []
    for entry in packed["sources"]:
        item = dict(entry["source"])
        for field, catalog in [("symptoms", "symptom_catalog"), ("events", "event_catalog")]:
            item[field] = {key: packed[catalog][target] for key, target in entry[field + "_bindings"].items()}
        restored.append(item)
    assert restored == original
    assert len(packed["event_catalog"]) == 4


def test_hybrid_maps_facts_once_then_computes_both_nodes(monkeypatch):
    provider = JsonDiagnosticKnowledgeProvider()
    nodes = [{"node_id": key, **definition} for key, definition in criterion_nodes("D013", knowledge=provider).items()
             if key.endswith("min_count") or key.endswith("must_include_one_of")]
    calls = []
    def respond(prompt, **kwargs):
        calls.append(prompt)
        packet, _ = json.JSONDecoder().raw_decode(prompt.split("原始知识与任务：", 1)[1])
        payload = {"facts": [{"fact_id": key, "state": "present", "evidence_refs": ["p1"], "confidence": .8}
                              for key in packet["requested_facts"]], "nodes": []}
        return json.dumps(payload)
    monkeypatch.setattr(client, "ask_llm_stream", respond)
    reasoner = ComputableCriteriaReasoner(knowledge=provider)
    judgments = reasoner._evaluate_tree_nodes({"key_narratives": [{"ref": "p1", "text": "synthetic"}], client.ALLOWED_EVIDENCE_REFS_KEY: ["p1"]}, nodes)
    assert len(calls) == 1 and all(j.status == "met" for j in judgments)
    assert len(reasoner.computation_log[0]["facts"]) == 9


def test_shared_prefix_rejects_unavailable_node_without_a_model_call(monkeypatch):
    monkeypatch.setattr(client, "ask_llm_stream", lambda *a, **kw: pytest.fail("no prefix model call"))
    reasoner = ComputableCriteriaReasoner(knowledge=JsonDiagnosticKnowledgeProvider(), shared_prefix={})
    with pytest.raises(ValueError, match="Shared prefix unavailable"):
        reasoner._evaluate_tree_nodes({}, [{"node_id": "screen.synthetic", "stage": "screening"}])


def test_full_tree_integration_preserves_candidate_and_reuses_identical_prefix(monkeypatch):
    calls = []
    def respond(prompt, **kwargs):
        calls.append(prompt)
        if "原始知识与任务：" not in prompt:
            return json.dumps(node_payload(prompt, labels=["D013"], ref="ev-mood"))
        packet, _ = json.JSONDecoder().raw_decode(prompt.split("原始知识与任务：", 1)[1])
        values = []
        for key, spec in packet["requested_facts"].items():
            value = {"fact_id": key, "evidence_refs": [], "confidence": .5}
            value.update(lower=None, upper=None, unit=None) if spec["kind"] == "duration" else value.update(state="unknown")
            values.append(value)
        return json.dumps({"facts": values, "nodes": []})
    monkeypatch.setattr(client, "ask_llm_stream", respond)
    provider = JsonDiagnosticKnowledgeProvider()
    profile = {"evidence_nodes": [{"evidence_id": "ev-mood", "quote": "synthetic"}], client.ALLOWED_EVIDENCE_REFS_KEY: ["ev-mood"]}
    full = client.APICLINICRReasoner(knowledge=provider, criterion_batching="per_label").generate_candidates(profile)
    graph = load_candidate_graph()
    prefix = {row.node_id: {"definition": {"node_id": row.node_id, **graph["nodes"][row.node_id]},
        "judgment": {key: row.model_dump()[key] for key in TreeNodeJudgment.model_fields}}
        for row in full.tree_trace if row.stage != "criterion"}
    start = len(calls)
    computed = ComputableCriteriaReasoner(knowledge=provider, shared_prefix=prefix).generate_candidates(profile)
    assert len(calls) - start == 1
    assert computed.candidate_coverage == full.candidate_coverage
    assert [(row.node_id, row.status) for row in computed.tree_trace] == [(row.node_id, row.status) for row in full.tree_trace]
