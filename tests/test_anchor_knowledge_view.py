"""Anchor evidence presence and bounded context, separate from diagnostic criteria."""
import copy
import json
from unittest.mock import Mock

import pytest

from layers.layer3.diagnosis.decision_tree import (
    DecisionTreeCandidateGenerator, anchor_frontier_state, load_candidate_graph,
)
from layers.layer3.diagnosis.schemas import TreeNodeJudgment
from layers.layer3.llm import client
from layers.layer3.llm.node_knowledge import ANCHOR_VIEW_VERSION, NodeKnowledgeView
from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider
from tests.tree_test_support import _node_answers, node_payload


def nodes(*ids):
    return [{"node_id": node, **copy.deepcopy(load_candidate_graph()["nodes"][node])} for node in ids]


def test_compact_anchor_deduplicates_pool_and_keeps_defining_qualifiers():
    provider = JsonDiagnosticKnowledgeProvider()
    symptoms = provider.get_symptoms()
    provider.get_candidate = Mock(side_effect=AssertionError("No complete label packages"))
    provider.get_symptoms = Mock(return_value=symptoms)
    batch = nodes("anchor.D001", "anchor.D003", "anchor.Insomnia")
    original = copy.deepcopy(batch)
    view = NodeKnowledgeView(provider, anchor_compact_view=True)
    result = view.sources_for_nodes(batch)
    assert len(result) == 1 and result[0]["view"] == ANCHOR_VIEW_VERSION
    assert {row["disorder"] for row in result[0]["labels"]} == {"D001", "D003", "Insomnia"}
    catalog = result[0]["symptom_catalog"]
    assert {row["id"] for row in catalog} == {sid for node in batch for sid in node["symptom_pool"]}
    assert len(catalog) == 21
    assert all(set(row) == {"id", "name", "description", "source"} for row in catalog)
    assert all(row["description"] == symptoms[row["id"]]["description"] for row in catalog)
    assert "adolescents or adults" in next(row["description"] for row in catalog if row["id"] == "S012")
    assert "criteria" not in result[0] and "events" not in result[0] and "differential" not in result[0]
    assert view.sources_for_nodes(batch) == result and provider.get_symptoms.call_count == 1
    provider.get_candidate.assert_not_called()
    assert batch == original


def test_anchor_defaults_to_full_and_other_stages_keep_prior_context():
    provider = Mock()
    provider.get_candidate.side_effect = lambda label: {"label": label, "criteria": "full"}
    assert NodeKnowledgeView(provider).sources_for_nodes(nodes("anchor.D013")) == [{"label": "D013", "criteria": "full"}]
    provider.get_candidate.assert_called_once_with("D013")
    provider.reset_mock()
    graph = load_candidate_graph()
    guards = [{"node_id": key, **value} for key, value in graph["nodes"].items() if value["stage"] == "exclusion"]
    batch = [guards[0], {"stage": "criterion", "disorders": ["D013"], "context_disorders": ["D011"]}]
    expected = list(dict.fromkeys(label for node in batch for label in [*node["disorders"], *node.get("context_disorders", [])]))
    result = NodeKnowledgeView(provider, anchor_compact_view=True).sources_for_nodes(batch)
    assert result == [{"label": label, "criteria": "full"} for label in expected]
    provider.get_symptoms.assert_not_called()


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("refs", [[], ["ev-patient"]])
def test_anchor_not_met_means_no_positive_evidence_and_does_not_require_denial(monkeypatch, compact, refs):
    calls = []
    def respond(prompt, **kwargs):
        calls.append(prompt)
        assert client.ANCHOR_GUIDANCE in prompt
        assert "不表示患者明确否认症状" in prompt
        return json.dumps({"nodes": [{"node_id": "anchor.D013", "status": "not_met",
                                      "rationale": "No pool symptom evidence in the supplied narrative",
                                      "evidence_refs": refs, "symptom_ids": [], "denial_scope": "unknown"}]})
    monkeypatch.setattr(client, "ask_llm_stream", respond)
    result = client.APICLINICRReasoner(knowledge=JsonDiagnosticKnowledgeProvider(), anchor_compact_view=compact)._evaluate_tree_nodes(
        {client.ALLOWED_EVIDENCE_REFS_KEY: ["ev-patient"]}, nodes("anchor.D013"))
    assert result[0].status == "not_met" and len(calls) == 1


@pytest.mark.parametrize("bad", ["no_evidence", "no_symptom", "foreign_reference", "foreign_symptom"])
def test_anchor_positive_evidence_and_whitelist_requirements_remain_strict(monkeypatch, bad):
    calls = []
    def respond(prompt, **kwargs):
        calls.append(prompt)
        item = {"node_id": "anchor.D013", "status": "met", "rationale": "fixture",
                "evidence_refs": ["ev-patient"], "symptom_ids": ["S024"]}
        if bad == "no_evidence": item["evidence_refs"] = []
        if bad == "no_symptom": item["symptom_ids"] = []
        if bad == "foreign_reference": item.update(status="not_met", symptom_ids=[], evidence_refs=["invented"])
        if bad == "foreign_symptom": item["symptom_ids"] = ["S001"]
        return json.dumps({"nodes": [item]})
    monkeypatch.setattr(client, "ask_llm_stream", respond)
    with pytest.raises(ValueError):
        client.APICLINICRReasoner(knowledge=JsonDiagnosticKnowledgeProvider(), anchor_compact_view=True)._evaluate_tree_nodes(
            {client.ALLOWED_EVIDENCE_REFS_KEY: ["ev-patient"]}, nodes("anchor.D013"))
    assert len(calls) == 2


def test_executor_accepts_anchor_absence_without_patient_denial():
    def evaluate(batch):
        payload = _node_answers(batch, ["D013"], "ev-patient", {}, (), {})
        for item in payload["nodes"]:
            if item["node_id"] == "anchor.D013":
                item.update(status="not_met", evidence_refs=[], symptom_ids=[])
        return [TreeNodeJudgment.model_validate(item) for item in payload["nodes"]]
    result = DecisionTreeCandidateGenerator(evaluate, knowledge=JsonDiagnosticKnowledgeProvider()).generate()
    assert result.candidates == []
    assert next(item for item in result.tree_trace if item.node_id == "anchor.D013").evidence_refs == []


def test_post_anchor_snapshot_compares_paths_and_coverage_not_screen_status():
    graph = load_candidate_graph()
    judgments = {node: TreeNodeJudgment(node_id=node, status="pending", rationale="fixture")
                 for node, definition in graph["nodes"].items() if definition["stage"] in ["screening", "anchor"]}
    judgments["anchor.D013"] = TreeNodeJudgment(node_id="anchor.D013", status="met", rationale="fixture", evidence_refs=["ev"], symptom_ids=["S024"])
    judgments["anchor.D001"] = TreeNodeJudgment(node_id="anchor.D001", status="not_met", rationale="no evidence")
    active, coverage = anchor_frontier_state(graph, judgments)
    assert set(active) == {"D013"} and len(coverage) == 24
    assert coverage["D013"] == "anchor_supported"
    assert coverage["D001"] == "entry_pending"
    assert coverage["D002"] == "anchor_pending"


def test_screen_downgrades_survive_later_frontiers_and_reset_once_per_generation(monkeypatch):
    def respond(prompt, **kwargs):
        payload = node_payload(prompt, ref="ev-patient")
        for item in payload["nodes"]:
            if item["node_id"] == "screen.appearance":
                item.update(status="not_met", denial_scope="partial", evidence_refs=["ev-patient"])
        return json.dumps(payload)
    monkeypatch.setattr(client, "ask_llm_stream", respond)
    reasoner = client.APICLINICRReasoner(knowledge=JsonDiagnosticKnowledgeProvider())
    for _ in range(2):
        result = reasoner.generate_candidates({client.ALLOWED_EVIDENCE_REFS_KEY: ["ev-patient"]})
        assert any(item.stage == "criterion" for item in result.tree_trace)
        assert len(reasoner.screen_denial_scope_events) == 1
        assert reasoner.screen_denial_scope_events[0]["node_id"] == "screen.appearance"
