"""Screen denials must be complete before they can remove a tree branch."""
import json

import pytest
from pydantic import ValidationError

from layers.layer3.diagnosis.decision_tree import DecisionTreeCandidateGenerator, load_candidate_graph
from layers.layer3.diagnosis.schemas import TreeNodeJudgment
from layers.layer3.diagnosis.screening import validate_screen_denial_scope
from layers.layer3.llm import client
from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider
from tests.tree_test_support import _node_answers


def judgment(**updates):
    return TreeNodeJudgment.model_validate({
        "node_id": "screen.appearance", "status": "not_met",
        "rationale": "Only denies gaining weight", "evidence_refs": ["ev-denial"],
        **updates,
    })


@pytest.mark.parametrize("scope", ["partial", "none", "unknown"])
def test_incomplete_denial_is_downgraded_and_logged_before_pruning(scope, caplog):
    original = judgment(denial_scope=scope)
    effective, event = validate_screen_denial_scope(original, stage="screening")
    assert original.status == "not_met"
    assert effective.status == "pending" and effective.evidence_refs == ["ev-denial"]
    assert effective.denial_scope == scope
    assert event["event"] == "screen_denial_scope_downgrade"
    assert event["original_status"] == "not_met" and event["effective_status"] == "pending"
    assert caplog.records[0].node_id == original.node_id
    assert validate_screen_denial_scope(effective, stage="screening") == (effective, None)
    assert len(caplog.records) == 1


def test_legacy_missing_scope_is_unknown_and_cannot_prune():
    assert judgment().denial_scope == "unknown"
    assert validate_screen_denial_scope(judgment(), stage="screening")[0].status == "pending"
    with pytest.raises(ValidationError):
        judgment(denial_scope="probably_full")


@pytest.mark.parametrize("status,scope,stage", [
    ("not_met", "full", "screening"), ("pending", "partial", "screening"),
    ("met", "partial", "screening"), ("not_met", "unknown", "criterion"),
    ("not_met", "partial", "exclusion"), ("not_met", "none", "anchor"),
])
def test_valid_denials_positive_cues_and_non_screen_nodes_are_unchanged(status, scope, stage):
    original = judgment(status=status, denial_scope=scope)
    assert validate_screen_denial_scope(original, stage=stage) == (original, None)


@pytest.mark.parametrize("scope,expected", [("full", "not_met"), ("partial", "pending"), ("none", "pending"), ("unknown", "pending")])
def test_api_boundary_preserves_raw_status_and_exports_downgrade_event(monkeypatch, scope, expected):
    raw = judgment(denial_scope=scope).model_dump()
    monkeypatch.setattr(client, "ask_llm_stream", lambda *_a, **_k: json.dumps({"nodes": [raw]}))
    reasoner = client.APICLINICRReasoner(knowledge=JsonDiagnosticKnowledgeProvider(), screening_zero_knowledge=True)
    node = {"node_id": "screen.appearance", **load_candidate_graph()["nodes"]["screen.appearance"]}
    answers = reasoner._evaluate_tree_nodes({client.ALLOWED_EVIDENCE_REFS_KEY: ["ev-denial"]}, [node])
    assert answers[0].status == expected and raw["status"] == "not_met"
    assert len(reasoner.screen_denial_scope_events) == (scope != "full")


def test_scope_guard_does_not_hide_unknown_evidence(monkeypatch):
    raw = judgment(denial_scope="partial", evidence_refs=["invented"]).model_dump()
    calls = []
    def respond(*args, **kwargs):
        calls.append(1)
        return json.dumps({"nodes": [raw]})
    monkeypatch.setattr(client, "ask_llm_stream", respond)
    reasoner = client.APICLINICRReasoner(knowledge=JsonDiagnosticKnowledgeProvider(), screening_zero_knowledge=True)
    node = {"node_id": "screen.appearance", **load_candidate_graph()["nodes"]["screen.appearance"]}
    with pytest.raises(ValueError, match="白名单"):
        reasoner._evaluate_tree_nodes({client.ALLOWED_EVIDENCE_REFS_KEY: ["ev-denial"]}, [node])
    assert len(calls) == 2 and reasoner.screen_denial_scope_events == []


@pytest.mark.parametrize("scope,retained", [("full", False), ("partial", True), ("none", True), ("unknown", True)])
def test_custom_evaluator_cannot_bypass_screen_guard(scope, retained):
    def evaluate(nodes):
        payload = _node_answers(nodes, ["D017"], "ev-denial", {}, (), {})
        for item in payload["nodes"]:
            if item["node_id"] == "screen.appearance":
                item.update(status="not_met", denial_scope=scope)
        return [TreeNodeJudgment.model_validate(item) for item in payload["nodes"]]
    result = DecisionTreeCandidateGenerator(evaluate, knowledge=JsonDiagnosticKnowledgeProvider()).generate()
    assert ("D017" in [item.disorder for item in result.candidates]) == retained
    screen = next(item for item in result.tree_trace if item.node_id == "screen.appearance")
    assert screen.status == ("pending" if retained else "not_met")
    assert screen.denial_scope == scope
