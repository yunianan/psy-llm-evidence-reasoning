import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from layers.layer3.diagnosis.decision_tree import positive_symptom_ids, criterion_nodes
from layers.layer3.diagnosis.g1_cache import (
    G1NodeCache, G1CacheMiss, CachedG1Reasoner, cache_context, knowledge_hash,
)
from layers.layer3.diagnosis.schemas import TreeNodeJudgment
from layers.layer3.llm import client
from shared.diagnostic_knowledge import (
    JsonDiagnosticKnowledgeProvider, UnconnectedDiagnosticKnowledgeProvider, DiagnosticKnowledgeNotConnected,
)
from tests.tree_test_support import controlled_generation
from tools.evaluate_candidate_predictions import evaluate


@pytest.mark.parametrize("status", ["pending", "not_met"])
def test_node_schema_rejects_positive_symptoms_for_unsupported_status(status):
    with pytest.raises(ValidationError):
        TreeNodeJudgment(node_id="n", status=status, rationale="fixture", symptom_ids=["S024"])
    assert TreeNodeJudgment(node_id="n", status=status, rationale="fixture").symptom_ids == []


def test_aggregation_ignores_ids_from_unvalidated_pending_and_negative_nodes():
    records = [SimpleNamespace(status="met", symptom_ids=["S024"]),
               SimpleNamespace(status="pending", symptom_ids=["S025"]),
               SimpleNamespace(status="not_met", symptom_ids=["S026"])]
    assert positive_symptom_ids(records, {"S024", "S025", "S026"}) == {"S024"}


def test_pending_output_is_repaired_without_promoting_status_or_dropping_reference(monkeypatch):
    calls = []
    def answer(prompt, **kwargs):
        calls.append(prompt)
        return json.dumps({"nodes": [{"node_id": "criterion.D013.depressive_episode/min_count", "status": "pending",
                                      "rationale": "Not enough known symptoms", "evidence_refs": ["ev"],
                                      "symptom_ids": ["S024"] if len(calls) == 1 else []}]})
    monkeypatch.setattr(client, "ask_llm_stream", answer)
    provider = JsonDiagnosticKnowledgeProvider()
    defs = criterion_nodes("D013", knowledge=provider)
    key = "criterion.D013.depressive_episode/min_count"
    result = client.APICLINICRReasoner(knowledge=provider)._evaluate_tree_nodes(
        {client.ALLOWED_EVIDENCE_REFS_KEY: ["ev"]}, [{"node_id": key, **defs[key]}])
    assert len(calls) == 2 and "不能为了保留症状 ID 把 pending 改成 met" in calls[-1]
    assert result[0].status == "pending" and result[0].evidence_refs == ["ev"] and result[0].symptom_ids == []


def test_persistent_invalid_output_fails_after_one_repair(monkeypatch):
    call = Mock(return_value=json.dumps({"nodes": [{"node_id": "n", "status": "pending", "rationale": "fixture", "symptom_ids": ["S024"]}]}))
    monkeypatch.setattr(client, "ask_llm_stream", call)
    with pytest.raises(ValueError):
        client.APICLINICRReasoner(knowledge=JsonDiagnosticKnowledgeProvider())._evaluate_tree_nodes(
            {}, [{"node_id": "n", "stage": "criterion", "disorders": ["D013"], "symptom_pool": ["S024"]}])
    assert call.call_count == 2


def test_per_label_batch_keeps_full_package_once_and_preserves_all_nodes():
    provider = JsonDiagnosticKnowledgeProvider()
    provider.get_candidate = Mock(wraps=provider.get_candidate)
    nodes = [{"node_id": key, **value} for label in ["D006", "D013"]
             for key, value in criterion_nodes(label, knowledge=provider).items()]
    provider.get_candidate.reset_mock()
    reasoner = client.APICLINICRReasoner(knowledge=provider, criterion_batching="per_label")
    batches = reasoner.node_batches(nodes)
    assert [len(batch) for batch in batches] == [17, 4]
    assert [node["node_id"] for batch in batches for node in batch] == [node["node_id"] for node in nodes]
    for batch in batches:
        source = reasoner.node_knowledge_view.sources_for_nodes(batch)
        assert len(source) == 1 and all(key in source[0] for key in ["criteria", "symptoms", "differential", "events"])
    assert provider.get_candidate.call_count == 2
    assert [len(b) for b in client.APICLINICRReasoner(knowledge=provider).node_batches(nodes)] == [8, 8, 5]


def test_static_prefix_keeps_complete_knowledge_before_patient_and_changes_no_source(monkeypatch):
    from tests.tree_test_support import node_payload
    prompts = []
    def answer(prompt, **kwargs):
        prompts.append(prompt)
        return json.dumps(node_payload(prompt, labels=["D013"], ref="ev"))
    monkeypatch.setattr(client, "ask_llm_stream", answer)
    provider = JsonDiagnosticKnowledgeProvider()
    nodes = [{"node_id": key, **value} for key, value in criterion_nodes("D013", knowledge=provider).items()]
    for text in ["first synthetic narrative", "second synthetic narrative"]:
        client.APICLINICRReasoner(knowledge=provider, criterion_batching="per_label", static_knowledge_prefix=True)._evaluate_tree_nodes(
            {client.ALLOWED_EVIDENCE_REFS_KEY: ["ev"], "key_narratives": [{"text": text, "sources": ["ev"]}]}, nodes)
    prefix = prompts[0].split("允许的 evidence_refs（", 1)[0]
    assert prompts[1].startswith(prefix)
    assert json.dumps([provider.get_candidate("D013")], ensure_ascii=False) in prefix
    assert prompts[0].index("原始来源知识：") < prompts[0].index("患者画像：")
    assert "first synthetic narrative" in prompts[0] and "second synthetic narrative" in prompts[1]


def fixture_cache(tmp_path):
    provider = JsonDiagnosticKnowledgeProvider()
    profile = {client.ALLOWED_EVIDENCE_REFS_KEY: ["ev"]}
    context = cache_context("case", profile, model="fixture", service_url="https://fixture.invalid/v1",
                            knowledge_digest=knowledge_hash(provider), prompt_version="fixture-v1", criterion_batching="per_label")
    generation = controlled_generation(["D013"], ref="ev")
    delegate = client.APICLINICRReasoner(model="fixture", knowledge=provider, criterion_batching="per_label")
    delegate.generate_candidates = Mock(return_value=generation)
    return provider, profile, context, generation, delegate, G1NodeCache(tmp_path)


def test_a2_a3_reuse_same_node_artifact_and_rebuild_candidates_without_model_calls(tmp_path):
    provider, profile, context, expected, delegate, cache = fixture_cache(tmp_path)
    a1 = CachedG1Reasoner(cache, context, knowledge=provider, delegate=delegate, mode="record")
    assert a1.generate_candidates(profile).model_dump() == expected.model_dump()
    payload = json.loads(cache.path_for(context).read_text(encoding="utf-8"))
    assert "candidates" not in payload and "final_diagnoses" not in payload and "nodes" in payload
    for _arm in ["A2", "A3"]:
        reused = CachedG1Reasoner(cache, context, knowledge=provider, delegate=delegate, mode="require")
        assert reused.generate_candidates(profile).model_dump() == expected.model_dump()
        assert reused.cache_hit
    assert delegate.generate_candidates.call_count == 1


def test_failed_g1_preserves_screen_downgrades_without_caching_partial_nodes(tmp_path):
    provider, profile, context, generation, delegate, cache = fixture_cache(tmp_path)
    event = {"node_id": "screen.appearance", "event": "screen_denial_scope_downgrade"}
    def fail_after_screen(_profile):
        delegate.screen_denial_scope_events.append(event)
        raise RuntimeError("synthetic transport failure after screening")
    delegate.generate_candidates.side_effect = fail_after_screen
    reasoner = CachedG1Reasoner(cache, context, knowledge=provider, delegate=delegate, mode="record")
    with pytest.raises(RuntimeError, match="synthetic transport failure"):
        reasoner.generate_candidates(profile)
    assert reasoner.screen_denial_scope_events == [event]
    assert reasoner.screen_denial_scope_events is not delegate.screen_denial_scope_events
    assert not cache.path_for(context).exists()


@pytest.mark.parametrize("field", ["case_id", "profile_hash", "model", "service_url", "tree_version", "prompt_version", "knowledge_hash", "criterion_batching"])
def test_changed_cache_version_does_not_reuse_previous_g1(tmp_path, field):
    provider, profile, context, generation, delegate, cache = fixture_cache(tmp_path)
    cache.save(context, generation.tree_trace)
    changed = {**context, field: "different"}
    assert cache.load(changed) is None


def test_missing_shared_cache_never_calls_model_and_unconnected_provider_stays_blocked(tmp_path):
    provider, profile, context, generation, delegate, cache = fixture_cache(tmp_path)
    with pytest.raises(G1CacheMiss):
        CachedG1Reasoner(cache, context, knowledge=provider, delegate=delegate).generate_candidates(profile)
    delegate.generate_candidates.assert_not_called()
    cache.save(context, generation.tree_trace)
    with pytest.raises(DiagnosticKnowledgeNotConnected):
        CachedG1Reasoner(cache, context, knowledge=UnconnectedDiagnosticKnowledgeProvider()).generate_candidates(profile)


def test_cache_integrity_and_frozen_result_conflicts_fail_closed(tmp_path):
    provider, profile, context, generation, delegate, cache = fixture_cache(tmp_path)
    cache.save(context, generation.tree_trace)
    changed = copy.deepcopy(generation.tree_trace)
    changed[0].rationale = "another model sample"
    with pytest.raises(ValueError, match="never replace"):
        cache.save(context, changed)
    path = cache.path_for(context)
    content = json.loads(path.read_text(encoding="utf-8"))
    content["nodes"][0]["rationale"] = "corrupted"
    path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        cache.load(context)


def test_failed_prediction_is_empty_and_retains_recall_denominator():
    gold = [{"case_id": "a", "gold_labels": ["D013"]}, {"case_id": "b", "gold_labels": ["D011"]}]
    predictions = [{"case_id": "a", "model_failure": True,
                    "candidates": [{"disorder": "D013", "confidence": .9}]},
                   {"case_id": "b", "candidates": [{"disorder": "D011", "confidence": .8}]}]
    score = evaluate(gold, predictions)
    assert score["candidate_recall_micro"] == .5 and score["candidate_ppv_micro"] == 1
    assert score["empty_candidate_rate"] == score["model_failure_rate"] == .5
    predictions[1]["model_failure"] = True
    score = evaluate(gold, predictions)
    assert score["candidate_recall_micro"] == 0 and score["candidate_ppv_micro"] is None
    assert score["model_failure_rate"] == 1
