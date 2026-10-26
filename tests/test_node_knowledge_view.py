"""Screening context isolation, without replacing production validation."""
import copy
import json

import pytest

from layers.layer3.diagnosis.decision_tree import load_candidate_graph
from layers.layer3.llm import client
from layers.layer3.llm.node_knowledge import NodeKnowledgeView


class SentinelKnowledge:
    def get_candidate(self, label):
        raise AssertionError("Screening must not read candidate knowledge")

    def get_symptoms(self):
        raise AssertionError("Screening must not read the symptom catalog")


@pytest.mark.parametrize("context_options", [{}, {"screening_zero_knowledge": True}])
def test_all_screens_use_only_existing_nodes_and_patient_evidence(monkeypatch, context_options):
    graph = load_candidate_graph()
    node_ids = list(dict.fromkeys(node for path in graph["paths"].values() for node in path["screens"]))
    nodes = [{"node_id": node, **copy.deepcopy(graph["nodes"][node])} for node in node_ids]
    original_nodes = copy.deepcopy(nodes)
    calls = []

    def controlled_api(prompt, **kwargs):
        calls.append((prompt, kwargs))
        assert "原始来源知识：[]\n" in prompt
        assert client.SCREENING_GUIDANCE in prompt
        assert "不等于诊断" in prompt
        assert "只有疲惫、压力或工作困难" in prompt
        batch, _ = json.JSONDecoder().raw_decode(prompt.split("当前节点 JSON：", 1)[1])
        return json.dumps({"nodes": [
            {"node_id": node["node_id"], "status": "met" if i == 0 else "pending",
             "rationale": "software fixture", "evidence_refs": ["ev-screen"] if i == 0 else [],
             "symptom_ids": [], "confidence": 0.7}
            for i, node in enumerate(batch)
        ]})

    monkeypatch.setattr(client, "ask_llm_stream", controlled_api)
    answers = client.APICLINICRReasoner(
        knowledge=SentinelKnowledge(), **context_options,
    )._evaluate_tree_nodes({
        "key_narratives": [{"value": "screen fixture", "sources": ["ev-screen"]}],
        client.ALLOWED_EVIDENCE_REFS_KEY: ["ev-screen"],
    }, nodes)

    assert len(answers) == len(node_ids) == 9
    assert {item.node_id for item in answers} == set(node_ids)
    assert len(calls) == 2
    assert all(options["response_format"] == {"type": "json_object"} for _, options in calls)
    assert all(options["temperature"] == 0 for _, options in calls)
    assert nodes == original_nodes


def test_explicit_full_view_preserves_the_offline_comparator():
    calls = []

    class RecordingKnowledge:
        def get_candidate(self, label):
            calls.append(label)
            return {"source_ref": "fixture", "criteria": {"fixture": True}}

    sources = NodeKnowledgeView(RecordingKnowledge(), screening_zero_knowledge=False).sources_for_nodes([
        {"stage": "screening", "disorders": ["D013"]},
    ])
    assert calls == ["D013"]
    assert sources[0]["criteria"] == {"fixture": True}


@pytest.mark.parametrize("context_options", [{}, {"screening_zero_knowledge": True},
                                            {"screening_zero_knowledge": False}, {"screening_compact_view": True},
                                            {"anchor_compact_view": True}])
def test_context_selection_does_not_enable_unconnected_runtime_model_requests(monkeypatch, context_options):
    monkeypatch.setattr(client, "ask_llm_stream", lambda *_a, **_k: pytest.fail("No connected provider"))
    reasoner = client.APICLINICRReasoner(**context_options)
    with pytest.raises(client.CLINICRModelUnavailable, match="诊断知识尚未接入"):
        reasoner.generate_candidates({})


def test_later_stages_keep_full_knowledge_and_context_disorders():
    calls = []
    original = {
        "criteria": {"required_criteria": {"duration": 14}},
        "symptoms": {"S-test": {"description": "untrimmed fixture"}},
        "differential": {"fixture": True}, "events": {"event": "untrimmed fixture"},
    }

    class RecordingKnowledge:
        def get_candidate(self, label):
            calls.append(label)
            return original

    nodes = [
        {"stage": stage, "disorders": ["D013"], "context_disorders": ["D011"]}
        for stage in ["anchor", "exclusion", "criterion"]
    ]
    view = NodeKnowledgeView(RecordingKnowledge()).sources_for_nodes(nodes)
    assert calls == ["D013", "D011"]
    assert view == [original, original]
    assert all(item is original for item in view)


def test_compact_screening_uses_only_needed_original_symptom_names():
    from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider
    from unittest.mock import Mock

    provider = JsonDiagnosticKnowledgeProvider()
    provider.get_candidate = Mock(side_effect=AssertionError("No full label packages"))
    provider.get_symptoms = Mock(wraps=provider.get_symptoms)
    graph = load_candidate_graph()
    nodes = [{"node_id": "screen.attention", **graph["nodes"]["screen.attention"]}]
    view = NodeKnowledgeView(provider, screening_compact_view=True)
    first = view.sources_for_nodes(nodes)[0]
    assert first["symptom_catalog"]
    assert set(first) == {"view", "entry_rules", "symptom_catalog_source", "symptom_catalog"}
    assert all(len(item) == 2 for item in first["symptom_catalog"])
    assert view.sources_for_nodes(nodes)[0] == first
    assert provider.get_symptoms.call_count == 1
    provider.get_candidate.assert_not_called()


def test_screening_context_modes_are_exclusive():
    with pytest.raises(ValueError, match="one screening"):
        NodeKnowledgeView(SentinelKnowledge(), screening_zero_knowledge=True, screening_compact_view=True)


def test_explicit_compact_api_option_overrides_zero_default(monkeypatch):
    from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider
    from unittest.mock import Mock

    provider = JsonDiagnosticKnowledgeProvider()
    provider.get_candidate = Mock(side_effect=AssertionError("No full packages in compact"))
    provider.get_symptoms = Mock(wraps=provider.get_symptoms)
    node_id = "screen.attention"
    node = {"node_id": node_id, **load_candidate_graph()["nodes"][node_id]}
    def respond(prompt, **kwargs):
        assert "screening-entry-compact-v1" in prompt
        return json.dumps({"nodes": [{"node_id": node_id, "status": "pending", "rationale": "fixture"}]})
    monkeypatch.setattr(client, "ask_llm_stream", respond)
    answers = client.APICLINICRReasoner(knowledge=provider, screening_compact_view=True)._evaluate_tree_nodes({}, [node])
    assert answers[0].status == "pending"
    provider.get_candidate.assert_not_called()
    assert provider.get_symptoms.call_count == 1


def test_explicit_full_api_option_overrides_zero_default(monkeypatch):
    calls = []
    class FullKnowledge:
        def get_candidate(self, label):
            calls.append(label)
            return {"fixture_full": True}
    node_id = "screen.mood"
    node = {"node_id": node_id, **load_candidate_graph()["nodes"][node_id]}
    def respond(prompt, **kwargs):
        assert '"fixture_full": true' in prompt
        return json.dumps({"nodes": [{"node_id": node_id, "status": "pending", "rationale": "fixture"}]})
    monkeypatch.setattr(client, "ask_llm_stream", respond)
    answers = client.APICLINICRReasoner(knowledge=FullKnowledge(), screening_zero_knowledge=False)._evaluate_tree_nodes({}, [node])
    assert answers[0].status == "pending"
    assert calls == node["disorders"]
