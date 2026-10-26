import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from tools.a0_no_knowledge import A0NoKnowledge
from shared.disorder_catalog import SUPPORTED_DISORDERS, disorder_metadata
from layers.layer3.llm.observability import capture_events


class SentinelProvider:
    def __init__(self): self.calls = 0
    def get_candidate(self, *args):
        self.calls += 1
        raise AssertionError("Knowledge touched")
    def get_symptoms(self):
        self.calls += 1
        raise AssertionError("Symptoms touched")


def profile():
    return {"key_narratives": [{"text": "A software fixture narrative.", "sources": ["p1"]}],
            "evidence_nodes": [{"evidence_id": "ev1", "target_path": "sleep_difficulty", "quote": "fixture"}],
            "sleep_difficulty": True, "_allowed_evidence_refs": ["forged"]}


def payload(ref="p1"):
    return {"candidates": [{"disorder": "D001", "label": disorder_metadata("D001")["label"],
        "symptom_match": {"matched": 1, "total": 1}, "core_evidence_summary": "software fixture",
        "evidence_refs": [ref], "confidence": .4, "decision_paths": []}], "missing_dimensions": []}


def test_a0_whole_prompt_path_has_no_knowledge_and_full_coverage(monkeypatch):
    from layers.layer3.diagnosis.clinicr import CLINICR
    monkeypatch.setattr(CLINICR, "run", Mock(side_effect=AssertionError("A0 must not run CLINICR")))
    sentinel, prompts = SentinelProvider(), []
    def answer(text, **kwargs):
        prompts.append(text)
        return json.dumps(payload())
    result = A0NoKnowledge(answer, model="fixture", knowledge=sentinel).generate_candidates(profile())
    assert sentinel.calls == 0
    assert result.considered_disorders == list(SUPPORTED_DISORDERS)
    assert len(result.candidate_coverage) == 24
    assert result.candidate_coverage["D001"] == "candidate"
    assert set(result.candidate_coverage.values()) == {"candidate", "not_triggered"}
    assert result.tree_trace == [] and result.candidates[0].decision_paths == []
    assert "forged" not in prompts[0]
    for forbidden in ("min_count", "source_condition", "原始来源知识", "症状节点释义", "differential", "knowledge_source"):
        assert forbidden not in prompts[0]
    display_json = prompts[0].split("展示标签：", 1)[1].split("\n", 1)[0]
    assert all(set(row) == {"disorder", "label"} for row in json.loads(display_json))


def test_sentinel_is_effective_in_a1_engine_path(monkeypatch):
    from tools.phase1_research_entry import run_arm
    from layers.layer3.llm import client
    sentinel = SentinelProvider()
    def answer(prompt, **kwargs):
        nodes, _ = json.JSONDecoder().raw_decode(prompt.split("当前节点 JSON：", 1)[1])
        return json.dumps({"nodes": [{"node_id": n["node_id"], "status": "pending", "rationale": "fixture",
                                    "evidence_refs": [], "symptom_ids": [], "denial_scope": "unknown"} for n in nodes]})
    monkeypatch.setattr(client, "ask_llm_stream", answer)
    with pytest.raises(Exception) as caught:
        run_arm("A1", {"case_id": "a", "profile": profile()}, knowledge=sentinel)
    causes = []
    error = caught.value
    while error:
        causes.append(str(error))
        error = error.__cause__ or error.__context__
    assert sentinel.calls > 0 and any("Knowledge touched" in text for text in causes)


def test_alias_normalization_and_repair_events_are_observable():
    replies = iter([json.dumps(payload("invented")), json.dumps(payload("sleep_difficulty=True"))])
    events = []
    with capture_events(events.append):
        result = A0NoKnowledge(lambda *a, **k: next(replies), model="fixture").generate_candidates(profile())
    assert result.candidates[0].evidence_refs == ["ev1"]
    assert sum(e["event"] == "repair_triggered" for e in events) == 1
    assert any(e["event"] == "references_normalized" and e["changed"] for e in events)
    succeeded = [e for e in events if e["event"] == "structured_validation_succeeded"]
    assert succeeded[-1]["structured_attempt"] == 2
    assert events[-1]["event"] == "repair_result" and events[-1]["result"] == "ok"


@pytest.mark.parametrize("mutation", ["foreign_ref", "duplicate", "bool_confidence", "tree_path", "missing_evidence"])
def test_a0_contract_failure_after_one_repair(mutation):
    value = payload()
    if mutation == "foreign_ref": value["candidates"][0]["evidence_refs"] = ["foreign"]
    elif mutation == "duplicate": value["candidates"] *= 2
    elif mutation == "bool_confidence": value["candidates"][0]["confidence"] = True
    elif mutation == "tree_path": value["candidates"][0]["decision_paths"] = [{"path_id": "bad"}]
    elif mutation == "missing_evidence": value["candidates"][0]["evidence_refs"] = []
    answer = Mock(return_value=json.dumps(value))
    with pytest.raises(ValueError):
        A0NoKnowledge(answer, model="fixture").generate_candidates(profile())
    assert answer.call_count == 2


def test_empty_candidate_does_not_force_labels():
    result = A0NoKnowledge(lambda *a, **k: '{"candidates":[]}', model="fixture").generate_candidates({})
    assert result.candidates == [] and len(result.considered_disorders) == 24
    assert set(result.candidate_coverage.values()) == {"not_triggered"}


def test_fresh_a0_import_and_prompt_never_read_symptoms_or_criteria():
    # A fresh process is needed: other pipeline tests explicitly import symptoms.
    script = '''
from shared import knowledge_sources as sources
original = sources._read
def guarded(root, name):
    if name != 'disorder.json':
        raise AssertionError('A0 read diagnostic content: ' + name)
    return original(root, name)
sources._read = guarded
from tools.a0_no_knowledge import A0NoKnowledge
value = A0NoKnowledge(lambda *a, **k: '{"candidates":[]}', model='fixture').generate_candidates({})
assert len(value.considered_disorders) == 24
from tempfile import TemporaryDirectory
from tools.phase1_research_entry import research_run
with TemporaryDirectory() as directory:
    rows, summary = research_run('A0', [{'case_id':'fresh','profile':{}}], directory,
        response_rows=[{'case_id':'fresh','responses':['{"candidates":[]}']}])
    assert summary['new_model_calls'] == 0 and len(rows[0]['report']['considered_disorders']) == 24
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
