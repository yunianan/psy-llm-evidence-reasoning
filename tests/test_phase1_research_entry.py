import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools import phase1_research_entry as entry
from tools.paired_evaluation import read_rows
from tools.a0_no_knowledge import known_patient_refs
from tests.test_a0_no_knowledge import payload, profile


def inputs():
    return [{"case_id": "c1", "profile": profile()}, {"case_id": "c2", "profile": profile()}]


def responses():
    return [{"case_id": "c1", "responses": [json.dumps(payload())]},
            {"case_id": "c2", "responses": ['{"candidates":[]}']}]


def test_offline_runner_manifest_logs_and_resume_never_call_model(tmp_path, monkeypatch):
    from shared import llm_client
    forbidden = Mock(side_effect=AssertionError("No network permitted"))
    monkeypatch.setattr(llm_client, "_bounded_client", forbidden)
    predictions, summary = entry.research_run("A0", inputs(), tmp_path, response_rows=responses())
    assert summary["new_model_calls"] == 0 and summary["complete"] == 2
    assert predictions[0]["report"]["considered_disorders"]
    assert predictions[0]["report"]["final_diagnoses"] == []
    manifest = json.loads((tmp_path / "manifest.A0.json").read_text(encoding="utf-8"))
    assert manifest["identity"]["knowledge_calls"] == 0
    assert manifest["identity"]["knowledge_hash"] is None
    assert manifest["identity"]["response_model_before_run"] is None
    before = (tmp_path / "runs/A0/c1.json").read_bytes()
    monkeypatch.setattr(entry.A0NoKnowledge, "generate_candidates", forbidden)
    again, _ = entry.research_run("A0", inputs(), tmp_path, response_rows=responses(), resume=True)
    assert again == predictions
    assert (tmp_path / "runs/A0/c1.json").read_bytes() == before
    forbidden.assert_not_called()
    events = read_rows(tmp_path / "runs/A0/events.jsonl")
    assert all("stage" in e and "node_ids" in e for e in events)
    assert any(e["event"] == "references_normalized" for e in events)
    assert all("usage_missing" in e for e in events if e["event"] == "case_finished")


def test_interrupted_run_preserves_completed_case_and_resumes_remaining(tmp_path):
    calls = []
    def execute(row):
        calls.append(row["case_id"])
        if len(calls) == 2: raise KeyboardInterrupt()
        return entry.export_prediction(row["case_id"], "A0", {"candidates": [], "final_diagnoses": []})
    with pytest.raises(KeyboardInterrupt):
        entry.research_run("A0", inputs(), tmp_path, executor=execute)
    before = (tmp_path / "runs/A0/c1.json").read_bytes()
    result, _ = entry.research_run("A0", inputs(), tmp_path, executor=execute, resume=True)
    assert calls == ["c1", "c2", "c2"]
    assert len(result) == 2
    assert (tmp_path / "runs/A0/c1.json").read_bytes() == before


def test_failed_case_is_not_retried_and_administrative_stop_is_not_model_failure(tmp_path):
    calls = []
    def fail(row):
        calls.append(row["case_id"])
        raise ValueError("invalid contract fixture")
    result, summary = entry.research_run("A0", inputs(), tmp_path, executor=fail)
    assert calls == ["c1"]
    assert result[0]["failure_class"] == "contract" and result[0]["model_failure"]
    assert result[1]["failure_class"] == "administrative" and not result[1]["model_failure"]
    assert summary["execution_stopped"]
    result2, _ = entry.research_run("A0", inputs(), tmp_path, executor=fail, resume=True)
    assert result2 == result and calls == ["c1"]


def test_changed_profile_or_source_cannot_resume(tmp_path):
    entry.research_run("A0", inputs(), tmp_path, response_rows=responses())
    changed = inputs()
    changed[0]["profile"]["sleep_difficulty"] = False
    with pytest.raises(ValueError, match="identical"):
        entry.research_run("A0", changed, tmp_path, response_rows=responses(), resume=True)


def test_budget_failure_exports_empty_but_model_failure_false():
    result = entry.export_prediction("a", "A1", {"candidates": [{"disorder": "D001"}]}, failure_class="budget")
    assert result["report"]["candidates"] == result["report"]["final_diagnoses"] == []
    assert result["run_failure"] and not result["model_failure"]


def test_corrupt_replay_source_is_integrity_failure_and_not_scoreable(tmp_path):
    from tools.prediction_contract import normalize_for_protocol
    source = [{"case_id": "c1", "report": {}}]
    result, summary = entry.research_run("A1", inputs()[:1], tmp_path, replay_rows=source)
    assert summary["protocol_integrity_failures"] == 1
    assert result[0]["failure_class"] == "administrative" and not result[0]["model_failure"]
    with pytest.raises(ValueError, match="integrity"):
        normalize_for_protocol(result[0])


def _runtime():
    return SimpleNamespace(local=SimpleNamespace(case_id="c", tags={}), max_attempts=3,
                           ask=Mock(return_value="fresh"))


def _prior(path, count, *, finished=False):
    key = entry.content_hash({"case_id": "c", "prompt": "p", "kwargs": {"model": "f"}})
    rows = [{"attempt_id": str(i), "logical_request_key": key,
             "event": "attempt_finished" if finished else "attempt_started",
             "raw_output": "stored", "retryable": True} for i in range(count)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_durable_request_reuses_successful_http_body_without_regeneration(tmp_path):
    path, runtime = tmp_path / "ledger.jsonl", _runtime()
    _prior(path, 1, finished=True)
    assert entry.DurableRequests(runtime, path)("p", model="f") == "stored"
    runtime.ask.assert_not_called()


def test_durable_request_counts_started_attempts_before_resume(tmp_path):
    path, runtime = tmp_path / "ledger.jsonl", _runtime()
    _prior(path, 2)
    observed = []
    runtime.ask.side_effect = lambda *a, **k: observed.append(runtime.max_attempts) or "fresh"
    assert entry.DurableRequests(runtime, path)("p", model="f") == "fresh"
    assert observed == [1] and runtime.max_attempts == 3
    _prior(path, 3)
    with pytest.raises(entry.PersistedRequestFailure):
        entry.DurableRequests(runtime, path)("p", model="f")
    assert runtime.ask.call_count == 1


def test_a1_actual_clinicr_can_replay_nodes_with_injected_verifier_without_api(tmp_path, monkeypatch):
    from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider
    from layers.layer3.diagnosis.g1_cache import cache_context, knowledge_hash, G1NodeCache, CachedG1Reasoner
    from layers.layer3.diagnosis.clinicr import CLINICR
    from layers.layer3.llm import client
    from tests.tree_test_support import controlled_generation
    provider = JsonDiagnosticKnowledgeProvider()
    row = {"case_id": "c", "profile": {"key_narratives": [{"text": "fixture", "sources": ["ev"]}]}}
    model_profile = {**row["profile"], client.ALLOWED_EVIDENCE_REFS_KEY: known_patient_refs(row["profile"])}
    context = cache_context("c", model_profile, model="fixture", service_url="https://fixture.invalid",
                            knowledge_digest=knowledge_hash(provider), prompt_version="fixture")
    generation = controlled_generation(["D013"], ref="ev", complete_labels=["D013"])
    cache = G1NodeCache(tmp_path)
    cache.save(context, generation.tree_trace)
    reasoner = CachedG1Reasoner(cache, context, knowledge=provider, mode="require")
    verifier = SimpleNamespace(verify=Mock(return_value={"score": .9, "decision": "confirmed", "routing": "pass",
        "checks": [], "fail_reasons": [], "missing_dimensions": []}))
    forbidden = Mock(side_effect=AssertionError("No model call permitted"))
    monkeypatch.setattr(client, "ask_llm_stream", forbidden)
    result = entry.run_arm("A1", row, model="fixture", knowledge=provider, reasoner=reasoner, verifier=verifier)
    assert result["candidate_generation_method"] == "dsm5_decision_tree"
    assert result["report"]["tree_trace"]
    assert reasoner.cache_hit
    forbidden.assert_not_called()
    normalized = entry._validate_replay(row, result, tmp_path)
    assert normalized["candidate_coverage"] == result["report"]["candidate_coverage"]


def test_cli_has_no_implicit_live_fallback(tmp_path):
    path = tmp_path / "dev.jsonl"
    path.write_text(json.dumps(inputs()[0]), encoding="utf-8")
    with pytest.raises(ValueError, match="Offline"):
        entry.main(["--arm", "A0", "--dev-list", str(path), "--out", str(tmp_path / "out")])


def test_cli_rejects_sealed_input_before_reading(tmp_path):
    with pytest.raises(SystemExit):
        entry.main(["--arm", "A1", "--dev-list", str(tmp_path / "sealed/eval.bundle.dpapi"), "--out", str(tmp_path / "out")])


@pytest.mark.parametrize("arm", ["A0", "A1"])
def test_execute_adapter_with_fake_sdk_records_usage_and_never_logs_key(tmp_path, monkeypatch, arm):
    from shared import llm_client
    from tests.tree_test_support import node_payload
    key = "synthetic-secret-never-log"
    monkeypatch.setattr(llm_client, "get_api_key", lambda: key)
    monkeypatch.setattr(llm_client, "get_base_url", lambda: "https://fixture.invalid/v1")
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    class FakeAPI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        def create(self, **request):
            prompt = request["messages"][0]["content"]
            value = payload() if arm == "A0" else node_payload(prompt, labels=["D013"], ref="p1")
            usage = {"prompt_tokens": 10, "completion_tokens": 10,
                     "prompt_tokens_details": {"cached_tokens": 4},
                     "completion_tokens_details": {"reasoning_tokens": 2}}
            chunk = SimpleNamespace(model="fixture", id="synthetic", choices=[SimpleNamespace(
                delta=SimpleNamespace(content=json.dumps(value), reasoning_content=None), finish_reason="stop")],
                usage=SimpleNamespace(model_dump=lambda **k: usage))
            return iter([chunk])
        def close(self): pass
    original = entry.ReliableStreaming
    monkeypatch.setattr(entry, "ReliableStreaming", lambda **kwargs: original(**kwargs, client_factory=FakeAPI, sleep=lambda _: None))
    result, summary = entry.research_run(arm, inputs()[:1], tmp_path, execute=True, model="fixture",
        budget_ref={"pricing_basis": "official_peak_all_input_miss_reference", "batch_limit_yuan": 5, "case_limit_yuan": 5})
    assert summary["complete"] == 1 and summary["new_model_calls"] >= 1
    assert summary["response_models"] == ["fixture"]
    assert summary["usage_missing_requests"] == 0
    assert result[0]["candidate_generation_method"] == ("no_knowledge_baseline" if arm == "A0" else "dsm5_decision_tree")
    for path in tmp_path.rglob("*.json*"):
        assert key not in path.read_text(encoding="utf-8")
