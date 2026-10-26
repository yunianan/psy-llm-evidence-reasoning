import io
import json
from types import SimpleNamespace as NS

import httpx
import httpx2
import pytest

from tools import l3_engineering_runtime as runtime
from tools import phase1_research_entry as entry
from tools import phase1_joint_run as joint
from tests.test_l3_engineering_runtime import transport, chunk, fake_factory


def test_httpx2_stream_disconnect_retries_discards_partial_and_keeps_cost():
    engine, records, calls, sleeps, _ = transport([
        [chunk("discard"), httpx2.RemoteProtocolError("fixture")],
        [chunk("ok", {"prompt_tokens": 10, "completion_tokens": 2})],
    ])
    assert engine.ask("fixture", model="fixture") == "ok"
    assert len(calls) == 2 and len(sleeps) == 1
    assert records[0]["retryable"] and records[0]["error_category"] == "connection_error"
    assert engine.budget.unknown_upper == records[0]["reference_reservation_upper_yuan"]


def test_httpx2_exhaustion_is_transport_and_stays_at_three_attempts():
    engine, records, calls, sleeps, _ = transport([[httpx2.RemoteProtocolError("fixture")] for _ in range(3)])
    with pytest.raises(httpx2.RemoteProtocolError) as exc:
        engine.ask("fixture", model="fixture")
    assert entry.classify_failure(exc.value) == "transport"
    assert len(calls) == 3 and len(sleeps) == 2


def test_httpx2_timeout_has_timeout_category():
    error = httpx2.ReadTimeout("fixture")
    assert runtime.retryable(error) and runtime.error_category(error) == "timeout"
    assert entry.classify_failure(error) == "transport"


@pytest.mark.parametrize("recovery", [False, True])
def test_historical_misclassification_only_reopens_with_engineering_recovery(tmp_path, recovery):
    engine, records, calls, _, _ = transport([[chunk("ok", {"prompt_tokens": 1, "completion_tokens": 1})]])
    engine.engineering_recovery = recovery
    prompt, kwargs = "fixture", {"model": "fixture"}
    previous = {"event": "attempt_finished", "attempt_id": "prior", "case_id": "case",
                "logical_request_key": entry.content_hash({"case_id": "case", "prompt": prompt, "kwargs": kwargs}),
                "status_code": 200, "error_type": "RemoteProtocolError", "retryable": False,
                "exception_types": ["httpx2.RemoteProtocolError"]}
    path = tmp_path / "requests.jsonl"
    path.write_text(json.dumps(previous) + "\n", encoding="utf-8")
    request = entry.DurableRequests(engine, path)
    if recovery:
        assert request(prompt, **kwargs) == "ok"
        assert records[-1]["attempt"] == 2 and records[-1]["prior_http_attempts"] == 1
    else:
        with pytest.raises(entry.PersistedRequestFailure):
            request(prompt, **kwargs)
        assert not calls


def test_historical_http_400_does_not_become_retryable():
    assert not runtime.legacy_stream_retryable({"status_code": 400, "exception_types": ["httpx2.RemoteProtocolError"]})


def test_recovery_reuses_successes_carries_budget_and_leaves_source_immutable(tmp_path, monkeypatch):
    from shared import llm_client
    import dotenv

    monkeypatch.setattr(llm_client, "get_api_key", lambda: "synthetic")
    monkeypatch.setattr(llm_client, "get_base_url", lambda: "https://fixture.invalid/v1")
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    calls = []
    responses = [[chunk("ok", {"prompt_tokens": 10, "completion_tokens": 10})] for _ in range(3)]
    responses.append([httpx2.RemoteProtocolError("synthetic stream disconnect")])
    factory = fake_factory(responses, calls)
    original_runtime = entry.ReliableStreaming
    monkeypatch.setattr(entry, "ReliableStreaming", lambda **kw: original_runtime(**kw, client_factory=factory, sleep=lambda _: None))
    def arm0(arm, row, *, request, model):
        request("A0:" + row["case_id"], model=model)
        return entry.export_prediction(row["case_id"], arm, {"candidates": []})
    def arm1(row, engine, out, request):
        engine.tags("anchor", ["anchor.fixture"])
        request("A1:" + row["case_id"], model=engine.local.model)
        return entry.export_prediction(row["case_id"], "A1", {"candidates": []}), {}
    monkeypatch.setattr(entry, "run_arm", arm0)
    monkeypatch.setattr(entry, "_live_a1", arm1)
    rows = [{"case_id": f"c{i:03}", "profile": {"key_narratives": ["fixture"]}} for i in range(2)]
    budget = {"pricing_basis": "official_peak_all_input_miss_reference", "batch_limit_yuan": 30, "case_limit_yuan": 2}
    source, destination = tmp_path / "source", tmp_path / "recovery"
    # Reproduce the actual old classification defect, without network access.
    with monkeypatch.context() as old:
        old.setattr(runtime, "TRANSPORT_ERRORS", (httpx.TransportError,))
        failed = joint.run_joint(rows, source, budget_ref=budget, model="fixture", execute=True)
    assert failed["complete"] == {"A0": 2, "A1": 1}
    before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    responses.append([chunk("ok", {"prompt_tokens": 10, "completion_tokens": 10})])
    recovered = joint.run_joint(rows, destination, budget_ref=budget, model="fixture", execute=True, recover_from=source)
    assert recovered["complete"] == {"A0": 2, "A1": 2} and not recovered["stop_reasons"]
    assert len(calls) == 5  # Four prior requests; only the failed batch was requested again.
    assert recovered["budget"]["reference_spent_upper_yuan"] == pytest.approx(failed["budget"]["reference_spent_upper_yuan"] + .0001)
    assert recovered["budget"]["unknown_usage_upper_yuan"] == failed["budget"]["unknown_usage_upper_yuan"]
    assert json.loads((destination / "summary.A1.json").read_text())["new_model_calls"] == 1
    assert json.loads((destination / "summary.A0.json").read_text())["new_model_calls"] == 0
    assert {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()} == before
    again = joint.run_joint(rows, destination, budget_ref=budget, model="fixture", execute=True, resume=True)
    assert len(calls) == 5 and again["budget"] == recovered["budget"]
