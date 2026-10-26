import io
import json
import threading
from types import SimpleNamespace as NS

import httpx
from openai import APIConnectionError, APIError, BadRequestError, APIStatusError
import pytest

from layers.layer3.diagnosis.schemas import TreeNodeJudgment
from layers.layer3.llm.client import _request_structured_output_with_one_retry
from tools.l3_engineering_runtime import (
    ReferenceBudget, CaseBudgetExceeded, BatchBudgetExceeded, ReliableStreaming,
    NonRetryableProviderError, NodeCheckpoint, attach_checkpoints, recover_ledger, reasoning_observation,
    error_category, retryable,
)


def chunk(text="", usage=None):
    return NS(model="fixture", id="completion-fixture", system_fingerprint=None,
              usage=NS(model_dump=lambda **kw: usage) if usage else None,
              choices=[NS(delta=NS(content=text, reasoning_content=None), finish_reason="stop" if usage else None)])


def fake_factory(responses, calls):
    class Stream:
        response = NS(headers={"x-request-id": "request-fixture"})
        def __init__(self, items):
            self.items = items
        def __iter__(self):
            for item in self.items:
                if isinstance(item, Exception):
                    raise item
                yield item
        def close(self):
            pass
    def factory(**options):
        calls.append({"client_options": options})
        def create(**request):
            calls[-1]["request"] = request
            result = responses.pop(0)
            if isinstance(result, Exception):
                raise result
            return Stream(result)
        return NS(chat=NS(completions=NS(create=create)), close=lambda: None)
    return factory


def transport(responses, *, max_attempts=3):
    calls, sleeps, journal = [], [], io.StringIO()
    runtime = ReliableStreaming(api_key="synthetic", base_url="https://fixture.invalid/v1", journal=journal,
        budget=ReferenceBudget(batch_limit=10, case_limit=5), max_attempts=max_attempts,
        client_factory=fake_factory(responses, calls), sleep=sleeps.append)
    records = []
    runtime.bind("case", records)
    runtime.tags("anchor", ["anchor.fixture"])
    return runtime, records, calls, sleeps, journal


def test_midstream_retry_discards_partial_text_and_logs_unknown_cost_and_backoff():
    error = APIConnectionError(request=httpx.Request("POST", "https://fixture.invalid"))
    runtime, records, calls, sleeps, journal = transport([
        [chunk("broken partial"), error],
        [chunk("valid"), chunk(usage={"prompt_tokens": 10, "completion_tokens": 2})]])
    assert runtime.ask("synthetic", model="fixture") == "valid"
    assert len(calls) == 2 and len(sleeps) == 1
    assert all(c["client_options"]["max_retries"] == 0 for c in calls)
    assert all(c["client_options"]["timeout"].read == 300 for c in calls)
    assert records[0]["usage"] is None and records[0]["raw_output"] == "broken partial"
    assert records[1]["response_request_id"] == "request-fixture"
    assert runtime.budget.unknown_upper > 0
    ends = [json.loads(x) for x in journal.getvalue().splitlines() if json.loads(x)["event"] == "attempt_finished"]
    assert ends[0]["backoff_seconds"] == sleeps[0]


def test_400_json_contract_error_is_not_retried_by_sdk_transport_or_structured_layer():
    response = httpx.Response(400, request=httpx.Request("POST", "https://fixture.invalid"))
    error = BadRequestError("invalid response_format json generation", response=response, body={})
    runtime, records, calls, sleeps, journal = transport([error])
    with pytest.raises(NonRetryableProviderError) as raised:
        _request_structured_output_with_one_retry(lambda p: runtime.ask(p, model="fixture"), json.loads, "p", "repair")
    assert raised.value.status_code == 400 and len(calls) == 1 and not sleeps


def test_transport_retries_have_three_attempt_total_limit():
    errors = [APIConnectionError(request=httpx.Request("POST", "https://fixture.invalid")) for _ in range(3)]
    runtime, records, calls, sleeps, journal = transport(errors)
    with pytest.raises(APIConnectionError):
        runtime.ask("synthetic", model="fixture")
    assert len(calls) == 3 and len(sleeps) == 2


def test_http_200_stream_internal_server_error_is_retried_without_inventing_http_500():
    error = APIError("synthetic provider error", request=httpx.Request("POST", "https://fixture.invalid"),
                     body={"code": "internal_server_error"})
    runtime, records, calls, sleeps, journal = transport([
        [chunk("discard partial"), error],
        [chunk("valid"), chunk(usage={"prompt_tokens": 10, "completion_tokens": 2})]])
    assert runtime.ask("synthetic", model="fixture") == "valid"
    assert len(calls) == 2 and len(sleeps) == 1
    assert records[0]["status_code"] == 200
    assert records[0]["provider_error_code"] == "internal_server_error"
    assert records[0]["error_category"] == "provider_internal_error"
    assert records[0]["failure_origin"] == "provider_stream"
    assert records[0]["retryable"] is True and runtime.budget.unknown_upper > 0


def test_stream_internal_server_error_keeps_three_total_attempt_limit():
    errors = [APIError("synthetic", request=httpx.Request("POST", "https://fixture.invalid"),
                       body={"code": "internal_server_error"}) for _ in range(3)]
    runtime, records, calls, sleeps, journal = transport([[error] for error in errors])
    with pytest.raises(APIError):
        runtime.ask("synthetic", model="fixture")
    assert len(calls) == 3 and len(sleeps) == 2


def test_unknown_stream_error_code_does_not_gain_a_retry():
    error = APIError("synthetic", request=httpx.Request("POST", "https://fixture.invalid"),
                     body={"code": "invalid_api_key"})
    runtime, records, calls, sleeps, journal = transport([[error]])
    with pytest.raises(APIError):
        runtime.ask("synthetic", model="fixture")
    assert len(calls) == 1 and not sleeps and records[0]["retryable"] is False


def test_explicit_400_cause_overrides_stream_internal_error_hint():
    response = httpx.Response(400, request=httpx.Request("POST", "https://fixture.invalid"))
    error = APIError("synthetic wrapper", request=response.request, body={"code": "internal_server_error"})
    error.__cause__ = BadRequestError("synthetic 400", response=response, body={})
    assert retryable(error) is False and error_category(error) == "http_client_error"


def test_stream_wrapper_with_timeout_cause_is_retried_and_logged():
    wrapper = RuntimeError("synthetic stream wrapper")
    wrapper.__cause__ = TimeoutError("synthetic socket timeout")
    runtime, records, calls, sleeps, journal = transport([
        [chunk("discard partial"), wrapper],
        [chunk("valid"), chunk(usage={"prompt_tokens": 10, "completion_tokens": 2})]])
    assert runtime.ask("synthetic", model="fixture") == "valid"
    assert len(calls) == 2 and len(sleeps) == 1
    assert records[0]["exception_types"] == ["builtins.RuntimeError", "builtins.TimeoutError"]
    assert records[0]["failure_origin"] == "client_transport"
    assert records[0]["error_category"] == "timeout"


@pytest.mark.parametrize("status,category,should_retry", [
    (400, "http_client_error", False), (429, "rate_limit", True), (503, "server_error", True),
])
def test_error_categories_and_http_retry_policy(status, category, should_retry):
    response = httpx.Response(status, request=httpx.Request("POST", "https://fixture.invalid"))
    error = APIStatusError("synthetic", response=response, body={})
    assert error_category(error) == category
    assert retryable(error) is should_retry


def test_connection_reset_cause_is_classified_and_retried():
    error = APIConnectionError(request=httpx.Request("POST", "https://fixture.invalid"))
    error.__cause__ = ConnectionResetError("synthetic reset")
    assert error_category(error) == "connection_reset" and retryable(error)


def test_connection_failure_keeps_category_with_generic_socket_cause():
    error = APIConnectionError(request=httpx.Request("POST", "https://fixture.invalid"))
    error.__cause__ = OSError("synthetic connection failure")
    assert error_category(error) == "connection_error" and retryable(error)


@pytest.mark.parametrize("wrapped", [False, True])
def test_nonretryable_http_status_takes_precedence_over_timeout_context(wrapped):
    response = httpx.Response(400, request=httpx.Request("POST", "https://fixture.invalid"))
    error = BadRequestError("synthetic bad request", response=response, body={})
    error.__context__ = TimeoutError("earlier synthetic timeout")
    if wrapped:
        outer = RuntimeError("synthetic provider wrapper")
        outer.__cause__ = error
        error = outer
    runtime, records, calls, sleeps, journal = transport([error])
    with pytest.raises((NonRetryableProviderError, RuntimeError)):
        runtime.ask("synthetic", model="fixture")
    assert len(calls) == 1 and not sleeps


def test_explicit_thinking_false_and_usage_are_sent_without_changing_default():
    runtime, records, calls, sleeps, journal = transport([[chunk("ok", {"prompt_tokens": 1, "completion_tokens": 1})]])
    runtime.bind("case", records, thinking=False)
    assert runtime.ask("synthetic", model="fixture", response_format={"type": "json_object"}) == "ok"
    assert calls[0]["request"]["extra_body"] == {"enable_thinking": False}
    assert calls[0]["request"]["stream_options"] == {"include_usage": True}


def test_budget_waits_for_other_inflight_reservation_instead_of_aborting_case():
    budget = ReferenceBudget(batch_limit=.2, case_limit=.2)
    first = budget.reserve("a", "fixture", 16000)
    entered, completed, amounts = threading.Event(), threading.Event(), []
    def second():
        entered.set()
        amounts.append(budget.reserve("b", "fixture", 16000))
        completed.set()
    worker = threading.Thread(target=second)
    worker.start()
    assert entered.wait(1) and not completed.wait(.05)
    budget.finish("a", first, {"prompt_tokens": 10, "completion_tokens": 2})
    assert completed.wait(1)
    worker.join()
    budget.finish("b", amounts[0], {"prompt_tokens": 10, "completion_tokens": 2})
    assert budget.reserved == pytest.approx(0) and budget.spent < .2


def test_case_budget_unknown_usage_does_not_stop_other_cases():
    budget = ReferenceBudget(batch_limit=1, case_limit=.15)
    first = budget.reserve("a", "fixture", 16000)
    budget.finish("a", first, None)
    with pytest.raises(CaseBudgetExceeded):
        budget.reserve("a", "fixture", 16000)
    another = budget.reserve("b", "fixture", 16000)
    budget.finish("b", another, {"prompt_tokens": 10, "completion_tokens": 2})
    assert budget.unknown_upper == first and budget.spent > first


def test_real_batch_limit_is_distinct_from_concurrent_reservation_pressure():
    budget = ReferenceBudget(batch_limit=.15, case_limit=1)
    amount = budget.reserve("a", "fixture", 16000)
    budget.finish("a", amount, None)
    with pytest.raises(BatchBudgetExceeded):
        budget.reserve("b", "fixture", 16000)


def test_resume_reuses_completed_batch_and_evaluates_only_missing_nodes(tmp_path):
    nodes = [{"node_id": n, "stage": "anchor"} for n in ["one", "two"]]
    calls = []
    def evaluate(profile, batch):
        calls.extend(n["node_id"] for n in batch)
        if batch[0]["node_id"] == "two":
            raise RuntimeError("transport failure")
        return [TreeNodeJudgment(node_id=n["node_id"], status="pending", rationale="synthetic") for n in batch]
    reasoner = NS(node_batches=lambda ns: [[n] for n in ns], _evaluate_tree_nodes=evaluate)
    fake_transport = NS(tags=lambda *args: None)
    checkpoint = NodeCheckpoint(tmp_path, {"model": "fixture", "thinking": None})
    attach_checkpoints(reasoner, checkpoint, fake_transport)
    with pytest.raises(RuntimeError):
        reasoner._evaluate_tree_nodes({}, nodes)
    assert checkpoint.get(nodes[0]) is not None and checkpoint.get(nodes[1]) is None
    resumed_calls = []
    def finish(profile, batch):
        resumed_calls.extend(n["node_id"] for n in batch)
        return [TreeNodeJudgment(node_id=n["node_id"], status="pending", rationale="synthetic") for n in batch]
    resumed = NS(node_batches=lambda ns: [[n] for n in ns], _evaluate_tree_nodes=finish)
    stats = attach_checkpoints(resumed, NodeCheckpoint(tmp_path, checkpoint.context), fake_transport)
    assert len(resumed._evaluate_tree_nodes({}, nodes)) == 2
    assert resumed_calls == ["two"] and stats == {"reused_nodes": 1, "new_nodes": 1}
    assert NodeCheckpoint(tmp_path, {"model": "fixture", "thinking": False}).get(nodes[0]) is None


def test_checkpoint_rejects_changed_node_and_corruption(tmp_path):
    checkpoint = NodeCheckpoint(tmp_path, {"model": "fixture"})
    node = {"node_id": "n", "stage": "anchor"}
    checkpoint.save([node], [TreeNodeJudgment(node_id="n", status="pending", rationale="fixture")], origin="fixture")
    with pytest.raises(ValueError, match="definition"):
        checkpoint.get({**node, "question": "changed"})
    data = json.loads(checkpoint.path.read_text(encoding="utf-8"))
    data["nodes"]["n"]["judgment"]["rationale"] = "corrupted"
    checkpoint.path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        NodeCheckpoint(tmp_path, checkpoint.context)


def test_ledger_resume_counts_unfinished_attempt_at_maximum_once(tmp_path):
    path = tmp_path / "requests.jsonl"
    rows = [{"attempt_id": "a", "event": "attempt_started", "case_id": "case", "reference_reservation_upper_yuan": .2},
            {"attempt_id": "a", "event": "attempt_finished", "case_id": "case", "reference_reservation_upper_yuan": .2,
             "usage": {"prompt_tokens": 100, "completion_tokens": 10}},
            {"attempt_id": "b", "event": "attempt_started", "case_id": "case", "reference_reservation_upper_yuan": .3}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    ledger = recover_ledger(path)
    assert ledger["spent"] == pytest.approx(.30028)
    assert ledger["unknown_upper"] == .3 and ledger["unfinished_http_attempts"] == 1


def test_null_reasoning_is_unknown_unless_text_accounting_fully_reconciles():
    entry = {"usage": {"completion_tokens": 80, "completion_tokens_details": {"reasoning_tokens": None, "text_tokens": 80}}, "reasoning_stream_seen": False}
    result = reasoning_observation(entry)
    assert result["raw_reasoning_tokens"] is None and result["effective_reasoning_tokens"] == 0
    entry["usage"]["completion_tokens_details"].pop("text_tokens")
    assert reasoning_observation(entry)["effective_reasoning_tokens"] is None
