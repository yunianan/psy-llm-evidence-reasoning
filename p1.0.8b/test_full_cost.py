"""Cost accounting checks with synthetic responses, no model or patient calls."""
import io
import json
from types import SimpleNamespace as NS

import pytest
import run_full_cost as cost


def request(stage="anchor", usage=None):
    return {"stage": stage, "usage": usage, "node_ids": ["n"], "is_structured_retry": False,
            "elapsed_seconds": 1, "transport_attempts": 1}


def usage(n):
    return {"prompt_tokens": n, "completion_tokens": 2, "total_tokens": n + 2}


def test_missing_usage_stays_unknown_and_zero_call_stage_is_zero():
    assert cost.accounting([])["total_tokens"]["total"] == 0
    result = cost.accounting([request(usage=usage(10)), request()])
    assert result["total_tokens"] == {"total": None, "known_sum": 12, "missing_request_usage": 1}


@pytest.mark.parametrize("stage,expected", [("screening", "screening"), ("anchor", "anchor"),
                                            ("exclusion", "guard"), ("criterion", "criteria")])
def test_node_stage_is_derived_from_actual_prompt(stage, expected):
    prompt = "当前节点 JSON：" + json.dumps([{"node_id": "n", "stage": stage}]) + "\nother text"
    assert cost.classify(prompt) == (expected, ["n"])


def test_verifier_is_separate_and_unknown_prompt_rejected():
    assert cost.classify("你是 L3 的独立 Reward Model Verifier 临时API实现。") == ("verifier", [])
    with pytest.raises(ValueError):
        cost.classify("not an engine prompt")


def test_observer_captures_usage_repairs_and_transport_retries():
    owner = {"case_id": "synthetic", "requests": []}
    cost.base.LOCAL.record = owner
    journal = io.StringIO()
    def bounded(**kwargs):
        hooks = {"request": []}
        def create(**request):
            for _ in range(2):
                for hook in hooks["request"]:
                    hook(None)
            return iter([NS(model="model", system_fingerprint="fp", usage=NS(model_dump=lambda **kwargs: usage(10)),
                            choices=[NS(delta=NS(content="{}"), finish_reason="stop")])])
        return NS(_client=NS(event_hooks=hooks), chat=NS(completions=NS(create=create)))
    wrapped = cost.observer(bounded, journal)
    for _ in range(2):
        api = wrapped(timeout=60)
        list(api.chat.completions.create(messages=[{"content": "你是 L3 的独立 Reward Model Verifier"}], model="model", temperature=0))
    assert [r["is_structured_retry"] for r in owner["requests"]] == [False, True]
    assert all(r["transport_attempts"] == 2 and r["raw_output"] == "{}" for r in owner["requests"])
    assert cost.accounting(owner["requests"])["total_tokens"]["total"] == 24
    assert len(journal.getvalue().splitlines()) == 2


def test_failed_requests_are_in_attempted_totals_and_not_successful_distribution():
    event = {"node_id": "n", "original_status": "not_met", "effective_status": "pending", "denial_scope": "partial"}
    rows = [dict(case_id="ok", ok=True, requests=[request("screening", usage(10))], elapsed_seconds=2,
                 result={"verifier_routing": "return_to_l2", "candidates": []}, reasoner_screen_denial_scope_events=[event, event],
                 prompt_tokens=10, completion_tokens=2, total_tokens=12),
            dict(case_id="failed", ok=False, requests=[request("anchor", usage(30))], elapsed_seconds=3,
                 reasoner_screen_denial_scope_events=[], exception_chain=[], prompt_tokens=30, completion_tokens=2, total_tokens=32)]
    result = cost.summarize(rows, expected=2)
    assert result["measurement_status"] == "incomplete" and result["successful_cases"] == 1
    assert result["cost_accounting"]["total_tokens"]["total"] == 44
    assert result["total_tokens_per_successful_case"]["sum"] == 12
    assert result["total_tokens_per_attempted_case"]["p50"] == 22
    assert result["verifier_called_cases"] == 0
    assert result["stages"]["verifier"]["cost_accounting"]["total_tokens"]["total"] == 0
    assert result["screen_denial_scope"]["downgrade_log_events"] == 2
    assert result["screen_denial_scope"]["unique_downgraded_nodes"] == 1
