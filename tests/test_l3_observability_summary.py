import copy
from fractions import Fraction
import json
from pathlib import Path

import pytest

from layers.layer3.llm.observability import capture_events, EventJournal, EVENT_VERSION, ENVELOPE_VERSION
from tools.l3_observability_summary import summarize, summarize_run, ObservabilityIntegrityError


@pytest.fixture
def synthetic():
    """Seven independently specified cases; no model or scorer is involved."""
    data = {"arm": "A1", "predictions": [], "events": [], "requests": [],
            "manifest": {"identity": {"arm": "A1", "case_ids": [f"c{i}" for i in range(1, 8)],
                                      "input_hash": "synthetic-list", "event_version": EVENT_VERSION,
                                      "envelope_version": ENVELOPE_VERSION}}}
    context = {}
    def event(name, **fields):
        value = {"version": EVENT_VERSION, "event_version": EVENT_VERSION, "envelope_version": ENVELOPE_VERSION,
                 "seq": len(data["events"]) + 1,
                 "ts": "2026-10-25T20:00:00+08:00", "run_id": "synthetic", "arm": "A1",
                 "scope": "node", "source": "live", "request_id": None, "attempt_id": None,
                 **context, **fields, "event": name}
        data["events"].append(value)
    def attempt(case, stage, serial, elapsed, *, missing=False, cache_missing=False, error=False, reserve=.01):
        context.clear()
        context.update(case_id=case, stage=stage, node_ids=[stage + ".fixture"], attempt_id=case + "-" + str(serial))
        row = {"arm": "A1", **context, "batch_number": 1 if case == "c1" else serial,
               "logical_request_key": case + ("retry" if case == "c1" else str(serial)),
               "reference_reservation_upper_yuan": reserve, "usage": None, "event": "attempt_started"}
        data["requests"].append(copy.deepcopy(row))
        event("attempt_started")
        usage = None if missing else {"prompt_tokens": 10, "completion_tokens": 4,
                 "completion_tokens_details": {"reasoning_tokens": 1}}
        if usage is not None and not cache_missing:
            usage.update(prompt_cache_hit_tokens=3, prompt_cache_miss_tokens=7)
        row.update(event="attempt_finished", elapsed_seconds=elapsed, usage=usage)
        if error:
            row.update(error_type="ReadTimeout", error_category="timeout")
        data["requests"].append(copy.deepcopy(row))
        event("attempt_finished")
    def raw():
        event("raw_response", body_characters=2, body_sha256="a" * 64, finish_reason="stop")
    def finish(case, failure=None):
        if failure:
            event("terminal_failure", failure_class=failure,
                  transport_category="timeout" if failure == "transport" else None)
        event("case_finished", scope="case", failure_class=failure)
        data["predictions"].append({"case_id": case, "arm": "A1", "run_failure": bool(failure), "failure_class": failure})

    attempt("c1", "criteria", 1, 300, missing=True, error=True)
    attempt("c1", "criteria", 2, 20)
    raw()
    finish("c1")
    attempt("c2", "guard", 1, 300, missing=True, error=True, reserve=.02)
    finish("c2", "transport")
    attempt("c3", "anchor", 1, 2)
    raw()
    event("validation_failed", illegal_refs=["foreign"], fields=["evidence_refs"])
    event("repair_triggered", repair_id="repair3")
    attempt("c3", "anchor", 2, 3)
    raw()
    event("refs_canonicalized", changed=True, rewrites=[{"from": "alias", "to": ["ev"]}])
    event("repair_result", repair_id="repair3", attempt_id="c3-1", repair_attempt_id="c3-2", result="ok")
    finish("c3")
    attempt("c4", "criteria", 1, 4)
    raw()
    event("parse_failed", error_type="JSONDecodeError", truncated=True)
    event("repair_triggered", repair_id="repair4")
    attempt("c4", "criteria", 2, 5)
    raw()
    event("parse_failed", error_type="JSONDecodeError", truncated=True)
    event("repair_result", repair_id="repair4", attempt_id="c4-1", repair_attempt_id="c4-2", result="failed")
    finish("c4", "contract")
    attempt("c5", "screening", 1, 6, cache_missing=True)
    raw()
    event("screen_denial_scope_downgrade", node_id="screening.fixture")
    finish("c5")
    attempt("c6", "criteria", 1, 7, missing=True, reserve=.03)
    raw()
    finish("c6")
    context.clear()
    context.update(case_id="c7", stage="administrative", node_ids=[], scope="run", source="offline")
    finish("c7", "administrative")
    return data


def quantiles(values):
    """Independent exact oracle: linear percentile positions on declared counts."""
    if any(v is None for v in values):
        return {"p50": None, "p95": None, "observations": len(values), "missing": values.count(None)}
    values = sorted(values)
    def at(q):
        index = Fraction(len(values) - 1) * q
        lo = index.numerator // index.denominator
        hi = min(lo + 1, len(values) - 1)
        return round(float(Fraction(values[lo]) + (index - lo) * (values[hi] - values[lo])), 6)
    return {"p50": at(Fraction(1, 2)), "p95": at(Fraction(95, 100)), "observations": len(values), "missing": 0}


def expected_stat(*, n=1, complete=1, terminal=None, http=(), case_latency=None, units=0,
                  missing=0, cache_missing=0, no_cache_units=0, cost=0, recovered=0, timeout=0,
                  repairs=(0, 0, 0), illegal=0, canonical=0, denial=0):
    # All numeric counts below are declared independently of synthetic()'s records.
    names = ("prompt", "completion", "reasoning", "cache_hit", "cache_miss", "total")
    known = dict(zip(names, (units * 10, units * 4, units, (units - no_cache_units) * 3,
                             (units - no_cache_units) * 7, units * 14)))
    tokens = dict(known)
    if missing:
        tokens = dict.fromkeys(names)
    elif cache_missing:
        tokens.update(cache_hit=None, cache_miss=None)
    return {"cases": n, "complete": complete,
        "terminal_by_class": {k: (terminal or {}).get(k, 0) for k in ("transport", "contract", "budget", "administrative")},
        "http_attempts": len(http), "retries_recovered": recovered, "timeouts_terminal": timeout,
        "repairs": dict(zip(("triggered", "succeeded", "failed"), repairs)),
        "illegal_refs": illegal, "refs_canonicalized": canonical, "refs_canonicalization_batches": int(bool(canonical)),
        "denial_scope_downgrades": denial, "tokens": tokens, "known_tokens": known, "usage_missing": missing,
        "cache_usage_missing": cache_missing, "reasoning_usage_missing": missing,
        "http_elapsed_seconds_sum": sum(http), "http_attempt_latency_seconds": quantiles(list(http)) if http else {"p50": None, "p95": None, "observations": 0, "missing": 0},
        "latency_seconds": {**quantiles(case_latency if case_latency is not None else [sum(http)]), "basis": "per_case_sum_http_elapsed"},
        "cost_reference": {"upper_bound_yuan": cost, "known_upper_bound_yuan": cost,
                           "basis": "official_peak_all_input_miss_reference", "actual_bill": None}}


def test_synthetic_full_json_equals_independent_expected(synthetic):
    cases = {
        "c1": expected_stat(http=(300, 20), units=1, missing=1, cache_missing=1, cost=.010052, recovered=1),
        "c2": expected_stat(complete=0, terminal={"transport": 1}, http=(300,), missing=1, cache_missing=1, cost=.02, timeout=1),
        "c3": expected_stat(http=(2, 3), units=2, cost=.000104, repairs=(1, 1, 0), illegal=1, canonical=1),
        "c4": expected_stat(complete=0, terminal={"contract": 1}, http=(4, 5), units=2, cost=.000104, repairs=(1, 0, 1)),
        "c5": expected_stat(http=(6,), units=1, cache_missing=1, no_cache_units=1, cost=.000052, denial=1),
        "c6": expected_stat(http=(7,), missing=1, cache_missing=1, cost=.03),
        "c7": expected_stat(complete=0, terminal={"administrative": 1})}
    overall = expected_stat(n=7, complete=4, terminal={"transport": 1, "contract": 1, "administrative": 1},
        http=(300, 20, 300, 2, 3, 4, 5, 6, 7), case_latency=[320, 300, 5, 9, 6, 7, 0], units=6, missing=3,
        cache_missing=4, no_cache_units=1, cost=.060312, recovered=1, timeout=1, repairs=(2, 1, 1), illegal=1, canonical=1, denial=1)
    overall["per_case_distributions"] = {"tokens": {key: quantiles([v["tokens"][key] for v in cases.values()]) for key in cases["c1"]["tokens"]},
        "http_elapsed_seconds_sum": quantiles([320, 300, 5, 9, 6, 7, 0]),
        "latency_basis": "sum of recorded HTTP elapsed per case, excludes unrecorded wall time and backoff"}
    stages = {"administrative": cases["c7"], "anchor": cases["c3"], "guard": cases["c2"], "screening": cases["c5"],
        "criteria": expected_stat(n=3, complete=2, terminal={"contract": 1}, http=(300, 20, 4, 5, 7),
            case_latency=[320, 9, 7], units=3, missing=2, cache_missing=2, cost=.040156, recovered=1, repairs=(1, 0, 1))}
    keys = ("version", "input_hash", "case_ids", "arm", "code_commit", "code_versions", "tree_blob", "knowledge_hash", "prompt_version", "event_version", "envelope_version", "mode", "requested_model")
    identity = {key: synthetic["manifest"]["identity"].get(key) for key in keys}
    expected = {"version": "l3-observability-summary-v1", "event_version": EVENT_VERSION, "envelope_version": ENVELOPE_VERSION,
        "mode": "strict", "strict_acceptance": True, "identity": {"A1": identity}, "per_arm": {"A1": overall},
        "per_stage": {"A1": stages}, "per_case": {"A1": cases},
        "balance": {"A1": {"prediction_rows": 7, "case_finished": 7, "failed_prediction_rows": 3,
                           "terminal_failure": 3, "unmatched": {"case_finished": [], "terminal_failure": []}}},
        "field_coverage": {"A1": {"missing_envelope_fields": {}, "historical_fields_fabricated": False, "unsupported_legacy_counters": []}},
        "formal_scoring": False}
    assert summarize([synthetic]) == expected


@pytest.mark.parametrize("mutation", ["missing_finish", "duplicate_finish", "missing_terminal", "orphan_repair",
    "missing_repair_result", "empty_nodes", "bad_sequence", "missing_stage", "unknown_attempt", "wrong_case_attempt", "duplicate_http"])
def test_invariants_raise_without_repairing_input(synthetic, mutation):
    if mutation in {"missing_finish", "missing_terminal", "missing_repair_result"}:
        name = {"missing_finish": "case_finished", "missing_terminal": "terminal_failure", "missing_repair_result": "repair_result"}[mutation]
        synthetic["events"].pop(next(i for i, e in enumerate(synthetic["events"]) if e["event"] == name))
    elif mutation == "duplicate_finish": synthetic["events"].append(copy.deepcopy(synthetic["events"][-1]))
    elif mutation == "orphan_repair": next(e for e in synthetic["events"] if e["event"] == "repair_result")["repair_id"] = "orphan"
    elif mutation == "empty_nodes": synthetic["events"][0]["node_ids"] = []
    elif mutation == "bad_sequence": synthetic["events"][0]["seq"] = 99
    elif mutation == "missing_stage": synthetic["events"][0].pop("stage")
    elif mutation == "unknown_attempt": synthetic["events"][0]["attempt_id"] = "foreign"
    elif mutation == "wrong_case_attempt": synthetic["events"][0]["attempt_id"] = "c2-1"
    elif mutation == "duplicate_http": synthetic["requests"].append(copy.deepcopy(synthetic["requests"][0]))
    before = copy.deepcopy(synthetic)
    with pytest.raises(ObservabilityIntegrityError): summarize([synthetic])
    assert before == synthetic


def test_journal_sequence_resume_and_null_request_ids(tmp_path):
    path = tmp_path / "events.jsonl"
    journal = EventJournal(path, run_id="test", arm="A0")
    journal({"event": "case_finished", "case_id": "c", "stage": "a0"})
    before = path.read_bytes()
    EventJournal(path, run_id="test", arm="A0")({"event": "batch_stopped", "case_id": "c", "reason": "budget"})
    assert path.read_bytes().startswith(before)
    rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]
    assert [r["seq"] for r in rows] == [1, 2]
    assert all(r["request_id"] is None and r["attempt_id"] is None for r in rows)


def test_shared_helper_metadata_parse_and_repair_chain():
    from layers.layer3.llm.client import _request_structured_output_with_one_retry, _parse_json_object
    values = iter(["{", "{}"])
    events = []
    with capture_events(events.append, stage="a0", attempt_id=None):
        result = _request_structured_output_with_one_retry(lambda _: next(values), _parse_json_object, "p", "r")
    assert result == {}
    assert any(e["event"] == "parse_failed" and e["truncated"] is None for e in events)
    assert all("raw_output" not in e for e in events if e["event"] == "raw_response")
    trigger = next(e for e in events if e["event"] == "repair_triggered")
    repaired = next(e for e in events if e["event"] == "repair_result")
    assert trigger["repair_id"] == repaired["repair_id"] and repaired["result"] == "ok"


def test_runner_exports_administrative_terminals_and_summary_without_calls(tmp_path):
    from tools.phase1_research_entry import research_run
    rows = [{"case_id": "c1", "profile": {}}, {"case_id": "c2", "profile": {}}]
    def fail(row): raise ValueError("synthetic contract failure")
    predictions, _ = research_run("A0", rows, tmp_path, executor=fail)
    result = summarize_run(tmp_path)
    assert result["per_arm"]["A0"]["terminal_by_class"] == {"transport": 0, "contract": 1, "budget": 0, "administrative": 1}
    assert result["balance"]["A0"]["terminal_failure"] == 2
    assert (tmp_path / "observability/A0.summary.md").exists()
    events = [json.loads(s) for s in (tmp_path / "runs/A0/events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(e["event"] == "batch_stopped" and e["reason"] == "completion_impossible" for e in events)
    assert result["per_arm"]["A0"]["http_attempts"] == 0


def test_legacy_audit_lists_missing_fields_and_never_claims_strict(synthetic):
    for event in synthetic["events"]:
        event.pop("seq")
        event.pop("request_id")
    result = summarize([synthetic], legacy_audit=True)
    assert not result["strict_acceptance"]
    assert result["field_coverage"]["A1"]["missing_envelope_fields"]["seq"] == len(synthetic["events"])
    assert result["per_arm"]["A1"]["illegal_refs"] is None
    with pytest.raises(ObservabilityIntegrityError): summarize([synthetic])


def test_strict_rejects_event_and_manifest_envelope_version_mismatch(synthetic):
    synthetic["events"][0]["envelope_version"] = "l3-event-envelope-v0"
    with pytest.raises(ObservabilityIntegrityError) as error:
        summarize([synthetic])
    assert any(issue["code"] == "event_envelope_version" for issue in error.value.issues)
    synthetic["events"][0]["envelope_version"] = ENVELOPE_VERSION
    synthetic["manifest"]["identity"]["envelope_version"] = "l3-event-envelope-v0"
    with pytest.raises(ObservabilityIntegrityError) as error:
        summarize([synthetic])
    assert any(issue["code"] == "manifest_envelope_version" for issue in error.value.issues)
    synthetic["manifest"]["identity"]["envelope_version"] = ENVELOPE_VERSION
    assert summarize([synthetic])["envelope_version"] == ENVELOPE_VERSION


def test_headline_versions_come_from_the_run_not_current_constants(synthetic):
    synthetic["manifest"]["identity"].pop("event_version")
    synthetic["manifest"]["identity"].pop("envelope_version")
    result = summarize([synthetic], legacy_audit=True)
    assert result["event_version"] is None and result["envelope_version"] is None
    synthetic["manifest"]["identity"].update(event_version=EVENT_VERSION, envelope_version="l3-event-envelope-v0")
    assert summarize([synthetic], legacy_audit=True)["envelope_version"] == "l3-event-envelope-v0"
