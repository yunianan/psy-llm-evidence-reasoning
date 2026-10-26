"""Recompute engineering ledgers. No inference, clinical scoring or log repair."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from fractions import Fraction
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layers.layer3.llm.observability import EVENT_VERSION, ENVELOPE_VERSION

VERSION = "l3-observability-summary-v1"
CLASSES = ("transport", "contract", "budget", "administrative")
TOKEN_FIELDS = ("prompt", "completion", "reasoning", "cache_hit", "cache_miss", "total")
ENVELOPE = ("version", "event_version", "envelope_version", "seq", "ts", "run_id", "arm", "case_id",
            "stage", "scope", "node_ids", "attempt_id", "request_id")


class ObservabilityIntegrityError(ValueError):
    def __init__(self, issues):
        self.issues = issues
        super().__init__(json.dumps(issues, ensure_ascii=False))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _integer(value):
    return type(value) is int and value >= 0


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _tokens(row):
    usage = row.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    values = {"prompt": usage.get("prompt_tokens"), "completion": usage.get("completion_tokens"),
              "reasoning": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
              "cache_hit": usage.get("prompt_cache_hit_tokens", details.get("cached_tokens")),
              "cache_miss": usage.get("prompt_cache_miss_tokens")}
    values = {key: value if _integer(value) else None for key, value in values.items()}
    # Reasoning is a completion subset. Miss tokens are never inferred from hit.
    values["total"] = values["prompt"] + values["completion"] if all(values[k] is not None for k in ("prompt", "completion")) else None
    return values


def _distribution(values):
    if not values or any(value is None for value in values):
        return {"p50": None, "p95": None, "observations": len(values), "missing": sum(v is None for v in values)}
    values = sorted(values)
    def percentile(q):
        index = (len(values) - 1) * q
        lo, hi = math.floor(index), math.ceil(index)
        return round(values[lo] + (values[hi] - values[lo]) * (index - lo), 6)
    return {"p50": percentile(.5), "p95": percentile(.95), "observations": len(values), "missing": 0}


def _requests(rows):
    grouped = defaultdict(dict)
    issues = []
    for row in rows:
        key, event = row.get("attempt_id"), row.get("event")
        if not key or event not in {"attempt_started", "attempt_finished"} or event in grouped[key]:
            issues.append({"code": "invalid_or_duplicate_http_attempt", "attempt_id": key})
            continue
        grouped[key][event] = row
    result = []
    for key, events in grouped.items():
        if "attempt_started" not in events:
            issues.append({"code": "http_finish_without_start", "attempt_id": key})
        start, finish = events.get("attempt_started"), events.get("attempt_finished")
        if start and finish and any(start.get(k) != finish.get(k) for k in ("case_id", "arm", "stage", "node_ids")):
            issues.append({"code": "http_identity_changed", "attempt_id": key})
        result.append(finish or start)
    if issues:
        raise ObservabilityIntegrityError(issues)
    return result


def _validate(data, *, legacy):
    arm, predictions, events, requests, manifest = (data[k] for k in ("arm", "predictions", "events", "requests", "manifest"))
    issues, missing = [], Counter()
    def issue(code, **detail):
        issues.append({"arm": arm, "code": code, **detail})
    ids = [r.get("case_id") for r in predictions]
    if any(not isinstance(k, str) or not k for k in ids) or len(ids) != len(set(ids)):
        issue("invalid_prediction_ids")
    if manifest.get("identity", {}).get("case_ids") != ids:
        issue("prediction_order_or_manifest_mismatch")
    identity = manifest.get("identity", {}) or {}
    if not legacy:
        for key, expected in (("event_version", EVENT_VERSION), ("envelope_version", ENVELOPE_VERSION)):
            if identity.get(key) != expected:
                issue("manifest_" + key, declared=identity.get(key))
    http = _requests(requests)
    http_by_id = {r["attempt_id"]: r for r in http}
    for r in http:
        if r.get("case_id") not in ids or r.get("arm") != arm or not r.get("stage"):
            issue("http_outside_registered_arm_or_case", attempt_id=r["attempt_id"])
    finished, failures, triggers, results = Counter(), Counter(), {}, {}
    finished_at, terminal_at, raw_at = {}, {}, {}
    mirrored = defaultdict(Counter)
    for index, event in enumerate(events, 1):
        name, case = event.get("event"), event.get("case_id")
        if case not in ids or event.get("arm") != arm or not event.get("stage"):
            issue("event_outside_registered_arm_case_or_stage", index=index)
        absent = [key for key in ENVELOPE if key not in event]
        if absent:
            missing.update(absent)
            if not legacy:
                issue("missing_envelope_fields", index=index, fields=absent)
        if not legacy:
            if event.get("version") != EVENT_VERSION or event.get("seq") != index:
                issue("event_version_or_sequence", index=index)
            if event.get("envelope_version") != ENVELOPE_VERSION:
                issue("event_envelope_version", index=index)
            try:
                if datetime.fromisoformat(event.get("ts", "")).utcoffset() is None:
                    raise ValueError()
            except (TypeError, ValueError):
                issue("invalid_timestamp", index=index)
            if event.get("request_id") is not None and (not isinstance(event["request_id"], str) or not event["request_id"]):
                issue("invalid_request_id", index=index)
            scope = event.get("scope")
            if scope not in {"node", "case", "run", "verifier"}:
                issue("invalid_scope", index=index)
            if scope == "node" and not event.get("node_ids"):
                issue("node_scope_without_nodes", index=index)
            if event.get("attempt_id") is not None and event["attempt_id"] not in http_by_id:
                issue("unknown_http_attempt", index=index)
            elif event.get("attempt_id") is not None:
                request = http_by_id[event["attempt_id"]]
                if any(event.get(k) != request.get(k) for k in ("case_id", "stage", "node_ids")):
                    issue("event_http_identity_mismatch", index=index)
                if event.get("request_id") is not None and event["request_id"] != (request.get("response_request_id") or request.get("completion_id")):
                    issue("event_server_request_id_mismatch", index=index)
            if case in finished_at and event.get("scope") != "run":
                issue("event_after_case_finished", index=index)
            if event.get("source") == "live" and name == "terminal_failure" and event.get("failure_class") in {"transport", "contract"}:
                if not event.get("attempt_id"):
                    issue("live_model_terminal_without_attempt", index=index)
            if name == "raw_response" and ("raw_output" in event or not isinstance(event.get("body_sha256"), str) or len(event["body_sha256"]) != 64):
                issue("invalid_raw_response_metadata", index=index)
            if name == "parse_failed" and (case, event.get("attempt_id")) not in raw_at:
                issue("parse_failure_without_raw_response", index=index)
        if name == "raw_response":
            raw_at[(case, event.get("attempt_id"))] = index
        if name in {"attempt_started", "attempt_finished"}:
            mirrored[event.get("attempt_id")][name] += 1
        if name == "case_finished":
            finished[case] += 1
            finished_at[case] = index
        if name == "terminal_failure":
            failures[case] += 1
            terminal_at[case] = index
        if name == "repair_triggered":
            repair_id = event.get("repair_id")
            if not repair_id:
                missing["repair_id"] += 1
                if not legacy:
                    issue("repair_trigger_without_id", index=index)
            elif repair_id in triggers:
                issue("duplicate_repair_trigger", index=index)
            else:
                triggers[repair_id] = (index, event)
        if name == "repair_result":
            repair_id = event.get("repair_id")
            trigger = triggers.get(repair_id)
            if not trigger or repair_id in results or any(trigger[1].get(k) != event.get(k) for k in ("case_id", "attempt_id")):
                issue("orphan_or_duplicate_repair_result", index=index)
            if event.get("result") not in {"ok", "failed"}:
                issue("invalid_repair_result", index=index)
            results[repair_id] = event
    unmatched = {"case_finished": sorted(k for k in set(ids) | set(finished) if finished[k] != 1),
                 "terminal_failure": sorted(k for k in set(ids) | set(failures)
                    if failures[k] != int(next((r for r in predictions if r["case_id"] == k), {}).get("run_failure", False)))}
    for name, values in unmatched.items():
        if values:
            issue("unmatched_" + name, cases=values)
    for row in predictions:
        if row.get("arm") != arm or row.get("run_failure") != bool(row.get("failure_class")):
            issue("prediction_failure_or_arm_mismatch", case_id=row["case_id"])
        if row.get("run_failure"):
            terminal = next((e for e in events if e.get("event") == "terminal_failure" and e.get("case_id") == row["case_id"]), {})
            if terminal.get("failure_class") != row.get("failure_class") or row.get("failure_class") not in CLASSES:
                issue("terminal_class_mismatch", case_id=row["case_id"])
    if not legacy:
        for repair_id in triggers.keys() - results.keys():
            issue("repair_without_result", repair_id=repair_id)
        for row in http:
            required = {"attempt_started": 1, "attempt_finished": int(row["event"] == "attempt_finished")}
            if dict(mirrored[row["attempt_id"]]) != {k: v for k, v in required.items() if v}:
                issue("request_event_mirror_mismatch", attempt_id=row["attempt_id"])
        if mirrored.keys() - http_by_id.keys():
            issue("event_attempt_not_in_ledger")
        for repair_id, event in results.items():
            if event.get("result") == "failed" and terminal_at.get(event.get("case_id"), 0) <= triggers.get(repair_id, (len(events),))[0]:
                issue("failed_repair_without_terminal", repair_id=repair_id)
            repaired = event.get("repair_attempt_id")
            if repaired is not None and (repaired not in http_by_id or http_by_id[repaired].get("case_id") != event.get("case_id")):
                issue("repair_http_identity_mismatch", repair_id=repair_id)
    if issues:
        raise ObservabilityIntegrityError(issues)
    return http, {"prediction_rows": len(ids), "case_finished": sum(finished.values()),
                  "failed_prediction_rows": sum(bool(r["run_failure"]) for r in predictions),
                  "terminal_failure": sum(failures.values()), "unmatched": unmatched}, dict(sorted(missing.items()))


def _aggregate(predictions, attempts, events, *, legacy):
    values = [_tokens(row) for row in attempts]
    tokens = {key: sum(v[key] for v in values) if all(v[key] is not None for v in values) else None for key in TOKEN_FIELDS}
    known = {key: sum(v[key] for v in values if v[key] is not None) for key in TOKEN_FIELDS}
    usage_missing = sum(v["total"] is None for v in values)
    cache_missing = sum(v["cache_hit"] is None or v["cache_miss"] is None for v in values)
    groups = defaultdict(list)
    for row in attempts:
        groups[(row["case_id"], row.get("logical_request_key") or (row.get("stage"), row.get("batch_number")))].append(row)
    recovered = sum(any(r.get("error_type") for r in group) and group[-1]["event"] == "attempt_finished" and not group[-1].get("error_type") for group in groups.values())
    latency, costs = [], []
    for row, token in zip(attempts, values):
        latency.append(row.get("elapsed_seconds") if _finite(row.get("elapsed_seconds")) else None)
        costs.append(float(Fraction(2 * token["prompt"] + 8 * token["completion"], 1_000_000)) if token["total"] is not None
                     else row.get("reference_reservation_upper_yuan") if _finite(row.get("reference_reservation_upper_yuan")) else None)
    terminal = {c: sum(r.get("failure_class") == c for r in predictions) for c in CLASSES}
    timed_out = {e["case_id"] for e in events if e.get("event") == "terminal_failure" and e.get("failure_class") == "transport"
                 and e.get("transport_category") == "timeout"}
    for row in predictions:
        case_requests = [a for a in attempts if a["case_id"] == row["case_id"]]
        if row.get("failure_class") == "transport" and case_requests and case_requests[-1].get("error_category") == "timeout":
            timed_out.add(row["case_id"])
    canonical = [e for e in events if e.get("event") == "refs_canonicalized"]
    case_latencies = []
    for prediction in predictions:
        elapsed = [r.get("elapsed_seconds") for r in attempts if r["case_id"] == prediction["case_id"]]
        case_latencies.append(sum(elapsed) if all(_finite(v) for v in elapsed) else None)
    return {"cases": len(predictions), "complete": sum(not r["run_failure"] for r in predictions),
            "terminal_by_class": terminal, "http_attempts": len(attempts), "retries_recovered": recovered,
            "timeouts_terminal": len(timed_out),
            "repairs": {"triggered": sum(e.get("event") == "repair_triggered" for e in events),
                        "succeeded": None if legacy else sum(e.get("event") == "repair_result" and e.get("result") == "ok" for e in events),
                        "failed": None if legacy else sum(e.get("event") == "repair_result" and e.get("result") == "failed" for e in events)},
            "illegal_refs": None if legacy else sum(len(e.get("illegal_refs", [])) for e in events if e.get("event") == "validation_failed"),
            "refs_canonicalized": None if legacy else sum(len(e.get("rewrites", [])) for e in canonical),
            "refs_canonicalization_batches": None if legacy else sum(bool(e.get("changed")) for e in canonical),
            "denial_scope_downgrades": None if legacy else sum(e.get("event") == "screen_denial_scope_downgrade" for e in events),
            "tokens": tokens, "known_tokens": known, "usage_missing": usage_missing,
            "cache_usage_missing": cache_missing, "reasoning_usage_missing": sum(v["reasoning"] is None for v in values),
            "http_elapsed_seconds_sum": None if any(v is None for v in latency) else round(sum(latency), 6),
            "http_attempt_latency_seconds": _distribution(latency),
            "latency_seconds": {**_distribution(case_latencies), "basis": "per_case_sum_http_elapsed"},
            "cost_reference": {"upper_bound_yuan": None if any(v is None for v in costs) else round(sum(costs), 9),
                               "known_upper_bound_yuan": round(sum(v for v in costs if v is not None), 9),
                               "basis": "official_peak_all_input_miss_reference", "actual_bill": None}}


def summarize(arms, *, legacy_audit=False):
    result = {"version": VERSION, "event_version": EVENT_VERSION, "envelope_version": ENVELOPE_VERSION,
              "mode": "legacy_field_audit" if legacy_audit else "strict",
              "strict_acceptance": not legacy_audit, "identity": {}, "per_arm": {}, "per_stage": {},
              "per_case": {}, "balance": {}, "field_coverage": {}, "formal_scoring": False}
    for data in arms:
        arm = data["arm"]
        if arm in result["per_arm"]:
            raise ObservabilityIntegrityError([{"code": "duplicate_arm", "arm": arm}])
        http, balance, missing = _validate(data, legacy=legacy_audit)
        predictions, events = data["predictions"], data["events"]
        identity = data["manifest"]["identity"]
        result["identity"][arm] = {key: identity.get(key) for key in ("version", "input_hash", "case_ids", "arm", "code_commit", "code_versions", "tree_blob", "knowledge_hash", "prompt_version", "event_version", "envelope_version", "mode", "requested_model")}
        result["balance"][arm] = balance
        result["field_coverage"][arm] = {"missing_envelope_fields": missing,
            "historical_fields_fabricated": False,
            "unsupported_legacy_counters": ["repairs.succeeded", "repairs.failed", "illegal_refs", "refs_canonicalized", "denial_scope_downgrades"] if legacy_audit else []}
        overall = _aggregate(predictions, http, events, legacy=legacy_audit)
        cases = {}
        for row in predictions:
            case = row["case_id"]
            cases[case] = _aggregate([row], [r for r in http if r["case_id"] == case],
                                     [e for e in events if e["case_id"] == case], legacy=legacy_audit)
        overall["per_case_distributions"] = {"tokens": {key: _distribution([c["tokens"][key] for c in cases.values()]) for key in TOKEN_FIELDS},
            "http_elapsed_seconds_sum": _distribution([c["http_elapsed_seconds_sum"] for c in cases.values()]),
            "latency_basis": "sum of recorded HTTP elapsed per case, excludes unrecorded wall time and backoff"}
        result["per_arm"][arm], result["per_case"][arm] = overall, cases
        stages = sorted({r["stage"] for r in http} | {e["stage"] for e in events})
        stage_summaries = {}
        for stage in stages:
            stage_events = [e for e in events if e["stage"] == stage]
            stage_http = [r for r in http if r["stage"] == stage]
            ids = {r["case_id"] for r in stage_http} | {e["case_id"] for e in stage_events}
            stage_predictions = []
            for row in predictions:
                if row["case_id"] in ids:
                    # Complete counts describe whole-case completion among cases seen at this stage;
                    # terminal classification belongs only to the actual failure stage.
                    row = dict(row)
                    if not any(e["case_id"] == row["case_id"] and e["event"] == "terminal_failure" for e in stage_events):
                        row["failure_class"] = None
                    stage_predictions.append(row)
            stage_summaries[stage] = _aggregate(stage_predictions, stage_http, stage_events, legacy=legacy_audit)
        if sum(v["http_attempts"] for v in stage_summaries.values()) != overall["http_attempts"]:
            raise ObservabilityIntegrityError([{"code": "stage_attempt_balance", "arm": arm}])
        result["per_stage"][arm] = stage_summaries
    if result["identity"]:
        for key, constant in (("event_version", EVENT_VERSION), ("envelope_version", ENVELOPE_VERSION)):
            declared = {values.get(key) for values in result["identity"].values()}
            if declared == {constant}:
                result[key] = constant
            elif len(declared) == 1 and None not in declared:
                result[key] = next(iter(declared))
            else:
                result[key] = None
    return result


def summarize_run(run, *, arms=None, legacy_audit=False):
    run = Path(run)
    names = arms or sorted(path.stem.split(".")[-1] for path in run.glob("manifest.A*.json"))
    if not names:
        raise ValueError("No arm manifests")
    data = []
    for arm in names:
        data.append({"arm": arm, "manifest": json.loads((run / f"manifest.{arm}.json").read_text(encoding="utf-8")),
                     "predictions": read_rows(run / "predictions" / f"{arm}.jsonl"),
                     "events": read_rows(run / "runs" / arm / "events.jsonl"),
                     "requests": read_rows(run / "runs" / arm / "requests.jsonl")})
    return summarize(data, legacy_audit=legacy_audit)


def write_summary(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# L3 工程可观测性汇总", "", f"模式：{result['mode']}；严格验收：{result['strict_acceptance']}。", "",
             "| 臂 | 病例 | 完整 | HTTP尝试 | usage缺失 | 总token | 参考上界/元 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, value in result["per_arm"].items():
        lines.append(f"| {arm} | {value['cases']} | {value['complete']} | {value['http_attempts']} | {value['usage_missing']} | {value['tokens']['total']} | {value['cost_reference']['upper_bound_yuan']} |")
    lines += ["", "reasoning 为 completion 子集，不再次加到 total。cache miss 只采用 API 明示值；缺失不反推。",
              "usage 缺失时相关总额为 null，known_tokens 仅列已知小计。费用为官方峰时全未命中参考，actual_bill=null。",
              "时延分布先按病例合计 HTTP elapsed；不包含未记录的墙钟时间或退避等待。", "",
              "## 字段覆盖", "", "```json", json.dumps(result["field_coverage"], ensure_ascii=False, indent=2), "```", "",
              "历史审计仅披露已有字段和缺项，不证明新事件契约通过，不产生 H1 结论。", ""]
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--legacy-audit", action="store_true", help="List historical field gaps; never a strict acceptance")
    args = parser.parse_args(argv)
    result = summarize_run(args.run, legacy_audit=args.legacy_audit)
    write_summary(result, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
