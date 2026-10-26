"""Phase 1 A0/A1 research CLI. Offline replay by default; never use /diagnose."""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.a0_no_knowledge import A0NoKnowledge, METHOD, PROMPT_VERSION, known_patient_refs
from tools.prediction_contract import keyed, normalize_for_protocol
from tools.paired_evaluation import monitor_block, read_rows
from layers.layer3.diagnosis.schemas import CandidateGeneration, DiagnosticCandidate
from layers.layer3.llm import client
from layers.layer3.llm.observability import (
    capture_events, EVENT_VERSION, ENVELOPE_VERSION, EventJournal, current_context, update_context,
)
from layers.layer3.diagnosis.g1_cache import G1NodeCache, CachedG1Reasoner, cache_context, content_hash, knowledge_hash
from layers.layer3.diagnosis.g1_cache import G1CacheIntegrityError
from tools.l3_engineering_runtime import (
    ReliableStreaming, ReferenceBudget, NodeCheckpoint, attach_checkpoints, recover_ledger,
    CaseBudgetExceeded, BatchBudgetExceeded, error_category, exception_types, now, legacy_stream_retryable,
)

VERSION = "phase1-research-entry-v4"
POLICY = {"thinking": True, "screening": "zero", "anchor_criteria_guard": "full", "read_timeout": 300,
          "stream": True, "include_usage": True, "sdk_max_retries": 0, "max_http_attempts": 3,
          "structured_repairs": 1, "max_tokens": 16000, "max_workers": 1,
          "criterion_batching": "per_label", "static_knowledge_prefix": True}


class ResearchIntegrityError(ValueError):
    pass


def _cache_directories(value):
    return [Path(x) for x in value] if isinstance(value, (list, tuple)) else [Path(value)]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _identity():
    names = ["tools/phase1_research_entry.py", "tools/a0_no_knowledge.py", "tools/prediction_contract.py",
             "tools/paired_evaluation.py", "tools/l3_engineering_runtime.py", "shared/disorder_catalog.py",
             "layers/layer3/llm/client.py", "layers/layer3/llm/observability.py",
             "layers/layer3/diagnosis/screening.py", "tools/l3_observability_summary.py",
             "tools/phase1_joint_run.py"]
    return {name: content_hash((ROOT / name).read_text(encoding="utf-8")) for name in names}


def _report_final(report):
    diagnosis = report.get("diagnosis", {})
    return [x["disorder"] for x in [diagnosis.get("primary"), *diagnosis.get("secondary", [])] if x]


def export_prediction(case_id, arm, report=None, *, failure_class=None):
    failed = failure_class is not None
    payload = copy.deepcopy(report) if report is not None else {}
    if failed:
        payload.update(candidates=[], candidate_coverage={}, tree_trace=[], final_diagnoses=[])
        payload.pop("diagnosis", None)
    else:
        payload.setdefault("final_diagnoses", _report_final(payload))
    method = METHOD if arm == "A0" else "dsm5_decision_tree"
    payload["candidate_generation_method"] = method
    row = {"case_id": case_id, "arm": arm, "candidate_generation_method": method,
           "model_failure": failure_class in {"transport", "contract"}, "run_failure": failed,
           "failure_class": failure_class, "report": payload}
    normalize_for_protocol(row)
    return row


def run_arm(arm, row, *, request=None, model="offline-fixture", knowledge=None, reasoner=None, verifier=None):
    """Injectable engine boundary, also used by sentinel and engine-replay tests."""
    if arm == "A0":
        generation = A0NoKnowledge(request, model=model, knowledge=knowledge).generate_candidates(row["profile"])
        return export_prediction(row["case_id"], arm, {**generation.model_dump(mode="json"), "final_diagnoses": []})
    if arm != "A1":
        raise ValueError("Phase 1 supports only A0/A1")
    from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider
    from layers.layer3.diagnosis.clinicr import CLINICR
    provider = knowledge if knowledge is not None else JsonDiagnosticKnowledgeProvider()
    delegate = reasoner or client.APICLINICRReasoner(model=model, knowledge=provider,
        criterion_batching="per_label", static_knowledge_prefix=True)
    return export_prediction(row["case_id"], arm, CLINICR(reasoner=delegate, verifier=verifier, knowledge=provider).run(row["profile"]))


def classify_failure(error):
    chain, seen = [], set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    if any(isinstance(e, (CaseBudgetExceeded, BatchBudgetExceeded)) for e in chain):
        return "budget"
    if any(isinstance(e, (ResearchIntegrityError, G1CacheIntegrityError, FileNotFoundError, PermissionError)) for e in chain):
        return "administrative"
    if any(isinstance(e, PersistedRequestFailure) for e in chain):
        return next(e.failure_class for e in chain if isinstance(e, PersistedRequestFailure))
    if any(error_category(e) in {"timeout", "connection_error", "connection_reset", "provider_internal_error"}
           or type(e).__name__ in {"CLINICRAPIError", "NonRetryableProviderError"} for e in chain):
        return "transport"
    if any(isinstance(e, (ValueError, KeyError)) for e in chain):
        return "contract"
    return "administrative"


class PersistedRequestFailure(RuntimeError):
    def __init__(self, failure_class):
        self.failure_class = failure_class
        super().__init__("A terminal or exhausted request cannot be regenerated")


class DurableRequests:
    """Recover exact request results/allowances; never reset attempts on resume."""
    def __init__(self, transport, journal_path):
        self.transport, self.path = transport, Path(journal_path)

    def __call__(self, prompt, **kwargs):
        runtime = self.transport
        key = content_hash({"case_id": runtime.local.case_id, "prompt": prompt, "kwargs": kwargs})
        events = read_rows(self.path) if self.path.exists() else []
        previous = {r["attempt_id"]: r for r in events if r.get("logical_request_key") == key}
        if previous:
            last = list(previous.values())[-1]
            if last.get("event") == "attempt_finished" and not last.get("error_type"):
                update_context(attempt_id=last["attempt_id"],
                               request_id=last.get("response_request_id") or last.get("completion_id"),
                               finish_reason=last.get("finish_reason"))
                return last["raw_output"]
            recovery = getattr(runtime, "engineering_recovery", False)
            if len(previous) >= 3 or (last.get("event") == "attempt_finished" and not last.get("retryable")
                                      and not (recovery and legacy_stream_retryable(last))):
                raise PersistedRequestFailure("transport")
        original = runtime.max_attempts
        runtime.max_attempts = 3 - len(previous)
        runtime.local.tags.update(logical_request_key=key, prior_http_attempts=len(previous))
        try:
            return runtime.ask(prompt, **kwargs)
        finally:
            runtime.max_attempts = original


def _validate_replay(row, source, cache_dir=None):
    if source.get("run_failure") or source.get("model_failure"):
        raise ResearchIntegrityError("Only complete historical A1 reports may be replayed")
    report = copy.deepcopy(source.get("report", source))
    generation = CandidateGeneration(
        considered_disorders=report["considered_disorders"],
        candidates=[DiagnosticCandidate.model_validate({k: c[k] for k in DiagnosticCandidate.model_fields if k in c})
                    for c in report["candidates"]],
        tree_trace=report["tree_trace"], missing_dimensions=report.get("candidate_missing_dimensions", []),
        candidate_coverage=report["candidate_coverage"],
    )
    from layers.layer3.diagnosis.clinicr import CLINICR
    CLINICR._validate_generation(generation, set(known_patient_refs(row["profile"])))
    if cache_dir is not None:
        from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider
        provider = JsonDiagnosticKnowledgeProvider()
        envelopes = [(p, json.loads(p.read_text(encoding="utf-8"))) for directory in _cache_directories(cache_dir) for p in directory.glob("*.json")]
        matches = [(p, e) for p, e in envelopes if e["context"]["case_id"] == row["case_id"]]
        if len(matches) != 1:
            raise ResearchIntegrityError("Exactly one matching source G1 artifact required")
        source_path, envelope = matches[0]
        context = envelope["context"]
        profile = {**row["profile"], client.ALLOWED_EVIDENCE_REFS_KEY: known_patient_refs(row["profile"])}
        if context["profile_hash"] != content_hash(profile) or context["knowledge_hash"] != knowledge_hash(provider):
            raise ResearchIntegrityError("G1 source profile/knowledge differs")
        from layers.layer3.diagnosis.decision_tree import load_candidate_graph
        if context["tree_hash"] != content_hash(load_candidate_graph()):
            raise ResearchIntegrityError("G1 source tree differs")
        replay = CachedG1Reasoner(G1NodeCache(source_path.parent), context, knowledge=provider, mode="require").generate_candidates(profile)
        replay.candidates = CLINICR._validate_generation(replay, set(known_patient_refs(row["profile"])))
        if replay.model_dump(mode="json") != generation.model_dump(mode="json"):
            raise ResearchIntegrityError("Historical report disagrees with read-only G1 replay")
    return report


def _live_a1(row, transport, out, request):
    from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider
    from layers.layer3.diagnosis.clinicr import CLINICR
    from layers.layer3.diagnosis.verifier import IndependentVerifier
    provider = JsonDiagnosticKnowledgeProvider()
    model = transport.local.model
    profile = {**row["profile"], client.ALLOWED_EVIDENCE_REFS_KEY: known_patient_refs(row["profile"])}
    context = cache_context(row["case_id"], profile, model=model, service_url=transport.base_url,
        knowledge_digest=knowledge_hash(provider), prompt_version=":".join((client.NODE_PROMPT_VERSION,
            client.SCREENING_PROMPT_VERSION, client.ANCHOR_PROMPT_VERSION)),
        criterion_batching="per_label", static_knowledge_prefix=True, generation_options=POLICY)
    delegate = client.APICLINICRReasoner(model=model, knowledge=provider, criterion_batching="per_label", static_knowledge_prefix=True)
    stats = attach_checkpoints(delegate, NodeCheckpoint(out / "checkpoints/A1", context), transport)
    reasoner = CachedG1Reasoner(G1NodeCache(out / "g1-nodes"), context, knowledge=provider, delegate=delegate,
        mode="record", provenance=lambda: {"source_run": str(out.name), "event_version": EVENT_VERSION,
            "envelope_version": ENVELOPE_VERSION,
            "code_versions": _identity(), "response_models": sorted({r.get("response_model", "provider_not_exposed") for r in transport.local.requests})})
    verifier = IndependentVerifier(model=client.APIRewardModelVerifier(model=model, knowledge=provider), knowledge=provider)
    with patch.object(client, "ask_llm_stream", request):
        result = CLINICR(reasoner=reasoner, verifier=verifier, knowledge=provider).run(row["profile"])
    return export_prediction(row["case_id"], "A1", result), stats


def research_run(arm, rows, out, *, replay_rows=None, response_rows=None, cache_dir=None,
                 resume=False, model="offline-replay", budget_ref=None, execute=False, executor=None,
                 coordination=None, continuation=None):
    if arm not in {"A0", "A1"}:
        raise ValueError("Phase 1 supports only A0/A1")
    indexed = keyed(rows)
    if not indexed or any(not isinstance(r.get("profile"), dict) or not re.fullmatch(r"[A-Za-z0-9_-]+", r["case_id"]) for r in rows):
        raise ValueError("Nonempty profiles and safe unique case IDs required")
    if execute and (replay_rows is not None or response_rows is not None or executor is not None):
        raise ValueError("Live and replay modes cannot be mixed")
    if execute:
        from urllib.parse import urlsplit
        from dotenv import load_dotenv
        from shared.llm_client import get_base_url
        load_dotenv(ROOT / ".env", override=False)
        if model.startswith("offline"):
            raise ValueError("Explicit live model name required")
        if not isinstance(budget_ref, dict) or budget_ref.get("pricing_basis") != "official_peak_all_input_miss_reference":
            raise ValueError("Explicit official-price reference budget required")
        if set(budget_ref) - {"pricing_basis", "batch_limit_yuan", "case_limit_yuan"}:
            raise ValueError("Unexpected budget configuration fields")
        if any(type(budget_ref.get(k)) not in (int, float) or not 0 < budget_ref[k] < float("inf")
               for k in ("batch_limit_yuan", "case_limit_yuan")):
            raise ValueError("Positive finite budget limits required")
        parsed = urlsplit(get_base_url())
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Credential-bearing service URLs cannot enter manifests")
    if not execute and executor is None and replay_rows is None and response_rows is None:
        raise ValueError("Offline response/prediction source required; live calls need --execute")
    out = Path(out)
    sources = keyed(replay_rows) if replay_rows is not None else {}
    responses = keyed(response_rows) if response_rows is not None else {}
    if sources and not set(indexed) <= set(sources) or responses and not set(indexed) <= set(responses):
        raise ValueError("Offline source misses planned IDs")
    # Read this explicitly selected workspace under the sandbox account as well;
    # do not modify the user's global safe.directory configuration.
    commit = subprocess.check_output(["git", "-c", f"safe.directory={ROOT}", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    identity = {"version": VERSION, "arm": arm, "input_hash": content_hash(rows), "case_ids": list(indexed),
                "code_commit": commit, "code_versions": _identity(), "policy": POLICY,
                "requested_model": model, "response_model_before_run": None,
                "response_provenance": "recorded after response; never a pre-request fingerprint",
                "seed": 20261015, "event_version": EVENT_VERSION, "envelope_version": ENVELOPE_VERSION,
                "budget_reference": budget_ref,
                "mode": "live" if execute else "offline_replay", "eval_opened": False,
                "source_hash": content_hash(replay_rows if replay_rows is not None else response_rows) if executor is None else "injected_test_executor"}
    identity["coordination"] = coordination.policy if coordination is not None else None
    identity["service_url"] = get_base_url() if execute else None
    identity["g1_source_directories"] = [str(p.resolve()) for p in _cache_directories(cache_dir)] if cache_dir else []
    if continuation is not None:
        identity["engineering_continuation"] = continuation
    if arm == "A1":
        from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider
        provider = JsonDiagnosticKnowledgeProvider()
        identity.update(tree_blob=content_hash((ROOT / "knowledge/dsm5_decision_tree/candidate_paths.json").read_text(encoding="utf-8")),
                        knowledge_hash=knowledge_hash(provider),
                        prompt_version=":".join((client.NODE_PROMPT_VERSION, client.SCREENING_PROMPT_VERSION, client.ANCHOR_PROMPT_VERSION)))
    else:
        identity.update(tree_blob=None, knowledge_hash=None, prompt_version=PROMPT_VERSION, knowledge_calls=0)
    manifest_path = out / f"manifest.{arm}.json"
    if manifest_path.exists():
        original = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not resume or original["identity"] != identity:
            raise ValueError("Resume requires identical inputs, code, source and parameters")
    else:
        if resume:
            raise ValueError("Cannot resume a missing manifest")
        write_json(manifest_path, {"identity": identity, "started_at": now()})
    run = out / "runs" / arm
    run.mkdir(parents=True, exist_ok=True)
    request_path = run / "requests.jsonl"
    transport = None
    budget = None
    if execute:
        if not isinstance(budget_ref, dict) or budget_ref.get("pricing_basis") != "official_peak_all_input_miss_reference":
            raise ValueError("Explicit official-price reference budget required for this transport")
        from dotenv import load_dotenv
        from shared.llm_client import get_api_key, get_base_url
        load_dotenv(ROOT / ".env", override=False)
        budget = coordination.arm_budget(arm) if coordination is not None else ReferenceBudget(batch_limit=budget_ref["batch_limit_yuan"], case_limit=budget_ref["case_limit_yuan"])
        if coordination is None and resume and request_path.exists():
            old = recover_ledger(request_path)
            budget.spent, budget.by_case, budget.unknown_upper = old["spent"], old["by_case"], old["unknown_upper"]
    journal = request_path.open("a", encoding="utf-8")
    if execute:
        transport = ReliableStreaming(api_key=get_api_key(), base_url=get_base_url(), journal=journal, budget=budget,
            max_attempts=3, read_timeout=300, max_tokens=16000, thinking=True)
        transport.engineering_recovery = continuation is not None
        durable = DurableRequests(transport, request_path)
    results, stopped = [], False
    events = EventJournal(run / "events.jsonl", run_id=out.name, arm=arm)
    with journal:
        for number, row in enumerate(rows):
            path = run / f"{row['case_id']}.json"
            if path.exists():
                result = json.loads(path.read_text(encoding="utf-8"))
                if result["case_id"] != row["case_id"]:
                    raise ValueError("Stored result ID differs")
                results.append(result)
                stopped = stopped or result.get("stop_after_case", False)
                continue
            if coordination is not None:
                joint_stopped = coordination.before_case(arm, row["case_id"])
                stopped = stopped or joint_stopped
            def event(value):
                tags = getattr(transport.local, "tags", {}) if transport else {"stage": "a0" if arm == "A0" else "replay", "node_ids": []}
                events({"case_id": row["case_id"], "arm": arm, **tags, **current_context(), **value})
            with capture_events(event, case_id=row["case_id"], arm=arm,
                                stage="a0" if arm == "A0" else "replay" if not execute else "administrative",
                                scope="case", node_ids=[], attempt_id=None, request_id=None,
                                source="live" if execute else "injected_test" if executor else "offline"):
                if transport:
                    transport.bind(row["case_id"], [], arm=arm)
                    transport.local.model = model
                    transport.tags("a0" if arm == "A0" else "administrative", [])
                stop_reasons, error_fields = [], {}
                try:
                    if stopped:
                        update_context(stage="administrative", scope="run", node_ids=[], attempt_id=None, request_id=None)
                        prediction = export_prediction(row["case_id"], arm, failure_class="administrative")
                    elif executor is not None:
                        prediction = executor(row)
                    elif execute:
                        if arm == "A0":
                            prediction = run_arm(arm, row, request=durable, model=model)
                        else:
                            prediction, _stats = _live_a1(row, transport, out, durable)
                    elif arm == "A1" and sources:
                        try:
                            prediction = export_prediction(row["case_id"], arm, _validate_replay(row, sources[row["case_id"]], cache_dir))
                        except Exception as error:
                            raise ResearchIntegrityError("Invalid historical A1/G1 replay source") from error
                    elif arm == "A0" and responses:
                        messages = iter(responses[row["case_id"]]["responses"])
                        def request(prompt, **kwargs):
                            event({"event": "offline_response_replayed", "stage": "a0", "new_model_calls": 0})
                            return next(messages)
                        prediction = run_arm(arm, row, request=request, model=model)
                    else:
                        raise ValueError("A0 requires responses; A1 requires saved reports")
                except Exception as error:
                    kind = classify_failure(error)
                    prediction = export_prediction(row["case_id"], arm, failure_class=kind)
                    error_fields = {"exception_types": exception_types(error), "transport_category": error_category(error)}
                    if isinstance(error, (ResearchIntegrityError, G1CacheIntegrityError, FileNotFoundError, PermissionError)):
                        prediction["protocol_integrity_failure"] = True
                        stopped = True
                        stop_reasons.append("protocol_integrity")
                    if isinstance(error, BatchBudgetExceeded):
                        stopped = True
                        stop_reasons.append("budget")
                normalize_for_protocol({k: v for k, v in prediction.items() if k != "protocol_integrity_failure"})
                if prediction["run_failure"]:
                    event({"event": "terminal_failure", "failure_class": prediction["failure_class"], **error_fields})
                event({"event": "case_finished", "scope": "case", "failure_class": prediction["failure_class"],
                       "usage_missing": None if not execute else any(r.get("usage") is None for r in transport.local.requests),
                       "mode": identity["mode"]})
                result = {**prediction, "completed_at": now(), "replay_only": not execute,
                          "new_model_calls": 0 if not execute else len(transport.local.requests)}
                failures_so_far = sum(r["run_failure"] for r in [*results, result])
                if not stopped and failures_so_far > len(rows) // 20:
                    stopped = True
                    stop_reasons.append("completion_impossible")
                block = [*results[-(number % 20):], result] if number % 20 else [result]
                if len(block) == 20 or number == len(rows) - 1:
                    result["monitoring"] = monitor_block(block)
                    if result["monitoring"]["contract_stop"]:
                        stop_reasons.append("monitor_contract")
                    if result["monitoring"]["transport_stop"]:
                        stop_reasons.append("monitor_transport")
                    stopped = stopped or result["monitoring"]["stop"]
                for reason in stop_reasons:
                    event({"event": "batch_stopped", "scope": "run", "reason": reason})
                result["stop_after_case"] = stopped
                write_json(path, result)
                results.append(result)
                if coordination is not None:
                    coordination.after_case(arm, row["case_id"], result)
    predictions = [{k: v for k, v in r.items() if k not in {"completed_at", "replay_only", "new_model_calls", "monitoring", "stop_after_case"}} for r in results]
    destination = out / "predictions" / f"{arm}.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in predictions), encoding="utf-8")
    ledger = read_rows(request_path)
    finished = {r["attempt_id"]: r for r in ledger}
    carried = set((continuation or {}).get("imported_attempt_ids", {}).get(arm, []))
    summary = {"arm": arm, "cases": len(results), "complete": sum(not r["run_failure"] for r in results),
               "mode": identity["mode"], "new_model_calls": len(set(finished) - carried),
               "carried_http_attempts": len(set(finished) & carried), "execution_stopped": stopped,
               "response_models": sorted({r["response_model"] for r in finished.values() if r.get("response_model")}),
               "fingerprints": sorted({r["system_fingerprint"] for r in finished.values() if r.get("system_fingerprint")}),
               "usage_missing_requests": sum(r.get("usage") is None for r in finished.values()),
               "protocol_integrity_failures": sum(r.get("protocol_integrity_failure", False) for r in results),
               "budget": budget.snapshot() if budget else {"new_model_cost_yuan": 0, "source_inference_cost_reused_not_recharged": True},
               "formal_scoring": False, "eval_opened": False}
    write_json(out / f"summary.{arm}.json", summary)
    index = {p.stem: p.name for p in out.glob("manifest.A*.json")}
    if coordination is None:
        write_json(out / "manifest.json", {"version": VERSION, "arm_manifests": index, "eval_opened": False})
    from tools.l3_observability_summary import summarize_run, write_summary
    write_summary(summarize_run(out, arms=[arm]), out / "observability" / f"{arm}.summary.json")
    return predictions, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["A0", "A1"], required=True)
    parser.add_argument("--dev-list", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--budget-ref", type=Path)
    parser.add_argument("--replay-predictions", type=Path)
    parser.add_argument("--a0-responses", type=Path)
    parser.add_argument("--g1-cache-dir", type=Path, action="append")
    parser.add_argument("--model", default="offline-replay")
    parser.add_argument("--execute", action="store_true", help="Explicitly enable paid calls; never needed for offline construction")
    args = parser.parse_args(argv)
    if args.dev_list.suffix.lower() != ".jsonl" or any(p.lower() == "sealed" or p.lower().startswith("eval.") for p in args.dev_list.parts):
        parser.error("This entry accepts DEV JSONL only, never sealed/EVAL input")
    rows = read_rows(args.dev_list)
    if args.max_cases is not None:
        if args.max_cases < 1:
            parser.error("max-cases must be positive")
        rows = rows[:args.max_cases]
    _, summary = research_run(args.arm, rows, args.out, resume=args.resume, model=args.model, execute=args.execute,
        replay_rows=read_rows(args.replay_predictions) if args.replay_predictions else None,
        response_rows=read_rows(args.a0_responses) if args.a0_responses else None, cache_dir=args.g1_cache_dir,
        budget_ref=json.loads(args.budget_ref.read_text(encoding="utf-8")) if args.budget_ref else None)
    print(json.dumps(summary, ensure_ascii=False))
    return 2 if summary["protocol_integrity_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
