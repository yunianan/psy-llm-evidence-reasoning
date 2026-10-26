"""Research-only repaired batches; production remains disconnected."""
import json
from pathlib import Path
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / "L1-L5系统"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent / "p1.1-engineering"))
import run_cached_smoke as prior
cost = prior.cost

from layers.layer3.diagnosis.g1_cache import G1NodeCache, CachedG1Reasoner, cache_context, knowledge_hash
from layers.layer3.diagnosis.schemas import DiagnosticCandidate
from layers.layer3.llm import client as node_client
from tools.l3_engineering_runtime import NodeCheckpoint, attach_checkpoints, now

POLICY = {"read_timeout_seconds": 300, "stream": True, "sdk_max_retries": 0,
          "max_http_attempts": 3, "retry": "exponential backoff + jitter; no retry for 4xx contract/auth errors",
          "max_workers": 2, "max_tokens": 16000, "structured_repairs_per_batch": 1,
          "reference_case_limit_yuan": 10, "reference_batch_limit_yuan": 100,
          "reference_pricing": "official peak all input cache miss; not actual platform billing"}


def read(path):
    return cost.base.read_rows(path)


def model_profile(row):
    profile = row["profile"]
    return {**profile, node_client.ALLOWED_EVIDENCE_REFS_KEY: sorted(cost.base.CLINICR._known_evidence_refs(profile))}


def context(row, provider, thinking, *, transport_policy=False):
    options = {"max_tokens": 16000, "enable_thinking": thinking, "temperature": 0,
               "response_format": {"type": "json_object"}}
    if transport_policy:
        options["transport_policy"] = POLICY
    return cache_context(row["case_id"], model_profile(row), model=cost.base.APICLINICRReasoner(knowledge=provider).model,
        service_url=cost.base.transport.get_base_url(), knowledge_digest=knowledge_hash(provider),
        prompt_version=":".join([node_client.NODE_PROMPT_VERSION, node_client.SCREENING_PROMPT_VERSION, node_client.ANCHOR_PROMPT_VERSION]),
        criterion_batching="per_label", static_knowledge_prefix=True, generation_options=options)


def run_case(row, *, arm, thinking, run, transport, checkpoint_root, g1_only=False):
    record = {"case_id": row["case_id"], "arm": arm, "thinking_override": thinking,
              "started_at": now(), "requests": []}
    transport.bind(row["case_id"], record["requests"], thinking=thinking, arm=arm)
    provider = cost.base.JsonDiagnosticKnowledgeProvider()
    delegate = cost.base.APICLINICRReasoner(knowledge=provider, criterion_batching="per_label", static_knowledge_prefix=True)
    checkpoint = NodeCheckpoint(checkpoint_root, context(row, provider, thinking))
    record["checkpoint_nodes_before"] = len(checkpoint.nodes)
    stats = attach_checkpoints(delegate, checkpoint, transport)
    full_context = context(row, provider, thinking, transport_policy=True)
    cache = G1NodeCache(run / "g1-nodes")
    reasoner = CachedG1Reasoner(cache, full_context, knowledge=provider, delegate=delegate, mode="record",
        provenance=lambda: {"source_run": run.name, "thinking_override": thinking,
                            "checkpoint_nodes_imported": record["checkpoint_nodes_before"], "transport_policy": POLICY})
    reward = cost.APIRewardModelVerifier(knowledge=provider)
    engine = cost.base.CLINICR(reasoner=reasoner, verifier=cost.IndependentVerifier(model=reward, knowledge=provider), knowledge=provider)
    started = perf_counter()
    try:
        if g1_only:
            generation = reasoner.generate_candidates(model_profile(row))
            generation.candidates = cost.base.CLINICR._validate_generation(generation, cost.base.CLINICR._known_evidence_refs(row["profile"]))
            result = generation.model_dump(mode="json")
        else:
            result = engine.run(row["profile"])
        record.update(ok=True, result=result, failure_kind=None, infrastructure_success=True)
    except Exception as error:
        chain = cost.exception_chain(error)
        types = {item["type"] for item in chain}
        budget = bool(types & {"CaseBudgetExceeded", "BatchBudgetExceeded"})
        network = any(r.get("error_type") for r in record["requests"]) and not all(
            r.get("error_type") is None for r in record["requests"][-1:])
        storage = bool(types & {"PermissionError", "FileNotFoundError", "G1CacheIntegrityError"})
        contract = bool(types & {"ValueError", "ValidationError"})
        kind = "budget" if budget else "transport_or_http" if network else "storage" if storage else "contract_failure" if contract else "model_failure"
        record.update(ok=False, exception_chain=chain, failure_kind=kind,
                      infrastructure_success=kind in {"contract_failure", "model_failure"})
    record.update(completed_at=now(), elapsed_seconds=round(perf_counter() - started, 3),
                  checkpoint_stats=stats, checkpoint_nodes_after=len(checkpoint.nodes),
                  screen_denial_scope_events=reasoner.screen_denial_scope_events,
                  g1_cache_key=cache.path_for(full_context).name,
                  g1_shared_validation={"source_g1_unavailable": True, "new_model_calls": 0})
    record["cost_accounting"] = cost.accounting(record["requests"])
    envelope = cache.load(full_context)
    if envelope:
        generations = []
        for _ in ["A2", "A3"]:
            generation = CachedG1Reasoner(cache, full_context, knowledge=provider, mode="require").generate_candidates(model_profile(row))
            generation.candidates = cost.base.CLINICR._validate_generation(generation, cost.base.CLINICR._known_evidence_refs(row["profile"]))
            generations.append(generation.model_dump(mode="json"))
        identical = generations[0] == generations[1]
        if record["ok"]:
            result = record["result"]
            expected_candidates = [DiagnosticCandidate.model_validate({k: c[k] for k in DiagnosticCandidate.model_fields}).model_dump(mode="json") for c in result["candidates"]]
            identical = (identical and generations[0]["candidate_coverage"] == result["candidate_coverage"]
                         and generations[0]["tree_trace"] == result["tree_trace"] and generations[0]["candidates"] == expected_candidates)
        record["g1_shared_validation"] = {"identical": identical, "new_model_calls": 0}
    return record
