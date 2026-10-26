"""Offline analysis of recorded full requests; no new predictions on old cases."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / "L1-L5系统"
sys.path.insert(0, str(REPO))
from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider
from layers.layer3.diagnosis.decision_tree import load_candidate_graph, criterion_nodes
from layers.layer3.diagnosis.g1_cache import content_hash, knowledge_hash, cache_context, G1NodeCache, CachedG1Reasoner
from layers.layer3.diagnosis.schemas import CandidateGeneration, DiagnosticCandidate
from layers.layer3.diagnosis.clinicr import CLINICR
from layers.layer3.llm.client import ALLOWED_EVIDENCE_REFS_KEY


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    source = HERE.parent / "p1.0.8b/runs/full-a1-v1"
    rows = [json.loads(x) for x in (source / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    provider = JsonDiagnosticKnowledgeProvider()
    packages = {label: provider.get_candidate(label) for label in load_candidate_graph()["paths"]}
    sizes = {label: len(json.dumps(value, ensure_ascii=False)) for label, value in packages.items()}
    scopes = {}
    for name, chosen, repairs in [("all_observed_with_repairs", rows, True),
                                  ("19_successful_initial_requests", [r for r in rows if r["ok"]], False)]:
        result = Counter()
        distinct = defaultdict(set)
        per_label = Counter()
        for row in chosen:
            for request in row["requests"]:
                if request["stage"] != "criteria" or (not repairs and request["is_structured_retry"]):
                    continue
                labels = list(dict.fromkeys(node.split(".")[1] for node in request["node_ids"]))
                result["requests"] += 1
                for label in labels:
                    result["label_package_transmissions"] += 1
                    result["serialized_package_characters"] += sizes[label]
                    distinct[row["case_id"]].add(label)
                    per_label[label] += 1
        result["distinct_case_label_packages"] = sum(len(labels) for labels in distinct.values())
        result["one_transmission_per_case_label_characters"] = sum(sizes[label] for labels in distinct.values() for label in labels)
        result["repeated_package_characters"] = result["serialized_package_characters"] - result["one_transmission_per_case_label_characters"]
        info = dict(result)
        info["repeated_package_character_fraction"] = result["repeated_package_characters"] / result["serialized_package_characters"]
        info["label_transmissions"] = dict(per_label)
        info["scope"] = "serialized complete label packages only, not total API tokens"
        scopes[name] = info
    requests = [q for row in rows for q in row["requests"]]
    cache_usage = {}
    for stage in ["all", "screening", "anchor", "guard", "criteria", "verifier"]:
        items = [q for q in requests if stage == "all" or q["stage"] == stage]
        total = sum(q["usage"]["prompt_tokens"] for q in items)
        hit = sum(q["usage"]["prompt_cache_hit_tokens"] for q in items)
        cache_usage[stage] = {"input_tokens": total, "cached_input_tokens": hit,
                              "uncached_input_tokens": total - hit, "cached_input_fraction": hit / total}
    inputs = {r["case_id"]: r["profile"] for r in map(json.loads, (HERE.parent / "p1.0.6/inputs.jsonl").read_text(encoding="utf-8").splitlines())}
    digest = knowledge_hash(provider)
    cache = G1NodeCache(HERE / "offline-cache-validation")
    replay = []
    for row in rows:
        if not row["ok"]:
            continue
        profile = {**inputs[row["case_id"]], ALLOWED_EVIDENCE_REFS_KEY: sorted(CLINICR._known_evidence_refs(inputs[row["case_id"]]))}
        context = cache_context(row["case_id"], profile, model="deepseek-flash", service_url="https://api.deepseek.com",
                                knowledge_digest=digest, prompt_version="historical-b564e1e-node-prompt", criterion_batching="legacy")
        report = row["result"]
        cache.save(context, report["tree_trace"], provenance={"source": "p1.0.8b", "offline_validation_only": True,
                   "fingerprints": ["e72d3c7f663d3e1383b00b3884cb4501"]})
        expected = CandidateGeneration(considered_disorders=report["considered_disorders"],
            candidates=[DiagnosticCandidate.model_validate({k: c[k] for k in DiagnosticCandidate.model_fields}) for c in report["candidates"]],
            tree_trace=report["tree_trace"], missing_dimensions=report["candidate_missing_dimensions"], candidate_coverage=report["candidate_coverage"])
        for arm in ["A2", "A3"]:
            generation = CachedG1Reasoner(cache, context, knowledge=provider, mode="require").generate_candidates(profile)
            generation.candidates = CLINICR._validate_generation(generation, CLINICR._known_evidence_refs(inputs[row["case_id"]]))
            assert generation.model_dump(mode="json") == expected.model_dump(mode="json")
            replay.append({"case_id": row["case_id"], "consumer": arm, "identical": True, "model_calls": 0,
                           "node_artifact": cache.path_for(context).name})
    output = {"historical_source": "p1.0.8b/full-a1-v1", "criteria_redundancy": scopes,
              "provider_prefix_cache": cache_usage, "historical_cache_replay": replay,
              "g1_only_reuse_verified_not_full_a2_a3": True,
              "knowledge_hash": digest, "no_new_model_calls_on_original_20": True}
    write(HERE / "baseline-analysis.json", output)
    print(json.dumps({"criteria_redundancy": {k: {x: v[x] for x in v if x != "label_transmissions"} for k, v in scopes.items()},
                      "cached_input_fraction": cache_usage["all"]["cached_input_fraction"],
                      "cached_g1_replays": len(replay), "new_model_calls": 0}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
