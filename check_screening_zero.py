"""P1.0.5: compare zero-knowledge screening with the preserved P1.0 result."""
import json
from pathlib import Path
from unittest.mock import patch

import check_dev_screens as smoke
from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider
from layers.layer3.llm.client import APICLINICRReasoner
from layers.layer3.llm.node_knowledge import NODE_KNOWLEDGE_VIEW_VERSION
from layers.layer3.diagnosis.decision_tree import load_candidate_graph


BASELINE = Path(__file__).with_name("dev-screen-smoke.json")
RESULT = Path(__file__).with_name("dev-screen-screening-zero-v2.json")


def token_total(requests, key):
    values = [(item.get("usage") or {}).get(key) for item in requests]
    return sum(values) if values and all(isinstance(value, int) for value in values) else None


def main():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert baseline["ok"], "Comparison requires the successful original baseline"
    provider = JsonDiagnosticKnowledgeProvider()
    smoke.RESULT = RESULT
    with (
        patch.object(smoke, "JsonDiagnosticKnowledgeProvider", return_value=provider),
        patch.object(provider, "get_candidate", wraps=provider.get_candidate) as candidate_calls,
        patch.object(provider, "get_symptoms", wraps=provider.get_symptoms) as symptom_calls,
        patch.object(smoke, "APICLINICRReasoner", side_effect=lambda **kwargs: APICLINICRReasoner(
            **kwargs, screening_zero_knowledge=True,
        )),
    ):
        smoke.main()
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    before = {item["node_id"]: item for item in baseline["judgments"]}
    after = {item["node_id"]: item for item in result.get("judgments", [])}
    comparison = [{
        "node_id": node,
        "before_status": before[node]["status"],
        "after_status": after.get(node, {}).get("status"),
        "before_confidence": before[node]["confidence"],
        "after_confidence": after.get(node, {}).get("confidence"),
    } for node in baseline["expected_node_ids"]]
    changed = [item for item in comparison if item["before_status"] != item["after_status"]]
    before_tokens = token_total(baseline["requests"], "prompt_tokens")
    after_tokens = token_total(result["requests"], "prompt_tokens")
    fingerprints = lambda rows: {item.get("system_fingerprint") for item in rows}
    same_inputs = all(result[key] == baseline[key] for key in (
        "case_id", "model", "tree_version", "expected_node_ids", "expected_batch_sizes",
    ))
    same_fingerprint = fingerprints(result["requests"]) == fingerprints(baseline["requests"])
    no_knowledge_reads = candidate_calls.call_count == symptom_calls.call_count == 0
    within_budget = after_tokens is not None and after_tokens <= 7000
    graph = load_candidate_graph()
    def active(statuses):
        return sorted(label for label, path in graph["paths"].items()
                      if any(statuses.get(node, {}).get("status") != "not_met" for node in path["screens"]))
    activity_same = bool(result["ok"] and active(before) == active(after))
    technical_pass = bool(result["ok"] and same_inputs and same_fingerprint and no_knowledge_reads
                          and within_budget and activity_same)
    result.update(
        phase="P1.0.5", node_knowledge_view_version=NODE_KNOWLEDGE_VIEW_VERSION,
        screening_zero_knowledge_enabled=True,
        baseline_file=BASELINE.name, input_identity_matches_baseline=same_inputs,
        response_fingerprint_matches_baseline=same_fingerprint,
        knowledge_provider_calls={"get_candidate": candidate_calls.call_count,
                                  "get_symptoms": symptom_calls.call_count},
        node_comparison=comparison, status_changes=changed,
        prompt_tokens_before=before_tokens, prompt_tokens_after=after_tokens,
        prompt_token_reduction_percent=(round(100 * (before_tokens - after_tokens) / before_tokens, 4)
                                        if before_tokens and after_tokens is not None else None),
        input_token_target=7000, input_token_target_met=within_budget,
        activity_sets_identical=activity_same,
        status_agreement_is_observation_only=True,
        p1_0_5_gate="awaiting_20_case_semantic_audit" if technical_pass else "hold",
    )
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "p1_0_5_gate", "prompt_tokens_before", "prompt_tokens_after",
        "prompt_token_reduction_percent", "knowledge_provider_calls", "status_changes",
    )}, ensure_ascii=False))
    return 0 if technical_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
