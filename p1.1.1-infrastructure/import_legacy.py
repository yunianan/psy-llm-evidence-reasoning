"""Recover only validated completed node batches; never overwrite first-attempt outputs."""
import json
from unittest.mock import Mock, patch

from common import HERE, cost, read, context, model_profile, node_client
from layers.layer3.diagnosis.decision_tree import load_candidate_graph, criterion_nodes
from shared.disorder_catalog import SUPPORTED_DISORDERS
from tools.l3_engineering_runtime import NodeCheckpoint


def main():
    previous = HERE.parent / "p1.1-engineering"
    rows = {r["case_id"]: r for r in read(HERE / "repair9.inputs.jsonl")}
    provider = cost.base.JsonDiagnosticKnowledgeProvider()
    definitions = dict(load_candidate_graph()["nodes"])
    for label in SUPPORTED_DISORDERS:
        definitions.update(criterion_nodes(label, knowledge=provider))
    summary = []
    for name in ["cached-full-smoke-v1", "cached-full-smoke25-continuation-v1", "cached-full-smoke25-bounded-v1"]:
        folder = previous / "runs" / name
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        for row in read(folder / "results.jsonl"):
            if not row["attempted"] or row["case_id"] not in rows:
                continue
            profile_row = rows[row["case_id"]]
            current = context(profile_row, provider, None)
            assert current["model"] == manifest["reasoner_model"] and current["service_url"] == manifest["service_url"]
            assert current["knowledge_hash"] == manifest["knowledge_hash"]
            checkpoint = NodeCheckpoint(HERE / "repair-checkpoints", current)
            delegate = cost.base.APICLINICRReasoner(knowledge=provider, criterion_batching="per_label", static_knowledge_prefix=True)
            rejected, accepted = [], 0
            for index, request in enumerate(row["requests"], 1):
                if request["stage"] == "verifier" or request.get("error_type") or not request.get("raw_output"):
                    continue
                nodes = [{"node_id": key, **definitions[key]} for key in request["node_ids"]]
                responder = Mock(return_value=request["raw_output"])
                try:
                    with patch.object(node_client, "ask_llm_stream", responder):
                        answers = delegate._evaluate_tree_nodes(model_profile(profile_row), nodes)
                    assert responder.call_count == 1, "A completed batch must pass validation without a fabricated repair"
                except (ValueError, AssertionError) as error:
                    rejected.append({"request_number": index, "reason": type(error).__name__})
                    continue
                checkpoint.save(nodes, answers, origin=f"validated_legacy:{name}:request-{index}")
                accepted += 1
            summary.append({"case_id": row["case_id"], "source_run": name, "accepted_batches": accepted,
                            "checkpoint_nodes": len(checkpoint.nodes), "rejected_batches": rejected,
                            "new_model_calls": 0})
    assert len(summary) == 9
    cost.write(HERE / "legacy-import.json", {"cases": summary, "new_model_calls": 0,
               "semantic_thinking_mode": "provider default retained; no mixed-mode cache reuse"})
    print(json.dumps({"cases": len(summary), "recovered_nodes": sum(r["checkpoint_nodes"] for r in summary), "new_model_calls": 0}))


if __name__ == "__main__":
    main()
