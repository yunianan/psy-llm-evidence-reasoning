"""Two identical synthetic full-anchor requests to inspect new-source usage."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "p1.0.8b"))
import run_full_cost as cost


def main():
    output = HERE / "provider-cache-probe.json"
    assert not output.exists(), "Preserve the first probe"
    graph = cost.base.load_candidate_graph()
    node = {"node_id": "anchor.Insomnia", **graph["nodes"]["anchor.Insomnia"]}
    quote = "Synthetic interface fixture only: for two weeks I have needed two hours to fall asleep."
    profile = {"key_narratives": [{"text": quote, "sources": ["probe-p1"]}],
               "evidence_nodes": [{"evidence_id": "probe-p1", "quote": quote}],
               cost.base.ALLOWED_EVIDENCE_REFS_KEY: ["probe-p1"]}
    record = {"case_id": "synthetic-cache-probe", "requests": []}
    cost.base.LOCAL.record = record
    observed = cost.observer(cost.base.transport._bounded_client)
    def bounded(*, timeout):
        api = observed(timeout=timeout)
        create = api.chat.completions.create
        def with_usage(**request):
            request["stream_options"] = {"include_usage": True}
            return create(**request)
        api.chat.completions.create = with_usage
        return api
    result = {"started_at": cost.base.now(), "model_alias": cost.base.transport.get_model(),
              "base_url": cost.base.transport.get_base_url(), "synthetic_only": True,
              "request_stream_usage": True, "attempts": []}
    with patch.object(cost.base.transport, "_bounded_client", bounded):
        for index in range(2):
            reasoner = cost.base.APICLINICRReasoner(knowledge=cost.base.JsonDiagnosticKnowledgeProvider())
            try:
                answers = reasoner._evaluate_tree_nodes(profile, [node])
                result["attempts"].append({"index": index, "valid_node_output": True,
                                           "judgments": [r.model_dump(mode="json") for r in answers]})
            except Exception as error:
                result["attempts"].append({"index": index, "valid_node_output": False,
                                           "exception_chain": cost.exception_chain(error)})
                break
    result.update(completed_at=cost.base.now(), requests=record["requests"])
    cost.write(output, result)
    print(json.dumps({"model_alias": result["model_alias"], "attempts": result["attempts"],
                      "requests": [{k: q.get(k) for k in ["usage", "response_model", "system_fingerprint", "error_type", "status_code"]} for q in record["requests"]]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
