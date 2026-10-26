"""One synthetic long request before continuing never-attempted cases."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "p1.0.8b"))
import run_full_cost as cost


def main():
    output = HERE / "provider-long-context-probe.json"
    assert not output.exists()
    graph = cost.base.load_candidate_graph()
    nodes = [{"node_id": key, **value} for key, value in graph["nodes"].items() if value["stage"] == "anchor"][:8]
    record = {"case_id": "synthetic-long-context", "requests": []}
    cost.base.LOCAL.record = record
    observed = cost.observer(cost.base.transport._bounded_client)
    def bounded(*, timeout):
        api = observed(timeout=180)
        create = api.chat.completions.create
        def with_usage(**request):
            request.update(max_tokens=16000, stream_options={"include_usage": True})
            return create(**request)
        api.chat.completions.create = with_usage
        return api
    quote = "This is a synthetic interface fixture, not a patient. No symptom information is available."
    profile = {"key_narratives": [{"text": quote, "sources": ["synthetic-p1"]}],
               cost.base.ALLOWED_EVIDENCE_REFS_KEY: ["synthetic-p1"]}
    result = {"started_at": cost.base.now(), "synthetic_only": True, "effective_read_timeout_seconds": 180}
    try:
        with patch.object(cost.base.transport, "_bounded_client", bounded):
            answers = cost.base.APICLINICRReasoner(knowledge=cost.base.JsonDiagnosticKnowledgeProvider(), static_knowledge_prefix=True)._evaluate_tree_nodes(profile, nodes)
        result.update(ok=True, judgments=[r.model_dump(mode="json") for r in answers])
    except Exception as error:
        result.update(ok=False, exception_chain=cost.exception_chain(error))
    result.update(completed_at=cost.base.now(), requests=record["requests"])
    cost.write(output, result)
    print(json.dumps({"ok": result["ok"], "request_count": len(record["requests"]),
                      "requests": [{k: q.get(k) for k in ["prompt_characters", "elapsed_seconds", "usage", "error_type", "status_code"]} for q in record["requests"]]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
