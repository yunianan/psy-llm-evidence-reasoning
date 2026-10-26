import argparse
import json
from unittest.mock import patch

from common import HERE, cost, node_client, POLICY
from layers.layer3.diagnosis.decision_tree import load_candidate_graph
from tools.l3_engineering_runtime import ReferenceBudget, ReliableStreaming, now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    folder = HERE / "runs/thinking-probe-v1"
    provider = cost.base.JsonDiagnosticKnowledgeProvider()
    model = cost.base.APICLINICRReasoner(knowledge=provider).model
    if not args.execute:
        print(json.dumps({"model": model, "service_url": cost.base.transport.get_base_url(), "synthetic_only": True,
                          "calls": 2, "thinking_modes": [True, False], "policy": POLICY}))
        return
    assert not folder.exists()
    folder.mkdir(parents=True)
    profile = {"key_narratives": [{"value": "合成接口夹具：连续三个月每周四晚半夜醒后很难再睡，白天疲倦。未描述其他症状。", "sources": ["synthetic-p1"]}],
               node_client.ALLOWED_EVIDENCE_REFS_KEY: ["synthetic-p1"]}
    definition = {"node_id": "anchor.Insomnia", **load_candidate_graph()["nodes"]["anchor.Insomnia"]}
    budget = ReferenceBudget(batch_limit=5, case_limit=2)
    result = {"started_at": now(), "synthetic_only": True, "model": model, "trials": []}
    with (folder / "requests.jsonl").open("w", encoding="utf-8") as journal:
        transport = ReliableStreaming(api_key=cost.base.transport.get_api_key(), base_url=cost.base.transport.get_base_url(),
                                      journal=journal, budget=budget)
        with patch.object(node_client, "ask_llm_stream", transport.ask):
            for enabled in [True, False]:
                records = []
                transport.bind("synthetic-thinking-probe", records, thinking=enabled, arm=str(enabled))
                transport.tags("anchor", [definition["node_id"]])
                trial = {"enable_thinking": enabled}
                try:
                    answers = cost.base.APICLINICRReasoner(knowledge=provider)._evaluate_tree_nodes(profile, [definition])
                    trial.update(ok=True, judgments=[a.model_dump(mode="json") for a in answers])
                except Exception as error:
                    trial.update(ok=False, exception_chain=cost.exception_chain(error))
                trial["requests"] = records
                result["trials"].append(trial)
    on, off = result["trials"]
    def reasoning(trial):
        values = [(r.get("usage_values") or {}).get("reasoning") for r in trial["requests"]]
        return None if not values or any(v is None for v in values) else sum(values)
    result.update(completed_at=now(), explicit_on_reasoning=reasoning(on), explicit_off_reasoning=reasoning(off),
                  thinking_disable_confirmed=on["ok"] and off["ok"] and reasoning(on) is not None and reasoning(on) > 0 and reasoning(off) == 0,
                  budget=budget.snapshot())
    cost.write(folder / "summary.json", result)
    print(json.dumps({k: result[k] for k in ["explicit_on_reasoning", "explicit_off_reasoning", "thinking_disable_confirmed", "budget"]}))


if __name__ == "__main__":
    main()
