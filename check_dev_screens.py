"""P1.0：仅测试 DEV 首例的 screen 前沿，复用生产提示词、分批与校验。"""
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1] / "L1-L5系统"
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=False, encoding="utf-8-sig")

from shared import llm_client as transport
from shared.diagnostic_knowledge import (
    JsonDiagnosticKnowledgeProvider, get_default_knowledge_provider,
)
from layers.layer3.diagnosis.clinicr import CLINICR
from layers.layer3.diagnosis.decision_tree import load_candidate_graph
from layers.layer3.diagnosis.schemas import TreeNodeJudgment
from layers.layer3.llm.client import ALLOWED_EVIDENCE_REFS_KEY, APICLINICRReasoner

RESULT = Path(__file__).with_name("dev-screen-smoke.json")


def main():
    with (ROOT / "artifacts/datasets/mentalbench_split_v2/engine/dev.inputs.jsonl").open(
        encoding="utf-8-sig"
    ) as stream:
        row = json.loads(next(stream))
    profile = row["profile"]
    allowed_refs = CLINICR._known_evidence_refs(profile)
    model_profile = {**profile, ALLOWED_EVIDENCE_REFS_KEY: sorted(allowed_refs)}
    graph = load_candidate_graph()
    node_ids = list(dict.fromkeys(
        node for path in graph["paths"].values() for node in path["screens"]
    ))
    nodes = [{"node_id": node, **graph["nodes"][node]} for node in node_ids]
    reasoner = APICLINICRReasoner(knowledge=JsonDiagnosticKnowledgeProvider())
    result = {
        "captured_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "scope": "DEV first profile, screen frontier only",
        "selection": "first JSONL row; no gold read or label-based selection",
        "case_id": row["case_id"],
        "model": reasoner.model,
        "tree_version": graph["version"],
        "provider": type(reasoner.knowledge).__name__,
        "default_provider": type(get_default_knowledge_provider()).__name__,
        "expected_node_ids": node_ids,
        "expected_batch_sizes": [len(nodes[i:i+8]) for i in range(0, len(nodes), 8)],
        "allowed_evidence_ref_count": len(allowed_refs),
        "sdk_max_retries": 1,
        "structured_output_retry_budget_per_batch": 1,
        "requests": [],
    }
    original_bounded_client = transport._bounded_client

    def observed_client(*, timeout):
        api = original_bounded_client(timeout=timeout)
        original_create = api.chat.completions.create

        def observed_create(**request):
            prompt = request["messages"][0]["content"]
            batch, _ = json.JSONDecoder().raw_decode(prompt.split("当前节点 JSON：", 1)[1])
            batch_ids = [node["node_id"] for node in batch]
            record = {
                "node_ids": batch_ids,
                "is_structured_retry": any(item["node_ids"] == batch_ids for item in result["requests"]),
                "prompt_characters": len(prompt),
                "temperature": request["temperature"],
                "response_format": request.get("response_format"),
                "timeout_seconds": timeout,
                "usage": None,
            }
            result["requests"].append(record)
            started = perf_counter()
            parts = []
            try:
                response_stream = original_create(**request)

                def observed_chunks():
                    try:
                        for chunk in response_stream:
                            record["response_model"] = chunk.model
                            if getattr(chunk, "system_fingerprint", None):
                                record["system_fingerprint"] = chunk.system_fingerprint
                            if getattr(chunk, "usage", None):
                                record["usage"] = chunk.usage.model_dump(mode="json")
                            for choice in chunk.choices:
                                if choice.delta.content:
                                    parts.append(choice.delta.content)
                                if choice.finish_reason:
                                    record["finish_reason"] = choice.finish_reason
                            yield chunk
                        try:
                            payload = json.loads("".join(parts))
                            assert isinstance(payload, dict) and set(payload) == {"nodes"}
                            assert isinstance(payload["nodes"], list)
                            judgments = []
                            for item in payload["nodes"]:
                                assert set(item) == set(TreeNodeJudgment.model_fields)
                                judgment = TreeNodeJudgment.model_validate(item, strict=True)
                                assert judgment.rationale.strip()
                                judgments.append(judgment)
                            definitions = {node["node_id"]: node for node in batch}
                            assert len(judgments) == len(batch)
                            assert {item.node_id for item in judgments} == set(definitions)
                            for item in judgments:
                                assert set(item.evidence_refs) <= allowed_refs
                                assert item.status == "pending" or item.evidence_refs
                                assert set(item.symptom_ids) <= set(definitions[item.node_id].get("symptom_pool", []))
                                assert item.status == "met" or not item.symptom_ids
                            record["raw_strict_json_and_whitelists_pass"] = True
                        except (ValueError, AssertionError, TypeError, KeyError):
                            record["raw_strict_json_and_whitelists_pass"] = False
                    except Exception as error:
                        record["error_type"] = type(error).__name__
                        raise
                    finally:
                        record["elapsed_seconds"] = round(perf_counter() - started, 3)

                return observed_chunks()
            except Exception as error:
                record["error_type"] = type(error).__name__
                record["status_code"] = getattr(error, "status_code", None)
                record["elapsed_seconds"] = round(perf_counter() - started, 3)
                raise

        api.chat.completions.create = observed_create
        return api

    print(json.dumps({"starting": result["scope"], "case_id": row["case_id"],
                      "batch_sizes": result["expected_batch_sizes"]}, ensure_ascii=False), flush=True)
    try:
        with patch.object(transport, "_bounded_client", observed_client):
            answers = reasoner._evaluate_tree_nodes(model_profile, nodes)
        assert len(answers) == len(nodes) and {item.node_id for item in answers} == set(node_ids)
        for node_ids_in_batch in [node_ids[i:i+8] for i in range(0, len(node_ids), 8)]:
            last = [item for item in result["requests"] if item["node_ids"] == node_ids_in_batch][-1]
            assert last.get("raw_strict_json_and_whitelists_pass") is True
            assert last.get("finish_reason") == "stop"
        result["judgments"] = [
            {**item.model_dump(exclude={"rationale"}), "rationale_nonempty": bool(item.rationale.strip())}
            for item in answers
        ]
        result.update(ok=True, screen_interface_gate="go", status_counts=dict(Counter(item.status for item in answers)))
    except Exception as error:
        result.update(ok=False, screen_interface_gate="no_go", error_type=type(error).__name__,
                      status_code=getattr(error, "status_code", None))
    result["structured_retry_count"] = sum(item["is_structured_retry"] for item in result["requests"])
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ["ok", "screen_interface_gate", "structured_retry_count"]}, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
