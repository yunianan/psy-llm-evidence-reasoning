"""单条纯软件测试节点的严格 JSON 检查；不使用任何患者或病例数据。"""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError


ROOT = Path(__file__).resolve().parents[1] / "L1-L5系统"
sys.path.insert(0, str(ROOT))
from layers.layer3.diagnosis.schemas import TreeNodeJudgment
from shared.llm_client import get_api_key, get_base_url, get_model


NODE_ID = "phase0.test.blue_signal"
EVIDENCE_ID = "TEST_ONLY_EVIDENCE_001"
SYMPTOM_ID = "TEST_ONLY_BLUE_SIGNAL"
FIXTURE = {
    "input": {"signal": "blue"},
    "evidence": [{"ref": EVIDENCE_ID, "quote": "signal=blue"}],
    "node": {
        "node_id": NODE_ID,
        "question": "输入是否明确给出 signal=blue？",
        "symptom_pool": [SYMPTOM_ID],
        "mapping": {"blue": SYMPTOM_ID},
    },
}


def main():
    load_dotenv(ROOT / ".env", override=False, encoding="utf-8-sig")
    model = os.getenv("L3_REASONER_MODEL") or get_model()
    schema = {
        "type": "object", "additionalProperties": False,
        "required": ["nodes"],
        "properties": {"nodes": {
            "type": "array", "minItems": 1, "maxItems": 1,
            "items": TreeNodeJudgment.model_json_schema(),
        }},
    }
    prompt = (
        "这是纯软件接口测试，不含患者、疾病或临床数据。只判定下面的一条测试节点。\n"
        "只返回一个符合 schema 的 JSON 对象，不要代码围栏、前后说明或额外字段。\n"
        "根对象仅有 nodes 数组。节点须显式包含 schema properties 的全部六个字段。\n"
        "status 为 met/not_met/pending：明确匹配条件为 met；明确反驳为 not_met；信息不足为 pending。\n"
        "node_id 逐字复制；evidence_refs 仅可使用测试 evidence 的 ref。\n"
        "symptom_ids 仅按测试 mapping 映射到 symptom_pool；非 met 时为空。\n"
        "rationale 只写一句简短、可核查的依据，不输出隐藏思维过程；confidence 为 0 到 1 的数字。\n"
        f"schema：{json.dumps(schema, ensure_ascii=False)}\n"
        f"测试夹具：{json.dumps(FIXTURE, ensure_ascii=False)}"
    )
    result = {
        "optional_test": True, "affects_p0_1_p0_2": False,
        "captured_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "model": model, "fixture": FIXTURE,
        "parameters": {"temperature": 0, "stream": True, "max_tokens": 2048,
                       "response_format": {"type": "json_object"},
                       "thinking": "provider_default", "timeout_seconds": 60},
        "json_repair_attempts": 0, "sdk_max_retries": 0,
    }
    try:
        with OpenAI(api_key=get_api_key(), base_url=get_base_url(), timeout=60, max_retries=0) as client:
            stream = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=0, stream=True, max_tokens=2048,
                response_format={"type": "json_object"},
            )
            parts = []
            for chunk in stream:
                if getattr(chunk, "system_fingerprint", None):
                    result["system_fingerprint"] = chunk.system_fingerprint
                result["response_model"] = chunk.model
                for choice in chunk.choices:
                    if choice.delta.content:
                        parts.append(choice.delta.content)
                    if choice.finish_reason:
                        result["finish_reason"] = choice.finish_reason
        raw = "".join(parts)
        result["raw_content"] = raw  # 仅保存纯软件夹具输出，不保存 reasoning_content。
        assert result.get("finish_reason") == "stop", "Output was not completed normally"
        payload = json.loads(raw)
        assert isinstance(payload, dict) and set(payload) == {"nodes"}, "Envelope fields differ"
        assert isinstance(payload["nodes"], list) and len(payload["nodes"]) == 1, "Expected one node"
        node = payload["nodes"][0]
        assert isinstance(node, dict) and set(node) == set(TreeNodeJudgment.model_fields), "Node fields differ"
        parsed = TreeNodeJudgment.model_validate(node, strict=True)
        assert parsed.node_id == NODE_ID and parsed.status == "met", "Wrong node or status"
        assert parsed.evidence_refs == [EVIDENCE_ID], "Invalid test evidence reference"
        assert parsed.symptom_ids == [SYMPTOM_ID], "Invalid test symptom mapping"
        assert parsed.rationale.strip(), "Empty rationale"
        result.update(ok=True, parsed=parsed.model_dump(mode="json"), first_response_passed=True)
    except OpenAIError as error:
        result.update(ok=False, error=type(error).__name__, status_code=getattr(error, "status_code", None))
    except (ValueError, AssertionError) as error:
        result.update(ok=False, error=type(error).__name__, detail=str(error)[:500], first_response_passed=False)
    baseline = json.loads(Path(__file__).with_name("model-check.json").read_text(encoding="utf-8"))
    result["matches_preflight_fingerprint"] = (
        result.get("system_fingerprint") == baseline["smoke_test"]["system_fingerprint"]
    )
    Path(__file__).with_name("node-json-check.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
