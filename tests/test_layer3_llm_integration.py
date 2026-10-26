import json

import pytest

from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider

from layers.layer3.diagnosis.clinicr import CLINICR
from layers.layer3.llm import client
from tests.tree_test_support import node_payload


def test_explicit_provider_clinicr_only_calls_step1_and_verifier(monkeypatch):
    monkeypatch.delenv("L3_LLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("L3_REASONING_EFFORT", raising=False)
    monkeypatch.setenv("LLM_MODEL", "qwen3.8-flash")
    profile = {"PHQ9": 16, "PHQ9_items": {"item2": 3}}
    calls = []

    def controlled_api(prompt, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        if "DSM-5 决策树节点判定器" in prompt:
            response = node_payload(
                prompt, ["D013", "D011", "D009"], "PHQ9_item2=3", complete_labels=["D013"]
            )
        elif "Reward Model Verifier" in prompt:
            response = {
                "score": .87,
                "dimensions": [{"dimension": name, "score": .87, "reason": "受控终验",
                                "evidence_refs": ["PHQ9_item2=3"]}
                               for name in ["evidence_sufficiency", "logic_coherence",
                                            "differential_coverage", "no_reasoning_leaps"]],
                "fail_reasons": [], "missing_dimensions": [],
            }
        else:
            pytest.fail("A duplicate per-candidate diagnostic request was made")
        return json.dumps(response, ensure_ascii=False)

    monkeypatch.setattr(client, "ask_llm_stream", controlled_api)
    result = CLINICR(knowledge=JsonDiagnosticKnowledgeProvider()).run(profile)
    assert result["diagnosis"]["primary"]["disorder"] == "D013"
    assert result["verifier_score"] == .87
    assert not any("ELIMINATIVE" in call["prompt"] for call in calls)
    assert sum("Reward Model Verifier" in call["prompt"] for call in calls) == 1
    assert not hasattr(client.APICLINICRReasoner, "evaluate_candidate")
    assert all(row["source"] == "step1_tree_records" for row in result["checked"])
    assert result["candidate_generation_method"] == "dsm5_decision_tree"
    assert result["tree_trace"]
    assert all(call["temperature"] == 0 for call in calls)
    assert all(call["timeout"] == 60.0 for call in calls)
    assert all(call["extra_body"] == {"enable_thinking": True, "reasoning_effort": "medium"}
               for call in calls)
    assert all(call["response_format"] == {"type": "json_object"} for call in calls)
    assert all("允许的 evidence_refs（只能复制 ref" in call["prompt"] for call in calls)
    assert "PHQ9=16" in calls[0]["prompt"]
    assert "_allowed_evidence_refs" not in calls[0]["prompt"]
    assert '"confidence"' not in calls[-1]["prompt"]


def test_step1_maps_exact_scalar_field_alias_to_traceable_evidence_node(monkeypatch):
    profile = {
        "symptom_duration_days": 45,
        "evidence_nodes": [{
            "evidence_id": "ev-duration",
            "target_path": "diagnostic_context.symptom_duration_days",
            "quote": "这一个多月",
        }],
        client.ALLOWED_EVIDENCE_REFS_KEY: ["ev-duration"],
    }
    prompts = []

    def controlled_api(prompt, **kwargs):
        prompts.append(prompt)
        return json.dumps({"nodes": [{"node_id": "criterion.D013.depressive_episode/min_duration",
                          "status": "met", "rationale": "45天病程",
                          "evidence_refs": ["symptom_duration_days=45"],
                          "symptom_ids": [], "confidence": 0.7}]}, ensure_ascii=False)

    monkeypatch.setattr("layers.layer3.llm.client.ask_llm_stream", controlled_api)

    result = client.APICLINICRReasoner(
        model="deepseek-chat", knowledge=JsonDiagnosticKnowledgeProvider(),
    )._evaluate_tree_nodes(profile, [{
        "node_id": "criterion.D013.depressive_episode/min_duration", "stage": "criterion", "disorders": ["D013"],
        "symptom_pool": [], "question": "是否达到来源规定病程？",
    }])

    assert all(item.evidence_refs == ["ev-duration"] for item in result)
    assert '"ref": "ev-duration"' in prompts[0]
    assert '"target_path": "diagnostic_context.symptom_duration_days"' in prompts[0]
    assert '"patient_quote": "这一个多月"' in prompts[0]


def test_l3_timeout_can_be_overridden(monkeypatch):
    monkeypatch.setenv("L3_LLM_TIMEOUT_SECONDS", "90")
    assert client._l3_timeout_seconds() == 90.0


@pytest.mark.parametrize("value", ["invalid", "0", "301"])
def test_invalid_l3_timeout_is_rejected(monkeypatch, value):
    monkeypatch.setenv("L3_LLM_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError, match="L3_LLM_TIMEOUT_SECONDS"):
        client._l3_timeout_seconds()


def test_non_qwen_model_does_not_receive_qwen_reasoning_options():
    assert client._l3_reasoning_extra("deepseek-chat") is None


@pytest.mark.parametrize("effort", ["low", "medium", "xhigh"])
def test_qwen_reasoning_effort_can_be_configured(monkeypatch, effort):
    monkeypatch.setenv("L3_REASONING_EFFORT", effort)
    assert client._l3_reasoning_extra("qwen3.8-flash") == {
        "enable_thinking": True,
        "reasoning_effort": effort,
    }


def test_invalid_qwen_reasoning_effort_is_rejected(monkeypatch):
    monkeypatch.setenv("L3_REASONING_EFFORT", "high")
    with pytest.raises(ValueError, match="L3_REASONING_EFFORT"):
        client._l3_reasoning_extra("qwen3.8-flash")


def _valid_assessment_payload():
    return {
        "disorder": "D013",
        "criteria_detail": [{
            "code": "A1",
            "name": "情绪低落",
            "status": "met",
            "rationale": "患者原话支持",
            "evidence_refs": ["ev-mood"],
        }],
        "conclusion": "watch",
        "conclusion_reason": "仍需补充其他标准",
        "missing_dimensions": [{
            "field": "mdd.additional_symptoms",
            "purpose": "确认其他症状",
            "suggested_question": "还出现了哪些变化？",
        }],
    }


def _valid_verifier_payload():
    return {
        "score": 0.72,
        "dimensions": [
            {
                "dimension": dimension,
                "score": 0.72,
                "reason": "推理过程基本完整",
                "evidence_refs": ["ev-mood"] if dimension == "evidence_sufficiency" else [],
            }
            for dimension in (
                "evidence_sufficiency",
                "logic_coherence",
                "differential_coverage",
                "no_reasoning_leaps",
            )
        ],
        "fail_reasons": [],
        "missing_dimensions": [{
            "field": "appetite_change",
            "purpose": "补充症状证据",
            "suggested_question": "最近食欲有什么变化？",
        }],
    }


def _score_with_test_inputs(model):
    profile = {
        "evidence_nodes": [{
            "evidence_id": "ev-mood",
            "target_path": "symptoms.depressed_mood",
            "quote": "最近一直情绪低落",
        }],
        client.ALLOWED_EVIDENCE_REFS_KEY: ["ev-mood"],
    }
    assessment = client.EliminativeAssessment.model_validate(_valid_assessment_payload())
    return model.score_reasoning(
        profile,
        {"disorder": "D013", "evidence_refs": ["ev-mood"]},
        assessment,
        [{"disorder": "D011", "conclusion": "watch"}],
    )


def test_step3_retries_once_when_first_payload_uses_wrong_field(monkeypatch):
    invalid = _valid_verifier_payload()
    invalid["missing_dimensions"][0]["suggested_question_for_L2"] = (
        invalid["missing_dimensions"][0].pop("suggested_question")
    )
    responses = iter([
        json.dumps(invalid, ensure_ascii=False),
        json.dumps(_valid_verifier_payload(), ensure_ascii=False),
    ])
    calls = []

    def controlled_api(prompt, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        return next(responses)

    monkeypatch.setattr(client, "ask_llm_stream", controlled_api)

    result = _score_with_test_inputs(client.APIRewardModelVerifier(
        "deepseek-chat", knowledge=JsonDiagnosticKnowledgeProvider(),
    ))

    assert result.score == 0.72
    assert len(calls) == 2
    assert all(call["response_format"] == {"type": "json_object"} for call in calls)
    assert "禁止使用 suggested_question_for_L2" not in calls[0]["prompt"]
    assert "禁止使用 suggested_question_for_L2" in calls[1]["prompt"]


def test_step3_fails_closed_after_one_invalid_schema_retry(monkeypatch):
    invalid = _valid_verifier_payload()
    invalid["missing_dimensions"][0]["suggested_question_for_L2"] = (
        invalid["missing_dimensions"][0].pop("suggested_question")
    )
    calls = []

    def controlled_api(prompt, **kwargs):
        calls.append(prompt)
        return json.dumps(invalid, ensure_ascii=False)

    monkeypatch.setattr(client, "ask_llm_stream", controlled_api)

    with pytest.raises(client.CLINICRModelUnavailable, match="ValidationError"):
        _score_with_test_inputs(client.APIRewardModelVerifier(
            "deepseek-chat", knowledge=JsonDiagnosticKnowledgeProvider(),
        ))

    assert len(calls) == 2


def test_step3_retries_unknown_evidence_and_keeps_whitelist_enforced(monkeypatch):
    invalid = _valid_verifier_payload()
    invalid["dimensions"][0]["evidence_refs"] = ["invented-evidence-id"]
    responses = iter([
        json.dumps(invalid, ensure_ascii=False),
        json.dumps(_valid_verifier_payload(), ensure_ascii=False),
    ])
    calls = []

    def controlled_api(prompt, **kwargs):
        calls.append(prompt)
        return next(responses)

    monkeypatch.setattr(client, "ask_llm_stream", controlled_api)

    result = _score_with_test_inputs(client.APIRewardModelVerifier(
        "deepseek-chat", knowledge=JsonDiagnosticKnowledgeProvider(),
    ))

    assert result.dimensions[0].evidence_refs == ["ev-mood"]
    assert len(calls) == 2
    assert "证据白名单外" in calls[1]


def test_step3_retries_provider_structured_json_api_error_once(monkeypatch):
    class FakeAPIError(Exception):
        pass

    calls = []

    def controlled_api(prompt, **kwargs):
        calls.append(prompt)
        if len(calls) == 1:
            raise FakeAPIError(
                "JSON response for response_format was incomplete and invalid"
            )
        return json.dumps(_valid_verifier_payload(), ensure_ascii=False)

    monkeypatch.setattr(client, "APIError", FakeAPIError)
    monkeypatch.setattr(client, "ask_llm_stream", controlled_api)

    result = _score_with_test_inputs(client.APIRewardModelVerifier(
        "deepseek-chat", knowledge=JsonDiagnosticKnowledgeProvider(),
    ))

    assert result.score == 0.72
    assert len(calls) == 2
