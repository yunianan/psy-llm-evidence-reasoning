"""当前临时 LLM 评分器的独立实验副本；标准知识由调用方注入。"""
from __future__ import annotations
import json
import os
from typing import Any, Callable
from openai import APIError
from .contracts import EliminativeAssessment, RewardModelAssessment
from ._transport import ask_llm_stream, get_model
from ._support import (
    VerifierAPIError, VerifierModelUnavailable, JSON_OBJECT_RESPONSE_FORMAT,
    _prompt_profile, _evidence_catalog, _parse_json_object,
    _canonicalize_payload_evidence_refs, _assert_payload_evidence_refs,
    _request_structured_output_with_one_retry, _l3_timeout_seconds, _l3_reasoning_extra,
)


class APIRewardModelVerifier:
    """训练奖励模型上线前的 API 适配器；接口可直接替换为本地奖励模型。"""

    def __init__(
        self,
        model: str | None = None,
        *,
        knowledge_provider: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self.model = model or os.getenv("L3_VERIFIER_MODEL") or get_model()
        self.knowledge_provider = knowledge_provider

    def score_reasoning(
        self,
        profile: dict[str, Any],
        candidate_without_confidence: dict[str, Any],
        assessment: EliminativeAssessment,
        differential_context: list[dict[str, Any]],
    ) -> RewardModelAssessment:
        if self.knowledge_provider is None:
            raise VerifierModelUnavailable("独立 API 评分需要调用方提供当前候选的标准知识快照")
        knowledge = self.knowledge_provider(candidate_without_confidence["disorder"])
        if not isinstance(knowledge, dict) or not knowledge:
            raise VerifierModelUnavailable("当前候选的标准知识快照不能为空")
        clinical_profile, allowed_refs = _prompt_profile(profile)
        evidence_catalog = _evidence_catalog(clinical_profile, allowed_refs)
        prompt = f"""你是 L3 的独立 Reward Model Verifier 临时API实现。
只评估排名第一诊断的推理过程质量，不重新生成诊断，也不知道候选生成器给出的置信度。
DSM-5 原始候选知识：{json.dumps(knowledge, ensure_ascii=False)}
请评估：证据是否充分、逻辑是否连贯、是否考虑鉴别诊断、是否存在跳跃性推理。
score 范围0至1。四个 dimensions 必须各出现一次：evidence_sufficiency、logic_coherence、differential_coverage、no_reasoning_leaps。
若需要补充信息，missing_dimensions 必须给出具体字段、目的和供L2使用的追问；低分时 fail_reasons 必须解释原因。
所有 evidence_refs 只能从“允许的 evidence_refs”列表中逐字复制；没有对应证据时返回空列表，禁止创造引用。

只输出严格 JSON：
{{
  "score": 0.87,
  "dimensions": [
    {{"dimension":"evidence_sufficiency","score":0.9,"reason":"理由","evidence_refs":["证据标识"]}},
    {{"dimension":"logic_coherence","score":0.9,"reason":"理由","evidence_refs":[]}},
    {{"dimension":"differential_coverage","score":0.8,"reason":"理由","evidence_refs":[]}},
    {{"dimension":"no_reasoning_leaps","score":0.9,"reason":"理由","evidence_refs":[]}}
  ],
  "fail_reasons": [],
  "missing_dimensions": []
}}

允许的 evidence_refs（只能复制 ref；target_path 和 patient_quote 仅帮助理解）：{json.dumps(evidence_catalog, ensure_ascii=False)}
患者画像：{json.dumps(clinical_profile, ensure_ascii=False)}
待终验候选（不含候选置信度）：{json.dumps(candidate_without_confidence, ensure_ascii=False)}
逐项排除结果：{assessment.model_dump_json()}
其他候选的排除上下文：{json.dumps(differential_context, ensure_ascii=False)}
"""
        try:
            def request(step_prompt: str) -> str:
                return ask_llm_stream(
                    step_prompt,
                    model=self.model,
                    temperature=0,
                    timeout=_l3_timeout_seconds(),
                    extra_body=_l3_reasoning_extra(self.model),
                    response_format=JSON_OBJECT_RESPONSE_FORMAT,
                )

            def validate(raw: str) -> RewardModelAssessment:
                payload = _canonicalize_payload_evidence_refs(
                    _parse_json_object(raw), clinical_profile, allowed_refs
                )
                _assert_payload_evidence_refs(payload, allowed_refs)
                return RewardModelAssessment.model_validate(payload)

            repair_prompt = (
                prompt
                + "\n上一次回答不是符合上述结构的合法结果，或包含证据白名单外的引用。"
                "请重新独立生成完整 JSON；missing_dimensions 中必须使用字段名 "
                "field、purpose、suggested_question，禁止使用 suggested_question_for_L2。"
                "所有 evidence_refs 必须从允许列表逐字复制。"
                "只输出一个合法 JSON 对象，不要解释、不要 Markdown、不要省略字段。"
            )
            return _request_structured_output_with_one_retry(
                request, validate, prompt, repair_prompt
            )
        except APIError as exc:
            raise VerifierAPIError(
                f"CLINICR Step 3 API错误：{type(exc).__name__}"
            ) from exc
        except Exception as exc:
            raise VerifierModelUnavailable(
                f"CLINICR Step 3模型输出不合法：{type(exc).__name__}"
            ) from exc
