"""从当前 L3 提取的 JSON、引用和请求配置辅助函数。"""
from __future__ import annotations
import json
import os
import re
from typing import Any, Callable, TypeVar
from openai import APIError

ALLOWED_EVIDENCE_REFS_KEY = "_allowed_evidence_refs"
JSON_OBJECT_RESPONSE_FORMAT = {"type": "json_object"}
StructuredResult = TypeVar("StructuredResult")

class VerifierModelUnavailable(RuntimeError):
    pass

class VerifierAPIError(VerifierModelUnavailable):
    pass


def _is_retryable_structured_output_api_error(error: APIError) -> bool:
    """只重试供应商明确标记的 JSON response_format 生成中止。"""
    message = str(error).casefold()
    return (
        "response_format" in message
        and "json" in message
        and any(marker in message for marker in ("invalid", "incomplete", "abnormal"))
    )


def _request_structured_output_with_one_retry(
    request: Callable[[str], str],
    validate: Callable[[str], StructuredResult],
    prompt: str,
    retry_prompt: str,
) -> StructuredResult:
    """解析失败或供应商中止 JSON 生成时，共享同一个单次重试额度。"""
    try:
        raw = request(prompt)
    except APIError as exc:
        if not _is_retryable_structured_output_api_error(exc):
            raise
        return validate(request(retry_prompt))
    try:
        return validate(raw)
    except ValueError:
        return validate(request(retry_prompt))


def _l3_timeout_seconds() -> float:
    """L3 诊断输出较长，为三个 CLINICR 模型步骤提供独立超时。"""
    raw = (os.getenv("L3_LLM_TIMEOUT_SECONDS") or "60").strip()
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ValueError("L3_LLM_TIMEOUT_SECONDS 必须是数字") from exc
    if not 1 <= timeout <= 300:
        raise ValueError("L3_LLM_TIMEOUT_SECONDS 必须在 1 到 300 秒之间")
    return timeout


def _l3_reasoning_extra(model: str) -> dict[str, Any] | None:
    """Qwen3 保留思考模式，但避免使用耗时过长的默认 xhigh。"""
    if "qwen3" not in model.casefold():
        return None
    effort = (os.getenv("L3_REASONING_EFFORT") or "medium").strip().lower()
    if effort not in {"low", "medium", "xhigh"}:
        raise ValueError("L3_REASONING_EFFORT 必须是 low、medium 或 xhigh")
    return {"enable_thinking": True, "reasoning_effort": effort}


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE
    )
    if fenced:
        text = fenced.group(1).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("模型输出必须是 JSON 对象")
    return value


def _prompt_profile(profile: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """从规则注入的元数据中拆出唯一允许引用列表，不把元数据当临床事实。"""
    raw_refs = profile.get(ALLOWED_EVIDENCE_REFS_KEY, [])
    allowed = (
        [item for item in raw_refs if isinstance(item, str) and item]
        if isinstance(raw_refs, list)
        else []
    )
    clinical_profile = {
        key: value
        for key, value in profile.items()
        if key != ALLOWED_EVIDENCE_REFS_KEY
    }
    return clinical_profile, allowed


def _evidence_catalog(
    clinical_profile: dict[str, Any], allowed_refs: list[str]
) -> list[dict[str, str]]:
    """给模型提供可选择证据的含义，避免它根据画像字段另造引用。"""
    node_by_id = {
        str(node["evidence_id"]): node
        for node in clinical_profile.get("evidence_nodes") or []
        if isinstance(node, dict) and node.get("evidence_id")
    }
    catalog: list[dict[str, str]] = []
    for ref in allowed_refs:
        item = {"ref": ref}
        node = node_by_id.get(ref)
        if node:
            if node.get("target_path"):
                item["target_path"] = str(node["target_path"])
            quote = node.get("quote") or node.get("evidence_quote")
            if quote:
                item["patient_quote"] = str(quote)
        catalog.append(item)
    return catalog


def _value_at_evidence_target(
    clinical_profile: dict[str, Any], target_path: str
) -> Any:
    """读取证据节点所指向的画像值；兼容 L2→L3 的扁平诊断字段。"""
    missing = object()
    parts = target_path.split(".")
    current: Any = clinical_profile
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            current = missing
            break
        current = current[part]
    if current is not missing:
        return current
    if len(parts) == 2 and parts[0] == "diagnostic_context":
        return clinical_profile.get(parts[1], missing)
    return missing


def _canonical_evidence_aliases(
    clinical_profile: dict[str, Any], allowed_refs: list[str]
) -> dict[str, list[str]]:
    """把模型常见的 ``字段=值`` 写法映射回可追溯证据节点。"""
    allowed = set(allowed_refs)
    aliases: dict[str, list[str]] = {}
    for node in clinical_profile.get("evidence_nodes") or []:
        if not isinstance(node, dict):
            continue
        evidence_id = str(node.get("evidence_id") or "")
        target_path = str(node.get("target_path") or "")
        if not evidence_id or evidence_id not in allowed or not target_path:
            continue
        value = _value_at_evidence_target(clinical_profile, target_path)
        if type(value) not in {str, int, float, bool}:
            continue
        serialized_values = {
            str(value),
            json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        }
        field = target_path.rsplit(".", 1)[-1]
        for serialized in serialized_values:
            for alias in (f"{field}={serialized}", f"{target_path}={serialized}"):
                aliases.setdefault(alias, []).append(evidence_id)
    return aliases


def _canonicalize_payload_evidence_refs(
    payload: dict[str, Any], clinical_profile: dict[str, Any], allowed_refs: list[str]
) -> dict[str, Any]:
    """仅将能由真实字段和值唯一解释的别名换成白名单证据 ID。"""
    allowed = set(allowed_refs)
    aliases = _canonical_evidence_aliases(clinical_profile, allowed_refs)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "evidence_refs" and isinstance(child, list):
                    normalized: list[Any] = []
                    for ref in child:
                        replacements = (
                            [ref]
                            if isinstance(ref, str) and ref in allowed
                            else aliases.get(ref, [ref]) if isinstance(ref, str) else [ref]
                        )
                        for replacement in replacements:
                            if replacement not in normalized:
                                normalized.append(replacement)
                    value[key] = normalized
                else:
                    visit(child)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)
    return payload


def _assert_payload_evidence_refs(
    payload: dict[str, Any], allowed_refs: list[str]
) -> None:
    """模型边界预检；未知证据不得被删除、替换或带入下游。"""
    allowed = set(allowed_refs)
    unknown: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "evidence_refs" and isinstance(child, list):
                    unknown.update(
                        ref for ref in child
                        if isinstance(ref, str) and ref not in allowed
                    )
                else:
                    visit(child)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)
    if unknown:
        raise ValueError("模型输出包含证据白名单之外的引用")
