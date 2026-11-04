import hashlib
import json
from typing import Any


ICD10_STANDARD = "WHO ICD-10"
ICD10_OUTPUT_POLICY = "L3_OUTPUT_ICD10_V1"

REQUIRED_L3_CHECKS = {
    "V3-CANDIDATE-CONFIDENCE",
    "V3-PRIMARY-NOT-RULED-OUT",
    "V3-CRITERIA-EVIDENCE",
    "V3-THERAPY-MATCH",
    "V3-PLUGIN-ROUTING",
    "V3-THERAPY-ADJUSTMENT",
    "V3-CONTRAINDICATION-OVERRIDE",
    "V3-TREATMENT-INSTRUCTIONS",
    "V3-ICD10-CLINICAL-REVIEW",
}


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _compact(value: Any) -> Any:
    """统一模型解析前后的空默认值，同时保留 0 和 False。"""
    if isinstance(value, dict):
        compacted = {key: _compact(item) for key, item in value.items()}
        return {
            key: item
            for key, item in compacted.items()
            if item is not None and item != "" and item != [] and item != {}
        }
    if isinstance(value, list):
        return [_compact(item) for item in value]
    return value


def l3_report_fingerprint(report: Any) -> str:
    """对影响 L3→L4 决策的字段生成稳定指纹，不包含验证凭证自身。"""
    source = _plain(report)
    payload = {
        "diagnosis": source.get("diagnosis"),
        "candidate_confidence": source.get("candidate_confidence", 0.0),
        "verifier_score": source.get("verifier_score", 0.0),
        "verifier_routing": source.get("verifier_routing", ""),
        "recommended_therapy": source.get("recommended_therapy"),
        "instructions": source.get("instructions"),
        "diagnostic_standard": source.get("diagnostic_standard"),
        "icd10_catalog_id": source.get("icd10_catalog_id"),
        "icd10_review": source.get("icd10_review"),
        "severity_level": source.get("severity_level"),
        "verifier_decision": source.get("verifier_decision"),
    }
    encoded = json.dumps(
        _compact(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
