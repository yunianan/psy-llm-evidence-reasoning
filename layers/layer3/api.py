from datetime import datetime, timezone
from typing import Any, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .diagnosis.clinicr import CLINICR
from .diagnosis.information_report import input_information_report
from .llm import CLINICRModelUnavailable
from .therapy.therapy_selector import TherapySelector
from .instructions.instruction_generator import InstructionGenerator
from .output_validator import L3OutputValidator
from shared.l2_l3_contract import (
    as_missing_dimension,
    missing_required_profile_evidence,
    missing_required_profile_fields,
)

from shared.disorder_catalog import (SUPPORTED_DISORDERS, TREATMENT_KEY_BY_NODE,
                                     grouped_catalog, disorder_metadata)

app = FastAPI(title="Diagnosis Layer API")


class MotivationInput(BaseModel):
    importance: int | None = Field(default=None, ge=0, le=10)
    confidence: int | None = Field(default=None, ge=0, le=10)
    readiness: int | None = Field(default=None, ge=0, le=10)


class DiagnoseRequest(BaseModel):
    PHQ9: float | None = Field(default=None, ge=0, le=27)
    GAD7: float | None = Field(default=None, ge=0, le=21)
    MDQ: float | None = Field(default=None, ge=0, le=13)
    PCL5: float | None = Field(default=None, ge=0, le=80)
    ISI: float | None = Field(default=None, ge=0, le=28)
    Y_BOCS: float | None = Field(default=None, ge=0, le=40)
    ADNM: float | None = Field(default=None, ge=0)
    RRS: float | None = Field(default=None, ge=0)
    AAQ_II: float | None = Field(default=None, ge=0)
    PHQ15: float | None = Field(default=None, ge=0, le=30)
    relapse_count: int | None = Field(default=None, ge=0)
    rhythm_disturbance: bool | None = None
    exposure_avoidant: bool | None = None
    ritual_behaviors: bool | None = None
    insomnia_chief_complaint: bool | None = None
    chronic_months: int | None = Field(default=None, ge=0)
    interpersonal_conflicts: int | None = Field(default=None, ge=0)
    interpersonal_impairment: bool | None = None
    single_focus: bool | None = None
    psychotic_symptoms: bool | None = None
    substance_active: bool | None = None
    severe_dissociation: bool | None = None
    risk_level: int = 0
    sensitive_topics: List[str] = []
    l1_flags: List[str] = []
    PHQ9_items: dict[str, float] = Field(default_factory=dict)
    key_symptoms_extracted: list[dict[str, Any]] = Field(default_factory=list)
    symptom_duration_days: int | None = None
    functional_impairment: bool | None = None
    mania_or_hypomania_history: bool | None = None
    substance_or_medical_cause: bool | None = None
    differential_diagnoses_assessed: list[str] | None = None
    evidence_conflicts: list[dict[str, Any]] | None = None
    diagnostic_details: dict[str, Any] = Field(default_factory=dict)
    emotions: list[dict[str, Any]] = Field(default_factory=list)
    cognitive_patterns: list[dict[str, Any]] = Field(default_factory=list)
    key_narratives: list[dict[str, Any]] = Field(default_factory=list)
    functional_impairments: list[dict[str, Any]] = Field(default_factory=list)
    patient_goals: list[str] = Field(default_factory=list)
    motivation: MotivationInput = Field(default_factory=MotivationInput)
    evidence_nodes: list[dict[str, Any]] = Field(default_factory=list)


class DiagnoseResponse(BaseModel):
    layer_id: Literal["L3"] = "L3"
    timestamp: str
    diagnostic_standard: Literal["DSM-5"] = "DSM-5"
    supported_disorders: list[str] = Field(default_factory=lambda: list(SUPPORTED_DISORDERS))
    diagnostic_catalog: list[dict[str, Any]] = Field(default_factory=grouped_catalog)
    considered_disorders: list[str] = Field(default_factory=list)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    checked: list[dict[str, Any]] = Field(default_factory=list)
    candidate_generation_method: str = "external_reasoner"
    tree_trace: list[dict[str, Any]] = Field(default_factory=list)
    candidate_coverage: dict[str, str] = Field(default_factory=dict)
    candidate_missing_dimensions: list[dict[str, str]] = Field(default_factory=list)
    diagnostic_readiness: dict[str, Any] = Field(default_factory=dict, description="仅描述必需判据/鉴别路径的满足、已知阻塞或信息不足，不替代最终诊断门")
    information_collection_plan: dict[str, Any] = Field(default_factory=dict, description="Top-1/Top-3补采优先队列；保留其余候选，不自动追问或改变判据")
    diagnosis: dict
    verifier_score: float
    verifier_decision: str
    candidate_confidence: float
    verifier_routing: str
    recommended_therapy: dict
    severity_level: int
    clinical_notes_for_L4: str
    summary: str
    structured_labels: dict
    evidence_refs: list
    confidence: float
    flags: list
    instruction_to_next: str
    instructions: dict
    missing_dimensions: list[dict[str, str]] = Field(default_factory=list)
    provisional_diagnosis: dict | None = None
    verifier_checks: list[dict[str, Any]] = Field(default_factory=list)
    verifier_fail_reasons: list[str] = Field(default_factory=list)
    l3_validation: dict[str, Any]


def _catalog_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {"diagnostic_standard": "DSM-5", "supported_disorders": list(SUPPORTED_DISORDERS),
            "diagnostic_catalog": grouped_catalog(),
            "considered_disorders": result.get("considered_disorders", []),
            "candidates": result.get("candidates", []), "checked": result.get("checked", []),
            "candidate_generation_method": result.get("candidate_generation_method", "external_reasoner"),
            "tree_trace": result.get("tree_trace", []),
            "candidate_coverage": result.get("candidate_coverage", {}),
            "candidate_missing_dimensions": result.get("candidate_missing_dimensions", []),
            "diagnostic_readiness": result.get("diagnostic_readiness", {}),
            "information_collection_plan": result.get("information_collection_plan", {})}


@app.get("/diagnostic-catalog")
def diagnostic_catalog():
    return _catalog_payload({})


def build_blocked_report(clinicr_result):
    missing = clinicr_result.get("missing_dimensions", [])
    routing = clinicr_result["verifier_routing"]
    readiness_status = clinicr_result.get("diagnostic_readiness", {}).get("status")
    if routing == "return_to_l2":
        action = "collect_missing_dimensions"
        instruction = "返回L2补充诊断所需信息，按补采队列安排顺序；未知不等于不满足"
    elif routing == "manual_review":
        action = "manual_review"
        instruction = "转人工复核，不进入L4自动治疗"
    else:
        action = "stop_diagnostic_route"
        instruction = "终验不支持当前诊断，停止进入L4"
    if routing == "manual_review":
        wording = "诊断候选保留，治疗方案需人工复核，不进入自动治疗"
    elif readiness_status == "known_blocker":
        wording = "候选存在已知不满足条件，未形成最终诊断，不进入治疗对话"
        if routing == "return_to_l2":
            instruction = "返回L2复核已知不满足条件，并补充仍缺的信息"
    elif readiness_status == "supported":
        wording = "判据支持不等于终验通过，未放行的诊断不得进入治疗对话"
    else:
        wording = "信息不足，不得形成诊断结论或进入治疗对话"
    return {
        "layer_id": "L3",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        **_catalog_payload(clinicr_result),
        "diagnosis": clinicr_result["diagnosis"],
        "provisional_diagnosis": clinicr_result.get("provisional_diagnosis"),
        "verifier_score": clinicr_result["verifier_score"],
        "verifier_decision": clinicr_result["verifier_decision"],
        "candidate_confidence": (
            clinicr_result.get("provisional_diagnosis") or {}
        ).get("confidence", 0.0),
        "verifier_routing": routing,
        "verifier_checks": clinicr_result.get("verifier_checks", []),
        "verifier_fail_reasons": clinicr_result.get("verifier_fail_reasons", []),
        "recommended_therapy": {
            "primary": "",
            "addon": [],
            "plugin_triggers": [],
            "strategy_adjustments": {},
            "contraindications": [],
        },
        "severity_level": clinicr_result["severity_level"],
        "clinical_notes_for_L4": "",
        "summary": clinicr_result["summary"],
        "structured_labels": {
            "therapy_routing": {
                "primary": "",
                "addon": [],
                "plugin_triggers": [],
                "strategy_adjustments": {},
                "l4_substage_priority": [],
            },
            "wording_constraint": wording,
            "risk_monitoring": [],
        },
        "evidence_refs": clinicr_result["evidence_refs"],
        "confidence": 0.0,
        "flags": clinicr_result["flags"],
        "instruction_to_next": instruction,
        "instructions": {
            "L2": {
                "action": action,
                "missing_dimensions": missing,
            }
        },
        "missing_dimensions": missing,
    }


def build_input_contract_report(profile: dict[str, Any]) -> dict[str, Any] | None:
    """在调用 CLINICR 前执行与 L2 相同的最低输入契约。"""

    requirements = [
        *missing_required_profile_fields(profile),
        *missing_required_profile_evidence(profile),
    ]
    unique = {item.field: item for item in requirements}
    if not unique:
        return None

    missing = [as_missing_dimension(item) for item in unique.values()]
    result = {
        "diagnosis": {
            "primary": None,
            "secondary": [],
            "under_observation": [],
            "ruled_out": [],
        },
        "provisional_diagnosis": None,
        "verifier_score": 0.0,
        "verifier_decision": "rejected",
        "verifier_routing": "return_to_l2",
        "verifier_checks": [],
        "verifier_fail_reasons": [
            "L2→L3共享输入契约未满足，未调用CLINICR模型"
        ],
        "missing_dimensions": missing,
        "severity_level": 1,
        "summary": "患者画像缺少诊断所需字段或对应证据，返回L2补充信息。",
        "evidence_refs": [],
        "flags": ["insufficient_diagnostic_evidence"],
    }
    result.update(input_information_report(missing))
    report = build_blocked_report(result)
    report["l3_validation"] = L3OutputValidator().validate(report)
    return report


THERAPY_SUBSTAGE_PRIORITY = {
    "CBT": ["behavioral_activation", "cognitive_restructuring", "sleep_hygiene"],
    "CBT+BA": ["behavioral_activation", "cognitive_restructuring", "sleep_hygiene"],
    "CBT-TF": ["trauma_processing", "cognitive_restructuring"],
    "CBT-I": ["sleep_hygiene", "stimulus_control", "sleep_restriction"],
    "CBT+IPSRT": ["behavioral_activation", "interpersonal_rhythm"],
    "CBT/SFBT": ["behavioral_activation", "solution_focused"],
}

PLUGIN_SUBSTAGE = {
    "MBCT": "mindfulness",
    "ACT": "cognitive_defusion",
    "ERP": "exposure_response_prevention",
    "EMDR": "trauma_processing",
    "CBT-I": "sleep_hygiene",
    "IPSRT": "interpersonal_rhythm",
    "IPT": "interpersonal_work",
    "SFBT": "solution_focused",
}

GUIDELINE_SOURCE = {
    "CBT+BA": "APA_2019; NICE_NG222",
    "CBT": "APA_2019; NICE_NG222",
    "CBT-TF": "ISTSS_2019; NICE_NG116",
    "CBT-I": "AASM_2021; ACP_Qaseem_2016",
    "CBT+IPSRT": "ISBD_2020; NICE_NG185",
    "CBT/SFBT": "APA_2019; NICE_NG222",
}


def build_report(profile, clinicr_result, therapy, safety_context=None):
    diagnosis = clinicr_result["diagnosis"]
    primary = diagnosis.get("primary")
    therapy_type = therapy.get("basic_therapy", "CBT")

    priority = list(THERAPY_SUBSTAGE_PRIORITY.get(therapy_type, ["behavioral_activation", "cognitive_restructuring"]))
    for plugin in therapy.get("plugins", []):
        extra = PLUGIN_SUBSTAGE.get(plugin)
        if extra and extra not in priority:
            priority.append(extra)

    safety = safety_context or {}
    sensitive_topics = safety.get("sensitive_topics", [])
    safety_flags = safety.get("l1_flags", []) or safety.get("flags", [])

    contraindications = list(therapy.get("contraindications", []))
    plugin_triggers = list(therapy.get("plugin_triggers", []))
    strategy_adjustments = dict(therapy.get("strategy_adjustments", {}))
    if "trauma" in sensitive_topics:
        contraindications.append("避免深度创伤暴露（per L1敏感话题标记）")

    risk_monitoring = []
    if "monitor_suicidal_ideation" in safety_flags or "passive_si_monitor" in safety_flags:
        risk_monitoring.append("passive_si_to_active_si_escalation")
    if strategy_adjustments.get("safety_monitoring") == "enhanced":
        risk_monitoring.append("enhanced_monitoring_frequency")

    evidence_refs = list(clinicr_result["evidence_refs"])
    evidence_refs.append({
        "type": "guideline",
        "therapy": f"{therapy_type}_for_{primary['disorder']}" if primary else therapy_type,
        "source": GUIDELINE_SOURCE.get(therapy_type, "APA_2019"),
    })
    trigger_by_plugin = {
        item.get("plugin"): item for item in plugin_triggers if item.get("plugin")
    }
    for plugin in therapy.get("plugins", []):
        trigger = trigger_by_plugin.get(plugin, {})
        evidence_refs.append({
            "type": "guideline",
            "therapy": f"{plugin}_plugin",
            "source": "L3治疗插件确定性规则",
            "threshold": "; ".join(trigger.get("conditions", [])),
        })

    focus_notes = []
    for stage in priority:
        focus_notes.append(f"({priority.index(stage) + 1}){stage}")

    flags = list(clinicr_result.get("flags", []))
    for f in safety_flags:
        if f not in flags:
            flags.append(f)

    if primary is None:
        raise HTTPException(status_code=400, detail="Cannot determine diagnosis")

    adjustment_text = "；".join(strategy_adjustments.get("instructions", []))
    clinical_notes = (
        f"L4治疗焦点按序推进：{'→'.join(priority)}。"
        f"主诊断{disorder_metadata(primary['disorder'])['display_name']}（置信度{primary['confidence']}，"
        f"严重度{primary.get('severity', 'unknown')}）。"
        f"方案微调：{adjustment_text or '无'}。"
    )
    if risk_monitoring:
        clinical_notes += "需按L1安全通行证执行风险监测，风险升级时立即返回L1。"

    instruction_to_next = (
        f"L4对话应聚焦{'→'.join(priority[:2])}。"
        f"执行方案微调：{adjustment_text or '无'}。"
        "L4-L2需温和反馈诊断（'初步印象'措辞），征求患者确认。"
    )

    report = {
        "layer_id": "L3",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        **_catalog_payload(clinicr_result),
        "diagnosis": diagnosis,
        "verifier_score": clinicr_result["verifier_score"],
        "verifier_decision": clinicr_result["verifier_decision"],
        "candidate_confidence": primary["confidence"] if primary else 0.0,
        "verifier_routing": clinicr_result["verifier_routing"],
        "verifier_checks": clinicr_result.get("verifier_checks", []),
        "verifier_fail_reasons": clinicr_result.get("verifier_fail_reasons", []),
        "recommended_therapy": {
            "primary": therapy_type,
            "addon": therapy.get("plugins", []),
            "plugin_triggers": plugin_triggers,
            "strategy_adjustments": strategy_adjustments,
            "contraindications": contraindications,
        },
        "severity_level": clinicr_result["severity_level"],
        "clinical_notes_for_L4": clinical_notes,
        "summary": clinicr_result["summary"],
        "structured_labels": {
            "therapy_routing": {
                "primary": therapy_type,
                "addon": therapy.get("plugins", []),
                "plugin_triggers": plugin_triggers,
                "strategy_adjustments": strategy_adjustments,
                "l4_substage_priority": priority,
            },
            "wording_constraint": "使用'初步印象'措辞，禁止'确诊'",
            "risk_monitoring": risk_monitoring,
        },
        "evidence_refs": evidence_refs,
        "confidence": clinicr_result["confidence"],
        "flags": flags,
        "instruction_to_next": instruction_to_next,
    }
    report["instructions"] = InstructionGenerator().generate(report, profile).model_dump(mode="json")
    return report


def diagnose(profile, safety_context=None, clinicr: CLINICR | None = None):
    contract_report = build_input_contract_report(profile)
    if contract_report is not None:
        return contract_report

    engine = clinicr or CLINICR()
    result = engine.run(profile, safety_context=safety_context)

    if result["verifier_routing"] not in {"pass", "conditional"}:
        report = build_blocked_report(result)
        report["l3_validation"] = L3OutputValidator().validate(report)
        return report

    primary = result["diagnosis"]["primary"]
    if primary is None:
        raise HTTPException(status_code=400, detail="Cannot determine diagnosis")

    if primary["disorder"] not in TREATMENT_KEY_BY_NODE:
        result = {**result, "verifier_routing": "manual_review",
                  "flags": list(dict.fromkeys(result.get("flags", []) + ["treatment_mapping_unavailable"])),
                  "summary": result["summary"] + "该诊断尚无项目治疗映射，保留诊断结果并转人工治疗方案复核。"}
        report = build_blocked_report(result)
        report["l3_validation"] = L3OutputValidator().validate(report)
        return report

    selector = TherapySelector()
    confirmed_diagnoses = [primary["disorder"]] + [
        item["disorder"] for item in result["diagnosis"].get("secondary", [])
    ]
    therapy = selector.run(
        primary["disorder"],
        profile,
        confirmed_diagnoses=confirmed_diagnoses,
        verifier_decision=result["verifier_decision"],
    )

    report = build_report(profile, result, therapy, safety_context)

    report["l3_validation"] = L3OutputValidator().validate(report)
    if not report["l3_validation"]["passed"]:
        report["flags"] = list(dict.fromkeys(report["flags"] + ["l3_output_validation_failed"]))
        report["instruction_to_next"] = "L3输出校验失败，留在L3修正诊断或疗法，不进入L4"
        report["instructions"] = {
            "L3": {
                "action": "correct_output",
                "failed_checks": [
                    item for item in report["l3_validation"]["checks"]
                    if item["status"] == "fail"
                ],
            }
        }
        return report

    return report


@app.post("/diagnose", response_model=DiagnoseResponse)
def diagnose_endpoint(request: DiagnoseRequest):
    profile = {
        "PHQ9": request.PHQ9,
        "GAD7": request.GAD7,
        "MDQ": request.MDQ,
        "PCL5": request.PCL5,
        "ISI": request.ISI,
        "Y_BOCS": request.Y_BOCS,
        "ADNM": request.ADNM,
        "RRS": request.RRS,
        "AAQ_II": request.AAQ_II,
        "PHQ15": request.PHQ15,
        "relapse_count": request.relapse_count,
        "rhythm_disturbance": request.rhythm_disturbance,
        "exposure_avoidant": request.exposure_avoidant,
        "ritual_behaviors": request.ritual_behaviors,
        "insomnia_chief_complaint": request.insomnia_chief_complaint,
        "chronic_months": request.chronic_months,
        "interpersonal_conflicts": request.interpersonal_conflicts,
        "interpersonal_impairment": request.interpersonal_impairment,
        "single_focus": request.single_focus,
        "psychotic_symptoms": request.psychotic_symptoms,
        "substance_active": request.substance_active,
        "severe_dissociation": request.severe_dissociation,
        "PHQ9_items": request.PHQ9_items,
        "key_symptoms_extracted": request.key_symptoms_extracted,
        "symptom_duration_days": request.symptom_duration_days,
        "functional_impairment": request.functional_impairment,
        "mania_or_hypomania_history": request.mania_or_hypomania_history,
        "substance_or_medical_cause": request.substance_or_medical_cause,
        "differential_diagnoses_assessed": request.differential_diagnoses_assessed,
        "evidence_conflicts": request.evidence_conflicts,
        "diagnostic_details": request.diagnostic_details,
        "emotions": request.emotions,
        "cognitive_patterns": request.cognitive_patterns,
        "key_narratives": request.key_narratives,
        "functional_impairments": request.functional_impairments,
        "patient_goals": request.patient_goals,
        "motivation": request.motivation.model_dump(mode="json"),
        "evidence_nodes": request.evidence_nodes,
    }
    safety_context = {
        "risk_level": request.risk_level,
        "sensitive_topics": request.sensitive_topics,
        "l1_flags": request.l1_flags,
    }
    try:
        return DiagnoseResponse(**diagnose(profile, safety_context))
    except CLINICRModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health():
    return {"status": "ok", "service": "Diagnosis Layer"}


@app.get("/")
def root():
    return {"message": "Diagnosis Layer API is running", "endpoints": ["/diagnose", "/health"]}
