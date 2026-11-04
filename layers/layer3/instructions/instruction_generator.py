"""规则编码会话级治疗指令，不重新诊断、不重新匹配疗法。"""

from copy import deepcopy

from shared.schemas.treatment_instructions import (
    BoundaryInstruction, DiagnosticAnchor, RoleInstruction, TreatmentInstructions,
)

ABSOLUTE_BANS = ["开药", "确诊", "诊断新疾病", "预测未来", "替代医生", "谈论其他患者", "保证治愈"]
SCALE_FIELDS = {
    "PHQ9": "PHQ-9", "GAD7": "GAD-7", "PCL5": "PCL-5",
    "Y_BOCS": "Y-BOCS", "ISI": "ISI", "MDQ": "MDQ", "RRS": "RRS",
    "AAQ_II": "AAQ-II", "PHQ15": "PHQ-15", "ADNM": "ADNM",
}
SUPPORT_TECHNIQUES = {
    "reflection": "开放式提问与共情反映",
    "feedback": "温和反馈与共同确认",
    "mi_evoking": "MI 唤出改变动机",
    "stabilization": "支持性稳定化",
    "review": "回顾进展与巩固",
}
BASE_TECHNIQUES = {
    "CBT": {"thought_identification": "识别自动化思维", "cognitive_restructuring": "苏格拉底式提问与认知重构", "behavioral_activation": "行为激活", "behavioral_experiment": "行为实验"},
    "BA": {"behavioral_activation": "行为激活"},
    "CBT-TF": {"thought_identification": "识别创伤相关思维", "trauma_processing": "聚焦创伤的 CBT"},
    "CBT-I": {"sleep_hygiene": "睡眠卫生", "stimulus_control": "刺激控制", "sleep_restriction": "睡眠限制"},
    "IPSRT": {"interpersonal_rhythm": "人际与社会节律稳定"},
    "SFBT": {"solution_focused": "焦点解决提问"},
}
PLUGIN_TECHNIQUES = {
    "MBCT": {"mindfulness": "正念练习"},
    "ACT": {"cognitive_defusion": "认知解离"},
    "ERP": {"exposure_response_prevention": "暴露与反应阻止"},
    "EMDR": {"emdr": "EMDR"},
    "IPT": {"interpersonal_work": "人际关系工作"},
    **{key: BASE_TECHNIQUES[key] for key in ("CBT-I", "IPSRT", "SFBT")},
}
CHALLENGING = {"behavioral_experiment", "trauma_processing", "exposure_response_prevention", "emdr"}


def allowed_techniques(anchor: DiagnosticAnchor) -> dict[str, str]:
    """仅把已经确定的疗法及禁忌转换为可执行技术白名单。"""
    allowed = dict(SUPPORT_TECHNIQUES)
    for part in anchor.therapy_type.replace("/", "+").split("+"):
        allowed.update(BASE_TECHNIQUES.get(part, {}))
    for plugin in anchor.plugins:
        allowed.update(PLUGIN_TECHNIQUES.get(plugin, {}))
    adjustment = anchor.strategy_adjustments
    if adjustment.get("challenge_interventions") == "deferred":
        for code in CHALLENGING:
            allowed.pop(code, None)
    if adjustment.get("motivation_band") == "low":
        allowed = dict(SUPPORT_TECHNIQUES)
    if adjustment.get("care_priority", "psychotherapy") != "psychotherapy":
        allowed = dict(SUPPORT_TECHNIQUES)
    if (any("暴露" in text for text in anchor.contraindications)
            or adjustment.get("exposure_policy") == "prohibited_until_stabilized"):
        for code in ("trauma_processing", "exposure_response_prevention", "emdr"):
            allowed.pop(code, None)
    return allowed


class InstructionGenerator:
    def generate(self, report: dict, profile: dict) -> TreatmentInstructions:
        diagnosis = deepcopy(report["diagnosis"])
        primary = diagnosis.get("primary")
        if primary is None:
            raise ValueError("没有主诊断，不能生成治疗指令")
        therapy = report["recommended_therapy"]
        cognitive = deepcopy(profile.get("cognitive_patterns") or [])
        anchor = DiagnosticAnchor(
            diagnostic_standard=report["diagnostic_standard"],
            primary_label=primary["label"],
            primary=primary["disorder"],
            comorbidities=[item["disorder"] for item in diagnosis.get("secondary", [])],
            excluded=[item["disorder"] for item in diagnosis.get("ruled_out", [])],
            # 保留原始语义分类，不把全部认知模式擅自升级为核心信念。
            core_beliefs=[item["value"] for item in cognitive if item.get("kind") == "core_belief"],
            cognitive_patterns=cognitive,
            triggering_situations=deepcopy([item for item in profile.get("key_narratives", []) if item.get("kind") == "triggering_situation"]),
            key_narratives=deepcopy(profile.get("key_narratives") or []),
            patient_goals=list(profile.get("patient_goals") or []),
            diagnosis_details=diagnosis,
            scale_scores={name: profile[key] for key, name in SCALE_FIELDS.items() if profile.get(key) is not None},
            plugins=list(therapy.get("addon", [])),
            plugin_triggers=deepcopy(therapy.get("plugin_triggers", [])),
            strategy_adjustments=deepcopy(therapy.get("strategy_adjustments", {})),
            contraindications=list(therapy.get("contraindications", [])),
            confidence=report["verifier_score"],
            candidate_confidence=report["candidate_confidence"],
            verifier_score=report["verifier_score"],
            verifier_decision=report["verifier_decision"],
            therapy_type=therapy["primary"],
        )
        anchor.allowed_techniques = allowed_techniques(anchor)
        return TreatmentInstructions(
            L1=RoleInstruction(
                identity="你是一名受过 CBT 训练的心理治疗师",
                style="温暖、共情、协作、结构化；平滑衔接，不播报阶段名称",
                boundaries=list(ABSOLUTE_BANS),
            ),
            L2=anchor,
            L4=BoundaryInstruction(
                absolute_bans=list(ABSOLUTE_BANS),
                safety_trigger="表达主动自伤、自杀或伤人意图时立即停止普通治疗并触发 L1 安全协议；不得自行解除 L1 限制",
            ),
        )


def validate_session_instructions(report: dict) -> None:
    bundle = TreatmentInstructions.model_validate(report.get("instructions"))
    anchor = bundle.L2
    if anchor.diagnosis_details != report["diagnosis"]:
        raise ValueError("诊断锚点与诊断报告不一致")
    diagnosis = report["diagnosis"]
    if anchor.diagnostic_standard != report.get("diagnostic_standard"):
        raise ValueError("诊断锚点与报告采用的诊断标准不一致")
    if anchor.primary_label != diagnosis["primary"]["label"]:
        raise ValueError("ICD-10 诊断锚点与报告名称不一致")
    if (anchor.primary != diagnosis["primary"]["disorder"]
            or anchor.comorbidities != [item["disorder"] for item in diagnosis.get("secondary", [])]
            or anchor.excluded != [item["disorder"] for item in diagnosis.get("ruled_out", [])]):
        raise ValueError("诊断锚点摘要与诊断详情不一致")
    therapy = report["recommended_therapy"]
    expected = (therapy["primary"], therapy.get("addon", []), therapy.get("plugin_triggers", []), therapy.get("strategy_adjustments", {}), therapy.get("contraindications", []))
    actual = (anchor.therapy_type, anchor.plugins, anchor.plugin_triggers, anchor.strategy_adjustments, anchor.contraindications)
    if actual != expected or anchor.allowed_techniques != allowed_techniques(anchor):
        raise ValueError("指令与已批准疗法或技术白名单不一致")
    if (anchor.confidence != report["verifier_score"]
            or anchor.candidate_confidence != report["candidate_confidence"]
            or anchor.verifier_score != report["verifier_score"]
            or anchor.verifier_decision != report["verifier_decision"]):
        raise ValueError("指令置信度或终验结论不一致")
    if not set(ABSOLUTE_BANS) <= set(bundle.L4.absolute_bans) or not bundle.L4.safety_trigger:
        raise ValueError("防越界指令缺失")
    if (not bundle.L1.identity.strip() or not bundle.L1.style.strip()
            or not set(ABSOLUTE_BANS) <= set(bundle.L1.boundaries)):
        raise ValueError("角色定义或角色边界缺失")
