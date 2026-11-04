"""生成患者可见、已脱敏的画像确认卡。"""

from layers.layer2.schemas import ClinicalIntake
from layers.layer2.symptom_vocabulary import ICD10_SYMPTOM_LABELS
from orchestration.schemas import PatientConfirmation


THRESHOLD_LABELS = {
    "minimal": "很轻",
    "mild": "轻度",
    "moderate": "中度",
    "moderately_severe": "中重度",
    "severe": "重度",
    "positive_screen": "筛查提示需要进一步关注",
    "subthreshold": "未达到常用筛查阈值",
    "no_clinically_significant_insomnia": "未见明显失眠困扰",
    "extreme": "非常严重",
    "subclinical": "未达到常用筛查阈值",
}


def build_patient_confirmation(
    intake: ClinicalIntake,
) -> PatientConfirmation:
    symptoms = [
        ICD10_SYMPTOM_LABELS.get(item.symptom, "一项已描述的困扰")
        for item in intake.key_symptoms_extracted
    ]
    profile_summary: list[str] = []
    if symptoms:
        profile_summary.append("你提到的主要困扰包括：" + "、".join(symptoms) + "。")
    context = intake.diagnostic_context
    if context.symptom_duration_days is not None:
        profile_summary.append(f"这些情况大约持续了{context.symptom_duration_days}天。")
    if context.functional_impairment is True:
        profile_summary.append("这些情况已经对工作、学习、人际或日常生活造成影响。")
    elif context.functional_impairment is False:
        profile_summary.append("你目前认为这些情况尚未明显影响日常功能。")
    if intake.patient_goals:
        profile_summary.append("你希望优先改善：" + "、".join(intake.patient_goals) + "。")

    scale_summary = []
    for name, result in intake.scale_results.items():
        if result.status != "completed" or result.score is None:
            continue
        label = THRESHOLD_LABELS.get(result.threshold or "", result.threshold or "已完成")
        scale_summary.append(f"{name}筛查得分为{result.score}，结果为{label}。")

    goals = "、".join(intake.patient_goals) or "你最关心的困扰"
    return PatientConfirmation(
        title="请确认我们整理的信息",
        profile_summary=profile_summary,
        scale_summary=scale_summary,
        initial_impression="当前只整理了你提供的信息，尚未进行诊断判断。",
        disclaimer=(
            "量表和系统整理只用于辅助了解情况，不能单独作为医学诊断，"
            "也不替代专业人员的面对面评估。"
        ),
        next_step=(
            f"如果信息基本准确，这份画像将交给下一层围绕{goals}进一步分析。"
        ),
    )
