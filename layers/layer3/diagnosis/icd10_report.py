"""Turn a trusted ICD-10 tree result and a clinical review into an L4 report."""

from datetime import datetime, timezone

from layers.layer3.instructions.instruction_generator import InstructionGenerator
from layers.layer4.schemas.l3_report import (
    Diagnosis, DiagnosticReport, ICD10ClinicalReview, ICD10CriterionEvidence,
    PrimaryDiagnosis, SecondaryDiagnosis,
)
from shared.report_integrity import ICD10_STANDARD


def reviewed_diagnosis(result: dict, review: ICD10ClinicalReview) -> Diagnosis:
    if (result.get("standard") != ICD10_STANDARD or result.get("status") != "completed"
            or result.get("pending") or result.get("session_id") != review.tree_session_id
            or not result.get("catalog_id")):
        raise ValueError("复核必须对应本会话已完成的 ICD-10 树运行")

    codes = [review.primary_code, *review.secondary_codes]
    if len(codes) != len(set(codes)):
        raise ValueError("主诊断和共病编码必须唯一")
    documents = result["documents"]
    messages = {m["message_id"]: m for m in result["profile"]["messages"]}
    diagnoses = []
    for code in codes:
        candidates = [c for c in result["candidates"] if c["code"] == code and c["answer"] == "yes"]
        if len(candidates) != 1 or candidates[0]["output_status"] != "criterion_supported_candidate":
            raise ValueError(f"{code} 不是本次 ICD-10 树唯一有依据的候选")
        candidate = candidates[0]
        criteria = [ICD10CriterionEvidence.model_validate(node) for node in candidate["path"]]
        if not criteria or len({node.node_id for node in criteria}) != len(criteria):
            raise ValueError(f"{code} 缺少可追溯的 ICD-10 路径")
        has_patient_evidence = False
        for node in criteria:
            if node.source.get("document_id") not in documents or not node.source.get("pdf_page"):
                raise ValueError(f"{node.node_id} 缺少 WHO 原文出处")
            if node.answer in {"yes", "no"} and (
                not node.original_clause.strip() or (node.calculation is None and not node.evidence)
            ):
                raise ValueError(f"{node.node_id} 缺少逐条证据")
            for quote in node.evidence:
                message = messages.get(quote.message_id)
                if (not message or quote.source_role != message["role"]
                        or not quote.quote.strip() or quote.quote not in message["text"]):
                    raise ValueError(f"{node.node_id} 引文无法对应当前会话材料")
                has_patient_evidence = True
        if not has_patient_evidence:
            raise ValueError(f"{code} 缺少患者或临床材料证据")
        diagnoses.append({
            "disorder": code, "label": candidate["label"], "name": candidate["label"],
            "display_name": candidate["label"], "knowledge_source": ICD10_STANDARD,
            "confidence": candidate.get("confidence"), "criteria_evidence": criteria,
        })
    return Diagnosis(primary=PrimaryDiagnosis(**diagnoses[0]),
                     secondary=[SecondaryDiagnosis(**row) for row in diagnoses[1:]])


def build_icd10_report(result: dict, review: ICD10ClinicalReview, profile: dict) -> dict:
    diagnosis = reviewed_diagnosis(result, review)
    report = DiagnosticReport(
        diagnostic_standard=ICD10_STANDARD,
        icd10_catalog_id=result["catalog_id"],
        icd10_review=review,
        timestamp=datetime.now(timezone.utc).isoformat(),
        candidate_generation_method="icd10_tree",
        diagnosis=diagnosis,
        candidate_confidence=diagnosis.primary.confidence,
        verifier_decision="clinician_confirmed",
        verifier_routing="pass",
        recommended_therapy=review.therapy,
        severity_level=review.severity_level,
    ).model_dump(mode="json")
    report["instructions"] = InstructionGenerator().generate(report, profile).model_dump(mode="json")
    return report
