import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from layers.layer3.output_validator import L3OutputValidator
from layers.layer4.config import Settings
from layers.layer4.llm.mock import MockBackend
from layers.layer4.schemas.l3_report import DiagnosticReport, RecommendedTherapy
from layers.layer4.service import L4Service, SessionRequest
from layers.layer4.validation import L3ToL4Gate, L3ToL4ValidationError


from orchestration.icd10_followup import Session, build_catalog
from tests.test_icd10_orchestrator import PENDING, ScriptedJudge


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _raw_report() -> dict:
    return json.loads((EXAMPLES / "sample_diagnostic_report.json").read_text(encoding="utf-8"))


def _report() -> DiagnosticReport:
    return DiagnosticReport.model_validate(_raw_report())


@pytest.fixture
def tree_result():
    raw = _raw_report()
    tree = Session(build_catalog())
    tree.state["id"] = raw["icd10_review"]["tree_session_id"]
    tree.add_message("我怕蜘蛛，近几个月也经常睡不好。", message_id="m1")
    for index, (_, reply) in enumerate(PENDING.values(), start=2):
        tree.add_message(reply, message_id=f"m{index}")
    judge = ScriptedJudge()
    return tree.advance(lambda node, messages: judge(node, messages, clinical_profile={}))


def _failed_l3_rules(raw: dict, tree_result: dict) -> set[str]:
    receipt = L3OutputValidator().validate(raw, icd10_result=tree_result)
    return {
        item["rule_id"] for item in receipt["checks"] if item["status"] == "fail"
    }


def test_l3_output_validator_accepts_valid_report(tree_result):
    receipt = L3OutputValidator().validate(_raw_report(), icd10_result=tree_result)

    assert receipt["passed"] is True
    assert receipt["route"] == "allow_l4"


def test_l3_rejects_low_candidate_confidence_after_review(tree_result):
    raw = _raw_report()
    raw["candidate_confidence"] = 0.49
    raw["verifier_routing"] = "pass"

    assert "V3-CANDIDATE-CONFIDENCE" in _failed_l3_rules(raw, tree_result)

    raw["candidate_confidence"] = 0.0
    assert "V3-CANDIDATE-CONFIDENCE" in _failed_l3_rules(raw, tree_result)


def test_l3_rejects_primary_diagnosis_that_is_also_ruled_out(tree_result):
    raw = _raw_report()
    raw["diagnosis"]["ruled_out"].append({
        "disorder": raw["diagnosis"]["primary"]["disorder"],
        "reason": "冲突测试",
        "evidence_ref": "test",
    })

    assert "V3-PRIMARY-NOT-RULED-OUT" in _failed_l3_rules(raw, tree_result)


def test_l3_rejects_therapy_that_does_not_match_review(tree_result):
    raw = _raw_report()
    raw["recommended_therapy"]["primary"] = "EMDR"

    assert "V3-THERAPY-MATCH" in _failed_l3_rules(raw, tree_result)


def test_l3_rejects_met_criterion_without_evidence(tree_result):
    raw = _raw_report()
    raw["diagnosis"]["primary"]["criteria_evidence"][0]["evidence"] = []

    assert "V3-CRITERIA-EVIDENCE" in _failed_l3_rules(raw, tree_result)


def test_l4_guard_accepts_valid_l3_receipt():
    result = L3ToL4Gate().validate(_report())

    assert result["passed"] is True
    assert result["route"] == "allow_l4"


def test_l4_rejects_retired_dsm_report_and_receipt():
    raw = _raw_report()
    raw["diagnostic_standard"] = "DSM-5"
    with pytest.raises(ValidationError, match="diagnostic_standard"):
        DiagnosticReport.model_validate(raw)

    report = _report()
    report.l3_validation.policy_version = "L3_OUTPUT_V1"
    result = L3ToL4Gate().validate(report)
    assert "V3-ICD10-VALIDATION-POLICY" in {
        item["rule_id"] for item in result["violations"]
    }


def test_l4_guard_does_not_recompute_l3_rules_but_detects_report_tampering():
    report = _report().model_copy(update={"candidate_confidence": 0.1})

    result = L3ToL4Gate().validate(report)
    rule_ids = {item["rule_id"] for item in result["violations"]}

    assert "V3-PAYLOAD-INTEGRITY" in rule_ids
    assert "V3-CANDIDATE-CONFIDENCE" not in rule_ids


def test_l4_guard_rejects_missing_l3_validation_receipt():
    report = _report().model_copy(update={"l3_validation": None})

    result = L3ToL4Gate().validate(report)

    assert "V3-VALIDATION-RECEIPT" in {
        item["rule_id"] for item in result["violations"]
    }


def test_l4_service_cannot_bypass_receipt_integrity_guard():
    report = _report().model_copy(update={
        "recommended_therapy": RecommendedTherapy(primary="EMDR")
    })
    service = L4Service(Settings(_env_file=None), backend=MockBackend())

    with pytest.raises(L3ToL4ValidationError, match="V3-PAYLOAD-INTEGRITY"):
        service.create_session(SessionRequest(
            report=report,
        ))
