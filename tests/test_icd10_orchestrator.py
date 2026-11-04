import pytest

from layers.layer3.output_validator import L3OutputValidator
from layers.layer4.schemas.l3_report import DiagnosticReport, ICD10ClinicalReview
from layers.layer4.validation import L3ToL4Gate, L3ToL4ValidationError
from main import create_pipeline
from orchestration import icd10_followup
from tests.test_top_level_orchestrator import _complete_l2, _orchestrator, _safe_passport


PENDING = {
    "specific.fear": ("具体是什么物体让你害怕？", "我看到蜘蛛就很害怕。"),
    "specific.distress": ("这份恐惧给你带来多大困扰？", "我因此很痛苦，不敢进储物间。"),
}


class ScriptedJudge:
    def __init__(self):
        self.calls = []
        self.profiles = []

    def __call__(self, node, messages, *, clinical_profile):
        self.profiles.append(clinical_profile)
        key = node["id"]
        self.calls.append(key)
        question, target = PENDING.get(key, ("", None))
        if target:
            match = next((m for m in messages if m["text"] == target), None)
            if match is None:
                return {"answer": "unknown", "rationale": "fixture", "follow_up": question, "evidence": [], "confidence": None}
        else:
            match = messages[0]
        answer = "no" if key.startswith("entry.") and key != "entry.specific" else "yes"
        return {"answer": answer, "rationale": "fixture", "follow_up": "",
                "confidence": .63 if key == "specific.distress" else .81,
                "evidence": [{"message_id": match["message_id"], "quote": match["text"]}]}


def start_session(monkeypatch, tmp_path):
    judge = ScriptedJudge()
    monkeypatch.setattr(icd10_followup, "APIJudge", lambda **config: judge)
    monkeypatch.setattr("orchestration.orchestrator.safety_detect", _safe_passport)
    orch = _orchestrator(tmp_path)
    sid = "icd10-e2e"
    orch.create_session(sid)
    assert _complete_l2(orch, sid).route == "AWAITING_PATIENT_CONFIRMATION"
    return orch, sid, judge


def test_terminal_entry_defaults_to_icd10():
    pipeline = create_pipeline()
    assert pipeline.icd10_tree_factory == pipeline._default_icd10_tree


def test_icd10_followups_reuse_original_message_ids_and_route_candidate_to_review(monkeypatch, tmp_path):
    orch, sid, judge = start_session(monkeypatch, tmp_path)
    first = orch.confirm_patient_profile(sid, "continue")
    assert first.route == "RETURN_TO_L2" and first.reply == PENDING["specific.fear"][0]
    assert len(judge.calls[:14]) == len(set(judge.calls[:14])) == 14
    assert all(key.startswith("entry.") for key in judge.calls[:14])
    intake = judge.profiles[0]
    assert intake["patient_confirmation"] == "confirmed"
    assert intake["information_sufficiency"] == "sufficient"
    assert intake["key_symptoms_extracted"] and intake["evidence_nodes"]
    assert intake["diagnostic_context"]["functional_impairment"]
    assert intake["scale_results"]["PHQ-9"]["items"]
    assert orch.get_session_snapshot(sid).l3_raw["model_result"]["standard"] == "WHO ICD-10"

    second = orch.handle_message(sid, PENDING["specific.fear"][1])
    assert second.route == "RETURN_TO_L2" and second.reply == PENDING["specific.distress"][0]
    assert any(node["quote"] == PENDING["specific.fear"][1] for node in judge.profiles[-1]["evidence_nodes"])
    final = orch.handle_message(sid, PENDING["specific.distress"][1])
    assert final.route == "MANUAL_REVIEW"

    state = orch.l2.get_session_snapshot(sid)
    snapshot = orch.get_session_snapshot(sid)
    assert snapshot.diagnostic_report is None
    result = snapshot.l3_raw["model_result"]
    assert result["status"] == "completed"
    assert result["standard"] == "WHO ICD-10"
    assert result["comparison"]["status"] == "not_implemented"
    assert [candidate["code"] for candidate in result["candidates"]] == ["F40.2"]
    assert result["candidates"][0]["confidence"] == .63
    path = {node["node_id"]: node for node in result["candidates"][0]["path"]}
    for node_id, (_, text) in PENDING.items():
        task = next(t for t in state.follow_up_tasks if t.node_id == node_id)
        assert task.status == "answered" and task.answer_evidence_refs
        evidence = next(e for e in state.evidence_nodes if e.evidence_id in task.answer_evidence_refs)
        assert path[node_id]["evidence"][0]["quote"] == evidence.quote == text
        assert path[node_id]["evidence"][0]["message_id"] == evidence.message_id
        assert path[node_id]["source"]["document_id"] == "WHO_DCR10_1993"


def test_unresolved_icd10_answer_is_not_repeated_or_counted_negative(monkeypatch, tmp_path):
    orch, sid, _ = start_session(monkeypatch, tmp_path)
    assert orch.confirm_patient_profile(sid, "continue").route == "RETURN_TO_L2"
    assert orch.handle_message(sid, "我记不清了。").route == "MANUAL_REVIEW"
    result = orch.get_session_snapshot(sid).l3_raw["model_result"]
    assert result["status"] == "unresolved" and result["pending"]["node_id"] == "specific.fear"
    assert not result["candidates"]
    task = next(t for t in orch.l2.get_session_snapshot(sid).follow_up_tasks if t.node_id == "specific.fear")
    assert task.status == "unresolved" and task.answer_evidence_refs


def test_l1_crisis_stops_before_icd10_receives_followup(monkeypatch, tmp_path):
    orch, sid, judge = start_session(monkeypatch, tmp_path)
    assert orch.confirm_patient_profile(sid, "continue").route == "RETURN_TO_L2"
    previous = len(judge.calls)
    assert orch.handle_message(sid, "立即危机").route == "STOPPED"
    assert len(judge.calls) == previous
    assert not any(m.get("text") == "立即危机" for m in orch.l2.get_session_snapshot(sid).messages)



def review_for(tree_id, **updates):
    return ICD10ClinicalReview.model_validate({
        "tree_session_id": tree_id,
        "reviewer_id": "fixture-reviewer",
        "primary_code": "F40.2",
        "rationale": "受控工程测试复核",
        "severity_level": 1,
        "therapy": {
            "primary": "CBT",
            "strategy_adjustments": {"severity_band": "mild", "instructions": ["核对患者希望改善的目标。"]},
        },
        **updates,
    })


def test_icd10_review_issues_validated_report_and_opens_l4(monkeypatch, tmp_path):
    orch, sid, _ = start_session(monkeypatch, tmp_path)
    orch.confirm_patient_profile(sid, "continue")
    for _, text in PENDING.values():
        result = orch.handle_message(sid, text)
    assert result.route == "MANUAL_REVIEW"
    raw_result = orch.get_session_snapshot(sid).l3_raw["model_result"]

    with pytest.raises(ValueError, match="对应本会话"):
        orch.submit_icd10_review(sid, review_for("stale-tree-id"))
    with pytest.raises(ValueError, match="有依据的候选"):
        orch.submit_icd10_review(sid, review_for(raw_result["session_id"], primary_code="D013"))

    assert orch.submit_icd10_review(sid, review_for(raw_result["session_id"])).route == "L4"
    report = orch.get_session_snapshot(sid).diagnostic_report
    assert report.diagnostic_standard == "WHO ICD-10"
    assert report.diagnosis.primary.disorder == "F40.2"
    assert report.candidate_confidence == report.diagnosis.primary.confidence == .63
    assert report.verifier_score is None
    assert report.l3_validation.policy_version == "L3_OUTPUT_ICD10_V1"
    assert {"V3-ICD10-CLINICAL-REVIEW", "V3-CANDIDATE-CONFIDENCE", "V3-CRITERIA-EVIDENCE"} <= {
        check.rule_id for check in report.l3_validation.checks if check.status == "pass"
    }
    assert any(node.calculation for node in report.diagnosis.primary.criteria_evidence)
    assert L3ToL4Gate().validate(report)["passed"]
    assert orch.l4.store.get(sid).anchor.primary_label == report.diagnosis.primary.label
    assert orch.handle_message(sid, "我想先谈谈工作中遇到的困难。").route == "L4"

    tampered = report.model_dump(mode="json")
    tampered["diagnosis"]["primary"]["criteria_evidence"][0]["evidence"][0]["quote"] = "未记录的引文"
    assert not L3ToL4Gate().validate(DiagnosticReport.model_validate(tampered))["passed"]
    receipt = L3OutputValidator().validate(tampered, icd10_result=raw_result)
    assert any(c["rule_id"] == "V3-CRITERIA-EVIDENCE" and c["status"] == "fail" for c in receipt["checks"])


def test_icd10_release_rejects_low_support_and_unusable_review_plan(monkeypatch, tmp_path):
    orch, sid, _ = start_session(monkeypatch, tmp_path)
    orch.confirm_patient_profile(sid, "continue")
    for _, text in PENDING.values():
        orch.handle_message(sid, text)
    tree = orch.store.get(sid).l3_raw["model_result"]
    review = review_for(tree["session_id"], therapy={"primary": "UNSUPPORTED"})
    with pytest.raises(L3ToL4ValidationError, match="V3-THERAPY-MATCH"):
        orch.submit_icd10_review(sid, review)
    tree["candidates"][0]["confidence"] = .49
    with pytest.raises(L3ToL4ValidationError, match="V3-CANDIDATE-CONFIDENCE"):
        orch.submit_icd10_review(sid, review_for(tree["session_id"]))
    session = orch.get_session_snapshot(sid)
    assert session.route == "MANUAL_REVIEW" and session.l4_session_id is None
