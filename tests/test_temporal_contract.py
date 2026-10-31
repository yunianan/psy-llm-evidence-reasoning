"""工程区间策略、三值病程门和L2/L3往返的离线回归。"""
from datetime import date

import pytest
from pydantic import ValidationError

from shared.clinical_contract import CONTRACT_VERSION, DurationEvidence, follow_up_task
from shared.temporal import parse_duration, compare_duration
from shared.temporal_gate import evaluate_temporal_node, apply_temporal_contract
from shared.clinical_profile import ClinicalProfile
from layers.layer2.schemas import L2SessionState, SemanticAnalysis
from layers.layer2.profile.ingest import ProfileIngestor
from layers.layer2.profile.builder import build_clinical_intake
from layers.layer2.adapters import L2ToL3Adapter
from layers.layer3.api import DiagnoseRequest
from layers.layer3.diagnosis.decision_tree import criterion_nodes
from layers.layer4.schemas.l3_report import DiagnosticReport
from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider


@pytest.mark.parametrize("text,lo,hi,status,approx,pattern", [
    ("约六个月", 165, 200, "ok", True, "unknown"),
    ("八个月", 243, 243, "ok", False, "unknown"),
    ("一年多", 365, 540, "ok", True, "unknown"),
    ("大半年", 150, 240, "ok", True, "unknown"),
    ("两周左右", 10, 18, "ok", True, "unknown"),
    ("几个月", None, None, "vague", True, "unknown"),
    ("去年冬天开始", None, None, "unresolved", False, "unknown"),
    ("时好时坏三年了", 1096, 1096, "ok", False, "episodic"),
    ("一个半月", 45, 45, "ok", False, "unknown"),
    ("1个半月", 45, 45, "ok", False, "unknown"),
    ("一到两个月", 30, 60, "ok", True, "unknown"),
    ("3-4个月", 91, 121, "ok", True, "unknown"),
    ("三四个月", 91, 121, "ok", True, "unknown"),
    ("俩月", 60, 60, "ok", False, "unknown"),
    ("十二天", 12, 12, "ok", False, "unknown"),
    ("1年两个月", 426, 426, "ok", False, "unknown"),
    ("好一阵子", None, None, "vague", True, "unknown"),
    ("about six months", 165, 200, "ok", True, "unknown"),
    ("maybe eight months now", 220, 267, "ok", True, "unknown"),
])
def test_interval_policy(text, lo, hi, status, approx, pattern):
    record = parse_duration(text)
    assert (record.min_days, record.max_days, record.parse_status, record.approx, record.pattern) == (lo, hi, status, approx, pattern)
    assert record.raw_span == text


@pytest.mark.parametrize("text,status", [("八个月", "met"), ("两个月", "not_met"),
    ("大半年", "unknown"), ("约六个月", "met_approx"), ("几个月", "unknown"),
    ("去年冬天开始", "unknown"), ("将近半年", "not_met"), ("不到半年", "not_met")])
def test_three_valued_duration_gate(text, status):
    assert compare_duration(parse_duration(text), "6 months").status == status


def test_pattern_and_episode_kind_are_not_interchangeable():
    overall = parse_duration("时好时坏三年了")
    assert compare_duration(overall, "2 weeks", required_pattern="continuous").status == "unknown"
    episode = parse_duration("每次两周左右")
    assert episode.duration_kind == "episode"
    assert compare_duration(episode, "6 months").status == "unknown"


def test_overall_and_per_episode_duration_remain_separate_in_l2():
    state = L2SessionState(session_id="episodes", safety_passport={}, turn_index=1)
    ingest(state, "反复三年了，每次两周左右")
    records = state.diagnostic_context.duration_evidence
    assert [(r.duration_kind, r.min_days, r.max_days) for r in records] == [("overall", 1096, 1096), ("episode", 10, 18)]
    assert all(r.pattern == "episodic" for r in records)


def test_relative_date_anchor_is_explicit_and_event_after_is_not_exact_onset():
    record = parse_duration("去年冬天开始", interview_date="2026-10-30")
    assert record.parse_status == "ok" and record.min_days < record.max_days
    assert record.interview_date == date(2026, 10, 30)
    unresolved = parse_duration("从高三起", interview_date="2026-10-30")
    assert unresolved.parse_status == "unresolved"
    event = parse_duration("换工作以后", interview_date="2026-10-30", event_dates={"换工作": "2026-01-01"})
    assert event.min_days == 0 and event.max_days == 302


@pytest.mark.parametrize("text", ["2025年12月开始", "去年12月开始", "3月开始", "2025年起"])
def test_calendar_onset_is_not_parsed_as_years_or_months_of_duration(text):
    unresolved = parse_duration(text)
    assert unresolved.parse_status == "unresolved"
    assert unresolved.min_days is None
    anchored = parse_duration(text, interview_date="2026-10-30")
    if text != "3月开始":
        assert anchored.parse_status == "ok" and anchored.max_days < 731


def ingest(state, text, *, attrs=None, value=None):
    analysis = SemanticAnalysis.model_validate({"input_type": "free_narrative",
        "symptoms": [{"value": "insomnia", "evidence_quote": text, "attributes": attrs or {}}],
        "structured_fields": [{"field": "symptom_duration_days", "value": value or text, "evidence_quote": text}]})
    return ProfileIngestor().ingest(state, analysis=analysis, patient_text=text, message_id=f"m{state.turn_index}")


def test_natural_language_duration_survives_ingest_adapter_and_api_roundtrip():
    state = L2SessionState(session_id="duration", safety_passport={}, turn_index=1)
    ingest(state, "睡不好已经3个月了", attrs={"duration": "3个月"})
    assert state.diagnostic_context.duration_evidence
    profile = L2ToL3Adapter.serialize_profile(build_clinical_intake(state))
    request = DiagnoseRequest.model_validate(profile)
    assert request.duration_evidence[0].raw_span == "睡不好已经3个月了"
    assert request.duration_evidence[0].symptom_id == "insomnia"
    assert request.duration_evidence[0].min_days == 91
    assert request.duration_evidence[0].evidence_refs[0] in {n["evidence_id"] for n in profile["evidence_nodes"]}


def test_conflicting_attribute_values_retain_history_and_conflict_state():
    state = L2SessionState(session_id="conflict", safety_passport={}, turn_index=1)
    ingest(state, "睡不好已经21天了", attrs={"duration_days": 21}, value=21)
    state.turn_index = 2
    ingest(state, "睡不好已经120天了", attrs={"duration_days": 120}, value=120)
    assert state.diagnostic_context.evidence_conflicts
    assert {d.min_days for d in state.diagnostic_context.duration_evidence} == {21, 120}
    assert any(f.state == "conflict" for f in state.clinical_facts)


def definition(label="Insomnia"):
    key = f"criterion.{label}.min_duration"
    return {"node_id": key, **criterion_nodes(label, knowledge=JsonDiagnosticKnowledgeProvider())[key]}


def test_l3_writeback_is_bound_to_the_cited_quote_and_approx_stays_pending():
    profile = {"evidence_nodes": [{"evidence_id": "e", "quote": "约六个月"}]}
    decision = evaluate_temporal_node(profile, definition("D001"), evidence_refs=["e"])
    assert decision.status == "met_approx"
    assert profile["duration_evidence"][0]["source"] == "L3_writeback"
    assert profile["duration_evidence"][0]["evidence_refs"] == ["e"]
    assert profile["duration_evidence"][0]["raw_span"] in profile["evidence_nodes"][0]["quote"]


def test_uncited_duration_and_wrong_symptom_do_not_satisfy_gate():
    record = parse_duration("八个月", symptom_id="depressed_mood", scope="symptom:depressed_mood", evidence_refs=["mood"])
    profile = {"duration_evidence": [record.model_dump(mode="json")], "evidence_nodes": [{"evidence_id": "mood", "quote": "八个月"}]}
    assert evaluate_temporal_node(profile, definition(), evidence_refs=["mood"]).status == "unknown"
    assert evaluate_temporal_node({"evidence_nodes": [{"evidence_id": "e", "quote": "八个月"}]}, definition()).status == "unknown"
    # 仅有候选节点作用域、没有症状ID时，也不能转借其他症状群。
    record.symptom_id, record.scope = "", "criterion.D001.min_duration"
    profile["duration_evidence"] = [record.model_dump(mode="json")]
    assert evaluate_temporal_node(profile, definition(), evidence_refs=["mood"]).status == "unknown"


def test_forged_structured_range_is_recomputed_from_patient_quote():
    record = parse_duration("两周", symptom_id="insomnia", evidence_refs=["e"])
    record.min_days = record.max_days = 5000
    profile = {"duration_evidence": [record.model_dump(mode="json")], "evidence_nodes": [{"evidence_id": "e", "quote": "两周"}]}
    assert evaluate_temporal_node(profile, definition(), evidence_refs=["e"]).status == "not_met"
    assert profile["temporal_audit"][0]["detected_errors"] == ["structured_interval_disagrees_with_raw_span"]


def test_explicit_elapsed_time_does_not_require_the_event_anchor():
    record = parse_duration("换工作以后睡不好，已经八个月了")
    assert record.parse_status == "ok" and record.min_days == 243


def test_non_temporal_few_and_lately_do_not_hide_explicit_duration():
    record = parse_duration("About six months now, I have missed a few appointments lately.")
    assert record.parse_status == "ok" and record.min_days == 165


def test_followup_task_preserves_node_target_and_binds_ellipsis_answer():
    state = L2SessionState(session_id="task", safety_passport={}, turn_index=1,
        follow_up_tasks=[follow_up_task({"field": "decision_tree.criterion.Insomnia.min_duration",
            "symptom_id": "insomnia", "suggested_question": "睡眠问题持续多久？", "priority": "top_1"})])
    ProfileIngestor().ingest(state, analysis=SemanticAnalysis(input_type="free_narrative"),
        patient_text="差不多有大半年了", message_id="m1")
    task = state.follow_up_tasks[0]
    assert task.status == "answered" and task.answer_evidence_refs
    record = state.diagnostic_context.duration_evidence[0]
    assert record.symptom_id == "insomnia" and record.scope == "criterion.Insomnia.min_duration"
    assert record.min_days == 150 and record.max_days == 240


def test_new_report_fields_and_all_scale_items_survive_public_models():
    value = {"information_collection_plan": {"items": [{"node_id": "n"}]}, "diagnostic_readiness": {"status": "insufficient_information"},
        "duration_evidence": [{"raw_span": "六个月"}], "temporal_audit": [{"node_id": "n"}]}
    parsed = DiagnosticReport.model_validate(value).model_dump()
    assert all(parsed[k] == v for k, v in value.items())
    profile = {f"{scale}_items": {"item1": 0} for scale in ("PHQ9", "GAD7", "MDQ", "PCL5", "ISI", "Y_BOCS")}
    request = DiagnoseRequest.model_validate(profile).model_dump()
    assert all(request[k] == v for k, v in profile.items())


def test_http_endpoint_forwards_the_complete_shared_profile(monkeypatch):
    from layers.layer3 import api
    captured = []
    original_guard = api.build_input_contract_report
    def capture(profile, safety):
        captured.append(profile)
        return original_guard(profile)
    monkeypatch.setattr(api, "diagnose", capture)
    request = DiagnoseRequest(GAD7_items={"item1": 2}, ISI_items={"item1": 1})
    api.diagnose_endpoint(request)
    assert captured[0]["GAD7_items"] == {"item1": 2}
    assert captured[0]["ISI_items"] == {"item1": 1}
    assert captured[0]["contract_version"] == CONTRACT_VERSION


def test_frequency_conflict_does_not_mark_duration_as_conflicting():
    state = L2SessionState(session_id="independent-slots", safety_passport={}, turn_index=1)
    ingestor = ProfileIngestor()
    for text, attrs in [("睡不好3个月，每周两次", {"duration": "3个月", "frequency": "每周两次"}),
                        ("现在每周四次", {"frequency": "每周四次"})]:
        analysis = SemanticAnalysis.model_validate({"input_type": "free_narrative", "symptoms": [
            {"value": "insomnia", "evidence_quote": text, "attributes": attrs}]})
        ingestor.ingest(state, analysis=analysis, patient_text=text, message_id=f"m{state.turn_index}")
        state.turn_index += 1
    assert all(f.state == "present" for f in state.clinical_facts if f.attribute == "duration")
    assert all(f.state == "conflict" for f in state.clinical_facts if f.attribute == "frequency")


def test_interval_schema_and_internal_profile_reject_invalid_values():
    with pytest.raises(ValidationError):
        DurationEvidence(raw_span="x", min_days=10, max_days=3)
    with pytest.raises(ValidationError):
        ClinicalProfile(symptom_duration_days="3个月")
    with pytest.raises(ValidationError):
        ClinicalProfile(functional_impairment="unknown")
    with pytest.raises(ValidationError):
        ClinicalProfile(contract_version="legacy_v1")
    with pytest.raises(ValidationError):
        ClinicalProfile(duration_evidence=[parse_duration("八个月").model_dump(mode="json")])


def test_live_v2_clinicr_corrects_model_duration_and_returns_writeback_without_mutating_input():
    from copy import deepcopy
    from layers.layer3.diagnosis.clinicr import CLINICR
    from tests.tree_test_support import controlled_generation

    class Reasoner:
        def generate_candidates(self, profile):
            return controlled_generation(["Insomnia"], complete_labels=["Insomnia"], ref="e")

    class NoVerifier:
        def run(self, *args, **kwargs):
            raise AssertionError("病程反证不应进入奖励终验")

    profile = {"contract_version": CONTRACT_VERSION, "evidence_nodes": [{"evidence_id": "e", "quote": "两周"}]}
    original = deepcopy(profile)
    result = CLINICR(reasoner=Reasoner(), verifier=NoVerifier(), knowledge=JsonDiagnosticKnowledgeProvider()).run(profile)
    assert profile == original
    assert result["diagnosis"]["primary"] is None
    audit = next(a for a in result["temporal_audit"] if a["node_id"] == "criterion.Insomnia.min_duration")
    assert audit["original_status"] == "met" and audit["effective_status"] == "not_met"
    assert audit["remaining_errors"] == []
    assert any(d["source"] == "L3_writeback" and d["raw_span"] == "两周" for d in result["duration_evidence"])


def test_l2_accepts_node_plan_and_keeps_l3_source_on_writeback():
    from tests.test_layer2_assessment import build_ready_session
    service, _, _, sid = build_ready_session()
    state = service.get_session_snapshot(sid)
    existing = state.diagnostic_context.duration_evidence[0]
    written = existing.model_copy(update={"source": "L3_writeback"}).model_dump(mode="json")
    service.receive_l3_follow_up(sid, [{"field": "decision_tree.criterion.Insomnia.min_duration",
        "symptom_id": "insomnia", "suggested_question": "睡不好多久了？", "priority": "top_1", "required_by": ["Insomnia"]}], [written])
    intake = service.build_clinical_intake(sid)
    assert intake.follow_up_tasks[0].node_id == "criterion.Insomnia.min_duration"
    assert intake.follow_up_tasks[0].priority == "top_1"
    assert any(d.source == "L3_writeback" for d in intake.diagnostic_context.duration_evidence)
