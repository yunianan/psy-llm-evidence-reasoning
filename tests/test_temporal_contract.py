"""工程区间策略、三值病程门和L2/L3往返的离线回归。"""
from datetime import date

import pytest
from pydantic import ValidationError

from shared.clinical_contract import CONTRACT_VERSION, DurationEvidence, follow_up_task
from shared.temporal import parse_duration, compare_duration
from shared.clinical_profile import ClinicalProfile
from layers.layer2.schemas import L2SessionState, SemanticAnalysis
from layers.layer2.profile.ingest import ProfileIngestor
from layers.layer2.profile.builder import build_clinical_intake
from layers.layer2.adapters import L2ToL3Adapter
from shared.clinical_profile import ClinicalProfile


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


def test_natural_language_duration_survives_ingest_adapter_and_profile_roundtrip():
    state = L2SessionState(session_id="duration", safety_passport={}, turn_index=1)
    ingest(state, "睡不好已经3个月了", attrs={"duration": "3个月"})
    assert state.diagnostic_context.duration_evidence
    profile = L2ToL3Adapter.serialize_profile(build_clinical_intake(state))
    request = ClinicalProfile.model_validate(profile)
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


def test_all_scale_items_survive_profile_model():
    profile = {f"{scale}_items": {"item1": 0} for scale in ("PHQ9", "GAD7", "MDQ", "PCL5", "ISI", "Y_BOCS")}
    request = ClinicalProfile.model_validate(profile).model_dump()
    assert all(request[k] == v for k, v in profile.items())


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
