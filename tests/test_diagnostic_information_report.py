from copy import deepcopy

import pytest

from layers.layer3.api import DiagnoseResponse, build_blocked_report, diagnose
from layers.layer3.diagnosis.assessment import assessments_from_step1
from layers.layer3.diagnosis.clinicr import CLINICR
from layers.layer3.diagnosis.decision_tree import criterion_nodes
from layers.layer3.diagnosis.information_report import build_information_report
from layers.layer3.diagnosis.verifier import IndependentVerifier
from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider
from tests.l3_test_double import ControlledRewardModelVerifier
from tests.tree_test_support import controlled_generation


def raw_report(labels=("D013",), *, complete=(), overrides=None):
    knowledge = JsonDiagnosticKnowledgeProvider()
    generation = controlled_generation(list(labels), complete_labels=list(complete), overrides=overrides)
    assessments = assessments_from_step1(generation, generation.candidates, knowledge=knowledge)
    return {
        **generation.model_dump(mode="json"),
        "checked": [a.model_dump(mode="json") for a in assessments],
        "diagnosis": {"primary": None, "secondary": []}, "verifier_routing": "return_to_l2",
    }, knowledge


@pytest.mark.parametrize("status,expected", [("pending", "insufficient_information"), ("not_met", "known_blocker")])
def test_pending_is_distinct_from_known_nonfulfillment_and_input_is_unchanged(status, expected):
    code = "criterion.D013.depressive_episode/min_duration"
    report, knowledge = raw_report(complete=("D013",), overrides={code: status})
    before = deepcopy(report)
    information = build_information_report(report, knowledge=knowledge)
    candidate = information["diagnostic_readiness"]["candidates"][0]
    assert candidate["status"] == expected
    if status == "pending":
        assert candidate["pending_required_nodes"] == [code]
        assert candidate["known_blockers"] == []
    else:
        assert candidate["known_blockers"][0]["node_id"] == code
    assert report == before


def test_known_blocker_and_pending_are_both_reported_but_only_pending_is_a_question():
    count = "criterion.D013.depressive_episode/min_count"
    duration = "criterion.D013.depressive_episode/min_duration"
    report, knowledge = raw_report(complete=("D013",), overrides={count: "not_met", duration: "pending"})
    info = build_information_report(report, knowledge=knowledge)
    candidate = info["diagnostic_readiness"]["candidates"][0]
    assert candidate["status"] == "known_blocker"
    assert candidate["pending_required_nodes"] == [duration]
    assert [q["node_id"] for q in info["information_collection_plan"]["items"]] == [duration]
    assert report["candidate_coverage"]["D013"] == "provisional_candidate"


@pytest.mark.parametrize("optional_status", ["pending", "not_met"])
def test_optional_condition_does_not_block_or_become_mandatory_question(optional_status):
    knowledge = JsonDiagnosticKnowledgeProvider()
    optional = next(n for n, d in criterion_nodes("D008", knowledge=knowledge).items() if not d["mandatory"])
    report, _ = raw_report(("D008",), complete=("D008",), overrides={optional: optional_status})
    info = build_information_report(report, knowledge=knowledge)
    assert info["diagnostic_readiness"]["status"] == "supported"
    assert info["information_collection_plan"]["items"] == []
    assert not info["diagnostic_readiness"]["final_diagnosis_reported"]
    assert info["diagnostic_readiness"]["verifier_routing"] == "return_to_l2"


def test_negative_guard_requirement_is_satisfied_by_raw_not_met():
    report, knowledge = raw_report(complete=("D013",))
    info = build_information_report(report, knowledge=knowledge)
    assert info["diagnostic_readiness"]["status"] == "supported"
    assert not info["diagnostic_readiness"]["candidates"][0]["known_blockers"]


def test_missing_path_record_is_unknown_even_when_global_trace_contains_met():
    code = "criterion.D013.depressive_episode/min_count"
    report, knowledge = raw_report(complete=("D013",))
    path = report["candidates"][0]["decision_paths"][0]
    path["decisions"] = [n for n in path["decisions"] if n["node_id"] != code]
    info = build_information_report(report, knowledge=knowledge)
    assert info["diagnostic_readiness"]["status"] == "insufficient_information"
    assert code in info["diagnostic_readiness"]["candidates"][0]["pending_required_nodes"]


def test_top_three_only_orders_questions_and_later_candidates_remain_in_queue():
    labels = ("D013", "D011", "D003", "D002", "D001")
    report, knowledge = raw_report(labels)
    report["candidates"].sort(key=lambda c: c["disorder"], reverse=True)
    for c in report["candidates"]:
        c["confidence"] = 0.8
    before = deepcopy(report)
    info = build_information_report(report, knowledge=knowledge)
    plan = info["information_collection_plan"]
    assert plan["candidate_order"] == ["D001", "D002", "D003", "D011", "D013"]
    assert plan["top_1_candidates"] == ["D001"]
    assert plan["top_3_candidates"] == ["D001", "D002", "D003"]
    assert plan["remaining_candidates"] == ["D011", "D013"]
    assert any("D013" in q["required_by"] and q["priority"] == "remaining" for q in plan["items"])
    assert {q["priority"] for q in plan["items"]} == {"top_1", "top_3", "remaining"}
    assert report == before


def test_shared_guard_question_is_deduplicated_and_preserves_each_expected_state():
    report, knowledge = raw_report(("D013", "D015"))
    info = build_information_report(report, knowledge=knowledge)
    questions = [q for q in info["information_collection_plan"]["items"] if q["node_id"] == "history.mania"]
    assert len(questions) == 1
    assert set(questions[0]["required_by"]) == {"D013", "D015"}
    assert questions[0]["expected_by_candidate"] == {"D013": "not_met", "D015": "not_met"}


def test_input_guard_reports_unknown_without_calling_engine_and_api_keeps_fields():
    class Sentinel:
        def run(self, *args, **kwargs):
            pytest.fail("Incomplete input must not reach engine")

    report = diagnose({}, clinicr=Sentinel())
    assert report["diagnostic_readiness"]["scope"] == "input_contract_only"
    assert report["diagnostic_readiness"]["status"] == "insufficient_information"
    assert report["information_collection_plan"]["items"]
    assert all(q["priority"] == "input_contract" for q in report["information_collection_plan"]["items"])
    parsed = DiagnoseResponse.model_validate(report)
    assert parsed.diagnostic_readiness == report["diagnostic_readiness"]
    assert parsed.information_collection_plan == report["information_collection_plan"]
    assert report["verifier_routing"] == "return_to_l2"


@pytest.mark.parametrize("node_status", ["pending", "not_met"])
def test_live_engine_reports_information_gap_without_rejudgment_or_verifier_call(node_status):
    generation = controlled_generation(["D013"], complete_labels=["D013"],
                                      overrides={"criterion.D013.depressive_episode/min_duration": node_status})

    class Reasoner:
        def generate_candidates(self, profile):
            return generation

    model = ControlledRewardModelVerifier()
    engine = CLINICR(reasoner=Reasoner(), verifier=IndependentVerifier(model=model),
                     knowledge=JsonDiagnosticKnowledgeProvider())
    result = engine.run({"evidence_nodes": [{"evidence_id": "ev-mood"}]})
    assert model.calls == []
    expected = "insufficient_information" if node_status == "pending" else "known_blocker"
    assert result["diagnostic_readiness"]["status"] == expected
    assert ("信息不足" if node_status == "pending" else "已知不满足") in result["summary"]
    assert result["diagnosis"]["primary"] is None
    assert result["verifier_routing"] == "return_to_l2"
    assert result["candidate_coverage"] == generation.candidate_coverage
    assert result["tree_trace"] == generation.model_dump(mode="json")["tree_trace"]
    blocked = build_blocked_report(result)
    assert blocked["information_collection_plan"] == result["information_collection_plan"]
    assert ("信息不足" if node_status == "pending" else "已知不满足") in blocked["structured_labels"]["wording_constraint"]
