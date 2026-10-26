import importlib
from pathlib import Path
import sys

import pytest

from step3_reward_verifier import (
    IndependentVerifier, ReplayScorer, VerifierModelUnavailable,
    case_from_layer3, verify_case,
)


@pytest.mark.parametrize("score,route", [
    (0.0, "return_to_l2"), (0.4999, "return_to_l2"),
    (0.5, "conditional"), (0.7999, "conditional"),
    (0.8, "pass"), (1.0, "pass"),
])
def test_threshold_boundaries(case, response_factory, score, route):
    result = verify_case(case, ReplayScorer(response_factory(score)))
    assert result["routing"] == route
    assert result["confidence"] == score
    assert result["candidate_confidence"] == case.candidate.confidence
    assert (result["diagnosis"]["primary"] is not None) == (route != "return_to_l2")


@pytest.mark.parametrize("score", [0.0, 0.4999, 0.5, 0.7999, 0.8, 1.0])
def test_extracted_verifier_matches_original_layer3(case, response_factory, score):
    original = Path(__file__).resolve().parents[2] / "L1-L5系统"
    if not original.is_dir():
        pytest.skip("原系统目录不存在，仅跳过基线对照")
    sys.path.insert(0, str(original))
    baseline = importlib.import_module("layers.layer3.diagnosis.verifier").IndependentVerifier
    expected = baseline(model=ReplayScorer(response_factory(score))).verify(
        case.candidate, case.assessment, case.profile, case.differential_context,
    )
    actual = IndependentVerifier(model=ReplayScorer(response_factory(score))).verify(
        case.candidate, case.assessment, case.profile, case.differential_context,
    )
    assert actual == expected


def test_custom_scorer_does_not_receive_initial_confidence(case, response_factory):
    class Scorer(ReplayScorer):
        def score_reasoning(self, profile, candidate_without_confidence, assessment, differential_context):
            assert "confidence" not in candidate_without_confidence
            for path in candidate_without_confidence.get("decision_paths", []):
                assert all("confidence" not in record for record in path["decisions"])
            assert profile["_allowed_evidence_refs"] == ["ev-interface"]
            return super().score_reasoning(profile, candidate_without_confidence, assessment, differential_context)
    case.profile["_allowed_evidence_refs"] = ["invented"]
    case.candidate.confidence = 0.01
    result = verify_case(case, Scorer(response_factory()))
    # Step 1 置信度门槛属于原 CLINICR，本实验只验证 Step 3。
    assert result["routing"] == "pass"
    assert case.candidate.confidence == 0.01


@pytest.mark.parametrize("score", [0.4, 0.6])
def test_nonpassing_result_needs_specific_follow_up(case, response_factory, score):
    response = response_factory(score)
    response["missing_dimensions"] = []
    with pytest.raises(VerifierModelUnavailable):
        verify_case(case, ReplayScorer(response))


def test_dimensions_must_be_unique(case, response_factory):
    response = response_factory()
    response["dimensions"][1]["dimension"] = "evidence_sufficiency"
    with pytest.raises(VerifierModelUnavailable, match="四个质量维度"):
        verify_case(case, ReplayScorer(response))


def test_unknown_evidence_is_rejected(case, response_factory):
    response = response_factory()
    response["dimensions"][0]["evidence_refs"] = ["invented"]
    with pytest.raises(ValueError, match="白名单"):
        verify_case(case, ReplayScorer(response))


def test_candidate_and_assessment_must_agree(case, response_factory):
    case.assessment.disorder = "D011"
    with pytest.raises(ValueError, match="ID 不一致"):
        verify_case(case, ReplayScorer(response_factory()))


def test_secondary_candidates_are_not_marked_as_independently_verified(case, response_factory):
    case.secondary_candidates = [{"disorder": "D011", "confidence": 0.55}]
    result = verify_case(case, ReplayScorer(response_factory()))
    assert result["diagnosis"]["secondary"] == case.secondary_candidates
    assert result["secondary_verification"] == "not_performed"


def test_saved_l3_result_can_be_packed_without_running_candidate_generation(case):
    assessment = case.assessment.model_dump(mode="json")
    candidate = case.candidate.model_dump(mode="json")
    result = {
        "provisional_diagnosis": {"disorder": "D013"},
        "candidates": [{**candidate, "display_name": "display metadata"}],
        "checked": [{**assessment, "source": "step1_tree_records"}],
        "diagnosis": {"secondary": []},
    }
    packed = case_from_layer3(case_id="packed", profile=case.profile, result=result)
    assert packed.candidate == case.candidate
    assert packed.assessment == case.assessment


@pytest.mark.xfail(strict=True, reason="保留原 Step 3 已知缺陷：高总分与全部维度失败时仍通过，待后续修复")
def test_contradictory_high_total_should_not_pass(case, response_factory):
    response = response_factory(0.99)
    for dimension in response["dimensions"]:
        dimension["score"] = 0.0
    response["fail_reasons"] = ["四项均失败"]
    assert verify_case(case, ReplayScorer(response))["routing"] != "pass"
