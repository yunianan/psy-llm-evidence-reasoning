import json
from pathlib import Path

import pytest

from tools.phase1_joint_run import JointControl, run_joint
from tools.phase1_research_entry import export_prediction
from tools.l3_engineering_runtime import BatchBudgetExceeded


def rows(n=25):
    return [{"case_id": f"c{i:03}", "profile": {}} for i in range(n)]


def budget(batch=30, case=1):
    return {"pricing_basis": "official_peak_all_input_miss_reference", "batch_limit_yuan": batch, "case_limit_yuan": case}


def test_two_arms_share_one_cap_and_separate_case_allowances(tmp_path):
    joint = JointControl(rows(2), tmp_path, budget(.5, .3), estimates={"A0": .001, "A1": .001})
    prompt = "x" * 50000
    for arm in ("A0", "A1"):
        adapter = joint.arm_budget(arm)
        reserved = adapter.reserve("same_case", prompt, 16000)
        adapter.finish("same_case", reserved, {"prompt_tokens": 50000, "completion_tokens": 16000})
    assert joint.budget.spent == pytest.approx(.456)
    assert set(joint.budget.by_case) == {"A0:same_case", "A1:same_case"}
    with pytest.raises(BatchBudgetExceeded):
        joint.arm_budget("A0").reserve("new_case", prompt, 16000)


def test_forecast_stops_before_spending(tmp_path):
    joint = JointControl(rows(25), tmp_path, budget(1), estimates={"A0": .04, "A1": .87469})
    assert joint.before_case("A0", "c000")
    assert joint.stop_reasons == ["budget_forecast"] and joint.budget.spent == 0


def test_same_order_and_no_new_calls_after_joint_stop(tmp_path):
    calls = []
    def execute(arm):
        def run(row):
            calls.append((arm, row["case_id"]))
            failure = "contract" if arm == "A1" and row["case_id"] in {"c001", "c002"} else None
            return export_prediction(row["case_id"], arm, {"candidates": [], "final_diagnoses": []}, failure_class=failure)
        return run
    result = run_joint(rows(), tmp_path, budget_ref=budget(), model="fixture", executors={a: execute(a) for a in ("A0", "A1")})
    assert calls == [(a, f"c{i:03}") for i in range(3) for a in ("A0", "A1")]
    assert result["stop_reasons"] and result["cases"] == 25
    assert result["complete"] == {"A0": 3, "A1": 1}
    for arm in ("A0", "A1"):
        predictions = [json.loads(line) for line in (tmp_path / f"predictions/{arm}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(predictions) == 25
        assert all(r["failure_class"] == "administrative" for r in predictions[3:])
    before = list(calls)
    again = run_joint(rows(), tmp_path, budget_ref=budget(), model="fixture", resume=True,
                      executors={a: execute(a) for a in ("A0", "A1")})
    assert calls == before and again["complete"] == result["complete"]


def test_joint_completion_catches_disjoint_single_arm_failures(tmp_path):
    calls = []
    def execute(arm):
        def run(row):
            calls.append((arm, row["case_id"]))
            failure = "contract" if (arm, row["case_id"]) in {("A0", "c000"), ("A1", "c001")} else None
            return export_prediction(row["case_id"], arm, {"candidates": []}, failure_class=failure)
        return run
    result = run_joint(rows(), tmp_path, budget_ref=budget(), model="fixture", executors={a: execute(a) for a in ("A0", "A1")})
    assert len(calls) == 4
    assert "joint_completion_impossible" in result["stop_reasons"]


def test_identity_change_cannot_resume_and_new_run_cannot_overwrite(tmp_path):
    executors = {a: lambda row, a=a: export_prediction(row["case_id"], a, {"candidates": []}) for a in ("A0", "A1")}
    run_joint(rows(2), tmp_path, budget_ref=budget(), model="fixture", executors=executors)
    with pytest.raises(ValueError):
        run_joint(rows(2), tmp_path, budget_ref=budget(), model="fixture", executors=executors)
    with pytest.raises(ValueError, match="identity"):
        run_joint(rows(2), tmp_path, budget_ref=budget(), model="other", executors=executors, resume=True)


def test_peer_error_does_not_deadlock(tmp_path, monkeypatch):
    import tools.phase1_joint_run as module
    monkeypatch.setattr(module, "research_run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fixture setup error")))
    with pytest.raises(RuntimeError, match="fixture setup"):
        run_joint(rows(2), tmp_path, budget_ref=budget(), model="fixture", executors={})
