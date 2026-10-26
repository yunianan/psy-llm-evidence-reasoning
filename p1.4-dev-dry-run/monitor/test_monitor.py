import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("local_progress_monitor", Path(__file__).with_name("serve.py"))
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def run(tmp_path):
    put(tmp_path / "joint-manifest.json", {"started_at": "2026-10-25T21:00:00+08:00", "identity": {"model": "fixture", "budget_reference": {"batch_limit_yuan": 30}}})
    put(tmp_path / "manifest.A0.json", {"identity": {"case_ids": ["c1"]}})
    put(tmp_path / "runs/A0/c1.json", {"run_failure": False, "failure_class": None})
    started = {"event": "attempt_started", "attempt_id": "a1", "case_id": "c1", "stage": "criteria",
               "started_at": "2026-10-25T21:00:00+08:00", "reference_reservation_upper_yuan": .2}
    path = tmp_path / "runs/A1/requests.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(started) + "\n", encoding="utf-8")
    return tmp_path


def test_running_progress_separates_completed_pairs_and_reservations(run):
    before = {str(p): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    value = monitor.build_status(run, alive=True)
    assert value["status"] == "running"
    assert value["terminal_arms"] == 1 and value["joint_complete"] == 0
    assert value["reference_budget"]["consumed_upper_yuan"] == 0
    assert value["reference_budget"]["pending_reservation_yuan"] == .2
    assert value["arms"]["A1"]["active"] == 1
    assert before == {str(p): p.read_bytes() for p in run.rglob("*") if p.is_file()}


def test_abnormal_exit_is_not_shown_as_running(run):
    assert monitor.build_status(run, alive=False)["status"] == "abnormal"


def test_final_summary_has_priority_over_process_exit(run):
    put(run / "joint-summary.json", {"stop_reasons": ["budget_forecast"], "finished_at": "2026-10-25T21:10:00+08:00"})
    value = monitor.build_status(run, alive=False)
    assert value["status"] == "stopped" and value["final_ready"]
    assert value["elapsed_seconds"] == 600
    put(run / "joint-summary.json", {"stop_reasons": []})
    assert monitor.build_status(run, alive=False)["status"] == "finished"


def test_unknown_finished_usage_uses_recorded_upper_not_zero(run):
    path = run / "runs/A1/requests.jsonl"
    started = json.loads(path.read_text(encoding="utf-8"))
    finished = {**started, "event": "attempt_finished", "usage": None, "elapsed_seconds": 10,
                "raw_output": "private patient text must never leave the logs"}
    path.write_text(json.dumps(started) + "\n" + json.dumps(finished) + "\n", encoding="utf-8")
    value = monitor.build_status(run, alive=True)
    assert value["http_attempts"] == 1
    assert value["reference_budget"]["consumed_upper_yuan"] == .2
    assert value["reference_budget"]["finished_usage_missing"] == 1
    assert "private patient text" not in json.dumps(value)


def test_live_partial_tail_is_not_changed(run):
    path = run / "runs/A1/requests.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + '{"event":', encoding="utf-8")
    before = path.read_bytes()
    assert len(monitor.load_lines(path)) == 1
    assert path.read_bytes() == before
    path.write_text('{"broken":\n{}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        monitor.load_lines(path)
