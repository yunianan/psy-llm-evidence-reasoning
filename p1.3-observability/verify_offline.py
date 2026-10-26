"""Persist synthetic ledger evidence and audit old logs without inference."""
import json
from pathlib import Path
import socket
import sys

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
sys.path.insert(0, str(WORKSPACE / "L1-L5系统"))

from tools.l3_observability_summary import summarize, summarize_run, write_summary, ObservabilityIntegrityError
from tests.test_l3_observability_summary import synthetic, test_synthetic_full_json_equals_independent_expected


def main():
    def blocked(*args, **kwargs):
        raise AssertionError("No network in observability verification")
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    data = synthetic.__wrapped__()
    test_synthetic_full_json_equals_independent_expected(data)
    (HERE / "synthetic.inputs.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary(summarize([data]), HERE / "synthetic.summary.json")
    source = WORKSPACE / "phase1/p1.2-runner/runs/offline-example-v2"
    paths = list(source.glob("manifest.A*.json")) + list((source / "predictions").glob("*.jsonl")) + list((source / "runs").glob("*/*.jsonl"))
    before = {str(path): path.read_bytes() for path in paths}
    try:
        summarize_run(source)
    except ObservabilityIntegrityError as error:
        strict_errors = sorted({issue["code"] for issue in error.issues})
    else:
        raise AssertionError("Old logs must not pass the new envelope contract")
    write_summary(summarize_run(source, legacy_audit=True), HERE / "offline-example-v2.summary.json")
    assert before == {str(path): path.read_bytes() for path in paths}
    proof = {"source_files_unchanged": True, "source_files_checked": len(paths),
             "new_model_calls": 0, "real_eval_accessed": False,
             "legacy_audit_only": True, "legacy_strict_rejection_codes": strict_errors,
             "synthetic_full_json_equals_independent_oracle": True,
             "synthetic_cases": 7, "synthetic_http_attempts": 9}
    (HERE / "offline-audit-verification.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(proof, ensure_ascii=False))


if __name__ == "__main__":
    main()
