"""Sequential paired DEV engineering run: one global budget and stop state."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import copy
import shutil
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.l3_engineering_runtime import ReferenceBudget, BatchBudgetExceeded, recover_ledger, now, legacy_stream_retryable
from tools.paired_evaluation import monitor_block, read_rows
from tools.phase1_research_entry import research_run, write_json, content_hash, _identity, POLICY, VERSION as RUNNER_VERSION
from tools.l3_observability_summary import summarize_run, write_summary

VERSION = "phase1-joint-run-v2"
DEFAULT_ESTIMATES = {"A0": .04, "A1": .87469}


def prepare_engineering_recovery(source, out, rows, identity):
    """Fork the pre-terminal operational prefix; leave the failed run immutable.

    Only the observed httpx2 classification defect is recoverable. Historical
    request costs and attempt IDs stay in the new ledger, without a fresh cap.
    """
    source, out = Path(source).resolve(), Path(out).resolve()
    if source == out or out.exists() or any(p.lower() == "sealed" for p in source.parts):
        raise ValueError("Engineering recovery requires a fresh, separate DEV directory")
    original = json.loads((source / "joint-manifest.json").read_text(encoding="utf-8"))
    old = original["identity"]
    if any(old.get(k) != identity.get(k) for k in ("input_hash", "model", "policy", "budget_reference", "estimates")):
        raise ValueError("Recovery must preserve inputs, model, policy and budget")
    allowed = {"tools/l3_engineering_runtime.py", "tools/phase1_research_entry.py", "tools/phase1_joint_run.py"}
    if old["code_versions"].keys() != identity["code_versions"].keys() or any(
        old["code_versions"][k] != value for k, value in identity["code_versions"].items() if k not in allowed):
        raise ValueError("Medical/prompt/knowledge code changed during transport recovery")
    summarize_run(source)  # Require a strictly balanced source before importing anything.
    imported, successful, source_manifests, event_prefixes = {}, {}, {}, {}
    recovery_cases = []
    for arm in ("A0", "A1"):
        data = read_rows(source / "runs" / arm / "requests.jsonl")
        attempts = {r["attempt_id"]: r for r in data}
        imported[arm], successful[arm] = list(attempts), []
        for row in rows:
            case = row["case_id"]
            value = json.loads((source / "runs" / arm / f"{case}.json").read_text(encoding="utf-8"))
            calls = [r for r in attempts.values() if r["case_id"] == case]
            if not value["run_failure"]:
                successful[arm].append(case)
            elif calls:
                last = calls[-1]
                same_request = [r for r in calls if r.get("logical_request_key") == last.get("logical_request_key")]
                if value.get("failure_class") != "administrative" or not legacy_stream_retryable(last) or len(same_request) >= 3:
                    raise ValueError("Only the identified, unexhausted httpx2 transport defect may be recovered")
                recovery_cases.append({"arm": arm, "case_id": case, "attempt_id": last["attempt_id"]})
        raw = (source / "runs" / arm / "events.jsonl").read_bytes().splitlines(keepends=True)
        prefix = []
        for line in raw:
            event = json.loads(line)
            if event["event"] in {"terminal_failure", "batch_stopped"}:
                break
            prefix.append(line)
        if {json.loads(line)["case_id"] for line in prefix if json.loads(line)["event"] == "case_finished"} != set(successful[arm]):
            raise ValueError("Source is not a recoverable sequential prefix")
        event_prefixes[arm] = b"".join(prefix)
        source_manifests[arm] = json.loads((source / f"manifest.{arm}.json").read_text(encoding="utf-8"))
    if not recovery_cases:
        raise ValueError("No identified transport classification defect to recover")
    continuation = {"version": "phase1-engineering-continuation-v1", "source_run": str(source),
                    "source_joint_manifest_hash": content_hash(original), "source_code_versions": old["code_versions"],
                    "imported_attempt_ids": imported, "imported_complete_cases": successful,
                    "recovery_cases": recovery_cases, "source_preserved": True,
                    "formal_scoring": False, "budget_and_attempts_carried": True}
    joint_identity = {**identity, "engineering_continuation": continuation}
    write_json(out / "joint-manifest.json", {"identity": joint_identity, "started_at": now(), "real_eval_accessed": False})
    for arm in ("A0", "A1"):
        directory = out / "runs" / arm
        directory.mkdir(parents=True)
        for name in ("requests.jsonl",):
            shutil.copyfile(source / "runs" / arm / name, directory / name)
        (directory / "events.jsonl").write_bytes(event_prefixes[arm])
        for case in successful[arm]:
            shutil.copyfile(source / "runs" / arm / f"{case}.json", directory / f"{case}.json")
        manifest = copy.deepcopy(source_manifests[arm])
        manifest["identity"].update(version=RUNNER_VERSION, code_versions=identity["code_versions"],
                                    engineering_continuation=continuation)
        manifest["identity"]["coordination"]["version"] = VERSION
        manifest["started_at"] = now()
        write_json(out / f"manifest.{arm}.json", manifest)
    for folder in ("checkpoints", "g1-nodes"):
        if (source / folder).exists():
            shutil.copytree(source / folder, out / folder)
    write_json(out / "engineering-continuation.json", continuation)
    return continuation


class ArmBudget:
    def __init__(self, joint, arm):
        self.joint, self.arm = joint, arm

    def reserve(self, case_id, prompt, max_tokens):
        self.joint.check_forecast()
        return self.joint.budget.reserve(f"{self.arm}:{case_id}", prompt, max_tokens)

    def finish(self, case_id, amount, usage):
        self.joint.budget.finish(f"{self.arm}:{case_id}", amount, usage)

    def snapshot(self):
        value = self.joint.budget.snapshot()
        prefix = self.arm + ":"
        arm_cases = {k[len(prefix):]: v for k, v in value["per_case_reference"].items() if k.startswith(prefix)}
        return {**value, "scope": "shared_across_A0_A1", "joint_reference_spent_upper_yuan": value["reference_spent_upper_yuan"],
                "reference_spent_upper_yuan": sum(arm_cases.values()), "per_case_reference": arm_cases}


class JointControl:
    def __init__(self, rows, out, budget_ref, *, estimates=None, resume=False):
        self.rows, self.out = rows, Path(out)
        self.ids = [r["case_id"] for r in rows]
        self.schedule = [(arm, case) for case in self.ids for arm in ("A0", "A1")]
        self.condition = threading.Condition()
        self.done, self.started, self.stop_reasons = {}, set(), []
        self.estimates = dict(estimates or DEFAULT_ESTIMATES)
        if set(self.estimates) != {"A0", "A1"} or any(type(v) not in (int, float) or not 0 < v < float("inf") for v in self.estimates.values()):
            raise ValueError("Positive finite per-arm forecasts required")
        self.budget = ReferenceBudget(batch_limit=budget_ref["batch_limit_yuan"], case_limit=budget_ref["case_limit_yuan"])
        self.policy = {"version": VERSION, "schedule": "case_id order, A0 then A1 per case; active inference concurrency 1",
                       "monitor_block_size": 20, "global_budget_reference": budget_ref,
                       "expected_case_reference_yuan": self.estimates, "forecast_multiplier": 1.3}
        if resume:
            for arm in ("A0", "A1"):
                ledger = self.out / "runs" / arm / "requests.jsonl"
                if ledger.exists():
                    old = recover_ledger(ledger)
                    self.budget.spent += old["spent"]
                    self.budget.unknown_upper += old["unknown_upper"]
                    self.budget.by_case.update({f"{arm}:{k}": v for k, v in old["by_case"].items()})
                    self.started.update((arm, k) for k in old["by_case"])
                for case in self.ids:
                    path = self.out / "runs" / arm / f"{case}.json"
                    if path.exists():
                        self.done[arm, case] = json.loads(path.read_text(encoding="utf-8"))
                        if self.done[arm, case].get("stop_after_case"):
                            self.stop_reasons.append("persisted_arm_stop")
            state = self.out / "joint-state.json"
            if state.exists():
                self.stop_reasons.extend(json.loads(state.read_text(encoding="utf-8"))["stop_reasons"])

    def arm_budget(self, arm):
        return ArmBudget(self, arm)

    def _forecast_ok(self):
        remaining = sum(self.estimates[arm] for arm, case in self.schedule if (arm, case) not in self.started and (arm, case) not in self.done)
        return self.budget.batch_limit - self.budget.spent - self.budget.reserved + 1e-10 >= 1.3 * remaining

    def check_forecast(self):
        with self.condition:
            if self.stop_reasons or not self._forecast_ok():
                if not self.stop_reasons:
                    self.stop_reasons.append("budget_forecast")
                self.save()
                raise BatchBudgetExceeded("Joint remaining budget below frozen 1.3x forecast")

    def before_case(self, arm, case):
        with self.condition:
            key = (arm, case)
            while next((k for k in self.schedule if k not in self.done), None) != key:
                if getattr(self, "aborted", False):
                    raise RuntimeError("Peer arm failed; orchestration stopped")
                self.condition.wait(timeout=1)
            if getattr(self, "aborted", False):
                raise RuntimeError("Peer arm failed; orchestration stopped")
            if not self._forecast_ok() and not self.stop_reasons:
                self.stop_reasons.append("budget_forecast")
            self.started.add(key)
            self.save()
            return bool(self.stop_reasons)

    def after_case(self, arm, case, result):
        with self.condition:
            self.done[arm, case] = result
            if result.get("stop_after_case") and not self.stop_reasons:
                self.stop_reasons.append("arm_stop")
            failed_cases = {c for (a, c), value in self.done.items() if value["run_failure"]}
            if len(failed_cases) > len(self.ids) // 20 and not self.stop_reasons:
                self.stop_reasons.append("joint_completion_impossible")
            number = self.ids.index(case)
            if arm == "A1" and ((number + 1) % 20 == 0 or number == len(self.ids) - 1):
                start = number // 20 * 20
                for name in ("A0", "A1"):
                    monitor = monitor_block([self.done[name, c] for c in self.ids[start:number + 1]])
                    if monitor["stop"] and not self.stop_reasons:
                        self.stop_reasons.append("joint_monitor_" + name)
            self.save()
            self.condition.notify_all()

    def save(self):
        write_json(self.out / "joint-state.json", {"version": VERSION, "updated_at": now(),
            "completed_arms": len(self.done), "planned_arms": len(self.schedule),
            "stop_reasons": list(dict.fromkeys(self.stop_reasons)), "budget": self.budget.snapshot(),
            "policy": self.policy})

    def abort(self):
        with self.condition:
            self.stop_reasons.append("orchestration_error")
            # Wake peers; their next gate fails rather than waiting indefinitely.
            self.aborted = True
            self.condition.notify_all()


def run_joint(rows, out, *, budget_ref, model, execute=False, resume=False, executors=None, estimates=None,
              recover_from=None):
    out = Path(out)
    if not rows or [r["case_id"] for r in rows] != sorted({r["case_id"] for r in rows}):
        raise ValueError("A fixed sorted unique list is required")
    if execute and executors is not None:
        raise ValueError("Do not mix live and synthetic executors")
    identity = {"version": VERSION, "input_hash": content_hash(rows), "model": model,
                "policy": POLICY, "code_versions": _identity(), "budget_reference": budget_ref,
                "estimates": dict(estimates or DEFAULT_ESTIMATES)}
    manifest = out / "joint-manifest.json"
    continuation = None
    if recover_from is not None:
        if resume or not execute:
            raise ValueError("Engineering recovery requires explicit live execution, separately from resume")
        continuation = prepare_engineering_recovery(recover_from, out, rows, identity)
        identity["engineering_continuation"] = continuation
        resume = True
    elif resume and manifest.exists():
        continuation = json.loads(manifest.read_text(encoding="utf-8"))["identity"].get("engineering_continuation")
        if continuation is not None:
            identity["engineering_continuation"] = continuation
    if out.exists() and any(out.iterdir()) and not resume:
        raise ValueError("New output directory required; use identical --resume")
    if resume:
        if not manifest.exists() or json.loads(manifest.read_text(encoding="utf-8"))["identity"] != identity:
            raise ValueError("Joint resume identity differs")
    else:
        write_json(manifest, {"identity": identity, "started_at": now(), "real_eval_accessed": False})
    control = JointControl(rows, out, budget_ref, estimates=estimates, resume=resume)
    def worker(arm):
        try:
            return research_run(arm, rows, out, execute=execute, model=model, budget_ref=budget_ref,
                resume=resume, executor=(executors or {}).get(arm), coordination=control, continuation=continuation)
        except BaseException:
            control.abort()
            raise
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {arm: pool.submit(worker, arm) for arm in ("A0", "A1")}
        results = {arm: value.result() for arm, value in pending.items()}
    write_json(out / "manifest.json", {"version": VERSION,
        "arm_manifests": {arm: f"manifest.{arm}.json" for arm in results}, "eval_opened": False})
    summary = summarize_run(out)
    write_summary(summary, out / "observability/joint.summary.json")
    counts = {arm: sum(not row["run_failure"] for row in results[arm][0]) for arm in results}
    joint_complete = sum(all(not results[a][0][i]["run_failure"] for a in results) for i in range(len(rows)))
    final = {"version": VERSION, "cases": len(rows), "complete": counts, "joint_complete": joint_complete,
             "stop_reasons": list(dict.fromkeys(control.stop_reasons)), "budget": control.budget.snapshot(),
             "finished_at": now(), "real_eval_accessed": False, "formal_scoring": False}
    if continuation is not None:
        final["engineering_continuation"] = continuation
    write_json(out / "joint-summary.json", final)
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-list", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget-ref", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recover-from", type=Path, help="New engineering run recovering the documented httpx2 defect; carries all prior costs")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute or args.dev_list.suffix != ".jsonl" or any(p.lower() == "sealed" or p.lower().startswith("eval.") for p in args.dev_list.parts):
        parser.error("Explicit execute and DEV JSONL required")
    print(json.dumps(run_joint(read_rows(args.dev_list), args.out, model=args.model, execute=True, resume=args.resume,
        recover_from=args.recover_from,
        budget_ref=json.loads(args.budget_ref.read_text(encoding="utf-8"))), ensure_ascii=False))


if __name__ == "__main__":
    main()
