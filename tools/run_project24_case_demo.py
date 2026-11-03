"""Run one preselected MentalBench development case through the current model stage."""
import json
import os
from collections import Counter
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import dotenv_values
from paper_dtree.__main__ import DEFAULT_TREE_BUNDLE
from paper_dtree.engine import DecisionTreeRunner, PAIRWISE_POLICY, RetryPolicy
from paper_dtree.entries import DEFAULT_ENTRY_PROFILE, EntryProfile
from paper_dtree.models import load_bundle
from paper_dtree.transport import OpenAICompatibleTransport

CASE_ID = "mb-000676"
DATA = ROOT.parent / "external/MentalBench/resources/dataset/开发集"
OUT = ROOT / "output/paper_dtree" / (sys.argv[1] if len(sys.argv) > 1 else "20261102_mentalbench_mb000676")

def select(path):
    with path.open(encoding="utf-8") as stream:
        return next(json.loads(line) for line in stream if json.loads(line)["case_id"] == CASE_ID)

def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def main():
    case, reference = select(DATA / "病例输入.jsonl"), select(DATA / "参考答案.jsonl")
    env = dotenv_values(ROOT / ".env")
    def setting(key): return os.environ.get(key) or env.get(key) or ""
    key = setting("PAPER_DT_API_KEY") or setting("LLM_API_KEY")
    endpoint = setting("PAPER_DT_BASE_URL") or setting("LLM_BASE_URL")
    model = setting("PAPER_DT_MODEL") or setting("L3_REASONER_MODEL") or setting("LLM_MODEL")
    if not all((key, endpoint, model)): raise RuntimeError("Missing API key, endpoint or model")
    OUT.mkdir(parents=True, exist_ok=False)
    bundle, profile = load_bundle(DEFAULT_TREE_BUNDLE), EntryProfile.load(DEFAULT_ENTRY_PROFILE)
    write("registration.json", {"case_id": CASE_ID, "source": str(DATA), "reference_kind": reference["reference_kind"],
          "model": model, "method": "dt", "bundle_id": bundle.id, "entry_profile_id": profile.id,
          "pairwise_policy": PAIRWISE_POLICY, "max_attempts": 5,
          "budget_policy": "monitor_and_notify_not_stop", "reference_budget": None,
          "price": "unknown; token usage recorded separately", "independent_runs": 1})
    write("case.json", case)
    write("reference.json", reference)
    (OUT / "bundle_snapshot.json").write_text(DEFAULT_TREE_BUNDLE.read_text(encoding="utf-8"), encoding="utf-8")
    (OUT / "entry_profile_snapshot.json").write_text(DEFAULT_ENTRY_PROFILE.read_text(encoding="utf-8"), encoding="utf-8")
    actual = OpenAICompatibleTransport(api_key=key, base_url=endpoint, timeout=60)
    class SafeTransport:
        def complete(self, request):
            try: return actual.complete(request)
            except Exception as exc:
                raise RuntimeError(f"model_request_failed type={type(exc).__name__} http_status={getattr(exc, 'status_code', None)}") from None
    events, start = [], time.monotonic()
    with (OUT / "attempts.jsonl").open("x", encoding="utf-8") as journal:
        def record(event):
            event = {**event, "elapsed_seconds": round(time.monotonic() - start, 3)}
            events.append(event)
            journal.write(json.dumps(event, ensure_ascii=False) + "\n")
            journal.flush()
            if event["event"] == "request_attempt":
                print(f"{len([e for e in events if e['event']=='request_attempt'])} {event['stage']} {event['status']}", flush=True)
        runner = DecisionTreeRunner(bundle, SafeTransport(), model=model, pairwise_policy=PAIRWISE_POLICY,
                 retry=RetryPolicy(max_attempts=5, initial_seconds=1, maximum_seconds=8), entry_profile=profile, record=record)
        try:
            result = runner.run(case["question"], case_id=CASE_ID, method="dt")
            write("model_result.json", result)
        except Exception as exc:
            write("failure.json", {"type": type(exc).__name__, "message": str(exc),
                                   "note": "Model stage incomplete; no final prediction or score."})
        finally:
            requests = [e for e in events if e.get("event") == "request_attempt"]
            responses = [e["response"] for e in requests if "response" in e]
            summary = {"requests": len(requests), "successes": sum(e["status"] == "success" for e in requests),
                       "failures": sum(e["status"] == "failed" for e in requests),
                       "by_stage": dict(Counter(e["stage"] for e in requests)),
                       "input_tokens": sum(r["input_tokens"] for r in responses),
                       "output_tokens": sum(r["output_tokens"] for r in responses),
                       "reference_cost": None, "actual_bill": None, "seconds": round(time.monotonic()-start, 1)}
            write("usage.json", summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)

if __name__ == "__main__": main()
