"""Run the published CMAJ vignette through the current interactive ADHD tree.

The study's later diagnosis is deliberately excluded from the model input.
The source vignette is paraphrased; it is not a verbatim patient statement.
"""
from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import dotenv_values
from paper_dtree.__main__ import DEFAULT_TREE_BUNDLE
from paper_dtree.engine import PAIRWISE_POLICY, RetryPolicy
from paper_dtree.entries import DEFAULT_ENTRY_PROFILE, EntryProfile
from paper_dtree.followup import FollowUpRunner
from paper_dtree.models import load_bundle
from paper_dtree.transport import OpenAICompatibleTransport

CASE = ("A seven-year-old boy is brought to his family doctor by his parents at the teacher's request. "
        "Since beginning first grade, he has been easily distracted, has difficulty focusing on assignments, "
        "has trouble staying seated in class, and argues with his teacher. His parents observe similar behavior "
        "at home and are concerned that it is interfering with his learning.")
SOURCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC3796598/"
OUT = ROOT / "output/paper_dtree" / ("published_adhd_cmaj_2013_" + datetime.now().strftime("%Y%m%d_%H%M%S"))


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    env = dotenv_values(ROOT / ".env")
    setting = lambda name: os.environ.get(name) or env.get(name) or ""
    key = setting("PAPER_DT_API_KEY") or setting("LLM_API_KEY")
    endpoint = setting("PAPER_DT_BASE_URL") or setting("LLM_BASE_URL")
    model = setting("PAPER_DT_MODEL") or setting("L3_REASONER_MODEL") or setting("LLM_MODEL")
    if not all((key, endpoint, model)):
        raise RuntimeError("Missing model configuration")
    OUT.mkdir(parents=True, exist_ok=False)
    bundle = load_bundle(DEFAULT_TREE_BUNDLE)
    profile = EntryProfile.load(DEFAULT_ENTRY_PROFILE)
    # This is a targeted ADHD-tree demonstration, not a whole-forest evaluation.
    entries = tuple(e for e in profile.entries if "p24.adhd" in e.tree_ids)
    profile = replace(profile, entries=entries)
    save("registration.json", {"source": SOURCE, "publication": "CMAJ 2013;185:1333–1335, doi:10.1503/cmaj.130042",
         "case_input": "initial vignette paraphrase; later author diagnosis withheld", "bundle_id": bundle.id,
         "entry_scope": [e.id for e in entries], "scope": "ADHD entries only, not whole-forest screening",
         "model": model, "method": "tri_state_followup_v1", "model_runs": 1,
         "reference_budget": None, "budget_policy": "monitor_and_notify_not_stop"})
    save("case.json", {"text": CASE, "source": SOURCE, "source_type": "clinician narrative paraphrase, not a patient interview"})
    transport = OpenAICompatibleTransport(api_key=key, base_url=endpoint, timeout=60)
    start = time.monotonic()
    events = []
    def record(event):
        if event.get("event") in {"request_attempt", "branch_taken", "run_paused"}:
            events.append(event)
        if event.get("event") == "request_attempt":
            print(event.get("stage"), event.get("node_id", event.get("entry_id", "")), event.get("status"), flush=True)
    runner = FollowUpRunner(bundle, transport, model=model, pairwise_policy=PAIRWISE_POLICY,
                            retry=RetryPolicy(max_attempts=3, initial_seconds=1, maximum_seconds=4),
                            entry_profile=profile, record=record)
    # Evidence verification requires messages tagged patient; the paraphrase is
    # passed as a simulated input message, not represented as a real quotation.
    runner.set_patient_evidence([{"role": "patient", "message_id": "case_paraphrase", "text": CASE}], [])
    try:
        result = runner.run(CASE, case_id="published_cmaj_2013_adhd", method="dt")
        save("result.json", result)
        print("result status:", result["status"], "follow-up node:", result.get("follow_up", {}).get("node_id"), flush=True)
    except Exception as exc:
        save("failure.json", {"type": type(exc).__name__, "message": "Model request or traversal failed; no diagnostic result."})
        print("run failed:", type(exc).__name__, flush=True)
    finally:
        save("events.json", events)
        requests = [e for e in events if e.get("event") == "request_attempt"]
        responses = [e["response"] for e in requests if "response" in e]
        save("usage.json", {"request_attempts": len(requests), "successes": sum(e.get("status") == "success" for e in requests),
             "input_tokens": sum(r.get("input_tokens", 0) for r in responses), "output_tokens": sum(r.get("output_tokens", 0) for r in responses),
             "reference_cost": None, "actual_bill": None, "elapsed_seconds": round(time.monotonic() - start, 1)})


if __name__ == "__main__":
    main()
