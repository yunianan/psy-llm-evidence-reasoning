"""Compare P2.1 before/after using read-only G1 nodes; never read gold or call APIs."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
import sys
from unittest.mock import patch

sys.dont_write_bytecode = True
WORKSPACE = Path(__file__).resolve().parents[2]
ENGINE = WORKSPACE / "L1-L5系统"
sys.path.insert(0, str(ENGINE))

from layers.layer3.diagnosis.clinicr import CLINICR
from layers.layer3.diagnosis.g1_cache import CachedG1Reasoner, G1NodeCache
from layers.layer3.diagnosis.schemas import CandidateGeneration, DiagnosticCandidate
from layers.layer3.diagnosis.verifier import IndependentVerifier
from layers.layer3.llm import client
from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider
from tests.l3_test_double import ControlledRewardModelVerifier
from tests.tree_test_support import controlled_generation
from tools.prediction_contract import normalize_for_protocol

BASE = WORKSPACE / "phase1/p1.4-dev15-case2"
RUN = BASE / "runs/p1.4-dev15-case2-recovery-v1"
INPUTS = BASE / "dev15.inputs.jsonl"
PREDICTIONS = RUN / "predictions/A1.jsonl"


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_files():
    files = [INPUTS, PREDICTIONS, *sorted((RUN / "g1-nodes").glob("*.json")),
             ENGINE / "layers/layer3/diagnosis/verifier.py",
             ENGINE / "layers/layer3/diagnosis/schemas.py",
             ENGINE / "layers/layer3/llm/client.py",
             ENGINE / "knowledge/dsm5_decision_tree/candidate_paths.json",
             WORKSPACE / "phase1/p1.5-closure/g1-freeze-v1/manifest.json"]
    files += sorted(p for p in (WORKSPACE / "step3_reward_verifier").rglob("*")
                    if p.is_file() and p.suffix in {".py", ".md", ".json"})
    return {str(p.relative_to(WORKSPACE)): digest(p) for p in files}


def run_replay():
    provider = JsonDiagnosticKnowledgeProvider()
    inputs, predictions = rows(INPUTS), rows(PREDICTIONS)
    assert len(inputs) == len(predictions) == 15
    assert [r["case_id"] for r in inputs] == [r["case_id"] for r in predictions]
    cache_by_case = {}
    for path in sorted((RUN / "g1-nodes").glob("*.json")):
        envelope = json.loads(path.read_text(encoding="utf-8"))
        cid = envelope["context"]["case_id"]
        assert cid not in cache_by_case
        cache_by_case[cid] = envelope["context"]
    assert set(cache_by_case) == {r["case_id"] for r in inputs}
    counts = {"model_calls": 0, "network_calls": 0, "cache_writes": 0, "real_verifier_calls": 0}

    def forbidden(kind):
        def fail(*args, **kwargs):
            counts[kind] += 1
            raise AssertionError(f"Offline replay attempted {kind}")
        return fail

    class VerifierMustNotRun:
        verify = staticmethod(forbidden("real_verifier_calls"))

    class ObservedReasoner:
        def __init__(self, reasoner):
            self.reasoner = reasoner
            self.generation = None

        def generate_candidates(self, profile):
            self.generation = self.reasoner.generate_candidates(profile)
            return self.generation

    outputs, checks = [], []
    with ExitStack() as stack:
        stack.enter_context(patch.object(client, "ask_llm_stream", forbidden("model_calls")))
        stack.enter_context(patch.object(socket.socket, "connect", forbidden("network_calls")))
        stack.enter_context(patch.object(socket.socket, "connect_ex", forbidden("network_calls")))
        stack.enter_context(patch.object(G1NodeCache, "save", forbidden("cache_writes")))
        for row, historical in zip(inputs, predictions):
            reasoner = ObservedReasoner(CachedG1Reasoner(
                G1NodeCache(RUN / "g1-nodes"), cache_by_case[row["case_id"]],
                knowledge=provider, mode="require"))
            report = CLINICR(reasoner=reasoner, verifier=VerifierMustNotRun(), knowledge=provider).run(row["profile"])
            source = historical["report"]
            expected_generation = CandidateGeneration(
                considered_disorders=source["considered_disorders"],
                candidates=[DiagnosticCandidate.model_validate({k: c[k] for k in DiagnosticCandidate.model_fields if k in c})
                            for c in source["candidates"]],
                tree_trace=source["tree_trace"], missing_dimensions=source["candidate_missing_dimensions"],
                candidate_coverage=source["candidate_coverage"])
            rebuilt = reasoner.generation.model_copy(deep=True)
            rebuilt.candidates = CLINICR._validate_generation(rebuilt, CLINICR._known_evidence_refs(row["profile"]))
            generation_equal = rebuilt.model_dump(mode="json") == expected_generation.model_dump(mode="json")
            core = ("candidates", "checked", "diagnosis", "provisional_diagnosis", "candidate_confidence",
                    "verifier_score", "verifier_decision", "verifier_routing", "verifier_checks",
                    "verifier_fail_reasons", "tree_trace", "candidate_coverage", "evidence_refs")
            core_equal = all(report[k] == source[k] for k in core)
            assert generation_equal and core_equal, row["case_id"]
            assert report["verifier_routing"] == "return_to_l2" and report["diagnosis"]["primary"] is None
            normalized = normalize_for_protocol({"case_id": row["case_id"], "report": {**report, "final_diagnoses": []}})
            outputs.append({"case_id": row["case_id"], "raw_generation": reasoner.generation.model_dump(mode="json"),
                            "report": report, "normalized": normalized})
            checks.append({"case_id": row["case_id"], "historical_generation_equal": generation_equal,
                           "historical_core_equal": core_equal,
                           "historical_summary_equal": report["summary"] == source["summary"],
                           "candidate_count": len(report["candidates"])})

        # Known complete engineering fixtures exercise the original verifier with
        # a controlled scorer. All four scorer arguments are recorded unchanged.
        controlled = []
        for score in (.87, .8, .79, .5, .49):
            generation = controlled_generation(["D001", "D002"], complete_labels=["D001", "D002"],
                                               confidences={"D001": .8, "D002": .8})

            class Reasoner:
                def generate_candidates(self, profile):
                    return generation.model_copy(deep=True)

            class Scorer(ControlledRewardModelVerifier):
                def score_reasoning(self, profile, candidate_without_confidence, assessment, differential_context):
                    self.arguments = deepcopy({"profile": profile, "candidate": candidate_without_confidence,
                                               "assessment": assessment.model_dump(mode="json"),
                                               "differential_context": differential_context})
                    return super().score_reasoning(profile, candidate_without_confidence, assessment, differential_context)

            scorer = Scorer(score)
            report = CLINICR(reasoner=Reasoner(), verifier=IndependentVerifier(model=scorer),
                             knowledge=provider).run({"evidence_nodes": [{"evidence_id": "ev-mood"}]})
            assert len(scorer.calls) == 1
            controlled.append({"score": score, "scorer_arguments": scorer.arguments, "report": report})
    assert not any(counts.values())
    return outputs, controlled, checks, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Evidence output already exists; never overwrite")
    protected_before = protected_files()
    outputs, controlled, checks, counts = run_replay()
    assert protected_files() == protected_before
    evidence = {"version": "p2.1-tie-replay-v1", "cases": outputs, "controlled": controlled,
                "checks": checks, "counts": counts, "protected_files": protected_before,
                "gold_read": False, "EVAL_accessed": False}
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        evidence["comparison"] = {
            "full_business_bytes_equal": canonical(outputs).encode() == canonical(baseline["cases"]).encode(),
            "controlled_verifier_inputs_outputs_equal": canonical(controlled) == canonical(baseline["controlled"]),
            "protected_files_equal": protected_before == baseline["protected_files"],
            "per_case": [{"case_id": a["case_id"], "equal": canonical(a) == canonical(b)}
                         for a, b in zip(outputs, baseline["cases"])],
        }
        assert len(outputs) == len(baseline["cases"])
        assert all(evidence["comparison"][k] for k in (
            "full_business_bytes_equal", "controlled_verifier_inputs_outputs_equal", "protected_files_equal"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "cases": len(outputs), "paths": sum(c["candidate_count"] for c in checks),
                      "controlled_cases": len(controlled), "counts": counts,
                      "comparison": evidence.get("comparison")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
