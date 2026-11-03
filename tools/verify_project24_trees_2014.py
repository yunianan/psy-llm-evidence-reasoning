"""Offline engine/graph verification of the new source-derived 24-label forest.

Scripted answers exercise the graph. They are not model predictions or evidence
of diagnostic accuracy. No network, credentials, billing, or clinical scoring.
"""
from __future__ import annotations

import argparse
from collections import deque
from itertools import combinations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_dtree.engine import DecisionTreeRunner, PAIRWISE_POLICY, RetryPolicy
from paper_dtree.entries import EntryProfile
from paper_dtree.models import Response, load_bundle
from tools.extract_project24_evidence import DATA


class ControlledTransport:
    """Deterministic test responses, independent of removed baseline tools."""
    def __init__(self, bundle, selected, answers=None, *, entry_profile=None):
        self.bundle, self.selected = bundle, set(selected)
        self.profile, self.calls = entry_profile, []
        self.question_answers = {}
        for tree in bundle.trees:
            for key, node in tree.nodes.items():
                if node.calculation is None and (tree.id, key) in (answers or {}):
                    value = answers[tree.id, key]
                    if node.question in self.question_answers and self.question_answers[node.question] != value:
                        raise ValueError("Conflicting scripted answers to identical questions")
                    self.question_answers[node.question] = value

    def complete(self, request):
        prompt = request.messages[1]["content"]
        self.calls.append(prompt)
        marker = "candidate diagnoses for this patient: "
        if marker in prompt:
            for a, b in combinations(self.bundle.diagnoses, 2):
                if any(f"{marker}{x} and {y}. We are interested" in prompt for x, y in [(a, b), (b, a)]):
                    return Response(json.dumps([a, b]), model="controlled-only")
            raise AssertionError("Unknown scripted pairwise prompt")
        question = prompt.rsplit("The question is as follows: ", 1)[1][:-1]
        if self.profile:
            for entry in self.profile.entries:
                if question == entry.question:
                    return Response("yes" if entry.id in self.selected else "no", model="controlled-only")
        else:
            for tree in self.bundle.trees:
                if question == tree.screening_question:
                    return Response("yes" if tree.id in self.selected else "no", model="controlled-only")
        return Response("yes" if self.question_answers.get(question, False) else "no", model="controlled-only")


def local_paths(tree):
    paths, pending = {}, deque((root, []) for root in tree.roots)
    while pending:
        key, path = pending.popleft()
        if key in paths:
            continue
        paths[key] = path
        node = tree.nodes[key]
        for answer, branch in [(True, node.yes), (False, node.no)]:
            pending.extend((dest, [*path, (key, answer)]) for dest in branch.next)
    if set(paths) != set(tree.nodes):
        raise AssertionError(f"Unreachable nodes in {tree.id}")
    return paths


def adhd_witnesses(tree):
    """Find executable branch witnesses respecting the deterministic counters."""
    witnesses = {}
    for adult in (False, True):
        threshold = 5 if adult else 6
        for a1 in range(10):
            for a2 in range(10):
                pending, seen = deque((root, []) for root in tree.roots), set()
                while pending:
                    key, path = pending.popleft()
                    if key in seen:
                        continue
                    seen.add(key)
                    node = tree.nodes[key]
                    if key == "age_17_or_over":
                        choices = [adult]
                    elif key.startswith(("a1_", "a2_")) and key.rsplit("_", 1)[-1].isdigit():
                        domain, number = key.split("_")
                        choices = [int(number) <= (a1 if domain == "a1" else a2)]
                    elif node.calculation:
                        domain = node.calculation["symptom_nodes"][0].split("_")[0]
                        choices = [(a1 if domain == "a1" else a2) >= threshold]
                    else:
                        choices = [True, False]
                    for answer in choices:
                        candidate = [*path, (key, answer)]
                        witnesses.setdefault((key, answer), candidate)
                        branch = node.yes if answer else node.no
                        pending.extend((dest, candidate) for dest in branch.next)
    if set(witnesses) != {(key, answer) for key in tree.nodes for answer in (True, False)}:
        raise AssertionError(f"ADHD tree has unreachable branches: {set((key, answer) for key in tree.nodes for answer in (True, False)) - set(witnesses)}")
    return witnesses


def run_scripted(bundle, selected, answers=None, profile=None):
    events = []
    transport = ControlledTransport(bundle, selected, answers, entry_profile=profile)
    result = DecisionTreeRunner(bundle, transport, model="scripted-engineering-only-no-api",
                                pairwise_policy=PAIRWISE_POLICY, retry=RetryPolicy(max_attempts=1, initial_seconds=0),
                                entry_profile=profile, record=events.append).run(
        "Engineering fixture: each yes/no response is scripted. This is not a clinical vignette.",
        case_id="project24-offline-graph", method="dt")
    return result, events


def verify(bundle=None):
    bundle = bundle or load_bundle(DATA / "refined_bundle.json")
    profile = EntryProfile.load(DATA / "entry_profile.json")
    probes, endpoints = [], []
    for tree in bundle.trees:
        paths = (adhd_witnesses(tree) if tree.id == "p24.adhd" else
                 {(key, answer): [*path, (key, answer)] for key, path in local_paths(tree).items() for answer in [True, False]})
        for (key, answer), path in paths.items():
                script = {(tree.id, node): value for node, value in path}
                result, _ = run_scripted(bundle, [tree.id], script)
                trace = [r for r in result["tree_trace"] if r["tree_id"] == tree.id and r["node_id"] == key]
                assert len(trace) == 1 and trace[0]["answer"] == ("yes" if answer else "no"), (tree.id, key, answer)
                assert result["status"] == "success"
                assert not result["pending_referrals"] and not result["pending_entry_trees"]
                branch = tree.nodes[key].yes if answer else tree.nodes[key].no
                assert set(branch.diagnoses) <= set(result["candidates_before_pairwise"])
                assert set(result["raw_diagnoses"]) <= set(bundle.diagnoses)
                probes.append({"tree_id": tree.id, "node_id": key, "answer": answer, "status": "passed"})
                for name in branch.diagnoses:
                    endpoints.append({"name": name, "tree_id": tree.id, "node_id": key,
                                      "script": {n: value for n, value in path}})
    assert {r["name"] for r in endpoints} == set(bundle.diagnoses)
    result, events = run_scripted(bundle, [e.id for e in profile.entries], profile=profile)
    requests = [e for e in events if e["event"] == "request_attempt"]
    expected = list(dict.fromkeys(t for e in profile.entries for t in e.tree_ids))
    assert len(expected) == len(bundle.trees)
    assert [r["stage"] for r in requests[:24]] == ["screening"] * 24
    assert [r["tree_id"] for r in result["tree_trace"][:19]] == expected
    assert len(result["selected_groups"]) == 9
    assert not result["pending_entry_trees"] and not result["pending_referrals"]
    return {"bundle_id": bundle.id, "network_calls": 0, "model_calls": 0,
            "method": "scripted graph/engine fixtures; not clinical or LLM validation",
            "tree_count": len(bundle.trees), "node_count": sum(len(t.nodes) for t in bundle.trees),
            "branch_probes_passed": len(probes), "labels_with_reachable_emission": len({r['name'] for r in endpoints}),
            "entry_count": 24, "group_count": 9, "all_screens_before_nodes": True,
            "shared_frontier_verified": True, "outside_scope_outputs": [],
            "label_endpoints": endpoints, "probes": probes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA / "offline_verification.json")
    args = parser.parse_args()
    report = verify()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"label_endpoints", "probes"}}, ensure_ascii=False, indent=2))
