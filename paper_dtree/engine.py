"""Figure 1 stage 2 model executor, including the Figure 2 DT subworkflow."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from itertools import combinations
import json
import math
import time
from typing import Callable, Protocol

from . import prompts
from .models import Bundle, Request, Response
from .entries import EntryProfile


class Transport(Protocol):
    def complete(self, request: Request) -> Response: ...


class RunFailed(RuntimeError):
    """A request failure is not a negative clinical answer or an empty prediction."""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int | None = None
    initial_seconds: float = 1.0
    maximum_seconds: float = 60.0

    def __post_init__(self):
        if self.max_attempts is not None and (type(self.max_attempts) is not int or self.max_attempts < 1):
            raise ValueError("max_attempts must be positive or None (paper retry-until-valid)")
        if (not math.isfinite(self.initial_seconds) or not math.isfinite(self.maximum_seconds)
                or self.initial_seconds < 0 or self.maximum_seconds < self.initial_seconds):
            raise ValueError("Invalid exponential backoff")


# The paper does not specify reduction/order/conflict handling. No silent default.
PAIRWISE_POLICY = "all_pairs_union_eliminations_v1"


class DecisionTreeRunner:
    def __init__(self, bundle: Bundle, transport: Transport, *, model: str,
                 pairwise_policy: str, retry: RetryPolicy | None = None,
                 record: Callable[[dict], None] | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 entry_profile: EntryProfile | None = None):
        bundle.validate_graph()
        if not isinstance(model, str) or not model.strip():
            raise ValueError("An explicit model snapshot is required")
        if pairwise_policy != PAIRWISE_POLICY:
            raise ValueError(f"Explicit reconstruction policy required: {PAIRWISE_POLICY}")
        self.bundle, self.transport, self.model = bundle, transport, model
        self.policy, self.retry = pairwise_policy, retry or RetryPolicy()
        self.record, self.sleep = record or (lambda event: None), sleep
        if entry_profile is not None and entry_profile.standard != bundle.standard:
            raise ValueError("Entry profile and tree bundle standards must match")
        self.entry_profile = entry_profile

    @staticmethod
    def _binary(raw: str) -> bool:
        normalized = raw.strip().lower()
        if normalized not in {"yes", "no"}:
            raise ValueError("Expected only yes or no; unknown is not silently converted to no")
        return normalized == "yes"

    @staticmethod
    def _diagnoses(raw: str) -> list[str]:
        value = json.loads(raw)
        if not isinstance(value, dict) or set(value) != {"diagnoses"}:
            raise ValueError("Expected exactly a diagnoses object")
        return DecisionTreeRunner._list(value["diagnoses"])

    @staticmethod
    def _list(value) -> list[str]:
        if not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value):
            raise ValueError("Expected a JSON list of nonempty diagnosis strings")
        if len(value) != len(set(value)):
            raise ValueError("Duplicate diagnoses in response")
        return value

    def _ask(self, text: str, parse, *, stage: str, **context):
        # Reconstruct a fresh message pair on EVERY attempt; never add previous
        # questions/responses, JSON repair instructions or hidden project context.
        attempt = 0
        while True:
            attempt += 1
            request = Request(self.model, (
                {"role": "system", "content": prompts.system(self.bundle.standard)},
                {"role": "user", "content": text},
            ))
            event = {"event": "request_attempt", "stage": stage, **context,
                     "attempt": attempt, "request": request.to_dict()}
            try:
                response = self.transport.complete(request)
                if not isinstance(response, Response):
                    raise TypeError("Transport must return a Response")
                event["response"] = asdict(response)
                parsed = parse(response.text)
            except Exception as exc:
                event.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                self.record(event)
                if self.retry.max_attempts is not None and attempt >= self.retry.max_attempts:
                    raise RunFailed(f"{stage} exhausted {attempt} attempts") from exc
                self.sleep(min(self.retry.maximum_seconds,
                               self.retry.initial_seconds * 2 ** min(attempt - 1, 20)))
                continue
            event.update(status="success", parsed=parsed)
            self.record(event)
            return parsed

    def run(self, case: str, *, case_id: str, method: str = "dt") -> dict:
        if not isinstance(case, str) or not case.strip() or not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("A nonempty raw case vignette and case_id are required")
        if method not in {"base", "dt", "dt_raw"}:
            raise ValueError("method must be base, dt or dt_raw")
        if method == "dt" and self.bundle.stage != "refined":
            raise ValueError("Paper DT requires a refined bundle; use dt_raw for the explicit raw-tree arm")
        if method == "dt_raw" and self.bundle.stage != "raw":
            raise ValueError("dt_raw requires a raw bundle")
        result = {
            "case_id": case_id, "method": method, "bundle_id": self.bundle.id,
            "standard": self.bundle.standard, "bundle_stage": self.bundle.stage,
            "coverage": self.bundle.coverage, "model": self.model,
            "referral_policy": self.bundle.referral_policy,
            "pairwise_policy": self.policy, "pairwise_case_wrapper": "prepend_full_case_v1",
            "pairwise_label_binding": "casefold_outer_whitespace_exact_presented_labels_v1",
            "traversal_policy": "screen_all_then_shared_node_frontier_v3",
            "entry_profile_id": self.entry_profile.id if self.entry_profile else "bundle_tree_screening",
            "selected_entry_ids": [], "selected_groups": [], "pending_entry_trees": [],
            "max_attempts": self.retry.max_attempts,
            "status": "running", "screening": [], "tree_trace": [],
            "candidates_before_pairwise": [], "candidate_sources": {},
            "pairwise_comparisons": [], "diagnoses_after_pairwise": [],
            "pending_referrals": [], "referral_trace": [], "non_diagnostic_outcomes": [],
            "postprocessing": "pending_simplification_and_reconciliation",
        }
        self.record({"event": "run_started", "case_id": case_id, "method": method,
                     "model": self.model, "bundle_id": self.bundle.id,
                     "standard": self.bundle.standard, "bundle_stage": self.bundle.stage,
                     "coverage": self.bundle.coverage, "refinements": list(self.bundle.refinements),
                     "referral_policy": self.bundle.referral_policy,
                     "pairwise_policy": self.policy, "pairwise_case_wrapper": "prepend_full_case_v1",
                     "pairwise_label_binding": "casefold_outer_whitespace_exact_presented_labels_v1",
                     "traversal_policy": "screen_all_then_shared_node_frontier_v3",
                     "entry_profile_id": self.entry_profile.id if self.entry_profile else "bundle_tree_screening",
                     "max_attempts": self.retry.max_attempts})
        try:
            if method == "base":
                result["raw_diagnoses"] = self._ask(
                    prompts.direct(case, self.bundle.standard), self._diagnoses,
                    stage="base", case_id=case_id)
            else:
                self._traverse(case, case_id, result)
                if result["pending_referrals"] or result["pending_entry_trees"]:
                    # An unavailable selected tree/referral is neither a negative
                    # answer nor a completed diagnostic model response.
                    result["status"] = "awaiting_tree_content"
                    self.record({"event": "run_finished", "result": result})
                    return result
                candidates = result["candidates_before_pairwise"]
                eliminated = set()
                for a, b in combinations(candidates, 2):
                    def parse(raw):
                        labels = self._list(json.loads(raw))
                        kept = []
                        for label in labels:
                            # Table 1 itself shows lowercase diagnosis examples.
                            # Bind case/outer-space variants to the two presented
                            # names, without introducing fuzzy clinical matching.
                            matches = [name for name in (a, b) if name.strip().casefold() == label.strip().casefold()]
                            if len(matches) != 1 or matches[0] in kept:
                                raise ValueError("Pairwise labels must uniquely identify the presented diagnoses")
                            kept.append(matches[0])
                        if not kept:
                            raise ValueError("Pairwise response must retain one or both presented diagnoses")
                        return kept
                    kept = self._ask(prompts.pairwise(case, a, b, self.bundle.standard), parse,
                                     stage="pairwise", case_id=case_id, pair=[a, b])
                    deleted = [d for d in (a, b) if d not in kept]
                    eliminated.update(deleted)
                    result["pairwise_comparisons"].append({"pair": [a, b], "kept": kept, "deleted": deleted})
                final = [d for d in candidates if d not in eliminated]
                result["diagnoses_after_pairwise"] = final
                result["raw_diagnoses"] = final
                result["pairwise_all_eliminated"] = bool(candidates) and not final
            result["status"] = "success"
        except BaseException as exc:
            if getattr(exc, "is_followup_pause", False):
                result.update(status="awaiting_patient", follow_up=exc.task)
                self.record({"event": "run_paused", "result": result})
                return result
            result.update(status="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                          error_type=type(exc).__name__, error=str(exc))
            self.record({"event": "run_finished", "result": result})
            raise
        self.record({"event": "run_finished", "result": result})
        return result

    def _traverse(self, case: str, case_id: str, result: dict) -> None:
        trees = {tree.id: tree for tree in self.bundle.trees}
        selected = []
        # Complete all independent screens first. Group metadata does not add
        # another clinical gate or automatically turn a label into a diagnosis.
        if self.entry_profile is not None:
            for entry in self.entry_profile.entries:
                answer = self._ask(prompts.question(case, entry.question), self._binary,
                                   stage="screening", case_id=case_id, entry_id=entry.id,
                                   group_id=entry.group_id, source_refs=list(entry.source_refs))
                result["screening"].append({"entry_id": entry.id, "group_id": entry.group_id,
                                             "disorder_id": entry.disorder_id, "label": entry.label,
                                             "answer": "yes" if answer else "no", "tree_ids": list(entry.tree_ids)})
                if not answer:
                    continue
                result["selected_entry_ids"].append(entry.id)
                if entry.group_id not in result["selected_groups"]:
                    result["selected_groups"].append(entry.group_id)
                for target in entry.tree_ids:
                    if target in trees:
                        if target not in selected:
                            selected.append(target)
                    else:
                        result["pending_entry_trees"].append({
                            "entry_id": entry.id, "group_id": entry.group_id, "disorder_id": entry.disorder_id,
                            "target_tree_id": target, "source_refs": list(entry.source_refs),
                            "reason": "selected_entry_tree_not_yet_transcribed"})
        else:
            for tree in self.bundle.trees:
                answer = self._ask(prompts.question(case, tree.screening_question), self._binary,
                                   stage="screening", case_id=case_id, tree_id=tree.id,
                                   source=asdict(tree.screening_source))
                result["screening"].append({"tree_id": tree.id, "answer": "yes" if answer else "no"})
                if answer:
                    selected.append(tree.id)
        visited, normal_by_tree, fallback_by_tree = set(), {}, {}

        def add(diagnosis, source):
            if diagnosis not in result["candidate_sources"]:
                result["candidates_before_pairwise"].append(diagnosis)
                result["candidate_sources"][diagnosis] = []
            if source not in result["candidate_sources"][diagnosis]:
                result["candidate_sources"][diagnosis].append(source)

        # All applicable trees enter one ready-node queue after screening ends.
        # A tree's next node requires its preceding answer, but no tree has to
        # finish before another tree can advance. Referrals use the same queue.
        queue = deque((tree_id, root) for tree_id in selected for root in trees[tree_id].roots)
        while queue:
            tree_id, node_id = queue.popleft()
            if (tree_id, node_id) in visited:
                continue
            visited.add((tree_id, node_id))
            node = trees[tree_id].nodes[node_id]
            calculation = None
            if node.calculation:
                prior = {r["node_id"]: r for r in result["tree_trace"] if r["tree_id"] == tree_id}
                inputs = node.calculation["symptom_nodes"]
                age = prior[node.calculation["age_node"]]
                threshold = 5 if age["answer"] == "yes" else 6
                count = sum(prior[key]["answer"] == "yes" for key in inputs)
                answer = count >= threshold
                calculation = {"kind": node.calculation["kind"], "symptom_count": count,
                               "threshold": threshold, "age_17_or_over": age["answer"] == "yes",
                               "inputs": [{"node_id": key, "answer": prior[key]["answer"]} for key in inputs],
                               "age_node": node.calculation["age_node"]}
            else:
                answer = self._ask(prompts.question(case, node.question), self._binary,
                                   stage="node", case_id=case_id, tree_id=tree_id,
                                   node_id=node_id, source=asdict(node.source))
            branch = node.yes if answer else node.no
            record = {"tree_id": tree_id, "node_id": node_id, "question": node.question,
                      "answer": "yes" if answer else "no", "branch": asdict(branch),
                      "source": asdict(node.source)}
            evidence_hook = getattr(self, "_node_evidence", None)
            if evidence_hook is not None:
                if calculation:
                    related = [prior[calculation["age_node"]], *(prior[k] for k in inputs)]
                    record["patient_evidence"] = [e for r in related for e in r.get("patient_evidence", [])]
                else:
                    record["patient_evidence"] = evidence_hook(tree_id, node_id)
            if calculation:
                record["calculation"] = calculation
            result["tree_trace"].append(record)
            self.record({"event": "branch_taken", "case_id": case_id, **record})
            for diagnosis in branch.diagnoses:
                normal_by_tree.setdefault(tree_id, set()).add(diagnosis)
                add(diagnosis, {"tree_id": tree_id, "node_id": node_id, "fallback": False})
            for diagnosis in branch.fallback_diagnoses:
                fallback_by_tree.setdefault(tree_id, []).append((diagnosis, node_id))
            for target in branch.pending_referrals:
                result["pending_referrals"].append({
                    "tree_id": tree_id, "node_id": node_id, "target_tree_id": target,
                    "source": asdict(node.source), "reason": "source_referral_tree_not_yet_transcribed"})
            for text in branch.observations:
                result["non_diagnostic_outcomes"].append({
                    "tree_id": tree_id, "node_id": node_id, "text": text, "source": asdict(node.source)})
            queue.extend((tree_id, dest) for dest in branch.next)
            for target in branch.referrals:
                result["referral_trace"].append({
                    "tree_id": tree_id, "node_id": node_id, "target_tree_id": target,
                    "status": "already_visited" if all((target, root) in visited for root in trees[target].roots)
                    else "queued", "source": asdict(node.source)})
                queue.extend((target, root) for root in trees[target].roots)
        # Defer catch-all leaves until all paths of their owning tree finish.
        for tree_id, emissions in fallback_by_tree.items():
            if not normal_by_tree.get(tree_id):
                for diagnosis, node_id in emissions:
                    add(diagnosis, {"tree_id": tree_id, "node_id": node_id, "fallback": True})
