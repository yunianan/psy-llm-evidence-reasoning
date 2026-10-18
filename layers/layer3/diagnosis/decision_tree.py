"""DSM-5 candidate subgraph. Traversal is code; clinical facts are LLM judgments.

This is a project implementation, not the paper authors' source code or a
reproduction of all 29 handbook trees. Original MentalKG remains read-only.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Callable

from shared.disorder_catalog import SUPPORTED_DISORDERS, candidate_knowledge, disorder_metadata
from .schemas import (
    CandidateDecisionPath, CandidateGeneration, DiagnosticCandidate,
    MissingDimension, SymptomMatch, TreeDecisionRecord, TreeNodeJudgment,
)

TREE_FILE = Path(__file__).resolve().parents[3] / "knowledge/dsm5_decision_tree/candidate_paths.json"
NodeEvaluator = Callable[[list[dict[str, Any]]], list[TreeNodeJudgment]]


@lru_cache
def load_candidate_graph() -> dict[str, Any]:
    graph = json.loads(TREE_FILE.read_text(encoding="utf-8"))
    if graph["standard"] != "DSM-5":
        raise ValueError("Decision graph must use DSM-5")
    nodes = graph["nodes"]
    if set(graph["paths"]) != set(SUPPORTED_DISORDERS):
        raise ValueError("Decision graph must cover all 24 project labels")
    for label, path in graph["paths"].items():
        if not path["screens"] or not path["anchors"]:
            raise ValueError(f"Missing symptom entry for {label}")
        for node_id in [*path["screens"], *path["anchors"],
                        *[rule["node_id"] for rule in path["guards"]]]:
            if node_id not in nodes or not nodes[node_id]["source_refs"]:
                raise ValueError(f"Unreferenced node: {node_id}")
    return graph


def _symptom_pool(value: Any) -> set[str]:
    if isinstance(value, dict):
        result = set(value.get("symptom_pool", []))
        for child in value.values():
            result |= _symptom_pool(child)
        return result
    if isinstance(value, list):
        return set().union(*(_symptom_pool(child) for child in value)) if value else set()
    return set()


def criterion_nodes(label: str) -> dict[str, dict[str, Any]]:
    """Expand source conditions once in the same form used by traversal and reporting."""
    knowledge = candidate_knowledge(label)
    entries = []
    for key, value in knowledge["criteria"]["required_criteria"].items():
        if isinstance(value, list):
            entries.extend((f"{key}/{i}", item, {}, True) for i, item in enumerate(value))
        elif isinstance(value, dict):
            mandatory = value.get("relation") != "include"
            entries.extend((f"{key}/{part}", condition, value, mandatory)
                           for part, condition in value.items()
                           if part not in {"relation", "symptom_pool"})
        else:
            entries.append((key, value, {}, True))
    result = {}
    for key, value, group, mandatory in entries:
        node_id = f"criterion.{label}.{key}"
        part = key.rsplit("/", 1)[-1]
        result[node_id] = {
            "stage": "criterion", "question": f"患者相关病程是否满足来源字段 {key} 的原始条件？",
            "follow_up": f"请补充与 {key} 对应的症状、时间和情境信息。",
            "source_refs": [knowledge["source_ref"] + f"/required_criteria/{key}"],
            "source_condition": value, "source_group": group, "mandatory": mandatory,
            "symptom_pool": group.get("symptom_pool", []) if part in {
                "min_count", "must_include_one_of", "must_include_all"
            } else [],
            "disorders": [label],
        }
    return result


class DecisionTreeCandidateGenerator:
    def __init__(self, evaluate: NodeEvaluator) -> None:
        self.evaluate = evaluate
        self.graph = load_candidate_graph()

    def generate(self) -> CandidateGeneration:
        nodes = {key: dict(value) for key, value in self.graph["nodes"].items()}
        cache: dict[str, TreeDecisionRecord] = {}
        missing: dict[str, MissingDimension] = {}

        def assess(node_ids: list[str]) -> None:
            wanted = list(dict.fromkeys(n for n in node_ids if n not in cache))
            if not wanted:
                return
            # Independent questions at one frontier share an API batch, not a
            # conversation history. Branch selection still happens in code.
            requests = [{"node_id": n, **nodes[n]} for n in wanted]
            answers = self.evaluate(requests)
            by_id = {item.node_id: item for item in answers}
            if len(by_id) != len(answers) or set(by_id) != set(wanted):
                raise ValueError("Node answers must match the requested frontier exactly")
            for node_id in wanted:
                answer = by_id[node_id]
                definition = nodes[node_id]
                if answer.status != "pending" and not answer.evidence_refs:
                    raise ValueError("Known node judgments require patient evidence")
                if not set(answer.symptom_ids) <= set(definition.get("symptom_pool", [])):
                    raise ValueError("Node mapped symptoms outside its source pool")
                if definition["stage"] == "anchor" and answer.status == "met" and not answer.symptom_ids:
                    raise ValueError("A positive anchor requires a mapped patient symptom")
                if answer.status != "met" and answer.symptom_ids:
                    raise ValueError("Only supported nodes may supply positive symptoms")
                record = TreeDecisionRecord(
                    **answer.model_dump(), question=definition["question"],
                    source_refs=definition["source_refs"], stage=definition["stage"],
                )
                cache[node_id] = record
                if answer.status == "pending":
                    missing[node_id] = MissingDimension(
                        field=f"decision_tree.{node_id}",
                        purpose=definition["question"],
                        suggested_question=definition["follow_up"],
                    )

        paths = self.graph["paths"]
        coverage = {label: "not_triggered" for label in SUPPORTED_DISORDERS}
        assess([n for path in paths.values() for n in path["screens"]])
        active = {
            label: path for label, path in paths.items()
            if any(cache[n].status != "not_met" for n in path["screens"])
        }
        assess([n for path in active.values() for n in path["anchors"]])
        active = {
            label: path for label, path in active.items()
            if any(cache[n].status == "met" for n in path["anchors"])
        }
        for label in active:
            coverage[label] = "anchor_supported"
        for label, path in paths.items():
            if label not in active and any(n in cache and cache[n].status == "pending" for n in path["anchors"]):
                coverage[label] = "anchor_pending"
            elif label not in active and any(cache[n].status == "pending" for n in path["screens"]):
                coverage[label] = "entry_pending"

        # Each guard is a sourced exclusion / subtype discriminator. Unknown
        # keeps a branch alive. Explicit contrary evidence prunes that branch.
        for depth in range(max((len(p["guards"]) for p in active.values()), default=0)):
            assess([p["guards"][depth]["node_id"] for p in active.values()
                    if depth < len(p["guards"])])
            for label, path in list(active.items()):
                if depth >= len(path["guards"]):
                    continue
                rule = path["guards"][depth]
                state = cache[rule["node_id"]].status
                if state != "pending" and state != rule["expected"]:
                    coverage[label] = f"excluded:{rule['node_id']}"
                    del active[label]

        # Full source criteria are checked to produce evidence, not to require
        # a final diagnosis at candidate time. Failed/unknown criteria remain
        # visible for deterministic aggregation and the user's downstream scoring.
        criteria_by_label: dict[str, list[str]] = {}
        for label in active:
            definitions = criterion_nodes(label)
            # Keep per-request generated nodes outside the cached graph.
            nodes.update(definitions)
            criteria_by_label[label] = list(definitions)
        assess([n for ids in criteria_by_label.values() for n in ids])

        candidates = []
        for label, path in active.items():
            screen_ids = [n for n in path["screens"] if cache[n].status != "not_met"]
            anchor_ids = [n for n in path["anchors"] if cache[n].status == "met"]
            ids = list(dict.fromkeys([*screen_ids, *anchor_ids,
                                     *[r["node_id"] for r in path["guards"]],
                                     *criteria_by_label[label]]))
            records = [cache[n] for n in ids]
            pool = _symptom_pool(candidate_knowledge(label)["criteria"])
            mapped = set().union(*(set(r.symptom_ids) for r in records)) & pool
            refs = list(dict.fromkeys(ref for r in records if r.status == "met" for ref in r.evidence_refs))
            provisional = any(r.status != "met" for r in records if r.stage == "criterion" and nodes[r.node_id]["mandatory"]) or any(
                cache[r["node_id"]].status == "pending" for r in path["guards"]
            )
            trace = CandidateDecisionPath(
                path_id=label, tree_refs=path["tree_refs"], provisional=provisional, decisions=records,
            )
            summary = "；".join(f"{r.node_id}={r.status}: {r.rationale}" for r in records)
            candidates.append(DiagnosticCandidate(
                disorder=label, label=disorder_metadata(label)["label"],
                symptom_match=SymptomMatch(matched=len(mapped), total=max(1, len(pool))),
                core_evidence_summary=summary, evidence_refs=refs,
                # Compatibility field: subjective anchor support, not a DSM
                # score or calibrated probability. Downstream ranking is unchanged.
                confidence=max(cache[n].confidence for n in anchor_ids), decision_paths=[trace],
            ))
            coverage[label] = "provisional_candidate" if provisional else "candidate"
        return CandidateGeneration(
            considered_disorders=list(SUPPORTED_DISORDERS), candidates=candidates,
            tree_trace=list(cache.values()), missing_dimensions=list(missing.values()),
            candidate_coverage=coverage,
        )
