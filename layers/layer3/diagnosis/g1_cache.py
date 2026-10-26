"""Explicit offline G1 node cache. Candidates are always rebuilt by traversal."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from shared.disorder_catalog import SUPPORTED_DISORDERS
from .decision_tree import DecisionTreeCandidateGenerator, load_candidate_graph
from .schemas import TreeDecisionRecord, TreeNodeJudgment

CACHE_FORMAT = "g1-nodes-v1"


def content_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def knowledge_hash(knowledge) -> str:
    # Explicit reads preserve the intentionally unconnected production boundary.
    return content_hash({"candidates": {label: knowledge.get_candidate(label) for label in SUPPORTED_DISORDERS},
                         "symptoms": knowledge.get_symptoms()})


def cache_context(case_id, profile, *, model, service_url, knowledge_digest, prompt_version,
                  criterion_batching="legacy", static_knowledge_prefix=False, generation_options=None, tree=None):
    graph = tree if tree is not None else load_candidate_graph()
    return {"case_id": case_id, "profile_hash": content_hash(profile), "model": model,
            "service_url": service_url, "knowledge_hash": knowledge_digest,
            "tree_version": graph["version"], "tree_hash": content_hash(graph),
            "prompt_version": prompt_version, "criterion_batching": criterion_batching,
            "static_knowledge_prefix": static_knowledge_prefix,
            "generation_options": generation_options or {},
            "screening_mode": "zero", "anchor_guard_criteria": "full",
            "temperature": 0, "response_format": {"type": "json_object"},
            "node_contract": "only-met-positive-symptoms-v1"}


class G1CacheMiss(RuntimeError):
    pass


class G1CacheIntegrityError(ValueError):
    pass


class G1NodeCache:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def path_for(self, context):
        return self.root / (content_hash(context) + ".json")

    def load(self, context, *, expected_fingerprint=None):
        path = self.path_for(context)
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise G1CacheIntegrityError("G1 cache is not valid UTF-8 JSON") from exc
        if envelope.get("format") != CACHE_FORMAT or envelope.get("context") != context:
            raise G1CacheIntegrityError("G1 cache format/context mismatch")
        if content_hash(envelope["nodes"]) != envelope.get("nodes_hash"):
            raise G1CacheIntegrityError("G1 cache node integrity mismatch")
        try:
            records = [TreeDecisionRecord.model_validate(row) for row in envelope["nodes"]]
        except ValueError as exc:
            raise G1CacheIntegrityError("G1 cached node violates the contract") from exc
        if not records or len({r.node_id for r in records}) != len(records):
            raise G1CacheIntegrityError("G1 cache must contain unique complete traversal records")
        if expected_fingerprint is not None:
            if envelope.get("provenance", {}).get("fingerprints") != [expected_fingerprint]:
                raise G1CacheIntegrityError("G1 cache response fingerprint differs from freeze")
        return envelope

    def save(self, context, records, *, provenance=None):
        nodes = [TreeDecisionRecord.model_validate(r).model_dump(mode="json") for r in records]
        if not nodes or len({r["node_id"] for r in nodes}) != len(nodes):
            raise ValueError("Do not cache empty, partial-failure or duplicate node sets")
        envelope = {"format": CACHE_FORMAT, "context": context, "nodes": nodes,
                    "nodes_hash": content_hash(nodes), "provenance": provenance or {}}
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.path_for(context)
        lock = target.with_suffix(".lock")
        # Fail rather than overwrite a concurrently frozen or conflicting result.
        with lock.open("x", encoding="utf-8"):
            pass
        temporary = None
        try:
            existing = self.load(context)
            if existing is not None:
                if existing["nodes_hash"] != envelope["nodes_hash"]:
                    raise ValueError("A G1 result already exists; never replace it with another sample")
                return
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.root,
                                             suffix=".tmp", delete=False) as output:
                temporary = Path(output.name)
                json.dump(envelope, output, ensure_ascii=False, indent=2)
                output.write("\n")
            os.replace(temporary, target)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            lock.unlink(missing_ok=True)


class CachedG1Reasoner:
    """A1 records G1 once; A2/A3 must use mode='require' with the same context."""
    def __init__(self, cache, context, *, knowledge, delegate=None, mode="require",
                 provenance=None, expected_fingerprint=None):
        if mode not in {"record", "require"}:
            raise ValueError("G1 cache mode must be record or require")
        if mode == "record" and delegate is None:
            raise ValueError("Recording requires an explicit connected offline reasoner")
        self.cache, self.context, self.knowledge = cache, context, knowledge
        self.delegate, self.mode = delegate, mode
        self.provenance, self.expected_fingerprint = provenance, expected_fingerprint
        self.cache_hit = False
        self.screen_denial_scope_events = []

    def generate_candidates(self, profile):
        if content_hash(profile) != self.context["profile_hash"]:
            raise ValueError("G1 cache profile differs from the actual engine input")
        if knowledge_hash(self.knowledge) != self.context["knowledge_hash"]:
            raise ValueError("G1 cache knowledge differs from the active provider")
        graph = load_candidate_graph()
        if content_hash(graph) != self.context["tree_hash"]:
            raise ValueError("G1 cache tree differs from the active tree")
        envelope = self.cache.load(self.context, expected_fingerprint=self.expected_fingerprint)
        self.cache_hit = envelope is not None
        self.screen_denial_scope_events = []
        if envelope is None:
            if self.mode == "require":
                raise G1CacheMiss("Shared G1 missing; A2/A3 may not make a fresh G1 prediction")
            if self.delegate.model != self.context["model"]:
                raise ValueError("G1 delegate model differs from the cache context")
            view = self.delegate.node_knowledge_view
            if (not view.screening_zero_knowledge or view.screening_compact_view or view.anchor_compact_view
                    or self.delegate.criterion_batching != self.context["criterion_batching"]
                    or self.delegate.static_knowledge_prefix != self.context["static_knowledge_prefix"]):
                raise ValueError("G1 delegate policy differs from the cache context")
            try:
                generation = self.delegate.generate_candidates(profile)
            finally:
                self.screen_denial_scope_events = list(self.delegate.screen_denial_scope_events)
            provenance = self.provenance() if callable(self.provenance) else self.provenance
            # Store judgments, not CandidateGeneration, candidates, ranking or final diagnoses.
            self.cache.save(self.context, generation.tree_trace, provenance=provenance)
            return generation
        allowed = set(profile.get("_allowed_evidence_refs", []))
        records = {r["node_id"]: r for r in envelope["nodes"]}
        consumed = set()

        def replay(nodes):
            answers = []
            for node in nodes:
                record = records.get(node["node_id"])
                if record is None:
                    raise ValueError("Shared G1 does not cover the requested traversal")
                if (record["stage"] != node["stage"] or record["question"] != node["question"]
                        or record["source_refs"] != node["source_refs"]):
                    raise ValueError("Cached node definition differs from current traversal")
                if not set(record["evidence_refs"]) <= allowed:
                    raise ValueError("Cached node references unknown patient evidence")
                answers.append(TreeNodeJudgment.model_validate({k: record[k] for k in TreeNodeJudgment.model_fields}))
                consumed.add(node["node_id"])
            return answers
        generation = DecisionTreeCandidateGenerator(replay, knowledge=self.knowledge).generate()
        if consumed != set(records):
            raise ValueError("Shared G1 has extra nodes beyond the actual traversal")
        return generation
