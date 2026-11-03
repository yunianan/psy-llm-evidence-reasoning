"""Independent contracts: no project labels, profiles, confidence or Pydantic models."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    items = tuple(_text(item, name) for item in value)
    if len(items) != len(set(items)):
        raise ValueError(f"{name} contains duplicates")
    return items


def _keys(value: Any, allowed: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError(f"{name}: expected an object with only {sorted(allowed)}")
    return value


@dataclass(frozen=True)
class SourceRef:
    document: str
    print_pages: tuple[int, ...]
    pdf_pages: tuple[int, ...]
    note: str

    @classmethod
    def parse(cls, data: Any) -> SourceRef:
        d = _keys(data, {"document", "print_pages", "pdf_pages", "note"}, "source")
        for key in ("print_pages", "pdf_pages"):
            allow_unknown_pdf_page = key == "pdf_pages" and d.get("document") == "DSM5_2013_user_excerpt"
            if (not isinstance(d.get(key), list) or (not d[key] and not allow_unknown_pdf_page)
                    or any(type(p) is not int or p < 1 for p in d[key])):
                raise ValueError(f"source.{key} requires positive page numbers")
        return cls(_text(d.get("document"), "source.document"),
                   tuple(d["print_pages"]), tuple(d["pdf_pages"]),
                   _text(d.get("note"), "source.note"))


@dataclass(frozen=True)
class Branch:
    next: tuple[str, ...] = ()
    diagnoses: tuple[str, ...] = ()
    fallback_diagnoses: tuple[str, ...] = ()
    referrals: tuple[str, ...] = ()
    pending_referrals: tuple[str, ...] = ()
    observations: tuple[str, ...] = ()

    @classmethod
    def parse(cls, data: Any) -> Branch:
        names = ("next", "diagnoses", "fallback_diagnoses", "referrals", "pending_referrals", "observations")
        d = _keys(data, set(names), "branch")
        return cls(**{name: _strings(d.get(name, []), f"branch.{name}") for name in names})


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    yes: Branch
    no: Branch
    source: SourceRef
    calculation: dict[str, Any] | None = None


@dataclass(frozen=True)
class Tree:
    id: str
    title: str
    screening_question: str
    screening_source: SourceRef
    roots: tuple[str, ...]
    nodes: dict[str, Question]


@dataclass(frozen=True)
class Bundle:
    id: str
    standard: str
    stage: str
    coverage: str
    documents: tuple[str, ...]
    diagnoses: tuple[str, ...]
    trees: tuple[Tree, ...]
    refinements: tuple[dict[str, Any], ...]
    referral_policy: str = "acyclic"

    @classmethod
    def parse(cls, data: Any) -> Bundle:
        d = _keys(data, {"id", "standard", "stage", "coverage", "documents", "diagnoses",
                         "trees", "refinements", "review_status", "referral_policy"}, "bundle")
        if d.get("review_status") != "source_verified":
            raise ValueError("Only source_verified tree bundles are executable; a catalog is not a tree")
        standard = _text(d.get("standard"), "standard")
        if standard not in {"DSM-5", "DSM-5-TR"}:
            raise ValueError("Explicit DSM-5 or DSM-5-TR standard required")
        if d.get("stage") not in {"raw", "refined"}:
            raise ValueError("stage must be raw or refined")
        if d.get("coverage") not in {"partial", "full"}:
            raise ValueError("coverage must be partial or full")
        documents = _strings(d.get("documents"), "documents")
        diagnoses = _strings(d.get("diagnoses"), "diagnoses")
        if not documents or not diagnoses or not isinstance(d.get("trees"), list) or not d["trees"]:
            raise ValueError("Executable bundles need documents, diagnoses and trees")
        trees = []
        for raw in d["trees"]:
            t = _keys(raw, {"id", "title", "screening_question", "screening_source", "roots", "nodes"}, "tree")
            nodes = {}
            if not isinstance(t.get("nodes"), dict) or not t["nodes"]:
                raise ValueError("tree.nodes must be a nonempty object")
            for key, raw_node in t["nodes"].items():
                _text(key, "node id")
                n = _keys(raw_node, {"question", "yes", "no", "source", "calculation"}, "node")
                calculation = n.get("calculation")
                if calculation is not None:
                    c = _keys(calculation, {"kind", "age_node", "symptom_nodes"}, "calculation")
                    symptoms = _strings(c.get("symptom_nodes"), "calculation.symptom_nodes")
                    if c.get("kind") != "adhd_domain_threshold_v1" or len(symptoms) != 9:
                        raise ValueError("ADHD threshold calculation requires nine distinct symptom nodes")
                    _text(c.get("age_node"), "calculation.age_node")
                    allowed_domains = {tuple(f"{domain}_{i}" for i in range(1, 10)) for domain in ("a1", "a2")}
                    if t.get("id") != "p24.adhd" or c["age_node"] != "age_17_or_over" or symptoms not in allowed_domains:
                        raise ValueError("ADHD calculation must use exactly one complete symptom domain and its age node")
                nodes[key] = Question(key, _text(n.get("question"), "question"),
                                      Branch.parse(n.get("yes")), Branch.parse(n.get("no")),
                                      SourceRef.parse(n.get("source")), calculation)
            trees.append(Tree(_text(t.get("id"), "tree.id"), _text(t.get("title"), "tree.title"),
                              _text(t.get("screening_question"), "screening_question"),
                              SourceRef.parse(t.get("screening_source")),
                              _strings(t.get("roots"), "roots"), nodes))
        refinements = d.get("refinements", [])
        if not isinstance(refinements, list):
            raise ValueError("refinements must be a list")
        for change in refinements:
            _keys(change, {"tree_id", "node_id", "category", "before", "after", "reason", "source"}, "refinement")
            for name in ("tree_id", "node_id", "category", "before", "after", "reason"):
                _text(change.get(name), f"refinement.{name}")
            SourceRef.parse(change.get("source"))
        if d["stage"] == "refined" and not refinements:
            raise ValueError("Refined bundles must record their changes; raw trees are not refined trees")
        referral_policy = d.get("referral_policy", "acyclic")
        if referral_policy not in {"acyclic", "visit_each_node_once"}:
            raise ValueError("Unknown referral policy")
        bundle = cls(_text(d.get("id"), "id"), standard, d["stage"], d["coverage"],
                     documents, diagnoses, tuple(trees), tuple(refinements), referral_policy)
        bundle.validate_graph()
        return bundle

    def validate_graph(self) -> None:
        trees = {tree.id: tree for tree in self.trees}
        if len(trees) != len(self.trees):
            raise ValueError("Duplicate tree id")
        edges: dict[tuple[str, str], list[tuple[str, str]]] = {}
        local_edges: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for tree in self.trees:
            if not tree.roots or any(root not in tree.nodes for root in tree.roots):
                raise ValueError(f"Invalid roots in {tree.id}")
            sources = [tree.screening_source]
            for key, node in tree.nodes.items():
                sources.append(node.source)
                dests = []
                for branch in (node.yes, node.no):
                    if any(n not in tree.nodes for n in branch.next):
                        raise ValueError(f"Unknown next node in {tree.id}/{key}")
                    if set(branch.diagnoses + branch.fallback_diagnoses) - set(self.diagnoses):
                        raise ValueError(f"Unknown diagnosis in {tree.id}/{key}")
                    if set(branch.pending_referrals) & set(trees):
                        raise ValueError("Available referral trees must use referrals, not pending_referrals")
                    dests.extend((tree.id, n) for n in branch.next)
                    local_edges.setdefault((tree.id, key), []).extend((tree.id, n) for n in branch.next)
                    for target in branch.referrals:
                        if target not in trees:
                            raise ValueError(f"Unknown referral: {target}")
                        dests.extend((target, root) for root in trees[target].roots)
                edges[(tree.id, key)] = dests
            if any(s.document not in self.documents for s in sources):
                raise ValueError(f"Unlisted source document in {tree.id}")
            # A calculation may only use questions guaranteed to have run on
            # every path from this tree's roots, never missing or future answers.
            predecessors = {key: set() for key in tree.nodes}
            for key, node in tree.nodes.items():
                for branch in (node.yes, node.no):
                    for target in branch.next:
                        predecessors[target].add(key)
            universe = set(tree.nodes)
            dominators = {k: ({k} if k in tree.roots else set(universe)) for k in tree.nodes}
            changed = True
            while changed:
                changed = False
                for key in tree.nodes:
                    if key in tree.roots or not predecessors[key]:
                        continue
                    common = set.intersection(*(dominators[p] for p in predecessors[key]))
                    new = {key} | common
                    if new != dominators[key]:
                        dominators[key] = new
                        changed = True
            for key, node in tree.nodes.items():
                if node.calculation:
                    dependencies = {node.calculation["age_node"], *node.calculation["symptom_nodes"]}
                    if key in dependencies or not dependencies <= dominators[key]:
                        raise ValueError(f"Calculation inputs must precede it on every path: {tree.id}/{key}")
        if self.referral_policy not in {"acyclic", "visit_each_node_once"}:
            raise ValueError("Unknown referral policy")
        visited, active = set(), set()

        def visit(key):
            if key in active:
                raise ValueError("Tree bundle contains a cycle, including referrals")
            if key in visited:
                return
            active.add(key)
            for dest in (local_edges.get(key, []) if self.referral_policy == "visit_each_node_once" else edges[key]):
                visit(dest)
            active.remove(key)
            visited.add(key)

        # Local question cycles always remain invalid. Source cross-tree feedback
        # is allowed only by an explicit policy and terminates via runner.visited.
        for key in edges:
            visit(key)
        reachable, pending = set(), [(t.id, r) for t in self.trees for r in t.roots]
        while pending:
            key = pending.pop()
            if key not in reachable:
                reachable.add(key)
                pending.extend(edges[key])
        if reachable != set(edges):
            raise ValueError("Tree bundle contains unreachable nodes")
        for change in self.refinements:
            tree = trees.get(change["tree_id"])
            if tree is None or change["node_id"] not in {*tree.nodes, "screening", "structure"}:
                raise ValueError("Refinement references an unknown tree/node")
            if SourceRef.parse(change["source"]).document not in self.documents:
                raise ValueError("Refinement references an unlisted document")


def load_bundle(path: str | Path) -> Bundle:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return Bundle.parse(json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_pairs))


@dataclass(frozen=True)
class Request:
    model: str
    messages: tuple[dict[str, str], ...]
    temperature: int = 0
    top_p: int = 1
    frequency_penalty: int = 0
    presence_penalty: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Response:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    request_id: str = ""
    fingerprint: str | None = None
