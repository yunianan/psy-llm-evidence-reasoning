"""Reuse Step 1 evidence for candidate routing; never call a diagnostic model again."""
from __future__ import annotations

from shared.diagnostic_knowledge import DiagnosticKnowledgeProvider

from .decision_tree import criterion_nodes, load_candidate_graph
from .schemas import (
    CandidateGeneration, CriterionAssessment, DiagnosticCandidate,
    EliminativeAssessment, MissingDimension,
)


def assessments_from_step1(
    generation: CandidateGeneration, candidates: list[DiagnosticCandidate],
    *, knowledge: DiagnosticKnowledgeProvider | None = None,
) -> list[EliminativeAssessment]:
    """Keep the existing report shape, with all judgments taken from Step 1."""
    graph = load_candidate_graph()
    assessments = []
    for candidate in candidates:
        definitions = criterion_nodes(candidate.disorder, knowledge=knowledge)
        guards = graph["paths"][candidate.disorder]["guards"]
        records = {}
        for path in candidate.decision_paths:
            if path.path_id == candidate.disorder:
                for record in path.decisions:
                    if record.node_id in records:
                        raise ValueError(f"Duplicate Step 1 record: {record.node_id}")
                    records[record.node_id] = record
        details, missing = [], []
        complete = bool(definitions) and bool(records)
        for node_id, definition in definitions.items():
            record = records.get(node_id)
            status = record.status if record else "pending"
            details.append(CriterionAssessment(
                code=node_id, name=definition["question"], status=status,
                rationale=record.rationale if record else "Step 1 未提供该标准的判断记录",
                evidence_refs=list(record.evidence_refs) if record else [],
            ))
            if definition["mandatory"] and status != "met":
                complete = False
            if status == "pending":
                missing.append(MissingDimension(
                    field=f"decision_tree.{node_id}", purpose=definition["question"],
                    suggested_question=definition["follow_up"],
                ))
        for guard in guards:
            node_id, expected = guard["node_id"], guard["expected"]
            definition = graph["nodes"][node_id]
            record = records.get(node_id)
            raw_status = record.status if record else "pending"
            status = "pending" if raw_status == "pending" else (
                "met" if raw_status == expected else "not_met"
            )
            details.append(CriterionAssessment(
                code=node_id,
                name=f"鉴别路径要求 {expected}：{definition['question']}",
                status=status,
                rationale=record.rationale if record else "Step 1 未提供该鉴别节点的判断记录",
                evidence_refs=list(record.evidence_refs) if record else [],
            ))
            if status != "met":
                complete = False
            if status == "pending":
                missing.append(MissingDimension(
                    field=f"decision_tree.{node_id}", purpose=definition["question"],
                    suggested_question=definition["follow_up"],
                ))
        if any(path.provisional for path in candidate.decision_paths):
            complete = False
        assessments.append(EliminativeAssessment(
            disorder=candidate.disorder, criteria_detail=details,
            conclusion="meets_criteria" if complete else "watch",
            conclusion_reason=(
                "Step 1 的必需标准及鉴别路径均已满足，直接复用原判断"
                if complete else "Step 1 标准未完全满足、信息不足或记录不完整，保留暂定候选"
            ),
            missing_dimensions=missing,
        ))

    # Branches already pruned by Step 1 are reported without a second judgment.
    trace = {record.node_id: record for record in generation.tree_trace}
    retained = {candidate.disorder for candidate in candidates}
    for label, coverage in generation.candidate_coverage.items():
        if label in retained or not coverage.startswith("excluded:"):
            continue
        node_id = coverage.removeprefix("excluded:")
        record = trace[node_id]
        guard = next(rule for rule in graph["paths"][label]["guards"]
                     if rule["node_id"] == node_id)
        if record.status == "pending" or record.status == guard["expected"]:
            raise ValueError(f"Step 1 exclusion contradicts its node record: {label}")
        assessments.append(EliminativeAssessment(
            disorder=label,
            criteria_detail=[CriterionAssessment(
                code=node_id, name=f"鉴别路径要求 {guard['expected']}：{record.question}",
                status="not_met", rationale=record.rationale,
                evidence_refs=list(record.evidence_refs),
            )],
            conclusion="ruled_out", conclusion_reason=f"Step 1 鉴别分支已排除：{record.rationale}",
        ))
    return assessments
