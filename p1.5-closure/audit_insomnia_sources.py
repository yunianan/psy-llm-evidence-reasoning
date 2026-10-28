"""Offline provenance audit; no new clinical judgments, prompts, or model calls."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import socket
import sys
from unittest.mock import patch

WORKSPACE = Path(__file__).resolve().parents[2]
ENGINE = WORKSPACE / "L1-L5系统"
sys.path.insert(0, str(ENGINE))

from layers.layer3.diagnosis.clinicr import CLINICR
from layers.layer3.diagnosis.decision_tree import criterion_nodes, load_candidate_graph
from layers.layer3.llm import client
from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider
from tests.tree_test_support import controlled_generation

VERSION = "insomnia-source-coverage-v1"
RUN = WORKSPACE / "phase1/p1.4-dev15-case2/runs/p1.4-dev15-case2-recovery-v1"
# Explicit crosswalk of existing fields, not a new interpretation of medical rules.
FIELDS = {
    "insomnia_symptoms/min_count": "A",
    "min_duration": "D",
    "functional_impairment_required": "B",
    "additional_requirements/0": "A",
    "additional_requirements/1": "B",
    "additional_requirements/2": "C",
    "additional_requirements/3": "E",
    "additional_requirements/4": "F",
    "additional_requirements/5": "G",
    "additional_requirements/6": "H",
}
GUARDS = {"sleep.other_disorder": "F", "sleep.substance": "G", "sleep.explained": "H"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(ref):
    filename, pointer = ref.split("#", 1)
    path = (ENGINE / filename).resolve()
    # Only the local supplement is needed; do not resolve handbook URLs or EVAL.
    path.relative_to((WORKSPACE / "external/MentalBench/resources/dsm5_supplement/EN").resolve())
    value = read_json(path)
    for part in pointer.lstrip("/").split("/") if pointer else []:
        part = part.replace("~1", "/").replace("~0", "~")
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def count(rows, field):
    return {"mapped": sum(bool(row[field]) for row in rows), "total": len(rows)}


def audit(run=RUN):
    provider = JsonDiagnosticKnowledgeProvider()
    source = provider.get_candidate("Insomnia")
    graph = load_candidate_graph()
    definitions = criterion_nodes("Insomnia", knowledge=provider)
    assert set(definitions) == {"criterion.Insomnia." + key for key in FIELDS}
    assert {rule["node_id"] for rule in graph["paths"]["Insomnia"]["guards"]} == set(GUARDS)
    watched = [ENGINE / "shared/knowledge_sources.py", ENGINE / "layers/layer3/diagnosis/decision_tree.py",
               ENGINE / "layers/layer3/diagnosis/assessment.py", ENGINE / "layers/layer3/diagnosis/clinicr.py",
               ENGINE / "knowledge/dsm5_decision_tree/candidate_paths.json"]
    watched += sorted(provider.insomnia.root.rglob("*.json"))
    watched += [Path(run) / "predictions/A1.jsonl"]
    before = {str(path.relative_to(WORKSPACE)): sha256(path) for path in watched}
    adapter_rows = [{"criterion": key, "source_ref": ref, "resolves": resolve(ref) is not None,
                     "granularity": "required_criteria_root" if key == "A" else "specific_field"}
                    for key, ref in source["criterion_source_refs"].items()]

    blocked = {"model_calls": 0, "network_calls": 0}

    def no_model(*args, **kwargs):
        blocked["model_calls"] += 1
        raise AssertionError("Provenance audit must not call a model")

    def no_network(*args, **kwargs):
        blocked["network_calls"] += 1
        raise AssertionError("Provenance audit must remain offline")

    with ExitStack() as stack:
        stack.enter_context(patch.object(client, "ask_llm_stream", no_model))
        stack.enter_context(patch.object(socket.socket, "connect", no_network))
        stack.enter_context(patch.object(socket.socket, "connect_ex", no_network))
        # All Insomnia criteria/guards are pending: the source-report path executes,
        # but no clinical conclusion or external Verifier is requested.
        generation = controlled_generation(["Insomnia"], ref="synthetic-insomnia-source")

        class FixtureReasoner:
            def generate_candidates(self, profile):
                return generation

        report = CLINICR(reasoner=FixtureReasoner(), knowledge=provider).run({
            "evidence_nodes": [{"evidence_id": "synthetic-insomnia-source", "quote": "synthetic provenance fixture"}]
        })
    assert report["diagnosis"]["primary"] is None
    assessed = {item["code"]: item for a in report["checked"] if a["disorder"] == "Insomnia"
                for item in a["criteria_detail"]}
    trace = {item["node_id"]: item for item in report["tree_trace"]}
    exported = {item["criterion_code"]: item for item in report["evidence_refs"]
                if item["type"] == "DSM-5-criterion" and item["disorder"] == "Insomnia"}
    node_rows = []
    for node_id, definition in {**definitions, **{key: graph["nodes"][key] for key in GUARDS}}.items():
        guard = node_id in GUARDS
        field = node_id.removeprefix("criterion.Insomnia.")
        letter = GUARDS[node_id] if guard else FIELDS[field]
        refs = [ref for ref in definition["source_refs"] if ref.startswith(source["source_ref"])]
        values = [resolve(ref) for ref in refs]
        condition_equal = True if guard else values == [definition["source_condition"]]
        specific = bool(refs) and condition_equal
        retained = node_id in assessed and node_id in trace and all(ref in trace[node_id]["source_refs"] for ref in refs)
        node_rows.append({"node_id": node_id, "stage": definition["stage"], "criterion": letter,
                          "criterion_source_ref": source["criterion_source_refs"][letter],
                          "specific_source_refs": refs, "mapped_and_resolvable": specific,
                          "source_condition_equal": condition_equal,
                          "assessment_code_and_trace_preserved": retained,
                          "export_source_doc": exported[node_id]["source_doc"],
                          "export_source_doc_is_specific_json": exported[node_id]["source_doc"] in refs})
    assert set(assessed) == set(exported) == {row["node_id"] for row in node_rows}
    symptom_rows = [{"symptom_id": sid, "criterion": f"A{i}",
                     "source_ref": source["criterion_source_refs"][f"A{i}"],
                     "object_equal": resolve(source["criterion_source_refs"][f"A{i}"]) == source["symptoms"][sid]}
                    for i, sid in enumerate(("S085", "S086", "S087"), 1)]
    real_rows = read_rows(Path(run) / "predictions/A1.jsonl")
    real = {"cases": len(real_rows), "candidate_cases": 0, "criterion_node_instances": 0,
            "assessment_instances": 0, "mapped_instances": 0}
    for row in real_rows:
        saved = row["report"]
        real["candidate_cases"] += any(c["disorder"] == "Insomnia" for c in saved["candidates"])
        real["criterion_node_instances"] += sum(t["node_id"].startswith("criterion.Insomnia.") for t in saved["tree_trace"])
        by_node = {t["node_id"]: t for t in saved["tree_trace"]}
        for assessment in saved["checked"]:
            if assessment["disorder"] == "Insomnia":
                for item in assessment["criteria_detail"]:
                    real["assessment_instances"] += 1
                    expected = next(n["specific_source_refs"] for n in node_rows if n["node_id"] == item["code"])
                    real["mapped_instances"] += all(ref in by_node.get(item["code"], {}).get("source_refs", []) for ref in expected)
    real["assessment_mapping_rate"] = (real["mapped_instances"] / real["assessment_instances"]
                                         if real["assessment_instances"] else None)
    missing = [row["node_id"] for row in node_rows if not row["mapped_and_resolvable"]
               or not row["assessment_code_and_trace_preserved"]]
    json_export_gaps = [row["node_id"] for row in node_rows if not row["export_source_doc_is_specific_json"]]
    after = {str(path.relative_to(WORKSPACE)): sha256(path) for path in watched}
    assert before == after, "Audit modified an input"
    passed = (not missing and all(row["resolves"] for row in adapter_rows)
              and all(row["object_equal"] for row in symptom_rows)
              and {row["criterion"] for row in node_rows} == set("ABCDEFGH") and not any(blocked.values()))
    return {"version": VERSION, "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "accepted_scope": "local source resolution + expanded nodes + controlled assessment/report provenance",
            "passed": passed, "clinical_validation": False, "real_eval_accessed": False,
            **blocked, "inputs_unchanged": before == after, "input_sha256": before,
            "coverage": {"adapter_refs": count(adapter_rows, "resolves"),
                         "expanded_criteria": count([r for r in node_rows if r["stage"] == "criterion"], "mapped_and_resolvable"),
                         "guards": count([r for r in node_rows if r["stage"] == "exclusion"], "mapped_and_resolvable"),
                         "assessment_via_code_and_trace": count(node_rows, "assessment_code_and_trace_preserved"),
                         "criterion_report_json_source": count([r for r in node_rows if r["stage"] == "criterion"], "export_source_doc_is_specific_json"),
                         "guard_report_json_source": count([r for r in node_rows if r["stage"] == "exclusion"], "export_source_doc_is_specific_json"),
                         "symptom_objects": count(symptom_rows, "object_equal"), "letters": {"mapped": 8, "total": 8}},
            "unmapped_nodes": missing, "single_field_json_export_gaps": json_export_gaps,
            "adapter_refs": adapter_rows, "nodes": node_rows, "symptoms": symptom_rows,
            "real_dev15": real,
            "limitations": ["A adapter ref points to a criteria root; actual expanded nodes carry finer field refs.",
                            "B functional_impairment_required is an existing alias; distress OR impairment meaning comes from existing event/source text, not this audit.",
                            "Three guard source_doc values retain First2014; exact supplement JSON refs are in tree_trace.source_refs and this crosswalk.",
                            "DEV15 did not activate Insomnia criteria; real clinical Insomnia assessment coverage is unobserved (0/0 = null).",
                            "This validates provenance plumbing, not clinical validity or model understanding of A-H."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a new audit directory")
    result = audit(args.run)
    args.out.mkdir(parents=True)
    (args.out / "coverage.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Insomnia A–H 逐节点来源覆盖", "", f"工程来源核验通过：{result['passed']}；模型调用0，EVAL访问0。", "",
             "只验收本地来源及受控assessment/report链，不作临床标准复核。", "",
             "| 口径 | 映射/分母 |", "|---|---:|"]
    for key, value in result["coverage"].items():
        lines.append(f"| {key} | {value['mapped']}/{value['total']} |")
    lines += ["", "| 节点 | A–H | 具体来源 | assessment可回溯 |", "|---|---|---|---|"]
    for row in result["nodes"]:
        lines.append(f"| {row['node_id']} | {row['criterion']} | {'; '.join(row['specific_source_refs'])} | {row['assessment_code_and_trace_preserved']} |")
    lines += ["", "未映射节点：" + json.dumps(result["unmapped_nodes"]), "",
              "三条guard的单一source_doc仍为First2014（具体JSON单字段0/3）；完整JSON来源保留在同节点tree_trace.source_refs，不能把这两种口径混为13/13。",
              "实际DEV15未激活Insomnia候选/criteria/assessment：0个实例，实际映射率null；不把受控13/13冒充真实病例13/13。", "",
              "adapter A保留原required_criteria根指针；min_count与additional_requirements/0提供具体字段。B沿用原functional_impairment_required及OR语义来源，不修改医学条件。", ""]
    (args.out / "coverage.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "coverage": result["coverage"],
                      "unmapped_nodes": result["unmapped_nodes"], "out": str(args.out)}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
