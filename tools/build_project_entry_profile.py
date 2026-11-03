"""Export the selected 9-group/24-label symptom entries with current family routes.

Offline conversion only. The new runtime does not import this builder, shared,
the old tree executor or the original JSON knowledge providers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def source_symptoms(root, prefix):
    result = {}
    for path in sorted((root / "symptom").glob("*.json")):
        for key, row in read(path).items():
            if key in result:
                raise ValueError(f"Duplicate source symptom id: {key}")
            result[key] = {"id": key, "description": row["description"],
                           "source_ref": f"{prefix}/symptom/{path.name}#/{key}"}
    return result


def pool(value):
    result = set()
    if isinstance(value, dict):
        result.update(value.get("symptom_pool", []))
        for child in value.values():
            result.update(pool(child))
    elif isinstance(value, list):
        for child in value:
            result.update(pool(child))
    return result


def build_profile():
    # Explicit offline metadata read; never selected by the runtime executor.
    from shared.disorder_catalog import grouped_catalog

    mental = ROOT.parent / "external/MentalBench/resources/knowledge_graph/EN"
    supplement = ROOT.parent / "external/MentalBench/resources/dsm5_supplement/EN"
    mental_ref = "../external/MentalBench/resources/knowledge_graph/EN"
    supplement_ref = "../external/MentalBench/resources/dsm5_supplement/EN"
    symptoms = source_symptoms(mental, mental_ref)
    supplement_symptoms = source_symptoms(supplement, supplement_ref)
    if set(symptoms) & set(supplement_symptoms):
        raise ValueError("Supplement ids must not overwrite MentalKG symptoms")
    symptoms.update(supplement_symptoms)
    criteria = read(mental / "diagnostic_criteria.json")
    criteria.update(read(supplement / "diagnostic_criteria.json"))
    from tools.project24_tree_specifications import specifications
    routes = {label: spec["id"] for spec in specifications() for label in spec["target_ids"]}
    groups, entries = [], []
    for group in grouped_catalog():
        groups.append({"id": group["group"], "label": group["label"]})
        for item in group["disorders"]:
            label = item["disorder"]
            ids = sorted(pool(criteria[label]))
            if not ids or set(ids) - set(symptoms):
                raise ValueError(f"Missing source symptoms for {label}")
            targets = [routes[label]]
            refs = [f"first2014_project24/source_audit.json#/{routes[label]}"]
            descriptions = "\n".join(f"- {symptoms[sid]['description']}" for sid in ids)
            question = (
                "For entry screening only, does the clinical case contain a presentation relevant to "
                "the following source-defined symptom descriptions? Select a reasoning path based on "
                "the case facts; do not determine a diagnosis, presentation subtype, severity, or "
                "whether full DSM criteria are met at this screening step.\n" + descriptions
            )
            criterion_ref = (supplement_ref if label == "Insomnia" else mental_ref) + f"/diagnostic_criteria.json#/{label}"
            entries.append({"id": f"entry.{label}", "group_id": group["group"], "disorder_id": label,
                            "name": item["name"], "label": item["label"], "question": question,
                            "source_refs": [criterion_ref, *refs], "tree_ids": targets,
                            "symptoms": [symptoms[sid] for sid in ids]})
    if len(groups) != 9 or len(entries) != 24:
        raise ValueError("This export implements the user's selected nine groups and 24 label entries")
    return {
        "id": "project-nine-groups-twenty-four-source-family-entries-v1", "standard": "DSM-5", "review_status": "source_verified",
        "groups": groups, "entries": entries,
        "provenance": {
            "group_and_label_metadata": "shared/disorder_catalog.py:grouped_catalog",
            "route_references": "tools/project24_tree_specifications.py:specifications/*/target_ids",
            "symptom_content": [mental_ref + "/symptom", supplement_ref + "/symptom"],
            "scope_decision": "User selected nine groups and 24 label-associated symptom entries, then replaced the previous source tree package",
            "route_status": "All 24 entries route to the 19 source-derived family trees; no old symptom-tree route dependencies",
            "runtime_isolation": "Exported data only; no old executor, pending-state policy, confidence or criteria thresholds imported"},
        "notes": [
            "沿用用户选择的 9 类/24 症状入口；入口只选路径，不直接给诊断。",
            "24 入口全部筛选完，再在共享队列推进所有适用树；共享树只执行一次。",
            "症状文本来源保留；路由来自当前 19 棵家族树，不读取旧执行器或旧树映射。",
            "树内容取自 PDF；入口症状是离线导出的项目数据，来源在每项 source_ref 中区分。",
            "候选仅有 24 项；范围外原因记录为观察/终止，不强制归入最近标签。",
            "手册没有完整重印 DSM 诊断标准；完整性缺口见 first2014_project24/source_gaps.json。"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "knowledge/paper_dtree/first2014_project24")
    args = parser.parse_args(argv)
    profile = build_profile()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "entry_profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = ["# 当前九类二十四个症状筛选入口", "", "分类为组织信息；每个标签对应独立症状筛选，命中不代表确诊。", "",
            "| 分类 | 标签 | 名称 | 来源症状数 | 当前家族树 |", "|---|---|---|---:|---|"]
    groups = {g["id"]: g["label"] for g in profile["groups"]}
    for entry in profile["entries"]:
        rows.append(f"| {groups[entry['group_id']]} | {entry['disorder_id']} | {entry['label']} | {len(entry['symptoms'])} | {', '.join(entry['tree_ids'])} |")
    rows.extend(["", "症状正文及逐项来源位于 entry_profile.json；当前目标树均在 refined_bundle.json 内。", ""])
    (args.output_dir / "entries.md").write_text("\n".join(rows), encoding="utf-8")
    print(f"Exported {len(profile['groups'])} groups and {len(profile['entries'])} entry screens to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
