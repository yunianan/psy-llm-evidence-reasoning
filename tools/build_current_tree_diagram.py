"""Refresh the standalone tree diagram; historical trace is explicitly labeled."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "knowledge/paper_dtree/first2014_project24"
OUT = ROOT / "output/paper_dtree/tree_diagram"
RUN = ROOT / "output/paper_dtree/20261102_mentalbench_mb000676_network_retry/model_result.json"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    bundle, audit, catalog, gaps, previous = [load(p) for p in (
        DATA / "refined_bundle.json", DATA / "source_audit.json", DATA / "label_catalog.json",
        DATA / "source_gaps.json", RUN)]
    rows = {(n["tree_id"], n["node_id"]): n for n in audit["nodes"]}
    trace = {(n["tree_id"], n["node_id"]): n for n in previous["tree_trace"]}
    by_gap = {g["id"]: g["detail"] for g in gaps}
    special = {"p24.schizoaffective": {"bipolar_type_confirmed": "schizoaffective_subtype", "depressive_type_confirmed": "schizoaffective_subtype"},
               "p24.bipolar1": {"mania_symptoms": "episode_criteria_expansion", "mania_duration": "episode_criteria_expansion"},
               "p24.bipolar2": {"mania_symptoms": "episode_criteria_expansion", "mania_duration": "episode_criteria_expansion", "mde_associated": "episode_criteria_expansion"},
               "p24.mdd": {"mde_associated": "episode_criteria_expansion"},
               "p24.ptsd": {"trauma_pattern": "asd_ptsd_different_counts"}, "p24.asd": {"trauma_pattern": "asd_ptsd_different_counts"},
               "p24.adjustment": {"timing_confirmed": "adjustment_timing"}}
    trees = []
    for t in bundle["trees"]:
        labels = {c["id"] for c in catalog if c["tree_id"] == t["id"]}
        missing = [g["detail"] for g in gaps if isinstance(g["targets"], list)
                   and labels.intersection(g["targets"]) and g["id"] != "adhd_additional_review" and g["id"] not in set(special.get(t["id"], {}).values())]
        nodes = []
        for key, n in t["nodes"].items():
            row = rows[(t["id"], key)]
            gapid = special.get(t["id"], {}).get(key)
            old = trace.get((t["id"], key))
            # Never attach an old answer to a changed question, even if the ID
            # was kept; the A–E changes have not been tested on this case.
            node_trace = old["answer"] if old and old["question"] == n["question"] else None
            nodes.append({"id": key, "text": row["question_zh"] + ("（由程序根据前述逐项结果和年龄计算，不由模型直接回答）" if n.get("calculation") else ""), "yes": n["yes"], "no": n["no"],
                          "source_label": ("原书页：" + "、".join(map(str, n["source"]["print_pages"])))
                            if n["source"]["document"] == "First2014" else "用户提供的 DSM-5 摘录，印刷页 " + "、".join(map(str, n["source"]["print_pages"])) + "；原版待核验",
                          "answer": node_trace, "gap": by_gap[gapid] if gapid else ""})
        trees.append({"id": t["id"], "title": t["title"], "roots": t["roots"], "nodes": nodes,
                      "tested": any(x["tree_id"] == t["id"] for x in previous["tree_trace"]), "missing": missing})
    payload = {"trees": trees, "labels": {c["name"]: c["id"] + " " + c["label"] for c in catalog}}
    template = (OUT / "template.html").read_text(encoding="utf-8")
    result = template.replace("__FOREST_DATA__", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"))
    (OUT / "决策树与病例路径.html").write_text(result, encoding="utf-8")
    print(f"Refreshed {len(trees)} trees; historical trace from {previous['bundle_id']}; current bundle {bundle['id']}")


if __name__ == "__main__":
    main()
