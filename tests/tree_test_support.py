"""Controlled node responses for engineering tests; not clinical gold cases."""
import json


def _node_answers(batch, labels, ref, overrides, complete_labels, confidences):
    from layers.layer3.diagnosis.decision_tree import load_candidate_graph

    graph = load_candidate_graph()
    if not set(labels) <= set(graph["paths"]):
        raise ValueError("Unsupported controlled candidate")
    expected = {rule["node_id"]: rule["expected"]
                for label in labels for rule in graph["paths"][label]["guards"]}
    answers = []
    for node in batch:
        selected = bool(set(node["disorders"]) & set(labels))
        status = "pending"
        if node["stage"] == "screening":
            status = "met" if selected else "not_met"
        if node["stage"] == "anchor":
            status = "met" if selected else "not_met"
        if node["stage"] == "exclusion" and complete_labels:
            status = expected.get(node["node_id"], "pending")
        if node["stage"] == "criterion" and set(node["disorders"]) & set(complete_labels):
            status = "met"
        status = (overrides or {}).get(node["node_id"], status)
        answers.append({"node_id": node["node_id"], "status": status,
                        "rationale": "controlled engineering fixture",
                        "evidence_refs": [ref] if status != "pending" else [],
                        "symptom_ids": (node.get("symptom_pool", []) if node["stage"] == "criterion"
                                        else node.get("symptom_pool", [])[:1]) if status == "met" else [],
                        "confidence": (confidences or {}).get(node["disorders"][0], 0.75)})
    return {"nodes": answers}


def node_payload(prompt, labels=("D013",), ref="ev-mood", overrides=None, *, complete_labels=()):
    batch = json.JSONDecoder().raw_decode(prompt.split("当前节点 JSON：", 1)[1].lstrip())[0]
    return _node_answers(batch, labels, ref, overrides, complete_labels, None)


def controlled_generation(labels, *, ref="ev-mood", complete_labels=(), overrides=None, confidences=None):
    from layers.layer3.diagnosis.decision_tree import DecisionTreeCandidateGenerator
    from layers.layer3.diagnosis.schemas import TreeNodeJudgment

    def evaluate(nodes):
        payload = _node_answers(nodes, labels, ref, overrides, complete_labels, confidences)
        return [TreeNodeJudgment.model_validate(item) for item in payload["nodes"]]
    return DecisionTreeCandidateGenerator(evaluate).generate()
