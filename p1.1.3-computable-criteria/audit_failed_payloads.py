"""Offline diagnosis of failed JSON contracts, with no new inference calls."""
from collections import Counter
import json

import run_experiment as experiment
from tools.l3_computable_criteria import rule_for, validate_fact


def main():
    records = experiment.infra.read(experiment.RUN / "results.jsonl")
    profiles = {row["case_id"]: row for row in experiment.infra.read(experiment.HERE / "inputs.jsonl")}
    definitions = {}
    for path in (experiment.HERE / "checkpoints/full").glob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        definitions[value["context"]["case_id"]] = {key: node["definition"] for key, node in value["nodes"].items()}
    findings = []
    for record in records:
        if record["arm"] != "computed" or record["ok"]:
            continue
        profile, allowed = experiment.infra.node_client._prompt_profile(experiment.infra.model_profile(profiles[record["case_id"]]))
        for request in record["requests"]:
            if request["stage"] != "criteria" or request.get("error_type"):
                continue
            batch = [definitions[record["case_id"]][key] for key in request["node_ids"]]
            rules = [rule_for(node) for node in batch]
            specs = {key: spec for rule in rules if rule for key, spec in rule["facts"].items()}
            semantic = {node["node_id"] for node, rule in zip(batch, rules) if rule is None}
            finding = {"case_id": record["case_id"], "attempt_id": request["attempt_id"],
                       "structured_attempt": request["structured_attempt"], "finish_reason": request.get("finish_reason"),
                       "expected_facts": len(specs), "expected_semantic_nodes": len(semantic)}
            try:
                payload = experiment.infra.node_client._parse_json_object(request["raw_output"])
                payload = experiment.infra.node_client._canonicalize_payload_evidence_refs(payload, profile, allowed)
                experiment.infra.node_client._assert_payload_evidence_refs(payload, allowed)
                finding["payload_field_types"] = {key: type(value).__name__ for key, value in payload.items()}
                if set(payload) != {"facts", "nodes"} or not isinstance(payload["facts"], list) or not isinstance(payload["nodes"], list):
                    finding["contract_shape_error"] = "Expected exactly facts:list and nodes:list"
                    raise ValueError("Wrong top-level schema")
                facts = payload.get("facts", [])
                keys = [value.get("fact_id") for value in facts]
                finding.update(returned_facts=len(facts), missing_facts=sorted(set(specs) - set(keys)),
                               extra_facts=sorted(set(keys) - set(specs)),
                               duplicate_facts=[key for key, count in Counter(keys).items() if count > 1], invalid_facts=[])
                for fact in facts:
                    if fact.get("fact_id") in specs:
                        try:
                            validate_fact(fact, specs[fact["fact_id"]], allowed)
                        except ValueError as error:
                            finding["invalid_facts"].append({"fact_id": fact["fact_id"], "error": str(error)})
                nodes = payload.get("nodes", [])
                parsed = [experiment.TreeNodeJudgment.model_validate(node) for node in nodes]
                finding["missing_semantic_nodes"] = sorted(semantic - {node.node_id for node in parsed})
                finding["extra_semantic_nodes"] = sorted({node.node_id for node in parsed} - semantic)
            except Exception as error:
                finding["decode_or_schema_error"] = type(error).__name__
            findings.append(finding)
    experiment.infra.cost.write(experiment.HERE / "failed-payload-audit.json", findings)
    print(json.dumps({"audited_requests": len(findings),
        "finish_reasons": dict(Counter(row["finish_reason"] for row in findings)),
        "decode_or_schema_errors": dict(Counter(row.get("decode_or_schema_error") for row in findings)),
        "top_level_shape_errors": sum("contract_shape_error" in row for row in findings)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
