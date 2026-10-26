"""Opt-in research reasoner: full sources, batched fact extraction + arithmetic."""
from __future__ import annotations

import json

from layers.layer3.llm import client
from layers.layer3.diagnosis.schemas import TreeNodeJudgment
from tools.l3_computable_criteria import VERSION, calculate, rule_for, validate_fact

BATCHING = "four_labels_full_sources_lossless_dedup"

class SharedPrefixUnavailable(ValueError):
    pass


def pack_sources(sources):
    """Lossless deduplication; conflicting definitions keep separate bindings."""
    result = {"sources": [], "symptom_catalog": {}, "event_catalog": {}}
    for index, source in enumerate(sources):
        record = {key: value for key, value in source.items() if key not in {"symptoms", "events"}}
        entry = {"source": record}
        for field, catalog_name in [("symptoms", "symptom_catalog"), ("events", "event_catalog")]:
            bindings, catalog = {}, result[catalog_name]
            for key, value in source.get(field, {}).items():
                target = key
                if target in catalog and catalog[target] != value:
                    target = f"source{index}::{key}"
                catalog[target] = value
                bindings[key] = target
            entry[field + "_bindings"] = bindings
        result["sources"].append(entry)
    return result


class ComputableCriteriaReasoner(client.APICLINICRReasoner):
    """Never selected by the production engine. A separate holdout is required."""

    def __init__(self, *args, shared_prefix=None, batch_observer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion_batching = BATCHING
        self.shared_prefix = shared_prefix
        self.batch_observer = batch_observer
        self.computation_log = []

    def node_batches(self, nodes):
        if nodes and all(node["stage"] == "criterion" for node in nodes):
            groups = {}
            for node in nodes:
                if len(node["disorders"]) != 1:
                    raise ValueError("Criterion needs one owning label")
                groups.setdefault(node["disorders"][0], []).append(node)
            labels = list(groups)
            return [[node for label in labels[offset:offset + 4] for node in groups[label]]
                    for offset in range(0, len(labels), 4)]
        return super().node_batches(nodes)

    def _evaluate_tree_nodes(self, profile, nodes):
        if not nodes or not all(node["stage"] == "criterion" for node in nodes):
            if self.shared_prefix is None:
                return super()._evaluate_tree_nodes(profile, nodes)
            answers = []
            for node in nodes:
                entry = self.shared_prefix.get(node["node_id"])
                if entry is None or entry["definition"] != node:
                    raise SharedPrefixUnavailable("Shared prefix unavailable or changed; do not generate a different prefix")
                answers.append(TreeNodeJudgment.model_validate(entry["judgment"]))
            return answers
        clinical_profile, allowed_refs = client._prompt_profile(profile)
        evidence = client._evidence_catalog(clinical_profile, allowed_refs)
        answers = []
        for batch in self.node_batches(nodes):
            if self.batch_observer:
                self.batch_observer("criteria", [node["node_id"] for node in batch])
            rules = {node["node_id"]: rule_for(node) for node in batch}
            fact_specs = {key: spec for rule in rules.values() if rule for key, spec in rule["facts"].items()}
            semantic = [node for node in batch if rules[node["node_id"]] is None]
            packet = {"knowledge": pack_sources(self.node_knowledge_view.sources_for_nodes(batch)),
                "all_source_nodes": batch, "requested_facts": fact_specs,
                "semantic_node_ids": [node["node_id"] for node in semantic]}
            prompt = """你是L3研究用证据抽取器。完整来源只发送一次，不能改变任何医学规则。
先阅读每个标签完整criteria、symptoms、differential和events。来源数据不等于患者事实。
同一症状ID在不同source_group的事实不得混用，保留严重度、情境和排除解释。
本任务不输出候选、不选分支；只返回requested_facts和semantic_node_ids。
症状/predicate事实state为present|absent|unknown：present必须有直接患者证据，absent必须有明确反证。
未提及、部分否认、矛盾或信息不足必须unknown；anchor命中的ID不是完整症状清单。
predicate判断原始来源的临床显著损害/痛苦或事件暴露，必须结合该标签events定义。
尤其失眠的临床显著条件是显著痛苦OR功能干扰，不要求两者同时存在。
duration只抽取明确病程范围lower/upper，未知端点填null；只允许day/week/month/year。
精确6个月填lower=upper=6,unit=month；至少6个月填lower=6,upper=null。
完全未知填lower=upper=unit=null。禁止用系统日期补病程，禁止把一个月近似30天。
各duration必须指向本node所属病程，不能用其他发作或其他标签的时间替代。
semantic节点仍按完整原规则三值判断：met有支持，not_met有明确反证，缺失/冲突pending。
所有present/absent事实及semantic met/not_met必须有允许的患者evidence_refs，不能创造引用。
semantic pending/not_met的symptom_ids必须为空；met也只能使用该节点symptom_pool。
不输出计数/核心症状/简单病程/required predicate节点的status，程序将从facts计算。
输出严格JSON，只有facts和nodes。每个fact_id及semantic node_id恰好一次，不得添加。
症状/predicate事实字段：fact_id,state,evidence_refs,confidence。
duration事实字段：fact_id,lower,upper,unit,evidence_refs,confidence。
nodes字段使用原schema：node_id,status,rationale,evidence_refs,symptom_ids,confidence,denial_scope。
denial_scope填unknown，rationale简短；confidence是证据支持度，不是临床概率。
""" + "\n原始知识与任务：" + json.dumps(packet, ensure_ascii=False) + "\n允许患者引用：" + json.dumps(evidence, ensure_ascii=False) + "\n患者画像：" + json.dumps(clinical_profile, ensure_ascii=False)

            def request(text):
                return client.ask_llm_stream(text, model=self.model, temperature=0,
                    timeout=client._l3_timeout_seconds(), extra_body=client._l3_reasoning_extra(self.model),
                    response_format=client.JSON_OBJECT_RESPONSE_FORMAT)

            def validate(raw):
                payload = client._canonicalize_payload_evidence_refs(client._parse_json_object(raw), clinical_profile, allowed_refs)
                client._assert_payload_evidence_refs(payload, allowed_refs)
                if set(payload) != {"facts", "nodes"} or not isinstance(payload["facts"], list) or not isinstance(payload["nodes"], list):
                    raise ValueError("Expected facts and nodes lists")
                facts = {}
                for value in payload["facts"]:
                    if not isinstance(value, dict):
                        raise ValueError("Fact must be an object")
                    key = value.get("fact_id")
                    if not isinstance(key, str) or key not in fact_specs or key in facts:
                        raise ValueError("Unexpected or duplicate fact")
                    facts[key] = validate_fact(value, fact_specs[key], allowed_refs)
                if set(facts) != set(fact_specs):
                    raise ValueError("All facts must be returned, including unknowns")
                parsed = [TreeNodeJudgment.model_validate(value) for value in payload["nodes"]]
                definitions = {node["node_id"]: node for node in semantic}
                if len(parsed) != len(definitions) or {value.node_id for value in parsed} != set(definitions):
                    raise ValueError("Semantic node set changed")
                for value in parsed:
                    if value.status != "pending" and not value.evidence_refs:
                        raise ValueError("Known semantic judgment needs patient evidence")
                    if not set(value.symptom_ids) <= set(definitions[value.node_id].get("symptom_pool", [])):
                        raise ValueError("Unlisted semantic symptom")
                calculated = [calculate(node, rules[node["node_id"]], facts) for node in batch if rules[node["node_id"]]]
                by_id = {value.node_id: value for value in parsed + calculated}
                return [by_id[node["node_id"]] for node in batch], facts

            judgments, facts = client._request_structured_output_with_one_retry(request, validate, prompt,
                prompt + "\n上次不合法；完整重发全部facts和semantic nodes。未知必须显式保留；不得以删除节点、补造证据或把pending改met修复。")
            self.computation_log.append({"node_ids": [node["node_id"] for node in batch],
                "computed_nodes": sum(rule is not None for rule in rules.values()),
                "semantic_nodes": len(semantic), "facts": facts})
            answers.extend(judgments)
        return answers
