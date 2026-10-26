"""One synthetic full-run/cache/prefix/reporter integration, zero API calls."""
import json
import argparse
from unittest.mock import patch

import run_experiment as experiment
from common import cost
from tests.tree_test_support import node_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="offline-runner-v1")
    args = parser.parse_args()
    fixture = experiment.HERE / args.fixture
    assert not fixture.exists(), "Keep the original synthetic integration result"
    fixture.mkdir()
    calls = []
    def respond(prompt, **kwargs):
        calls.append("computed" if "原始知识与任务：" in prompt else "full")
        if "原始知识与任务：" not in prompt:
            return json.dumps(node_payload(prompt, labels=["D013"], ref="ev-mood"))
        packet, _ = json.JSONDecoder().raw_decode(prompt.split("原始知识与任务：", 1)[1])
        values = []
        for key, spec in packet["requested_facts"].items():
            value = {"fact_id": key, "evidence_refs": [], "confidence": .5}
            if spec["kind"] == "duration":
                value.update(lower=None, upper=None, unit=None)
            else:
                value["state"] = "unknown"
            values.append(value)
        return json.dumps({"facts": values, "nodes": []})
    class Transport:
        def bind(self, *args, **kwargs):
            pass
        def tags(self, *args):
            pass
    row = {"case_id": "synthetic-computed-runner", "profile": {"evidence_nodes": [{"evidence_id": "ev-mood", "quote": "synthetic"}]}}
    transport = Transport()
    with patch.object(experiment, "HERE", fixture), patch.object(experiment, "RUN", fixture), patch.object(experiment.infra.node_client, "ask_llm_stream", respond):
        full = experiment.infra.run_case(row, arm="full_llm", thinking=True, run=fixture, transport=transport,
                                         checkpoint_root=fixture / "checkpoints/full")
        computed = experiment.hybrid_case(row, transport)
        comparison = experiment.comparison(full, computed)
    cost.write(fixture / "full-record.json", full)
    cost.write(fixture / "computed-record.json", computed)
    cost.write(fixture / "comparison.json", comparison)
    assert full["ok"] and computed["ok"] and comparison["passed"] and comparison["final_identical"]
    assert full["g1_shared_validation"]["identical"] and computed["g1_shared_validation"]["identical"]
    result = {"passed": True, "network_model_calls": 0, "synthetic_model_responses": len(calls),
        "computed_responses": calls.count("computed"), "shared_prefix_nodes": computed["shared_prefix_nodes"],
        "g1_replays_identical": True, "final_snapshot_identical": True, "comparison": comparison}
    cost.write(fixture / "verification.json", result)
    print(json.dumps({key: result[key] for key in ["passed", "network_model_calls", "computed_responses", "shared_prefix_nodes", "g1_replays_identical", "final_snapshot_identical"]}))


if __name__ == "__main__":
    main()
