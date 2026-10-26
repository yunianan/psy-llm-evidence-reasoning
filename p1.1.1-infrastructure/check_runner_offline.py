import json
from types import SimpleNamespace
from unittest.mock import patch

from common import HERE, run_case, node_client, cost
from tests.tree_test_support import node_payload


def main():
    root = HERE / "offline-runner-fixture-v1"
    assert not root.exists()
    root.mkdir()
    transport = SimpleNamespace(bind=lambda *args, **kwargs: None, tags=lambda *args: None)
    row = {"case_id": "synthetic-offline", "profile": {"key_narratives": [{"value": "Synthetic offline fixture", "sources": ["ev"]}]}}
    with patch.object(node_client, "ask_llm_stream", lambda prompt, **kw: json.dumps(node_payload(prompt, labels=["D013"], ref="ev"))):
        record = run_case(row, arm="fixture", thinking=True, run=root, transport=transport,
                          checkpoint_root=root / "partial", g1_only=True)
    assert record["ok"] and record["g1_shared_validation"]["identical"]
    assert not record["requests"] and record["checkpoint_stats"]["new_nodes"] > 0
    cost.write(root / "verification.json", {"synthetic_only": True, "ok": True, "new_model_calls": 0,
                                          "checkpoint_nodes": record["checkpoint_nodes_after"], "shared_g1_identical": True})
    print("Synthetic full G1 + checkpoints + A2/A3 replay: passed, 0 model calls")


if __name__ == "__main__":
    main()
