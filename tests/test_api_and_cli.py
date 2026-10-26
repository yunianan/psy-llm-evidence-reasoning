import json
from pathlib import Path
import subprocess
import sys

import pytest

from step3_reward_verifier import APIRewardModelVerifier, VerifierModelUnavailable, verify_case
from step3_reward_verifier import api

WORKSPACE = Path(__file__).resolve().parents[2]


def test_api_requires_caller_supplied_standard_knowledge(case):
    with pytest.raises(VerifierModelUnavailable, match="标准知识快照"):
        verify_case(case, APIRewardModelVerifier(model="offline-test"))


def test_api_repairs_once_and_uses_supplied_knowledge(case, response_factory, monkeypatch):
    invalid = response_factory()
    invalid["dimensions"][0]["evidence_refs"] = ["invented"]
    replies = iter([invalid, response_factory()])
    prompts = []
    def request(prompt, **kwargs):
        prompts.append(prompt)
        return json.dumps(next(replies))
    monkeypatch.setattr(api, "ask_llm_stream", request)
    scorer = APIRewardModelVerifier(
        model="offline-test", knowledge_provider=lambda disorder: {"test_only": "supplied-snapshot-marker"},
    )
    result = verify_case(case, scorer)
    assert result["routing"] == "pass"
    assert len(prompts) == 2
    assert all("supplied-snapshot-marker" in prompt for prompt in prompts)
    assert all('"confidence"' not in prompt for prompt in prompts)


def test_api_fails_after_one_invalid_schema_repair(case, monkeypatch):
    calls = []
    def request(prompt, **kwargs):
        calls.append(prompt)
        return "{}"
    monkeypatch.setattr(api, "ask_llm_stream", request)
    scorer = APIRewardModelVerifier(model="offline-test", knowledge_provider=lambda disorder: {"test_only": True})
    with pytest.raises(VerifierModelUnavailable):
        verify_case(case, scorer)
    assert len(calls) == 2


def test_cli_replay_runs_without_layer3_or_api(tmp_path):
    output = tmp_path / "replay.json"
    completed = subprocess.run([
        sys.executable, "-B", "-m", "step3_reward_verifier",
        "--input", "step3_reward_verifier/examples/interface_input.json",
        "--response", "step3_reward_verifier/examples/pass_response.json",
        "--output", str(output),
    ], cwd=WORKSPACE, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["scoring_mode"] == "replay"
    assert result["routing"] == "pass"


def test_module_import_does_not_load_main_system():
    code = (
        "import sys; import step3_reward_verifier; "
        "assert not any(name == 'layers' or name.startswith('layers.') "
        "or name == 'shared' or name.startswith('shared.') for name in sys.modules)"
    )
    completed = subprocess.run([sys.executable, "-B", "-c", code], cwd=WORKSPACE, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_cli_requires_explicit_replay_or_api_mode():
    completed = subprocess.run([
        sys.executable, "-B", "-m", "step3_reward_verifier",
        "--input", "step3_reward_verifier/examples/interface_input.json",
    ], cwd=WORKSPACE, capture_output=True, text=True)
    assert completed.returncode != 0
