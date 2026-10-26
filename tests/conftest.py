import json
from pathlib import Path
import sys

import pytest

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))

from step3_reward_verifier import VerificationCase


@pytest.fixture
def case():
    path = WORKSPACE / "step3_reward_verifier/examples/interface_input.json"
    return VerificationCase.model_validate_json(path.read_text(encoding="utf-8"))


@pytest.fixture
def response_factory():
    def make(score=0.87):
        return {
            "score": score,
            "dimensions": [
                {"dimension": name, "score": score, "reason": "接口夹具评分",
                 "evidence_refs": ["ev-interface"] if name == "evidence_sufficiency" else []}
                for name in ("evidence_sufficiency", "logic_coherence", "differential_coverage", "no_reasoning_leaps")
            ],
            "fail_reasons": ["接口夹具：低分"] if score < 0.5 else [],
            "missing_dimensions": [{
                "field": "diagnostic_context.additional_detail",
                "purpose": "接口夹具：请求补充信息",
                "suggested_question": "请补充待核查的信息。",
            }] if score < 0.8 else [],
        }
    return make
