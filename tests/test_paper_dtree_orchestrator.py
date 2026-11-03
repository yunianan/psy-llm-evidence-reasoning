import json

from paper_dtree.engine import PAIRWISE_POLICY, RetryPolicy
from paper_dtree.followup import FollowUpRunner, L2FollowUpSession
from paper_dtree.models import Bundle, Response
from tests.test_paper_dtree import bundle_data, node, tree
from tests.test_top_level_orchestrator import _complete_l2, _orchestrator, _safe_passport


class Transport:
    def __init__(self):
        self.values = iter(["yes", json.dumps({"answer": "unknown", "follow_up": "你12岁前是否经常注意力不集中？", "evidence": []}),
                            None])
        self.calls = []
    def complete(self, request):
        self.calls.append(request)
        value = next(self.values)
        if value is None:
            text = request.messages[1]["content"]
            import re
            match = re.search(r'"message_id": "([^"]+)", "text": "上小学时经常走神。"', text)
            assert match
            value = json.dumps({"answer": "yes", "follow_up": "", "evidence": [{"message_id": match.group(1), "quote": "上小学时经常走神。"}]})
        return Response(value, model="fixture")


def test_l1_l2_tree_followup_resumes_and_preserves_patient_quote(monkeypatch, tmp_path):
    seen = []
    def safety(text, **kwargs):
        seen.append(text)
        return _safe_passport(text, **kwargs)
    monkeypatch.setattr("orchestration.orchestrator.safety_detect", safety)
    orch = _orchestrator(tmp_path)
    transport = Transport()
    bundle = Bundle.parse(bundle_data(tree("T", {"onset": node("Onset before age 12?", yes={"diagnoses": ["Dx-A"]})})))
    def factory(sid, case):
        runner = FollowUpRunner(bundle, transport, model="fixture", pairwise_policy=PAIRWISE_POLICY,
                                retry=RetryPolicy(max_attempts=1, initial_seconds=0))
        return L2FollowUpSession(runner, orch.l2, sid, case, case_id=sid)
    orch.paper_tree_factory = factory
    orch.create_session("tree-e2e")
    assert _complete_l2(orch, "tree-e2e").route == "AWAITING_PATIENT_CONFIRMATION"
    prompt = orch.confirm_patient_profile("tree-e2e", "continue")
    assert prompt.route == "RETURN_TO_L2" and "12岁前" in prompt.reply
    response = orch.handle_message("tree-e2e", "上小学时经常走神。")
    assert seen[-1] == "上小学时经常走神。"
    assert response.route == "MANUAL_REVIEW"
    state = orch.l2.get_session_snapshot("tree-e2e")
    task = next(t for t in state.follow_up_tasks if t.node_id == "onset")
    assert task.status == "answered" and task.answer_evidence_refs
    assert any(f.patient_quotes == ["上小学时经常走神。"] for f in state.key_narratives)
    tree_result = orch.get_session_snapshot("tree-e2e").l3_raw["model_result"]
    assert tree_result["raw_diagnoses"] == ["Dx-A"]
    evidence = tree_result["candidate_evidence"][0]["paths"][0]["nodes"][0]["patient_evidence"][0]
    assert evidence["quote"] == "上小学时经常走神。" and evidence["evidence_refs"] == task.answer_evidence_refs
    assert len(transport.calls) == 3


def test_l1_crisis_stops_before_followup_response_reaches_l2(monkeypatch, tmp_path):
    monkeypatch.setattr("orchestration.orchestrator.safety_detect", _safe_passport)
    orch = _orchestrator(tmp_path)
    transport = Transport()
    bundle = Bundle.parse(bundle_data(tree("T", {"onset": node("Onset?", yes={"diagnoses": ["Dx-A"]})})))
    orch.paper_tree_factory = lambda sid, case: L2FollowUpSession(
        FollowUpRunner(bundle, transport, model="fixture", pairwise_policy=PAIRWISE_POLICY,
                       retry=RetryPolicy(max_attempts=1, initial_seconds=0)), orch.l2, sid, case, case_id=sid)
    orch.create_session("tree-crisis")
    _complete_l2(orch, "tree-crisis")
    assert orch.confirm_patient_profile("tree-crisis", "continue").route == "RETURN_TO_L2"
    response = orch.handle_message("tree-crisis", "立即危机")
    assert response.route == "STOPPED" and len(transport.calls) == 2
    assert not any(m.get("text") == "立即危机" for m in orch.l2.get_session_snapshot("tree-crisis").messages)
