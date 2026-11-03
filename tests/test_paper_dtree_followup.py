import json
from types import SimpleNamespace

from layers.layer2.schemas import L2SessionState
from layers.layer2.session.store import L2SessionStore
from paper_dtree.engine import PAIRWISE_POLICY, RetryPolicy
from paper_dtree.followup import FollowUpRunner, L2FollowUpSession
from paper_dtree.models import Bundle, Response
from tests.test_paper_dtree import bundle_data, node, tree


class Scripted:
    def __init__(self, answers): self.answers, self.requests = iter(answers), []
    def complete(self, request):
        self.requests.append(request)
        return Response(next(self.answers), model="fixture")


class L2Fixture:
    def __init__(self, accepted=True):
        self.accepted = accepted
        self.store = L2SessionStore()
        self.store.create(L2SessionState(session_id="s", safety_passport={}, turn_index=1,
                                        messages=[{"message_id": "original", "role": "patient", "text": "Original case"}]))
    def handle_message(self, session_id, text, *, expected_input, message_id):
        if self.accepted:
            state = self.store.snapshot(session_id)
            state.turn_index += 1
            state.messages.append({"message_id": message_id, "role": "patient", "text": text})
            self.store.replace(state)
        return SimpleNamespace(accepted=self.accepted, error_code=None if self.accepted else "safety", l2_turn=None)


def make(answers, accepted=True):
    bundle = Bundle.parse(bundle_data(tree("T", {"onset": node("Onset before age 12?", yes={"diagnoses": ["Dx-A"]})})))
    transport = Scripted(answers)
    runner = FollowUpRunner(bundle, transport, model="fixture", pairwise_policy=PAIRWISE_POLICY,
                            retry=RetryPolicy(max_attempts=1, initial_seconds=0))
    l2 = L2Fixture(accepted)
    return L2FollowUpSession(runner, l2, "s", "Original case", case_id="case"), l2, transport


def test_unknown_pauses_then_patient_answer_updates_l2_and_resumes_without_rescreening():
    session, l2, api = make(["yes", json.dumps({"answer": "unknown", "follow_up": "12岁以前是否有注意力或活动方面的困难？", "evidence": []}),
                             json.dumps({"answer": "yes", "follow_up": "", "evidence": [{"message_id": "patient-1", "quote": "上小学时经常走神。"}]})])
    first = session.start()
    assert first["status"] == "awaiting_patient" and first["follow_up"]["node_id"] == "onset"
    assert len(l2.store.get("s").follow_up_tasks) == 1
    result = session.answer("上小学时经常走神。", message_id="patient-1")
    assert result["status"] == "success" and result["raw_diagnoses"] == ["Dx-A"]
    assert len(api.requests) == 3
    assert "上小学时经常走神。" in api.requests[-1].messages[1]["content"]
    state = l2.store.get("s")
    assert state.key_narratives[-1].patient_quotes == ["上小学时经常走神。"]
    assert state.follow_up_tasks[0].status == "answered"
    assert state.follow_up_tasks[0].answer_evidence_refs == [state.evidence_nodes[-1].evidence_id]
    criterion = result["candidate_evidence"][0]["paths"][0]["nodes"][0]
    assert criterion["answer"] == "yes" and criterion["source"]["document"]
    assert criterion["patient_evidence"][0]["evidence_refs"] == [state.evidence_nodes[-1].evidence_id]


def test_l2_rejection_does_not_write_answer_or_resume():
    session, l2, api = make(["yes", '{"answer":"unknown","follow_up":"请问小时候是否有这些困难？","evidence":[]}'], accepted=False)
    session.start()
    result = session.answer("有。", message_id="patient-1")
    assert result["status"] == "l2_rejected" and len(api.requests) == 2
    assert not l2.store.get("s").evidence_nodes


def test_remains_unresolved_without_repeated_automatic_question():
    unknown = '{"answer":"unknown","follow_up":"请问小时候是否有这些困难？","evidence":[]}'
    session, l2, _ = make(["yes", unknown, unknown])
    session.start()
    result = session.answer("我不记得了。", message_id="patient-1")
    assert result["status"] == "unresolved"
    assert all(t.status == "unresolved" for t in l2.store.get("s").follow_up_tasks)


def test_seven_stage_pipeline_stops_at_follow_up_without_scoring():
    from paper_dtree.pipeline import SevenStagePipeline
    from paper_dtree.postprocess import Catalog, ReconciliationStore
    from tests.test_paper_pipeline import catalog_data
    session, _, _ = make(["yes", '{"answer":"unknown","follow_up":"小时候是否有这些困难？","evidence":[]}'])
    pipeline = SevenStagePipeline(Catalog.parse(catalog_data()), ReconciliationStore(), runner=session.runner)
    result = pipeline.run("Original case", case_id="case", reference=["Dx-A"], reference_source="fixture")
    assert result["status"] == "awaiting_patient"
    assert result["stages"][1]["status"] == "awaiting_patient"
    assert all(stage["status"] == "not_started" for stage in result["stages"][2:])
    assert "scoring" not in result


def test_cannot_accept_fabricated_or_unreferenced_patient_evidence():
    import pytest
    from paper_dtree.engine import RunFailed
    fake = json.dumps({"answer": "yes", "follow_up": "", "evidence": [{"message_id": "original", "quote": "invented"}]})
    session, _, _ = make(["yes", fake])
    with pytest.raises(RunFailed): session.start()
    session, _, _ = make(["yes", '{"answer":"no","follow_up":"","evidence":[]}'])
    with pytest.raises(RunFailed): session.start()


def test_each_candidate_gets_only_its_own_branch_and_pairwise_status():
    bundle = Bundle.parse(bundle_data(tree("T", {
        "root": node("Root?", yes={"next": ["left", "right"]}),
        "left": node("Left?", yes={"diagnoses": ["Dx-A"]}),
        "right": node("Right?", yes={"diagnoses": ["Dx-B"]}),
    }), diagnoses=["Dx-A", "Dx-B"]))
    def decision(quote):
        return json.dumps({"answer": "yes", "follow_up": "", "evidence": [{"message_id": "m", "quote": quote}]})
    api = Scripted(["yes", decision("root fact"), decision("left fact"), decision("right fact"), '["Dx-A"]'])
    runner = FollowUpRunner(bundle, api, model="fixture", pairwise_policy=PAIRWISE_POLICY,
                            retry=RetryPolicy(max_attempts=1, initial_seconds=0))
    runner.set_patient_evidence([{"message_id": "m", "role": "patient", "text": "root fact; left fact; right fact"}], [])
    result = runner.run("root fact; left fact; right fact", case_id="c")
    by_name = {c["diagnosis"]: c for c in result["candidate_evidence"]}
    assert [n["node_id"] for n in by_name["Dx-A"]["paths"][0]["nodes"]] == ["root", "left"]
    assert [n["node_id"] for n in by_name["Dx-B"]["paths"][0]["nodes"]] == ["root", "right"]
    assert by_name["Dx-A"]["retained_after_pairwise"] is True
    assert by_name["Dx-B"]["retained_after_pairwise"] is False
    assert by_name["Dx-A"]["paths"][0]["nodes"][0]["patient_evidence"][0]["reference_status"] == "patient_message_only"
