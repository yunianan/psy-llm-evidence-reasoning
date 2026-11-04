"""Bind the standalone ICD-10 session to L1-screened L2 patient evidence."""

import os

from layers.layer2.evidence import EvidenceRegistry
from layers.layer2.schemas import ClinicalIntake, ProfileFact
from shared.clinical_contract import follow_up_task
from shared.llm_client import get_api_key, get_base_url, get_model


from icd10_tree import Session, build_catalog
from icd10_tree.transport import APIJudge


class ICD10FollowUpSession:
    def __init__(self, l2_service, session_id: str, intake: ClinicalIntake):
        self.l2 = l2_service
        self.session_id = session_id
        self.intake = intake.model_dump(mode="json")
        shared_base_url = get_base_url()
        base_url = (os.getenv("ICD10_BASE_URL") or shared_base_url).strip().rstrip("/")
        api_key = os.getenv("ICD10_API_KEY") or (get_api_key() if base_url == shared_base_url else "")
        model = os.getenv("ICD10_MODEL") or get_model()
        self.judge = APIJudge(api_key=api_key, base_url=base_url, model=model)
        self.session = Session(build_catalog())

    def _judge_with_profile(self, node, messages):
        return self.judge(node, messages, clinical_profile=self.intake)

    @staticmethod
    def _task_id(pending):
        return f"icd10:{pending['node_id']}"

    def _route_question(self, result):
        if result["status"] != "awaiting_patient":
            return result

        pending = result["pending"]
        state = self.l2.get_session_snapshot(self.session_id)
        question = pending["question"].lower()
        trauma = (pending["tree_id"] == "ptsd"
                  or any(term in question for term in ("创伤", "性侵", "性暴力", "trauma", "sexual violence")))
        if trauma and "avoid_trauma_probing" in state.active_safety_controls:
            return {**result, "status": "manual_review", "reason": "active_trauma_safety_control"}

        task_id = self._task_id(pending)
        source = pending["source"]
        document = result["documents"][source["document_id"]]
        state.follow_up_tasks.append(follow_up_task({
            "task_id": task_id,
            "field": f"icd10.{pending['node_id']}",
            "node_id": pending["node_id"],
            "purpose": "补充 ICD-10 节点缺失的患者资料",
            "suggested_question": pending["question"],
            "source_refs": [f"{document['publication_url']} (PDF p.{source['pdf_page']})"],
        }))
        self.l2.store.replace(state)
        return result

    def start(self):
        if self.session.state["status"] != "new":
            raise ValueError("ICD-10 assessment already started")
        state = self.l2.get_session_snapshot(self.session_id)
        for message in state.messages:
            if message.get("role") == "patient" and message.get("text", "").strip():
                self.session.add_message(message["text"], message_id=message["message_id"])
        return self._route_question(self.session.advance(self._judge_with_profile))

    def answer(self, patient_text, *, message_id):
        pending = self.session.state["pending"]
        if not pending:
            raise ValueError("No pending ICD-10 question")
        reply = self.l2.handle_message(self.session_id, patient_text, expected_input="free_text", message_id=message_id)
        if not reply.accepted:
            return {"status": "l2_rejected", "error_code": reply.error_code}

        state = self.l2.get_session_snapshot(self.session_id)
        if not any(message.get("message_id") == message_id and message.get("role") == "patient"
                   and message.get("text") == patient_text for message in state.messages):
            raise ValueError("L2 did not persist the exact patient response")
        task_id = self._task_id(pending)
        task = next(t for t in state.follow_up_tasks if t.task_id == task_id and t.status == "pending")
        evidence = EvidenceRegistry.add_patient_evidence(
            state, message_id=message_id, quote=patient_text, target_path=f"follow_up_tasks.{task_id}",
        )
        state.key_narratives.append(ProfileFact(
            value=patient_text, sources=[evidence.evidence_id], patient_quotes=[patient_text],
        ))
        task.answer_evidence_refs.append(evidence.evidence_id)
        task.status = "unresolved"
        self.l2.store.replace(state)

        if state.confirmation_status == "paused" or (reply.l2_turn and reply.l2_turn.input_type in {"pause", "refusal"}):
            return {**self.session.result(), "status": "patient_declined_or_paused",
                    "patient_evidence_id": evidence.evidence_id}

        self.intake = self.l2.build_clinical_intake(self.session_id).model_dump(mode="json")
        result = self.session.answer(patient_text, self._judge_with_profile, message_id=message_id)
        if self.session.state["decisions"][pending["node_id"]]["answer"] != "unknown":
            state = self.l2.get_session_snapshot(self.session_id)
            task = next(t for t in state.follow_up_tasks if t.task_id == task_id
                        and evidence.evidence_id in t.answer_evidence_refs)
            task.status = "answered"
            self.l2.store.replace(state)
        return self._route_question({**result, "patient_evidence_id": evidence.evidence_id})
