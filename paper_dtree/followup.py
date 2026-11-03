"""Interactive tree evaluation with explicit unknowns, separate from paper reproduction."""
from __future__ import annotations

import json

from .engine import DecisionTreeRunner


class _NeedsAnswer(Exception):
    is_followup_pause = True

    def __init__(self, task):
        self.task = task


class FollowUpRunner(DecisionTreeRunner):
    """Pause on an unresolved node and resume with added patient statements.

    Resuming replays recorded screening/node decisions, not API calls. The
    unresolved node is evaluated again against the augmented vignette. The
    instance belongs to one case and is not a durable session store.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._answers = {}
        self._node_judgments = {}
        self._patient_messages = {}
        self._evidence_nodes = []
        self._case_id = None
        self._pending = None

    def set_patient_evidence(self, messages, evidence_nodes):
        patient = {}
        for message in messages:
            if message.get("role") == "patient" and message.get("message_id") and message.get("text"):
                mid = message["message_id"]
                if mid in patient and patient[mid] != message["text"]:
                    raise ValueError("Conflicting patient message IDs")
                patient[mid] = message["text"]
        self._patient_messages = patient
        self._evidence_nodes = [e.model_dump(mode="json") if hasattr(e, "model_dump") else dict(e) for e in evidence_nodes]

    def _tri_state(self, raw):
        value = json.loads(raw)
        if not isinstance(value, dict) or set(value) != {"answer", "follow_up", "evidence"}:
            raise ValueError("Expected answer, follow_up and evidence")
        if value["answer"] not in {"yes", "no", "unknown"} or not isinstance(value["follow_up"], str):
            raise ValueError("Invalid tri-state answer")
        if value["answer"] == "unknown" and not value["follow_up"].strip():
            raise ValueError("Unknown requires a specific patient question")
        if not isinstance(value["evidence"], list):
            raise ValueError("Evidence must be a list")
        verified = []
        for item in value["evidence"]:
            if not isinstance(item, dict) or set(item) != {"message_id", "quote"}:
                raise ValueError("Evidence requires message_id and verbatim quote")
            mid, quote = item["message_id"], item["quote"]
            if not isinstance(mid, str) or not isinstance(quote, str) or not quote.strip() or quote not in self._patient_messages.get(mid, ""):
                raise ValueError("Evidence must quote an actual patient message")
            refs = [e["evidence_id"] for e in self._evidence_nodes if e.get("source_type") == "patient_message"
                    and e.get("message_id") == mid and isinstance(e.get("quote"), str) and e["quote"]
                    and (quote in e["quote"] or e["quote"] in quote)]
            verified.append({"message_id": mid, "quote": quote, "evidence_refs": refs,
                             "reference_status": "linked_l2_evidence" if refs else "patient_message_only"})
        if value["answer"] in {"yes", "no"} and not verified:
            raise ValueError("A yes/no node judgment requires patient evidence")
        value["evidence"] = verified
        return value

    def _ask(self, text, parse, *, stage, **context):
        if stage not in {"screening", "node"}:
            return super()._ask(text, parse, stage=stage, **context)
        key = (stage, context.get("entry_id") or context.get("tree_id"), context.get("node_id"))
        if key in self._answers:
            self.record({"event": "decision_replayed", "stage": stage, **context, "answer": self._answers[key]})
            return self._answers[key]
        if stage == "screening":
            result = super()._ask(text, parse, stage=stage, **context)
        else:
            instruction = ("Please answer the following question 'yes' or 'no' without explanation, based on the "
                           "facts stated in the clinical case. Answer with only the words 'yes' or 'no'. "
                           "If there is insufficient information, answer 'no'. ")
            replacement = ("Use only explicitly supported patient facts. Distinguish yes, no and unknown; "
                           "when evidence is absent or ambiguous answer unknown, never no. "
                           "Return only JSON with keys answer, follow_up and evidence. answer is yes, no, or unknown. "
                           "For each yes/no judgment, evidence must be a nonempty list of objects with exactly message_id and quote; "
                           "quote must be copied verbatim from the specified patient message below and support the actual answer. "
                           "Do not quote the question, assume absent facts are negative, invent IDs, or use assistant text as evidence. "
                           "For unknown use evidence=[] when no evidence exists. "
                           "For unknown provide one short, neutral Chinese question to the patient about the specific missing fact; "
                           "do not assert a diagnosis or assume symptoms. Otherwise follow_up is an empty string. "
                           "Patient messages (the only allowed evidence): "
                           + json.dumps([{"message_id": k, "text": v} for k, v in self._patient_messages.items()], ensure_ascii=False) + "\n")
            if instruction not in text:
                raise ValueError("Unexpected binary prompt template")
            result = super()._ask(text.replace(instruction, replacement, 1), self._tri_state, stage=stage, **context)
            if result["answer"] == "unknown":
                raise _NeedsAnswer({"tree_id": context["tree_id"], "node_id": context["node_id"],
                                    "question": result["follow_up"].strip(), "source": context.get("source"),
                                    "patient_evidence": result["evidence"]})
            self._node_judgments[(context["tree_id"], context["node_id"])] = result["evidence"]
            result = result["answer"] == "yes"
        self._answers[key] = result
        return result

    def _node_evidence(self, tree_id, node_id):
        return self._node_judgments.get((tree_id, node_id), [])

    def _candidate_evidence(self, result):
        """Only selected-branch ancestors of each emission belong to its path."""
        records = {(r["tree_id"], r["node_id"]): r for r in result.get("tree_trace", [])}
        output = []
        for diagnosis in result.get("candidates_before_pairwise", []):
            paths = []
            for origin in result.get("candidate_sources", {}).get(diagnosis, []):
                tid, end = origin["tree_id"], origin["node_id"]
                ancestors = {end}
                while True:
                    previous = {nid for (tree_id, nid), row in records.items() if tree_id == tid
                                and any(dest in ancestors for dest in row["branch"].get("next", []))}
                    expanded = ancestors | previous
                    if expanded == ancestors:
                        break
                    ancestors = expanded
                nodes = [{"node_id": r["node_id"], "criterion_question": r["question"],
                          "answer": r["answer"], "source": r["source"],
                          "patient_evidence": r.get("patient_evidence", []),
                          "evidence_status": "present" if r.get("patient_evidence") else "missing"}
                         for r in result["tree_trace"] if r["tree_id"] == tid and r["node_id"] in ancestors]
                paths.append({"tree_id": tid, "emission_node_id": end, "nodes": nodes})
            output.append({"diagnosis": diagnosis, "retained_after_pairwise": diagnosis in result.get("raw_diagnoses", []),
                           "criterion_scope": "source-derived tree questions, not a complete DSM criterion set",
                           "paths": paths})
        return output

    def run(self, case, *, case_id, method="dt"):
        if self._case_id is None:
            self._case_id = case_id
        elif self._case_id != case_id:
            raise ValueError("Interactive runner cannot be reused for a different case")
        result = super().run(case, case_id=case_id, method=method)
        self._pending = result.get("follow_up")
        result["candidate_evidence"] = self._candidate_evidence(result)
        result["policy"] = "tri_state_followup_v1"
        result["paper_protocol_deviation"] = True
        return result


class L2FollowUpSession:
    """Bridge an interactive runner to an existing L2 session.

    The caller must present each returned question to the patient, obtain a new
    L1 safety passport/message ID, update L2's passport, then submit the answer.
    Neither an LLM-inferred response nor a diagnostic label is written as fact.
    """

    def __init__(self, runner, l2_service, session_id, case, *, case_id):
        if not isinstance(runner, FollowUpRunner):
            raise TypeError("FollowUpRunner required")
        self.runner, self.l2, self.session_id = runner, l2_service, session_id
        self.case, self.case_id = case, case_id
        self.pending = None

    def _advance(self):
        from shared.clinical_contract import follow_up_task
        state = self.l2.store.snapshot(self.session_id)
        self.runner.set_patient_evidence(state.messages, state.evidence_nodes)
        result = self.runner.run(self.case, case_id=self.case_id)
        self.pending = result.get("follow_up")
        if self.pending:
            state = self.l2.store.snapshot(self.session_id)
            question_lower = self.pending["question"].lower()
            trauma_question = (self.pending["tree_id"] in {"p24.ptsd", "p24.asd"}
                               or any(term in question_lower for term in ("创伤", "性侵", "性暴力", "trauma", "sexual violence")))
            if "avoid_trauma_probing" in state.active_safety_controls and trauma_question:
                self.pending = None
                return {"status": "manual_review", "reason": "active_trauma_safety_control"}
            task_id = f"paper_dtree:{self.pending['tree_id']}:{self.pending['node_id']}"
            if not any(t.task_id == task_id and t.status == "pending" for t in state.follow_up_tasks):
                task = follow_up_task({"task_id": task_id, "field": "paper_dtree." + task_id,
                    "node_id": self.pending["node_id"], "purpose": "补充决策节点缺失信息",
                    "suggested_question": self.pending["question"], "source_refs": [str(self.pending.get("source"))]})
                state.follow_up_tasks.append(task)
                self.l2.store.replace(state)
        return result

    def start(self):
        if self.pending is not None or self.runner._case_id is not None:
            raise ValueError("Interactive session already started")
        return self._advance()

    def answer(self, patient_text, *, message_id):
        from layers.layer2.evidence import EvidenceRegistry
        from layers.layer2.schemas import ProfileFact
        if not self.pending or not patient_text.strip():
            raise ValueError("A pending question and nonempty patient response are required")
        pending = self.pending
        reply = self.l2.handle_message(self.session_id, patient_text, expected_input="free_text", message_id=message_id)
        if not reply.accepted:
            return {"status": "l2_rejected", "error_code": reply.error_code}
        state = self.l2.store.snapshot(self.session_id)
        if not any(m.get("message_id") == message_id and m.get("role") == "patient" and m.get("text") == patient_text for m in state.messages):
            raise ValueError("L2 did not persist the exact patient response")
        task_id = f"paper_dtree:{pending['tree_id']}:{pending['node_id']}"
        task = next(t for t in state.follow_up_tasks if t.task_id == task_id and t.status == "pending")
        evidence = EvidenceRegistry.add_patient_evidence(state, message_id=message_id, quote=patient_text,
                                                          target_path=f"follow_up_tasks.{task_id}")
        state.key_narratives.append(ProfileFact(value=patient_text, sources=[evidence.evidence_id], patient_quotes=[patient_text]))
        task.answer_evidence_refs.append(evidence.evidence_id)
        task.status = "unresolved"
        self.l2.store.replace(state)
        if state.confirmation_status == "paused" or (reply.l2_turn and reply.l2_turn.input_type in {"pause", "refusal"}):
            self.pending = None
            return {"status": "patient_declined_or_paused", "evidence_id": evidence.evidence_id}
        self.case += "\n\nPatient follow-up answer to " + pending["question"] + ": " + patient_text
        self.pending = None
        result = self._advance()
        state = self.l2.store.snapshot(self.session_id)
        task = next(t for t in state.follow_up_tasks if t.task_id == task_id)
        still_unknown = result.get("follow_up", {}).get("tree_id") == pending["tree_id"] and result.get("follow_up", {}).get("node_id") == pending["node_id"]
        if still_unknown:
            # Do not immediately repeat a question the patient could not answer.
            for item in state.follow_up_tasks:
                if item.task_id == task_id and item.status == "pending":
                    item.status = "unresolved"
            self.pending = None
            result = {"status": "unresolved", "reason": "patient_response_did_not_resolve_node", "follow_up": pending}
        else:
            task.status = "answered"
        self.l2.store.replace(state)
        result["patient_evidence_id"] = evidence.evidence_id
        return result
