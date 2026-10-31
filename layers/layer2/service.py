from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from shared.clinical_contract import DurationEvidence, follow_up_task
from typing import Any

from layers.layer2.guards import (
    L2EntranceGuard,
    L2EntranceGuardError,
    merge_persistent_controls,
)
from layers.layer2.llm import APIL2Model, L2Model, L2ModelOutputInvalid, L2ModelUnavailable
from layers.layer2.mi import MIResponseValidationError, MIResponseValidator
from layers.layer2.profile import ProfileIngestor, build_clinical_intake, is_sufficient
from layers.layer2.scales import ScaleAnswerError, ScaleEngine
from layers.layer2.schemas import (
    ClinicalIntake,
    ConfirmationDecision,
    DialogueResult,
    ExpectedDialogueInput,
    L2SessionState,
    L2Turn,
    PendingScaleItem,
    ScaleResult,
    SemanticAnalysis,
)
from layers.layer2.session import L2SessionStore, MIStageController


class L2AssessmentService:
    """L2 应用服务；默认使用生产 API，测试必须显式注入受控模型。"""

    def __init__(
        self,
        *,
        model: L2Model | None = None,
        store: L2SessionStore | None = None,
        entrance_guard: L2EntranceGuard | None = None,
        require_fresh_passport: bool = True,
    ) -> None:
        self.model = model or APIL2Model()
        self.store = store or L2SessionStore()
        self.entrance_guard = entrance_guard or L2EntranceGuard()
        self.require_fresh_passport = require_fresh_passport
        self.scales = ScaleEngine()
        self.ingestor = ProfileIngestor()
        self.stages = MIStageController()
        self.response_validator = MIResponseValidator()

    def create_session(
        self,
        safety_passport: dict[str, Any],
        session_id: str | None = None,
    ) -> str:
        sid = session_id or str(safety_passport.get("session_id") or uuid.uuid4().hex)
        self.entrance_guard.verify(safety_passport, session_id=sid)
        controls = merge_persistent_controls([], safety_passport)
        state = L2SessionState(
            session_id=sid,
            interview_date=datetime.now(timezone(timedelta(hours=8))).date().isoformat(),
            safety_passport=dict(safety_passport),
            active_safety_controls=controls,
        )
        self._sync_scale_skips(state)
        self.store.create(state)
        return sid

    def update_safety_passport(
        self, session_id: str, safety_passport: dict[str, Any]
    ) -> None:
        self.entrance_guard.verify(safety_passport, session_id=session_id)
        state = self.store.snapshot(session_id)
        if state.interview_date is None:
            state.interview_date = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
        state.active_safety_controls = merge_persistent_controls(
            state.active_safety_controls, safety_passport
        )
        state.safety_passport = dict(safety_passport)
        self._sync_scale_skips(state)
        self.store.replace(state)

    def handle_message(
        self,
        session_id: str,
        patient_text: str,
        *,
        expected_input: ExpectedDialogueInput = "auto",
        message_id: str | None = None,
    ) -> DialogueResult:
        if not patient_text.strip():
            return DialogueResult(
                accepted=False,
                reply="请告诉我你现在想表达的内容。",
                awaiting=expected_input,
                error_code="L2_EMPTY_INPUT",
            )

        original = self.store.get(session_id)
        if original.confirmation_status == "paused":
            return DialogueResult(
                accepted=False,
                reply="当前评估已经按你的选择暂停。",
                awaiting="profile_confirmation",
                error_code="L2_SESSION_PAUSED",
            )
        passport_message_id = original.safety_passport.get("message_id")
        if self.require_fresh_passport and (
            not isinstance(message_id, str)
            or not message_id
            or message_id != passport_message_id
            or message_id in original.consumed_l1_message_ids
        ):
            return DialogueResult(
                accepted=False,
                reply="安全接口暂时不可用，当前消息无法继续处理。",
                awaiting=expected_input,
                error_code="L2_SAFETY_MESSAGE_BINDING_INVALID",
            )

        state = self.store.snapshot(session_id)
        revising = state.confirmation_status == "revision_requested"
        if revising:
            state.confirmation_status = "not_requested"
        state.turn_index += 1
        evidence_message_id = message_id or uuid.uuid4().hex
        stage_before = state.current_stage
        answered_scale_item = self.scales.pending_item(state)
        answered_scale_score: int | None = None
        context = self._analysis_context(state, expected_input)
        try:
            analysis = self.model.analyze(patient_text, context)
        except L2ModelOutputInvalid:
            return self._model_output_error()
        except L2ModelUnavailable:
            return self._model_error()

        state.messages.append({
            "message_id": evidence_message_id,
            "role": "patient",
            "text": patient_text,
        })
        evidence_ids: list[str] = []
        triggered_scales: list[str] = []
        confirmation_decision: ConfirmationDecision | None = None
        response_override = ""

        if analysis.input_type == "pause":
            state.confirmation_status = "paused"
            confirmation_decision = "pause"
            response_override = "尊重患者选择，确认本次评估已暂停"
        elif state.current_stage == "planning" or expected_input == "profile_confirmation":
            confirmation_decision = self._confirmation_decision(analysis, patient_text)
            if confirmation_decision is None:
                response_override = "请患者明确选择信息准确并继续、需要修改或暂时停止"
            else:
                self._apply_confirmation(state, confirmation_decision)
                if (
                    confirmation_decision == "continue"
                    and state.confirmation_status != "confirmed"
                ):
                    confirmation_decision = None
                    response_override = "说明画像仍有信息缺失，邀请患者继续补充"
                else:
                    response_override = {
                        "continue": "感谢患者确认，并说明将把画像交给下一层进一步分析",
                        "revise": "感谢患者指出问题，邀请说明需要修改的具体信息",
                        "pause": "尊重患者选择，确认本次评估已暂停",
                    }[confirmation_decision]
        elif self._pending_consent(state):
            consent = self._structured_value(analysis, "scale_consent", patient_text)
            if not isinstance(consent, bool):
                response_override = "请患者明确说明是否愿意进行当前量表"
            else:
                self.scales.apply_consent(state, accepted=consent)
                if not consent:
                    response_override = "尊重患者拒绝量表的选择并继续自然访谈"
        elif self.scales.pending_item(state) is not None:
            try:
                self.scales.record_answer(
                    state,
                    message_id=evidence_message_id,
                    patient_text=patient_text,
                    analysis=analysis,
                )
                evidence_ids.extend(
                    node.evidence_id for node in state.evidence_nodes
                    if node.message_id == evidence_message_id
                )
                if answered_scale_item is not None:
                    answered_scale_score = state.scale_results[
                        answered_scale_item.scale_name
                    ].items[answered_scale_item.item_number]
            except ScaleAnswerError:
                response_override = "说明尚无法确定本题分值，请患者使用题目给出的数字或选项回答"
        else:
            evidence_ids.extend(self.ingestor.apply_corrections(
                state,
                analysis=analysis,
                patient_text=patient_text,
                message_id=evidence_message_id,
            ))
            evidence_ids.extend(self.ingestor.ingest(
                state,
                analysis=analysis,
                patient_text=patient_text,
                message_id=evidence_message_id,
            ))

        if state.current_stage == "engaging":
            self.stages.advance(state, analysis)

        suggested = None
        if not response_override and state.current_stage == "focusing":
            suggested = self.scales.suggest(state, analysis)
            if suggested:
                triggered_scales.append(suggested)

        if not suggested and not response_override:
            self.stages.advance(state, analysis)
        if revising and evidence_ids and is_sufficient(state) and not response_override:
            state.current_stage = "planning"
            state.confirmation_status = "pending"

        decision = self.stages.decide(
            state,
            stage_before=stage_before,
            analysis=analysis,
            response_override=response_override,
        )
        try:
            reply = self.model.generate(
                decision,
                self._response_context(
                    state,
                    patient_text,
                    answered_scale_item=answered_scale_item,
                    answered_scale_score=answered_scale_score,
                ),
            )
        except L2ModelUnavailable:
            return self._model_error()
        try:
            self.response_validator.assert_valid(
                reply,
                decision,
                active_safety_controls=state.active_safety_controls,
                patient_message=patient_text,
                scale_discussion_started=any(
                    item.status in {"pending_consent", "active", "completed", "declined"}
                    for item in state.scale_results.values()
                ),
            )
        except MIResponseValidationError:
            return DialogueResult(
                accepted=False,
                reply="对话回复未通过安全校验，当前消息未被记录，请重新尝试。",
                awaiting=expected_input,
                error_code="L2_RESPONSE_VALIDATION_FAILED",
            )

        if response_override and "无法确定本题分值" in response_override:
            return DialogueResult(
                accepted=False,
                reply=reply,
                awaiting=decision.awaiting,
                error_code="L2_SCALE_ANSWER_AMBIGUOUS",
            )

        state.missing_dimensions = build_clinical_intake(state).missing_dimensions
        if self.require_fresh_passport:
            state.consumed_l1_message_ids.append(evidence_message_id)
        state.messages.append({
            "message_id": f"{evidence_message_id}:l2",
            "role": "assistant",
            "text": reply,
        })
        self.store.replace(state)
        pending = self.scales.pending_item(state)
        turn = L2Turn(
            stage_before=stage_before,
            stage_after=state.current_stage,
            input_type=analysis.input_type,
            triggered_scales=triggered_scales,
            pending_scale_item=pending,
            evidence_ids=evidence_ids,
            transition_reason=decision.transition_reason,
            response_goal=decision.response_goal,
        )
        return DialogueResult(
            accepted=True,
            reply=reply,
            awaiting=decision.awaiting,
            l2_turn=turn,
            confirmation_decision=confirmation_decision,
        )

    def confirm_profile(
        self, session_id: str, decision: ConfirmationDecision
    ) -> ClinicalIntake:
        state = self.store.snapshot(session_id)
        self._apply_confirmation(state, decision)
        state.missing_dimensions = build_clinical_intake(state).missing_dimensions
        self.store.replace(state)
        return build_clinical_intake(state)

    def resume_profile_confirmation(self, session_id: str) -> ClinicalIntake:
        state = self.store.snapshot(session_id)
        if state.confirmation_status != "paused":
            raise ValueError("L2画像确认当前没有暂停")
        if not is_sufficient(state):
            state.confirmation_status = "revision_requested"
            state.current_stage = "focusing"
        else:
            state.confirmation_status = "pending"
            state.current_stage = "planning"
        self.store.replace(state)
        return build_clinical_intake(state)

    def build_clinical_intake(self, session_id: str) -> ClinicalIntake:
        return build_clinical_intake(self.store.snapshot(session_id))

    def receive_l3_follow_up(self, session_id: str, items: list[dict] | None, durations: list[dict] | None = None) -> None:
        state = self.store.snapshot(session_id)
        if items is not None:
            state.follow_up_tasks = [follow_up_task(item) for item in items]
        known = {(d.raw_span, d.scope, tuple(d.evidence_refs), d.source) for d in state.diagnostic_context.duration_evidence}
        for value in durations or []:
            record = DurationEvidence.model_validate(value)
            key = (record.raw_span, record.scope, tuple(record.evidence_refs), record.source)
            if key not in known:
                state.diagnostic_context.duration_evidence.append(record)
                known.add(key)
        self.store.replace(state)

    def get_session_snapshot(self, session_id: str) -> L2SessionState:
        return self.store.snapshot(session_id)

    @staticmethod
    def _apply_confirmation(
        state: L2SessionState, decision: ConfirmationDecision
    ) -> None:
        if decision == "continue":
            if not is_sufficient(state):
                return
            state.confirmation_status = "confirmed"
        elif decision == "revise":
            state.confirmation_status = "revision_requested"
            state.current_stage = "focusing"
        else:
            state.confirmation_status = "paused"

    @staticmethod
    def _structured_value(
        analysis: SemanticAnalysis, field: str, patient_text: str
    ) -> Any:
        for candidate in analysis.structured_fields:
            if candidate.field == field and candidate.evidence_quote in patient_text:
                return candidate.value
        return None

    def _confirmation_decision(
        self, analysis: SemanticAnalysis, patient_text: str
    ) -> ConfirmationDecision | None:
        value = self._structured_value(
            analysis, "confirmation_decision", patient_text
        )
        return value if value in {"continue", "revise", "pause"} else None

    @staticmethod
    def _pending_consent(state: L2SessionState) -> bool:
        if not state.pending_scale_name:
            return False
        return state.scale_results[state.pending_scale_name].status == "pending_consent"

    @staticmethod
    def _analysis_context(
        state: L2SessionState, expected_input: ExpectedDialogueInput
    ) -> dict[str, Any]:
        pending = ScaleEngine.pending_item(state)
        return {
            "current_stage": state.current_stage,
            "expected_input": expected_input,
            "pending_scale": state.pending_scale_name,
            "pending_scale_item": pending.model_dump(mode="json") if pending else None,
            "active_safety_controls": state.active_safety_controls,
            "validated_focus": [item.value for item in state.focus_topics],
            "missing_dimensions": state.missing_dimensions,
            "follow_up_task": next((t.model_dump(mode="json") for t in state.follow_up_tasks if t.status == "pending"), None),
            "interview_date": state.interview_date,
            "symptom_context": [s.model_dump(mode="json") for s in state.symptoms.values()],
        }

    @staticmethod
    def _response_context(
        state: L2SessionState,
        patient_text: str,
        *,
        answered_scale_item: PendingScaleItem | None = None,
        answered_scale_score: int | None = None,
    ) -> dict[str, Any]:
        facts = [
            *(f"症状:{item.symptom}" for item in state.symptoms.values()),
            *(f"情绪:{item.value}" for item in state.emotions),
            *(f"目标:{item.value}" for item in state.patient_goals),
            *(f"重点:{item.value}" for item in state.focus_topics),
        ]
        completed_history = list(state.messages)
        if (
            completed_history
            and completed_history[-1].get("role") == "patient"
            and completed_history[-1].get("text") == patient_text
        ):
            completed_history.pop()
        recent_dialogue = [
            {"role": item.get("role", ""), "text": item.get("text", "")}
            for item in completed_history[-6:]
            if item.get("role") in {"patient", "assistant"}
        ]
        return {
            "patient_message": patient_text,
            "validated_facts": facts,
            "active_safety_controls": state.active_safety_controls,
            "recent_dialogue": recent_dialogue,
            "answered_scale_item": (
                answered_scale_item.model_dump(mode="json")
                if answered_scale_item else None
            ),
            "answered_scale_score": answered_scale_score,
        }

    @staticmethod
    def _model_error() -> DialogueResult:
        return DialogueResult(
            accepted=False,
            reply="对话评估模型暂时不可用，当前消息未被记录，本次会话已结束。",
            error_code="L2_MODEL_UNAVAILABLE",
        )

    @staticmethod
    def _model_output_error() -> DialogueResult:
        return DialogueResult(
            accepted=False,
            reply="对话评估模型返回的数据格式不符合要求，当前消息未被记录，本次会话已结束。",
            error_code="L2_MODEL_OUTPUT_INVALID",
        )

    @staticmethod
    def _sync_scale_skips(state: L2SessionState) -> None:
        for scale, current in list(state.scale_results.items()):
            control = f"skip_scale:{scale}"
            if (
                current.status == "skipped"
                and current.skip_reason == "L1安全控制禁止触发该量表"
                and control not in state.active_safety_controls
            ):
                state.scale_results[scale] = ScaleResult(scale_name=scale)
        for control in state.active_safety_controls:
            if not control.startswith("skip_scale:"):
                continue
            scale = control.split(":", 1)[1]
            current = state.scale_results.get(scale)
            if current is None or current.status in {
                "not_started", "pending_consent", "active"
            }:
                state.scale_results[scale] = ScaleResult(
                    scale_name=scale,
                    status="skipped",
                    skip_reason="L1安全控制禁止触发该量表",
                )
                if state.pending_scale_name == scale:
                    state.pending_scale_name = None
