import uuid
from pathlib import Path
from typing import Any, Callable, cast

from layers.layer1.detector import reset_session as reset_l1_session
from layers.layer1.detector import safety_detect, start_baseline_asq
from layers.layer2 import (
    L2AssessmentService,
    L2DialogueAdapter,
    L2EntranceGuardError,
    L2ToL3Adapter,
)
from layers.layer2.schemas import ClinicalIntake, ExpectedDialogueInput
from layers.layer3.api import diagnose
from layers.layer3.instructions.turn_planner import TurnInstructionPlanner, TreatmentPlanningUnavailable
from layers.layer4.schemas.instructions import (
    SafetyPassport,
)
from layers.layer4.schemas.l3_report import DiagnosticReport
from layers.layer4.service import L4Service, SessionRequest, TurnRequest
from layers.layer4.validation import L3ToL4Gate
from patient_presentation import (
    DeveloperProfileRenderer,
    build_patient_confirmation,
)
from orchestration.schemas import (
    ConfirmationDecision,
    OrchestrationOutcome,
    PatientConfirmation,
    PipelineSession,
)
from orchestration.session_store import PipelineSessionStore


DIAGNOSTIC_EXPECTATIONS = {
    "symptom_duration_days",
    "functional_impairment",
    "mania_or_hypomania_history",
    "substance_or_medical_cause",
    "psychotic_symptoms",
    "differential_diagnoses_assessed",
    "evidence_conflicts",
}


class FiveLayerOrchestrator:
    """五层系统的应用级控制器；只路由，不重做各层临床判断。"""

    def __init__(
        self,
        *,
        l2_service: L2AssessmentService | None = None,
        l4_service: L4Service | None = None,
        store: PipelineSessionStore | None = None,
        profile_renderer: DeveloperProfileRenderer | None = None,
        l3_diagnose: Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]] | None = None,
        max_l3_to_l2_loops: int = 3,
        instruction_planner: TurnInstructionPlanner | None = None,
    ) -> None:
        if max_l3_to_l2_loops < 1:
            raise ValueError("max_l3_to_l2_loops 必须至少为1")
        self.l2 = l2_service or L2AssessmentService()
        self.dialogue = L2DialogueAdapter(self.l2)
        self.l2_to_l3 = L2ToL3Adapter()
        self.l4 = l4_service or L4Service()
        self.store = store or PipelineSessionStore()
        self.profile_renderer = profile_renderer or DeveloperProfileRenderer()
        self.l3_diagnose = l3_diagnose or diagnose
        self.max_l3_to_l2_loops = max_l3_to_l2_loops
        self.instruction_planner = instruction_planner or TurnInstructionPlanner()

    def create_session(self, session_id: str | None = None) -> OrchestrationOutcome:
        sid = session_id or uuid.uuid4().hex
        reset_l1_session(sid)
        session = PipelineSession(session_id=sid)
        self.store.create(session)
        return self._finish(
            session,
            reply="我们可以先从你现在最希望获得帮助的事情开始。",
            waiting_for="free_text",
        )

    def handle_message(self, session_id: str, user_message: str) -> OrchestrationOutcome:
        session = self.store.get(session_id)
        if not user_message.strip():
            return self._finish(
                session,
                reply="请输入你想说的内容。",
                waiting_for=session.expected_input,
            )
        if session.route == "STOPPED":
            return self._finish(
                session,
                reply=session.stopped_reason or "当前流程已经停止。",
                terminal=True,
            )
        if session.route == "PAUSED_BY_PATIENT":
            return self._finish(
                session,
                reply="当前流程已按你的选择暂停。如需继续，请先恢复会话。",
                waiting_for="resume",
            )

        passport = safety_detect(
            user_message,
            session_id=session.session_id,
            message_id=uuid.uuid4().hex,
            assessment_context=self._l1_assessment_context(session),
        )
        if not isinstance(passport, dict):
            session.l1_passport = {
                "malformed_result_type": type(passport).__name__,
            }
            session.route = "STOPPED"
            session.stopped_reason = "L1安全出口校验失败，当前消息未进入后续流程。"
            session.internal_flags.append("l1_exit_guard_rejected")
            return self._finish(
                session,
                reply=session.stopped_reason,
                terminal=True,
            )
        session.l1_passport = passport
        if passport.get("status") == "error":
            session.route = "STOPPED"
            session.stopped_reason = (
                "安全评估API不可用"
                if passport.get("error_code") == "L1_API_ERROR"
                else "安全评估模型输出不合法"
            )
            return self._finish(
                session,
                reply=session.stopped_reason + "，当前无法继续处理这条消息。",
                terminal=True,
            )
        raw_execution = passport.get("execution")
        raw_guard = passport.get("guard")
        execution = raw_execution if isinstance(raw_execution, dict) else {}
        guard = raw_guard if isinstance(raw_guard, dict) else {}
        if execution.get("route") == "l1":
            l1_assessment_valid = (
                passport.get("decision_path") == "safety_assessment"
                and execution.get("terminal") is False
                and execution.get("consume_message") is True
                and bool(str(execution.get("user_response") or "").strip())
                and guard.get("status") == "pass"
                and guard.get("can_call_l2") is False
            )
            if not l1_assessment_valid:
                session.route = "STOPPED"
                session.stopped_reason = "L1安全评估出口校验失败。"
                session.internal_flags.append("l1_assessment_exit_guard_rejected")
                return self._finish(session, reply=session.stopped_reason, terminal=True)
            if execution.get("resume_l2_next_turn") is True:
                resume_input = (
                    session.deferred_l2_expected_input
                    or session.expected_input
                )
                if resume_input == "auto":
                    resume_input = "free_text"
                session.expected_input = resume_input
                session.route = (
                    "L2_SCALE" if resume_input == "scale_answer" else "L2_INTERVIEW"
                )
                reply = str(execution["user_response"])
                if session.deferred_l2_reply:
                    reply = reply.rstrip() + "\n\n" + session.deferred_l2_reply
                session.deferred_l2_reply = ""
                session.deferred_l2_expected_input = None
            else:
                session.route = "L1_SAFETY_ASSESSMENT"
                reply = str(execution["user_response"])
            return self._finish(
                session,
                reply=reply,
                waiting_for=(
                    str(session.expected_input)
                    if execution.get("resume_l2_next_turn") is True
                    else str(execution.get("waiting_for") or "safety_answer")
                ),
            )
        if execution.get("route") == "safety":
            safety_exit_valid = (
                passport.get("decision_path") in {"clinical_crisis", "safety_pause"}
                and execution.get("terminal") is False
                and execution.get("consume_message") is True
                and bool(str(execution.get("user_response") or "").strip())
                and guard.get("status") == "pass"
                and guard.get("can_call_l2") is False
            )
            if not safety_exit_valid:
                session.route = "STOPPED"
                session.stopped_reason = "L1安全处置出口校验失败。"
                session.internal_flags.append("l1_safety_exit_guard_rejected")
                return self._finish(session, reply=session.stopped_reason, terminal=True)
            session.route = "L1_SAFETY_HOLD"
            return self._finish(
                session,
                reply=str(execution["user_response"]),
                waiting_for=str(execution.get("waiting_for") or "safety_handoff"),
            )
        if execution.get("terminal") is True:
            session.route = "STOPPED"
            session.stopped_reason = (
                execution.get("user_response") or "检测到需要优先处理的安全风险。"
            )
            return self._finish(
                session,
                reply=session.stopped_reason,
                terminal=True,
            )
        if execution.get("route") == "security":
            security_exit_valid = (
                passport.get("decision_path") == "security_reject"
                and execution.get("terminal") is False
                and bool(str(execution.get("user_response") or "").strip())
                and guard.get("status") == "pass"
                and guard.get("can_call_l2") is False
            )
            if not security_exit_valid:
                session.route = "STOPPED"
                session.stopped_reason = "L1系统安全出口校验失败，当前消息未进入后续流程。"
                session.internal_flags.append("l1_security_exit_guard_rejected")
                return self._finish(
                    session,
                    reply=session.stopped_reason,
                    terminal=True,
                )
            waiting_for = (
                session.expected_input
                if session.expected_input != "auto"
                else "free_text"
            )
            return self._finish(
                session,
                reply=execution["user_response"],
                waiting_for=waiting_for,
            )
        l1_exit_valid = (
            isinstance(raw_execution, dict)
            and isinstance(raw_guard, dict)
            and execution.get("route") == "l2"
            and execution.get("terminal") is False
            and guard.get("status") == "pass"
            and guard.get("can_call_l2") is True
        )
        if not l1_exit_valid:
            session.route = "STOPPED"
            session.stopped_reason = "L1安全出口校验失败，当前消息未进入后续流程。"
            session.internal_flags.append("l1_exit_guard_rejected")
            return self._finish(
                session,
                reply=session.stopped_reason,
                terminal=True,
            )

        # 待人工复核只暂停下游生成，不能绕过新消息的 L1 安全检查。
        if session.route == "MANUAL_REVIEW":
            return self._finish(
                session,
                reply="当前会话需要专业人员复核，自动流程已暂停。",
                waiting_for="manual_review",
            )

        if session.route == "AWAITING_PATIENT_CONFIRMATION":
            if session.l2_session_id is None:
                raise RuntimeError("患者确认阶段缺少 L2 会话")
            try:
                self.l2.update_safety_passport(
                    session.l2_session_id, session.l1_passport or {}
                )
            except L2EntranceGuardError:
                session.route = "STOPPED"
                session.stopped_reason = "L1到L2的安全接口校验失败，当前消息未进入L2。"
                return self._finish(
                    session, reply=session.stopped_reason, terminal=True
                )
            confirmation_result = self.dialogue.handle_message(
                session.l2_session_id,
                user_message,
                expected_input="profile_confirmation",
                message_id=str((session.l1_passport or {}).get("message_id") or ""),
            )
            decision = confirmation_result.confirmation_decision
            if confirmation_result.error_code in {"L2_MODEL_UNAVAILABLE", "L2_MODEL_OUTPUT_INVALID"}:
                return self._stop_l2_model_failure(session, confirmation_result.reply)
            if not confirmation_result.accepted or decision is None:
                return self._finish(
                    session,
                    reply=confirmation_result.reply,
                    waiting_for="patient_confirmation",
                    confirmation=session.patient_confirmation,
                )
            return self._apply_confirmation(session, decision, already_applied=True)

        if session.route == "L4":
            return self._handle_l4(session, user_message)

        return self._handle_l2(session, user_message)

    def confirm_patient_profile(
        self,
        session_id: str,
        decision: ConfirmationDecision,
    ) -> OrchestrationOutcome:
        session = self.store.get(session_id)
        if session.route != "AWAITING_PATIENT_CONFIRMATION":
            raise ValueError("当前会话不在患者确认阶段")
        return self._apply_confirmation(session, decision)

    def resume_session(self, session_id: str) -> OrchestrationOutcome:
        session = self.store.get(session_id)
        if session.route != "PAUSED_BY_PATIENT":
            raise ValueError("当前会话没有被患者暂停")
        if session.l2_session_id is None:
            raise RuntimeError("暂停的患者画像缺少 L2 会话")
        session.clinical_intake = self.l2.resume_profile_confirmation(
            session.l2_session_id
        )
        session.route = "AWAITING_PATIENT_CONFIRMATION"
        return self._finish(
            session,
            reply="流程已经恢复，请确认整理的信息是否准确。",
            waiting_for="patient_confirmation",
            confirmation=session.patient_confirmation,
        )

    def get_developer_profile_path(self, session_id: str) -> Path | None:
        """开发者专用入口；患者可见 outcome 永远不包含此路径。"""
        value = self.store.get(session_id).latest_developer_profile_path
        return Path(value) if value else None

    def get_session_snapshot(self, session_id: str) -> PipelineSession:
        """开发/测试专用只读快照。"""
        return self.store.snapshot(session_id)

    def _l1_assessment_context(self, session: PipelineSession) -> dict[str, Any]:
        """提供编排器生成的可信上下文，不把L2输出当成患者消息。"""
        context: dict[str, Any] = {
            "route": session.route,
            "expected_input": session.expected_input,
        }
        if session.l2_session_id is None:
            return context
        try:
            l2_state = self.l2.get_session_snapshot(session.l2_session_id)
            context["l2_stage"] = l2_state.current_stage
            pending = self.l2.scales.pending_item(l2_state)
        except (KeyError, ValueError):
            return context
        if pending is not None:
            context["pending_scale_item"] = pending.model_dump(mode="json")
        return context

    def _handle_l2(
        self,
        session: PipelineSession,
        user_message: str,
    ) -> OrchestrationOutcome:
        try:
            if session.l2_session_id is None:
                session.l2_session_id = self.l2.create_session(
                    safety_passport=session.l1_passport,
                    session_id=session.session_id,
                )
            else:
                self.l2.update_safety_passport(
                    session.l2_session_id,
                    session.l1_passport or {},
                )
        except L2EntranceGuardError:
            session.route = "STOPPED"
            session.stopped_reason = "L1到L2的安全接口校验失败，当前消息未进入L2。"
            session.internal_flags.append("l2_entrance_guard_rejected")
            return self._finish(
                session,
                reply=session.stopped_reason,
                terminal=True,
            )

        result = self.dialogue.handle_message(
            session.l2_session_id,
            user_message,
            expected_input=session.expected_input,
            message_id=str((session.l1_passport or {}).get("message_id") or ""),
        )
        if result.error_code in {"L2_MODEL_UNAVAILABLE", "L2_MODEL_OUTPUT_INVALID"}:
            return self._stop_l2_model_failure(session, result.reply)
        if not result.accepted:
            return self._finish(
                session,
                reply=result.reply,
                waiting_for=result.awaiting or session.expected_input,
            )
        if result.confirmation_decision == "pause":
            return self._apply_confirmation(session, "pause", already_applied=True)

        if (
            result.l2_turn is not None
            and result.l2_turn.stage_before == "engaging"
            and result.l2_turn.stage_after != "engaging"
            and (
                ((session.l1_passport or {}).get("structured_labels") or {})
                .get("l2_handoff", {})
                .get("baseline_asq_status")
                == "not_due"
            )
        ):
            asq_passport = start_baseline_asq(
                session_id=session.session_id,
                message_id=f"baseline-asq-{uuid.uuid4().hex}",
            )
            if asq_passport is not None:
                session.deferred_l2_reply = result.reply
                session.deferred_l2_expected_input = cast(
                    ExpectedDialogueInput,
                    result.awaiting or "free_text",
                )
                session.l1_passport = asq_passport
                asq_execution = asq_passport.get("execution") or {}
                asq_guard = asq_passport.get("guard") or {}
                if not (
                    asq_execution.get("route") == "l1"
                    and asq_execution.get("consume_message") is True
                    and asq_guard.get("status") == "pass"
                    and asq_guard.get("can_call_l2") is False
                ):
                    session.route = "STOPPED"
                    session.stopped_reason = "L1常规ASQ启动失败。"
                    return self._finish(session, reply=session.stopped_reason, terminal=True)
                session.route = "L1_SAFETY_ASSESSMENT"
                return self._finish(
                    session,
                    reply=str(asq_execution.get("user_response") or ""),
                    waiting_for="asq_answer",
                )

        intake = self.l2.build_clinical_intake(session.l2_session_id)
        session.clinical_intake = intake

        pending = result.l2_turn.pending_scale_item if result.l2_turn else None
        if pending is not None:
            session.route = "L2_SCALE"
            session.expected_input = "scale_answer"
            return self._finish(
                session,
                reply=result.reply,
                waiting_for=f"scale_answer:{pending.scale_name}:{pending.item_number}",
            )

        if intake.information_sufficiency != "sufficient":
            expected = cast(ExpectedDialogueInput, result.awaiting or "auto")
            session.expected_input = expected
            session.route = "L2_INTERVIEW"
            return self._finish(
                session,
                reply=result.reply,
                waiting_for=expected,
            )

        if intake.patient_confirmation != "confirmed":
            confirmation = build_patient_confirmation(intake)
            session.patient_confirmation = confirmation
            session.route = "AWAITING_PATIENT_CONFIRMATION"
            session.expected_input = "profile_confirmation"
            return self._finish(
                session,
                reply=result.reply,
                waiting_for="patient_confirmation",
                confirmation=confirmation,
            )

        return self._run_l3(session, intake)

    def _run_l3(
        self,
        session: PipelineSession,
        intake: ClinicalIntake,
    ) -> OrchestrationOutcome:
        session.route = "L3_PROCESSING"
        try:
            self.l2_to_l3.assert_valid(intake)
            profile = self.l2_to_l3.to_profile(intake)
        except ValueError as exc:
            session.route = "RETURN_TO_L2"
            session.expected_input = "free_text"
            session.internal_flags.append("l2_to_l3_guard_rejected")
            return self._finish(
                session,
                reply="患者画像还没有通过完整性检查，需要继续补充信息。",
                waiting_for="free_text",
            )

        passport = session.l1_passport or {}
        raw = self.l3_diagnose(profile, {
            "risk_level": passport.get("risk_level", 0),
            "sensitive_topics": passport.get("sensitive_topics", []),
            "flags": passport.get("flags", []),
            "l1_flags": passport.get("flags", []),
        })
        report = DiagnosticReport.model_validate(raw)
        session.l3_raw = raw
        session.diagnostic_report = report
        if session.l2_session_id is not None:
            self.l2.receive_l3_follow_up(session.l2_session_id, None, report.duration_evidence)

        if report.verifier_routing == "return_to_l2":
            session.l3_to_l2_loop_count += 1
            if session.l3_to_l2_loop_count > self.max_l3_to_l2_loops:
                session.route = "MANUAL_REVIEW"
                return self._finish(
                    session,
                    reply="多次补充后仍无法形成稳定判断，需要专业人员进一步复核。",
                    waiting_for="manual_review",
                )
            session.route = "RETURN_TO_L2"
            if session.l2_session_id is not None:
                self.l2.confirm_profile(session.l2_session_id, "revise")
            session.pending_l3_dimensions = [
                item.model_dump(mode="json") for item in report.missing_dimensions
            ]
            planned = report.information_collection_plan.get("items") or session.pending_l3_dimensions
            session.pending_l3_dimensions = list(planned)
            if session.l2_session_id is not None:
                self.l2.receive_l3_follow_up(session.l2_session_id, list(planned), report.duration_evidence)
            expected, question = self._l3_missing_requirement(
                session.pending_l3_dimensions
            )
            session.expected_input = expected
            return self._finish(
                session,
                reply=(
                    "为了避免作出不准确的判断，还需要补充一个方面。"
                    f"{question}"
                ),
                waiting_for=expected,
            )

        if report.verifier_routing == "manual_review":
            session.route = "MANUAL_REVIEW"
            return self._finish(
                session,
                reply="当前信息中存在需要进一步核对的地方，系统暂不进入下一阶段。",
                waiting_for="manual_review",
            )

        if report.verifier_routing == "reject":
            session.route = "STOPPED"
            session.stopped_reason = "当前证据不支持进入治疗层。"
            return self._finish(
                session,
                reply=session.stopped_reason,
                terminal=True,
            )

        try:
            L3ToL4Gate().assert_valid(report)
        except ValueError:
            session.route = "STOPPED"
            session.stopped_reason = "诊断输出没有通过进入治疗层前的完整性检查。"
            session.internal_flags.append("l3_to_l4_guard_rejected")
            return self._finish(
                session,
                reply=session.stopped_reason,
                terminal=True,
            )

        session.pending_l3_dimensions = []
        return self._start_l4(session, report)

    def _apply_confirmation(
        self,
        session: PipelineSession,
        decision: ConfirmationDecision,
        *,
        already_applied: bool = False,
    ) -> OrchestrationOutcome:
        if session.l2_session_id is None:
            raise RuntimeError("患者确认阶段缺少 L2 会话")
        if not already_applied:
            intake = self.l2.confirm_profile(session.l2_session_id, decision)
            session.clinical_intake = intake
        if decision == "pause":
            session.route = "PAUSED_BY_PATIENT"
            return self._finish(
                session,
                reply="好的，流程已经暂停，不会进入下一阶段。",
                waiting_for="resume",
            )
        if decision == "revise":
            session.route = "L2_INTERVIEW"
            session.expected_input = "free_text"
            return self._finish(
                session,
                reply="请告诉我哪些信息需要修改，以及更准确的情况是什么。",
                waiting_for="free_text",
            )

        intake = self.l2.build_clinical_intake(session.l2_session_id)
        session.clinical_intake = intake
        if intake.patient_confirmation != "confirmed":
            session.route = "L2_INTERVIEW"
            session.expected_input = "free_text"
            return self._finish(
                session,
                reply="画像仍有信息需要补充，暂时不会进入下一层。",
                waiting_for="free_text",
            )
        return self._run_l3(session, intake)

    def _start_l4(
        self,
        session: PipelineSession,
        report: DiagnosticReport,
    ) -> OrchestrationOutcome:
        passport = session.l1_passport or {}
        session.l4_session_id = self.l4.create_session(SessionRequest(
            session_id=session.session_id,
            report=report,
            safety=SafetyPassport(
                risk_level=passport.get("risk_level", 1),
                sensitive_topics=passport.get("sensitive_topics", []),
                flags=passport.get("flags", []),
            ),
        ))
        session.route = "L4"
        return self._finish(
            session,
            reply="谢谢你的确认。接下来我们会围绕你最希望改善的方面继续交流。",
            waiting_for="patient_message",
        )

    def _handle_l4(
        self,
        session: PipelineSession,
        user_message: str,
    ) -> OrchestrationOutcome:
        if session.l4_session_id is None:
            raise RuntimeError("L4 路由缺少 L4 会话")
        sd = self.l4.store.get(session.l4_session_id)
        passport = session.l1_passport or {}
        try:
            turn = self.instruction_planner.plan(
                session_id=session.l4_session_id,
                instructions=sd.instructions(), records=sd.treatment_records(),
                user_message=user_message, safety=passport,
            )
        except (TreatmentPlanningUnavailable, ValueError):
            session.internal_flags.append("l3_turn_instruction_failed")
            return self._finish(session, reply="治疗指令模型不可用或指令未通过校验，本轮未进入治疗生成，请稍后重试。", waiting_for="patient_message")
        sd.safety = SafetyPassport(
            risk_level=passport.get("risk_level", 1),
            sensitive_topics=passport.get("sensitive_topics", []),
            flags=passport.get("flags", []),
        )
        outcome = self.l4.handle_turn(TurnRequest(
            session_id=session.l4_session_id,
            user_message=user_message,
            turn=turn,
        ))
        if outcome.fallback_used:
            session.route = "MANUAL_REVIEW"
            session.internal_flags.append("l5_manual_review_required")
            # 仅保留在内部会话中，不将审核意见和治疗指令发给患者。
            session.pending_l5_review = outcome.model_dump(mode="json")
            return self._finish(
                session,
                reply="本轮回复多次未通过审核，自动治疗已暂停，需要专业人员复核后再继续。",
                waiting_for="manual_review",
            )
        return self._finish(
            session,
            reply=outcome.reply,
            waiting_for="patient_message",
        )

    @staticmethod
    def _l3_missing_requirement(
        missing: list[dict],
    ) -> tuple[ExpectedDialogueInput, str]:
        if not missing:
            return "free_text", "请再补充与当前困扰有关的具体表现。"
        item = missing[0]
        field = str(item.get("field") or "")
        allowed: set[str] = set(DIAGNOSTIC_EXPECTATIONS)
        expected: ExpectedDialogueInput = (
            field if field in allowed else "free_text"  # type: ignore[assignment]
        )
        question = str(item.get("suggested_question") or "请补充这方面的信息。")
        return expected, question

    def _stop_l2_model_failure(self, session: PipelineSession, reply: str) -> OrchestrationOutcome:
        session.route = "STOPPED"
        session.stopped_reason = reply
        session.internal_flags.append("l2_model_unavailable")
        return self._finish(session, reply=reply, terminal=True)

    def _finish(
        self,
        session: PipelineSession,
        *,
        reply: str,
        waiting_for: str | None = None,
        terminal: bool = False,
        confirmation: PatientConfirmation | None = None,
    ) -> OrchestrationOutcome:
        session.touch()
        try:
            session.latest_developer_profile_path = str(
                self.profile_renderer.render(session)
            )
        except Exception as exc:
            session.internal_flags.append(
                f"developer_profile_render_failed:{type(exc).__name__}"
            )
        return OrchestrationOutcome(
            session_id=session.session_id,
            route=session.route,
            reply=reply,
            waiting_for=waiting_for,
            terminal=terminal,
            confirmation=confirmation,
        )
