import uuid

from pydantic import BaseModel, ConfigDict

from layers.layer4.config import Settings
from layers.layer4.generation.responder import Responder
from layers.layer4.generation.summary_builder import (
    build_evidence_refs,
    build_l5_context,
    build_summary,
)
from layers.layer4.llm.base import LLMBackend
from layers.layer4.llm import get_backend
from layers.layer4.review.mock_reviewer import MockReviewer
from layers.layer4.review.l5_adapter import L5Reviewer
from layers.layer4.review.orchestrator import ReviewOrchestrator
from layers.layer4.review.reviewer import Reviewer
from layers.layer4.schemas.instructions import (
    DiagnosticAnchor,
    SafetyPassport,
    Substage,
    TurnInstruction,
)
from layers.layer4.schemas.l3_report import DiagnosticReport
from layers.layer4.schemas.l5_context import L5Context
from layers.layer4.schemas.review import ReviewResult
from layers.layer4.schemas.summary import StructuredSummary
from layers.layer4.session.history import ConversationTurn, SessionData, SessionStore
from layers.layer4.validation import L3ToL4Gate
from shared.schemas.treatment_instructions import TreatmentInstructions
from shared.treatment_instruction_guard import validate_turn_instructions

THERAPY_KEYWORDS = {
    "EMDR": "EMDR", "CPT": "CPT", "IPSRT": "IPSRT", "ERP": "ERP",
    "CBT-I": "CBT-I", "ACT": "ACT", "MBCT": "MBCT", "IPT": "IPT",
    "SFBT": "SFBT", "CBT": "CBT",
}


class SessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str | None = None
    report: DiagnosticReport
    safety: SafetyPassport = SafetyPassport()


class TurnRequest(BaseModel):
    session_id: str
    user_message: str
    turn: TurnInstruction


class TurnOutcome(BaseModel):
    session_id: str
    reply: str
    l4_substage: Substage
    l5_context: L5Context
    summary: StructuredSummary
    review_history: list[ReviewResult]
    fallback_used: bool = False


class L4Service:
    """L4 治疗对话层门面：一轮完整处理（输入指令 → 回复 + L5Context + 摘要）。"""

    def __init__(
        self,
        config: Settings | None = None,
        backend: LLMBackend | None = None,
        reviewer: Reviewer | None = None,
    ) -> None:
        self.config = config or Settings()
        self.backend = backend or get_backend(self.config)
        if reviewer is not None:
            self.reviewer = reviewer
        elif self.config.review_mode == "l5_local":
            from layers.layer5.reviewer import review_response

            self.reviewer = L5Reviewer(review_response)
        elif self.config.review_mode == "mock":
            self.reviewer = MockReviewer()
        else:
            raise NotImplementedError(
                "review_mode='l5_endpoint' 尚未实现，请使用默认的 'l5_local'"
            )
        self.store = SessionStore()

    def create_session(self, req: SessionRequest) -> str:
        L3ToL4Gate().assert_valid(req.report)
        session_id = req.session_id or uuid.uuid4().hex
        bundle = TreatmentInstructions.model_validate(req.report.instructions)
        anchor = bundle.L2
        therapy_options = [t for t in [anchor.therapy_type] + list(anchor.plugins) if t]
        data = SessionData(
            session_id=session_id,
            role=bundle.L1,
            anchor=anchor,
            boundary=bundle.L4,
            safety=req.safety,
            therapy_options=therapy_options,
        )
        self.store.create(data)
        return session_id

    def handle_turn(self, req: TurnRequest, initial_fix: str | None = None,
                    persist: bool = True) -> TurnOutcome:
        sd = self.store.get(req.session_id)
        if sd.manual_review_required:
            raise ValueError("当前会话需要人工复核，自动生成和重写已暂停")
        records = sd.treatment_records()
        # 外部回环 A 重写同一轮，不允许借重写替换已经批准的治疗决策。
        if not persist and records and req.turn.turn_number == len(records):
            if req.turn != records[-1].instruction or req.user_message != records[-1].user_message:
                raise ValueError("重写必须沿用原轮次的患者消息和 L3 指令")
            records = records[:-1]
        instructions = sd.instructions().model_copy(update={"L3": req.turn}, deep=True)
        # 使用进入本轮前的快照；同轮外部重写需排除本轮已存的消息对。
        history = [item.model_copy(deep=True) for item in sd.history[:2 * len(records)]]
        validate_turn_instructions(
            instructions,
            session_id=req.session_id, user_message=req.user_message, records=records,
            safety=sd.safety.model_dump(mode="json"),
        )
        anchor = sd.anchor or DiagnosticAnchor()
        substage = sd.machine.resolve(req.turn.force_substage)

        responder = Responder(
            self.backend, sd.role, anchor, sd.boundary,
            temperature=self.config.llm_temperature,
        )
        orchestrator = ReviewOrchestrator(
            responder, self.reviewer,
            self.config.max_rewrite_attempts, self.config.fallback_reply,
        )

        def build_ctx(reply: str) -> L5Context:
            return build_l5_context(reply, substage, sd.safety, instructions,
                                    req.user_message, history)

        def validate(reply: str) -> list[str]:
            return self._validate(reply, req.turn, sd.therapy_options,
                                  sd.boundary.absolute_bans)

        outcome = orchestrator.run(
            req.turn, history, req.user_message,
            build_ctx, validate, initial_fix=initial_fix,
        )

        flags: list[str] = list(sd.machine.flags)
        if outcome.fallback_used:
            flags.extend(["loop_A_fallback", "manual_review_required"])
            sd.manual_review_required = True

        summary = build_summary(
            req.turn, substage, outcome.reply,
            build_evidence_refs(req.turn, history, sd.safety),
            flags=flags,
        )

        if persist:
            sd.history.append(ConversationTurn(role="user", content=req.user_message))
            sd.history.append(ConversationTurn(role="assistant", content=outcome.reply))
            sd.l3_history.append(req.turn)
            sd.fallback_history.append(outcome.fallback_used)
            if not outcome.fallback_used:
                sd.machine.advance(force=req.turn.force_substage)

        return TurnOutcome(
            session_id=req.session_id,
            reply=outcome.reply,
            l4_substage=substage,
            l5_context=build_ctx(outcome.reply),
            summary=summary,
            review_history=outcome.review_history,
            fallback_used=outcome.fallback_used,
        )

    @staticmethod
    def _validate(reply: str, turn: TurnInstruction,
                  therapy_options: list[str],
                  absolute_bans: list[str] | None = None) -> list[str]:
        """V4 本地自校验：禁止词 + 绝对禁令 + V4-3 疗法一致性（启发式）。"""
        violations: list[str] = []
        for word in turn.forbidden:
            if word and word in reply:
                violations.append(f"回复包含禁止词：{word}")
        for ban in absolute_bans or []:
            if not ban:
                continue
            target = ban
            for prefix in ("请勿", "禁止", "不要"):
                if target.startswith(prefix):
                    target = target[len(prefix):]
                    break
            if target and target in reply:
                violations.append(f"回复违反绝对禁令：{ban}")
        allowed = set(therapy_options)
        allowed.update(
            keyword
            for keyword in THERAPY_KEYWORDS
            if any(keyword in option for option in therapy_options)
        )
        for kw in THERAPY_KEYWORDS:
            if kw in reply and kw not in allowed and kw != "CBT":
                violations.append(f"回复提及未授权疗法：{kw}")
        return violations
