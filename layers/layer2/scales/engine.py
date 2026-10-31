from __future__ import annotations

from layers.layer2.evidence import EvidenceRegistry
from layers.layer2.scales.catalog import SCALE_CATALOG, ScaleItemDefinition
from layers.layer2.schemas import (
    L2SessionState,
    PendingScaleItem,
    ScaleName,
    ScaleResult,
    SemanticAnalysis,
)
from shared.scale_answer_parser import extract_explicit_scale_score


class ScaleAnswerError(ValueError):
    pass


class ScaleEngine:
    def suggest(
        self, state: L2SessionState, analysis: SemanticAnalysis
    ) -> ScaleName | None:
        if state.current_stage != "focusing" or not self._interview_ready(state):
            return None
        for scale in analysis.possible_scale_domains:
            current = state.scale_results.get(scale)
            if current and current.status != "not_started":
                continue
            if f"skip_scale:{scale}" in state.active_safety_controls:
                state.scale_results[scale] = ScaleResult(
                    scale_name=scale,
                    status="skipped",
                    skip_reason="L1安全控制禁止触发该量表",
                )
                continue
            state.scale_results[scale] = ScaleResult(
                scale_name=scale,
                status="pending_consent",
            )
            state.pending_scale_name = scale
            return scale
        return None

    @staticmethod
    def _interview_ready(state: L2SessionState) -> bool:
        """量表仅在明确重点并补充临床访谈信息后作为辅助工具。"""
        context = state.diagnostic_context
        if (
            not state.focus_topics
            or (context.symptom_duration_days is None and not context.duration_evidence)
            or context.functional_impairment is None
        ):
            return False
        evidence_by_id = {node.evidence_id: node for node in state.evidence_nodes}
        # 同一重点被反复提取时，以其首次出现为准；新增重点仍需后续访谈。
        focus_turn = max(
            (
                min(
                    (
                        evidence_by_id[source].turn_index
                        for source in fact.sources if source in evidence_by_id
                    ),
                    default=0,
                )
                for fact in state.focus_topics
            ),
            default=0,
        )
        if not focus_turn:
            return False
        return any(
            node.source_type == "patient_message"
            and node.turn_index > focus_turn
            and (
                node.target_path.startswith(("symptoms.", "functional_impairments."))
                or node.target_path in {
                    "diagnostic_context.symptom_duration_days",
                    "diagnostic_context.functional_impairment",
                }
            )
            for node in state.evidence_nodes
        )

    def apply_consent(
        self, state: L2SessionState, *, accepted: bool
    ) -> PendingScaleItem | None:
        scale = state.pending_scale_name
        if not scale:
            return None
        result = state.scale_results[scale]
        if result.status != "pending_consent":
            return None
        if not accepted:
            result.status = "declined"
            state.pending_scale_name = None
            return None
        result.status = "active"
        return self.pending_item(state)

    def record_answer(
        self,
        state: L2SessionState,
        *,
        message_id: str,
        patient_text: str,
        analysis: SemanticAnalysis,
    ) -> PendingScaleItem | None:
        scale = state.pending_scale_name
        if not scale:
            raise ScaleAnswerError("当前没有进行中的量表")
        result = state.scale_results[scale]
        if result.status != "active":
            raise ScaleAnswerError("量表尚未获得患者许可")
        definition = SCALE_CATALOG[scale]
        item_number = len(result.items) + 1
        if item_number > len(definition.items):
            raise ScaleAnswerError("量表已经完成")
        item = definition.items[item_number - 1]
        score = self._extract_score(patient_text, analysis, item)
        if not item.min_score <= score <= item.max_score:
            raise ScaleAnswerError(
                f"分值必须在 {item.min_score}-{item.max_score} 之间"
            )
        result.items[item_number] = score
        node = EvidenceRegistry.add_patient_evidence(
            state,
            message_id=message_id,
            quote=patient_text.strip(),
            target_path=f"scale_results.{scale}.items.{item_number}",
            source_type="scale_item",
        )
        result.evidence_ids.append(node.evidence_id)

        if len(result.items) == len(definition.items):
            result.status = "completed"
            result.score = definition.total_score(result.items)
            result.threshold = definition.threshold(result.items)
            state.pending_scale_name = None
            return None
        return self.pending_item(state)

    @staticmethod
    def pending_item(state: L2SessionState) -> PendingScaleItem | None:
        scale = state.pending_scale_name
        if not scale:
            return None
        result = state.scale_results[scale]
        if result.status != "active":
            return None
        definition = SCALE_CATALOG[scale]
        number = len(result.items) + 1
        if number > len(definition.items):
            return None
        item = definition.items[number - 1]
        return PendingScaleItem(
            scale_name=scale,
            item_number=number,
            prompt=item.prompt,
            min_score=item.min_score,
            max_score=item.max_score,
            anchors=item.anchors,
        )

    @staticmethod
    def _extract_score(
        text: str, analysis: SemanticAnalysis, item: ScaleItemDefinition
    ) -> int:
        explicit = extract_explicit_scale_score(
            text, min_score=item.min_score, max_score=item.max_score
        )
        if explicit is not None:
            return explicit

        for candidate in analysis.structured_fields:
            if candidate.field == "scale_answer_score":
                if candidate.evidence_quote not in text:
                    continue
                if isinstance(candidate.value, bool) or not isinstance(candidate.value, int):
                    continue
                return candidate.value
        raise ScaleAnswerError("无法确定量表分值，请使用题目给出的数字或选项回答")
