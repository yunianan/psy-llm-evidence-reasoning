"""Select model context without changing source data or decision-tree rules."""
from __future__ import annotations

from typing import Any

from shared.diagnostic_knowledge import DiagnosticKnowledgeProvider


NODE_KNOWLEDGE_VIEW_VERSION = "screening-zero-v1"
COMPACT_SCREEN_VIEW_VERSION = "screening-entry-compact-v1"
ANCHOR_VIEW_VERSION = "anchor-symptom-compact-v1"


class NodeKnowledgeView:
    """Build the selected screening context; later stages keep prior context.

    disorders describes the branches affected by a screen, not its knowledge
    requirements. Stage-specific views for later stages are a separate change.
    """

    def __init__(
        self, knowledge: DiagnosticKnowledgeProvider, *, screening_zero_knowledge: bool = False,
        screening_compact_view: bool = False,
        anchor_compact_view: bool = False,
    ) -> None:
        if screening_zero_knowledge and screening_compact_view:
            raise ValueError("Choose one screening context mode")
        self.knowledge = knowledge
        self.screening_zero_knowledge = screening_zero_knowledge
        self.screening_compact_view = screening_compact_view
        self.anchor_compact_view = anchor_compact_view
        self._symptom_catalog = None

    def sources_for_nodes(self, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        labels = list(dict.fromkeys(
            label
            for node in nodes
            if not (((self.screening_zero_knowledge or self.screening_compact_view) and node["stage"] == "screening")
                    or (self.anchor_compact_view and node["stage"] == "anchor"))
            for label in [*node["disorders"], *node.get("context_disorders", [])]
        ))
        sources = [self.knowledge.get_candidate(label) for label in labels]
        screens = [node for node in nodes if node["stage"] == "screening"]
        if self.screening_compact_view and screens:
            sources.append(self._compact_screen_view(screens))
        anchors = [node for node in nodes if node["stage"] == "anchor"]
        if self.anchor_compact_view and anchors:
            sources.append(self._compact_anchor_view(anchors))
        return sources

    def _compact_anchor_view(self, nodes: list[dict[str, Any]]) -> dict[str, Any]:
        from shared.disorder_catalog import disorder_metadata

        if self._symptom_catalog is None:
            self._symptom_catalog = self.knowledge.get_symptoms()
        pool = sorted({sid for node in nodes for sid in node.get("symptom_pool", [])})
        catalog = []
        for sid in pool:
            record = self._symptom_catalog[sid]
            # The source description is already a short definition. Keep its
            # qualifiers (notably the adult-restlessness note in S012).
            description = " ".join(record["description"].split())
            catalog.append({"id": sid, "name": record["name"], "description": description,
                            "source": "DSM5InsomniaSupplement" if sid in {"S085", "S086", "S087"} else "MentalKG"})
        labels = list(dict.fromkeys(label for node in nodes for label in node["disorders"]))
        return {"view": ANCHOR_VIEW_VERSION,
                "labels": [{key: disorder_metadata(label)[key] for key in ["disorder", "label", "name"]} for label in labels],
                "symptom_catalog_source": "provider.get_symptoms; original ID/name/description, qualifiers retained",
                "symptom_catalog": catalog}

    def _compact_screen_view(self, nodes: list[dict[str, Any]]) -> dict[str, Any]:
        from ..diagnosis.decision_tree import load_candidate_graph

        graph = load_candidate_graph()
        pool = sorted({sid for node in nodes for label in node["disorders"]
                       for anchor in graph["paths"][label]["anchors"]
                       for sid in graph["nodes"][anchor].get("symptom_pool", [])})
        if self._symptom_catalog is None:
            self._symptom_catalog = self.knowledge.get_symptoms()
        return {
            "view": COMPACT_SCREEN_VIEW_VERSION,
            "entry_rules": [
                "入口问题中的“或”表示任一线索即可支持 met，不要求同时满足全部方向，不要求达到对应疾病或完整发作标准。",
                "met 必须引用患者明确线索；不得因病因、持续时间或诊断归属未明将已有入口线索改成 pending。",
                "not_met 需要明确反驳入口所覆盖的线索；只否认其中一种可能、未提及其他方向时仍为 pending，不得扩大否认范围。",
                "睡眠入口询问入睡、维持或早醒困难；少睡而精力旺盛、单独的噩梦不自动等于这些困难。",
                "词表只解释入口用语，不要求命中计数，不把词表当成患者事实或诊断标准。",
            ],
            "symptom_catalog_source": "MentalKG + DSM5InsomniaSupplement; provider.get_symptoms; original ID/name only",
            "symptom_catalog": [[sid, self._symptom_catalog[sid]["name"]] for sid in pool],
        }
