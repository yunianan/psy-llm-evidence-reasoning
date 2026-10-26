"""Knowledge access boundary shared by candidate generation and verification."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Protocol

from .disorder_catalog import disorder_metadata
from .knowledge_sources import (
    InsomniaSupplementSource, MentalKGSource,
    get_insomnia_source, get_mental_kg_source,
)


class DiagnosticKnowledgeProvider(Protocol):
    """Return sourced knowledge, independently of its storage/retrieval backend.

    get_candidate must retain the complete criteria, relevant symptoms,
    differential rules, events, and source_ref for the requested project label.
    Patient facts and retrieved case examples are not diagnostic criteria.
    """

    def get_candidate(self, node: str) -> dict[str, Any]:
        ...

    def get_symptoms(self) -> dict[str, Any]:
        ...


class JsonDiagnosticKnowledgeProvider:
    """Explicit JSON adapter for offline tools and tests; never selected by default."""

    def __init__(
        self,
        mental_kg: MentalKGSource | None = None,
        insomnia: InsomniaSupplementSource | None = None,
    ) -> None:
        self.mental_kg = mental_kg if mental_kg is not None else get_mental_kg_source()
        self.insomnia = insomnia if insomnia is not None else get_insomnia_source()

    def get_candidate(self, node: str) -> dict[str, Any]:
        source = self.insomnia if node == "Insomnia" else self.mental_kg
        return {"metadata": disorder_metadata(node), **source.get_candidate(node)}

    def get_symptoms(self) -> dict[str, Any]:
        if self.mental_kg.symptoms.keys() & self.insomnia.symptoms.keys():
            raise ValueError("Supplement symptom IDs must not overwrite MentalKG")
        return {**self.mental_kg.symptoms, **self.insomnia.symptoms}


class DiagnosticKnowledgeNotConnected(RuntimeError):
    """The diagnostic engine has no explicitly connected knowledge backend."""


class UnconnectedDiagnosticKnowledgeProvider:
    """Keep the injection boundary without automatically connecting source data."""

    def get_candidate(self, node: str) -> dict[str, Any]:
        raise DiagnosticKnowledgeNotConnected(
            "Layer3 诊断知识尚未接入，等待 GraphRAG；不会自动读取 MentalKG 或失眠补充库。"
        )

    def get_symptoms(self) -> dict[str, Any]:
        raise DiagnosticKnowledgeNotConnected("Layer3 诊断知识尚未接入，等待 GraphRAG。")


@lru_cache
def get_default_knowledge_provider() -> UnconnectedDiagnosticKnowledgeProvider:
    return UnconnectedDiagnosticKnowledgeProvider()


@lru_cache
def _offline_json_provider() -> JsonDiagnosticKnowledgeProvider:
    return JsonDiagnosticKnowledgeProvider()


def candidate_knowledge(node: str) -> dict[str, Any]:
    """Explicit JSON lookup for offline tools, independent of the runtime default."""
    return _offline_json_provider().get_candidate(node)
