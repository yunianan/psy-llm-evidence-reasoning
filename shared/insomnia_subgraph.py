"""Standalone, sourced insomnia extension of MentalKG; no Layer3 integration.

The original disorder IDs are external attachment points. Conditions retain
their original grouping, counts and text; graph edges are not diagnostic rules.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from .knowledge_sources import (
    SUPPLEMENT_REF, InsomniaSupplementSource, MentalKGSource,
    get_insomnia_source, get_mental_kg_source,
)


NAMESPACE = "DSM5InsomniaSupplement"
Direction = Literal["out", "in", "both"]


class InsomniaExtensionSubgraph:
    def __init__(
        self,
        mental_kg: MentalKGSource | None = None,
        insomnia: InsomniaSupplementSource | None = None,
    ) -> None:
        base = mental_kg if mental_kg is not None else get_mental_kg_source()
        supplement = insomnia if insomnia is not None else get_insomnia_source()
        knowledge = supplement.get_candidate("Insomnia")
        if set(knowledge["symptoms"]) & set(base.symptoms):
            raise ValueError("Extension symptom IDs must not overwrite MentalKG")
        if "Insomnia" in base.disorders:
            raise ValueError("Extension disorder ID must not overwrite MentalKG")

        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: list[dict[str, Any]] = []
        self._out: dict[str, list[dict[str, Any]]] = {}
        self._in: dict[str, list[dict[str, Any]]] = {}

        self._add_node("Insomnia", "disorder", supplement.disorders["Insomnia"],
                       f"{SUPPLEMENT_REF}/disorder.json#/Insomnia")
        for sid, record in knowledge["symptoms"].items():
            self._add_node(sid, "symptom", record,
                           f"{SUPPLEMENT_REF}/symptom/10insomnia.json#/{sid}")
        for key, record in knowledge["events"].items():
            self._add_node(f"{NAMESPACE}:event:{key}", "event", record,
                           f"{SUPPLEMENT_REF}/event.json#/{key}")

        criteria_id = f"{NAMESPACE}:criteria:Insomnia"
        self._add_node(criteria_id, "diagnostic_criteria", knowledge["criteria"], knowledge["source_ref"])
        self._add_edge("Insomnia", criteria_id, "has_diagnostic_criteria", knowledge["source_ref"])
        for key, value in knowledge["criteria"]["required_criteria"].items():
            pointer = f"/required_criteria/{key}"
            condition_id = f"{NAMESPACE}:condition:Insomnia{pointer}"
            ref = knowledge["source_ref"] + pointer
            self._add_node(condition_id, "condition", value, ref)
            self._add_edge(criteria_id, condition_id, "has_condition", ref)
            if isinstance(value, dict):
                for sid in value.get("symptom_pool", []):
                    # Counts and group relations remain on the condition node.
                    self._add_edge(condition_id, sid, "uses_symptom", ref + "/symptom_pool")
            if isinstance(value, list):
                for index, item in enumerate(value):
                    item_id = f"{condition_id}/{index}"
                    item_ref = f"{ref}/{index}"
                    self._add_node(item_id, "condition", item, item_ref)
                    self._add_edge(condition_id, item_id, "has_condition", item_ref)
            if key in knowledge["events"]:
                event_id = f"{NAMESPACE}:event:{key}"
                self._add_edge(condition_id, event_id, "interpreted_by", ref)

        for index, record in enumerate(knowledge["differential"]["Insomnia"]):
            target = record["differential_diagnosis"]
            if target not in base.disorders:
                raise ValueError(f"Unknown MentalKG attachment point: {target}")
            if set(record["rules"]) != {"Insomnia", target}:
                raise ValueError("Differential rules must reference their graph endpoints")
            if target not in self._nodes:
                self._add_node(target, "disorder", base.disorders[target],
                               f"MentalKG:disorder.json#/{target}", external=True)
            self._add_edge("Insomnia", target, "differential_diagnosis",
                           f"{SUPPLEMENT_REF}/differential_diagnosis/Insomnia.json#/Insomnia/{index}",
                           data=record)

    def _add_node(
        self, node_id: str, kind: str, data: Any, source_ref: str, *, external: bool = False,
    ) -> None:
        if node_id in self._nodes:
            raise ValueError(f"Duplicate extension node: {node_id}")
        self._nodes[node_id] = {
            "id": node_id, "kind": kind,
            "namespace": "MentalKG" if external else NAMESPACE,
            "external": external, "source_ref": source_ref, "data": deepcopy(data),
        }

    def _add_edge(
        self, source: str, target: str, relation: str, source_ref: str, *, data: Any = None,
    ) -> None:
        if source not in self._nodes or target not in self._nodes:
            raise ValueError(f"Unknown graph endpoint: {source} -> {target}")
        edge = {"source": source, "target": target, "relation": relation,
                "namespace": NAMESPACE, "source_ref": source_ref}
        if data is not None:
            edge["data"] = deepcopy(data)
        self._edges.append(edge)
        self._out.setdefault(source, []).append(edge)
        self._in.setdefault(target, []).append(edge)

    def get_node(self, node_id: str) -> dict[str, Any]:
        return deepcopy(self._nodes[node_id])

    def relations(
        self, node_id: str, *, relation: str | None = None, direction: Direction = "both",
    ) -> list[dict[str, Any]]:
        if node_id not in self._nodes:
            raise KeyError(node_id)
        if direction not in {"out", "in", "both"}:
            raise ValueError("direction must be out, in or both")
        edges = []
        if direction in {"out", "both"}:
            edges.extend(self._out.get(node_id, []))
        if direction in {"in", "both"}:
            edges.extend(self._in.get(node_id, []))
        return deepcopy([edge for edge in edges if relation is None or edge["relation"] == relation])

    def neighbors(
        self, node_id: str, *, relation: str | None = None, direction: Direction = "both",
    ) -> list[dict[str, Any]]:
        edges = self.relations(node_id, relation=relation, direction=direction)
        ids = dict.fromkeys(edge["target"] if edge["source"] == node_id else edge["source"]
                            for edge in edges)
        return [self.get_node(adjacent) for adjacent in ids]

    def to_dict(self) -> dict[str, Any]:
        return deepcopy({
            "schema_version": 1,
            "graph_id": NAMESPACE,
            "extends": "MentalKG",
            "attachment_points": [node["id"] for node in self._nodes.values() if node["external"]],
            "nodes": list(self._nodes.values()),
            "edges": self._edges,
        })
