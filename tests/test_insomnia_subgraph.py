import copy
import json
from pathlib import Path

import pytest

from shared.insomnia_subgraph import InsomniaExtensionSubgraph, NAMESPACE
from shared.knowledge_sources import (
    MENTALKG_ROOT, SUPPLEMENT_ROOT, InsomniaSupplementSource, MentalKGSource,
)


def resolve(ref):
    filename, pointer = ref.split("#", 1)
    if filename.startswith("MentalKG:"):
        path = MENTALKG_ROOT / filename.removeprefix("MentalKG:")
    else:
        path = Path(__file__).resolve().parents[1] / filename
    value = json.loads(path.read_text(encoding="utf-8"))
    for part in pointer.lstrip("/").split("/"):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def test_external_attachment_points_support_incoming_queries_without_reverse_rules():
    graph = InsomniaExtensionSubgraph()
    neighbors = graph.neighbors("Insomnia", relation="differential_diagnosis", direction="out")
    assert {node["id"] for node in neighbors} == {"D011", "D013"}
    assert all(node["external"] and node["namespace"] == "MentalKG" for node in neighbors)
    for label in ("D011", "D013"):
        incoming = graph.neighbors(label, relation="differential_diagnosis", direction="in")
        assert [node["id"] for node in incoming] == ["Insomnia"]
        assert graph.relations(label, relation="differential_diagnosis", direction="out") == []
    for edge in graph.relations("Insomnia", relation="differential_diagnosis"):
        assert edge["data"] == resolve(edge["source_ref"])
        assert set(edge["data"]["rules"]) == {"Insomnia", edge["target"]}


def test_symptom_group_preserves_full_condition_and_does_not_merge_sleep_symptoms():
    graph = InsomniaExtensionSubgraph()
    criteria = graph.neighbors("Insomnia", relation="has_diagnostic_criteria")[0]
    assert criteria["data"] == resolve(criteria["source_ref"])
    conditions = graph.neighbors(criteria["id"], relation="has_condition", direction="out")
    group = next(node for node in conditions if node["source_ref"].endswith("/insomnia_symptoms"))
    assert group["data"] == {"relation": "must_include", "min_count": 1,
                             "symptom_pool": ["S085", "S086", "S087"]}
    assert {node["id"] for node in graph.neighbors(group["id"], relation="uses_symptom")} == {
        "S085", "S086", "S087",
    }
    # The broad original S027 is not silently equated with the supplement symptoms.
    with pytest.raises(KeyError):
        graph.get_node("S027")


def test_all_graph_references_resolve_and_event_definition_is_source_scoped():
    graph = InsomniaExtensionSubgraph()
    exported = graph.to_dict()
    ids = {node["id"] for node in exported["nodes"]}
    assert len(ids) == len(exported["nodes"])
    for node in exported["nodes"]:
        assert node["data"] == resolve(node["source_ref"])
    for edge in exported["edges"]:
        assert edge["source"] in ids and edge["target"] in ids
        assert resolve(edge["source_ref"]) is not None
    event = graph.get_node(f"{NAMESPACE}:event:functional_impairment_required")
    base_events = json.loads((MENTALKG_ROOT / "event.json").read_text(encoding="utf-8"))
    assert event["data"] != base_events["functional_impairment_required"]
    assert event["namespace"] == NAMESPACE
    assert graph.relations(event["id"], relation="interpreted_by", direction="in")


def test_export_and_query_results_cannot_mutate_source_content():
    graph = InsomniaExtensionSubgraph()
    snapshot = graph.to_dict()
    changed = graph.get_node("S085")
    changed["data"]["name"] = "changed"
    changed_export = graph.to_dict()
    changed_export["edges"].clear()
    assert graph.to_dict() == snapshot


@pytest.mark.parametrize("fault,error", [
    ("unknown_disorder", "Unknown MentalKG attachment point"),
    ("symptom_collision", "must not overwrite"),
    ("unknown_symptom", "Unknown graph endpoint"),
    ("mismatched_rules", "must reference their graph endpoints"),
])
def test_invalid_extension_cannot_export_dangling_or_colliding_nodes(monkeypatch, fault, error):
    source = InsomniaSupplementSource()
    data = copy.deepcopy(source.get_candidate("Insomnia"))
    if fault == "unknown_disorder":
        data["differential"]["Insomnia"][0]["differential_diagnosis"] = "D999"
    elif fault == "symptom_collision":
        data["symptoms"]["S001"] = data["symptoms"]["S085"]
    elif fault == "mismatched_rules":
        data["differential"]["Insomnia"][0]["differential_diagnosis"] = "D012"
    else:
        data["criteria"]["required_criteria"]["insomnia_symptoms"]["symptom_pool"].append("S999")
    monkeypatch.setattr(source, "get_candidate", lambda node: data)
    with pytest.raises(ValueError, match=error):
        InsomniaExtensionSubgraph(mental_kg=MentalKGSource(), insomnia=source)


def test_saved_graph_matches_current_source_files():
    saved = json.loads((SUPPLEMENT_ROOT.parent / "extension_graph.json").read_text(encoding="utf-8"))
    assert saved == InsomniaExtensionSubgraph().to_dict()
