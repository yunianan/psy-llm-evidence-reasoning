"""Independent JSON adapters for MentalKG and the DSM-5 insomnia supplement.

These adapters preserve source content and references. They have no dependency
on Layer3, display metadata, model clients, or a retrieval framework.
"""
from __future__ import annotations

from functools import cached_property, lru_cache
import json
import os
from pathlib import Path
from typing import Any


MENTALKG_ROOT = Path(os.getenv("MENTALKG_PATH") or (
    Path(__file__).resolve().parents[2] / "external/MentalBench/resources/knowledge_graph/EN"
))
SUPPLEMENT_ROOT = Path(__file__).resolve().parents[2] / "external/MentalBench/resources/dsm5_supplement/EN"
SUPPLEMENT_REF = "../external/MentalBench/resources/dsm5_supplement/EN"


def _read(root: Path, relative: str) -> dict[str, Any]:
    return json.loads((root / relative).read_text(encoding="utf-8-sig"))


class MentalKGSource:
    def __init__(self, root: Path = MENTALKG_ROOT) -> None:
        self.root = root

    @cached_property
    def disorders(self) -> dict[str, Any]:
        disorders = _read(self.root, "disorder.json")
        if set(disorders) != {f"D{i:03d}" for i in range(1, 24)}:
            raise ValueError("MentalKG must contain the original D001-D023 disorder mapping")
        return disorders

    @cached_property
    def symptoms(self) -> dict[str, Any]:
        return {
            sid: record
            for path in sorted((self.root / "symptom").glob("*.json"))
            for sid, record in _read(self.root, f"symptom/{path.name}").items()
        }

    @cached_property
    def criteria(self) -> dict[str, Any]:
        return _read(self.root, "diagnostic_criteria.json")

    @lru_cache
    def get_candidate(self, node: str) -> dict[str, Any]:
        if node not in self.disorders:
            raise ValueError(f"Unsupported diagnostic candidate: {node}")
        criteria = self.criteria[node]
        ids: set[str] = set()

        def collect(obj: Any) -> None:
            if isinstance(obj, dict):
                ids.update(obj.get("symptom_pool", []))
                for value in obj.values():
                    collect(value)
            elif isinstance(obj, list):
                for value in obj:
                    collect(value)

        collect(criteria)
        return {
            "source_ref": f"MentalKG:diagnostic_criteria.json#/{node}",
            "criteria": criteria,
            "symptoms": {sid: self.symptoms[sid] for sid in sorted(ids)},
            "differential": _read(self.root, f"differential_diagnosis/{node}.json"),
            "events": _read(self.root, "event.json"),
        }


class InsomniaSupplementSource:
    def __init__(self, root: Path = SUPPLEMENT_ROOT) -> None:
        self.root = root

    @cached_property
    def disorders(self) -> dict[str, Any]:
        return _read(self.root, "disorder.json")

    @cached_property
    def symptoms(self) -> dict[str, Any]:
        return _read(self.root, "symptom/10insomnia.json")

    @cached_property
    def criteria(self) -> dict[str, Any]:
        return _read(self.root, "diagnostic_criteria.json")

    @lru_cache
    def get_candidate(self, node: str) -> dict[str, Any]:
        if node != "Insomnia":
            raise ValueError(f"Unsupported diagnostic candidate: {node}")
        root_ref = f"{SUPPLEMENT_REF}/diagnostic_criteria.json#/Insomnia/required_criteria"
        return {
            "source_ref": f"{SUPPLEMENT_REF}/diagnostic_criteria.json#/Insomnia",
            "criteria": self.criteria[node],
            "symptoms": self.symptoms,
            "differential": _read(self.root, "differential_diagnosis/Insomnia.json"),
            "events": _read(self.root, "event.json"),
            "criterion_source_refs": {
                "A": root_ref,
                "A1": f"{SUPPLEMENT_REF}/symptom/10insomnia.json#/S085",
                "A2": f"{SUPPLEMENT_REF}/symptom/10insomnia.json#/S086",
                "A3": f"{SUPPLEMENT_REF}/symptom/10insomnia.json#/S087",
                "B": f"{root_ref}/additional_requirements/1",
                "C": f"{root_ref}/additional_requirements/2",
                "D": f"{root_ref}/min_duration",
                "E": f"{root_ref}/additional_requirements/3",
                "F": f"{root_ref}/additional_requirements/4",
                "G": f"{root_ref}/additional_requirements/5",
                "H": f"{root_ref}/additional_requirements/6",
            },
            "note": "DSM-5 项目补充，沿用 MentalKG 格式，非原图谱作者提供或审核；共存疾病不构成自动排除。",
        }


@lru_cache
def get_mental_kg_source() -> MentalKGSource:
    return MentalKGSource()


@lru_cache
def get_insomnia_source() -> InsomniaSupplementSource:
    return InsomniaSupplementSource()
