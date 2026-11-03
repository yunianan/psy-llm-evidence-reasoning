"""Independent symptom-entry profiles, loaded from exported data only."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .models import _keys, _strings, _text


DEFAULT_ENTRY_PROFILE = Path(__file__).resolve().parents[1] / "knowledge/paper_dtree/first2014_project24/entry_profile.json"


@dataclass(frozen=True)
class ScreeningEntry:
    id: str
    group_id: str
    disorder_id: str
    name: str
    label: str
    question: str
    source_refs: tuple[str, ...]
    tree_ids: tuple[str, ...]
    symptoms: tuple[dict, ...]


@dataclass(frozen=True)
class EntryProfile:
    id: str
    standard: str
    groups: tuple[dict, ...]
    entries: tuple[ScreeningEntry, ...]

    @classmethod
    def parse(cls, data):
        d = _keys(data, {"id", "standard", "review_status", "groups", "entries", "provenance", "notes"}, "entry profile")
        if d.get("standard") not in {"DSM-5", "DSM-5-TR"} or d.get("review_status") != "source_verified":
            raise ValueError("An explicitly versioned, source_verified entry profile is required")
        groups = []
        if not isinstance(d.get("groups"), list) or not d["groups"]:
            raise ValueError("Entry groups must be a nonempty list")
        for group in d["groups"]:
            g = _keys(group, {"id", "label"}, "entry group")
            groups.append({"id": _text(g.get("id"), "group id"), "label": _text(g.get("label"), "group label")})
        group_ids = {g["id"] for g in groups}
        if len(group_ids) != len(groups):
            raise ValueError("Duplicate entry group")
        entries = []
        if not isinstance(d.get("entries"), list) or not d["entries"]:
            raise ValueError("Entry profile must contain screening entries")
        for entry in d["entries"]:
            e = _keys(entry, {"id", "group_id", "disorder_id", "name", "label", "question", "source_refs", "tree_ids", "symptoms"}, "entry")
            for key in ("id", "group_id", "disorder_id", "name", "label", "question"):
                _text(e.get(key), f"entry.{key}")
            if e["group_id"] not in group_ids:
                raise ValueError("Unknown entry group")
            sources = _strings(e.get("source_refs"), "entry.source_refs")
            trees = _strings(e.get("tree_ids"), "entry.tree_ids")
            if not sources or not trees or not isinstance(e.get("symptoms"), list) or not e["symptoms"]:
                raise ValueError("Each entry needs source symptoms, references and target tree ids")
            symptoms = []
            for symptom in e["symptoms"]:
                s = _keys(symptom, {"id", "description", "source_ref"}, "entry symptom")
                symptoms.append({key: _text(s.get(key), f"symptom.{key}") for key in ("id", "description", "source_ref")})
            if len({s["id"] for s in symptoms}) != len(symptoms):
                raise ValueError("Duplicate symptom id within entry")
            entries.append(ScreeningEntry(e["id"], e["group_id"], e["disorder_id"], e["name"], e["label"],
                                          e["question"], sources, trees, tuple(symptoms)))
        if len({e.id for e in entries}) != len(entries) or len({e.disorder_id for e in entries}) != len(entries):
            raise ValueError("Entry ids and diagnostic label bindings must be unique")
        if {e.group_id for e in entries} != group_ids:
            raise ValueError("Entry profile contains an empty group")
        return cls(_text(d.get("id"), "entry profile id"), d["standard"], tuple(groups), tuple(entries))

    @classmethod
    def load(cls, path):
        return cls.parse(json.loads(Path(path).read_text(encoding="utf-8")))
