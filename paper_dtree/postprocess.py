"""Figure 1 stages 4-6, driven by reviewed names rather than guessed medical rules."""
from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
import sqlite3


SIMPLIFICATION_RULES = {
    1: "remove_specifiers_modifiers_codes_and_merge",
    2: "collapse_neurocognitive_disorders",
    3: "collapse_substance_specific_disorders",
    4: "collapse_breathing_related_sleep_disorders",
    5: "generalize_causative_medical_condition_by_category",
    6: "merge_other_specified_and_unspecified_by_category",
}
RECONCILIATION_RULES = {
    1: "retain_nonpsychiatric_as_non_dsm_false_positive",
    2: "map_to_z_code_condition",
    3: "normalize_minor_typography_or_extra_language",
    4: "generalize_excess_specification_or_specific_instance",
    5: "map_insufficiently_specific_output_to_other_unspecified",
    6: "map_prior_dsm_entity_to_current_entity",
    7: "retain_other_unmatched_output_as_non_dsm_false_positive",
}


def nonempty(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value


def diagnosis_list(value):
    if not isinstance(value, list):
        raise ValueError("Diagnoses must be a list")
    return [nonempty(name, "diagnosis") for name in value]


@dataclass(frozen=True)
class Catalog:
    id: str
    standard: str
    source_ref: str
    entries: dict[str, dict]
    simplifications: dict[int, dict[str, dict]]

    @classmethod
    def parse(cls, data):
        if not isinstance(data, dict) or data.get("review_status") != "source_verified":
            raise ValueError("A source_verified diagnostic catalog is required")
        if data.get("standard") not in {"DSM-5", "DSM-5-TR"}:
            raise ValueError("Catalog must explicitly select DSM-5 or DSM-5-TR")
        entries = {}
        for entry in data.get("diagnoses", []):
            name = nonempty(entry.get("name"), "catalog name")
            if name in entries:
                raise ValueError("Duplicate diagnostic catalog name")
            nonempty(entry.get("category"), "category")
            nonempty(entry.get("source_ref"), "diagnosis source")
            if entry.get("kind") not in {"dsm", "z_code"}:
                raise ValueError("Catalog entries must be DSM diagnoses or Z-code conditions")
            entries[name] = dict(entry)
        if not entries:
            raise ValueError("An explicit diagnostic vocabulary is required for WRatio")
        rules = {rule: {} for rule in SIMPLIFICATION_RULES}
        for row in data.get("simplifications", []):
            rule = row.get("rule")
            if type(rule) is not int or rule not in rules:
                raise ValueError("Simplification rule must be an Appendix C number, 1-6")
            for key in ("input", "output", "source_ref"):
                nonempty(row.get(key), f"simplification.{key}")
            if row["input"] in rules[rule]:
                raise ValueError("Duplicate simplification mapping for the same rule/string")
            rules[rule][row["input"]] = dict(row)
        return cls(nonempty(data.get("id"), "catalog id"), data["standard"],
                   nonempty(data.get("source_ref"), "catalog source"), entries, rules)

    @classmethod
    def load(cls, path):
        return cls.parse(json.loads(Path(path).read_text(encoding="utf-8")))

    def simplify(self, raw_names):
        rows = []
        current = [{"name": name, "raw_names": [name], "steps": []}
                   for name in diagnosis_list(raw_names)]
        for rule in SIMPLIFICATION_RULES:
            merged = {}
            for row in current:
                before = row["name"]
                mapping = self.simplifications[rule].get(before)
                after = mapping["output"] if mapping else before
                steps = list(row["steps"])
                if mapping:
                    steps.append({"rule": rule, "rule_name": SIMPLIFICATION_RULES[rule],
                                  "before": before, "after": after, "source_ref": mapping["source_ref"]})
                if after not in merged:
                    merged[after] = {"name": after, "raw_names": [], "steps": []}
                for name in row["raw_names"]:
                    if name not in merged[after]["raw_names"]:
                        merged[after]["raw_names"].append(name)
                for step in steps:
                    if step not in merged[after]["steps"]:
                        merged[after]["steps"].append(step)
            current = list(merged.values())
            rows.append({"rule": rule, "rule_name": SIMPLIFICATION_RULES[rule],
                         "names_after_rule": [item["name"] for item in current]})
        return {"items": current, "rule_trace": rows,
                "mapping_policy": "reviewed_exact_mappings_in_appendix_order",
                "unmapped_policy": "preserve_string_for_fuzzy_match_and_reconciliation"}

    def fuzzy_match(self, simplified):
        from rapidfuzz.fuzz import WRatio

        rows = []
        for item in simplified["items"]:
            scores = [(name, float(WRatio(item["name"], name, processor=None))) for name in self.entries]
            best_score = max(score for _, score in scores)
            best = [name for name, score in scores if score == best_score]
            over_threshold = [name for name, score in scores if score > 99]
            automatic = over_threshold[0] if len(over_threshold) == 1 else None
            rows.append({"input": item["name"], "raw_names": list(item["raw_names"]),
                         "best_matches": best, "best_score": best_score,
                         "matches_above_99": over_threshold, "automatic_match": automatic,
                         "status": "automatic_match" if automatic else "requires_reconciliation"})
        return {"items": rows, "algorithm": "rapidfuzz.fuzz.WRatio", "rapidfuzz_version": version("rapidfuzz"),
                "processor": None, "automatic_rule": "exactly_one_score_strictly_greater_than_99",
                "boundary_note": "score 99 requires review, following Appendix C >99"}


class ReconciliationStore:
    """Persistent reuse of reviewed strings, separated by standard and catalog version."""

    def __init__(self, path=":memory:"):
        self.connection = sqlite3.connect(str(path))
        self.connection.execute("""CREATE TABLE IF NOT EXISTS reconciliations (
            standard TEXT NOT NULL, catalog_id TEXT NOT NULL, input TEXT NOT NULL,
            decision TEXT NOT NULL, PRIMARY KEY (standard, catalog_id, input))""")
        self.connection.commit()

    def close(self):
        self.connection.close()

    def get(self, catalog, name):
        row = self.connection.execute(
            "SELECT decision FROM reconciliations WHERE standard=? AND catalog_id=? AND input=?",
            (catalog.standard, catalog.id, name)).fetchone()
        return json.loads(row[0]) if row else None

    def add(self, catalog, decision):
        d = dict(decision)
        for key in ("input", "target", "reviewer", "reviewed_at", "reason"):
            nonempty(d.get(key), f"review.{key}")
        rule = d.get("appendix_rule")
        if type(rule) is not int or rule not in RECONCILIATION_RULES:
            raise ValueError("Use English Appendix C reconciliation rule 1-7, not supplement reason codes")
        if rule in {1, 7}:
            if d["target"] != d["input"]:
                raise ValueError("Non-DSM output must be retained, not deleted or renamed to a DSM diagnosis")
            d.update(category="Non-DSM", kind="non_dsm")
        else:
            target = catalog.entries.get(d["target"])
            if target is None or (rule == 2 and target["kind"] != "z_code"):
                raise ValueError("Reconciliation target must be a catalog diagnosis (Z-code for rule 2)")
            d.update(category=target["category"], kind=target["kind"])
        d["appendix_rule_name"] = RECONCILIATION_RULES[rule]
        existing = self.get(catalog, d["input"])
        if existing is not None:
            if existing != d:
                raise ValueError("Conflicting reconciliation; version the catalog instead of overwriting review history")
            return existing
        with self.connection:
            self.connection.execute("INSERT INTO reconciliations VALUES (?,?,?,?)",
                                    (catalog.standard, catalog.id, d["input"], json.dumps(d, ensure_ascii=False)))
        return d


def reconcile(catalog, matches, store):
    accepted, pending = [], []
    for row in matches["items"]:
        decision = store.get(catalog, row["input"])
        if decision is not None:
            record = {"input": row["input"], "name": decision["target"], "category": decision["category"],
                      "kind": decision["kind"], "resolution": "reused_review", "review": decision}
        elif row["automatic_match"] is not None:
            target = catalog.entries[row["automatic_match"]]
            record = {"input": row["input"], "name": target["name"], "category": target["category"],
                      "kind": target["kind"], "resolution": "automatic_match", "review": None}
        else:
            pending.append({**row, "allowed_appendix_rules": RECONCILIATION_RULES,
                            "required_review_fields": ["input", "target", "appendix_rule", "reviewer", "reviewed_at", "reason"]})
            continue
        accepted.append(record)
    return {"status": "awaiting_reconciliation" if pending else "completed", "accepted": accepted,
            "pending": pending, "reference_scope": {"standard": catalog.standard, "catalog_id": catalog.id}}
