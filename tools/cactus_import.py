"""CACTUS (evaluation_chinese.json) -> Patient Agent case adapter.

Direct use of tools/db_import.py fails on CACTUS data ("missing record_id"):
db_import expects normalized records (record_id + symptoms) or transcripts
([{conversation: [{doctor, patient}]}]), while CACTUS evaluation_chinese.json
is [{idx, AI_client: {intake_form: <English free text>, attitude}, AI_counselor,
ground_truth_CBT}]. This script adapts CACTUS to normalized records
(450 rows = 150 clients x 3 attitudes, deduped to 150 cases), then reuses
db_import.build_scenario / import_one to generate cases. Symptom mapping is
sentence-level heuristic over intake sections 2/3/5 with negation filtering;
evidence is recorded in tools/cactus_import_details.json for human review.
Risk: full-text scan found no suicide/self-harm wording (0/150), so
risk_screen=none for all. Outcome: CACTUS has no outcome data, so
treatment_response=stable for all. Patient utterances remain db_import's
Chinese templates and need human rewriting, same as standard DB imports.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location("db_import", ROOT / "tools" / "db_import.py")
assert _SPEC and _SPEC.loader
db_import: Any = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(db_import)

from patient_agent.schemas import PatientScenario  # noqa: E402

LEXICON: dict[str, tuple[str, ...]] = {
    "depressed_mood": (
        "depress", "feeling down", "feel down", "low mood", "feeling low", "feel low",
        "unhappy", "gloomy", "tearful", "grief", "grieving", "lonely", "loneliness",
        "isolated", "isolation", "sadness", "feeling sad",
    ),
    "anhedonia": (
        "lose interest", "lost interest", "losing interest", "no longer enjoy",
        "no longer find", "enjoyment", "demotivat", "no motivation",
        "lack of motivation", "apathy", "apathetic", "lost my passion",
        "lost his passion", "lost her passion", "lose my passion",
        "unable to enjoy", "never be able to enjoy", "affecting my motivation",
        "losing my motivation", "lost my motivation", "my excitement",
    ),
    "insomnia": (
        "sleep", "insomnia", "wake up at night", "waking up", "fall asleep",
        "falling asleep", "trouble sleeping",
    ),
    "fatigue": (
        "fatigue", "tired", "exhaust", "drained", "lethargic", "no energy",
        "low energy", "feel drained",
    ),
    "worthlessness": (
        "worthless", "self-esteem", "self-worth", "self-doubt", "doubt my", "doubt myself",
        "not good enough", "inadequat", "inadequac", "inferior", "failure", "useless",
        "self-confidence", "my confidence", "confidence in pursuing", "ashamed", "shame",
        "self-critic", "self-blame", "self-conscious", "insecur", "incompeten",
        "incapable", "not capable", "rejection", "rejected", "perfectionis",
        "blame myself", "feel guilty", "guilt", "nothing special", "insignifican",
        "not significant", "discounting",
    ),
    "concentration_difficulty": (
        "concentrat", "distracted", "forgetful", "procrastinat",
        "difficulty focusing", "hard to focus", "trouble focusing",
        "attention span", "memory problem", "mind keeps wandering",
    ),
    "anxiety": (
        "anxiet", "anxious", "worry", "worried", "worrying", "nervous", "fear",
        "afraid", "panic", "dread", "overwhelm", "tense", "tension", "uneasy",
        "restless", "overthinking", "apprehens", "perfectionis",
    ),
    "irritability": (
        "irritab", "angry", "anger", "snap at", "short temper", "impatient",
        "easily annoyed", "losing my temper", "lose my temper",
    ),
    "trauma_related": (
        "trauma", "abuse", "abusive", "nightmare", "flashback", "ptsd",
    ),
    "obsession_compulsion": (
        "obsess", "compulsi", "intrusive thought", "ritual",
        "repeatedly check", "checking over and over",
    ),
    "elevated_mood": ("euphor", "manic", "mania"),
}

_NEGATION_EXCEPTIONS = (
    "no longer enjoy", "no longer find", "not good enough", "not capable",
    "not significant", "never be able to enjoy",
)
_NEGATION_CUES = (
    " not ", "n't", "never experienced", "never had", "have not", "has not",
    "do not", "does not", "did not", "i am not", "no significant",
    "not experienced", "not received", "not been impacted", "not impacted",
)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


def _is_denied(sentence: str) -> bool:
    low = " " + sentence.lower() + " "
    if any(p in low for p in _NEGATION_EXCEPTIONS):
        return False
    return any(cue in low for cue in _NEGATION_CUES)


def parse_intake(form: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": "", "age": "", "gender": "", "occupation": "", "education": "",
        "marital": "", "family": "", "presenting": "", "reason": "", "past": "",
        "functioning": "", "support": "",
    }
    m = re.search(
        r"Name:\s*\n(?P<name>.*?)\nAge:\s*\n(?P<age>.*?)\nGender:\s*\n(?P<gender>.*?)\n"
        r"Occupation:\s*(?P<occ>.*?)\nEducation:\s*(?P<edu>.*?)\n"
        r"Marital Status:\s*(?P<mar>.*?)\nFamily Details:\s*(?P<fam>.*?)\n",
        form, re.S,
    )
    if m:
        out.update({k: (v or "").strip() for k, v in m.groupdict().items()})
        out["occupation"] = out.pop("occ")
        out["education"] = out.pop("edu")
        out["marital"] = out.pop("mar")
        out["family"] = out.pop("fam")
    else:
        nm = re.search(r"Name:\s*\n?([^\n]+)", form)
        if nm:
            out["name"] = nm.group(1).strip()

    def section(start: str, ends: tuple[str, ...]) -> str:
        if start not in form:
            return ""
        rest = form.split(start, 1)[1]
        for end in ends:
            if end in rest:
                rest = rest.split(end, 1)[0]
        return rest.strip()

    out["presenting"] = section("2. Presenting Problem", ("3. Reason",))
    out["reason"] = section("3. Reason for Seeking Counseling", ("4. Past History",))
    out["past"] = section("4. Past History", ("5.",))
    out["functioning"] = section("5.", ("6. Social Support",))
    out["support"] = section("6. Social Support System", ())
    if not out["support"]:
        out["support"] = section("6. Social Support", ())
    return out


def extract_onset(presenting: str) -> str:
    low = presenting.lower()
    m = re.search(r"(\d+)\s*(week|month|year)s?\s+ago", low)
    if m:
        unit = {"week": "周", "month": "个月", "year": "年"}[m.group(2)]
        return f"{m.group(1)}{unit}"
    for pat, label in (
        ("few months", "几个月"), ("past month", "1个月"), ("a month", "1个月"),
        ("few weeks", "几周"), ("past year", "1年"), ("several months", "几个月"),
        ("several years", "几年"), ("few years", "几年"),
        ("since i was a teenager", "很久"), ("since childhood", "很久"),
        ("teenage years", "很久"), ("recently", "最近"), ("months", "几个月"),
        ("years", "几年"), ("weeks", "几周"),
    ):
        if pat in low:
            return label
    return "一段时间"


def extract_symptoms(parsed: dict[str, Any]) -> tuple[dict[str, int], dict[str, list[str]], bool]:
    weighted: dict[str, int] = {}
    evidence: dict[str, list[str]] = {}
    sources = ((parsed["presenting"], 2), (parsed["reason"], 1), (parsed["functioning"], 1))
    for text, weight in sources:
        for sentence in _sentences(text):
            # clause-level denial: a "not" in one clause must not erase a
            # symptom reported in another clause of the same sentence
            # (e.g. "...for not liking vegetables" vs "I often feel guilty...")
            clauses = [c.strip() for c in re.split(r"[,;]|\bbut\b", sentence) if c.strip()]
            for clause in clauses or [sentence]:
                if _is_denied(clause):
                    continue
                low = clause.lower()
                for code, keywords in LEXICON.items():
                    if any(kw in low for kw in keywords):
                        weighted[code] = weighted.get(code, 0) + weight
                        evidence.setdefault(code, []).append(sentence[:120])
    symptoms = {code: (3 if w >= 6 else 2 if w >= 3 else 1) for code, w in weighted.items()}
    fallback = False
    if not symptoms:
        symptoms = {"worthlessness": 1}
        fallback = True
    return symptoms, evidence, fallback


def build_records(data: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for rec in data:
        form = rec["AI_client"]["intake_form"]
        if form not in groups:
            groups[form] = []
            order.append(form)
        groups[form].append(rec)
    records: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for i, form in enumerate(order, start=1):
        group = sorted(groups[form], key=lambda r: r["idx"])
        rep = group[0]
        parsed = parse_intake(form)
        symptoms, evidence, fallback = extract_symptoms(parsed)
        onset = extract_onset(parsed["presenting"])
        record_id = f"CACTUS_{i:03d}"
        ground_truth = list(rep.get("ground_truth_CBT") or [])
        idxs = [r["idx"] for r in group]
        attitudes = sorted({r["AI_client"].get("attitude", "") for r in group})
        chinese_opening = ""
        for r in group:
            init = (r.get("AI_counselor") or {}).get("CBT", {}).get("dialogue_history_init") or ""
            if init:
                chinese_opening = init
                break
        top_code = max(symptoms, key=lambda c: symptoms[c])
        chief = db_import.CATALOG[top_code]["utterance"].format(onset=onset)
        occupation = parsed["occupation"]
        if occupation.lower() in ("", "not specified", "undisclosed", "-"):
            occupation = "未知"
        family_bits = [b for b in (parsed["marital"], parsed["family"]) if b and b != "-"]
        presenting_first = " ".join(_sentences(parsed["presenting"])[:2])
        reason_first = " ".join(_sentences(parsed["reason"])[:1])
        records.append({
            "record_id": record_id,
            "scenario_id": record_id,
            "version": "1.0.0",
            "title": f"CACTUS导入：{parsed['name'] or record_id}",
            "age_range": f"{parsed['age']}岁" if parsed["age"].isdigit() else "未知",
            "occupation": occupation,
            "family_relationship": "；".join(family_bits) if family_bits else "未知",
            "chief_complaint": chief,
            "onset": onset,
            "risk_screen": "none",
            "symptoms": symptoms,
            "denied_conditions": ["manic_history", "psychotic_features"],
            "treatment_response": "stable",
            "expected_diagnosis_codes": [],
            "must_not_diagnose": ["双相障碍", "精神分裂症"],
            "major_events": [
                f"来源 CACTUS evaluation_chinese.json idx {idxs}（同一来访者的 {', '.join(attitudes)} 三种态度变体，去重为 1 例）",
                f"CBT ground truth: {', '.join(ground_truth) if ground_truth else '无'}",
                f"Presenting problem (原文): {presenting_first[:300]}",
                f"Reason for counseling (原文): {reason_first[:200]}",
                f"CACTUS 中文版开场 (原文): {chinese_opening[:200]}",
            ],
        })
        details.append({
            "record_id": record_id, "name": parsed["name"], "age": parsed["age"],
            "idxs": idxs, "ground_truth_CBT": ground_truth, "symptoms": symptoms,
            "evidence": evidence, "fallback": fallback, "onset": onset,
        })
    return records, details


def enrich(out_root: Path, details: list[dict[str, Any]]) -> None:
    for d in details:
        path = out_root / d["record_id"] / "1.0.0.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        extra_tags = ["cactus", "needs_review"]
        for t in d["ground_truth_CBT"]:
            extra_tags.append("cbt_" + re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_"))
        payload["tags"] = sorted(set(payload.get("tags", []) + extra_tags))
        desc = (
            f"由 tools/cactus_import.py 从 CACTUS 数据集（evaluation_chinese.json "
            f"idx {d['idxs']}，来访者 {d['name'] or '未具名'}）适配后经 tools/db_import.py 生成；"
            f"症状由 intake 英文主诉启发式映射（{', '.join(d['symptoms'])}），"
            "话术为模板句、期望路径为推导值，均需人工复核。"
        )
        if d["fallback"]:
            desc += " [注意: 症状无命中, 已兜底为 worthlessness=1, 必须人工改判]"
        payload["description"] = desc
        payload["psychology"]["cognitive_patterns"] = [f"CBT目标技术：{t}" for t in d["ground_truth_CBT"]]
        scenario = PatientScenario.model_validate(payload)
        path.write_text(
            json.dumps(scenario.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        # keep index.json in sync with the enriched case file
        index_path = out_root / "index.json"
        if index_path.is_file():
            index_payload = json.loads(index_path.read_text(encoding="utf-8"))
            for entry in index_payload.get("scenarios", []):
                if entry.get("scenario_id") == d["record_id"]:
                    entry["tags"] = payload["tags"]
                    entry["description"] = desc
            index_path.write_text(
                json.dumps(index_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cactus_import", description="CACTUS -> Patient Agent 病例适配")
    parser.add_argument("--source", default="evaluation_chinese.json")
    parser.add_argument("--out", default="imported_scenarios")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="overwrite existing case files")
    parser.add_argument("--records-out", default="tools/cactus_normalized_records.json")
    parser.add_argument("--report-out", default="tools/cactus_import_details.json")
    args = parser.parse_args(argv)
    data = json.loads((ROOT / args.source).read_text(encoding="utf-8"))
    records, details = build_records(data)
    if args.limit:
        records, details = records[: args.limit], details[: args.limit]
    (ROOT / args.records_out).write_text(
        json.dumps({"dataset": "CACTUS", "dataset_version": "evaluation_chinese-450", "records": records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (ROOT / args.report_out).write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    out_root = ROOT / args.out
    results = []
    for record in records:
        results.append(db_import.import_one(
            record, version=None, dataset_version="evaluation_chinese-450",
            out_root=out_root, force=args.force, verify=True, baseline=False, dry_run=False,
        ))
    enrich(out_root, details)
    passed = sum(1 for r in results if r.get("passed"))
    print(f"CACTUS import done: {len(results)} cases, verify passed {passed}/{len(results)}")
    for r in results:
        if not r.get("passed"):
            print(f"  FAILED {r['scenario_id']}: {r.get('failures')}")
    fallback = [d["record_id"] for d in details if d["fallback"]]
    if fallback:
        print(f"  fallback (must recheck manually): {fallback}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
