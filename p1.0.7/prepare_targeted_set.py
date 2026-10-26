"""Predeclare 16 synthetic denial fixtures, independent of DEV/gold and outputs."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEXT = {
    "appearance": {
        "full": "I have never been worried about my appearance or any perceived defect in my body. I do not repeatedly check, groom, compare, seek reassurance, or avoid things because of my appearance. None of those appearance or bodily preoccupations applies to me now or in the past.",
        "partial": "I am not afraid of gaining weight and I am not concerned about getting fatter. We have not discussed any other aspect of my appearance, checking, grooming, or comparing.",
        "none": "I work at a library and take the bus to work. I would like to discuss my weekly schedule.",
        "unknown": "When asked whether I worry about my appearance or body or repeatedly focus on it, I honestly cannot tell. I am not sure how to answer that question.",
    },
    "mood": {
        "full": "I have never experienced elevated or unusually irritable mood, depression, feeling emotionally low, or loss of interest. I clearly deny all of those mood experiences both currently and in my past.",
        "partial": "I have never felt unusually high or euphoric. We have not discussed depression, loss of interest, or irritability.",
        "none": "I work at a library and take the bus to work. I would like to discuss my weekly schedule.",
        "unknown": "When asked about elevated mood, irritability, depression, or losing interest, I cannot tell whether I have experienced any of them. I am unsure.",
    },
    "anxiety": {
        "full": "I clearly deny anxiety, excessive worrying, fear, panic, and avoidance. None of these has ever occurred, either now or in the past. I do not avoid anything out of anxiety or fear.",
        "partial": "I am not afraid of spiders. We have not discussed other fears, worries, panic, or avoidance.",
        "none": "I work at a library and take the bus to work. I would like to discuss my weekly schedule.",
        "unknown": "When asked about anxiety, worry, fear, panic, or avoidance, I do not know whether those words describe my experience. I cannot give a clear answer.",
    },
    "sleep": {
        "full": "I have never had trouble falling asleep, staying asleep, or waking too early. I explicitly deny all three difficulties, currently and throughout my past.",
        "partial": "I have no difficulty falling asleep. We have not discussed staying asleep or waking too early.",
        "none": "I work at a library and take the bus to work. I would like to discuss my weekly schedule.",
        "unknown": "When asked about falling asleep, staying asleep, or waking too early, I cannot recall clearly and am not sure whether I have had these difficulties.",
    },
}


def fixtures():
    rows = []
    for entry, cases in TEXT.items():
        for scope, text in cases.items():
            case_id = f"synthetic-denial-{entry}-{scope}"
            ref = case_id + "-p001"
            rows.append({
                "case_id": case_id, "synthetic": True, "node_id": f"screen.{entry}",
                "expected_denial_scope": scope,
                "expected_status": "not_met" if scope == "full" else "pending",
                "profile": {"key_narratives": [{"value": text, "sources": [ref]}]},
                "reference_support": {"ref": ref, "quote": text},
            })
    return rows


if __name__ == "__main__":
    path = HERE / "targeted-denial-set.jsonl"
    assert not path.exists(), "Do not overwrite a preregistered fixture set"
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in fixtures()), encoding="utf-8")
    print("Saved 16 synthetic fixtures: 4 entrances x 4 denial scopes")
