import json
from common import HERE, cost
from tools.l3_engineering_runtime import reasoning_observation


def main():
    folder = HERE / "runs/thinking-probe-v1"
    raw = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    observations = []
    for trial in raw["trials"]:
        observations.append({"enabled": trial["enable_thinking"], "ok": trial["ok"],
                             "requests": [reasoning_observation(r) for r in trial["requests"]]})
    on, off = observations
    supported = on["ok"] and off["ok"] and any((r["effective_reasoning_tokens"] or 0) > 0 for r in on["requests"]) and all(r["effective_reasoning_tokens"] == 0 for r in off["requests"])
    result = {"thinking_control_observed": supported, "observations": observations,
              "note": "Original summary and raw null fields retained. Off-mode returns all completion as text; this is accounting-derived zero, not a fabricated API field.",
              "does_not_establish_clinical_equivalence": True}
    assert not (folder / "verification.json").exists()
    cost.write(folder / "verification.json", result)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
