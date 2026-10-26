"""Preserve original counts, disclose reference errata, and export quote review."""
from collections import Counter
import json
from pathlib import Path

from audit_metrics import decision, error_counts

HERE = Path(__file__).resolve().parent


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main():
    references = rows(HERE / "reference.jsonl")
    original = {(row["case_id"], row["node_id"]): row for row in references}
    corrected = json.loads(json.dumps(original_values := list(original.values())))
    by_key = {(row["case_id"], row["node_id"]): row for row in corrected}
    inputs = rows(HERE / "inputs.jsonl")
    text_by_case = {row["case_id"]: {ref: item["value"] for item in row["profile"]["key_narratives"] for ref in item["sources"]} for row in inputs}
    errata = json.loads((HERE / "reference-errata.json").read_text(encoding="utf-8"))
    for change in errata["changes"]:
        row = by_key[(change["case_id"], change["node_id"])]
        assert row["reference_status"] == change["before"]
        for evidence in change["support"]:
            assert evidence["quote"] in text_by_case[change["case_id"]][evidence["ref"]]
        row.update(reference_status=change["after"], support=change["support"], erratum_reason=change["reason"])
    (HERE / "reference-v1.1-sensitivity.jsonl").write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in corrected), encoding="utf-8")

    primary_results = rows(HERE / "runs/zero-vs-full-v1/results.jsonl")
    compact_results = [row for row in rows(HERE / "runs/compact-vs-full-v1/results.jsonl") if row["arm"] == "compact"]
    results = primary_results + compact_results
    assert len(results) == 60 and all(row["ok"] for row in results)
    counts_by_arm = {}
    quote_reviews = []
    for arm in ["full", "zero", "compact"]:
        counts = Counter()
        for record in [row for row in results if row["arm"] == arm]:
            reference = [row for row in corrected if row["case_id"] == record["case_id"]]
            counts.update(error_counts(reference, record["judgments"]))
            for item in record["judgments"]:
                if item["status"] == "pending":
                    continue
                ref = by_key[(record["case_id"], item["node_id"])]
                cited = set(item["evidence_refs"])
                known_support = {evidence["ref"] for evidence in ref["support"]}
                review = ("supported_by_reference_quote" if item["status"] == ref["reference_status"] == "met" and cited & known_support else
                          "negation_scope_requires_review" if item["status"] == "not_met" else
                          "not_supported_under_reference_rubric" if ref["reference_status"] != "met" else
                          "alternative_citation_requires_review")
                quote_reviews.append({"case_id": record["case_id"], "arm": arm, "node_id": item["node_id"],
                                      "status": item["status"], "reference_status_v1_1": ref["reference_status"],
                                      "review": review, "review_method": "agent_reference_guided_post_review_not_clinical_gold",
                                      "rationale": item["rationale"], "reference_support": ref["support"],
                                      "cited_quotes": [{"ref": evidence, "quote": text_by_case[record["case_id"]][evidence],
                                                        "matched_reference_support": evidence in known_support} for evidence in item["evidence_refs"]],
                                      "additional_unmatched_citations": sorted(cited-known_support) if review == "supported_by_reference_quote" else []})
        counts_by_arm[arm] = {**dict(counts), "overcall_rate": counts["overcall"]/counts["nonmet_denominator"],
                              "undercall_rate": counts["undercall"]/counts["met_denominator"]}
    output = {"analysis_type": "post_run_reference_errata_sensitivity", "reference_original_preserved": True,
              "primary_reference_unchanged": True, "v1_counts": dict(Counter(row["reference_status"] for row in original_values)),
              "v1_1_counts": dict(Counter(row["reference_status"] for row in corrected)), "arms": counts_by_arm,
              "zero_gate": decision(counts_by_arm["full"], counts_by_arm["zero"], 19, 20),
              "compact_gate": decision(counts_by_arm["full"], counts_by_arm["compact"], 19, 20),
              "quote_review_counts": {arm: dict(Counter(row["review"] for row in quote_reviews if row["arm"] == arm)) for arm in counts_by_arm},
              "known_limitations": ["Agent audit references, no independent clinician gold", "No reference not_met cases", "Same DEV sample used for compact-view development", "Additional citations without reference support exported for review"]}
    (HERE / "sensitivity-summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n', encoding="utf-8")
    (HERE / "nonpending-quote-review.jsonl").write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in quote_reviews), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=True))


if __name__ == "__main__":
    main()
