from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

from .engine import DecisionTreeRunner, PAIRWISE_POLICY, RetryPolicy
from .models import load_bundle
from .pipeline import SevenStagePipeline
from .postprocess import Catalog, ReconciliationStore
from .entries import DEFAULT_ENTRY_PROFILE, EntryProfile

DEFAULT_TREE_BUNDLE = Path(__file__).resolve().parents[1] / "knowledge/paper_dtree/first2014_project24/refined_bundle.json"


def read_reference(path, case_id):
    if path is None:
        return None, None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("case_id") != case_id:
        raise ValueError("Author reference case_id does not match the query")
    return data["diagnoses"], data["source_ref"]


def stage_writer(output_dir):
    def write(stage):
        name = f"{stage['number']:02d}_{stage['name']}.json"
        (output_dir / name).write_text(json.dumps(stage, ensure_ascii=False, indent=2), encoding="utf-8")
    return write


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Paper Figure 1 seven-stage pipeline; Figure 2 is inside stage 2")
    sub = parser.add_subparsers(dest="command", required=True)
    catalog = sub.add_parser("source-index", help="Index the supplied 2014 PDF, without inventing edges")
    catalog.add_argument("--pdf", required=True)
    catalog.add_argument("--output", required=True)
    inspect = sub.add_parser("inspect", help="Validate an explicit executable tree bundle offline")
    inspect.add_argument("--bundle", default=str(DEFAULT_TREE_BUNDLE))
    entry_inspect = sub.add_parser("inspect-entries", help="Inspect the selected symptom-entry profile offline")
    entry_inspect.add_argument("--entry-profile", default=str(DEFAULT_ENTRY_PROFILE))
    run = sub.add_parser("run", help="Seven-stage Base/DT experiment, with explicit source data")
    run.add_argument("--bundle", default=str(DEFAULT_TREE_BUNDLE),
                     help="Default: project-24 local refined forest; use --method dt")
    run.add_argument("--case-file", required=True, help="UTF-8 raw vignette, without ground truth")
    run.add_argument("--case-id", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--method", choices=["base", "dt", "dt_raw"], required=True)
    run.add_argument("--pairwise-policy", choices=[PAIRWISE_POLICY], required=True,
                     help="Author aggregation is undisclosed; acknowledge explicit reconstruction")
    run.add_argument("--output-dir", required=True, help="A new directory; existing runs are never overwritten")
    run.add_argument("--max-attempts", type=int, default=None,
                     help="Optional engineering deviation; omitted = retry until valid, as in paper")
    run.add_argument("--catalog", required=True, help="Reviewed diagnostic vocabulary and Appendix C mappings")
    run.add_argument("--reconciliation-db", required=True, help="Reusable SQLite review store")
    run.add_argument("--reference-file", help="Separate author reference JSON; never sent to the model")
    run.add_argument("--entry-profile", default=str(DEFAULT_ENTRY_PROFILE),
                     help="Default: project-24 nine groups/24 entries routed to 19 family trees; 'bundle' uses tree screens")
    resume = sub.add_parser("resume", help="Resume stages 4-7 from saved raw results, without model calls")
    resume.add_argument("--input-result", required=True)
    resume.add_argument("--catalog", required=True)
    resume.add_argument("--reconciliation-db", required=True)
    resume.add_argument("--reference-file")
    resume.add_argument("--output-dir", required=True, help="New directory; preserve original artifacts")
    review = sub.add_parser("import-reviews", help="Import explicit human decisions, without inferring diagnoses")
    review.add_argument("--catalog", required=True)
    review.add_argument("--reconciliation-db", required=True)
    review.add_argument("--decisions-file", required=True)
    args = parser.parse_args(argv)
    if args.command == "source-index":
        from .source_index import build_source_index
        result = build_source_index(args.pdf)
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as output:
            json.dump(result, output, ensure_ascii=False, indent=2)
            output.write("\n")
        print(f"Indexed {len(result['trees'])} trees; catalog is not an executable bundle: {target}")
        return 0
    if args.command == "inspect-entries":
        profile = EntryProfile.load(args.entry_profile)
        print(json.dumps({"id": profile.id, "standard": profile.standard, "entry_count": len(profile.entries),
                          "groups": [{**group, "entry_count": sum(e.group_id == group["id"] for e in profile.entries)}
                                     for group in profile.groups]}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "import-reviews":
        catalog = Catalog.load(args.catalog)
        decisions = json.loads(Path(args.decisions_file).read_text(encoding="utf-8"))
        if not isinstance(decisions, list):
            parser.error("decisions-file must contain a JSON list of human review records")
        store = ReconciliationStore(args.reconciliation_db)
        try:
            for decision in decisions:
                store.add(catalog, decision)
        finally:
            store.close()
        print(f"Imported/reused {len(decisions)} reviewed strings")
        return 0
    if args.command == "resume":
        catalog = Catalog.load(args.catalog)
        saved = json.loads(Path(args.input_result).read_text(encoding="utf-8"))
        reference, source = read_reference(args.reference_file, saved["case_id"])
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "catalog_snapshot.json").write_text(Path(args.catalog).read_text(encoding="utf-8"), encoding="utf-8")
        (output_dir / "resume_source.json").write_text(json.dumps({"input_result": str(Path(args.input_result).resolve())}), encoding="utf-8")
        store = ReconciliationStore(args.reconciliation_db)
        try:
            result = SevenStagePipeline(catalog, store, record_stage=stage_writer(output_dir)).resume(
                saved, reference=reference, reference_source=source)
        finally:
            store.close()
        (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{result['status']}: {output_dir / 'result.json'}")
        return 0 if result["status"] == "completed" else 2
    bundle = load_bundle(args.bundle)
    if args.command == "inspect":
        print(json.dumps({"bundle_id": bundle.id, "standard": bundle.standard,
                          "stage": bundle.stage, "coverage": bundle.coverage,
                          "trees": len(bundle.trees), "nodes": sum(len(t.nodes) for t in bundle.trees)},
                         ensure_ascii=False, indent=2))
        return 0
    catalog = Catalog.load(args.catalog)
    if catalog.standard != bundle.standard:
        parser.error("Diagnostic catalog and tree bundle standards must match")
    entry_profile = None if args.entry_profile == "bundle" else EntryProfile.load(args.entry_profile)
    if entry_profile is not None and entry_profile.standard != bundle.standard:
        parser.error("Entry profile and tree bundle standards must match")
    retry = RetryPolicy(max_attempts=args.max_attempts)
    if (args.method == "dt" and bundle.stage != "refined") or (args.method == "dt_raw" and bundle.stage != "raw"):
        parser.error("Method and raw/refined bundle stage do not match")
    case = Path(args.case_file).read_text(encoding="utf-8")
    if not case.strip() or not args.case_id.strip() or not args.model.strip():
        parser.error("Nonempty case, case_id and model are required")
    reference, reference_source = read_reference(args.reference_file, args.case_id)
    SevenStagePipeline._check_reference(reference, reference_source)
    from .transport import OpenAICompatibleTransport
    transport = OpenAICompatibleTransport(
        api_key=os.environ.get("PAPER_DT_API_KEY", ""),
        base_url=os.environ.get("PAPER_DT_BASE_URL", "https://api.openai.com/v1"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    # Preserve the exact input/configuration beside the attempt journal.
    (output_dir / "case.txt").write_text(case, encoding="utf-8")
    (output_dir / "bundle_snapshot.json").write_text(
        Path(args.bundle).read_text(encoding="utf-8"), encoding="utf-8")
    (output_dir / "catalog_snapshot.json").write_text(
        Path(args.catalog).read_text(encoding="utf-8"), encoding="utf-8")
    if entry_profile is not None:
        (output_dir / "entry_profile_snapshot.json").write_text(Path(args.entry_profile).read_text(encoding="utf-8"), encoding="utf-8")
    (output_dir / "run_config.json").write_text(json.dumps({
        "case_id": args.case_id, "model": args.model, "method": args.method,
        "standard": bundle.standard, "bundle_id": bundle.id, "coverage": bundle.coverage,
        "catalog_id": catalog.id, "reconciliation_db": str(Path(args.reconciliation_db).resolve()),
        "entry_profile_id": entry_profile.id if entry_profile else "bundle_tree_screening",
        "pairwise_policy": args.pairwise_policy, "pairwise_case_wrapper": "prepend_full_case_v1",
        "retry": asdict(retry),
        "endpoint": os.environ.get("PAPER_DT_BASE_URL", "https://api.openai.com/v1"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "attempts.jsonl").open("x", encoding="utf-8") as journal:
        def record(event):
            journal.write(json.dumps(event, ensure_ascii=False) + "\n")
            journal.flush()
        runner = DecisionTreeRunner(bundle, transport, model=args.model,
                                    pairwise_policy=args.pairwise_policy, retry=retry, record=record,
                                    entry_profile=entry_profile)
        store = ReconciliationStore(args.reconciliation_db)
        try:
            pipeline = SevenStagePipeline(catalog, store, runner=runner, record_stage=stage_writer(output_dir))
            result = pipeline.run(case, case_id=args.case_id, method=args.method,
                                  reference=reference, reference_source=reference_source)
        finally:
            store.close()
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{result['status']}: {output_dir / 'result.json'}")
    return 0 if result["status"] == "completed" else 1 if result["status"] == "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
