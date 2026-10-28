"""Package and verify the tested G1 engineering baseline, entirely offline.

This is a post-run engineering evidence snapshot, not a preregistered EVAL freeze.
No .env, API credentials, KB cases, or sealed EVAL data are read or copied.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import socket
import sys
from unittest.mock import patch
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True

WORKSPACE = Path(__file__).resolve().parents[2]
VERSION = "g1-engineering-freeze-v1"
RUN_REL = Path("phase1/p1.4-dev15-case2/runs/p1.4-dev15-case2-recovery-v1")
BASE_REL = Path("phase1/p1.4-dev15-case2")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def acceptance(root, engine, run, inputs):
    sys.path.insert(0, str(engine))
    from tools.phase1_research_entry import _validate_replay
    from tools.l3_observability_summary import summarize_run
    from layers.layer3.diagnosis.g1_cache import content_hash, knowledge_hash
    from layers.layer3.diagnosis.decision_tree import load_candidate_graph
    from layers.layer3.llm import client
    from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider, get_default_knowledge_provider

    identity = read_json(run / "manifest.A1.json")["identity"]
    code_checks = {name: content_hash((engine / name).read_text(encoding="utf-8")) == digest
                   for name, digest in identity["code_versions"].items()}
    assert code_checks and all(code_checks.values()), "Measured code signature differs"
    provider = JsonDiagnosticKnowledgeProvider()
    graph = load_candidate_graph()
    digest = knowledge_hash(provider)
    assert digest == identity["knowledge_hash"]
    assert content_hash((engine / "knowledge/dsm5_decision_tree/candidate_paths.json").read_text(encoding="utf-8")) == identity["tree_blob"]
    assert type(get_default_knowledge_provider()).__name__ == "UnconnectedDiagnosticKnowledgeProvider"
    original_freeze = read_json(root / "evidence/recovery-freeze.json")
    assert original_freeze["code_versions"] == identity["code_versions"]
    assert original_freeze["A1"]["knowledge_hash"] == digest
    assert original_freeze["policy"] == identity["policy"]
    joint = read_json(run / "joint-summary.json")
    assert joint["complete"] == {"A0": 15, "A1": 15} and joint["joint_complete"] == 15
    assert not joint["stop_reasons"] and joint["budget"]["reference_reserved_yuan"] == 0
    blocks = {"model_calls": 0, "network_calls": 0}

    def block_model(*args, **kwargs):
        blocks["model_calls"] += 1
        raise AssertionError("Frozen replay must not call models")

    def block_network(*args, **kwargs):
        blocks["network_calls"] += 1
        raise AssertionError("Frozen replay must not open the network")

    rows = read_rows(inputs)
    predictions = read_rows(run / "predictions/A1.jsonl")
    assert len(rows) == len(predictions) == 15
    assert [r["case_id"] for r in rows] == [r["case_id"] for r in predictions] == identity["case_ids"]
    assert identity["input_hash"] == content_hash(rows)
    assert sha256(inputs) == original_freeze["input_sha256"]
    by_id = {row["case_id"]: row for row in predictions}
    checks = []
    with ExitStack() as stack:
        stack.enter_context(patch.object(client, "ask_llm_stream", block_model))
        stack.enter_context(patch.object(socket.socket, "connect", block_network))
        stack.enter_context(patch.object(socket.socket, "connect_ex", block_network))
        for row in rows:
            report = _validate_replay(row, by_id[row["case_id"]], cache_dir=run / "g1-nodes")
            checks.append({"case_id": row["case_id"], "generation_matches": report == by_id[row["case_id"]]["report"]})
        observed = summarize_run(run)
    assert all(item["generation_matches"] for item in checks)
    assert observed == read_json(run / "observability/joint.summary.json"), "Strict ledger recomputation differs"
    assert observed["strict_acceptance"] is True
    caches = []
    for path in sorted((run / "g1-nodes").glob("*.json")):
        envelope = read_json(path)
        context = envelope["context"]
        assert context["knowledge_hash"] == digest and context["tree_hash"] == content_hash(graph)
        assert context["prompt_version"] == identity["prompt_version"]
        caches.append({"file": path.name, "case_id": context["case_id"], "format": envelope["format"],
                       "nodes": len(envelope["nodes"]), "nodes_hash": envelope["nodes_hash"],
                       "context_hash": content_hash(context), "sha256": sha256(path)})
    assert len(caches) == 15
    assert {c["case_id"] for c in caches} == set(identity["case_ids"])
    junit = []
    for name in ("transport-recovery-tests.xml", "source-and-cache-tests.xml"):
        suites = ET.parse(root / "evidence" / name).getroot()
        values = list(suites) if suites.tag == "testsuites" else [suites]
        counts = {key: sum(int(s.get(key, 0)) for s in values) for key in ("tests", "failures", "errors", "skipped")}
        assert counts["failures"] == counts["errors"] == counts["skipped"] == 0
        junit.append({"file": name, **counts})
    assert junit[0]["tests"] == 698 and junit[1]["tests"] == 79
    source_audit = read_json(root / "evidence/source-audit/coverage.json")
    assert source_audit["passed"] and not source_audit["unmapped_nodes"]
    # Bind the provenance audit to the same code, tree, KB and predictions.
    for relative, expected in source_audit["input_sha256"].items():
        relative = relative.replace("\\", "/")
        actual = run / "predictions/A1.jsonl" if relative.endswith("/predictions/A1.jsonl") else engine.parent / relative
        assert sha256(actual) == expected, f"Provenance audit input changed: {relative}"
    finished = [record for arm in ("A0", "A1") for record in read_rows(run / "runs" / arm / "requests.jsonl")
                if record["event"] == "attempt_finished"]
    models = sorted({r["response_model"] for r in finished if r.get("response_model")})
    fingerprints = sorted({r["system_fingerprint"] for r in finished if r.get("system_fingerprint")})
    return {"passed": True, "scope": "G1/L3 offline engineering acceptance on the recorded recovery chain",
            "code_signature_checks": code_checks, "code_files_bound_to_run": len(code_checks),
            "tree": {"version": graph["version"], "canonical_json_hash": content_hash(graph),
                     "manifest_json_string_hash": identity["tree_blob"],
                     "file_sha256": sha256(engine / "knowledge/dsm5_decision_tree/candidate_paths.json"),
                     "historical_git_blob": original_freeze["A1"]["tree_blob"]},
            "knowledge_hash": digest, "prompt_version": identity["prompt_version"], "policy": identity["policy"],
            "model": {"request_name": identity["requested_model"], "observed_response_names": models,
                      "system_fingerprints": fingerprints or None, "fingerprint_status": "exposed" if fingerprints else "provider_not_exposed",
                      "service_url": identity["service_url"], "source": "recorded HTTP responses, no new model probe",
                      "observed_dates": sorted({r.get("completed_at", r.get("started_at", ""))[:10] for r in finished})},
            "event_version": identity["event_version"], "envelope_version": identity["envelope_version"],
            "replay": {"cases": 15, "matched": len(checks), "checks": checks, "read_only": True, **blocks},
            "cache_inventory": caches, "strict_ledger_recomputed_equal": True, "junit": junit,
            "insomnia_source_coverage": source_audit["coverage"],
            "insomnia_online_assessment_instances": source_audit["real_dev15"]["assessment_instances"],
            "production_provider": type(get_default_knowledge_provider()).__name__,
            "research_freeze_complete": False, "eval_authorized": False, "real_eval_accessed": False}


def verify(bundle):
    manifest = read_json(bundle / "manifest.json")
    assert manifest["version"] == VERSION
    actual_files = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()}
    assert actual_files == set(manifest["files"]) | {"manifest.json"}, "Unexpected or missing package file"
    for relative, info in manifest["files"].items():
        path = bundle / relative
        assert sha256(path) == info["sha256"] and path.stat().st_size == info["bytes"], relative
    result = acceptance(bundle, bundle / "snapshot/L1-L5系统", bundle / "evidence/recovery-run",
                        bundle / "snapshot" / BASE_REL / "dev15.inputs.jsonl")
    assert result == read_json(bundle / "acceptance.json"), "Packaged acceptance differs from build acceptance"
    return {"version": "g1-freeze-verification-v1", "passed": True, "verified_files": len(manifest["files"]),
            "replay": result["replay"], "strict_ledger_recomputed_equal": True,
            "model_calls": 0, "network_calls": 0, "real_eval_accessed": False,
            "manifest_sha256": sha256(bundle / "manifest.json")}


def build(out):
    if out.exists():
        raise FileExistsError("Use a new freeze package directory")
    out.mkdir(parents=True)
    origins = {}

    def copy(source, destination):
        destination = out / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        origins[destination.relative_to(out).as_posix()] = str(source.relative_to(WORKSPACE)).replace("\\", "/")

    engine = WORKSPACE / "L1-L5系统"
    code = set()
    for folder in ("layers/layer3", "shared"):
        code.update((engine / folder).rglob("*.py"))
    code.add(engine / "layers/__init__.py")
    code.update(engine / name for name in read_json(WORKSPACE / RUN_REL / "manifest.A1.json")["identity"]["code_versions"])
    code.update(engine / name for name in ("requirements.txt", "tools/__init__.py", "tools/mentalbench_partition.py",
                                          "tests/__init__.py", "tests/tree_test_support.py"))
    for path in sorted(code):
        if path.is_file():
            copy(path, Path("snapshot") / path.relative_to(WORKSPACE))
    kb_roots = (WORKSPACE / "external/MentalBench/resources/knowledge_graph/EN",
                WORKSPACE / "external/MentalBench/resources/dsm5_supplement/EN",
                engine / "knowledge/dsm5_decision_tree")
    for folder in kb_roots:
        for path in sorted(folder.rglob("*.json")):
            copy(path, Path("snapshot") / path.relative_to(WORKSPACE))
    for name in ("audit_insomnia_sources.py", "build_g1_freeze.py"):
        path = WORKSPACE / "phase1/p1.5-closure" / name
        copy(path, Path("snapshot") / path.relative_to(WORKSPACE))
    copy(WORKSPACE / BASE_REL / "dev15.inputs.jsonl", Path("snapshot") / BASE_REL / "dev15.inputs.jsonl")
    run = WORKSPACE / RUN_REL
    run_files = [run / name for name in ("manifest.json", "manifest.A0.json", "manifest.A1.json", "joint-manifest.json",
                                         "joint-summary.json", "joint-state.json", "engineering-continuation.json")]
    for folder in ("g1-nodes", "predictions", "observability"):
        run_files += [p for p in (run / folder).rglob("*") if p.is_file()]
    for arm in ("A0", "A1"):
        run_files += [run / "runs" / arm / name for name in ("requests.jsonl", "events.jsonl")]
    for path in run_files:
        copy(path, Path("evidence/recovery-run") / path.relative_to(run))
    for name in ("recovery-freeze.json", "recovery-g1-replay.json", "recovery-registration.md", "transport-recovery-tests.xml"):
        copy(WORKSPACE / BASE_REL / name, Path("evidence") / name)
    copy(WORKSPACE / "phase1/p1.5-closure/source-and-cache-tests.xml", Path("evidence/source-and-cache-tests.xml"))
    for name in ("coverage.json", "coverage.md"):
        copy(WORKSPACE / "phase1/p1.5-closure/source-audit-v1" / name, Path("evidence/source-audit") / name)
    for label, folder in (("dev25-budget-stop", WORKSPACE / "phase1/p1.4-dev-dry-run/runs/p1.4-dev-dry-run-v1"),
                          ("dev15-transport-misclassification", WORKSPACE / BASE_REL / "runs/p1.4-dev15-case2-v1")):
        for name in ("joint-summary.json", "joint-manifest.json"):
            copy(folder / name, Path("evidence/negative-runs") / label / name)
    result = acceptance(out, out / "snapshot/L1-L5系统", out / "evidence/recovery-run",
                        out / "snapshot" / BASE_REL / "dev15.inputs.jsonl")
    dump(out / "acceptance.json", result)
    (out / "README.md").write_text(
        "# G1 工程冻结件 v1\n\n"
        "这是运行后整理的本地工程快照，绑定DEV-15修复链；不是正式研究/EVAL预注册freeze。\n\n"
        "包含代码、原始KG/失眠JSON、树、同名单输入、15份节点缓存、预测、事件/请求账本、严格汇总与JUnit。\n"
        "含患者画像与推理数据，仅留本地；不上传GitHub。无.env或API密钥文件，无真实EVAL输入/密封包。\n\n"
        "运行身份中11份代码签名逐项核对；其余代码快照为当前本地补充，不能倒称在线运行当时已完整签名。\n"
        "tree的Git blob、文件SHA256、JSON字符串hash、规范化对象hash算法不同，acceptance.json分别列示。\n\n"
        "验证命令（使用已有Python依赖）：\n\n"
        "```powershell\npython snapshot/phase1/p1.5-closure/build_g1_freeze.py --verify .\n```\n\n"
        "验证先检查每份文件的字节SHA256，再用包内代码、KG和节点缓存只读重建15份G1，重算严格账本；阻断网络和模型调用。\n"
        "cache key绑定画像/模型源/tree/knowledge/prompt/生成策略；缺失或不匹配即失败，不回退在线。\n\n"
        "适用范围仅当前模型源、已记录配置和病例；provider未暴露fingerprint，模型名不是不可变快照。\n"
        "完整研究未freeze、EVAL未授权；G2/C未接入，剪枝提案待确认；生产Unconnected/503不变。\n",
        encoding="utf-8")
    files = {p.relative_to(out).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size,
             "origin": origins.get(p.relative_to(out).as_posix(), "generated_by_closure_builder")}
             for p in sorted(out.rglob("*")) if p.is_file()}
    dependencies = {}
    for name in ("openai", "pydantic", "httpx", "httpx2", "pytest"):
        try:
            dependencies[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            dependencies[name] = None
    dump(out / "manifest.json", {"version": VERSION, "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
         "run_id": RUN_REL.name, "scope": "post-run G1/L3 engineering evidence freeze",
         "python": platform.python_version(), "dependencies_at_packaging": dependencies,
         "recorded_commit_is_not_full_worktree_snapshot": True,
         "signature_algorithms": {"files": "SHA256(raw file bytes)", "run_code_versions": "SHA256(canonical JSON encoding of UTF-8 source text)",
                                  "knowledge_and_cache": "SHA256(canonical JSON object encoding)", "historical_git_blob": "Git blob object ID"},
         "research_freeze_complete": False, "eval_authorized": False, "real_eval_accessed": False,
         "files": files})
    print(json.dumps({"built": str(out), "files": len(files), "code_files_bound_to_run": result["code_files_bound_to_run"],
                      "replay_matches": result["replay"]["matched"], "strict_ledger_equal": True, "model_calls": 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--out", type=Path)
    group.add_argument("--verify", type=Path)
    parser.add_argument("--verification-out", type=Path)
    args = parser.parse_args()
    if args.out:
        build(args.out.resolve())
    else:
        result = verify(args.verify.resolve())
        if args.verification_out:
            if args.verification_out.exists():
                raise FileExistsError("Do not overwrite a verification record")
            dump(args.verification_out, result)
        print(json.dumps({"passed": result["passed"], "verified_files": result["verified_files"],
                          "replay_matches": result["replay"]["matched"], "model_calls": result["model_calls"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
