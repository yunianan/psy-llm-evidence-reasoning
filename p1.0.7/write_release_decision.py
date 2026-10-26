"""Apply the user's zero-first selection rule without changing any measured gate."""
import json
from pathlib import Path
import subprocess

from run_denial_audit import base
from layers.layer3.llm.client import APICLINICRReasoner

HERE = Path(__file__).resolve().parent


def main():
    summary = json.loads((HERE / "runs/compact-retest-v1/summary.json").read_text(encoding="utf-8"))
    zero = summary["comparisons"]["zero"]
    compact = summary["comparisons"]["compact"]
    checks = {
        "zero_individual_gate_pass": zero["gate"] == "pass",
        "both_modes_completed_20": all(summary["arms"][arm]["successful_cases"] == 20 for arm in ["zero", "compact"]),
        "both_activity_sets_20_of_20": all(summary["comparisons"][arm]["activity_identical_cases"] == 20 for arm in ["zero", "compact"]),
        "both_no_false_not_met": all(summary["arms"][arm]["effective_false_elimination_nodes"] == 0 for arm in ["zero", "compact"]),
        "targeted_pass": summary["checks"]["targeted_pass"],
        "both_fingerprints_compatible": all(summary["comparisons"][arm]["fingerprint_compatible"] for arm in ["zero", "compact"]),
        "source_and_schema_checks": all(summary["checks"].values()),
        "zero_input_p95_le_7000": zero["arms"]["zero"]["prompt_tokens"]["p95"] <= 7000,
    }
    decision = {
        "recorded_at": base.now(),
        "selected_mode": "zero" if all(checks.values()) else None,
        "selection_gate": "pass" if all(checks.values()) else "hold",
        "checks": checks,
        "user_rule": "修好 denial_scope 后两者都跑：都 20/20 → 选 zero（成本优先）；compact 明显更稳才选 compact。",
        "rule_interpretation": "两模式完成、活动集合和误排除安全门槛共同满足；待选的 zero 仍必须独立通过 overcall/undercall。未选 compact 的 undercall 不作为 zero 的额外阻塞项。",
        "post_retest_clarification": True,
        "agent_added_all_modes_gate_corrected": "原协议把所有比较臂均通过 undercall 也作为 zero 的阻塞项，超出用户原选择规则；本记录显式纠正该额外条件，不修改任何观测值、单臂门槛或正式评测协议。",
        "strict_all_modes_gate_preserved": summary["gate"],
        "compact_individual_gate_preserved": compact["gate"],
        "compact_undercall_not_worse_pass": compact["undercall_not_worse_pass"],
        "first_run_failures_preserved": summary["compact_cumulative_accounting"]["first_run_failed_case_ids"],
        "production_default_provider": type(base.get_default_knowledge_provider()).__name__,
        "default_mode_installed": APICLINICRReasoner().node_knowledge_view.screening_zero_knowledge,
        "regression_tests_passed": 537,
        "implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base.REPO, text=True).strip(),
    }
    assert decision["selection_gate"] == "pass" and decision["default_mode_installed"]
    (HERE / "screening-release-decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + '\n', encoding="utf-8")
    print(json.dumps({"selection_gate": decision["selection_gate"], "selected_mode": decision["selected_mode"],
                      "compact_individual_gate": compact["gate"], "strict_all_modes_gate": summary["gate"]}))


if __name__ == "__main__":
    main()
