from typing import Any

from layers.layer4.schemas.l3_report import DiagnosticReport
from shared.report_integrity import (
    ICD10_OUTPUT_POLICY, REQUIRED_L3_CHECKS, l3_report_fingerprint,
)


class L3ToL4ValidationError(ValueError):
    def __init__(self, violations: list[dict[str, str]]):
        self.violations = violations
        message = "；".join(
            f"{item['rule_id']}: {item['message']}" for item in violations
        )
        super().__init__(f"L3报告不允许进入L4：{message}")


class L3ToL4Gate:
    """L4 只验证 L3 凭证和报告完整性，不重复执行 L3 业务规则。"""

    def validate(self, report: DiagnosticReport) -> dict[str, Any]:
        violations: list[dict[str, str]] = []
        if report.verifier_routing != "pass":
            self._add(
                violations, "V3-ROUTING", f"verifier_routing={report.verifier_routing}不允许进入L4"
            )
        if report.diagnosis.primary is None:
            self._add(violations, "V3-PRIMARY-REQUIRED", "缺少主诊断")

        if report.icd10_review is None or report.verifier_decision != "clinician_confirmed":
            self._add(violations, "V3-ICD10-REVIEW-REQUIRED", "缺少 ICD-10 临床复核确认")

        receipt = report.l3_validation
        if receipt is None:
            self._add(violations, "V3-VALIDATION-RECEIPT", "缺少L3输出校验凭证")
            return self._result(violations)
        if not receipt.passed or receipt.route != "allow_l4":
            self._add(violations, "V3-VALIDATION-FAILED", "L3输出校验未通过")

        if receipt.policy_version != ICD10_OUTPUT_POLICY:
            self._add(violations, "V3-ICD10-VALIDATION-POLICY", "ICD-10 报告须使用对应放行规则")

        required = REQUIRED_L3_CHECKS
        check_status = {item.rule_id: item.status for item in receipt.checks}
        missing_checks = sorted(required - set(check_status))
        failed_checks = sorted(
            rule_id for rule_id in required
            if check_status.get(rule_id) != "pass"
        )
        if missing_checks:
            self._add(
                violations, "V3-VALIDATION-CHECKS-MISSING",
                f"L3校验凭证缺少检查项：{missing_checks}",
            )
        if failed_checks:
            self._add(
                violations, "V3-VALIDATION-CHECKS-FAILED",
                f"L3校验凭证存在未通过项：{failed_checks}",
            )

        expected_hash = l3_report_fingerprint(report)
        if receipt.payload_hash != expected_hash:
            self._add(
                violations, "V3-PAYLOAD-INTEGRITY",
                "L3报告内容与校验凭证指纹不一致，可能在校验后被修改",
            )
        return self._result(violations)

    def assert_valid(self, report: DiagnosticReport) -> dict[str, Any]:
        result = self.validate(report)
        if not result["passed"]:
            raise L3ToL4ValidationError(result["violations"])
        return result

    @staticmethod
    def _add(violations, rule_id, message):
        violations.append({"rule_id": rule_id, "severity": "block", "message": message})

    @staticmethod
    def _result(violations):
        return {
            "passed": not violations,
            "boundary": "L3_TO_L4",
            "violations": violations,
            "route": "allow_l4" if not violations else "return_to_l3",
        }
