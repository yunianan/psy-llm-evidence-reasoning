from typing import Any

from layers.layer4.schemas.l3_report import ICD10ClinicalReview
from shared.report_integrity import (
    ICD10_OUTPUT_POLICY, ICD10_STANDARD, REQUIRED_L3_CHECKS, l3_report_fingerprint,
)
from .diagnosis.icd10_report import reviewed_diagnosis
from .instructions.instruction_generator import (
    BASE_TECHNIQUES, PLUGIN_TECHNIQUES, validate_session_instructions,
)


MIN_CANDIDATE_CONFIDENCE = 0.5


class L3OutputValidator:
    """L3 对自身诊断输出进行确定性自检，并生成供 L4 验证的凭证。"""

    def validate(self, report: dict[str, Any], *, icd10_result: dict | None = None) -> dict[str, Any]:
        checks: list[dict[str, str]] = []
        diagnosis = report.get("diagnosis") or {}
        primary = diagnosis.get("primary")

        if primary is None:
            for rule_id in sorted(REQUIRED_L3_CHECKS):
                self._check(checks, rule_id, False, "缺少主诊断，无法执行该检查")
            return self._receipt(report, checks)

        try:
            validate_session_instructions(report)
            self._check(checks, "V3-TREATMENT-INSTRUCTIONS", True, "会话指令与诊断、疗法一致")
        except (ValueError, KeyError, TypeError) as exc:
            self._check(checks, "V3-TREATMENT-INSTRUCTIONS", False, str(exc))

        candidate_confidence = report.get("candidate_confidence")
        confidence_ok = (
            type(candidate_confidence) in {int, float}
            and MIN_CANDIDATE_CONFIDENCE <= candidate_confidence <= 1
            and candidate_confidence == primary.get("confidence")
            and all(
                type(item.get("confidence")) in {int, float}
                and MIN_CANDIDATE_CONFIDENCE <= item["confidence"] <= 1
                for item in diagnosis.get("secondary", [])
            )
        )
        self._check(
            checks, "V3-CANDIDATE-CONFIDENCE", confidence_ok,
            f"候选诊断支持度={candidate_confidence}，最低门槛={MIN_CANDIDATE_CONFIDENCE}",
        )

        primary_code = primary.get("disorder", "")
        ruled_out_codes = {item.get("disorder", "") for item in diagnosis.get("ruled_out") or []}
        self._check(
            checks,
            "V3-PRIMARY-NOT-RULED-OUT",
            primary_code not in ruled_out_codes,
            f"主诊断={primary.get('disorder', '')}，排除诊断={sorted(ruled_out_codes)}",
        )

        self._check_icd10(report, icd10_result, checks)

        recommended = report.get("recommended_therapy") or {}
        plugins = recommended.get("addon") or []
        triggers = recommended.get("plugin_triggers") or []
        trigger_plugins = [item.get("plugin") for item in triggers]
        plugin_routing_ok = (
            len(plugins) <= 2
            and len(plugins) == len(set(plugins))
            and plugins == trigger_plugins
        )
        plugin_routing_ok = plugin_routing_ok and all(
            plugin in PLUGIN_TECHNIQUES and isinstance(item.get("conditions"), list)
            and item["conditions"] and all(isinstance(text, str) and text.strip()
                                           for text in item["conditions"])
            for plugin, item in zip(plugins, triggers)
        )
        self._check(
            checks,
            "V3-PLUGIN-ROUTING",
            plugin_routing_ok,
            f"插件={plugins}，触发记录={trigger_plugins}，最多允许2个且必须逐项携带触发依据",
        )

        adjustments = recommended.get("strategy_adjustments") or {}
        adjustment_ok = bool(adjustments.get("severity_band")) and isinstance(
            adjustments.get("instructions"), list
        )
        adjustment_ok = adjustment_ok and bool(adjustments.get("instructions")) and all(
            isinstance(item, str) and item.strip() for item in adjustments["instructions"]
        )
        self._check(
            checks,
            "V3-THERAPY-ADJUSTMENT",
            adjustment_ok,
            "复核计划必须提供严重程度分组和可供 L4 执行的微调指令",
        )

        exposure_blocked = adjustments.get("exposure_policy") == "prohibited_until_stabilized"
        exposure_plugins = {"EMDR", "ERP"} & set(plugins)
        addiction_first = adjustments.get("care_priority") == "addiction_treatment_first"
        contraindications = recommended.get("contraindications") or []
        requires_psychiatry = any("精神病性症状" in item for item in contraindications)
        requires_addiction_first = any("先处理成瘾" in item for item in contraindications)
        requires_stabilization = any("先做稳定化" in item for item in contraindications)
        care_priorities = set(adjustments.get("care_priorities") or [])
        if adjustments.get("care_priority"):
            care_priorities.add(adjustments["care_priority"])
        contraindication_ok = (
            not (exposure_blocked and exposure_plugins)
            and not (addiction_first and plugins)
            and (
                not requires_psychiatry
                or "psychiatric_referral" in care_priorities
            )
            and (
                not requires_addiction_first
                or "addiction_treatment_first" in care_priorities
            )
            and (
                not requires_stabilization
                or "stabilization_first" in care_priorities
            )
        )
        self._check(
            checks,
            "V3-CONTRAINDICATION-OVERRIDE",
            contraindication_ok,
            f"暴露禁用={exposure_blocked}，暴露插件={sorted(exposure_plugins)}，成瘾优先={addiction_first}",
        )

        return self._receipt(report, checks)

    def _check_icd10(self, report, result, checks):
        review = None
        try:
            review = ICD10ClinicalReview.model_validate(report.get("icd10_review"))
        except ValueError:
            pass
        review_ok = bool(
            report.get("diagnostic_standard") == ICD10_STANDARD
            and review and result and result.get("status") == "completed"
            and result.get("session_id") == review.tree_session_id
            and result.get("catalog_id") == report.get("icd10_catalog_id")
            and report.get("verifier_decision") == "clinician_confirmed"
            and report.get("verifier_routing") == "pass"
            and report.get("verifier_score") is None
            and report.get("severity_level") == review.severity_level
        )
        self._check(checks, "V3-ICD10-CLINICAL-REVIEW", review_ok, "复核必须对应当前已完成的 ICD-10 树及确认结论")

        evidence_ok = False
        detail = "缺少与当前树运行绑定的 ICD-10 复核"
        if review_ok:
            try:
                expected = reviewed_diagnosis(result, review).model_dump(mode="json")
                evidence_ok = expected == report["diagnosis"]
                detail = "与 ICD-10 路径、WHO 出处和原话证据一致" if evidence_ok else "报告诊断或证据与 ICD-10 路径不一致"
            except (KeyError, ValueError) as exc:
                detail = str(exc)
        self._check(checks, "V3-CRITERIA-EVIDENCE", evidence_ok, detail)

        therapy = report.get("recommended_therapy") or {}
        therapy_parts = therapy.get("primary", "").replace("/", "+").split("+")
        therapy_ok = bool(
            review and review.primary_code == report["diagnosis"]["primary"].get("disorder")
            and review.therapy.model_dump(mode="json") == therapy
            and all(part in BASE_TECHNIQUES for part in therapy_parts)
        )
        self._check(checks, "V3-THERAPY-MATCH", therapy_ok, "ICD-10 编码和受支持的治疗计划须与复核一致")

    @staticmethod
    def _check(checks, rule_id, passed, detail):
        checks.append({
            "rule_id": rule_id,
            "status": "pass" if passed else "fail",
            "detail": detail,
        })

    @staticmethod
    def _receipt(report, checks):
        passed = all(item["status"] == "pass" for item in checks)
        return {
            "passed": passed,
            "route": "allow_l4" if passed else "reject",
            "checks": checks,
            "payload_hash": l3_report_fingerprint(report),
            "policy_version": ICD10_OUTPUT_POLICY,
        }
