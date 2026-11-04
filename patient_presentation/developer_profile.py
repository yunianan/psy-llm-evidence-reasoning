"""生成仅供开发者查看的完整患者画像 HTML。"""

import html
import json
import re
from pathlib import Path
from typing import Any

from orchestration.schemas import PipelineSession


class DeveloperProfileRenderer:
    """生成仅供开发调试的患者画像 HTML；不会进入患者可见返回对象。"""

    def __init__(self, output_dir: str | Path = "artifacts/developer_profiles") -> None:
        self.output_dir = Path(output_dir)

    def render(self, session: PipelineSession) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", session.session_id)
        target = self.output_dir / f"{safe_id}.html"
        temporary = target.with_suffix(".html.tmp")
        temporary.write_text(self._document(session), encoding="utf-8")
        temporary.replace(target)
        return target.resolve()

    def _document(self, session: PipelineSession) -> str:
        passport = session.l1_passport or {}
        intake = session.clinical_intake
        report = session.diagnostic_report

        overview = {
            "session_id": session.session_id,
            "route": session.route,
            "expected_input": session.expected_input,
            "updated_at": session.updated_at,
            "stopped_reason": session.stopped_reason,
        }
        l1 = {
            "risk_level": passport.get("risk_level"),
            "safety_flag": passport.get("safety_flag"),
            "crisis_flag": passport.get("crisis_flag"),
            "sensitive_topics": passport.get("sensitive_topics", []),
            "flags": passport.get("flags", []),
            "execution": passport.get("execution", {}),
            "guard": passport.get("guard", {}),
        }
        intake_data = intake.model_dump(mode="json") if intake else {}
        report_data = report.model_dump(mode="json") if report else {}
        icd10 = (session.l3_raw or {}).get("model_result", {})
        review_material = {
            key: icd10[key] for key in ("standard", "session_id", "catalog_id", "status", "candidates")
            if key in icd10
        } if icd10.get("standard") == "WHO ICD-10" else {}

        return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>开发者患者画像 · {html.escape(session.session_id)}</title>
<style>
:root {{ color-scheme: light; font-family: Inter, "Microsoft YaHei", sans-serif; }}
body {{ margin: 0; background: #f4f7fb; color: #1f2937; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
.warning {{ background: #2b1b1b; color: #fecaca; padding: 14px 18px; border-radius: 12px; font-weight: 700; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(320px,1fr)); gap: 16px; margin-top: 18px; }}
.card {{ background: white; border: 1px solid #dbe4ef; border-radius: 14px; padding: 18px; box-shadow: 0 4px 14px #1f29370d; }}
.wide {{ grid-column: 1 / -1; }}
h1 {{ margin-bottom: 8px; }} h2 {{ margin: 0 0 12px; font-size: 18px; }}
.route {{ display: inline-block; padding: 5px 10px; border-radius: 999px; background: #dbeafe; color: #1d4ed8; font-weight: 700; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #f8fafc; border-radius: 10px; padding: 14px; font-size: 12px; }}
details summary {{ cursor: pointer; font-weight: 700; }}
</style>
</head>
<body><main>
<div class="warning">仅限开发者调试：包含患者原话、量表明细、内部评分和证据链，不得展示给患者。</div>
<h1>患者画像调试视图</h1>
<span class="route">{html.escape(session.route)}</span>
<div class="grid">
{self._card("编排状态", overview)}
{self._card("L1 安全通行证", l1)}
{self._card("L2 患者画像", intake_data, wide=True)}
{self._card("L3 ICD-10 候选与复核材料", review_material, wide=True) if review_material else ""}
{self._card("L3 诊断与放行校验", report_data, wide=True)}
</div>
</main></body></html>"""

    @staticmethod
    def _card(title: str, value: Any, *, wide: bool = False) -> str:
        encoded = html.escape(json.dumps(value, ensure_ascii=False, indent=2))
        klass = "card wide" if wide else "card"
        return (
            f'<section class="{klass}"><h2>{html.escape(title)}</h2>'
            f"<pre>{encoded}</pre></section>"
        )
