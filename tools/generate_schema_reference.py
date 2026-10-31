"""从实际类型定义生成 L1-L5 Schema 阅读文档。"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin, is_typeddict

from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT = PROJECT_ROOT / "docs" / "layer-schema-reference.md"

SECTIONS = {
    "L1：安全入口": [
        "layers.layer1.models",
        "layers.layer1.risk_classifier",
        "shared.schemas.safety_schema",
    ],
    "L2：评估、量表与患者画像": ["shared.clinical_contract", "shared.clinical_profile", "layers.layer2.schemas.models"],
    "L3：诊断、疗法选择与治疗指令": [
        "layers.layer3.diagnosis.schemas",
        "layers.layer3.api",
        "layers.layer3.instructions.turn_planner",
        "shared.schemas.treatment_instructions",
    ],
    "L4：治疗对话": [
        "layers.layer4.schemas.l3_report",
        "layers.layer4.schemas.summary",
        "layers.layer4.service",
        "layers.layer4.api.server",
        "layers.layer4.generation.responder",
        "layers.layer4.review.orchestrator",
        "layers.layer4.llm.base",
        "layers.layer4.session.history",
        "layers.layer4.session.state_machine",
    ],
    "L5：独立输出审核": [
        "layers.layer4.schemas.l5_context",
        "layers.layer4.schemas.review",
    ],
    "顶层编排器": ["orchestration.schemas"],
}

INTERFACE_MODELS = {
    "SafetyPassport", "SafetyModelUnavailable", "DialogueResult", "ClinicalIntake",
    "DiagnoseRequest", "DiagnoseResponse", "TreatmentInstructions", "DiagnosticReport",
    "SessionRequest", "TurnRequest", "TurnOutcome", "L5Context", "ReviewResult",
    "SessionCreated", "RewriteRequest", "OrchestrationOutcome", "PatientConfirmation",
}


def _type_name(annotation: Any) -> str:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Literal:
        return "Literal[" + ", ".join(repr(item) for item in args) + "]"
    if origin in (list, dict, tuple, set):
        name = origin.__name__
        return f"{name}[{', '.join(_type_name(item) for item in args)}]"
    if origin in (Union, UnionType):
        return " | ".join(_type_name(item) for item in args)
    if annotation is Any:
        return "Any"
    return getattr(annotation, "__name__", str(annotation).replace("typing.", ""))


def _source_link(obj: type) -> str:
    source = Path(inspect.getsourcefile(obj) or "").resolve()
    line = inspect.getsourcelines(obj)[1]
    relative = source.relative_to(PROJECT_ROOT).as_posix()
    return f"[`{relative}:{line}`](../{relative}#L{line})"


def _default(field: Any) -> str:
    if field.is_required():
        return "—"
    if field.default_factory is not None:
        return "工厂生成"
    value = field.default
    if isinstance(value, (str, int, float, bool)) or value is None:
        return f"`{value!r}`"
    return "有默认值"


def _constraints(schema: dict[str, Any]) -> str:
    result = []
    for key, label in (
        ("minimum", "最小"), ("maximum", "最大"),
        ("minLength", "最短长度"), ("maxLength", "最长长度"),
        ("minItems", "最少项"), ("maxItems", "最多项"),
    ):
        if key in schema:
            result.append(f"{label}={schema[key]}")
    if "enum" in schema:
        result.append("枚举=" + ", ".join(repr(item) for item in schema["enum"]))
    return "；".join(result) or "—"


def _pydantic_table(model: type[BaseModel]) -> list[str]:
    schema = model.model_json_schema()
    properties = schema.get("properties", {})
    lines = [
        "| 字段 | 类型 | 必填 | 默认值 | 约束 |",
        "|---|---|:---:|---|---|",
    ]
    for name, field in model.model_fields.items():
        prop = properties.get(name, {})
        lines.append(
            f"| `{name}` | `{_type_name(field.annotation)}` | "
            f"{'是' if field.is_required() else '否'} | {_default(field)} | {_constraints(prop)} |"
        )
    return lines


def _typed_dict_table(model: type) -> list[str]:
    required = getattr(model, "__required_keys__", set())
    lines = ["| 字段 | 类型 | 必填 |", "|---|---|:---:|"]
    for name, annotation in model.__annotations__.items():
        lines.append(f"| `{name}` | `{_type_name(annotation)}` | {'是' if name in required else '否'} |")
    return lines


def _models(module_name: str) -> list[type]:
    if module_name == "layers.layer4.api.server":
        previous = {
            name: os.environ.get(name) for name in ("L4_LLM_PROVIDER", "L4_REVIEW_MODE")
        }
        os.environ["L4_LLM_PROVIDER"] = "mock"
        os.environ["L4_REVIEW_MODE"] = "mock"
        try:
            module = importlib.import_module(module_name)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    else:
        module = importlib.import_module(module_name)
    found = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module_name:
            continue
        if is_typeddict(obj) or (isinstance(obj, type) and issubclass(obj, BaseModel)):
            found.append(obj)
    return sorted(found, key=lambda item: inspect.getsourcelines(item)[1])


def render() -> str:
    lines = [
        "# L1–L5 Schema 参考",
        "",
        "> 本文档由 `tools/generate_schema_reference.py` 从当前代码自动生成，请勿手工修改。",
        "> 修改 Schema 后运行 `python tools/generate_schema_reference.py --write`；一致性测试会阻止代码与本文档不同步。",
        "",
        "## 如何阅读",
        "",
        "- **层间接口**：会跨层、API 或患者界面传递，应视为稳定契约。",
        "- **内部结构**：只服务于本层的解析、推理或会话状态，可以演进，但仍受严格校验。",
        "- `extra='forbid'` 的 Pydantic 模型会拒绝未知字段；`Literal` 只接受列出的精确值。",
        "- L1 同时存在 Pydantic 模型输出和 `TypedDict` 通行证；`TypedDict` 只提供静态类型提示，运行时守卫另行验证。",
        "",
        "## 层间主链",
        "",
        "```text",
        "患者文本 → L1 SafetyPassport → L2 ClinicalIntake → L3 DiagnosticReport",
        "→ TreatmentInstructions → L4 TurnOutcome/L5Context → L5 ReviewResult",
        "→ OrchestrationOutcome",
        "```",
        "",
    ]
    for section, modules in SECTIONS.items():
        lines.extend([f"## {section}", ""])
        for module_name in modules:
            lines.extend([f"### `{module_name}`", ""])
            for model in _models(module_name):
                kind = "层间接口" if model.__name__ in INTERFACE_MODELS else "内部结构"
                description = inspect.getdoc(model) or "无类级说明。"
                if is_typeddict(model) and description.startswith("dict("):
                    description = "TypedDict 字段契约；运行时校验由对应守卫负责。"
                if description.startswith("Usage docs:"):
                    description = "无类级说明。"
                lines.extend([
                    f"#### `{model.__name__}` · {kind}", "",
                    f"源码：{_source_link(model)}", "",
                    description.splitlines()[0], "",
                ])
                if is_typeddict(model):
                    lines.extend(_typed_dict_table(model))
                else:
                    lines.extend(_pydantic_table(model))
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    content = render()
    if args.write:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(content, encoding="utf-8")
        print(OUTPUT)
    else:
        print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
