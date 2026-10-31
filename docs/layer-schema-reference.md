# L1–L5 Schema 参考

> 本文档由 `tools/generate_schema_reference.py` 从当前代码自动生成，请勿手工修改。
> 修改 Schema 后运行 `python tools/generate_schema_reference.py --write`；一致性测试会阻止代码与本文档不同步。

## 如何阅读

- **层间接口**：会跨层、API 或患者界面传递，应视为稳定契约。
- **内部结构**：只服务于本层的解析、推理或会话状态，可以演进，但仍受严格校验。
- `extra='forbid'` 的 Pydantic 模型会拒绝未知字段；`Literal` 只接受列出的精确值。
- L1 同时存在 Pydantic 模型输出和 `TypedDict` 通行证；`TypedDict` 只提供静态类型提示，运行时守卫另行验证。

## 层间主链

```text
患者文本 → L1 SafetyPassport → L2 ClinicalIntake → L3 DiagnosticReport
→ TreatmentInstructions → L4 TurnOutcome/L5Context → L5 ReviewResult
→ OrchestrationOutcome
```

## L1：安全入口

### `layers.layer1.models`

#### `SafetySignalAnalysis` · 内部结构

源码：[`layers/layer1/models.py:19`](../layers/layer1/models.py#L19)

LLM 只判定 L1 三条顶层路线，不评估即时危险。

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `suicide_signal_detected` | `bool` | 是 | — | — |
| `jailbreak_detected` | `bool` | 是 | — | — |
| `safe_to_enter_l2` | `bool` | 是 | — | — |
| `confidence` | `float` | 是 | — | 最小=0；最大=1 |
| `evidence_quotes` | `list[str]` | 否 | 工厂生成 | — |
| `jailbreak_evidence` | `list[str]` | 否 | 工厂生成 | — |
| `asq_transition_support` | `str` | 否 | `''` | 最长长度=80 |
| `summary` | `str` | 是 | — | — |

#### `ASQState` · 内部结构

源码：[`layers/layer1/models.py:48`](../layers/layer1/models.py#L48)

NIMH ASQ 2025 成人筛查状态；题目与判定由规则控制。

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `status` | `Literal['not_started', 'in_progress', 'negative', 'non_acute_positive', 'acute_positive', 'refused']` | 否 | `'not_started'` | 枚举='not_started', 'in_progress', 'negative', 'non_acute_positive', 'acute_positive', 'refused' |
| `trigger` | `Literal['baseline', 'signal']` | 否 | `'baseline'` | 枚举='baseline', 'signal' |
| `current_question` | `int` | 否 | `0` | 最小=0；最大=5 |
| `awaiting_attempt_timing` | `bool` | 否 | `False` | — |
| `answers` | `dict[int, bool | NoneType]` | 否 | 工厂生成 | — |
| `most_recent_attempt_timing` | `str` | 否 | `''` | — |

#### `BSSAState` · 内部结构

源码：[`layers/layer1/models.py:64`](../layers/layer1/models.py#L64)

成人门诊 BSSA 的结构化事实与安全计划状态。

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `status` | `Literal['not_started', 'in_progress', 'completed']` | 否 | `'not_started'` | 枚举='not_started', 'in_progress', 'completed' |
| `turn_count` | `int` | 否 | `0` | 最小=0 |
| `current_ideation` | `Literal['yes', 'no', 'unknown', 'refused']` | 否 | `'unknown'` | 枚举='yes', 'no', 'unknown', 'refused' |
| `ideation_frequency` | `str` | 否 | `''` | — |
| `last_occurrence` | `str` | 否 | `''` | — |
| `plan_present` | `Literal['yes', 'no', 'unknown', 'refused']` | 否 | `'unknown'` | 枚举='yes', 'no', 'unknown', 'refused' |
| `plan_description` | `str` | 否 | `''` | — |
| `access_to_means` | `Literal['yes', 'no', 'unknown', 'refused']` | 否 | `'unknown'` | 枚举='yes', 'no', 'unknown', 'refused' |
| `intent_present` | `Literal['yes', 'no', 'unknown', 'refused']` | 否 | `'unknown'` | 枚举='yes', 'no', 'unknown', 'refused' |
| `past_attempts` | `Literal['yes', 'no', 'unknown', 'refused']` | 否 | `'unknown'` | 枚举='yes', 'no', 'unknown', 'refused' |
| `most_recent_attempt_timing` | `str` | 否 | `''` | — |
| `preparatory_behavior` | `Literal['yes', 'no', 'unknown', 'refused']` | 否 | `'unknown'` | 枚举='yes', 'no', 'unknown', 'refused' |
| `hopelessness` | `Literal['yes', 'no', 'unknown', 'refused']` | 否 | `'unknown'` | 枚举='yes', 'no', 'unknown', 'refused' |
| `impulsivity` | `Literal['yes', 'no', 'unknown', 'refused']` | 否 | `'unknown'` | 枚举='yes', 'no', 'unknown', 'refused' |
| `psychosis` | `Literal['yes', 'no', 'unknown', 'refused']` | 否 | `'unknown'` | 枚举='yes', 'no', 'unknown', 'refused' |
| `substance_impairment` | `Literal['yes', 'no', 'unknown', 'refused']` | 否 | `'unknown'` | 枚举='yes', 'no', 'unknown', 'refused' |
| `trusted_support_available` | `Literal['yes', 'no', 'unknown', 'refused']` | 否 | `'unknown'` | 枚举='yes', 'no', 'unknown', 'refused' |
| `reasons_for_living` | `list[str]` | 否 | 工厂生成 | — |
| `can_maintain_safety` | `Literal['yes', 'no', 'unknown', 'refused']` | 否 | `'unknown'` | 枚举='yes', 'no', 'unknown', 'refused' |
| `trusted_contact` | `str` | 否 | `''` | — |
| `coping_strategy` | `str` | 否 | `''` | — |
| `means_restriction` | `str` | 否 | `''` | — |
| `professional_contact` | `str` | 否 | `''` | — |
| `refused_fields` | `list[str]` | 否 | 工厂生成 | — |
| `evidence_quotes` | `dict[str, str]` | 否 | 工厂生成 | — |

#### `BSSAExtraction` · 内部结构

源码：[`layers/layer1/models.py:96`](../layers/layer1/models.py#L96)

BSSA LLM 单轮结构化提取与下一问。

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `current_ideation` | `Literal['yes', 'no', 'unknown', 'refused'] | NoneType` | 否 | `None` | — |
| `ideation_frequency` | `str | NoneType` | 否 | `None` | — |
| `last_occurrence` | `str | NoneType` | 否 | `None` | — |
| `plan_present` | `Literal['yes', 'no', 'unknown', 'refused'] | NoneType` | 否 | `None` | — |
| `plan_description` | `str | NoneType` | 否 | `None` | — |
| `access_to_means` | `Literal['yes', 'no', 'unknown', 'refused'] | NoneType` | 否 | `None` | — |
| `intent_present` | `Literal['yes', 'no', 'unknown', 'refused'] | NoneType` | 否 | `None` | — |
| `past_attempts` | `Literal['yes', 'no', 'unknown', 'refused'] | NoneType` | 否 | `None` | — |
| `most_recent_attempt_timing` | `str | NoneType` | 否 | `None` | — |
| `preparatory_behavior` | `Literal['yes', 'no', 'unknown', 'refused'] | NoneType` | 否 | `None` | — |
| `hopelessness` | `Literal['yes', 'no', 'unknown', 'refused'] | NoneType` | 否 | `None` | — |
| `impulsivity` | `Literal['yes', 'no', 'unknown', 'refused'] | NoneType` | 否 | `None` | — |
| `psychosis` | `Literal['yes', 'no', 'unknown', 'refused'] | NoneType` | 否 | `None` | — |
| `substance_impairment` | `Literal['yes', 'no', 'unknown', 'refused'] | NoneType` | 否 | `None` | — |
| `trusted_support_available` | `Literal['yes', 'no', 'unknown', 'refused'] | NoneType` | 否 | `None` | — |
| `reasons_for_living` | `list[str] | NoneType` | 否 | `None` | — |
| `can_maintain_safety` | `Literal['yes', 'no', 'unknown', 'refused'] | NoneType` | 否 | `None` | — |
| `trusted_contact` | `str | NoneType` | 否 | `None` | — |
| `coping_strategy` | `str | NoneType` | 否 | `None` | — |
| `means_restriction` | `str | NoneType` | 否 | `None` | — |
| `professional_contact` | `str | NoneType` | 否 | `None` | — |
| `refused_fields` | `list[str]` | 否 | 工厂生成 | — |
| `evidence_quotes` | `dict[str, str]` | 否 | 工厂生成 | — |
| `next_question_target` | `str` | 是 | — | — |
| `next_question` | `str` | 是 | — | — |

#### `SupportResponse` · 内部结构

源码：[`layers/layer1/models.py:128`](../layers/layer1/models.py#L128)

安全过渡阶段的个性化支持回应。

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `supportive_message` | `str` | 是 | — | 最短长度=20；最长长度=360 |

#### `L1SessionState` · 内部结构

源码：[`layers/layer1/models.py:136`](../layers/layer1/models.py#L136)

L1 内部会话状态，不跨层传递。

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `session_id` | `str` | 是 | — | — |
| `phase` | `Literal['monitoring', 'asq', 'bssa', 'safety_hold']` | 否 | `'monitoring'` | 枚举='monitoring', 'asq', 'bssa', 'safety_hold' |
| `baseline_asq_status` | `Literal['not_due', 'due', 'negative', 'refused', 'positive']` | 否 | `'not_due'` | 枚举='not_due', 'due', 'negative', 'refused', 'positive' |
| `asq` | `ASQState` | 否 | 工厂生成 | — |
| `bssa` | `BSSAState` | 否 | 工厂生成 | — |
| `disposition` | `Literal['not_assessed', 'emergency_evaluation', 'further_evaluation', 'non_urgent_follow_up', 'no_further_safety_intervention']` | 否 | `'not_assessed'` | 枚举='not_assessed', 'emergency_evaluation', 'further_evaluation', 'non_urgent_follow_up', 'no_further_safety_intervention' |
| `last_signal` | `SafetySignalAnalysis | NoneType` | 否 | `None` | — |
| `active_controls` | `list[str]` | 否 | 工厂生成 | — |
| `last_support_message` | `str` | 否 | `''` | — |

### `layers.layer1.risk_classifier`

### `shared.schemas.safety_schema`

#### `EvidenceRef` · 内部结构

源码：[`shared/schemas/safety_schema.py:4`](../shared/schemas/safety_schema.py#L4)

TypedDict 字段契约；运行时校验由对应守卫负责。

| 字段 | 类型 | 必填 |
|---|---|:---:|
| `source` | `str` | 是 |
| `finding` | `str` | 是 |
| `basis` | `str` | 是 |

#### `RiskScoreBreakdown` · 内部结构

源码：[`shared/schemas/safety_schema.py:10`](../shared/schemas/safety_schema.py#L10)

TypedDict 字段契约；运行时校验由对应守卫负责。

| 字段 | 类型 | 必填 |
|---|---|:---:|
| `rule` | `int` | 是 |
| `llm` | `int` | 是 |
| `scale` | `int` | 是 |

#### `SecurityEvent` · 内部结构

源码：[`shared/schemas/safety_schema.py:16`](../shared/schemas/safety_schema.py#L16)

TypedDict 字段契约；运行时校验由对应守卫负责。

| 字段 | 类型 | 必填 |
|---|---|:---:|
| `detected` | `bool` | 是 |
| `event_type` | `str` | 是 |
| `severity` | `int` | 是 |
| `matched_patterns` | `list[str]` | 是 |

#### `SafetyPassport` · 层间接口

源码：[`shared/schemas/safety_schema.py:23`](../shared/schemas/safety_schema.py#L23)

TypedDict 字段契约；运行时校验由对应守卫负责。

| 字段 | 类型 | 必填 |
|---|---|:---:|
| `schema_version` | `Literal['1.0']` | 是 |
| `layer_id` | `Literal['L1']` | 是 |
| `session_id` | `str` | 是 |
| `message_id` | `str` | 是 |
| `timestamp` | `str` | 是 |
| `safety_flag` | `Literal['pass', 'warn', 'block']` | 是 |
| `risk_level` | `int` | 是 |
| `risk_types` | `list[str]` | 是 |
| `crisis_flag` | `bool` | 是 |
| `decision_path` | `Literal['clinical_crisis', 'safety_assessment', 'safety_pause', 'security_reject', 'normal']` | 是 |
| `security_event` | `SecurityEvent` | 是 |
| `sensitive_topics` | `list[str]` | 是 |
| `summary` | `str` | 是 |
| `structured_labels` | `dict[str, Any]` | 是 |
| `evidence_refs` | `list[EvidenceRef]` | 是 |
| `confidence` | `float` | 是 |
| `flags` | `list[str]` | 是 |
| `instruction_to_next` | `str` | 是 |
| `execution` | `dict[str, Any]` | 是 |
| `guard` | `dict[str, Any]` | 是 |

#### `SafetyModelUnavailable` · 层间接口

源码：[`shared/schemas/safety_schema.py:49`](../shared/schemas/safety_schema.py#L49)

L1 的 LLM 连续失败时返回的终止错误，不能传递给 L2。

| 字段 | 类型 | 必填 |
|---|---|:---:|
| `schema_version` | `Literal['1.0']` | 是 |
| `layer_id` | `Literal['L1']` | 是 |
| `session_id` | `str` | 是 |
| `message_id` | `str` | 是 |
| `timestamp` | `str` | 是 |
| `status` | `Literal['error']` | 是 |
| `error_code` | `Literal['L1_API_ERROR', 'L1_MODEL_OUTPUT_INVALID']` | 是 |
| `message` | `str` | 是 |
| `terminal` | `Literal[True]` | 是 |
| `can_call_l2` | `Literal[False]` | 是 |
| `details` | `dict[str, Any]` | 是 |
| `evidence_refs` | `list[EvidenceRef]` | 是 |

## L2：评估、量表与患者画像

### `shared.clinical_contract`

#### `DurationEvidence` · 内部结构

源码：[`shared/clinical_contract.py:23`](../shared/clinical_contract.py#L23)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `symptom_id` | `str` | 否 | `''` | — |
| `scope` | `str` | 否 | `'unknown'` | — |
| `raw_span` | `str` | 是 | — | 最短长度=1 |
| `min_days` | `float | NoneType` | 否 | `None` | — |
| `max_days` | `float | NoneType` | 否 | `None` | — |
| `unit` | `Literal['day', 'week', 'month', 'year', 'relative', 'unknown']` | 否 | `'unknown'` | 枚举='day', 'week', 'month', 'year', 'relative', 'unknown' |
| `quantity` | `float | NoneType` | 否 | `None` | — |
| `approx` | `bool` | 否 | `False` | — |
| `approximation` | `Literal['none', 'around', 'more', 'under', 'near', 'colloquial']` | 否 | `'none'` | 枚举='none', 'around', 'more', 'under', 'near', 'colloquial' |
| `anchor` | `str` | 否 | `'relative_to_interview'` | — |
| `interview_date` | `date | NoneType` | 否 | `None` | — |
| `pattern` | `Literal['continuous', 'episodic', 'unknown']` | 否 | `'unknown'` | 枚举='continuous', 'episodic', 'unknown' |
| `duration_kind` | `Literal['overall', 'episode', 'unknown']` | 否 | `'overall'` | 枚举='overall', 'episode', 'unknown' |
| `parse_status` | `Literal['ok', 'vague', 'failed', 'unresolved']` | 否 | `'failed'` | 枚举='ok', 'vague', 'failed', 'unresolved' |
| `source` | `Literal['L2', 'L3_writeback', 'legacy_adapter']` | 否 | `'L2'` | 枚举='L2', 'L3_writeback', 'legacy_adapter' |
| `evidence_refs` | `list[str]` | 否 | 工厂生成 | — |
| `policy_version` | `str` | 否 | `'duration-interval-v1'` | — |

#### `TemporalDecision` · 内部结构

源码：[`shared/clinical_contract.py:53`](../shared/clinical_contract.py#L53)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `status` | `Literal['met', 'not_met', 'unknown', 'met_approx']` | 是 | — | 枚举='met', 'not_met', 'unknown', 'met_approx' |
| `threshold_days` | `float | NoneType` | 否 | `None` | — |
| `reason` | `str` | 是 | — | — |
| `evidence_refs` | `list[str]` | 否 | 工厂生成 | — |
| `suggested_question` | `str` | 否 | `'大概从几月份开始的？期间一直如此，还是反复发作？'` | — |
| `policy_version` | `str` | 否 | `'duration-interval-v1'` | — |

#### `ClinicalFact` · 内部结构

源码：[`shared/clinical_contract.py:63`](../shared/clinical_contract.py#L63)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `fact_id` | `str` | 是 | — | — |
| `symptom_id` | `str` | 否 | `''` | — |
| `attribute` | `Literal['existence', 'onset', 'duration', 'frequency', 'severity', 'impact', 'trigger', 'denial', 'clarification']` | 是 | — | 枚举='existence', 'onset', 'duration', 'frequency', 'severity', 'impact', 'trigger', 'denial', 'clarification' |
| `state` | `Literal['present', 'absent', 'unknown', 'conflict']` | 否 | `'unknown'` | 枚举='present', 'absent', 'unknown', 'conflict' |
| `value` | `Any` | 否 | `None` | — |
| `raw_span` | `str` | 是 | — | — |
| `evidence_refs` | `list[str]` | 否 | 工厂生成 | — |
| `duration` | `DurationEvidence | NoneType` | 否 | `None` | — |

#### `FollowUpTask` · 内部结构

源码：[`shared/clinical_contract.py:75`](../shared/clinical_contract.py#L75)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `task_id` | `str` | 是 | — | — |
| `field` | `str` | 是 | — | — |
| `node_id` | `str` | 否 | `''` | — |
| `symptom_id` | `str` | 否 | `''` | — |
| `attribute` | `Literal['existence', 'onset', 'duration', 'frequency', 'severity', 'impact', 'trigger', 'denial', 'clarification']` | 否 | `'clarification'` | 枚举='existence', 'onset', 'duration', 'frequency', 'severity', 'impact', 'trigger', 'denial', 'clarification' |
| `purpose` | `str` | 否 | `''` | — |
| `suggested_question` | `str` | 是 | — | — |
| `priority` | `str` | 否 | `'remaining'` | — |
| `required_by` | `list[str]` | 否 | 工厂生成 | — |
| `source_refs` | `list[str]` | 否 | 工厂生成 | — |
| `expected_by_candidate` | `dict[str, str]` | 否 | 工厂生成 | — |
| `status` | `Literal['pending', 'answered', 'unresolved', 'conflict']` | 否 | `'pending'` | 枚举='pending', 'answered', 'unresolved', 'conflict' |
| `answer_evidence_refs` | `list[str]` | 否 | 工厂生成 | — |

### `shared.clinical_profile`

#### `ClinicalProfile` · 内部结构

源码：[`shared/clinical_profile.py:8`](../shared/clinical_profile.py#L8)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `contract_version` | `Literal['l2-l3-clinical-v2']` | 否 | `'l2-l3-clinical-v2'` | — |
| `interview_date` | `date | NoneType` | 否 | `None` | — |
| `clinical_facts` | `list[ClinicalFact]` | 否 | 工厂生成 | — |
| `duration_evidence` | `list[DurationEvidence]` | 否 | 工厂生成 | — |
| `follow_up_tasks` | `list[FollowUpTask]` | 否 | 工厂生成 | — |
| `temporal_audit` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `PHQ9` | `float | NoneType` | 否 | `None` | — |
| `GAD7` | `float | NoneType` | 否 | `None` | — |
| `MDQ` | `float | NoneType` | 否 | `None` | — |
| `PCL5` | `float | NoneType` | 否 | `None` | — |
| `ISI` | `float | NoneType` | 否 | `None` | — |
| `Y_BOCS` | `float | NoneType` | 否 | `None` | — |
| `ADNM` | `float | NoneType` | 否 | `None` | — |
| `RRS` | `float | NoneType` | 否 | `None` | — |
| `AAQ_II` | `float | NoneType` | 否 | `None` | — |
| `PHQ15` | `float | NoneType` | 否 | `None` | — |
| `PHQ9_items` | `dict[str, float]` | 否 | 工厂生成 | — |
| `GAD7_items` | `dict[str, float]` | 否 | 工厂生成 | — |
| `MDQ_items` | `dict[str, float]` | 否 | 工厂生成 | — |
| `PCL5_items` | `dict[str, float]` | 否 | 工厂生成 | — |
| `ISI_items` | `dict[str, float]` | 否 | 工厂生成 | — |
| `Y_BOCS_items` | `dict[str, float]` | 否 | 工厂生成 | — |
| `ADNM_items` | `dict[str, float]` | 否 | 工厂生成 | — |
| `RRS_items` | `dict[str, float]` | 否 | 工厂生成 | — |
| `AAQ_II_items` | `dict[str, float]` | 否 | 工厂生成 | — |
| `PHQ15_items` | `dict[str, float]` | 否 | 工厂生成 | — |
| `relapse_count` | `Annotated | NoneType` | 否 | `None` | — |
| `chronic_months` | `Annotated | NoneType` | 否 | `None` | — |
| `interpersonal_conflicts` | `Annotated | NoneType` | 否 | `None` | — |
| `symptom_duration_days` | `Annotated | NoneType` | 否 | `None` | — |
| `functional_impairment` | `Annotated | NoneType` | 否 | `None` | — |
| `mania_or_hypomania_history` | `Annotated | NoneType` | 否 | `None` | — |
| `substance_or_medical_cause` | `Annotated | NoneType` | 否 | `None` | — |
| `rhythm_disturbance` | `Annotated | NoneType` | 否 | `None` | — |
| `exposure_avoidant` | `Annotated | NoneType` | 否 | `None` | — |
| `ritual_behaviors` | `Annotated | NoneType` | 否 | `None` | — |
| `insomnia_chief_complaint` | `Annotated | NoneType` | 否 | `None` | — |
| `interpersonal_impairment` | `Annotated | NoneType` | 否 | `None` | — |
| `single_focus` | `Annotated | NoneType` | 否 | `None` | — |
| `psychotic_symptoms` | `Annotated | NoneType` | 否 | `None` | — |
| `substance_active` | `Annotated | NoneType` | 否 | `None` | — |
| `severe_dissociation` | `Annotated | NoneType` | 否 | `None` | — |
| `differential_diagnoses_assessed` | `list[str] | NoneType` | 否 | `None` | — |
| `evidence_conflicts` | `list[dict[str, Any]] | NoneType` | 否 | `None` | — |
| `diagnostic_details` | `dict[str, Any]` | 否 | 工厂生成 | — |
| `key_symptoms_extracted` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `emotions` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `cognitive_patterns` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `key_narratives` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `functional_impairments` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `patient_goals` | `list[str]` | 否 | 工厂生成 | — |
| `motivation` | `dict[str, Any]` | 否 | 工厂生成 | — |
| `evidence_nodes` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |

### `layers.layer2.schemas.models`

#### `SemanticCandidate` · 内部结构

源码：[`layers/layer2/schemas/models.py:50`](../layers/layer2/schemas/models.py#L50)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `value` | `str` | 是 | — | — |
| `evidence_quote` | `str` | 是 | — | — |
| `attributes` | `dict[str, Any]` | 否 | 工厂生成 | — |

#### `StructuredFieldCandidate` · 内部结构

源码：[`layers/layer2/schemas/models.py:66`](../layers/layer2/schemas/models.py#L66)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `field` | `Literal['symptom_duration_days', 'functional_impairment', 'mania_or_hypomania_history', 'substance_or_medical_cause', 'differential_diagnoses_assessed', 'evidence_conflicts', 'diagnostic_details', 'relapse_count', 'rhythm_disturbance', 'exposure_avoidant', 'ritual_behaviors', 'insomnia_chief_complaint', 'chronic_months', 'interpersonal_conflicts', 'interpersonal_impairment', 'single_focus', 'psychotic_symptoms', 'substance_active', 'severe_dissociation', 'motivation_importance', 'motivation_confidence', 'motivation_readiness', 'scale_consent', 'scale_answer_score', 'confirmation_decision']` | 是 | — | 枚举='symptom_duration_days', 'functional_impairment', 'mania_or_hypomania_history', 'substance_or_medical_cause', 'differential_diagnoses_assessed', 'evidence_conflicts', 'diagnostic_details', 'relapse_count', 'rhythm_disturbance', 'exposure_avoidant', 'ritual_behaviors', 'insomnia_chief_complaint', 'chronic_months', 'interpersonal_conflicts', 'interpersonal_impairment', 'single_focus', 'psychotic_symptoms', 'substance_active', 'severe_dissociation', 'motivation_importance', 'motivation_confidence', 'motivation_readiness', 'scale_consent', 'scale_answer_score', 'confirmation_decision' |
| `value` | `Any` | 是 | — | — |
| `evidence_quote` | `str` | 是 | — | — |

#### `CorrectionCandidate` · 内部结构

源码：[`layers/layer2/schemas/models.py:101`](../layers/layer2/schemas/models.py#L101)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `target_path` | `str` | 是 | — | — |
| `action` | `Literal['remove', 'replace']` | 是 | — | 枚举='remove', 'replace' |
| `value` | `Any` | 否 | `None` | — |
| `evidence_quote` | `str` | 是 | — | — |

#### `SemanticAnalysis` · 内部结构

源码：[`layers/layer2/schemas/models.py:118`](../layers/layer2/schemas/models.py#L118)

语义分析 API 的严格输出；这里的内容尚未成为患者事实。

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `input_type` | `Literal['free_narrative', 'focus_selection', 'scale_consent', 'scale_answer', 'goal_statement', 'motivation_rating', 'profile_confirmation', 'correction', 'refusal', 'pause', 'off_topic']` | 是 | — | 枚举='free_narrative', 'focus_selection', 'scale_consent', 'scale_answer', 'goal_statement', 'motivation_rating', 'profile_confirmation', 'correction', 'refusal', 'pause', 'off_topic' |
| `symptoms` | `list[SemanticCandidate]` | 否 | 工厂生成 | — |
| `emotions` | `list[SemanticCandidate]` | 否 | 工厂生成 | — |
| `cognitive_patterns` | `list[SemanticCandidate]` | 否 | 工厂生成 | — |
| `patient_goals` | `list[SemanticCandidate]` | 否 | 工厂生成 | — |
| `key_narratives` | `list[SemanticCandidate]` | 否 | 工厂生成 | — |
| `functional_impairments` | `list[SemanticCandidate]` | 否 | 工厂生成 | — |
| `focus_topics` | `list[SemanticCandidate]` | 否 | 工厂生成 | — |
| `change_talk` | `list[SemanticCandidate]` | 否 | 工厂生成 | — |
| `sustain_talk` | `list[SemanticCandidate]` | 否 | 工厂生成 | — |
| `possible_scale_domains` | `list[Literal['PHQ-9', 'GAD-7', 'PCL-5', 'Y-BOCS', 'ISI', 'MDQ']]` | 否 | 工厂生成 | — |
| `structured_fields` | `list[StructuredFieldCandidate]` | 否 | 工厂生成 | — |
| `corrections` | `list[CorrectionCandidate]` | 否 | 工厂生成 | — |
| `clarification_needed` | `list[str]` | 否 | 工厂生成 | — |

#### `EvidenceNode` · 内部结构

源码：[`layers/layer2/schemas/models.py:139`](../layers/layer2/schemas/models.py#L139)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `evidence_id` | `str` | 是 | — | — |
| `source_type` | `Literal['patient_message', 'scale_item', 'benchmark_narrative']` | 是 | — | 枚举='patient_message', 'scale_item', 'benchmark_narrative' |
| `message_id` | `str` | 是 | — | — |
| `turn_index` | `int` | 是 | — | 最小=1 |
| `quote` | `str` | 是 | — | — |
| `target_path` | `str` | 是 | — | — |
| `created_by` | `Literal['llm_candidate_rule_verified', 'scale_rule', 'offline_source_adapter']` | 是 | — | 枚举='llm_candidate_rule_verified', 'scale_rule', 'offline_source_adapter' |
| `parent_ids` | `list[str]` | 否 | 工厂生成 | — |

#### `SymptomEvidence` · 内部结构

源码：[`layers/layer2/schemas/models.py:152`](../layers/layer2/schemas/models.py#L152)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `symptom` | `str` | 是 | — | — |
| `sources` | `list[str]` | 是 | — | 最少项=1 |
| `patient_quotes` | `list[str]` | 是 | — | 最少项=1 |
| `attributes` | `dict[str, Any]` | 否 | 工厂生成 | — |
| `attribute_facts` | `list[ClinicalFact]` | 否 | 工厂生成 | — |

#### `ProfileFact` · 内部结构

源码：[`layers/layer2/schemas/models.py:162`](../layers/layer2/schemas/models.py#L162)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `value` | `str` | 是 | — | — |
| `sources` | `list[str]` | 是 | — | 最少项=1 |
| `patient_quotes` | `list[str]` | 是 | — | 最少项=1 |

#### `MotivationScores` · 内部结构

源码：[`layers/layer2/schemas/models.py:170`](../layers/layer2/schemas/models.py#L170)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `importance` | `int | NoneType` | 否 | `None` | — |
| `confidence` | `int | NoneType` | 否 | `None` | — |
| `readiness` | `int | NoneType` | 否 | `None` | — |
| `sources` | `list[str]` | 否 | 工厂生成 | — |

#### `ScaleResult` · 内部结构

源码：[`layers/layer2/schemas/models.py:179`](../layers/layer2/schemas/models.py#L179)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `scale_name` | `Literal['PHQ-9', 'GAD-7', 'PCL-5', 'Y-BOCS', 'ISI', 'MDQ']` | 是 | — | 枚举='PHQ-9', 'GAD-7', 'PCL-5', 'Y-BOCS', 'ISI', 'MDQ' |
| `status` | `Literal['not_started', 'pending_consent', 'active', 'completed', 'declined', 'skipped']` | 否 | `'not_started'` | 枚举='not_started', 'pending_consent', 'active', 'completed', 'declined', 'skipped' |
| `items` | `dict[int, int]` | 否 | 工厂生成 | — |
| `score` | `int | NoneType` | 否 | `None` | — |
| `threshold` | `str | NoneType` | 否 | `None` | — |
| `evidence_ids` | `list[str]` | 否 | 工厂生成 | — |
| `skip_reason` | `str` | 否 | `''` | — |

#### `PendingScaleItem` · 内部结构

源码：[`layers/layer2/schemas/models.py:191`](../layers/layer2/schemas/models.py#L191)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `scale_name` | `Literal['PHQ-9', 'GAD-7', 'PCL-5', 'Y-BOCS', 'ISI', 'MDQ']` | 是 | — | 枚举='PHQ-9', 'GAD-7', 'PCL-5', 'Y-BOCS', 'ISI', 'MDQ' |
| `item_number` | `int` | 是 | — | 最小=1 |
| `prompt` | `str` | 是 | — | — |
| `min_score` | `int` | 是 | — | — |
| `max_score` | `int` | 是 | — | — |
| `anchors` | `dict[int, str]` | 是 | — | — |

#### `DiagnosticContext` · 内部结构

源码：[`layers/layer2/schemas/models.py:202`](../layers/layer2/schemas/models.py#L202)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `symptom_duration_days` | `int | NoneType` | 否 | `None` | — |
| `duration_evidence` | `list[DurationEvidence]` | 否 | 工厂生成 | — |
| `functional_impairment` | `bool | NoneType` | 否 | `None` | — |
| `mania_or_hypomania_history` | `bool | NoneType` | 否 | `None` | — |
| `substance_or_medical_cause` | `bool | NoneType` | 否 | `None` | — |
| `differential_diagnoses_assessed` | `list[str] | NoneType` | 否 | `None` | — |
| `evidence_conflicts` | `list[dict[str, Any]] | NoneType` | 否 | `None` | — |
| `relapse_count` | `int | NoneType` | 否 | `None` | — |
| `rhythm_disturbance` | `bool | NoneType` | 否 | `None` | — |
| `exposure_avoidant` | `bool | NoneType` | 否 | `None` | — |
| `ritual_behaviors` | `bool | NoneType` | 否 | `None` | — |
| `insomnia_chief_complaint` | `bool | NoneType` | 否 | `None` | — |
| `chronic_months` | `int | NoneType` | 否 | `None` | — |
| `interpersonal_conflicts` | `int | NoneType` | 否 | `None` | — |
| `interpersonal_impairment` | `bool | NoneType` | 否 | `None` | — |
| `single_focus` | `bool | NoneType` | 否 | `None` | — |
| `psychotic_symptoms` | `bool | NoneType` | 否 | `None` | — |
| `substance_active` | `bool | NoneType` | 否 | `None` | — |
| `severe_dissociation` | `bool | NoneType` | 否 | `None` | — |
| `diagnostic_details` | `dict[str, Any]` | 否 | 工厂生成 | — |
| `sources` | `dict[str, list[str]]` | 否 | 工厂生成 | — |

#### `L2SessionState` · 内部结构

源码：[`layers/layer2/schemas/models.py:228`](../layers/layer2/schemas/models.py#L228)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `session_id` | `str` | 是 | — | — |
| `contract_version` | `str` | 否 | `'l2-l3-clinical-v2'` | — |
| `interview_date` | `str | NoneType` | 否 | `None` | — |
| `clinical_facts` | `list[ClinicalFact]` | 否 | 工厂生成 | — |
| `follow_up_tasks` | `list[FollowUpTask]` | 否 | 工厂生成 | — |
| `current_stage` | `Literal['engaging', 'focusing', 'evoking', 'planning']` | 否 | `'engaging'` | 枚举='engaging', 'focusing', 'evoking', 'planning' |
| `turn_index` | `int` | 否 | `0` | — |
| `safety_passport` | `dict[str, Any]` | 是 | — | — |
| `active_safety_controls` | `list[str]` | 否 | 工厂生成 | — |
| `messages` | `list[dict[str, str]]` | 否 | 工厂生成 | — |
| `consumed_l1_message_ids` | `list[str]` | 否 | 工厂生成 | — |
| `symptoms` | `dict[str, SymptomEvidence]` | 否 | 工厂生成 | — |
| `emotions` | `list[ProfileFact]` | 否 | 工厂生成 | — |
| `cognitive_patterns` | `list[ProfileFact]` | 否 | 工厂生成 | — |
| `patient_goals` | `list[ProfileFact]` | 否 | 工厂生成 | — |
| `key_narratives` | `list[ProfileFact]` | 否 | 工厂生成 | — |
| `functional_impairments` | `list[ProfileFact]` | 否 | 工厂生成 | — |
| `focus_topics` | `list[ProfileFact]` | 否 | 工厂生成 | — |
| `change_talk` | `list[ProfileFact]` | 否 | 工厂生成 | — |
| `sustain_talk` | `list[ProfileFact]` | 否 | 工厂生成 | — |
| `motivation` | `MotivationScores` | 否 | 工厂生成 | — |
| `scale_results` | `dict[str, ScaleResult]` | 否 | 工厂生成 | — |
| `pending_scale_name` | `Literal['PHQ-9', 'GAD-7', 'PCL-5', 'Y-BOCS', 'ISI', 'MDQ'] | NoneType` | 否 | `None` | — |
| `evidence_nodes` | `list[EvidenceNode]` | 否 | 工厂生成 | — |
| `diagnostic_context` | `DiagnosticContext` | 否 | 工厂生成 | — |
| `confirmation_status` | `Literal['not_requested', 'pending', 'confirmed', 'revision_requested', 'paused']` | 否 | `'not_requested'` | 枚举='not_requested', 'pending', 'confirmed', 'revision_requested', 'paused' |
| `missing_dimensions` | `list[str]` | 否 | 工厂生成 | — |
| `flags` | `list[str]` | 否 | 工厂生成 | — |

#### `StageDecision` · 内部结构

源码：[`layers/layer2/schemas/models.py:261`](../layers/layer2/schemas/models.py#L261)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `stage_before` | `Literal['engaging', 'focusing', 'evoking', 'planning']` | 是 | — | 枚举='engaging', 'focusing', 'evoking', 'planning' |
| `stage_after` | `Literal['engaging', 'focusing', 'evoking', 'planning']` | 是 | — | 枚举='engaging', 'focusing', 'evoking', 'planning' |
| `transition_reason` | `str` | 是 | — | — |
| `response_goal` | `str` | 是 | — | — |
| `oars_techniques` | `list[Literal['open_question', 'affirmation', 'reflection', 'complex_reflection', 'summary']]` | 是 | — | — |
| `awaiting` | `Literal['auto', 'free_text', 'focus_selection', 'scale_consent', 'scale_answer', 'patient_goals', 'motivation', 'profile_confirmation', 'symptom_duration_days', 'functional_impairment', 'mania_or_hypomania_history', 'substance_or_medical_cause', 'psychotic_symptoms', 'differential_diagnoses_assessed', 'evidence_conflicts']` | 否 | `'auto'` | 枚举='auto', 'free_text', 'focus_selection', 'scale_consent', 'scale_answer', 'patient_goals', 'motivation', 'profile_confirmation', 'symptom_duration_days', 'functional_impairment', 'mania_or_hypomania_history', 'substance_or_medical_cause', 'psychotic_symptoms', 'differential_diagnoses_assessed', 'evidence_conflicts' |
| `required_question` | `str` | 否 | `''` | — |

#### `L2Turn` · 内部结构

源码：[`layers/layer2/schemas/models.py:273`](../layers/layer2/schemas/models.py#L273)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `stage_before` | `Literal['engaging', 'focusing', 'evoking', 'planning']` | 是 | — | 枚举='engaging', 'focusing', 'evoking', 'planning' |
| `stage_after` | `Literal['engaging', 'focusing', 'evoking', 'planning']` | 是 | — | 枚举='engaging', 'focusing', 'evoking', 'planning' |
| `input_type` | `Literal['free_narrative', 'focus_selection', 'scale_consent', 'scale_answer', 'goal_statement', 'motivation_rating', 'profile_confirmation', 'correction', 'refusal', 'pause', 'off_topic']` | 是 | — | 枚举='free_narrative', 'focus_selection', 'scale_consent', 'scale_answer', 'goal_statement', 'motivation_rating', 'profile_confirmation', 'correction', 'refusal', 'pause', 'off_topic' |
| `triggered_scales` | `list[Literal['PHQ-9', 'GAD-7', 'PCL-5', 'Y-BOCS', 'ISI', 'MDQ']]` | 否 | 工厂生成 | — |
| `pending_scale_item` | `PendingScaleItem | NoneType` | 否 | `None` | — |
| `evidence_ids` | `list[str]` | 否 | 工厂生成 | — |
| `transition_reason` | `str` | 是 | — | — |
| `response_goal` | `str` | 是 | — | — |

#### `DialogueResult` · 层间接口

源码：[`layers/layer2/schemas/models.py:286`](../layers/layer2/schemas/models.py#L286)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `accepted` | `bool` | 是 | — | — |
| `reply` | `str` | 是 | — | — |
| `awaiting` | `str | NoneType` | 否 | `None` | — |
| `l2_turn` | `L2Turn | NoneType` | 否 | `None` | — |
| `error_code` | `str | NoneType` | 否 | `None` | — |
| `confirmation_decision` | `Literal['continue', 'revise', 'pause'] | NoneType` | 否 | `None` | — |

#### `ClinicalIntake` · 层间接口

源码：[`layers/layer2/schemas/models.py:297`](../layers/layer2/schemas/models.py#L297)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `layer_id` | `Literal['L2']` | 否 | `'L2'` | — |
| `timestamp` | `str` | 否 | 工厂生成 | — |
| `session_id` | `str` | 是 | — | — |
| `contract_version` | `str` | 否 | `'l2-l3-clinical-v2'` | — |
| `interview_date` | `str | NoneType` | 否 | `None` | — |
| `clinical_facts` | `list[ClinicalFact]` | 否 | 工厂生成 | — |
| `follow_up_tasks` | `list[FollowUpTask]` | 否 | 工厂生成 | — |
| `current_stage` | `Literal['engaging', 'focusing', 'evoking', 'planning']` | 是 | — | 枚举='engaging', 'focusing', 'evoking', 'planning' |
| `key_symptoms_extracted` | `list[SymptomEvidence]` | 否 | 工厂生成 | — |
| `emotions` | `list[ProfileFact]` | 否 | 工厂生成 | — |
| `cognitive_patterns` | `list[ProfileFact]` | 否 | 工厂生成 | — |
| `patient_goals` | `list[str]` | 否 | 工厂生成 | — |
| `patient_goal_facts` | `list[ProfileFact]` | 否 | 工厂生成 | — |
| `key_narratives` | `list[ProfileFact]` | 否 | 工厂生成 | — |
| `functional_impairments` | `list[ProfileFact]` | 否 | 工厂生成 | — |
| `focus_topics` | `list[str]` | 否 | 工厂生成 | — |
| `focus_topic_facts` | `list[ProfileFact]` | 否 | 工厂生成 | — |
| `change_talk` | `list[ProfileFact]` | 否 | 工厂生成 | — |
| `sustain_talk` | `list[ProfileFact]` | 否 | 工厂生成 | — |
| `motivation` | `MotivationScores` | 是 | — | — |
| `scale_results` | `dict[str, ScaleResult]` | 否 | 工厂生成 | — |
| `diagnostic_context` | `DiagnosticContext` | 是 | — | — |
| `evidence_nodes` | `list[EvidenceNode]` | 否 | 工厂生成 | — |
| `information_sufficiency` | `Literal['insufficient', 'sufficient']` | 是 | — | 枚举='insufficient', 'sufficient' |
| `missing_dimensions` | `list[str]` | 否 | 工厂生成 | — |
| `patient_confirmation` | `Literal['not_requested', 'pending', 'confirmed', 'revision_requested', 'paused']` | 是 | — | 枚举='not_requested', 'pending', 'confirmed', 'revision_requested', 'paused' |
| `safety_controls` | `list[str]` | 否 | 工厂生成 | — |
| `flags` | `list[str]` | 否 | 工厂生成 | — |
| `l2_validation` | `dict[str, Any]` | 否 | 工厂生成 | — |

## L3：诊断、疗法选择与治疗指令

### `layers.layer3.diagnosis.schemas`

#### `MissingDimension` · 内部结构

源码：[`layers/layer3/diagnosis/schemas.py:14`](../layers/layer3/diagnosis/schemas.py#L14)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `field` | `str` | 是 | — | 最短长度=1 |
| `purpose` | `str` | 是 | — | 最短长度=1 |
| `suggested_question` | `str` | 是 | — | 最短长度=1 |

#### `TreeNodeJudgment` · 内部结构

源码：[`layers/layer3/diagnosis/schemas.py:22`](../layers/layer3/diagnosis/schemas.py#L22)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `node_id` | `str` | 是 | — | 最短长度=1 |
| `status` | `Literal['met', 'not_met', 'pending']` | 是 | — | 枚举='met', 'not_met', 'pending' |
| `rationale` | `str` | 是 | — | 最短长度=1 |
| `evidence_refs` | `list[str]` | 否 | 工厂生成 | — |
| `temporal_decision` | `TemporalDecision | NoneType` | 否 | `None` | — |
| `symptom_ids` | `list[str]` | 否 | 工厂生成 | — |
| `confidence` | `float` | 否 | `0.5` | 最小=0；最大=1 |
| `denial_scope` | `Literal['full', 'partial', 'none', 'unknown']` | 否 | `'unknown'` | 枚举='full', 'partial', 'none', 'unknown' |

#### `TreeDecisionRecord` · 内部结构

源码：[`layers/layer3/diagnosis/schemas.py:47`](../layers/layer3/diagnosis/schemas.py#L47)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `node_id` | `str` | 是 | — | 最短长度=1 |
| `status` | `Literal['met', 'not_met', 'pending']` | 是 | — | 枚举='met', 'not_met', 'pending' |
| `rationale` | `str` | 是 | — | 最短长度=1 |
| `evidence_refs` | `list[str]` | 否 | 工厂生成 | — |
| `temporal_decision` | `TemporalDecision | NoneType` | 否 | `None` | — |
| `symptom_ids` | `list[str]` | 否 | 工厂生成 | — |
| `confidence` | `float` | 否 | `0.5` | 最小=0；最大=1 |
| `denial_scope` | `Literal['full', 'partial', 'none', 'unknown']` | 否 | `'unknown'` | 枚举='full', 'partial', 'none', 'unknown' |
| `question` | `str` | 是 | — | — |
| `source_refs` | `list[str]` | 是 | — | — |
| `stage` | `Literal['screening', 'anchor', 'exclusion', 'criterion']` | 是 | — | 枚举='screening', 'anchor', 'exclusion', 'criterion' |

#### `CandidateDecisionPath` · 内部结构

源码：[`layers/layer3/diagnosis/schemas.py:53`](../layers/layer3/diagnosis/schemas.py#L53)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `path_id` | `str` | 是 | — | — |
| `tree_refs` | `list[str]` | 是 | — | — |
| `provisional` | `bool` | 是 | — | — |
| `decisions` | `list[TreeDecisionRecord]` | 是 | — | — |

#### `SymptomMatch` · 内部结构

源码：[`layers/layer3/diagnosis/schemas.py:62`](../layers/layer3/diagnosis/schemas.py#L62)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `matched` | `int` | 是 | — | 最小=0 |
| `total` | `int` | 是 | — | 最小=1 |

#### `DiagnosticCandidate` · 内部结构

源码：[`layers/layer3/diagnosis/schemas.py:75`](../layers/layer3/diagnosis/schemas.py#L75)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `disorder` | `Literal['D001', 'D002', 'D003', 'D004', 'D005', 'D006', 'D007', 'D008', 'D009', 'D010', 'D011', 'D012', 'D013', 'D014', 'D015', 'D016', 'D017', 'D018', 'D019', 'D020', 'D021', 'D022', 'D023', 'Insomnia']` | 是 | — | 枚举='D001', 'D002', 'D003', 'D004', 'D005', 'D006', 'D007', 'D008', 'D009', 'D010', 'D011', 'D012', 'D013', 'D014', 'D015', 'D016', 'D017', 'D018', 'D019', 'D020', 'D021', 'D022', 'D023', 'Insomnia' |
| `label` | `str` | 是 | — | 最短长度=1 |
| `symptom_match` | `SymptomMatch` | 是 | — | — |
| `core_evidence_summary` | `str` | 是 | — | 最短长度=1 |
| `evidence_refs` | `list[str]` | 是 | — | 最少项=1 |
| `confidence` | `float` | 是 | — | 最小=0.0；最大=1.0 |
| `decision_paths` | `list[CandidateDecisionPath]` | 否 | 工厂生成 | — |

#### `CandidateGeneration` · 内部结构

源码：[`layers/layer3/diagnosis/schemas.py:87`](../layers/layer3/diagnosis/schemas.py#L87)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `considered_disorders` | `list[Literal['D001', 'D002', 'D003', 'D004', 'D005', 'D006', 'D007', 'D008', 'D009', 'D010', 'D011', 'D012', 'D013', 'D014', 'D015', 'D016', 'D017', 'D018', 'D019', 'D020', 'D021', 'D022', 'D023', 'Insomnia']]` | 是 | — | 最少项=24；最多项=24 |
| `candidates` | `list[DiagnosticCandidate]` | 否 | 工厂生成 | 最多项=24 |
| `tree_trace` | `list[TreeDecisionRecord]` | 否 | 工厂生成 | — |
| `missing_dimensions` | `list[MissingDimension]` | 否 | 工厂生成 | — |
| `candidate_coverage` | `dict[str, str]` | 否 | 工厂生成 | — |

#### `CriterionAssessment` · 内部结构

源码：[`layers/layer3/diagnosis/schemas.py:97`](../layers/layer3/diagnosis/schemas.py#L97)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `code` | `str` | 是 | — | 最短长度=1 |
| `name` | `str` | 是 | — | 最短长度=1 |
| `status` | `Literal['met', 'not_met', 'pending']` | 是 | — | 枚举='met', 'not_met', 'pending' |
| `rationale` | `str` | 是 | — | 最短长度=1 |
| `evidence_refs` | `list[str]` | 否 | 工厂生成 | — |

#### `EliminativeAssessment` · 内部结构

源码：[`layers/layer3/diagnosis/schemas.py:107`](../layers/layer3/diagnosis/schemas.py#L107)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `disorder` | `Literal['D001', 'D002', 'D003', 'D004', 'D005', 'D006', 'D007', 'D008', 'D009', 'D010', 'D011', 'D012', 'D013', 'D014', 'D015', 'D016', 'D017', 'D018', 'D019', 'D020', 'D021', 'D022', 'D023', 'Insomnia']` | 是 | — | 枚举='D001', 'D002', 'D003', 'D004', 'D005', 'D006', 'D007', 'D008', 'D009', 'D010', 'D011', 'D012', 'D013', 'D014', 'D015', 'D016', 'D017', 'D018', 'D019', 'D020', 'D021', 'D022', 'D023', 'Insomnia' |
| `criteria_detail` | `list[CriterionAssessment]` | 是 | — | 最少项=1 |
| `conclusion` | `Literal['meets_criteria', 'watch', 'ruled_out']` | 是 | — | 枚举='meets_criteria', 'watch', 'ruled_out' |
| `conclusion_reason` | `str` | 是 | — | 最短长度=1 |
| `missing_dimensions` | `list[MissingDimension]` | 否 | 工厂生成 | — |

#### `VerifierDimension` · 内部结构

源码：[`layers/layer3/diagnosis/schemas.py:117`](../layers/layer3/diagnosis/schemas.py#L117)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `dimension` | `Literal['evidence_sufficiency', 'logic_coherence', 'differential_coverage', 'no_reasoning_leaps']` | 是 | — | 枚举='evidence_sufficiency', 'logic_coherence', 'differential_coverage', 'no_reasoning_leaps' |
| `score` | `float` | 是 | — | 最小=0.0；最大=1.0 |
| `reason` | `str` | 是 | — | 最短长度=1 |
| `evidence_refs` | `list[str]` | 否 | 工厂生成 | — |

#### `RewardModelAssessment` · 内部结构

源码：[`layers/layer3/diagnosis/schemas.py:131`](../layers/layer3/diagnosis/schemas.py#L131)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `score` | `float` | 是 | — | 最小=0.0；最大=1.0 |
| `dimensions` | `list[VerifierDimension]` | 是 | — | 最少项=4；最多项=4 |
| `fail_reasons` | `list[str]` | 否 | 工厂生成 | — |
| `missing_dimensions` | `list[MissingDimension]` | 否 | 工厂生成 | — |

### `layers.layer3.api`

#### `MotivationInput` · 内部结构

源码：[`layers/layer3/api.py:26`](../layers/layer3/api.py#L26)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `importance` | `int | NoneType` | 否 | `None` | — |
| `confidence` | `int | NoneType` | 否 | `None` | — |
| `readiness` | `int | NoneType` | 否 | `None` | — |
| `sources` | `list[str]` | 否 | 工厂生成 | — |

#### `DiagnoseRequest` · 层间接口

源码：[`layers/layer3/api.py:33`](../layers/layer3/api.py#L33)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `contract_version` | `Literal['l2-l3-clinical-v2']` | 否 | `'l2-l3-clinical-v2'` | — |
| `interview_date` | `date | NoneType` | 否 | `None` | — |
| `clinical_facts` | `list[ClinicalFact]` | 否 | 工厂生成 | — |
| `duration_evidence` | `list[DurationEvidence]` | 否 | 工厂生成 | — |
| `follow_up_tasks` | `list[FollowUpTask]` | 否 | 工厂生成 | — |
| `temporal_audit` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `PHQ9` | `float | NoneType` | 否 | `None` | — |
| `GAD7` | `float | NoneType` | 否 | `None` | — |
| `MDQ` | `float | NoneType` | 否 | `None` | — |
| `PCL5` | `float | NoneType` | 否 | `None` | — |
| `ISI` | `float | NoneType` | 否 | `None` | — |
| `Y_BOCS` | `float | NoneType` | 否 | `None` | — |
| `ADNM` | `float | NoneType` | 否 | `None` | — |
| `RRS` | `float | NoneType` | 否 | `None` | — |
| `AAQ_II` | `float | NoneType` | 否 | `None` | — |
| `PHQ15` | `float | NoneType` | 否 | `None` | — |
| `PHQ9_items` | `dict[str, float]` | 否 | 工厂生成 | — |
| `GAD7_items` | `dict[str, float]` | 否 | 工厂生成 | — |
| `MDQ_items` | `dict[str, float]` | 否 | 工厂生成 | — |
| `PCL5_items` | `dict[str, float]` | 否 | 工厂生成 | — |
| `ISI_items` | `dict[str, float]` | 否 | 工厂生成 | — |
| `Y_BOCS_items` | `dict[str, float]` | 否 | 工厂生成 | — |
| `ADNM_items` | `dict[str, float]` | 否 | 工厂生成 | — |
| `RRS_items` | `dict[str, float]` | 否 | 工厂生成 | — |
| `AAQ_II_items` | `dict[str, float]` | 否 | 工厂生成 | — |
| `PHQ15_items` | `dict[str, float]` | 否 | 工厂生成 | — |
| `relapse_count` | `Annotated | NoneType` | 否 | `None` | — |
| `chronic_months` | `Annotated | NoneType` | 否 | `None` | — |
| `interpersonal_conflicts` | `Annotated | NoneType` | 否 | `None` | — |
| `symptom_duration_days` | `Annotated | NoneType` | 否 | `None` | — |
| `functional_impairment` | `Annotated | NoneType` | 否 | `None` | — |
| `mania_or_hypomania_history` | `Annotated | NoneType` | 否 | `None` | — |
| `substance_or_medical_cause` | `Annotated | NoneType` | 否 | `None` | — |
| `rhythm_disturbance` | `Annotated | NoneType` | 否 | `None` | — |
| `exposure_avoidant` | `Annotated | NoneType` | 否 | `None` | — |
| `ritual_behaviors` | `Annotated | NoneType` | 否 | `None` | — |
| `insomnia_chief_complaint` | `Annotated | NoneType` | 否 | `None` | — |
| `interpersonal_impairment` | `Annotated | NoneType` | 否 | `None` | — |
| `single_focus` | `Annotated | NoneType` | 否 | `None` | — |
| `psychotic_symptoms` | `Annotated | NoneType` | 否 | `None` | — |
| `substance_active` | `Annotated | NoneType` | 否 | `None` | — |
| `severe_dissociation` | `Annotated | NoneType` | 否 | `None` | — |
| `differential_diagnoses_assessed` | `list[str] | NoneType` | 否 | `None` | — |
| `evidence_conflicts` | `list[dict[str, Any]] | NoneType` | 否 | `None` | — |
| `diagnostic_details` | `dict[str, Any]` | 否 | 工厂生成 | — |
| `key_symptoms_extracted` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `emotions` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `cognitive_patterns` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `key_narratives` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `functional_impairments` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `patient_goals` | `list[str]` | 否 | 工厂生成 | — |
| `motivation` | `MotivationInput` | 否 | 工厂生成 | — |
| `evidence_nodes` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `risk_level` | `int` | 否 | `0` | — |
| `sensitive_topics` | `list[str]` | 否 | 有默认值 | — |
| `l1_flags` | `list[str]` | 否 | 有默认值 | — |

#### `DiagnoseResponse` · 层间接口

源码：[`layers/layer3/api.py:40`](../layers/layer3/api.py#L40)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `layer_id` | `Literal['L3']` | 否 | `'L3'` | — |
| `timestamp` | `str` | 是 | — | — |
| `diagnostic_standard` | `Literal['DSM-5']` | 否 | `'DSM-5'` | — |
| `supported_disorders` | `list[str]` | 否 | 工厂生成 | — |
| `diagnostic_catalog` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `considered_disorders` | `list[str]` | 否 | 工厂生成 | — |
| `candidates` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `checked` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `candidate_generation_method` | `str` | 否 | `'external_reasoner'` | — |
| `tree_trace` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `candidate_coverage` | `dict[str, str]` | 否 | 工厂生成 | — |
| `candidate_missing_dimensions` | `list[dict[str, str]]` | 否 | 工厂生成 | — |
| `diagnostic_readiness` | `dict[str, Any]` | 否 | 工厂生成 | — |
| `information_collection_plan` | `dict[str, Any]` | 否 | 工厂生成 | — |
| `duration_evidence` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `temporal_audit` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `diagnosis` | `dict` | 是 | — | — |
| `verifier_score` | `float` | 是 | — | — |
| `verifier_decision` | `str` | 是 | — | — |
| `candidate_confidence` | `float` | 是 | — | — |
| `verifier_routing` | `str` | 是 | — | — |
| `recommended_therapy` | `dict` | 是 | — | — |
| `severity_level` | `int` | 是 | — | — |
| `clinical_notes_for_L4` | `str` | 是 | — | — |
| `summary` | `str` | 是 | — | — |
| `structured_labels` | `dict` | 是 | — | — |
| `evidence_refs` | `list` | 是 | — | — |
| `confidence` | `float` | 是 | — | — |
| `flags` | `list` | 是 | — | — |
| `instruction_to_next` | `str` | 是 | — | — |
| `instructions` | `dict` | 是 | — | — |
| `missing_dimensions` | `list[dict[str, str]]` | 否 | 工厂生成 | — |
| `provisional_diagnosis` | `dict | NoneType` | 否 | `None` | — |
| `verifier_checks` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `verifier_fail_reasons` | `list[str]` | 否 | 工厂生成 | — |
| `l3_validation` | `dict[str, Any]` | 是 | — | — |

### `layers.layer3.instructions.turn_planner`

#### `TurnProposal` · 内部结构

源码：[`layers/layer3/instructions/turn_planner.py:54`](../layers/layer3/instructions/turn_planner.py#L54)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `goal` | `str` | 是 | — | 最短长度=1 |
| `technique_code` | `str` | 是 | — | 最短长度=1 |
| `stage` | `Literal['L4-L1', 'L4-L2', 'L4-L3', 'L4-L4']` | 是 | — | 枚举='L4-L1', 'L4-L2', 'L4-L3', 'L4-L4' |
| `link_previous` | `str` | 否 | `''` | — |
| `plugin_steps` | `dict[str, str]` | 否 | 工厂生成 | — |
| `forbidden` | `list[str]` | 否 | 工厂生成 | — |
| `rationale` | `str` | 是 | — | 最短长度=1 |
| `evidence` | `list[SourceQuote]` | 是 | — | 最少项=1 |
| `observations` | `list[ProgressObservation]` | 否 | 工厂生成 | — |
| `previous_goal_met` | `bool` | 否 | `False` | — |
| `patient_ready` | `bool` | 否 | `False` | — |

### `shared.schemas.treatment_instructions`

#### `InstructionModel` · 内部结构

源码：[`shared/schemas/treatment_instructions.py:6`](../shared/schemas/treatment_instructions.py#L6)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|

#### `RoleInstruction` · 内部结构

源码：[`shared/schemas/treatment_instructions.py:12`](../shared/schemas/treatment_instructions.py#L12)

L1 角色定义（全会话复用）。

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `identity` | `str` | 是 | — | — |
| `style` | `str` | 是 | — | — |
| `boundaries` | `list[str]` | 否 | 工厂生成 | — |

#### `DiagnosticAnchor` · 内部结构

源码：[`shared/schemas/treatment_instructions.py:20`](../shared/schemas/treatment_instructions.py#L20)

L3 生成的会话级锚点；L4 只消费，不重新推导。

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `primary` | `str` | 否 | `''` | — |
| `comorbidities` | `list[str]` | 否 | 有默认值 | — |
| `excluded` | `list[str]` | 否 | 有默认值 | — |
| `core_beliefs` | `list[str]` | 否 | 有默认值 | — |
| `plugins` | `list[str]` | 否 | 有默认值 | — |
| `plugin_triggers` | `list[dict[str, Any]]` | 否 | 有默认值 | — |
| `strategy_adjustments` | `dict[str, Any]` | 否 | 有默认值 | — |
| `contraindications` | `list[str]` | 否 | 有默认值 | — |
| `confidence` | `float` | 否 | `0.0` | — |
| `candidate_confidence` | `float` | 否 | `0.0` | — |
| `verifier_score` | `float` | 否 | `0.0` | — |
| `therapy_type` | `str` | 否 | `''` | — |
| `verifier_decision` | `str` | 否 | `''` | — |
| `diagnosis_details` | `dict[str, Any]` | 否 | 工厂生成 | — |
| `scale_scores` | `dict[str, float]` | 否 | 工厂生成 | — |
| `cognitive_patterns` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `triggering_situations` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `key_narratives` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `patient_goals` | `list[str]` | 否 | 工厂生成 | — |
| `allowed_techniques` | `dict[str, str]` | 否 | 工厂生成 | — |
| `wording_constraint` | `str` | 否 | `'使用初步印象措辞，禁止确诊'` | — |

#### `SourceQuote` · 内部结构

源码：[`shared/schemas/treatment_instructions.py:46`](../shared/schemas/treatment_instructions.py#L46)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `source_id` | `str` | 是 | — | 最短长度=1 |
| `quote` | `str` | 是 | — | 最短长度=1 |

#### `ProgressObservation` · 内部结构

源码：[`shared/schemas/treatment_instructions.py:51`](../shared/schemas/treatment_instructions.py#L51)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `kind` | `Literal['patient_response', 'homework', 'unresolved']` | 是 | — | 枚举='patient_response', 'homework', 'unresolved' |
| `interpretation` | `str` | 是 | — | 最短长度=1 |
| `evidence` | `SourceQuote` | 是 | — | — |

#### `TurnInstruction` · 内部结构

源码：[`shared/schemas/treatment_instructions.py:57`](../shared/schemas/treatment_instructions.py#L57)

L3 本轮动态指令（每轮重新生成）。

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `goal` | `str` | 是 | — | — |
| `technique` | `str` | 是 | — | — |
| `forbidden` | `list[str]` | 否 | 有默认值 | — |
| `link_previous` | `str` | 否 | `''` | — |
| `plugin_guidance` | `str | NoneType` | 否 | `None` | — |
| `force_substage` | `Literal['L4-L1', 'L4-L2', 'L4-L3', 'L4-L4'] | NoneType` | 否 | `None` | — |
| `session_id` | `str` | 否 | `''` | — |
| `turn_number` | `int` | 否 | `0` | 最小=0 |
| `source_message` | `str` | 否 | `''` | — |
| `technique_code` | `str` | 否 | `''` | — |
| `plugin_steps` | `dict[str, str]` | 否 | 工厂生成 | — |
| `evidence` | `list[SourceQuote]` | 否 | 工厂生成 | — |
| `observations` | `list[ProgressObservation]` | 否 | 工厂生成 | — |
| `previous_goal_met` | `bool` | 否 | `False` | — |
| `patient_ready` | `bool` | 否 | `False` | — |
| `rationale` | `str` | 否 | `''` | — |

#### `BoundaryInstruction` · 内部结构

源码：[`shared/schemas/treatment_instructions.py:78`](../shared/schemas/treatment_instructions.py#L78)

L4 防越界指令（每轮复用）。

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `absolute_bans` | `list[str]` | 否 | 有默认值 | — |
| `safety_trigger` | `str` | 否 | `''` | — |

#### `TreatmentInstructions` · 层间接口

源码：[`shared/schemas/treatment_instructions.py:85`](../shared/schemas/treatment_instructions.py#L85)

会话建立时 L3 为空；每次生成回复前必须有绑定本轮的 L3。

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `L1` | `RoleInstruction` | 是 | — | — |
| `L2` | `DiagnosticAnchor` | 是 | — | — |
| `L3` | `TurnInstruction | NoneType` | 否 | `None` | — |
| `L4` | `BoundaryInstruction` | 是 | — | — |

#### `TreatmentTurnRecord` · 内部结构

源码：[`shared/schemas/treatment_instructions.py:94`](../shared/schemas/treatment_instructions.py#L94)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `instruction` | `TurnInstruction` | 是 | — | — |
| `user_message` | `str` | 是 | — | — |
| `reply` | `str` | 是 | — | — |
| `fallback_used` | `bool` | 否 | `False` | — |

#### `SafetyPassport` · 层间接口

源码：[`shared/schemas/treatment_instructions.py:101`](../shared/schemas/treatment_instructions.py#L101)

L1 安全通行证中 L4 需要的部分（每轮复用）。

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `risk_level` | `int` | 否 | `1` | — |
| `sensitive_topics` | `list[str]` | 否 | 有默认值 | — |
| `flags` | `list[str]` | 否 | 有默认值 | — |

## L4：治疗对话

### `layers.layer4.schemas.l3_report`

#### `DSMCriterionEvidence` · 内部结构

源码：[`layers/layer4/schemas/l3_report.py:9`](../layers/layer4/schemas/l3_report.py#L9)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `code` | `str` | 是 | — | — |
| `name` | `str` | 是 | — | — |
| `status` | `Literal['met', 'not_met', 'unknown']` | 是 | — | 枚举='met', 'not_met', 'unknown' |
| `met` | `bool` | 是 | — | — |
| `evidence_refs` | `list[str]` | 否 | 工厂生成 | — |

#### `DiagnosticDisplay` · 内部结构

源码：[`layers/layer4/schemas/l3_report.py:17`](../layers/layer4/schemas/l3_report.py#L17)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `name` | `str` | 否 | `''` | — |
| `label` | `str` | 否 | `''` | — |
| `display_group` | `str` | 否 | `''` | — |
| `display_group_label` | `str` | 否 | `''` | — |
| `display_name` | `str` | 否 | `''` | — |
| `knowledge_source` | `str` | 否 | `''` | — |

#### `PrimaryDiagnosis` · 内部结构

源码：[`layers/layer4/schemas/l3_report.py:26`](../layers/layer4/schemas/l3_report.py#L26)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `name` | `str` | 否 | `''` | — |
| `label` | `str` | 否 | `''` | — |
| `display_group` | `str` | 否 | `''` | — |
| `display_group_label` | `str` | 否 | `''` | — |
| `display_name` | `str` | 否 | `''` | — |
| `knowledge_source` | `str` | 否 | `''` | — |
| `disorder` | `str` | 是 | — | — |
| `confidence` | `float` | 否 | `0.0` | — |
| `severity` | `str` | 否 | `''` | — |
| `evidence` | `str` | 否 | `''` | — |
| `dsm5_criteria_met` | `list[str]` | 否 | 有默认值 | — |
| `dsm5_criteria_detail` | `list[DSMCriterionEvidence]` | 否 | 工厂生成 | — |

#### `SecondaryDiagnosis` · 内部结构

源码：[`layers/layer4/schemas/l3_report.py:35`](../layers/layer4/schemas/l3_report.py#L35)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `name` | `str` | 否 | `''` | — |
| `label` | `str` | 否 | `''` | — |
| `display_group` | `str` | 否 | `''` | — |
| `display_group_label` | `str` | 否 | `''` | — |
| `display_name` | `str` | 否 | `''` | — |
| `knowledge_source` | `str` | 否 | `''` | — |
| `disorder` | `str` | 是 | — | — |
| `confidence` | `float` | 否 | `0.0` | — |
| `evidence` | `str` | 否 | `''` | — |

#### `RuledOutDiagnosis` · 内部结构

源码：[`layers/layer4/schemas/l3_report.py:41`](../layers/layer4/schemas/l3_report.py#L41)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `name` | `str` | 否 | `''` | — |
| `label` | `str` | 否 | `''` | — |
| `display_group` | `str` | 否 | `''` | — |
| `display_group_label` | `str` | 否 | `''` | — |
| `display_name` | `str` | 否 | `''` | — |
| `knowledge_source` | `str` | 否 | `''` | — |
| `disorder` | `str` | 是 | — | — |
| `reason` | `str` | 否 | `''` | — |
| `evidence_ref` | `str` | 否 | `''` | — |

#### `UnderObservationDiagnosis` · 内部结构

源码：[`layers/layer4/schemas/l3_report.py:47`](../layers/layer4/schemas/l3_report.py#L47)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `name` | `str` | 否 | `''` | — |
| `label` | `str` | 否 | `''` | — |
| `display_group` | `str` | 否 | `''` | — |
| `display_group_label` | `str` | 否 | `''` | — |
| `display_name` | `str` | 否 | `''` | — |
| `knowledge_source` | `str` | 否 | `''` | — |
| `disorder` | `str` | 是 | — | — |
| `confidence` | `float` | 否 | `0.0` | — |
| `reason` | `str` | 否 | `''` | — |
| `missing_dimensions` | `list[dict]` | 否 | 工厂生成 | — |

#### `Diagnosis` · 内部结构

源码：[`layers/layer4/schemas/l3_report.py:54`](../layers/layer4/schemas/l3_report.py#L54)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `primary` | `PrimaryDiagnosis | NoneType` | 否 | `None` | — |
| `secondary` | `list[SecondaryDiagnosis]` | 否 | 有默认值 | — |
| `under_observation` | `list[UnderObservationDiagnosis]` | 否 | 工厂生成 | — |
| `ruled_out` | `list[RuledOutDiagnosis]` | 否 | 有默认值 | — |

#### `RecommendedTherapy` · 内部结构

源码：[`layers/layer4/schemas/l3_report.py:61`](../layers/layer4/schemas/l3_report.py#L61)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `primary` | `str` | 否 | `''` | — |
| `addon` | `list[str]` | 否 | 有默认值 | — |
| `plugin_triggers` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `strategy_adjustments` | `dict[str, Any]` | 否 | 工厂生成 | — |
| `contraindications` | `list[str]` | 否 | 有默认值 | — |

#### `TherapyRouting` · 内部结构

源码：[`layers/layer4/schemas/l3_report.py:69`](../layers/layer4/schemas/l3_report.py#L69)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `primary` | `str` | 否 | `''` | — |
| `addon` | `list[str]` | 否 | 有默认值 | — |
| `plugin_triggers` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `strategy_adjustments` | `dict[str, Any]` | 否 | 工厂生成 | — |
| `l4_substage_priority` | `list[str]` | 否 | 有默认值 | — |

#### `StructuredLabels` · 内部结构

源码：[`layers/layer4/schemas/l3_report.py:77`](../layers/layer4/schemas/l3_report.py#L77)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `therapy_routing` | `TherapyRouting` | 否 | 有默认值 | — |
| `wording_constraint` | `str` | 否 | `''` | — |
| `risk_monitoring` | `list[str]` | 否 | 有默认值 | — |

#### `EvidenceRefIn` · 内部结构

源码：[`layers/layer4/schemas/l3_report.py:83`](../layers/layer4/schemas/l3_report.py#L83)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `type` | `str` | 否 | `''` | — |
| `disorder` | `str` | 否 | `''` | — |
| `criteria_met` | `str` | 否 | `''` | — |
| `source_doc` | `str` | 否 | `''` | — |
| `threshold` | `str` | 否 | `''` | — |
| `therapy` | `str` | 否 | `''` | — |
| `source` | `str` | 否 | `''` | — |
| `criterion_code` | `str` | 否 | `''` | — |
| `criterion_name` | `str` | 否 | `''` | — |
| `status` | `str` | 否 | `''` | — |
| `evidence_refs` | `list[str]` | 否 | 工厂生成 | — |

#### `MissingDimension` · 内部结构

源码：[`layers/layer4/schemas/l3_report.py:97`](../layers/layer4/schemas/l3_report.py#L97)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `field` | `str` | 是 | — | — |
| `purpose` | `str` | 否 | `''` | — |
| `suggested_question` | `str` | 否 | `''` | — |

#### `VerifierCheck` · 内部结构

源码：[`layers/layer4/schemas/l3_report.py:103`](../layers/layer4/schemas/l3_report.py#L103)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `check_id` | `str` | 是 | — | — |
| `status` | `Literal['pass', 'fail']` | 是 | — | 枚举='pass', 'fail' |
| `score` | `float` | 是 | — | — |
| `weight` | `float` | 是 | — | — |
| `reasons` | `list[str]` | 否 | 工厂生成 | — |
| `evidence_refs` | `list[str]` | 否 | 工厂生成 | — |

#### `L3ValidationCheck` · 内部结构

源码：[`layers/layer4/schemas/l3_report.py:112`](../layers/layer4/schemas/l3_report.py#L112)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `rule_id` | `str` | 是 | — | — |
| `status` | `Literal['pass', 'fail']` | 是 | — | 枚举='pass', 'fail' |
| `detail` | `str` | 否 | `''` | — |

#### `L3ValidationReceipt` · 内部结构

源码：[`layers/layer4/schemas/l3_report.py:118`](../layers/layer4/schemas/l3_report.py#L118)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `passed` | `bool` | 是 | — | — |
| `route` | `str` | 是 | — | — |
| `checks` | `list[L3ValidationCheck]` | 否 | 工厂生成 | — |
| `payload_hash` | `str` | 是 | — | — |
| `policy_version` | `str` | 是 | — | — |

#### `DiagnosticReport` · 层间接口

源码：[`layers/layer4/schemas/l3_report.py:126`](../layers/layer4/schemas/l3_report.py#L126)

接口三：L3 → L4 DiagnosticReport（诊断报告）。未知字段忽略，真实 L3 完整 JSON 可解析。

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `diagnostic_standard` | `Literal['DSM-5']` | 否 | `'DSM-5'` | — |
| `supported_disorders` | `list[str]` | 否 | 工厂生成 | — |
| `diagnostic_catalog` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `considered_disorders` | `list[str]` | 否 | 工厂生成 | — |
| `candidates` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `checked` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `candidate_generation_method` | `str` | 否 | `'external_reasoner'` | — |
| `tree_trace` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `candidate_coverage` | `dict[str, str]` | 否 | 工厂生成 | — |
| `candidate_missing_dimensions` | `list[dict[str, str]]` | 否 | 工厂生成 | — |
| `diagnostic_readiness` | `dict[str, Any]` | 否 | 工厂生成 | — |
| `information_collection_plan` | `dict[str, Any]` | 否 | 工厂生成 | — |
| `duration_evidence` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `temporal_audit` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `layer_id` | `str` | 否 | `'L3'` | — |
| `timestamp` | `str` | 否 | `''` | — |
| `diagnosis` | `Diagnosis` | 否 | 有默认值 | — |
| `provisional_diagnosis` | `PrimaryDiagnosis | NoneType` | 否 | `None` | — |
| `verifier_score` | `float` | 否 | `0.0` | — |
| `verifier_decision` | `str` | 否 | `''` | — |
| `candidate_confidence` | `float` | 否 | `0.0` | — |
| `verifier_routing` | `str` | 否 | `''` | — |
| `verifier_checks` | `list[VerifierCheck]` | 否 | 工厂生成 | — |
| `verifier_fail_reasons` | `list[str]` | 否 | 工厂生成 | — |
| `recommended_therapy` | `RecommendedTherapy` | 否 | 有默认值 | — |
| `severity_level` | `int` | 否 | `1` | — |
| `clinical_notes_for_L4` | `str` | 否 | `''` | — |
| `summary` | `str` | 否 | `''` | — |
| `structured_labels` | `StructuredLabels` | 否 | 有默认值 | — |
| `evidence_refs` | `list[EvidenceRefIn]` | 否 | 有默认值 | — |
| `confidence` | `float` | 否 | `0.0` | — |
| `flags` | `list[str]` | 否 | 有默认值 | — |
| `instruction_to_next` | `str` | 否 | `''` | — |
| `missing_dimensions` | `list[MissingDimension]` | 否 | 有默认值 | — |
| `l3_validation` | `L3ValidationReceipt | NoneType` | 否 | `None` | — |
| `instructions` | `dict[str, Any]` | 否 | 工厂生成 | — |

### `layers.layer4.schemas.summary`

#### `EvidenceRef` · 内部结构

源码：[`layers/layer4/schemas/summary.py:6`](../layers/layer4/schemas/summary.py#L6)

循证依据引用：来源、发现、依据。

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `source` | `str` | 是 | — | — |
| `finding` | `str` | 是 | — | — |
| `basis` | `str` | 是 | — | — |

#### `StructuredSummary` · 内部结构

源码：[`layers/layer4/schemas/summary.py:14`](../layers/layer4/schemas/summary.py#L14)

每层输出附带的结构化临床摘要（通用格式，L4 填充）。

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `layer_id` | `Literal['L4']` | 否 | `'L4'` | — |
| `timestamp` | `str` | 是 | — | — |
| `summary` | `str` | 是 | — | — |
| `structured_labels` | `dict[str, Any]` | 是 | — | — |
| `evidence_refs` | `list[EvidenceRef]` | 否 | 有默认值 | — |
| `confidence` | `float` | 否 | `0.0` | — |
| `flags` | `list[str]` | 否 | 有默认值 | — |

### `layers.layer4.service`

#### `SessionRequest` · 层间接口

源码：[`layers/layer4/service.py:40`](../layers/layer4/service.py#L40)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `session_id` | `str | NoneType` | 否 | `None` | — |
| `report` | `DiagnosticReport` | 是 | — | — |
| `safety` | `SafetyPassport` | 否 | 有默认值 | — |

#### `TurnRequest` · 层间接口

源码：[`layers/layer4/service.py:47`](../layers/layer4/service.py#L47)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `session_id` | `str` | 是 | — | — |
| `user_message` | `str` | 是 | — | — |
| `turn` | `TurnInstruction` | 是 | — | — |

#### `TurnOutcome` · 层间接口

源码：[`layers/layer4/service.py:53`](../layers/layer4/service.py#L53)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `session_id` | `str` | 是 | — | — |
| `reply` | `str` | 是 | — | — |
| `l4_substage` | `Literal['L4-L1', 'L4-L2', 'L4-L3', 'L4-L4']` | 是 | — | 枚举='L4-L1', 'L4-L2', 'L4-L3', 'L4-L4' |
| `l5_context` | `L5Context` | 是 | — | — |
| `summary` | `StructuredSummary` | 是 | — | — |
| `review_history` | `list[ReviewResult]` | 是 | — | — |
| `fallback_used` | `bool` | 否 | `False` | — |

### `layers.layer4.api.server`

#### `SessionCreated` · 层间接口

源码：[`layers/layer4/api/server.py:9`](../layers/layer4/api/server.py#L9)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `session_id` | `str` | 是 | — | — |

#### `RewriteRequest` · 层间接口

源码：[`layers/layer4/api/server.py:13`](../layers/layer4/api/server.py#L13)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `session_id` | `str` | 是 | — | — |
| `user_message` | `str` | 是 | — | — |
| `turn` | `TurnInstruction` | 是 | — | — |
| `fix_instruction` | `str` | 是 | — | — |

### `layers.layer4.generation.responder`

#### `GenerationResult` · 内部结构

源码：[`layers/layer4/generation/responder.py:15`](../layers/layer4/generation/responder.py#L15)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `reply` | `str` | 是 | — | — |
| `stage_complete` | `bool` | 否 | `False` | — |
| `reason` | `str` | 否 | `''` | — |

### `layers.layer4.review.orchestrator`

#### `ReviewOutcome` · 内部结构

源码：[`layers/layer4/review/orchestrator.py:15`](../layers/layer4/review/orchestrator.py#L15)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `reply` | `str` | 是 | — | — |
| `review_history` | `list[ReviewResult]` | 是 | — | — |
| `attempts` | `int` | 是 | — | — |
| `fallback_used` | `bool` | 是 | — | — |
| `fix_instructions` | `list[str]` | 是 | — | — |
| `stage_complete` | `bool` | 否 | `False` | — |

### `layers.layer4.llm.base`

#### `Message` · 内部结构

源码：[`layers/layer4/llm/base.py:8`](../layers/layer4/llm/base.py#L8)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `role` | `Literal['system', 'user', 'assistant']` | 是 | — | 枚举='system', 'user', 'assistant' |
| `content` | `str` | 是 | — | — |

### `layers.layer4.session.history`

#### `ConversationTurn` · 内部结构

源码：[`layers/layer4/session/history.py:17`](../layers/layer4/session/history.py#L17)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `role` | `Literal['user', 'assistant']` | 是 | — | 枚举='user', 'assistant' |
| `content` | `str` | 是 | — | — |

#### `SessionData` · 内部结构

源码：[`layers/layer4/session/history.py:22`](../layers/layer4/session/history.py#L22)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `session_id` | `str` | 是 | — | — |
| `machine` | `SubstageMachine` | 否 | 有默认值 | — |
| `history` | `list[ConversationTurn]` | 否 | 有默认值 | — |
| `l3_history` | `list[TurnInstruction]` | 否 | 有默认值 | — |
| `fallback_history` | `list[bool]` | 否 | 有默认值 | — |
| `manual_review_required` | `bool` | 否 | `False` | — |
| `role` | `RoleInstruction` | 否 | 有默认值 | — |
| `anchor` | `DiagnosticAnchor | NoneType` | 否 | `None` | — |
| `boundary` | `BoundaryInstruction` | 否 | 有默认值 | — |
| `safety` | `SafetyPassport` | 否 | 有默认值 | — |
| `therapy_options` | `list[str]` | 否 | 有默认值 | — |
| `meta` | `dict` | 否 | 有默认值 | — |

### `layers.layer4.session.state_machine`

#### `SubstageMachine` · 内部结构

源码：[`layers/layer4/session/state_machine.py:10`](../layers/layer4/session/state_machine.py#L10)

L4 子层状态机：L4-L1 共情 → L4-L2 结果反馈 → L4-L3 技术执行 → L4-L4 作业布置。

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `current` | `Literal['L4-L1', 'L4-L2', 'L4-L3', 'L4-L4']` | 否 | `'L4-L1'` | 枚举='L4-L1', 'L4-L2', 'L4-L3', 'L4-L4' |
| `flags` | `list[str]` | 否 | 有默认值 | — |

## L5：独立输出审核

### `layers.layer4.schemas.l5_context`

#### `DiagnosisContext` · 内部结构

源码：[`layers/layer4/schemas/l5_context.py:8`](../layers/layer4/schemas/l5_context.py#L8)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `primary` | `str` | 否 | `''` | — |
| `secondary` | `list[str]` | 否 | 有默认值 | — |
| `suicidal_ideation` | `str | NoneType` | 否 | `None` | — |

#### `RiskContext` · 内部结构

源码：[`layers/layer4/schemas/l5_context.py:14`](../layers/layer4/schemas/l5_context.py#L14)

!!! abstract "Usage Documentation"

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `risk_level` | `int` | 否 | `1` | — |
| `sensitive_topics` | `list[str]` | 否 | 有默认值 | — |
| `flags` | `list[str]` | 否 | 有默认值 | — |

#### `L5Context` · 层间接口

源码：[`layers/layer4/schemas/l5_context.py:20`](../layers/layer4/schemas/l5_context.py#L20)

接口四：L4 + L3 → L5 ContextualSafetyContext（上下文安全上下文）。

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `l4_raw_output` | `str` | 是 | — | — |
| `l4_substage` | `Literal['L4-L1', 'L4-L2', 'L4-L3', 'L4-L4']` | 是 | — | 枚举='L4-L1', 'L4-L2', 'L4-L3', 'L4-L4' |
| `therapy_type` | `str` | 是 | — | — |
| `user_message` | `str` | 是 | — | 最短长度=1 |
| `history` | `list[ConversationTurn]` | 是 | — | — |
| `instructions` | `TreatmentInstructions` | 是 | — | — |
| `diagnosis_context` | `DiagnosisContext` | 否 | 有默认值 | — |
| `risk_context` | `RiskContext` | 否 | 有默认值 | — |

### `layers.layer4.schemas.review`

#### `ReviewResult` · 层间接口

源码：[`layers/layer4/schemas/review.py:8`](../layers/layer4/schemas/review.py#L8)

L5 审核结果（Mock 审核器与真实 L5 共用）。

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `harm` | `Literal['pass', 'fail']` | 是 | — | 枚举='pass', 'fail' |
| `boundary` | `Literal['pass', 'fail']` | 是 | — | 枚举='pass', 'fail' |
| `quality` | `Literal['pass', 'fail']` | 是 | — | 枚举='pass', 'fail' |
| `fail_reason` | `str` | 否 | `''` | — |
| `fix_instruction` | `str` | 否 | `''` | — |

## 顶层编排器

### `orchestration.schemas`

#### `PatientConfirmation` · 层间接口

源码：[`orchestration/schemas.py:32`](../orchestration/schemas.py#L32)

患者可见的脱敏确认卡，不包含内部评分、规则或证据编号。

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `title` | `str` | 是 | — | — |
| `profile_summary` | `list[str]` | 否 | 工厂生成 | — |
| `scale_summary` | `list[str]` | 否 | 工厂生成 | — |
| `initial_impression` | `str` | 是 | — | — |
| `disclaimer` | `str` | 是 | — | — |
| `next_step` | `str` | 是 | — | — |
| `actions` | `list[Literal['continue', 'revise', 'pause']]` | 否 | 有默认值 | — |

#### `OrchestrationOutcome` · 层间接口

源码：[`orchestration/schemas.py:48`](../orchestration/schemas.py#L48)

患者/API 可见的统一输出；刻意排除开发者画像和内部临床对象。

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `session_id` | `str` | 是 | — | — |
| `route` | `Literal['L1_SAFETY_ASSESSMENT', 'L1_SAFETY_HOLD', 'L2_INTERVIEW', 'L2_SCALE', 'L2_DIAGNOSTIC_CONTEXT', 'L3_PROCESSING', 'RETURN_TO_L2', 'MANUAL_REVIEW', 'AWAITING_PATIENT_CONFIRMATION', 'L4', 'PAUSED_BY_PATIENT', 'STOPPED']` | 是 | — | 枚举='L1_SAFETY_ASSESSMENT', 'L1_SAFETY_HOLD', 'L2_INTERVIEW', 'L2_SCALE', 'L2_DIAGNOSTIC_CONTEXT', 'L3_PROCESSING', 'RETURN_TO_L2', 'MANUAL_REVIEW', 'AWAITING_PATIENT_CONFIRMATION', 'L4', 'PAUSED_BY_PATIENT', 'STOPPED' |
| `reply` | `str` | 是 | — | — |
| `waiting_for` | `str | NoneType` | 否 | `None` | — |
| `terminal` | `bool` | 否 | `False` | — |
| `confirmation` | `PatientConfirmation | NoneType` | 否 | `None` | — |

#### `PipelineSession` · 内部结构

源码：[`orchestration/schemas.py:61`](../orchestration/schemas.py#L61)

只保存在编排层的全局状态。

| 字段 | 类型 | 必填 | 默认值 | 约束 |
|---|---|:---:|---|---|
| `session_id` | `str` | 是 | — | — |
| `route` | `Literal['L1_SAFETY_ASSESSMENT', 'L1_SAFETY_HOLD', 'L2_INTERVIEW', 'L2_SCALE', 'L2_DIAGNOSTIC_CONTEXT', 'L3_PROCESSING', 'RETURN_TO_L2', 'MANUAL_REVIEW', 'AWAITING_PATIENT_CONFIRMATION', 'L4', 'PAUSED_BY_PATIENT', 'STOPPED']` | 否 | `'L2_INTERVIEW'` | 枚举='L1_SAFETY_ASSESSMENT', 'L1_SAFETY_HOLD', 'L2_INTERVIEW', 'L2_SCALE', 'L2_DIAGNOSTIC_CONTEXT', 'L3_PROCESSING', 'RETURN_TO_L2', 'MANUAL_REVIEW', 'AWAITING_PATIENT_CONFIRMATION', 'L4', 'PAUSED_BY_PATIENT', 'STOPPED' |
| `created_at` | `str` | 否 | 工厂生成 | — |
| `updated_at` | `str` | 否 | 工厂生成 | — |
| `l2_session_id` | `str | NoneType` | 否 | `None` | — |
| `l4_session_id` | `str | NoneType` | 否 | `None` | — |
| `expected_input` | `Literal['auto', 'free_text', 'focus_selection', 'scale_consent', 'scale_answer', 'patient_goals', 'motivation', 'profile_confirmation', 'symptom_duration_days', 'functional_impairment', 'mania_or_hypomania_history', 'substance_or_medical_cause', 'psychotic_symptoms', 'differential_diagnoses_assessed', 'evidence_conflicts']` | 否 | `'auto'` | 枚举='auto', 'free_text', 'focus_selection', 'scale_consent', 'scale_answer', 'patient_goals', 'motivation', 'profile_confirmation', 'symptom_duration_days', 'functional_impairment', 'mania_or_hypomania_history', 'substance_or_medical_cause', 'psychotic_symptoms', 'differential_diagnoses_assessed', 'evidence_conflicts' |
| `deferred_l2_reply` | `str` | 否 | `''` | — |
| `deferred_l2_expected_input` | `Literal['auto', 'free_text', 'focus_selection', 'scale_consent', 'scale_answer', 'patient_goals', 'motivation', 'profile_confirmation', 'symptom_duration_days', 'functional_impairment', 'mania_or_hypomania_history', 'substance_or_medical_cause', 'psychotic_symptoms', 'differential_diagnoses_assessed', 'evidence_conflicts'] | NoneType` | 否 | `None` | — |
| `l1_passport` | `dict[str, Any] | NoneType` | 否 | `None` | — |
| `clinical_intake` | `ClinicalIntake | NoneType` | 否 | `None` | — |
| `diagnostic_report` | `DiagnosticReport | NoneType` | 否 | `None` | — |
| `l3_raw` | `dict[str, Any] | NoneType` | 否 | `None` | — |
| `pending_l3_dimensions` | `list[dict[str, Any]]` | 否 | 工厂生成 | — |
| `pending_l5_review` | `dict[str, Any] | NoneType` | 否 | `None` | — |
| `l3_to_l2_loop_count` | `int` | 否 | `0` | — |
| `patient_confirmation` | `PatientConfirmation | NoneType` | 否 | `None` | — |
| `stopped_reason` | `str` | 否 | `''` | — |
| `internal_flags` | `list[str]` | 否 | 工厂生成 | — |
| `latest_developer_profile_path` | `str | NoneType` | 否 | `None` | — |
