# Phase 1 研究入口与配对评分

实现范围为 A0/A1 研究入口和协议 v1.1 的离线评分。生产默认仍为 Unconnected，未接入 G2、病例轨 C 或 A2/A3 引擎。

## 入口约定

- A0 独立返回 `CandidateGeneration`，不运行 `CLINICR.run`。prompt 仅包含患者画像、输出约束和 24 个 `{disorder, label}` 展示项。整条组装路径经 sentinel 和独立进程文件读取拦截验证，不读取 criteria、症状定义、鉴别、事件、树问题或病例摘要。
- `considered_disorders` 保持目录固定顺序的 24 个 ID；预测标签的 coverage 为 `candidate`，其他标签为 `not_triggered`。`tree_trace=[]`、`decision_paths=[]`、`final_diagnoses=[]`。证据引用只能来自患者输入白名单。
- A1 显式注入 `JsonDiagnosticKnowledgeProvider` 并运行现有 CLINICR；研究入口绕过生产 `/diagnose`。保留 thinking on、screening zero、anchor/criteria/guard full，不改医学条件或既有 A1 prompt。
- 每例保存终态 JSON，汇总为 `predictions/{arm}.jsonl`。传输/契约失败标 `model_failure=true`；预算/行政失败标 `model_failure=false, run_failure=true`。两者均导出显式空预测，不删病例。

## 离线运行

以下 PowerShell 命令在 `L1-L5系统` 目录执行。例子使用已准备的 25 例 DEV 输入、人工合成 A0 回复和历史 A1 报告，不调用模型。输出目录必须是新目录；已有目录只能用相同配置加 `--resume`。

```powershell
$example = '..\phase1\p1.2-runner\runs\offline-example-v2\inputs'
$output = '..\phase1\p1.2-runner\runs\my-offline-replay'
.\.venv\Scripts\python.exe tools\phase1_research_entry.py --arm A0 --dev-list "$example\dev25.inputs.jsonl" --a0-responses "$example\synthetic-a0-responses.jsonl" --out $output
.\.venv\Scripts\python.exe tools\phase1_research_entry.py --arm A1 --dev-list "$example\dev25.inputs.jsonl" --replay-predictions "$example\historical-a1.predictions.jsonl" --g1-cache-dir '..\phase1\p1.1-engineering\runs\cached-full-smoke25-bounded-v1\g1-nodes' --g1-cache-dir '..\phase1\p1.1.1-infrastructure\runs\repair9-v1\g1-nodes' --out $output
```

恢复时在对应命令后加 `--resume`。manifest 绑定名单、代码文件版本、源数据、模型、tree/knowledge/prompt、日志和运行策略；不匹配则拒绝恢复。已有病例终态成功或失败均不重新生成。批次 checkpoint 和请求账本保留已完成节点、原始响应与已消耗重试额度。

主要产物：

| 路径 | 内容 |
|---|---|
| `manifest.{arm}.json` / `manifest.json` | 运行前身份与各臂索引 |
| `runs/{arm}/{case_id}.json` | 不被恢复流程覆盖的病例终态 |
| `runs/{arm}/requests.jsonl` | HTTP 尝试、usage、异常分类和原始输出；离线回放为空 |
| `runs/{arm}/events.jsonl` | 引用规范化、校验、修复及终态关联事件 |
| `predictions/{arm}.jsonl` | 包含全部登记病例的统一预测行 |
| `summary.{arm}.json` | 完整率、响应模型、usage 缺失和参考预算 |

请求名预先登记；响应模型和 fingerprint 只能在调用后记录，不能伪造为请求参数。API 密钥不进入 manifest 或日志。

## 配对评分

```powershell
.\.venv\Scripts\python.exe tools\paired_evaluation.py "$example\dev25.gold.jsonl" "$output\predictions\A0.jsonl" "$output\predictions\A1.jsonl" --kind single_reference_label --hypothesis H1 --out "$output\scoring\single.json"
.\.venv\Scripts\python.exe tools\paired_evaluation.py "$example\dev25.gold.jsonl" "$output\predictions\A0.jsonl" "$output\predictions\A1.jsonl" --kind admissible_candidates --hypothesis H1 --out "$output\scoring\dual.json"
```

差值方向为 `y−x`，本例为 A1−A0。两层分开输出 JSON 和 Markdown，不能混池。该示例的 A0 为合成数据，其差值没有真实 H1 性能解释。

- `compare()` 校验两臂 ID 与合格 gold 完全相同，拒绝缺行、重复、非法标签和非有限置信度。
- 候选仅包括 `candidate` / `provisional_candidate`，按 `(-confidence, disorder)` 排序，K=5。
- 公开整数分子分母，按 `family_key(source_code)` 整组抽样。B=1000、seed=20261015，同一索引序列用于两臂；使用 `Fraction` 和第 24/974 下标判界，JSON 中分数为字符串。
- H1 为 recall 优效且 PPV 非劣 1pp 的 IUT。任一抽样 PPV 未定义则 PPV CI 为 null，不补采；Top-5 区间独立保留。
- `evaluate_study()` 可接受合成 A0–A3 预测，所有比较共享抽样序列，并执行 H1→H2→H3 顺序门。它不实现 A2/A3 引擎。单臂旧评分器仅复用规范化函数，其原有统计行为保持不变。
- 质量门、描述性点估计与统计推断状态分别输出；失败保留 recall 分母，不增加 PPV 分母。20 例终态监测块的契约/传输阈值分别为 >10% / >15%，内部恢复超时不计终态失败。

## 验证和后续运行边界

`tests/fixtures/paired/toy-1..9/` 每个目录含 gold、两臂预测、独立预期 JSON。测试逐份深度比较，并单独验证原 v1 三例合并结果、共享抽样、精确 −1pp、未定义 PPV、顺序门及停批边界。

当前验证为离线测试和历史回放。付费适配路径用 fake SDK 验证过调用/日志接口，尚未在线验收。未来真实 DEV 运行必须另行登记名单、预算和版本，显式使用 `--execute --model <请求名> --budget-ref <json>`；预算引用字段为 `pricing_basis=official_peak_all_input_miss_reference`、`batch_limit_yuan`、`case_limit_yuan`，不是代理实际账单。

## P1.3 工程事件汇总

入口已升级为 `phase1-research-entry-v2`，新运行写入 `l3-event-envelope-v1` 信封，并在运行/恢复结束后自动出 `observability/{arm}.summary.json/.md`。新旧代码身份不同，不能以新版本恢复P1.2旧manifest；旧产物保持原样。

```powershell
.\.venv\Scripts\python.exe tools\l3_observability_summary.py --run <新运行目录> --out <汇总目录>\summary.json
.\.venv\Scripts\python.exe tools\l3_observability_summary.py --run '..\phase1\p1.2-runner\runs\offline-example-v2' --legacy-audit --out '..\phase1\p1.3-observability\offline-example-v2.summary.json'
```

默认严格校验事件序号、病例/请求身份、修复闭环及终态/预测对齐，违反则报错，不改输入。历史审计显式使用 `--legacy-audit`，只列缺项，`strict_acceptance=false`，不能冒充新契约通过。

汇总按臂、阶段和病例计数；usage缺失时总额为null，已知小计单列；reasoning属于completion，cache miss不反推。时延先按病例合计已记录HTTP elapsed再取p50/p95，不代表未记录的全流程墙钟时间。参考费用不等于实际账单。node作用域要求真实节点；行政、病例完成、回放和Verifier使用明确的非节点作用域，不造attempt/node ID。

## P1.4 联合工程运行

`phase1-research-entry-v3`新增可选联合控制，默认单臂行为保留。`tools/phase1_joint_run.py`按共同名单每例A0后A1，活跃推断并发1；全局预算共享，两臂按arm:case_id分别执行单例限额。每20例两臂同块检查，完成率/预算/完整性提前触发时停止新模型调用，剩余病例保留行政终态。恢复合并两臂请求账本，不重置预算、HTTP额度或已有失败行。

```powershell
.\.venv\Scripts\python.exe tools\phase1_joint_run.py --dev-list <DEV名单.jsonl> --out <全新目录> --budget-ref <预算引用.json> --model deepseek-v4.1-flash-1017 --execute
```

当前联合控制版本phase1-joint-run-v1，部署前冻结的规划费用预测为A0每例0.04元、A1每例0.87469元，均是官方峰时参考值；A0尚未实测，不保证完成。实际费用升高导致剩余余额不足1.3倍未启动工作预测时停止，不自动提高预算。模型请求名与各臂prompt/tree/knowledge分别登记，不使用历史源fingerprint填当前响应。

新入口当前顺序执行（`max_workers=1`），manifest 如实记录；不能将历史并发 2 的实测直接归给它。流式、read timeout 300s、最多 3 次传输尝试和一次结构修复均在 manifest 中登记。正式工程冻结、真实 DEV 成本/缓存计量、Insomnia 来源覆盖验收及 Phase 1 关门报告仍需后续完成。
