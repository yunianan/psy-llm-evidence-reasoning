# MentalBench 离线适配至 L2→L3 格式

> 历史记录。文中的旧 L3 诊断及实验入口已退役。当前会话使用 [ICD-10 诊断与临床复核](../README.md#icd-10-人工复核与-l4-放行)。

2026-11-04：项目内的 MentalBench 离线评测工具已退役。以下保留为历史协议与数据说明。

此工具将本地 MentalBench 英文病例正文转换为现有 `ClinicalIntake` 和 L2→L3 `profile` 格式，供第三层实验准备输入，不经过患者对话。原始 MentalBench 和 MentalKG 文件只读。

2026-10-15 已完成一次完整导出：24,750 条，转换失败 0 条。Type 1 为 1,725 条、Type 2 为 3,450 条、Type 3 为 6,525 条、Type 4 为 13,050 条；三个生成模型各 8,250 条。输出统计见导出目录的 `conversion_report.json`。

## 本轮转换的含义

转换方式是 **lossless_narrative_only（保留原文的格式适配）**：按段落保存 `key_narratives`、`patient_quotes` 和一一对应的 `evidence_nodes`。节点来源明确标为 `benchmark_narrative`，生成方式为 `offline_source_adapter`，不冒充真实患者消息或 LLM 已抽取事实。

这一步完成统一格式与证据追溯，**尚未执行语义抽取**：不会从生成用的 `features` 复制症状 ID、病程、损害、发作史或排除信息，也不凭规则猜测它们。当前导出的 `key_symptoms_extracted` 为空，结构化诊断字段和未施测量表、动机评分保持未知。病例事实完整保留在原文段落中，第三层节点模型仍能读取这些叙述。

因此这些文件不是已完成真实 L2 问诊、量表和患者确认的画像。`ClinicalIntake` 明确标记信息不足、未请求患者确认和抽取待完成。它们可作为候选生成研究的叙述输入；若要研究“充分结构化画像条件下的候选生成”，还需先抽取并核对临床字段，再冻结输入。应在实验中区分这两种输入条件。

## 运行

在 L1-L5 项目目录执行：

```powershell
.\.venv\Scripts\python.exe tools/adapt_mentalbench_profiles.py
```

默认读取 `../external/MentalBench/resources/dataset`，输出 `artifacts/datasets/mentalbench_l2_format`。可用 `--source`、`--output` 指定目录；`--limit` 用于小样本检查。已有导出文件时拒绝覆盖，避免误改已冻结的数据。

## 输出

| 文件 | 内容 |
|---|---|
| `profiles.jsonl` | 匿名案例 ID 和 L2→L3 同格式画像；不带参考标签、选项、生成提示词或来源疾病目录 |
| `l2_intakes.jsonl` | 同案例的完整 `ClinicalIntake` 结构，保留离线与未确认标记 |
| `gold.jsonl` | 独立参考答案与标签、任务类型、生成模型、来源文件和原始案例 ID |
| `sources.jsonl` | 转换方式、关联案例分组、抽取状态及完整 L3 入口缺失字段 |
| `conversion_report.json` | 数量、任务分布、生成模型分布、转换失败及缺失字段统计 |

导入时只读取普通任务文件，跳过 `features` 目录，避免重复计算生成副本。参考标签由 answer 对应 options 的名称映射而来，无法解释或相互矛盾则记录失败，不根据疾病文件夹猜答案。

Type 3 的参考标签标记为“当前信息下仍可能成立的候选集合”，不当成已确诊共病。与专家编写病例分开记录其 `MentalBench_synthetic_graph_grounded` 标签来源。

## 与生产入口的关系

复用 `L2ToL3Adapter.serialize_profile` 的同一套字段序列化；生产 `to_profile` 继续先执行 L2 出口守卫。未采集的五个诊断字段保持未知，完整 `diagnose()` 仍会返回 L2 补充信息，不为通过守卫伪造阴性病史。

仅做 Step 1 离线实验时，可将一条 `profiles.jsonl` 的 `profile` 交给候选推理器，先由 `CLINICR._known_evidence_refs` 得到原文证据白名单，再调用 `generate_candidates`。这会调用配置的模型 API，转换工具本身不调用模型。该入口是候选层实验，不生成进入治疗的诊断通行证。

2026-10-16 已在独立的 `artifacts/datasets/mentalbench_split_v2` 中完成 KB/DEV/EVAL 分组和质量筛查，详见 [划分与整个 L3 评测规范 v2](mentalbench-split-v2.md)。本目录的全量格式适配文件保持原样。v2 尚未接入运行时案例检索，也未运行正式模型评测；临床字段抽取和字段级证据补全仍待完成。`related_case_group` 保留可直接确认的原始编号关系，不证明种子级独立性。
