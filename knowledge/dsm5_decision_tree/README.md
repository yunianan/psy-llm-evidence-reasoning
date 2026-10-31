# DSM-5 第三层候选生成决策子图

更新：2026-10-20。项目：基于大语言模型的心理诊疗循证推理研究。

## 保留的执行器（需显式知识提供者）

`CLINICR.run → APICLINICRReasoner.generate_candidates → DecisionTreeCandidateGenerator`

输入是 L2 结构化患者画像。执行器代码保留；当前默认诊断知识已断开，等待后续 GraphRAG 接入。旧的一次性自由生成仅保留为显式调用的 `generate_candidates_baseline`，没有自动回退到它的逻辑。

诊断内容通过可注入的 `DiagnosticKnowledgeProvider` 提供。默认提供者处于未连接状态，不读取 MentalKG 或失眠补充的诊断内容；原 JSON 适配器仅供离线工具和测试显式使用。候选生成、标准汇总、终验及报告来源共享注入接口。见 [知识访问说明](../../docs/l3-knowledge-access.md)。

```text
患者画像及合法证据引用
  → 9 个症状入口（met / not_met / pending）
  → 相关候选的症状证据核对
  → 有来源的发作史、时间关系、病因等鉴别分支
  → 原始标准分解：症状数量 / 核心症状 / 病程 / 功能影响 / 附加要求
  → 合并为 0–24 个候选及证据记录
  → 程序复用 Step 1 记录汇总候选状态（不再调用 Step 2 模型）
  → 现有独立 Verifier 接口
```

适用范围是 D001–D023＋Insomnia。九个入口是针对这些标签归并的工程入口，不是原书树数，也不等同于九个界面展示分组；入口与展示分组按各自用途定义。原版手册有 29 棵树；本配置提取相关鉴别方向构成项目子图，未完整转录每棵原树的所有节点、跨树边和范围外终点。不能据此宣称完全复现论文、穷尽 DSM-5 或已经获得临床验证。

## 医学依据与程序策略

| 内容 | 来源/处理 |
|---|---|
| 症状入口与鉴别方向 | Michael B. First，DSM-5 Handbook of Differential Diagnosis，2014，第二章及相关第三章表；配置记录树号、印刷页码 |
| 原始 23 个标签、症状、标准及鉴别说明 | 原作者 MentalBench/MentalKG 仓库 `resources/knowledge_graph/EN`，只读 |
| 失眠 A–H | 工作区 `external/MentalBench/resources/dsm5_supplement/EN`；标准来源见其 README；不是原 MentalKG 节点 |
| 程序组织方法 | Sarma 等论文的入口筛选、迭代问题、收集候选及复杂条件分解，Methods 2.2.3、2.3 |
| 未知值处理、批量请求、最低召回条件 | 项目工程适配，需要独立评测；不构成新的 DSM 医学标准 |

执行规则：

- 入口明确否定才停止该入口；入口未知可继续检查实际症状证据。候选至少需要一个原始症状节点有患者依据，这只是宽松召回条件，不能解释为达到诊断阈值。
- 每个已判定节点须有患者证据；未知不作为否。显式发作史/病因鉴别反证可裁剪分支；其他完整标准不足或未知仍保留为 provisional 候选并汇总为 watch，不由第二个模型重新判定。
- `required_criteria` 从知识库实时读取。发作组分解为计数、核心症状、病程及损害等独立问题；附加要求逐条读取，JSON Pointer 精确到字段/数组项。程序检查模型声称满足的计数与核心症状是否对应它实际返回的症状 ID。
- 只统计不同症状 ID；病程、损害、病因不算症状。`symptom_match` 是来源症状池中的映射数量，包含来源可选症状时也不变成通用 DSM 分数。`confidence` 是节点主观证据支持度，未作临床概率校准。
- 病因关系不能由“患者有躯体病/用药/物质使用”直接推出；失眠与精神或躯体疾病共存不自动排除失眠。ADHD 亚型及精神病性特征保留原标签，不把基础疾病名称解释为没有该特征。
- 来源未覆盖的年龄条件、来源语义冲突和无法确证的完整发作须保持 pending，不用模型记忆偷偷补医学规则。原 MentalKG 的标准完整性与适用人群限制仍存在。
- 多个入口归并到同一候选时合并访问记录；记录的是项目实际访问节点顺序，不冒充一条原书原图路径。范围外终点不被强制映射到 24 个标签；当前实现记录排除节点与覆盖状态，不提供范围外疾病的完整候选输出。

同一遍历层的独立节点每批最多 8 个，一次回答全部当前节点。程序决定分支，模型不能指定新节点或候选 ID。结构错误/证据越界共享一次修复机会；仍失败则沿用 L3 错误路径，不降级自由生成。模型调用次数会多于旧实现。

## 输出与来源定位

- `candidates[].decision_paths`：当前候选的入口、鉴别及标准记录，带 `provisional`。
- `tree_trace`：所有访问节点，包含已裁剪分支；每项有 `status`、简短 `rationale`、患者 `evidence_refs`、知识 `source_refs`、症状 ID。
- `candidate_coverage`：24 个标签分别为 not_triggered、entry_pending、anchor_pending、excluded:<node>、provisional_candidate 或 candidate。
- `candidate_missing_dimensions`：未知节点对应的信息缺口；没有可用诊断时参与 L2 补问。

`MentalKG:diagnostic_criteria.json#/Dxxx/...` 定位原图谱；`../external/MentalBench/resources/dsm5_supplement/EN/...#/Insomnia/...` 从 L1-L5 项目根目录定位补充条目；`First2014:...#print-pages=...` 按下列 manifest 中的版本和页码解释。本手册副本的 PDF 页面序号＝印刷页码＋17。

逐候选 ELIMINATIVE 模型调用及其并发配置已删除。`diagnosis/assessment.py` 只汇总 Step 1 的原始记录：必需标准与鉴别路径齐备且满足时进入后续终验，未满足/未知保持 watch，已被 Step 1 剪枝的候选记录为 ruled_out。`checked` 为兼容报告字段，每项以 `source=step1_tree_records` 标记来源；不代表又做了一次独立验证。现有 Verifier 的打分与路由策略保留。为继续满足独立验证接口的“不读候选置信度”要求，新增路径中嵌套的节点置信度也不传给验证器。没有引入论文的候选成对比较。API 与 L4 报告可保存新增记录。

## 文件与复现

- `candidate_paths.json`：实际执行配置；不含患者数据。
- `source_manifest.json`：版本与原始链接。
- `../../layers/layer3/diagnosis/decision_tree.py`：执行器。
- `../../tools/build_dsm5_candidate_graph.py`：从可追溯定义重新生成配置；不会写入 MentalKG。
- `dataset_plan.md`：专家病例纳入、分层与评测准备状态。

在 L1-L5 根目录执行：

```powershell
.\.venv\Scripts\python.exe tools/build_dsm5_candidate_graph.py
.\.venv\Scripts\python.exe tools/generate_schema_reference.py --write
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe tools/verify_interfaces.py
```

当前测试使用受控模型响应，验证遍历、证据约束、未知/空候选、24 标签可达性、并发隔离及现有系统兼容性。新增测试覆盖重复调用移除、必需/可选条件汇总、证据与 JSON 来源保留、剪枝输出及补问路由。它们没有测量真实模型诊断 PPV、召回或医生一致性。
