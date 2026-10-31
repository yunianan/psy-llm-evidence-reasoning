# Layer3 诊断知识访问

**当前状态：诊断知识未连接，等待用户后续通过 GraphRAG 接入。**

默认 `get_default_knowledge_provider()` 返回 `UnconnectedDiagnosticKnowledgeProvider`。
它不读取 MentalKG 或失眠补充的诊断内容，访问时抛出
`DiagnosticKnowledgeNotConnected`。默认候选生成和独立终验在请求 LLM 前停止，
由现有 `CLINICRModelUnavailable` 错误通道返回；HTTP `/diagnose` 返回 503 并提示
“诊断知识尚未接入”。初始化应用和查询疾病目录仍可使用。

```text
MentalKG 原始 JSON ── MentalKGSource                 独立保留
失眠补充 JSON     ── InsomniaSupplementSource         独立保留
失眠扩展子图     ── extension_graph.json              独立保留
Embedding       ── text-embedding-v4 / 1024 维        独立客户端

Layer3 → 未连接的知识提供者 → 提示诊断知识尚未接入
```

| 模块 | 职责 |
|---|---|
| `shared/knowledge_sources.py` | 两个独立数据源适配器，负责 JSON 读取、原始内容组装和来源定位；不依赖 Layer3 或检索框架 |
| `shared/diagnostic_knowledge.py` | 知识访问协议、未连接的运行时默认值，以及供离线工具和测试显式使用的 JSON 适配器 |
| `shared/disorder_catalog.py` | 标签、别名、中文展示、症状词表及治疗映射；不再组装诊断标准或鉴别知识包 |

标签和症状词表仍由原始数据源提供，供 L2 症状识别、Schema 候选范围和展示使用。
诊断条件、鉴别说明和事件定义不在目录模块导入时加载；这不建立 L3 的图遍历、
诊断知识提示词或病例检索连接。

两份数据在工作区 `external/MentalBench/resources/` 下并列存放：
`knowledge_graph/EN` 为 MentalKG，`dsm5_supplement/EN` 为失眠补充。
补充库来源引用以 `../external/MentalBench/resources/dsm5_supplement/EN` 开头，
相对于 L1-L5 项目根目录解析。

失眠库另提供[独立扩展子图](insomnia-extension-subgraph.md)，以 D011、D013 为挂接点。
该子图尚未接入这里描述的 Layer3 运行链路。

保留 `CLINICR(knowledge=...)` 注入接口。省略 `knowledge` 时使用未连接的提供者，
没有自动选择 JSON 的运行时逻辑，也不会自动回退到旧自由生成基线。默认创建的
候选生成器和 Verifier，以及标准汇总和报告引用均使用同一个提供者。若自行传入
`reasoner` 或 `verifier`，其知识依赖由调用方配置。

`JsonDiagnosticKnowledgeProvider` 仅由离线工具或测试显式构造。
离线 `candidate_knowledge()` 用它读取来源，用于知识检查和生成决策树配置；
此离线入口独立于运行时默认值。测试原执行器时显式注入 JSON 提供者，
另有回归测试保证默认 Layer3 不读取两份诊断知识、不请求 LLM 并返回未接入状态。

当前协议包含：

- `get_candidate(node)`：按项目疾病 ID 提供完整 `criteria`、相关 `symptoms`、
  `differential`、`events` 和 `source_ref`；保留失眠的具体条件引用与补充说明。
- `get_symptoms()`：提供症状词表，供显式调用的历史候选生成基线使用。

这是保留的结构化知识访问边界，尚未实现 GraphRAG、向量索引、患者查询构造或
相似病例检索。病例池仍在 `external/MentalBench/resources/dataset/知识库`，
开发集、评测集、隔离记录和生成种子未进入运行时输入。后续 RAG 接入需要实现
检索与知识适配，完整诊断条件仍须满足当前决策树的结构要求。

## 离线 screening 上下文试验

`APICLINICRReasoner(..., screening_zero_knowledge=True)` 可显式选择
`NodeKnowledgeView` 的 screening 零知识视图：只在该 stage 跳过标签知识读取，
其他 stage 保留原知识包。P1.0.7 验证后该模式已成为 reasoner 默认；单例 mood 的状态变化已按入口线索
接受为 intentional change，逐节点与旧输出一致不是验收门槛。

`screening_compact_view=True` 是另一显式试验选项，仅发送相关原始症状 ID/名称
和入口判定规则，不发送完整标签包；与零知识选项互斥。P1.0.6 中 zero/compact
均完成 20 例，overcall/undercall 的相对门槛通过，但活动集合一致率均为 19/20，
未满足预设 100%。用户复核将 mb-017541 的 appearance 改判登记为 false elimination candidate：
只否认体重/体形不足以排除整个入口；不得登记为 intentional change。

P1.0.7 新增 `TreeNodeJudgment.denial_scope=full|partial|none|unknown`（缺失默认 unknown）。
screening 的 `not_met` 只有 full 才能剪枝，其余在 API 输出边界和树执行器中降级为 pending，
记录 `screen_denial_scope_downgrade`；未知证据引用仍拒绝。此规则也覆盖自定义 evaluator，
不改变 anchor/guard/criterion 的判定。审计保留 raw/effective 两套结果，不把降级冒充模型首次正确。
默认选择须以 P1.0.7 的 targeted denial set 和 20 例复测全部门槛为依据。

两模式合成集均为 16/16；zero 完成 20/20 DEV、活动集合 20/20、错误 not_met=0，
输入 p95=6,565.8 token。compact 首轮 18/20，另有 schema 修复失败及 HTTP 402；充值后完整复测
达到 20/20、活动集合 20/20、错误 not_met=0，但 undercall=17/71，高于 full 的 14/71。
原失败与该未通过结论保留；按用户原有成本优先规则选择独立达标的 zero 默认，未要求未选 compact
也通过 undercall 才允许 zero 放行。解释纠正与原严格联合 fail 均有记录。

不传模式时默认 zero；单独传 `screening_compact_view=True` 选择 compact；
显式 `screening_zero_knowledge=False` 保留完整包离线对照。同时显式启用两模式仍报错。
详见 [P1.0.7 记录](D:/develop/vscode/心理诊疗系统/phase1/p1.0.7/report.md)及
[整轮复测与默认选择](D:/develop/vscode/心理诊疗系统/phase1/p1.0.7/compact-retest-report.md)。

screening compact 当前仅为失败研究对照，不作为以后备选默认。

## P1.0.8 anchor 紧凑视图实验

`APICLINICRReasoner(..., anchor_compact_view=True)` 仅对 anchor 使用紧凑视图，默认 False。
节点问题和 symptom_pool 不变；只发送本批标签名和一份 pool 症状 ID/名称/原始短定义/来源，
不发送完整 criteria、differential、events、examples/subtypes。原短定义中的成人等限定保留。
其他 stage 的上下文保持原样。

anchor 的 not_met 表示没有找到来源症状正证据，不是患者明确否认；因此允许该状态引用为空，
met 仍必须有患者 ref 和至少一个合法症状 ID，未知 ref 仍拒绝。此共同口径同时用于 full/compact 对照。
实际分支仍只保留 anchor met，原 Schema 的 pending 不强行转换，审计将其计入阳性证据被漏掉的路径风险。

screen 降级事件在每次完整候选生成开始重置一次，进入后续 frontier 时保留，供未来 A1 按整例累积。
P1.1 exporter 尚未实现；后续需持久化 case_id、screen 总数、降级数和原始/有效状态，累计比例高时回查 prompt。
正式 A1/A0 评分依赖 protocol v1.1（H1/H2/H3），当前 stage smoke 不冒用 v1 的 A3 vs A2 主终点。
实验结果与默认选择见 [P1.0.8 记录](D:/develop/vscode/心理诊疗系统/phase1/p1.0.8/report.md)。

本轮 full/compact 各 20/20 完成、480 合法节点、最终 schema 失败为 0；两臂各有 1/60 批结构化修复。
compact 输入减少 77.61%，请求输入 p95=5,899，但 anchor met/保留标签/coverage 均仅 7/20 一致。
相对运行前代理症状参考，compact false_not_met=6/282，false_met=20/198（full 为 13/282、18/198）。
参考存在 S023/S068 映射争议，不是专家金标；首次 fail 和原参考均保留。
**P1.0.8 不通过，anchor 继续默认 full；不因成本达标或漏项较少而推广。**
两臂共同的“not_met 为无正症状证据”口径保留，不恢复患者否认的强制要求。

公开候选生成入口在模型请求前检查未连接的默认 provider；该试验不能绕过默认
未接入边界，也不建立 GraphRAG 或病例轨连接。详见
[P1.0.5 对照记录](D:/develop/vscode/心理诊疗系统/phase1/p1.0.5-screening-zero.md)。
20 例结果及代理参考的勘误限制见
[P1.0.6 报告](D:/develop/vscode/心理诊疗系统/phase1/p1.0.6/report.md)。

## 工程降本：输出契约、标签分批与 G1 缓存

TreeNodeJudgment 现直接拒绝 pending/not_met 的非空 symptom_ids；旧 API/执行器约束继续保留，
聚合仅统计 met 的症状。修复仍最多一次，不会清空非法字段后冒充模型首次成功。
评分器显式 model_failure=true 时使用空候选并保留 recall 分母，另报 model_failure_rate。

离线 reasoner 可显式设置 criterion_batching="per_label"，每个标签的全部 criterion 使用一个请求，
完整标签包只发一次；不删除 criteria、症状完整对象、differential 或 events。默认 legacy 保留。
static_knowledge_prefix=True 将同样完整的来源知识和节点放在患者内容之前，以利供应商前缀缓存；
字段和原文不删减，默认 False。两项策略均写入 G1 缓存 context，不能与旧策略混用。

G1NodeCache 只存完整遍历的节点记录和 provenance；CachedG1Reasoner 命中后重新遍历和聚合，
不缓存最终候选。A2/A3 消费须使用 require 模式：缺缓存、损坏、版本/画像/来源变化均不得隐式补跑。
context 包括 case_id、画像/知识/tree 内容摘要、tree/prompt/model、服务地址、批次/prefix 与生成参数。
响应指纹在响应后登记，供应商不暴露则如实记录，不套用其他模型源的指纹。

原 19 成功病例的离线节点回放，对 A2/A3 两个 G1 消费入口均复现原候选/路径/coverage，
共 38 次回放、0 模型调用；这不表示 G2、病例轨或 A2/A3 整体已接入。
工程产物与 300 元预算情景记录在工作区独立 phase1/p1.1-engineering；正式样本量仍待平台单价、
新增 G2/C 成本及 v1.1 协议冻结。生产默认 provider 和未连接边界保持不变。
