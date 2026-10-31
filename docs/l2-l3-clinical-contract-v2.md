# L2/L3患者事实与补采契约v2

项目：基于大语言模型的心理诊疗循证推理研究。日期：2026-10-30。

版本：`l2-l3-clinical-v2`；病程工程策略：`duration-interval-v1`。
实现定义位于`shared/clinical_contract.py`、`shared/clinical_profile.py`、`shared/temporal.py`和`shared/temporal_gate.py`。
自动生成的完整字段表见同目录`layer-schema-reference.md`。

## 共享边界

L2保留患者原话，按统一字段枚举校验抽取候选，再形成逐属性事实和病程区间。
`L2ToL3Adapter`与HTTP请求共同使用`ClinicalProfile`校验，完整转发量表逐题数据。
L3报告保留`duration_evidence`、`temporal_audit`、`information_collection_plan`和`diagnostic_readiness`。
编排器将病程快照回写L2，并将完整补采任务交给L2；L2分析上下文能看到当前任务的节点、症状、属性及已有症状信息。

每个`ClinicalFact`包含`fact_id/symptom_id/attribute/state/value/raw_span/evidence_refs`。
状态为`present/absent/unknown/conflict`；病程事实额外携带`DurationEvidence`。
来源引用必须存在，且原文片段必须实际包含在被引用的患者证据中。
属性更新保留逐次事实；不同属性分别登记冲突，不把新值静默覆盖当成已核实的一致事实。

`symptom_duration_days`只保留为老整数兼容视图。新病程事实以`duration_evidence`为准；
字符串病程不再被跳过，解析失败或模糊时仍保留原文和出处。
自然语言病程不能被提升成所有症状共同的持续时间。

## 病程记录

核心字段为：

```json
{
  "symptom_id": "insomnia",
  "scope": "symptom:insomnia",
  "raw_span": "差不多有大半年了",
  "min_days": 150,
  "max_days": 240,
  "unit": "year",
  "quantity": 0.5,
  "approx": true,
  "approximation": "colloquial",
  "anchor": "relative_to_interview",
  "pattern": "unknown",
  "duration_kind": "overall",
  "parse_status": "ok",
  "source": "L2",
  "evidence_refs": ["患者原话证据ID"],
  "policy_version": "duration-interval-v1"
}
```

`scope`可以是症状作用域或明确的L3节点；没有可靠归属时使用`unspecified/unknown`。
`parse_status`支持`ok/vague/failed/unresolved`。`source`支持`L2/L3_writeback/legacy_adapter`，
其中`legacy_adapter`用于明确的兼容解析，不是从参考诊断生成患者事实。

`pattern`区分`continuous/episodic/unknown`，`duration_kind`区分整体病程与单次发作。
“反复三年，每次两周左右”形成两条记录：整体1096天、单次10–18天；两者不相加。
单次时长不能证明整体病程，反复多年的总体跨度也不能证明某次发作连续持续了多久。

## 工程区间策略

这些宽度是工程约定，不是文献提供的临床标准。医学门槛继续取自原知识来源。

|原文|min–max天|解析|
|---|---|---|
|约六个月|165–200|ok，around|
|八个月|243–243|ok|
|一年多|365–540|ok，more|
|大半年|150–240|ok，colloquial|
|两周左右|10–18|ok，around|
|一个半月／1个半月|45–45|ok|
|三四个月／3-4个月|91–121|ok，colloquial|
|几个月／好一阵子／很久|空|vague|
|去年冬天开始|缺访谈日期时为空|unresolved；有日期时按12月至次年2月的起点范围计算|
|换工作以后／从高三起|缺事件日期时为空|unresolved|
|时好时坏三年了|1096–1096|ok，episodic|

月按30.44天、年按365.25天计算，并保留原始数量与单位。
日精度的月、周、天和非整年向下取整；整年按最近整天取值，门槛使用相同转换函数。
因此八个月记录243天、三年记录1096天；不把一个月当30天、一个年当12个30天月。
复合时长先按原始换算因子累加，再统一取整，避免逐段截断累计误差；例如“1年两个月”为426天。
通常月/年“约数”取中心值±10%并向区间内部取整；周的“左右”取±4天。
“一年多”“大半年”使用表中的显式约定；“不到”“将近”的上界保持严格小于相应整日门槛。

日历起病如“2025年12月开始”按日历起点区间处理，不解释为2025年的病程。
仅有“3月开始”且缺年份时继续追问。访谈日期在会话建立时固定，避免后续运行日期改变含义。
事件日期只有明确提供时才可用于计算；“换工作以后”给出0至事件后经过天数的保守范围，不等于事件当天起病。

## 三值门与约数边界

- 最短病程：下界达到门槛为`met`；上界仍低于门槛为`not_met`。
- 最长病程使用对应的上界／下界比较。
- 跨门槛、模糊、失败、缺锚点、模式不符、归属不明或区间冲突为`unknown`，节点映射为`pending`。
- “约六个月”恰在六个月门槛中心时记为`met_approx`，保留审计标记；策略v1在最终诊断门按`pending`处理并追问。
- 需要事件结束锚点的门不能使用普通总体病程替代。

L3 v2在奖励终验之前重新解析引用原文并执行该比较，模型临时判断不能覆盖区间门。
外部提供的区间若与原文不符，会记录`structured_interval_disagrees_with_raw_span`并按原文重算。
每次采用的L2区间和新解析的原文均生成节点级`L3_writeback`快照，保留原L2记录。
审计分别记录模型原判定、程序生效判定、检测到的不一致及剩余不一致。

## 补采往返与版本兼容

`FollowUpTask`包含任务ID、字段、节点、症状、属性、问句、优先级、涉及候选、来源、状态及回答证据。
病程补采回答按任务节点绑定；“记不清”“去年冬天”等不会变成否认或固定天数。
节点归属不同且没有共同症状事实绑定的时长不得互借。

新的L2序列化和HTTP入口使用v2，HTTP会拒绝显式旧版本号。
未带版本的冻结研究字典仍可由内部研究重放代码按原口径复现；不原地迁移、修改指纹或重写旧预测。
旧批次要采用新口径必须显式生成v2审计overlay。新L2门禁允许交接带出处的未知病程，
这只代表已经采集到患者的不确定回答，最终病程门仍保持pending。

## DEV旧批次审计

本轮对15例、30条参考路径另存病程审计，保留历史预测和原始引用。
6例原“病程字段为空但已判定”的结构化空值归零。
新口径还会把约数边界与跨症状群归属列为待澄清，因此未知总数可以上升，不能以未知减少作为唯一验收标准。
审计文件仅用于对照引用及区间，不作为新增临床金标准或正式PPV试验输入。
