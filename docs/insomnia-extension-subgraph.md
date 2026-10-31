# MentalKG 的失眠扩展子图

子图位于工作区 `external/MentalBench/resources/dsm5_supplement/extension_graph.json`。
它从同目录 `EN/` 下现有 JSON 构建，以原 MentalKG 的 D011、D013 为外部挂接点。
原图谱文件与失眠标准文件均保持原样。

子图构建和查询独立于 Layer3。它尚未接入候选诊断、提示词、知识提供者或 RAG；
当前诊断运行链路不会读取 `extension_graph.json`。

## 节点与关系

```text
Insomnia
  ├─ has_diagnostic_criteria → 完整失眠标准
  │    ├─ has_condition → 症状组（保留 relation、min_count）
  │    │    └─ uses_symptom → S085 / S086 / S087
  │    ├─ has_condition → 病程条件
  │    ├─ has_condition → 临床显著性条件
  │    │    └─ interpreted_by → 补充库的事件定义
  │    └─ has_condition → 附加要求列表及各条要求
  ├─ differential_diagnosis → D013（MentalKG 外部节点）
  └─ differential_diagnosis → D011（MentalKG 外部节点）
```

当前共 19 个节点、18 条边。疾病、症状沿用现有 ID；标准、条件、事件节点使用
`DSM5InsomniaSupplement:` 命名空间。节点及边均记录 `source_ref`，外部挂接点
另标记 `external=true`、`namespace=MentalKG`。

两个鉴别边的 `data` 完整保留原条目的情境、差异和双侧规则。边保留原始方向，
反向查询只是查找入边，不生成新的反向医学规则。子图没有新增症状等同关系，
也没有将 S027 与 S085–S087 合并。

## 构建与查询

在 L1-L5 项目根目录执行：

```powershell
.\.venv\Scripts\python.exe tools/build_insomnia_extension_graph.py
```

生成文件是现有源数据的图表示；更改源数据后用同一命令重新生成。

```python
from shared.insomnia_subgraph import InsomniaExtensionSubgraph

graph = InsomniaExtensionSubgraph()
graph.neighbors("Insomnia", relation="differential_diagnosis", direction="out")
# D013、D011，保留其 MentalKG 身份及来源。

graph.neighbors("D013", relation="differential_diagnosis", direction="in")
# Insomnia。

graph.relations("D013", relation="differential_diagnosis", direction="in")
# 原始有向边及完整鉴别规则、来源引用。

graph.to_dict()
# 可序列化的节点、边和挂接点；返回副本，不修改原始数据。
```

实现会拒绝症状 ID 覆盖、未知挂接点、悬空边及鉴别规则与端点不一致的数据。
