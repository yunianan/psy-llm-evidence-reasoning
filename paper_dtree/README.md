> 公开仓库版本：原文页、逐行转录及言语树原图转录未随仓库分发；节点和页码保留。来源重建测试在本版本跳过，详见 [公开版范围](../knowledge/paper_dtree/first2014_project24/PUBLIC_EXPORT.md)。

# 独立论文七阶段实验框架

2026-11-02 更新。项目：基于大语言模型的心理诊疗循证推理研究。

本模块按 Sarma 等论文 **Figure 1 的七阶段**组织整个实验；Figure 2 的决策树流程属于第 2 阶段内部。
对应 Methods 2.2–2.5、Table 1–2、Figure 1–2、Appendix B–D。
用户指定医学来源为 **First 2014《DSM-5 Handbook of Differential Diagnosis》**，因此当前来源目录使用
DSM-5，不冒充论文的 DSM-5-TR 内容复现。执行器也支持显式声明为 TR 的独立数据包，但不会自动切换标准。

**当前默认使用 24 标签版本：19 棵诊断家族树；9 类 / 24 项症状入口全部路由到当前树。
原默认 18 棵原图树包及其专用构建、校验和展示文件已按用户要求删除。历史实验快照保留。
这是来源可追溯的本地改编，作者优化树及完整诊断标准仍待单独完善。**
`knowledge/paper_dtree/first2014_source_catalog.json` 是按用户 PDF 逐一核对标题与起始页得到的目录，
不是可执行临床规则。当前默认树包及逐节点审计见
[First2014 当前 24 标签树](../knowledge/paper_dtree/first2014_project24/README.md)。
此前的 2.3 言语树示例保存在本地研究档案，未包含在公开版中。
原图包使用 `dt_raw`，不能冒充 `dt` 优化组。

### 患者追问的独立接口

`followup.py` 提供 `FollowUpRunner` 和 `L2FollowUpSession`：已选树节点使用 yes/no/unknown，unknown 暂停并返回追问；收到患者回答后由 L2 的 `handle_message` 执行现有消息和安全验证，保存患者原话及证据引用，再重评该节点。此前筛选和已确定节点以本次会话内的记录重放，不重复调用模型。不能回答或拒绝回答时返回未解决/暂停，不强行选 no。七阶段流水线在等待患者时停止在第二阶段，不生成评分。

交互模式下每个 yes/no 节点必须返回患者消息 ID 和逐字引用；程序检查消息角色与原文一致性，并尝试关联 L2 患者证据 ID。找不到 L2 证据节点但能验证患者原话时，保留消息来源并明确标记 `patient_message_only`；不存在的原话、缺少证据的 yes/no 会被拒绝，不作为有效节点判断。各候选的 `candidate_evidence` 保存实际到达该候选的路径节点、判断、书籍来源、患者引文以及两两比较后的保留状态。程序核对引用真实性，不等同于自动证明引文在医学上足以支持判断；手册来源的节点也不等于完整 DSM 标准。

该功能是与论文“信息不足回答 no”不同的交互模式。默认 `FiveLayerOrchestrator` 在患者确认画像后启动该树，待追问时路由至 L2；每条患者回复先通过 L1，之后才进入 L2 和树节点续跑。注入自定义 L3 的调用方可显式提供 `paper_tree_factory` 以启用。树候选尚未形成经过 L3→L4 校验的诊断报告：遍历完成、无法解决或需要安全复核时转人工复核，不自动进入治疗层。模型问题会受现有创伤相关安全控制限制。

当前编排会话和树续跑状态均保存在服务进程内；进程重启后不能恢复，进度缺失时转人工复核，不伪造结果。默认论文 CLI 仍使用原始二态复现模式；追问模式适用于系统对话流程。当前测试使用受控模型，不代表已完成真实患者或 API 验证。

当前入口按用户后续确认改为旧项目的 **9 个分类、24 个标签对应的症状入口**，详见
[入口配置和清单](../knowledge/paper_dtree/first2014_project24/entries.md)。主 CLI 默认加载独立导出的配置，
9 类用于组织 24 个筛选项，不增加分类门控。原书入口模式可用 `--entry-profile bundle` 显式选择。
这个范围改动来自用户选择，不再声称入口清单等同论文原始 28 棵树。

## 当前入口

```text
1 Query         病例原文形成查询；参考答案不进入此阶段
2 Model         同一指定模型使用 Base 或 DT 提示方法
                  Base：直接诊断
                  DT：所有入口筛选 → 所有适用树按回答推进分支 → 候选汇总 → 全部候选对比较
3 Raw Response  保存原始响应、模型名称与未简化诊断字符串
4 Simplify      按附录 C 六项规则简化并合并；保留原名与变更来源
5 Fuzzy Match   RapidFuzz WRatio 与显式标准名称表比对并记录分数
6 Reconcile     唯一 >99 分自动匹配；其余人工协调并入 SQLite 复用
7 Scoring       对照作者标签计 TP / FP / FN；逐例 PPV / 敏感度 / F1
```

第 2 阶段 `DecisionTreeRunner` 的结果保留 `postprocessing=pending_simplification_and_reconciliation`，
表示它是模型阶段输出，必须进入外层 `SevenStagePipeline` 的第 4–7 阶段。主 CLI 的 `run` 已调用完整流水线。
该标志不能解读为整个主框架仍缺后处理。七阶段各自生成独立 JSON 文件，并汇总为 `result.json`。

人工协调未完成时状态为 `awaiting_reconciliation`，第 7 阶段不评分；未提供参考标签时为
`awaiting_reference`。模型失败时后续阶段记未启动，不能按漏诊评分。参考标签只在模型结束后进入同一套
简化、匹配与协调流程，不能用参考答案改写模型输出。原始诊断集合不自动进入现有 L3 治疗流程。

## 对旧实现的检查与处理

| 原实现 | 与论文不同之处 | 当前处理 |
|---|---|---|
| `diagnosis/decision_tree.py` | 9 个合并症状入口、24 个项目标签；按各标签收集锚点与 guards | 保留为历史代码；默认 L3 不再自动调用 |
| `APICLINICRReasoner` 的节点生成 | 一批多个问题；三值状态及患者证据引用；依赖知识提供者 | 旧后端改名 `LegacyDSM5TreeReasoner`；默认类变成显式委托端口 |
| `criterion_nodes` | 从 MentalKG / 失眠 JSON 展开条件，候选可带未满足标准 | 新框架只执行独立数据包定义的二值问题和分支 |
| `assessment` / temporal gate / confidence 排序 | 标准汇总、病程门和项目排序影响候选收敛 | 新模块不导入这些模块 |
| 首位候选 Verifier | 只核验项目排名第一候选，不是论文逐对比较 | 新模块实现候选全部无序对比较，不经过原 Verifier |
| G1 缓存和既有实验 | 保存历史三值节点与项目策略 | 继续留存，显式旧入口重放；不用于新模块的候选生成 |

`CLINICR` 现在仍接受显式 `reasoner`，其默认 `APICLINICRReasoner` 不再选择任何树。
即使提供旧知识提供者，没有候选生成器也返回生成器未接入错误；HTTP 通道沿用 503。
这是本次解耦后的实际行为。需要保留历史流程的工具已改为显式构造 `LegacyDSM5TreeReasoner`。
原始 MentalKG、失眠数据、历史实验结果和用户已有其他模块改动未在本次覆盖。

## 论文中明确披露的执行约束

- 每个入口、每个节点分别提问，当前请求包含完整病例和当前问题。
- 每次请求新建系统/用户两条消息，不发送前面问题的对话历史。
- 节点只接受 yes 或 no；提示词要求信息不足回答 no。无效输出要重试，不由程序强行转成 no。
- 模型不输出下一节点、诊断 ID、置信度或证据包；程序按配置跳转和产生候选。
- 多入口可以同时选中；数据结构允许同一树多候选、多条来源路径，以及跨树转介。
- 当前 9 类/24 标签入口只选路径，不直接生成诊断；共用目标树去重，未录入目标明确挂起。
- 确切互斥关系应通过有医学来源的分支表达；不能用排名强制只保留一个疾病。
- `Other/Unspecified` 通过 `fallback_diagnoses` 延后到所属树全部路径结束；只有该树没有常规诊断时才采用。
- 候选两两比较可保留两者或其中一个。0/1 个候选不发成对请求。
- 成对响应按当前两个候选的名称精确绑定，仅忽略大小写和首尾空白；原始响应仍完整保存。
  Table 1 的示例本身使用小写，因此小写名称不视作无效诊断。不在此阶段做同义词、近似词或模糊医学匹配。
- 参数固定为 temperature=0、top_p=1、frequency_penalty=0、presence_penalty=0。
- 错误或不可解析时按指数退避重发同一个独立请求，默认直到有效响应；可选最大尝试次数记为工程偏离。
- 所有尝试及原始响应写日志。失败、重试和中断不冒充临床 no，也不成为新增样本。

## 必须明确的重建选择

论文没有公开足够细节支持逐字还原作者程序。代码没有将这些选择隐藏成“论文规定”：

1. **成对请求病例包装**：Table 1 成对模板未直接显示病例插入位置；本实现每次在该模板前加入完整病例，标记为 `prepend_full_case_v1`。
2. **成对结果汇总**：必须显式选择 `all_pairs_union_eliminations_v1`，没有静默默认。固定候选产生顺序，遍历原始候选集合的所有无序对，最后统一应用删除集合。发生循环、全部淘汰时如实记录，不补造胜者。该规则是重建方案，不能称为作者已披露的汇总算法。
3. **分支推进与共享节点**：先完成全部入口筛选，再将所有适用树的起始节点放入统一待处理队列。每次只根据当前节点回答加入其后续节点；不要求一棵树全部完成后再处理另一棵。原图显式转介也加入同一队列。所有适用分支处理完并汇总候选后，才统一两两比较。同一个 `(tree_id,node_id)` 仅判断一次，不同病例和运行之间不缓存。按用户 2026-11-01 的明确要求，当前策略记录为 `screen_all_then_shared_node_frontier_v3`；它是调度选择，不冒充论文规定的并发实现。
4. **医学版本**：Table 1 模板中的 DSM-5-TR 按用户指定替换为 DSM-5；其他请求模式保持论文形式。书籍医学内容和执行框架分别记录来源。
5. **本地来源改编**：当前 19 棵家族树的节点、分支和来源页码见公开版 source_audit.json；原文锚点保存在独立本地研究档案，公开版不含原文摘录。缺少的完整标准见 source_gaps.json。当前包采用 `acyclic`；执行器仍支持其他显式数据包的 `visit_each_node_once` 转介策略。

## 代码职责

| 文件 | 职责 |
|---|---|
| `models.py` | 独立树数据契约、来源和图结构校验、请求及响应结构 |
| `entries.py` | 独立症状入口配置；9/24 由导出数据决定，不依赖旧运行时 |
| `prompts.py` | Table 1 系统、直接诊断、二值问题和成对比较模板 |
| `engine.py` | 筛选、遍历、候选汇总、兜底、逐对比较、重试及事件记录 |
| `pipeline.py` | Figure 1 七阶段主编排，以及人工协调后的离线继续 |
| `postprocess.py` | 附录 C 六项简化、WRatio、七项英文原文协调规则及持久复用 |
| `scoring.py` | 诊断/类别粒度 TP/FP/FN、逐例指标及共同成功配对病例宏平均 |
| `transport.py` | 可选独立 SDK 适配器，不读取项目 LLM 客户端或隐藏修复策略 |
| `source_index.py` | 从用户提供的 2014 PDF 校验 29 个目录标题及印刷/PDF 页码 |
| `__main__.py` | 目录导入、离线配置校验和显式单病例运行入口 |

二值树执行器只依赖 Python 标准库；七阶段主流水线需要 `rapidfuzz`（可选依赖组 `paper`），
目录提取另需 `pypdf`，真实 API 传输另需既有的 `openai` SDK。本次安装验证的 RapidFuzz 版本为 3.14.6。
模块不导入 `layers`、`shared`、MentalKG、24 标签目录或项目临床画像契约。

## 树数据包的格式

只有 `review_status=source_verified` 且通过图校验的显式数据包可运行。
以下是非临床结构示例，不代表原书任何诊断规则：

```json
{
  "id": "engineering-structure-example",
  "standard": "DSM-5",
  "stage": "raw",
  "coverage": "partial",
  "review_status": "source_verified",
  "documents": ["engineering-fixture"],
  "diagnoses": ["Example-A"],
  "refinements": [],
  "trees": [{
    "id": "example-tree",
    "title": "Nonclinical engineering example",
    "screening_question": "Does this fictional example enter this branch?",
    "screening_source": {
      "document": "engineering-fixture", "print_pages": [1], "pdf_pages": [1],
      "note": "Nonclinical engineering example"
    },
    "roots": ["q1"],
    "nodes": {
      "q1": {
        "question": "Is the fictional example condition present?",
        "yes": {"diagnoses": ["Example-A"]},
        "no": {},
        "source": {
          "document": "engineering-fixture", "print_pages": [1], "pdf_pages": [1],
          "note": "Nonclinical engineering example"
        }
      }
    }
  }]
}
```

每个 yes/no 分支可包含：`next`（本树节点 ID 列表）、`diagnoses`、`fallback_diagnoses`、
`referrals`（已录入的目标树 ID 列表）、`pending_referrals`（原图明确指向但尚未录入的树），
以及 `observations`（原图的非诊断终点）。命中未完成转介会返回 `awaiting_tree_content`，
七阶段中模型阶段保持未完成，后续不评分；不会伪装成阴性或将转介名当诊断。原书没有的分支不能凭医学记忆补造。
有多个可共存的判断路径时，显式配置多根节点或分支继续；有来源的互斥关系放在分叉中。
`stage=refined` 要求记录修改日志，含树/节点、分类、修改前后、理由和来源；原图转录不自动算优化版。

## 使用方式

以下命令在 `L1-L5系统` 目录执行。本轮完成原图转录和受控测试，未调用真实模型。

离线校验完成录入的树包：

```powershell
.\.venv\Scripts\python.exe -m paper_dtree inspect
.\.venv\Scripts\python.exe tools/verify_project24_trees_2014.py
```

真实执行需准备一个只含病例原文的 UTF-8 文本文件。默认树包为当前 19 棵本地改编树，使用 `--method dt`；
默认入口同时使用本包的 24 项配置。可用 `--bundle` 和 `--entry-profile` 显式指定其他匹配的数据包。
最初的来源目录清单不能作为树包传入。当前 label_catalog.json 不是后处理词表；七阶段运行仍须显式提供审核过的 `--catalog`，
并区分论文去修饰项后的指标与保留分型的 24 标签指标。
在本地环境中设置 `PAPER_DT_API_KEY` 和需要时的 `PAPER_DT_BASE_URL`。本模块不自动读取现有项目密钥、
模型别名、JSON 强制格式、推理扩展或旧树缓存。下例中的占位符要先替换：

```powershell
.\.venv\Scripts\python.exe -m paper_dtree run `
  --catalog <已核对的诊断目录和简化映射.json> `
  --reconciliation-db <共享的人工协调.sqlite> `
  --case-file <病例原文.txt> --case-id <病例编号> `
  --reference-file <作者参考标签.json> `
  --model <明确模型版本> --method dt `
  --pairwise-policy all_pairs_union_eliminations_v1 `
  --output-dir <新的运行目录>
```

`--method base` 为直接诊断；`dt_raw` 是额外的原图树组；`dt` 必须使用优化树包。
输出保存原始病例、树包及名称表快照、运行配置、每次请求日志、路径、筛选前后候选、七阶段文件及运行状态。
运行目录必须新建，不能覆盖历史运行。模型失败会保留七阶段状态；中断及部分路径保存在 `attempts.jsonl`，不会产出成功评分。

作者参考标签文件与病例分开，格式为：

```json
{"case_id":"与查询一致的编号","diagnoses":["作者诊断原文"],"source_ref":"病例书版本与病例号"}
```

若需人工协调，从 `06_reconcile.json` 的 pending 清单形成审核决定，再显式导入数据库。
之后用 `resume` 在新目录继续第 4–7 阶段，不重新请求模型，也不覆盖原始实验记录：

```powershell
.\.venv\Scripts\python.exe -m paper_dtree import-reviews `
  --catalog <诊断目录.json> --reconciliation-db <共享协调.sqlite> `
  --decisions-file <人工审核决定.json>
.\.venv\Scripts\python.exe -m paper_dtree resume `
  --input-result <原运行目录/result.json> --catalog <相同版本诊断目录.json> `
  --reconciliation-db <共享协调.sqlite> --output-dir <新的继续目录>
```

## 附录 C 的数据与规则

简化按 1–6 顺序执行：移除修饰语与编码、合并神经认知诊断、物质类别泛化、呼吸相关睡眠诊断合并、
按精神诊断类别泛化具体躯体病因，以及逐类别合并 Other Specified / Unspecified。
医学名称和转换目标必须由核对过的映射表提供，不用通用正则表达式猜测哪些词可以删除。
未知字符串原样进入第 5 阶段，再根据分数自动匹配或进入人工协调；不能静默丢弃。

标准目录格式如下。以下名称仅为工程示例，不能作为医学目录运行临床实验：

```json
{
  "id": "fictional-catalog-v1",
  "standard": "DSM-5",
  "review_status": "source_verified",
  "source_ref": "nonclinical-fixture",
  "diagnoses": [{"name":"Example-A","category":"Example-Category","kind":"dsm","source_ref":"fixture"}],
  "simplifications": [{"rule":1,"input":"Example-A, mild","output":"Example-A","source_ref":"fixture"}]
}
```

每项简化包含 `rule`（1–6）、原始 `input`、目标 `output`、`source_ref`。
同名输出每轮合并，但保留所有原名与变更记录。标准目录应包含经过相同简化后的名称及类别；
不能拿项目原来的 24 标签表冒充完整 DSM 标准名称表。

模糊匹配直接使用 `rapidfuzz.fuzz.WRatio(..., processor=None)`，记录实际库版本；API 依据为
[RapidFuzz 官方文档](https://rapidfuzz.github.io/RapidFuzz/Usage/fuzz.html#WRatio)。
严格采用英文附录 C 的“唯一匹配且 >99”：99 分、低分或多个超过阈值者都进入人工协调。
Figure 1 图注中的 `<99` 与附录正文 `>99` 对 99 边界表述不完全一致，此处按附录明确规则处理。

英文附件协调规则共有七项，其中第 6 项为旧 DSM 名称转换，第 7 项为其他未匹配输出保留为 Non-DSM；
用户贴出的译文省略了旧名称转换项。本实现用 `appendix_rule` 明确指向英文条目编号，
不把它冒充补充材料 2 中另有定义的原因编码。

人工审核决定为 JSON 数组，每条记录例如：

```json
[{"input":"待协调的简化后字符串","target":"审核后的标准名称","appendix_rule":3,
  "reviewer":"审核者","reviewed_at":"2026-11-01","reason":"核对依据"}]
```

规则 1/7 必须原样保留非 DSM 名称，记 Non-DSM，并在评分中计为 FP；不能为提升 PPV 删除。
规则 2 必须对应目录中 `kind=z_code` 的条件。其他规则也必须指向显式标准目录。
SQLite 按标准版本、目录版本与输入字符串保存决定，并复用于后续病例和组别；冲突决定不能静默覆盖。

评分同时产生诊断与类别粒度的 TP/FP/FN 列表、PPV、敏感度和 F1，分母为零按论文记 0。
`paired_macro` 按本项目约定仅比较各组共同成功且已评分的病例，失败/待协调/未启动单列；
合法空预测仍计成功并正常产生 FN。宏平均是先逐例算指标再求均值，不将失败当漏诊，也不拿微平均替代。
本轮未增加模型实验，也没有用这套代码生成论文性能结论。

## 现有材料和后续数据工作

用户 PDF 共 339 页，版权页为 2014，ISBN 为 `9781585624621`。
目录中 2.1–2.29 的标题和起始页已逐项通过页面文字校验；本副本 PDF 序号为印刷页码加 17。
完整索引见 [first2014_source_catalog.json](../knowledge/paper_dtree/first2014_source_catalog.json)。

当前 19 棵树按本地 `refined` 记录，说明和来源审计位于 `knowledge/paper_dtree/first2014_project24/`；
2.3 言语示例独立于当前默认树，保留在本地研究档案，不包含在公开版中。
原图组合候选已经在当前家族树中细分判断，范围外原因保留为鉴别观察，不强制映射为范围内标签。
所有本地改编均记录来源；复杂标准仍须依据 DSM-5 对应原文补齐。
仅这本鉴别手册不保证涵盖所有诊断的完整标准。仍缺的完整标准或作者优化细节保持缺项，不能由旧知识库自动补上。
目录里的 29 棵树与论文报告的 28 棵尚未建立作者确认的对应；也不能因为有 29 个标题就声称已复现完整论文树集。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_paper_dtree.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_paper_pipeline.py
```

受控测试使用虚构诊断，验证多入口、yes/no 分支、缺失信息提示、独立请求、指数退避、失败与中断、
同树多候选、转树、兜底、去重来源、逐对比较、循环淘汰和默认 L3 解耦。
它们验证工程行为，不是对真实病例诊断性能的验证。本次没有启动付费对照试验。

历史 18 棵源图版本的测试曾涵盖源图终点、全部 726 个出口、24 入口目标闭合、交错推进、互相转介终止、
条件共病、医学结果保留与源图修正记录。整体回归 924 项通过，排除已说明需系统密钥访问的五项 DPAPI 测试。
当前整体回归为 846 项通过（明确排除需系统密钥访问的 `test_mentalbench_eval_protocol.py` 五项）；
接口检查仍为 45/45 通过，schema-reference 一致性检查通过。本次未改 L1–L5 的 Pydantic 或 TypedDict。

后续 2.3 言语问题树录入轮次：新增本树专项 30 项通过；树框架、七阶段、本树及 schema 一致性检查
合计 93 项通过。当前整体回归更新为 876 项通过，仍明确排除上述五项系统密钥测试。

后续 9 类/24 入口配置轮次：入口、树框架、七阶段、本树来源和 schema 一致性检查合计 110 项通过；
整体回归更新为 893 项通过，仍明确排除上述五项系统密钥测试。未启动付费模型实验。

前一轮（只包含模型阶段）验证结果：新框架及相关旧树回归共 182 项通过，接口检查 45/45 通过。
全量测试在沙箱内为 818 项通过、5 项 Windows DPAPI 密钥访问失败；
该 DPAPI 测试文件在沙箱外使用全新临时目录重跑为 5/5 通过。没有修改加密实现或跳过这些测试。
