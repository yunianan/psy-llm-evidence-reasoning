# 心理诊疗 LLM — 五层认知架构

基于五层认知架构的心理评估与治疗对话项目，将信息采集、诊断推理、治疗语言生成和输出审核分开。

各层当前 Schema、字段类型、必填项、默认值和约束见 [L1–L5 Schema 参考](docs/layer-schema-reference.md)。该文档从实际代码生成，Schema 改动后必须同步生成，并由自动化测试检查一致性。

`main.py` 的诊断阶段使用 [WHO ICD-10 十三入口决策树](icd10_tree/catalog.py)。
编排器将 L2 确认后的完整画像与患者原话交给 ICD-10 树，候选携带编码、证据支持度、原文出处和患者证据，进入人工复核。L2 使用基于 WHO ICD-10 本地条目的症状词表整理患者原话，症状标签用于画像，诊断判据由 L3 单独核对。树引擎随仓库提供，出处记录保留参考扫描件的页码和校验值。WHO 出版物见 [1993 年研究诊断标准](https://www.who.int/publications/i/item/9241544554) 和 [1992 年临床描述与诊断指南](https://www.who.int/publications-detail-redirect/9241544228)。

旧流程的候选处理与终验规则见 [CLINICR 内部诊断流程](docs/clinicr-reuse-notes.md)。

**本地完整系统入口是根目录 `main.py`。** 在 VS Code 中运行它即可输入自然语言，不需要分别启动五层服务，也不需要手动填写患者画像或诊断 JSON。

## 快速开始：在 VS Code 中聊天

### 1. 选择 Python 环境

需要 Python **3.11 或更高版本**。在 VS Code 中打开项目文件夹，通过命令面板的 **Python: Select Interpreter / Python: 选择解释器** 选择项目 `.venv\Scripts\python.exe`。

若刚克隆仓库、尚无 `.venv`，在项目根目录的 PowerShell 终端执行（`python` 应指向已安装的 Python 3.11+）：

```powershell
python --version

python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -e '.[dev]'
```

`.venv` 不随 Git 提交，新电脑需要重新创建。已有环境和依赖时不必重复安装。

### 2. 运行入口

打开 `main.py`，右键选择 **在终端中运行 Python 文件**。不要使用不支持交互输入的“输出”窗口。

也可以在项目根目录执行：

```powershell
& '.\.venv\Scripts\python.exe' main.py
```

首次运行若未配置共享连接，会提示选择平台：`1` 为 DeepSeek 官方（回车默认），`2` 为阿里云百炼。选择阿里后，依次粘贴控制台的 OpenAI 兼容 `base_url`、模型 ID `model` 和 API Key。已有 `.env` 或环境变量配置时直接使用配置，不再询问平台。

缺少密钥时会提示明文输入，按 Enter 提交。请勿录屏或分享包含密钥的截图。默认五层沿用同一平台连接和模型，各层独立配置仍可覆盖默认值。

入口会调用真实 API，消息会发送到配置的模型服务，并可能产生费用。建议先用虚构案例体验；不要将密钥提交到仓库。

### 3. 输入自然语言

出现 `你：` 后直接输入，例如：

> 最近工作压力比较大，晚上经常睡不好，白天也没有什么精力，我想先聊聊睡眠的问题。

根据系统提问继续回答：选择当前重点、描述持续时间和功能影响、同意或拒绝量表、按题目选项作答、表达目标和改变动机。画像整理完成后，可回复“信息准确，可以继续”“需要修改”或“暂停”。

终端显示系统回复、患者确认卡，以及单独标注的开发调试路由。内部路由标签不是治疗师对患者的阶段播报。

| 终端命令 | 作用 |
|---|---|
| `/help` | 查看帮助 |
| `/profile` | 显示开发者画像 HTML 路径 |
| `/resume` | 仅恢复患者主动暂停，不能解除人工复核 |
| `/exit` | 退出当前终端会话 |

也可按 `Ctrl+C` 退出。当前会话保存在内存中，退出程序后不会自动恢复。

## API 配置

`main.py` 自动读取**项目根目录**的 `.env`，环境变量优先于 `.env`；未提供密钥时再进行终端明文输入。终端输入的密钥只写入当前进程环境，不保存到文件；程序不会额外打印已读取的密钥。

如需保存配置，可参照 `.env.example` 手动创建 `.env`，不要覆盖已有配置。以下是默认服务的配置格式：

```dotenv
DEEPSEEK_API_KEY=你的共享API密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com

L3_REASONER_MODEL=deepseek-chat
L3_VERIFIER_MODEL=deepseek-chat
L3_INSTRUCTION_MODEL=deepseek-chat

L4_LLM_PROVIDER=openai_compat
L4_LLM_BASE_URL=https://api.deepseek.com
L4_LLM_API_KEY=
L4_LLM_MODEL=deepseek-chat
L4_LLM_TEMPERATURE=0.7
L4_REVIEW_MODE=l5_local
L4_MAX_REWRITE_ATTEMPTS=3
```

| 配置 | 用途 |
|---|---|
| `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL` | 通用平台配置；适用于阿里云百炼。密钥和地址优先于旧 `DEEPSEEK_*` 配置，模型默认 `deepseek-chat` |
| `L1_LLM_TIMEOUT_SECONDS` | L1语义安全分类使用内部流式接收，各次调用超时默认 `60` 秒，允许 `1–300` |
| `L1_BSSA_TIMEOUT_SECONDS` | L1 BSSA 访谈提取的单次 API 超时；未设置时沿用 `L1_LLM_TIMEOUT_SECONDS` |
| `L1_SUPPORT_TIMEOUT_SECONDS` | L1 个性化安全支持回应的单次 API 超时，默认 `12` 秒；失败时使用审计过的保底文本 |
| `L2_LLM_TIMEOUT_SECONDS` | L2 语义分析和 MI 回复各次 API 调用的超时秒数，默认 `60`，允许 `1–300` |
| `L3_INSTRUCTION_TIMEOUT_SECONDS` | L3 每轮动态治疗指令模型的调用超时，默认 `60` 秒，允许 `1–300` |
| `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL` | L1、L2、L3、L5 共用的 API 连接 |
| `ICD10_API_KEY`、`ICD10_BASE_URL`、`ICD10_MODEL` | 主会话 ICD-10 节点核对；缺省沿用共享配置，独立地址需提供对应密钥 |
| `L3_INSTRUCTION_MODEL` | 每轮治疗指令模型；未设置时沿用共享模型配置 |
| `L4_LLM_*` | L4 回复生成的独立 API 配置 |
| `L4_REVIEW_MODE=l5_local` | 本地 L5 适配器调用真实审核 API，**不是本地离线模型** |
| `L5_LLM_TIMEOUT_SECONDS=60` | L5 审核 API 的单次超时，允许范围 1–300 秒 |
| `L4_MAX_REWRITE_ATTEMPTS=3` | 默认最多 3 次生成审核：首次生成 + 2 次重写 |

在 `main.py` 中，若 L4 密钥为空且两组 API 地址相同（忽略末尾 `/`），会复用共享密钥；地址不同时会单独询问 L4 密钥。已有 L4 密钥不会被覆盖。

本地终端入口要求 `openai_compat` 和 `l5_local`，不接受 Mock 模式。自动化测试通过显式注入受控模型运行。

### Embedding 向量接口

独立客户端 `shared/embedding_client.py` 使用百炼 `text-embedding-v4`，默认 **1024 维**，提供 `embed_query(text)` 和 `embed_documents(texts)`。批量输入自动按每批最多 10 条调用，并检查返回数量、索引与维度。当前客户端供后续检索使用，尚未接入 L3 图谱或病例检索流程。

在本地 `.env` 中配置：

```dotenv
EMBEDDING_API_KEY=你的百炼API密钥
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSIONS=1024
```

默认地址对应北京地域。也可使用控制台对应业务空间的 OpenAI 兼容 `base_url`，不要包含末尾 `/embeddings`；地域必须与密钥匹配。`EMBEDDING_API_KEY` 未填写时可使用 `DASHSCOPE_API_KEY`，聊天模型密钥由原有配置管理。接口参数和地址见[百炼官方文档](https://help.aliyun.com/zh/model-studio/embedding-interfaces-compatible-with-openai)。

填写密钥后，在项目根目录检查真实接口（只发送一条固定测试文本，输出模型名和维度）：

```powershell
& '.\.venv\Scripts\python.exe' tools/check_embedding.py
```

成功输出 `{"ok": true, "model": "text-embedding-v4", "dimensions": 1024}`。应用入口已加载 `.env`；其他独立脚本调用客户端前也需加载配置。调用示例：

```python
from shared.embedding_client import EmbeddingClient

with EmbeddingClient() as client:
    vector = client.embed_query("最近入睡困难，白天疲倦")
    vectors = client.embed_documents(["第一段文本", "第二段文本"])
```

### 使用阿里平台的 DeepSeek

可以在无共享连接配置时运行 `main.py` 并选择 `2`，也可以在 `.env` 中填写以下三项（把示意文字替换为控制台实际值）：

```dotenv
LLM_API_KEY=你的阿里API密钥
LLM_BASE_URL=控制台提供的OpenAI兼容base_url
LLM_MODEL=控制台实际模型ID
```

`base_url` 必须与密钥所在地域、业务空间匹配；不要粘贴末尾带 `/chat/completions` 的完整请求地址，也不要保留 `{WorkspaceId}` 等占位符。模型 ID 使用控制台可用的名称，不要把 DeepSeek 官方的 `deepseek-chat` 直接套到阿里平台。接口说明见[阿里官方 DeepSeek 文档](https://help.aliyun.com/zh/model-studio/deepseek-api)。本项目使用 Chat Completions 接口，所选模型需支持该接口及现有文本/JSON 输出要求。

如果旧配置中显式写有 `L3_REASONER_MODEL`、`L3_VERIFIER_MODEL`、`L3_INSTRUCTION_MODEL`、`L4_LLM_BASE_URL`、`L4_LLM_MODEL`，请删除不再需要的覆盖项或改成阿里对应值，否则它们仍然优先。L4 密钥留空时，终端入口仅在地址相同的情况下复用共享密钥；已有 L4 密钥需自行更新。独立启动 L4 HTTP 服务时仍需显式设置 `L4_LLM_API_KEY`。

平台接入不会绕过 L1 安全阻断、L3 证据校验或 L5 审核；模型调用或校验失败仍按原流程处理。

L2 语义分析或回复生成的 API、JSON 解析、模型输出结构校验失败时，当前会话进入 `STOPPED`，终端提示模型不可用并退出，不再读取患者输入，失败这一轮不提交到 L2 对话和画像。普通访谈和画像确认阶段都执行此规则。

L2 的自然语言语义分析使用内部流式接收，服务端片段先在后端拼接，得到完整 JSON 并通过 Schema 校验后才写入画像；不会把未完成或未经校验的片段展示给患者。L2 的语义分析和 MI 回复调用默认各等待 60 秒，可通过 `L2_LLM_TIMEOUT_SECONDS` 调整；MI 回复保持非流式。诊断完成后的每轮动态治疗指令通过 `L3_INSTRUCTION_TIMEOUT_SECONDS` 设置，默认 60 秒。L5 审核使用 `L5_LLM_TIMEOUT_SECONDS`，默认也是 60 秒；未显式覆盖的其他共享调用仍默认 10 秒。排查时查看终端的 `[开发调试]` 事件：`L2_OUTPUT_REPAIR` 表示首次输出格式不合格并正在自动修复，`L2_MODEL_OUTPUT_INVALID` 表示修复后仍不合格，`L2_MODEL_UNAVAILABLE` 才表示 API 请求失败；`L3_INSTRUCTION_API_ERROR` 表示动态治疗指令 API 调用失败，`L3_INSTRUCTION_MODEL_OUTPUT_INVALID` 表示 API 已返回但输出解析或 schema 校验失败。`stage` 指明具体处理位置，同时显示异常类型和可用的 HTTP 状态码；不会输出密钥、患者原话、模型原始输出或完整异常内容。这些诊断信息只写入日志输出，不放入患者回复，也不自动保存日志文件。

L2 语义提取提示中的完整输出约束直接由 `SemanticAnalysis.model_json_schema()` 生成，不再维护一份容易漂移的手写 Schema。提示额外明确 `possible_scale_domains` 的六个精确枚举、`structured_fields` 的 `field/value/evidence_quote` 对象结构，以及 `clarification_needed` 的字符串数组结构。首次 JSON 解析或 Schema 校验失败时，不提交当前输出，而是要求同一模型只进行一次结构修复，并记录 `L2_OUTPUT_REPAIR` 调试事件；修复时不能新增临床事实，无法绑定患者原话的候选必须删除。第二次仍不合格时以 `L2_MODEL_OUTPUT_INVALID` 终止会话，并与 API 不可用错误区分。当前未启用阿里原生 `response_format=json_schema`，以避免为了结构化输出而隐式关闭模型思考模式；本地 Pydantic 仍执行最终严格校验。

## 系统如何运行

顶层编排器是 `orchestration/orchestrator.py` 中的 `FiveLayerOrchestrator`；`main.py` 只负责配置与终端输入输出。

1. **安全入口**：每条可处理消息先进入 L1。LLM 仅识别患者本人自杀风险语义和越狱风险，并输出带原文证据的结构化事实；规则按“自杀安全优先、越狱其次、正常放行最后”路由。存在自杀风险或无法排除风险时，由规则启动 NIMH ASQ；ASQ 非急性阳性或拒答后进入成人门诊 BSSA。BSSA 的自然访谈和事实提取由 API 完成，急性条件、处置路由、普通疗程暂停和 L2 放行均由规则与守卫决定。急性危机及需要进一步专业评估时不关闭界面，而是暂停普通疗程并保留安全转介页面。API错误与模型结构错误分别返回 `L1_API_ERROR` 和 `L1_MODEL_OUTPUT_INVALID`，都不放行到 L2。
2. **L2 评估**：入口守卫验证本轮安全通行证；API 负责自然语言理解、语义提取和 MI 回复，规则负责证据记录、量表许可与计分、信息充分度和阶段转换。
3. **画像确认**：信息充分后展示患者确认卡。修改则继续采集，暂停则停止自动推进；确认并通过 L2 出口守卫后进入 L3。
4. **L3 诊断与复核**。ICD-10 树核对患者原话和节点判据，缺失时向 L2 请求补充。树完成后提交候选、证据与支持度供临床复核。经复核批准的诊断和治疗计划通过 L3 校验后，生成 L4 会话级治疗指令。
5. **治疗轮次**：通过 L3→L4 守卫后建立治疗会话。之后每轮根据新消息、历史和静态指令调用 L3 指令 API，经过规则守卫后交给 L4 生成回复，再由 L5 审核。

不是每条消息都完整跑一次五层：评估期间主要运行 L1/L2；治疗期间运行 L1、L3 本轮指令、L4/L5，不会每轮重新诊断和匹配疗法。

> 临床边界：本仓库是研发原型，不等同于医疗机构完成 NPSG.15.01.01 合规。ASQ 中文题目是项目翻译，不是 NIMH 官方验证中文版；NIMH 的成人门诊 BSSA 面向具备资质的临床人员，系统给出的处置不能替代专业判断、当地急救流程或人工交接。

### 阶段与安全回环

- **L2 MI 阶段**：参与 → 聚焦 → 唤出 → 计划。由信息条件推进，不按固定轮数推进，不向患者播报内部阶段名称。
- **核心量表**：PHQ-9、GAD-7、PCL-5、Y-BOCS、ISI、MDQ。相关领域由 API 提议；规则决定许可、题号、分值范围和计分。题干与选项按规则保留，不由模型自由改写。
- **ASQ 插入点**：首条自然语言先完成 L1 语义分流；无即时风险时进入 L2 参与阶段。L2 从参与平滑转入聚焦时，编排器启动一次基线 ASQ，并暂存 L2 已生成的下一问；筛查完成后恢复该问题，避免丢失对话上下文。
- **L1→L2 放行**：只有 `decision_path=normal`、`execution.route=l2`、出口守卫通过且 `l2_handoff.ordinary_care_allowed=true` 时才能进入 L2。ASQ/BSSA 进行中、安全暂停、急性危机和越狱消息均由 L1 消费，不交给 L2。
- **L2→L3共享输入契约**：L2 判定画像充分、L2→L3 守卫和 L3 诊断前置检查共用同一组最低字段与证据要求。除病程和功能影响外，还必须取得有患者原话证据的躁狂/轻躁狂史、物质/药物/躯体疾病影响及精神病性症状筛查；缺失时留在或返回 L2 平滑追问，不启动诊断树。
- **回环 B**：ICD-10 节点缺少可判定的证据时，编排器经 L2 提问。患者回答后按同一节点和证据继续判断，仍无法解析则转人工复核。
- **回环 A**：L4 本地校验或 L5 审核不通过时，携带修复指令重写；次数耗尽后进入 `MANUAL_REVIEW`，阻止自动治疗及外部重写。
- **人工复核等待**：仅暂停下游生成，新自然语言消息仍经过 L1。当前没有真人审核后台，也不会自动恢复。已经终止或患者主动暂停的会话，则先返回对应状态提示。

L4 的阶段包括共情、反馈、技术执行和巩固。下一阶段由 L3 动态指令与规则守卫决定，不依赖 L4 模型自行宣称“阶段完成”。四段治疗指令中的 L1–L4 是指令名称，不是四个系统层本身。

## 患者确认卡与开发者画像

两者由 `patient_presentation/` 单独生成，不与编排器放在同一个文件夹。

- **患者确认卡**：显示主要困扰、持续时间、功能影响、患者目标和量表摘要，供患者确认、修正或暂停，不作为最终诊断结论。
- **开发者画像**：位于 `artifacts/developer_profiles/<session_id>.html`。启动时显示路径，也可输入 `/profile` 查看；用浏览器打开后，每轮对话后刷新。

开发者画像包含编排状态、L1 安全结果、L2 画像与证据、L3 候选与复核材料、诊断报告和放行检查。它可能包含患者原话和内部评分，不应展示给患者或公开分享。

HTML 仅是调试视图，不是可恢复的会话存档。L5 连续审核失败的结果保存在内部会话 `pending_l5_review`，不随患者输出返回。

## 仓库结构

```
├── main.py                  # 完整系统的本地交互终端入口
├── .env.example             # API 配置示例；真实密钥放本机环境或 .env
├── pyproject.toml           # Python 版本、依赖和测试配置
├── layers/                  # L1-L5 各层实现
│   ├── layer1/              # L1 安全层（LLM语义分流 + 规则ASQ + LLM辅助BSSA + 规则处置/守卫）
│   ├── layer2/              # L2 MI评估、量表、患者画像与自然语言适配
│   ├── layer3/              # L3 诊断推理层（ICD-10 复核报告、放行检查与治疗指令）
│   │   ├── diagnosis/       # ICD-10 复核报告和证据校验
│   │   ├── instructions/   # 会话级四段指令与每轮动态指令
│   │   └── output_validator.py # L3 整体输出校验
│   ├── layer4/              # L4 治疗对话层（状态机/回环A/prompt组装/FastAPI）
│   └── layer5/              # L5 输出安全审核层（上下文规则生成 + 一次调用三检）
├── orchestration/           # 五层顶层编排、全局会话与患者确认路由
├── patient_presentation/    # 开发者画像 HTML 与患者脱敏确认卡
├── shared/                  # 公共模块
│   ├── llm_client.py        # 共享 OpenAI 兼容客户端（DeepSeek 官方 / 阿里等）
│   ├── prompts/             # 共享 prompt 模板
│   └── schemas/            # 共享结构及四段治疗指令模型
├── artifacts/developer_profiles/ # 运行时生成的开发者画像
├── examples/                # 各层独立示例，不是完整系统入口
├── tools/
│   └── verify_interfaces.py # 层间接口契约验证，使用受控 API 替身
├── tests/                   # 终端入口、各层单元测试和五层集成测试
└── docs/                    # 设计文档与实现计划
```

## 层间接口与输出检查

| 接口 | 格式 | 位置 |
|---|---|---|
| 接口一 L1→L2 | SafetyPassport + 会话/消息绑定 + 执行回执 | `layers/layer2/guards/entrance.py` |
| 接口二 L2→L3 | ClinicalIntake + 患者确认 + 证据守卫 | `layers/layer2/adapters.py`、`layers/layer2/guards/exit.py` |
| 接口三 L3→L4 | DiagnosticReport、四段治疗指令及 L3 校验凭证 | `layers/layer3/diagnosis/icd10_report.py`、`layers/layer4/validation/l3_to_l4.py` |
| 接口四 L4+L3→L5 | ContextualSafetyContext → L5 审核输出 | `layers/layer5/reviewer.py`、`layers/layer4/schemas/l5_context.py` |

各层间接口使用对应目录下的 Pydantic schema；`layers/layer4/review/l5_adapter.py` 将真实 L5 包装为 Reviewer 协议，回环A在 L4 内编排，默认最多 3 次生成审核（首次生成 + 2 次重写）。

L5 审核上下文包含患者本轮原话、此前的治疗对话、完整四段治疗指令（含插件依据、禁忌及本轮指导）与 L1 风险标记。缺少原话、历史字段或本轮指令时拒绝进入审核。三项 `pass` 与非空失败原因/修复指令并存时，整次审核按不通过处理。

L5 当前审核的是 **L4 治疗回复**，并非系统全部输出。L2 MI 回复由 L2 本地规则校验，危机提示、确认卡及部分编排提示走各自路径。已有证据字段及绑定检查不等同于完整端到端语义追溯或临床正确性保证。

### ICD-10 人工复核与 L4 放行

主会话的树运行完成后进入 `MANUAL_REVIEW`。可信应用调用方可从
`pipeline.get_session_snapshot(session_id).l3_raw["model_result"]` 读取候选及本次树会话 ID，
再通过 `pipeline.submit_icd10_review(session_id, review)` 提交 `ICD10ClinicalReview`。
复核资料包含 `tree_session_id`、`reviewer_id`、`primary_code`、`secondary_codes`、
`rationale`、`severity_level` 和经过复核的 `therapy`。治疗计划使用项目已有的可执行疗法与插件，
并提供严重程度分组、策略指令和适用的禁忌。患者终端消息不具备复核权限。

ICD-10 的节点模型在同一次判断中提供证据支持度。树按使当前路径成立的判断汇总候选支持度，
放行要求主诊断与共病候选的支持度均至少为 0.5。该分数用于工程放行，未作临床概率校准。
L3 还检查当前树运行、WHO 条文和引文、复核结果、治疗计划、插件、禁忌及治疗指令。
报告通过 `L3_OUTPUT_ICD10_V1` 后，L4 验证凭证与报告指纹再开始会话。

## 自动化测试（受控模型，不调用外部 API）

在项目根目录的 PowerShell 终端运行：

```powershell
& '.\.venv\Scripts\python.exe' -m pip install -e '.[dev]'
& '.\.venv\Scripts\python.exe' -m pytest -q
& '.\.venv\Scripts\python.exe' -m pytest tests/test_main.py -q
& '.\.venv\Scripts\python.exe' tools/verify_interfaces.py
```

`tests/test_main.py` 覆盖配置优先级、终端明文密钥输入、不同端点密钥隔离、自然语言输入、确认卡显示、暂停恢复、人工复核限制、退出和受控编排集成。测试数量以实际运行结果为准。

自动化测试不代表真实 API 的稳定性或临床正确性。`tests/test_icd10_orchestrator.py` 覆盖 ICD-10 追问、候选、临床复核与放行。完整体验运行 `main.py`。

## 独立服务与后续前端接入

仓库提供独立的 L4 FastAPI 服务。L4 CLI 使用经复核的 ICD-10 示例报告演示治疗对话。

若需要单独启动 L4 服务，在项目根目录运行：

```powershell
& '.\.venv\Scripts\python.exe' -m uvicorn layers.layer4.api.server:app --env-file .env --port 8000
```

此方式需事先创建 `.env`，并显式填写 `L4_LLM_API_KEY`；**不会执行 `main.py` 的密钥询问或复用逻辑**。已通过终端环境变量配置时可省略 `--env-file .env`。接口文档位于 `http://127.0.0.1:8000/docs`。

当前没有统一五层 HTTP 聊天接口或前端页面。后续前端应调用统一后端 API，由该 API 复用 `FiveLayerOrchestrator`；不应调用终端输入循环，也不需要逐个调用各层文件。

## 常见启动问题

| 现象 | 检查方式 |
|---|---|
| 提示缺少模块 | 确认 VS Code 选中项目 `.venv`，并用同一个解释器安装依赖 |
| 找不到 Python 或版本低于 3.11 | 安装或选择满足要求的 Python，再创建项目环境 |
| 不能输入密钥或消息 | 使用“在终端中运行 Python 文件”，点击终端后输入；不要在输出面板操作 |
| 配置了 `.env` 仍提示 Mock 模式 | 检查已有环境变量；其优先级高于 `.env` |
| 提示安全评估模型不可用 | L1 已停止后续处理，检查共享 API 密钥、地址、服务可用性和网络 |
| 进入 `MANUAL_REVIEW` | 自动治疗已暂停，`/resume` 不会绕过人工复核 |
| 画像没有更新 | 用 `/profile` 确认路径并刷新浏览器；若未生成，检查输出目录写入权限 |
| 退出后找不到原会话 | 当前仅内存会话，HTML 不能恢复对话状态 |

### L2 MI 回复的本机格式调试

默认不记录模型输出。在 `.env` 中设置 `L2_DEBUG_OUTPUT=true` 并重新启动程序即可开启；也可以仅在当前 PowerShell 中设置：

```powershell
$env:L2_DEBUG_OUTPUT = "true"
& '.\.venv\Scripts\python.exe' live_layer_test.py
```

每次 L2 MI 回复调用会在 `artifacts/debug/l2/` 保存一份 `.meta.json`，记录调用阶段、结束原因 `finish_reason`、文本长度、JSON 解析及输出校验状态。仅发生 `JSONDecodeError` 时保存同名 `.response.txt`（原始回复）；元数据同时记录错误行号、列号和位置。这些位置以去除外层空白和完整 Markdown 代码围栏后的解析文本为准。`finish_reason=length` 是响应被长度限制截断的线索；`stop` 本身不保证 JSON 合法。

终端只打印元数据文件路径，不打印原始回复。该功能不保存请求提示词、请求头、API 密钥或模型独立的思考字段；失败回复本身可能包含患者信息，应仅在本机查看。调试目录已排除 Git。排查后设为 `false`、重新启动，并手动删除不再需要的调试文件。写盘失败不会改变原来的模型调用结果。此开关仅记录 L2 MI 回复，不记录语义提取，也不自动修复 JSON。
