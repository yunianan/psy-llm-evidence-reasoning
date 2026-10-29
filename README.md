# Patient Agent（模拟患者 · 自动多轮测试）

当前诊断标签与主系统统一为 MentalKG 原始 D001–D023＋Insomnia（24 个）。病例中明确的旧代码在读取时转换，历史通用 bipolar 在模拟候选中保留为 D008/D009 两个待鉴别标签，不擅自指定亚型。内置 mock 只用于流程测试；其临床判断不是 MentalKG 标准验证结果。原病例和历史运行记录保留。


心理诊疗系统里负责「扮演患者」的那一层。它按病例脚本稳定地说话、答量表、确认画像、
推进疗程，通过**完整系统 API** 与顶层编排器交互，把整场会话逐轮记下来，再对照病例的
预期路径自动判分。

它是测试基建，不是临床组件：**不自己算风险分级、不下诊断、不算量表分、不做审核**，
这些都由被它调用的系统去做。

> 硬约束（来自任务说明第七节自检清单）
> 1. **不添加病例外事实**——患者说的每句话都必须能在病例里找到出处，越界由 PA4 守卫拦下。
> 2. **测试能按条件停止**——危机、人工复核、目标事件覆盖、违规、轮数上限都能让本轮收尾。

## CACTUS 病例扩充（2026-10-29）

默认 `imported_scenarios/` 已合入 CACTUS 150 例，加上原有 DB 4 例共 **154 例**；内置 `scenarios/` 的 6 例保持原样。当前 MentalKG 诊断标签适配保留，CACTUS 工具复用当前 `tools/db_import.py` 与 Schema。

- 导入入口：`tools/cactus_import.py`；中间记录与提取依据分别在 `tools/cactus_normalized_records.json`、`tools/cactus_import_details.json`。
- 原始 450 条记录按相同 intake 合并为 150 例，三种态度仅保留为溯源信息。病例均带 `needs_review`：症状为启发式提取，量表、阴性鉴别、风险否认与治疗轨迹含模拟设定，不能当作原始数据明确提供的事实。
- 本次使用当前代码验证：既有测试 **264 passed**，导入工具小样本重跑 **3/3 通过**，默认全库离线回放 **154/154 通过**。这是规则患者与 Mock 系统的流程验证，不代表真实模型、临床效果或 CBT 技术评分验证。
- 本次回放报告：[runs/cactus-merge-20261029/report.md](runs/cactus-merge-20261029/report.md)。原包的导入说明及本次合入说明见 [CACTUS导入报告.md](CACTUS导入报告.md)。

可在独立目录重跑少量导入，避免覆盖默认病例库或完整中间文件：

```bash
python tools/cactus_import.py --limit 3 --out runs/cactus-smoke/cases --records-out runs/cactus-smoke-records.json --report-out runs/cactus-smoke-details.json
python -m patient_agent run-all --scenario-dir imported_scenarios --llm-mode offline --system-api mock --out runs/cactus-validation
```

## 快速开始

不需要任何 API key，不出网。在 `patient agent/` 目录下：

```bash
# 跑测试（全部离线/受控，不请求真实 API）
python -m pytest

# 看病例库里有什么
python -m patient_agent list

# 跑单个病例，打印逐轮对话 + 评测明细
python -m patient_agent run S03_explicit_crisis

# 全库回放，出报告（JSON + Markdown）
python -m patient_agent run-all --out runs/latest
```

退出码：`0` 全部通过 / `1` 有病例未通过评测 / `2` 用法或配置错误。

想连真实模型或真实编排器时再显式打开：

```bash
python -m patient_agent run-all --llm-mode auto            # 有 OPENAI_API_KEY 就用真模型
python -m patient_agent run-all --system-api http \
       --system-api-base-url http://127.0.0.1:8000
```

配置项见 `.env.example`（复制成 `.env` 后填写；全部留空也能跑）。

## 目录结构

```
patient agent/
├── patient_agent/            各层代码
│   ├── schemas.py            PA1  病例 Schema（七组字段 + expected/stop）
│   ├── library.py            PA2  版本化病例库（index.json + <id>/<版本>.json）
│   ├── generator.py          PA3  患者生成器（决定这一轮说什么）
│   ├── guard.py              PA4  一致性守卫（这句话病例允不允许）
│   ├── runner.py             PA5  场景运行器（唯一碰系统 API 的地方）
│   ├── stopping.py           PA6  停止条件
│   ├── evaluator.py          PA7  评测器（对照 expected 判分）
│   ├── report.py             PA7  批量报告（JSON + Markdown）
│   ├── recorder.py           逐轮记录（评测的唯一数据来源）
│   ├── factory.py            装配（CLI 与测试共用同一条装配路径）
│   ├── cli.py / __main__.py  命令行入口
│   ├── config.py             环境变量 / .env → Settings
│   ├── errors.py             稳定错误码
│   ├── lexicon.py            与病例无关的共享词表
│   ├── client/               完整系统 API：契约 + Mock 五层桩 + HTTP 实现
│   └── llm/                  LLM 适配器：离线伪模型 / OpenAI 兼容
├── scenarios/                病例库（6 例，覆盖 5 条验收场景）
├── tests/                    181 个用例
├── .env.example
├── pyproject.toml
└── pytest.ini
```

## 与系统对接：只用这八个接口

契约在 `patient_agent/client/base.py` 的抽象类 `SystemApiClient` 上，与任务说明第五节一致。
运行器只依赖这个抽象，所以「离线跑 Mock」和「连真编排器」是同一条代码路径。

| # | 接口 | 用途 |
| --- | --- | --- |
| 1 | `create_session` | 建会话，拿 `session_id`、初始路由与患者可见的初始提示 |
| 2 | `send_message` | 把患者消息交给编排器，拿回复 / 等待类型 / 路由 / 事件 / 错误码 |
| 3 | `confirm_profile` | 提交画像确认决定（`continue` / `modify` / `pause`） |
| 4 | `submit_scale` | 提交当前许可量表题目的答案 |
| 5 | `set_pause` | 患者主动暂停 / 恢复 |
| 6 | `get_dev_state` | 读开发者可见的五层流程快照 |
| 7 | `get_episode_summary` | 读 L5 疗程总结（未结束时为 `None`） |
| 8 | `get_session_history` | 读脱敏历史疗程列表 |

两个实现：

* `client/mock_system.py` — 内置 **简化五层状态机**，供离线测试与演示。
  它按「L1 风险 → L2 访谈/量表/画像 → L3 候选与排除 → L4 治疗 → L5 复盘」推进，
  **不代表真实临床逻辑**，只保证接口形态与分支行为可被自动测试驱动。
* `client/http_client.py` — 真实编排器的 HTTP 客户端，只在 `SYSTEM_API_MODE=http` 时加载。

响应里带 `SAFETY_HOLD` / `HUMAN_REVIEW` 属于**设计内的专门路径**（安全转介、人工复核），
响应本身带着有效的患者可见回复，不该当异常抛掉：`base.NON_FATAL_CODES` 列了它们，
运行器照常记这一轮，再让 PA6 以「系统终止」收尾。

## 病例库

```
scenarios/index.json               # 有哪些病例、各有几个版本、最新是哪个
scenarios/S01_low_risk_depression/1.0.0.json
```

版本落到文件名上：改一句台词就可能让「不迎合」验证结果变化，测试结论必须能追溯
「当时用的是哪个版本」。改病例时新增版本文件，而不是就地覆盖。

一个病例 = 七组临床字段 + 两组测试字段：

* 临床七组：基本背景 / 临床事实 / 心理特征 / 风险脚本 / 披露策略 / 行为参数 / 变化轨迹；
* `expected`：期望观测到的事件、期望走过的层、期望终止原因、要求被排除的候选、轮数下限；
* `stop`：`max_turns`、是否在危机 / 人工复核 / 违规时停止、覆盖哪些事件即可停。

所有「患者会说的话」都写在病例的 `utterance` 模板里，生成器只允许照抄或改写。
「不添加病例外事实」因此有可检查的边界，而不是一句口号。

### 内置病例与验收场景

| 病例 | 覆盖 | 终止原因 | 轮数 |
| --- | --- | --- | --- |
| `S01_low_risk_depression` | 验收 1（画像确认 → L3）、轨迹「改善」 | `completed_improved` | 22 |
| `S02_insufficient_info` | 验收 2（L3 信息不足回退 L2 补字段）+ L1 模糊风险「记录但继续」 | 未终止（目标事件覆盖后由 PA6 停） | 21 |
| `S03_explicit_crisis` | 验收 3（危机只走 L1 安全路径） | `crisis_referral` | 1 |
| `S04_bipolar_exclusion` | 验收 4（易激惹 + 睡眠减少、无躁狂史 → 双相被排除；患者不迎合） | `completed_improved` | 23 |
| `S05_comorbid_anxiety_depression` | 轨迹「无变化」+ 暂停/恢复链路 + PHQ-9/GAD-7/ISI 三量表 | `completed_no_change` | 34 |
| `S06_treatment_dropout` | 轨迹「退出」+ 画像先改后确认 | `dropout` | 19 |

`run-all` 的实测报告（`runs/latest/report.md`）节选：

```
## 路径覆盖（每层覆盖场景数）
| L1 | 6 |   | L2 | 5 |   | L3 | 5 |   | L4 | 5 |   | L5 | 4 |

## 错误类型
- `SAFETY_HOLD` × 1          ← 设计内的安全路径信号，非故障
## 事实一致性
- 未发现一致性违规
## 终止原因分布
- completed_improved × 2 / completed_no_change × 1 / crisis_referral × 1 / dropout × 1 / 未终止 × 1
```

## 各层在做什么

**PA1 `schemas.py`** — `PatientScenario` 用 Pydantic v2 建模，全部 `extra="forbid"`，
病例里写错字段名会直接报错而不是静默忽略。`revealable_facts()` 把七组字段摊平成一串
「可说的事实」，`declared_numbers()` 收集病例里出现过的数字；守卫只核对本轮授权依据，不能借用其他字段的数字。

**PA2 `library.py`** — 读索引、按版本解析、做 Schema 校验；索引里登记了但文件缺失、
版本没登记、`scenario_id` 与目录不一致，都会报明确的错。

**PA3 `generator.py`** — 每轮根据「系统上一句说了什么 + 当前等待类型 + 已经说过什么」
挑一条事实，交给 LLM 适配器润色。离线模式下 `RuleBasedLLM` 直接回参考台词，所以
在线与离线走的是同一套 prompt 组装逻辑，结果可复现。要点：

* 否认类事实**必须被追问**才说（`is_denial`），不能被主动抛出；
* 每条事实最多说一次；
* `unlock_after_events` 是闸门：系统没先抛出某事件，挂在它上面的事实就不许说
  （S02 的补充症状就这么锁着，保证「回环之后才补」是可验证的）；
* 轨迹步按**治疗层轮次**推进，量表逐题作答不会把它提前；
* LLM 报错或返回空白一律退回参考台词，会话不能因此崩掉。

**PA4 `guard.py`** — 发送前的硬规则与全文事实复核。真实模型模式额外调用模型复核每段回答的依据；离线模式只接受可完整核对的参考台词。事实漂移、风险升级、数量/病程错误、严重度升级和非法披露均阻断。否认按风险词所在分句核对，不能靠 `is_denial` 标记跳过风险检查。详见 [守卫规则及字段说明](GUARD_RULES.md)。

**完整证据链**：`evidence.py` 从现有病例编号构建本轮依据快照；API 生成器返回带 `fact_ids` 的 `segments`，守卫逐段复核；`recorder.py` 保留每次候选、重试关联、实际提示词、完整复核和提交状态。病例源文件不增加编号表。手动入口 `run_patient_agent.py` 每轮更新 `runs/manual-.../recorder.json`，自动回放沿用原记录导出位置。

**PA5 `runner.py`** — 建会话 → 患者开场 → 循环「生成动作 → 守卫检查 → 调对应 API →
记录 → 判断是否停止」。负责系统 API；生成器和真实模式守卫也会调用模型 API。开场同样检查，重试失败或复核不可用时停止且不提交候选；每轮记下墙钟耗时。

**PA6 `stopping.py`** — 判序固定：「系统终止」最硬，其次是安全暂停 / 人工复核，
再次是患者侧违规与目标事件覆盖，`max_turns` 垫底。结果里区分 `system_terminated`，
避免把「测试主动收手」误记成系统行为。

**PA7 `evaluator.py` / `report.py`** — 评测器只读记录、不重跑会话，所以「跑一次评多次」
很便宜。八项判分：必需事件 / 禁止事件 / 层覆盖 / 终止原因 / 候选排除 / 守卫阻断违规 /
错误码 / 轮数下限。层覆盖取「路由层 ∪ 已观测事件声明的层」——低风险病例的 L1 只体现为
`l1_clear`、危机病例的 L5 只体现为结案事件，只看路由会漏判。报告把多份评测汇总成
路径覆盖、错误类型（拆出「非设计内故障」）、事实一致性、终止分布与逐病例明细。

## 测试

`python -m pytest`，181 个用例，默认离线确定性、不需要 key、不出网。

| 测试文件 | 用例 | 盯的是 |
| --- | --- | --- |
| `test_schemas.py` | 14 | PA1 字段词汇、`extra="forbid"`、派生视图、JSON 往返 |
| `test_library.py` | 10 | PA2 索引/版本解析、路径校验、缓存、坏文件报错 |
| `test_generator.py` | 28 | PA3 事实选择、否认需追问、量表计划、轨迹闸门、LLM 降级 |
| `test_guard.py` | 25 | PA4 每类违规的放行/拦截边界 |
| `test_mock_system.py` | 31 | 八接口契约（签名一致）、Mock 五层分支行为、逐轮记账只记一次 |
| `test_runner_stopping.py` | 20 | PA5/PA6 判序、危机一轮止、暂停插入、异常路由到 L5 |
| `test_evaluator.py` | 20 | PA7 八项判分口径 |
| `test_report.py` | 12 | PA7 汇总口径与 Markdown 渲染 |
| `test_acceptance.py` | 12 | 五条验收场景 + 全库回放端到端 + 逐轮账本自洽 |
| `test_cli.py` | 9 | CLI 子命令、选项位置、落盘、退出码 0/1/2 |

## 已知简化

* Mock 五层桩是**为了驱动测试而写的简化状态机**：症状识别靠关键词表，量表题面是固定的
  简化题库，L3 的候选与排除规则只覆盖内置 6 个病例需要的那几条。它验证的是「Patient Agent
  与系统的交互契约和分支覆盖」，不是「系统本身对不对」。
* 离线模式的患者台词是病例模板的直接渲染，固定且可复现；接真模型后措辞会变，
  但「说什么事实」仍由 PA3 决定、由 PA4 兜底。
