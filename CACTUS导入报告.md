# CACTUS 导入报告（2026-10-23）

> **2026-10-29 选择性合入说明**：下文是原包的历史导入说明。本次仅合入 CACTUS 工具、150 个病例、中间记录、提取依据并扩展现有索引；保留工作区的 MentalKG 适配、既有代码与测试、原 4 例 DB 病例、内置 6 例及配置，未复制旧版代码或历史运行报告。
>
> 当前代码实测：`pytest` **264 passed**；导入工具在独立目录小样本重跑 **3/3 通过**；全库离线回放 **154/154 通过**。本次报告见 `runs/cactus-merge-20261029/report.md`，实际模式为 `offline / rule_based / mock_system`。
>
> **数据解释补充**：下文“无风险”“稳定”等表述是生成模拟病例时的设定；原文未提及不能等同于明确否认，无结局记录不能等同于已观察到无变化。阴性鉴别与量表作答也包含模板或推导值。三种态度合并后不构成态度差异评测，CBT 标签未接入技术选择评分。当前 150 例中有 18 例未满 18 岁，最低 10 岁；均保留 `needs_review`，正式研究评测前需人工复核。

## 结论

- 数据集：CACTUS（GitHub coding-groot/cactus，EMNLP Findings 2024）的评测集文件 `evaluation_chinese.json`，本包里已有，不需要再下载。
- 结果：450 条去重为 **150 个病例**（CACTUS_001–CACTUS_150），已补进默认病例库 `imported_scenarios/`（现共 154 例：原 DB_P0001–P0004 + CACTUS 150 例）。
- 验证：生成时逐例试跑 150/150 通过；全库 `run-all` **154/154 通过**（报告 `runs/cactus-import-20261023/report.md`）；项目测试 `pytest` **264 passed**。

## 为什么不能直接用 db_import

实测 `python tools/db_import.py --source evaluation_chinese.json --dry-run` 在第一条即失败：`记录缺少 record_id`。db_import 只认「规范化记录（record_id + symptoms）」或「会谈记录（conversation: doctor/patient）」，CACTUS 的 intake 是英文自由文本，两样都不是。

解法：新增适配层 `tools/cactus_import.py`，把 CACTUS 适配成规范化记录后**复用 db_import 的 build_scenario / import_one 生成与试跑**，原转换器未改。

## 适配规则（都是启发式，需人工复核）

1. **去重**：450 条 = 150 个来访者 × 3 种态度（positive/neutral/negative），同组 ground_truth_CBT 一致，只生成 1 例；三种态度与原 idx 记在病例 `background.major_events` 溯源里。
2. **背景**：解析 intake 六段结构 —— 年龄（`age_range`，如 45岁）、职业、婚恋+家庭；解析失败留「未知」，不猜。
3. **病程**：主诉段时间表述映射成中文（如 few months → 几个月，2 weeks ago → 2周）。
4. **症状映射**：只扫第 2（主诉）/3（求助原因）/5（功能影响）段，句级切分后做**分句级否认剔除**（整句剔除会把「因没去看家人而内疚」这类一句话两半的病例误杀，第一版因此有 9 例无命中；改分句级后 0 例兜底）；关键词映射到 db_import 的 11 个症状 code，主诉段权重 2、其余 1，权重 ≥6→严重度 3、≥3→2、其余 1。逐例命中证据在 `tools/cactus_import_details.json`。
   - 分布：自我价值感低 102 例、焦虑 77、睡眠 53、兴趣减退 38、情绪低落 18、注意力 5、疲乏 5、易激惹 4、创伤 2；每例 1–4 个症状。这与 CACTUS 的性质相符——它是 CBT 咨询病例（认知/自我评价议题为主），不是抑郁诊断病例集。
5. **风险**：450 条全文扫描无自杀/自伤表述，risk 一律 none（有数据依据，不是默认乱填）。
6. **治疗走向**：CACTUS 没有结局数据，一律 stable（completed_no_change），不编造改善/退出；ground_truth_CBT 记进 tags（`cbt_*`）与 `psychology.cognitive_patterns`。

## 产物

| 文件 | 内容 |
| --- | --- |
| `tools/cactus_import.py` | 适配层脚本（可重跑：`--limit N` 小样本、`--force` 覆盖重跑） |
| `tools/cactus_normalized_records.json` | 150 条规范化记录（适配中间产物，可审计） |
| `tools/cactus_import_details.json` | 逐例症状命中证据句（人工复核用） |
| `imported_scenarios/CACTUS_*/1.0.0.json` + `index.json` | 150 个病例，已登记进默认病例库 |
| `runs/cactus-import-20261023/` | 全库试跑报告（154/154 通过） |

## 遗留（人工环节，不阻断使用）

1. 每例的患者台词仍是 db_import 的中文模板句（与标准库导入相同），临床真实感靠逐句复写；病例都带 `needs_review` 标签。
2. 症状映射是英文关键词启发式，`cactus_import_details.json` 里的 evidence 是复核入口；尤其只有 1 个症状、或命中靠功能段（第 5 段）的病例优先看。
3. CACTUS 来访者含个别低龄虚构个案（如 13 岁），与 intake 原文一致，未做年龄过滤；若病例库要用于正式评测，建议先定一条年龄口径再筛。
4. 本次只进了 `imported_scenarios/`（.env 已把它设为默认库），**没有动 `scenarios/`**——那是验收锁定的正式库，要不要把 CACTUS 例升进去由人工决定。
