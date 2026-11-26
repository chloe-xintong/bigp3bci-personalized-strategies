# 仓库整理方案

本方案只调整仓库结构和文档，不改动任何已报告的结果。

## 一、Notebook

新编号按分析流程排列：数据 → 预处理与眨眼 → epoch 策略 → SWLDA → 探索性分析 → EEGNet。

| 原文件 | 处理 | 新位置 | 输出 | 说明 |
|---|---|---|---|---|
| 00 MNE 教程 | 移出主线 | `learning/` | 原样 | 不是本项目分析 |
| 01 EDF 检查与预处理 | 并入新 01 | 01 数据清单与原始检查 | — | 内容与 02 重叠，几乎无输出 |
| 02 原始数据检查 | 作为新 01 主体 | 01 数据清单与原始检查 | 清空 | 数据清单与通道审计 |
| 03 伪迹 QC | 并入新 01 | 01 数据清单与原始检查 | — | 只有初步检查，EOG/ECG 已标注跳过 |
| 05 预处理与候选检测 | 保留，不与 06 合并 | 02 预处理与候选检测 | 压缩图片 | 已 15.5 MB，体积来自图片输出 |
| 06 眨眼检测与人工复核 | 保留，修复第 8 节 | 03 眨眼检测与人工复核 | 保留 | 含 Qt 人工标注单元，无法在干净环境重跑；第 8 节中文已丢失（git 历史中也没有），根据残留数字重写 |
| 08 主波边界与恢复区间 | 与 09 合并 | 04 眨眼边界与二次确认 | 清空 | 同属检测器开发 |
| 09 局部突出度二次确认 | 与 08 合并 | 04 眨眼边界与二次确认 | 清空 | 同上 |
| 07 三种 epoch 策略比较 | 保留 | 05 epoch 策略 A/B/C | 保留 | 策略 C 依赖眨眼候选，故排在眨眼之后 |
| 10 SWLDA 拒绝策略比较 | 保留 v3，旧章节移到文末附录 | 06 SWLDA 主结果 | 保留 | full_three_sessions、rejection_experiment 已被 v3 替代 |
| 11 校准预算（验证） | 保留 | 07 校准预算 | 保留 | |
| 12 历史数据量 | 保留 | 08 历史数据量 | 保留 | |
| 13 EA 敏感性 | 保留 | 09 EA 敏感性 | 保留 | |
| — | 新建 | 10 EEGNet | — | 协议确定后运行 |
| 04 连续 EEG 人工复核 | 作为附录，清理本地路径 | A1 连续 EEG 人工复核 | 保留 | 约 261 处绝对路径，多在输出中；人工交互，无法重跑 |

**输出规则：** `outputs/` 不在仓库中，notebook 输出是在 GitHub 上唯一能直接看到的结果。报告类 notebook（新 03、05–09、A1）保留一份执行后的输出；开发类（新 01、04）清空输出；新 02 保留输出并压缩图片。

## 二、脚本

全部保留，不删除。README 中补全尚未说明的脚本。

**已在 README 中说明**
- 预处理：`run_causal_pilot.py`、`run_all_studyf_preprocessing.py`、`run_full_three_session_preprocessing.py`
- epoch 策略与眨眼：`compare_epoch_policies.py`、`refine_blink_boundaries.py`、`run_bounded_prominence_secondary.py`、`summarize_blink_label_association.py`
- SWLDA：`run_swlda_rejection_experiment.py`、`audit_swlda_saved_experiment.py`、`plot_swlda_all_studyf_accuracy.py`
- 探索性分析：`run_swlda_calibration_budget.py`、`add_weighted_logloss.py`、`run_history_amount_exploratory.py`、`run_ea_sensitivity.py`

**补写进 README**

| 脚本 | 原因 |
|---|---|
| `run_expanded_pilot.py` | 生成 `outputs/expanded_pilot`，是后续预处理输入清单的一部分 |
| `check_session_drift.py` | 报告引用了漂移结果；对应 `session_drift_descriptive.json` |
| `measure_local_plateau_excursions.py` | 有独立配置 `local_plateau_metrics.json` 和输出 |
| `audit_history_amount_exploratory.py` | 生成 notebook 08 显示的独立审计 |
| `audit_studyf_abc_strategies.py` | v3 A/B/C 实验的独立审计 |
| `plot_swlda_calibration_budget.py` | 不重新拟合，直接从 CSV 重画校准预算图 |
| `build_double_blind_review.py`、`score_double_blind_review.py` | 双盲复核：生成匿名复核包并比较两份独立结果 |
| `export_blink_candidates.py`、`export_unmatched_blink_context.py` | 导出候选注释和未匹配候选的上下文，供人工复核 |
| `audit_main_false_positive_reviews.py` | 审计已保存的主波假阳性复核 |

**生成 notebook 的脚本**（`build_*_notebook.py`、`*_cells.py`、`notebook_reports.py`）保留在 `scripts/`，不移动目录：脚本按 `scripts/` 的相对层级定位仓库根目录，移动后路径都要改，收益很小。

## 三、src

19 个模块保持平铺，不拆分子目录。模块已有 `bigp3bci_` 前缀；冻结实验按路径记录源文件哈希，移动后已有输出的复现检查会失败。README 增加以下对照表：

| 用途 | 模块 |
|---|---|
| 预处理与质量控制 | `preprocessing`、`continuous_qc`、`policy_comparison` |
| 眨眼检测与复核 | `expanded_pilot`、`candidate_review`、`blink_boundaries`、`boundary_review`、`blind_review`、`bounded_prominence`、`plateau_metrics` |
| SWLDA 与评估 | `swlda`、`probability`、`flash_early_stopping` |
| 探索性分析 | `calibration_budget`、`history_amount`、`euclidean_alignment` |
| 绘图 | `budget_plot`、`figure_style` |

## 四、tests

- 现有 16 个测试保留。
- 新增测试集守卫：把各脚本分散的 `Train` 路径检查收拢成一个共用加载函数，并测试读取 Test、Static、Dyn、DynBigram 文件时报错。

## 五、configs

README 增加以下对照表：

| 配置 | 入口脚本 | 输出目录 |
|---|---|---|
| `causal_pilot.json` | `run_causal_pilot.py` | `outputs/causal_pilot` |
| `expanded_pilot.json` | `run_expanded_pilot.py` | `outputs/expanded_pilot` |
| `three_policy_comparison.json` | `compare_epoch_policies.py` | `outputs/three_policy_comparison` |
| `blink_boundary_localization.json` | `refine_blink_boundaries.py` | `outputs/blink_boundary_localization_v1` |
| `bounded_prominence_secondary.json` | `run_bounded_prominence_secondary.py` | `outputs/bounded_prominence_secondary_v1` |
| `local_plateau_metrics.json` | `measure_local_plateau_excursions.py` | `outputs/local_plateau_metrics_v1` |
| `swlda_rejection_experiment.json` | `run_swlda_rejection_experiment.py` | `outputs/swlda_rejection_experiment_v1` |
| `swlda_three_sessions.json` | 由 notebook 读取 | `outputs/swlda_three_sessions_v2` |
| `swlda_all_studyf.json` | 由 notebook 读取 | `outputs/swlda_all_studyf_v3` |
| `swlda_calibration_budget.json` | `run_swlda_calibration_budget.py` | `outputs/swlda_calibration_budget_train_validation_v1` |
| `session_drift_descriptive.json` | `check_session_drift.py` | `outputs/session_drift_descriptive_v1` |
| `history_amount_exploratory.json` | `run_history_amount_exploratory.py` | `outputs/history_amount_exploratory_v1` |
| `ea_sensitivity.json` | `run_ea_sensitivity.py` | `outputs/ea_sensitivity_v1` |

## 六、docs

- 新增 EEGNet 协议：固定 epoch 数或停止规则须在训练前写定。
- 新增眨眼检测器冻结说明：参数版本、哪些文件参与过调参。
- 新增实验日志：每次运行的日期、commit、是否读取 Test。
- 报告用图放入 `docs/figures/`（该目录的 PNG 已从 `.gitignore` 中豁免）。

## 七、README

- notebook 范围由“00–16”改为实际编号。
- 新增“测试集政策”：不读取哪些文件、原因。
- 新增“复现步骤”：按顺序列出命令。
- 补充第二节的脚本说明、第三节的模块表和第五节的配置表。

## 八、执行顺序

1. 修复 06 第 8 节，清理 04 的本地路径。
2. 按第一节合并、重排 notebook；报告类重新执行一次，确认结果数值不变。
3. 更新 README 和 docs（第二、三、五、六、七节）。
4. 新增测试集守卫测试。
5. 配置 nbstripout，只对开发类 notebook 清空输出。
