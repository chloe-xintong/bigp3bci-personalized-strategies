# 实验日志

每次运行追加一行。日期为输出目录中记录文件的日期；commit 为首次包含该入口脚本的提交（2026-11-26 之前的运行早于对应代码提交，提交后代码未改变实验逻辑）。

| 日期 | 实验 | 入口脚本 | 输出目录 | commit | 读取 Test |
|---|---|---|---|---|---|
| 未记录 | 原始数据检查（notebook 01 第 12 节等） | notebook | — | `8aeab4e` | 读取了 Test EDF 的文件头和状态通道（统计字符数），未读取 EEG |
| 未记录 | Static 实验（已删除） | — | 已删除 | — | 是，Test/Static；结果已查看，输出按要求删除 |
| 2026-11-24 | 因果预处理 pilot | `run_causal_pilot.py` | `outputs/causal_pilot` | `8aeab4e` | 否 |
| 2026-11-25 | 扩展训练 pilot | `run_expanded_pilot.py` | `outputs/expanded_pilot` | `8aeab4e` | 否 |
| 2026-11-25 | 三种 epoch 策略 | `compare_epoch_policies.py` | `outputs/three_policy_comparison` | `8aeab4e` | 否 |
| 2026-11-25 | 主波边界与恢复区间 | `refine_blink_boundaries.py` | `outputs/blink_boundary_localization_v1` | `8aeab4e` | 否 |
| 2026-11-25 | 局部突出度二次确认 | `run_bounded_prominence_secondary.py` | `outputs/bounded_prominence_secondary_v1` | `8aeab4e` | 否 |
| 2026-11-25 | 局部平台指标 | `measure_local_plateau_excursions.py` | `outputs/local_plateau_metrics_v1` | `8aeab4e` | 否 |
| 2026-11-25 | SWLDA 拒绝策略（v1） | `run_swlda_rejection_experiment.py` | `outputs/swlda_rejection_experiment_v1` | `8aeab4e` | 否 |
| 2026-11-25 | SWLDA 三 Session（v2） | `run_full_three_session_preprocessing.py` | `outputs/swlda_three_sessions_v2` | `8aeab4e` | 否 |
| 2026-11-25 | SWLDA 全部 StudyF（v3，主结果） | `run_all_studyf_preprocessing.py` | `outputs/swlda_all_studyf_v3` | `8aeab4e` | 否 |
| 2026-11-26 | SWLDA 校准预算 | `run_swlda_calibration_budget.py` | `outputs/swlda_calibration_budget_train_validation_v1` | `8514fdf` | 否 |
| 2026-11-26 | Session 漂移描述 | `check_session_drift.py` | `outputs/session_drift_descriptive_v1` | `03ff479` | 否 |
| 2026-11-26 | 历史数据量（探索性） | `run_history_amount_exploratory.py` | `outputs/history_amount_exploratory_v1` | `03ff479` | 否 |
| 2026-11-26 | EA 敏感性 | `run_ea_sensitivity.py` | `outputs/ea_sensitivity_v1` | `0ceb52f` | 否 |
| 2026-11-26 | 跨被试 SWLDA（留一被试，探索性，附加分析） | `run_cross_subject_swlda.py --stage controls`，随后 `--stage cross` | `outputs/cross_subject_swlda_exploratory_v1` | `88dd0e3`（协议预先提交于 `762ea1e`） | 0 次；仅读 Train 缓存，原始 EDF 读取也为 0 |

跨被试附加分析：10 名被试全部保留，先重算 30 个仅当前对照，再拟合 40 个跨被试模型。
对照 AUC、字符准确率及逐 flash 分数与原 notebook 11（现 07）完全一致，最大差为 0。
无调参、无被试排除，不改变主比较表、主图或主设计。k=3/6/12 的 X−C 宏平均
AUC 差依次为 −0.042472、−0.080191、−0.091591；按被试 bootstrap 10,000 次的
95% 区间依次为 [−0.073878, −0.005008]、[−0.121428, −0.035858]、
[−0.130375, −0.043919]。这是未作多重比较校正的探索性描述；直接混合其他被试的
S1/S2 数据在本设置下没有改善平均校准效果。所有逐被试差异保留在独立输出目录。
