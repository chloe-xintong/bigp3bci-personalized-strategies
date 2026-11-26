# 眨眼检测器冻结说明

## 冻结版本

SWLDA 主实验（`outputs/swlda_all_studyf_v3`）的策略 C 使用 `old_candidate_overlap_v1` 版本的眨眼候选。参数写在 `configs/expanded_pilot.json` 的 `blink_detector` 中，实现位于 `src/bigp3bci_expanded_pilot.py` 的 `detect_blinks`。全部 90 份处理记录中的检测参数一致（见 `scripts/audit_studyf_abc_strategies.py` 检查 II.4）。

| 参数 | 值 |
|---|---|
| 额区通道 | EEG_F3、EEG_Fz、EEG_F4 |
| 后部通道 | EEG_Pz、EEG_PO7、EEG_PO8、EEG_Oz |
| 辅助低通 | 5 Hz，2 阶 |
| 进入 / 退出阈值 | 20 µV / 10 µV |
| 最小额区幅值 | 25 µV |
| 最少支持通道 | 2（峰值比例 0.5，最小相关 0.6） |
| 持续时间 | 0.08–1.2 s |
| 恢复 / 合并间隔 | 0.08 s / 0.15 s |
| 最小峰间距 | 0.35 s |
| 额区／后部支持比 | 1.2；后部下限 1.0 µV |
| 最小慢波 PTP 比例 | 0.35 |

阈值都是固定常数，不按整段记录的百分位或标准差拟合。检测只用因果信号；把文件截成一半，截断点之前已确认的候选不变。

## 参与过参数开发的数据

- 被试 F_03、F_05、F_06、F_08、F_20、F_25 的文件用于开发；F_07、F_21、F_23、F_24 未参与。
- 人工眨眼标注（`outputs/artifact_qc/continuous_manual_qc/manual/`）和 12 份文件的检测评估（`outputs/three_policy_comparison`、`outputs/blink_annotation_update`）包含 F_05_SE003_CBCol_Train04 和 F_08_SE003_CBCol_Train03。这两份文件后来成为 SWLDA 验证 EDF（共 12/180 个验证字符）。人工标注的是眨眼，不是目标字符；`outputs/blink_validation/analysis_record.json` 记录阈值未改动（thresholds_changed=false）。

## 之后的开发版本

以下版本只用于检测器开发，没有进入 SWLDA 实验：

| 版本 | 配置 | 输出 |
|---|---|---|
| 主波边界与恢复区间 | `configs/blink_boundary_localization.json` | `outputs/blink_boundary_localization_v1` |
| 局部突出度二次确认 | `configs/bounded_prominence_secondary.json` | `outputs/bounded_prominence_secondary_v1` |
| 局部平台指标 | `configs/local_plateau_metrics.json` | `outputs/local_plateau_metrics_v1` |

修改检测参数须使用新的版本名和新的输出目录，旧输出保留。
