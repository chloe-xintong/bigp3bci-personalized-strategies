"""Frozen-v1 limitations and a separate prospective v2 workflow."""
import nbformat as nb


def duration_report_cells():
    return [nb.v4.new_markdown_cell('''## v1：全量时长拒绝审计与检测器局限

统计范围：全部 10 名 StudyF 被试、30 个 Train Session、90 份 EDF。这里的“被拒绝事件”指进入阈值触发后、尚未成为已接受候选的合并 excursion；不是最终候选表中的事件。v1 采用 20 µV 进入阈值、25 µV 峰值接受幅度阈值、0.08–1.2 s 持续时间范围。本次从原日志及缓存辅助信号重算指标，未修改任何检测参数。

日志中 558 个事件带有 `duration` 拒绝原因，其中 28 个位于启动期。非启动期 530 个中，过长 152 个、过短 378 个；达到 25 µV 峰值接受阈值的分别为 152 和 55 个。过长事件中 2 个与现有人工眨眼阳性标注相交；两者的唯一拒绝原因均为 `duration`，对应人工眨眼也均没有其他已接受候选覆盖：

| 文件 | 人工区间（s） | 被拒绝触发区间（s） | 触发时长（s） | 峰值幅度（µV） |
|---|---|---|---|---|
| F_05 SE002 Train01 | 55.615–56.063 | 55.715–56.992 | 1.277 | 74.95 |
| F_05 SE002 Train02 | 213.585–213.755 | 213.668–215.285 | 1.617 | 74.65 |

人工参考合并旧的疑似眨眼阳性标注与本次 30 个片段的标注，排除启动期后按 EDF 内区间重叠去重，共 265 次参考事件。上述 2 次占这份有限人工阳性参考的 0.75%；这**不是全局时长相关漏检率**。530 个拒绝事件中的未标注部分仍未知，不能把未相交当作非眨眼，也不能把 2/207 当作真实眨眼比例。两例仅证明时长规则确有丢失人工所标眨眼的实例；不能直接推断去掉 duration 后必然恢复多少事件，因为接受状态也会影响后续峰间距规则。

需要明确报告两项检测器局限：

- **时长规则会丢失人工标注的眨眼。** 人工眨眼区间与辅助信号滞回区间是不同的时间量；后者持续过长时，v1 将整段拒绝。
- **区间终点偏晚会扩大 epoch 重叠拒绝范围。** 83 个匹配参考事件的最佳 IoU 配对中，终点偏差中位数为 +306 ms。若用候选区间删 epoch，延后终点可能覆盖人工眨眼结束后的更多 flash 窗口，从而增加 epoch 和证据损失。这个机制是检测器边界的局限；本次尚未单独量化它造成的额外 epoch 数，不能将所有策略性能下降都归因于此。

本次召回结果必须写作 **“重叠标准下的候选中心召回率 83/84（98.8%）”**。全局召回率仍需要在独立、完整复核的连续时段内评估。

### 后续 v2：先诊断，再冻结，最后独立评估

v1 检测配置、输出和已有对照保持冻结。v2 可比较限制滞回持续时间、相对阈值终止区间、超上限时截断而非整段丢弃等方案；目前只列研究方向，不实现或挑选参数。各方案须先在训练数据及复核样本中诊断，并共同检查漏检、误报、边界误差、epoch 损失及在线确认时延，不能以放过单个事件为优化目标。

选择规则后，另存 v2 配置、实现版本和输出，冻结阈值及评估标准；最后才在 **Static 之外**预先固定的独立范围内评估。该独立范围须另列清单；本次没有运行任何此类评估，也没有读取最终 Test 来调整规则。

复现：`python scripts/audit_duration_rejections.py`。事件、人工对应及来源哈希保存于 `outputs/duration_rejection_audit_v1/`。
'''),nb.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display
_duration_root = next(p for p in [Path.cwd(), *Path.cwd().parents]
                      if (p / 'configs/expanded_pilot.json').exists())
_duration_out = _duration_root / 'outputs/duration_rejection_audit_v1'
_duration_summary = json.loads((_duration_out / 'summary.json').read_text())
display(pd.DataFrame(_duration_summary['nonstartup_by_duration_side']).T)
display(pd.read_csv(_duration_out / 'amplitude_pass_manual_overlap.csv'))
display(pd.read_csv(_duration_out / 'manual_blinks_lost_duration_only.csv'))
''')]
