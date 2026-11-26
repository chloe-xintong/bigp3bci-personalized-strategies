"""Standalone cells inserted first in the consolidated blink report."""
import nbformat as nb


def single_review_cells():
    return [nb.v4.new_markdown_cell('''## 单人交互标注：先做 30 个预试

**从这里开始，运行下面两格即可，不用运行后面的历史分析。** 使用 `bigp3bci` kernel。
这与之前拖选 `BAD` 区间的操作相同，不是点击通道名标坏通道。

1. 运行初始化和面板，选择匿名编号，点击 **打开波形**。默认显示前 30 个预试；编号后的 `[已保存]` 表示已填过结论。
2. 在弹出的 MNE 窗口按 **`a`** 打开 annotation 编辑器。直接用默认的 **`BAD_`** 或 `BAD`，鼠标拖选你认为是眨眼的区间。可以标多段，**无需覆盖灰色定位线也能保存**。
3. 本面板的 **`BAD` / `BLINK`＝真眨眼**；`RECOVERY`＝前一事件的反向恢复；`OTHER`＝其他伪迹；`UNCERTAIN`＝无法判断。**这与下方历史“多检复核”面板把 BAD 当作 not_blink 的规则不同，请只用本节面板完成此次任务。**
4. 填置信度与备注，点击 **保存标注** 或 **保存并下一个**。所有画出的区间都会保存到 FIF 和 CSV；覆盖灰色定位线且标签一致时，才给该候选填写结论。未覆盖定位线时显示 `spans_saved_no_focus_decision`，表示区间已正式保存、候选结论仍为空。右键删除/重画后需再保存；空白不自动判为其他伪迹。
5. **如果整个当前片段未见眨眼，不用画区间，点击“未见眨眼，保存并下一个”。** 会明确记录 `not_blink` 和 `reviewed_no_blink`，区分“已看过但未见眨眼”与“还没复核”。如果无法判断，请用 `UNCERTAIN` 标记，而非选未见眨眼。仍有区间时不会用未见眨眼覆盖它们。
6. 先完成 `先做 30 个`，之后再切到 `其余 90 个`。不确定如何提交时可先点击 **先保存草稿**：区间会落盘，但不生成结论。切换到其他候选时也会先保存区间草稿；重新打开当前编号会直接显示内存波形，不再阻止操作。关闭窗口不等于保存。

波形是原有因果分析信号，窗口约 8 秒，时间从当前片段起点算，定位线通常位于 4 秒；显示缩放不代表检测阈值。仅显示当前匿名片段与本次标注，不加载靶标线、检测器特征或历史人工判定。相邻波形可能使你认出个别记录，匿名显示不能保证完全不可识别。

结论写入 `outputs/single_blind_review_v1/reviewer/review_form.csv`；画出的区间另存为 `reviewer/annotations/<匿名编号>-annot.fif` 和 `_spans.csv`。每次保存前会备份旧版本，原始 EDF、旧人工标注和自动候选保持不变。仅复核候选，不代表其余时段已完整复核。
'''), nb.v4.new_code_cell('''from pathlib import Path
import sys
import importlib
import os

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'configs/expanded_pilot.json').exists())
os.environ.setdefault('_MNE_FAKE_HOME_DIR', str(ROOT))
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
PACKET = ROOT / 'outputs/single_blind_review_v1'
if not (PACKET / 'reviewer/review_form.csv').exists():
    raise FileNotFoundError('找不到单人复核包 outputs/single_blind_review_v1。')

get_ipython().run_line_magic('matplotlib', 'qt')
import bigp3bci_blind_review as blind_review
importlib.reload(blind_review)
print('已准备好；运行下一格显示面板。')
'''), nb.v4.new_code_cell('''# 可重复运行：接回已有窗口和内存标注，不需要重启 kernel。
single_blink_state = blind_review.review_panel(
    ROOT, PACKET, existing_state=globals().get('single_blink_state'))
'''), nb.v4.new_markdown_cell('''## 30 个候选中心片段：覆盖、定位与漏检日志

**这些片段围绕先前未匹配的自动候选抽取；不是随机连续时段，不能估计全局召回率。** 下表只描述候选中心复核片段内的事件覆盖。假设复核者检查了整个显示片段；不会据此新增整份 EDF 的完整复核覆盖。

111 段人工眨眼标注，按 EDF 转回原始时间后，将正时长相交的人工区间传递合并为 84 次事件。仅相接、不相交的区间不合并，不跨 EDF。25 次事件在多个重叠片段中被重复标注，共产生 27 段额外标注；没有在同一片段内合并多个区间。这是明确的操作性去重规则，并非经过独立裁决的事件身份真值。逐条对应见 `outputs/pilot_blink_misses/deduplication_audit.csv`。

**重叠标准下的候选中心召回率**为 83/84（98.8%），不等于全局召回率或准确定位。另列人工时长覆盖比例及 IoU（交集 / 并集）；阈值是事后描述性对照，不用于选最佳参数。严格标准不通过可源于边界偏移，不能全部解释为生理眨眼漏检。表中额外提供一对一最大匹配，避免同一候选重复计入多个事件。

对于有重叠的 83 次事件，各取 IoU 最高的候选：候选起点较人工起点的中位差为 +75 ms，终点中位差为 +306 ms，IoU 中位数为 0.249。边界误差为自动时间减人工时间。

唯一宽松标准未匹配事件：F_05 SE002 Train02，213.585–213.755 s。对应已保存日志的 `exclusion_reasons=duration`：辅助信号触发区间为 213.668–215.285 s，持续 1.617 s，超过 1.2 s 上限。重放缓存辅助信号显示是单个未合并的触发区间，非多段合并导致；人工区间内额区共同幅值最大约 45.5 µV，已超过 20 µV 进入阈值。人工标注时长 0.169 s 不是检测器用于持续时间判定的量，因此不能据人工时长排除 duration 原因。

日志：`outputs/pilot_blink_misses/missed_event_excursion_logs.csv`；匹配及边界明细：`best_iou_pair_timing.csv`。复现：先运行 `scripts/evaluate_pilot_blink_misses.py`，再运行 `scripts/audit_pilot_blink_matches.py`。所有结论均为单人、所选片段内的开发性描述。
'''), nb.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display
_audit_root = next(p for p in [Path.cwd(), *Path.cwd().parents]
                   if (p / 'configs/expanded_pilot.json').exists())
_audit = json.loads((_audit_root / 'outputs/pilot_blink_misses/matching_audit.json').read_text())
display(pd.DataFrame(_audit['matching']).T)
display(pd.DataFrame(_audit['best_iou_pair_boundary_errors_seconds']).T)
display(pd.read_csv(_audit_root / 'outputs/pilot_blink_misses/missed_event_excursion_logs.csv'))
''')]
