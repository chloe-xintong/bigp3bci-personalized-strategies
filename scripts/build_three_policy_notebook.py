"""Build the fixed-parameter three-strategy epoch comparison notebook."""
from pathlib import Path
import textwrap
import nbformat as nb
ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(textwrap.dedent(s).strip()))
def code(s):cells.append(nb.v4.new_code_cell(textwrap.dedent(s).strip()))
md('''
# 05 · 固定检测参数下的三种 epoch 处理策略

正式滤波、眨眼检测参数和输入数据固定，在 12 份已处理的训练 EDF 上比较：

1. **faults_only**：保留眨眼相关 epoch，只排除明确故障和共同的技术有效性条件。
2. **ptp_250uv**：在同一基础上，任一 EEG 通道 PTP>250µV 时拒绝；150µV 作为敏感性对照。
3. **blink_overlap**：在同一基础上，拒绝与 CANDIDATE_blink 有任意样本重叠的 epoch。

这是处理策略的模拟：所有事件、16 通道指标、拒绝原因、触发通道和时序都保留，不删除数据。保留与否不使用标签，本步不训练分类器。
''')
code('''
from pathlib import Path
import json
import numpy as np
import pandas as pd
from IPython.display import display,Image
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'configs/three_policy_comparison.json').exists())
OUT=ROOT/'outputs/three_policy_comparison'
config=json.loads((OUT/'config_snapshot.json').read_text())
record=json.loads((OUT/'comparison_record.json').read_text())
manifest=pd.read_csv(OUT/'input_manifest.csv')
overall=pd.read_csv(OUT/'overall_summary.csv')
events=pd.read_csv(OUT/'all_epoch_policy_decisions.csv',keep_default_na=False)
PRIMARY=['faults_only','ptp_250uv','blink_overlap']
display(pd.Series(record,name='comparison record').to_frame())
''')
md('''
## 1. 共用的基础条件

三种策略都排除：不完整窗口、滤波启动期、原始或滤波后的非有限值、非有限值之后的滤波恢复区，以及达到持续时间条件（0.25 秒）的原始恒定或全零片段。恒定/全零标记在 epoch 判定时已经可知。

跳变和低变化候选、持续异常通道只作标记，不作为排除条件；不做重参考或插值。faults_only 没有 PTP 上下限。近乎平线的原始片段保存下来供复核，只有精确恒定或全零按上述规则排除。

每份 EDF 单独初始化并连续因果处理，epoch 为 [onset, onset+ceil(0.8×fs))，baseline=None。三种策略使用同一批 epoch，差异只来自 PTP 上限或眨眼重叠条件。
''')
code('''
display(pd.Series(config,name='fixed comparison settings').to_frame())
display(pd.Series(json.loads((OUT/'detector_config_snapshot.json').read_text()),name='unchanged blink detector').to_frame())
reasons=pd.read_csv(OUT/'rejection_reasons.csv')
display(reasons[reasons.strategy.eq('faults_only')])
''')
md('''
## 2. 数据范围

8 份文件按预定规则选出；另外 4 份因有人工标注而补充处理。全部为训练数据。统计按 cohort 分层，站点为推断值。
''')
code('''
display(manifest.reset_index(names='zero_based_index'))
display(pd.read_csv(OUT/'cohort_summary.csv'))
''')
md('''
## 3. 保留率与额外拒绝

共同排除 264 个 epoch，全部来自各 EDF 前 10 秒的启动保护。这批数据中没有非有限、精确恒定或全零等额外故障。

250µV 方案与基础方案结果相同：通过基础条件的 epoch 都没有超过该上限。150µV 是更严格的敏感性方案。眨眼重叠策略额外拒绝了大量 epoch。
''')
code('''
display(overall)
base=int(overall.loc[overall.strategy.eq('faults_only'),'expected_kept'].iloc[0])
comparison=overall.copy()
comparison['extra_rejected_vs_faults_only']=base-comparison.expected_kept
comparison['extra_rejected_fraction_of_base_usable']=(base-comparison.expected_kept)/base
display(comparison[['strategy','expected_kept','expected_retention','extra_rejected_vs_faults_only','extra_rejected_fraction_of_base_usable']])
display(pd.read_csv(OUT/'additional_rejection_overlap.csv'))
''')
md('''
## 4. 各文件与被试、Session 的差异

文件按 manifest 编号 1–12。候选较多的被试会主导合并保留率，所以同时给出逐文件结果和实际拒绝集合。
''')
code('''
s=pd.read_csv(OUT/'edf_summary.csv')
display(s.pivot(index='edf_name',columns='strategy',values='expected_kept'))
display(Image(filename=str(OUT/'retention_by_file.png'),width=1250))
display(pd.read_csv(OUT/'subject_summary.csv'))
''')
md('''
## 5. target 与 non-target 的保留率

标签只用于这里的事后分组。分类效果在后续 SWLDA 实验（notebook 06）中按验证划分评估。
''')
code('''
label=events.groupby(['strategy','label']).agg(total=('would_keep','size'),kept=('would_keep','sum'),retention=('would_keep','mean')).reset_index()
display(label)
''')
md('''
## 6. 每个字符和刺激集合的剩余 flash 数

拒绝以 epoch 为单位，字符整体保留。下面给出每种策略下每字符可用 flash 数的范围，并以 F_05 SE003 Train04 为例。本步不设最小证据量和动态停止规则。

stimulus_group_id 是实际同时亮起的字符集合签名，原始 StimulusCode 同时保留；很多集合只出现一次。
''')
code('''
characters=pd.read_csv(OUT/'character_summary.csv')
display(characters.groupby('strategy').expected_kept.agg(['count','min','median','max']))
display(characters[characters.edf_name.eq('F_05_SE003_CBCol_Train04.edf')])
groups=pd.read_csv(OUT/'stimulus_group_summary.csv')
display(groups[groups.edf_name.eq('F_05_SE003_CBCol_Train04.edf') & groups.strategy.eq('blink_overlap')].head(15))
print('全部逐刺激集合计数：',(OUT/'stimulus_group_summary.csv').relative_to(ROOT).as_posix())
''')
md('''
## 7. 触发通道与拒绝原因

PTP 拒绝记录实际超过上限的通道，基础故障记录故障通道。眨眼重叠按事件关联记录，触发通道可在候选表的 supporting_channels 中查到。
''')
code('''
triggers=pd.read_csv(OUT/'trigger_channels.csv')
display(triggers.groupby(['strategy','channel'])['count'].sum().sort_values(ascending=False).to_frame())
display(pd.read_csv(OUT/'rejection_reasons.csv').groupby(['strategy','reason'])['count'].sum().to_frame())
''')
md('''
## 8. 两段完整人工复核区间内的保留情况

只看完全落在人工复核区间内、且通过共同基础条件的 epoch：F_05 SE003 Train04 的 0–35 秒、F_08 SE003 Train03 的 1–50 秒。与人工眨眼是否相交作为描述性标签，不参与策略判定。

自动候选区间常比人工边界更长：辅助因果分支有延迟，并包含眨眼的恢复尾波。因此会出现“与人工眨眼不相交、但被眨眼策略拒绝”的 epoch；反过来，与人工边界相交的 epoch 也不一定全部被拒绝。
''')
code('''
review=pd.read_csv(OUT/'reviewed_epoch_summary.csv')
display(review)
print('overlaps_manual_blink=True：epoch与人工疑似眨眼时间段相交；False：完整检查区间内未相交。')
''')
md('''
## 9. 在线判定时间

基础和 PTP 方案在 0–800 ms 样本到齐后即可判定。眨眼候选需要等待恢复与合并，所以眨眼策略使用统一的确认窗口：epoch 样本终点 + 最大候选持续时间 1.2 秒 + max(恢复 0.08 秒, 合并 0.15 秒)，各项按整数样本向上取整；在 256 Hz 下额外等待约 1.355 秒。

would_keep 是策略结果，policy_decision_seconds 是判定时刻。文件末尾数据不足时标记为 pending_at_eof；本批 epoch 的 pending 数为 0。
''')
code('''
display(overall[['strategy','mean_extra_wait_seconds','pending_at_eof']])
display(events.loc[events.strategy.eq('blink_overlap'),['edf_name','event_index','epoch_decision_seconds','policy_decision_seconds','additional_wait_seconds','decision_data_available']].head(8))
''')
md('''
## 10. 三种处理后的平均 ERP

同一正式因果信号，baseline=None，平均各方案保留的完整有限 epoch。每条曲线标出参与平均的数量，各通道和类别共享坐标。曲线按 epoch 合并 12 份 EDF，被试构成和类别数会影响形态。250µV 与基础方案的曲线重合，因为本批没有额外拒绝。
''')
code('''
display(pd.read_csv(OUT/'pooled_erp_counts.csv'))
display(Image(filename=str(OUT/'pooled_erp.png'),width=1300))
''')
md('''
## 11. 一致性检查与复现

运行：`python scripts/compare_epoch_policies.py`；配置：`configs/three_policy_comparison.json`。运行时检查眨眼检测参数哈希和原 8 文件清单。

结果包含全部 9,054 个 epoch × 每种方案。被拒绝的 flash 在后续分类中不给分数。13 项回归测试覆盖故障条件、标签不参与、策略差异、等待时间与文件末尾等行为；下面检查保存结果的一致性。
''')
code('''
assert record['detector_parameters_unchanged'] and record['actual_epochs_removed']==0
counts=events.groupby('strategy').size()
assert counts.eq(record['total_events']).all()
wide=events.pivot(index=['edf_name','event_index'],columns='strategy',values='would_keep')
for strategy in wide:
    assert (wide[strategy]<=wide.faults_only).all()
assert (events.onset_sample==events.original_event_sample).all()
assert events.groupby(['edf_name','event_index']).common_fault.nunique().eq(1).all()
assert events.loc[events.strategy.eq('blink_overlap'),'additional_wait_seconds'].gt(0).all()
assert events.loc[events.strategy.eq('faults_only'),'additional_wait_seconds'].eq(0).all()
print('PASS: all events preserved, common baseline identical, criteria nested, event samples unchanged, confirmation wait explicit')
''')
n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'bigp3bci','display_name':'Python (bigp3bci)','language':'python'}})
p=ROOT/'notebooks/05_epoch_policy_comparison.ipynb'
nb.write(n,p);print(p)
