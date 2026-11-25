"""Build the fixed-parameter three-strategy epoch comparison notebook."""
from pathlib import Path
import textwrap
import nbformat as nb
ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(textwrap.dedent(s).strip()))
def code(s):cells.append(nb.v4.new_code_cell(textwrap.dedent(s).strip()))
md('''
# 10 · 固定检测参数后的三种epoch处理策略

固定正式滤波、辅助眨眼检测参数及输入数据，在当前12份已处理训练EDF上比较：

1. **faults_only**：保留眨眼相关epoch，仅排除明确故障和共同技术有效性条件。
2. **ptp_250uv**：在同一基础上，任意一个指定EEG通道PTP>250µV则拒绝；150µV另列为预先已有的敏感性对照。
3. **blink_overlap**：在同一基础上，拒绝与现有CANDIDATE_blink有任意正样本重叠的epoch。

本轮是处理策略模拟，不是真正删除数据，也不代表默认启用眨眼拒绝。所有事件、16通道指标、原因、触发通道和时序都保留。没有训练分类器，不用标签决定epoch是否保留，不按结果选择参数。
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
## 1. 哪些基础条件共用，哪些只是候选？

三种策略都排除：不完整窗口、滤波启动期、原始/滤波非有限值、非有限值后的滤波恢复区、已经达到持续时间条件的原始恒定或全零片段。原始恒定/全零的关联标记必须在epoch决定时已经可知，仍使用原配置的0.25秒条件。

**没有把跳变或低变化候选直接当成已确认故障**，也不自动删持续异常通道；不增加重参考或插值。第一种策略没有PTP上限，也没有PTP下限拒绝；否则“只排故障”会混入待验证幅值门槛。原始近乎平线标记继续保存用于复核，只有精确恒定/全零按上述规则排除。

每份EDF仍单独初始化并连续因果处理，0–800ms为[onset,onset+ceil(0.8×fs))，baseline=None。所有策略共用同一批epoch，因此差异只来自新增的PTP上限或眨眼重叠条件。
''')
code('''
display(pd.Series(config,name='fixed comparison settings').to_frame())
display(pd.Series(json.loads((OUT/'detector_config_snapshot.json').read_text()),name='unchanged blink detector').to_frame())
reasons=pd.read_csv(OUT/'rejection_reasons.csv')
display(reasons[reasons.strategy.eq('faults_only')])
''')
md('''
## 2. 数据范围：原8文件和补充4文件分开

8份固定pilot按原预定规则选出；另外4份因用户保存人工标注而补充处理，不是独立盲测。所有文件均为训练数据。整体统计同时保留cohort分层，站点仍为inferred；文件数或epoch数不能当作独立被试数。
''')
code('''
display(manifest.reset_index(names='zero_based_index'))
display(pd.read_csv(OUT/'cohort_summary.csv'))
''')
md('''
## 3. 全体保留率与额外拒绝代价

共同排除264个epoch，本轮均来自各EDF前10秒启动保护。未在这批数据中发现会额外触发第一种策略的非有限、精确恒定/全零等故障；这不表示以后可以取消故障检查。

250µV方案与基础方案相同：没有epoch在基础条件通过后超过该上限。150µV是较严格敏感性方案，不作为自动选择的新阈值。眨眼重叠策略会剔除大量额外epoch，不能因为检测器与已标眨眼重叠较好，就推导这些epoch必须删除。
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
## 4. 各文件与被试/Session差异

文件索引按manifest1–12。不能只看合并保留率；某个被试候选很多时，合并统计可能主要反映这个被试。即使两个策略保留率相近，也应对照实际拒绝集合。
''')
code('''
s=pd.read_csv(OUT/'edf_summary.csv')
display(s.pivot(index='edf_name',columns='strategy',values='expected_kept'))
display(Image(filename=str(OUT/'retention_by_file.png'),width=1250))
display(pd.read_csv(OUT/'subject_summary.csv'))
''')
md('''
## 5. target/non-target保留是否不同？

标签仅用于这里的事后分组。任何保留率差异都不意味着使用标签做拒绝，也不单独证明策略有益或有害。P300分类效果需要后续按原验证划分训练/评估，本轮没有这一步。
''')
code('''
label=events.groupby(['strategy','label']).agg(total=('would_keep','size'),kept=('would_keep','sum'),retention=('would_keep','mean')).reset_index()
display(label)
''')
md('''
## 6. 字符及刺激集合剩余flash数

不因一次拒绝删除整个字符。下面显示每种策略下每字符可用flash数的范围，再列F_05 SE003 Train04作例子。没有加入最小证据量、字符评分或动态停止规则；剩余非零不能等同于证据已经足够。

stimulus_group_id仍是实际同时亮起的字符状态集合签名，原始StimulusCode继续保留。不能把这些签名冒称固定行列组；很多集合只出现一次。
''')
code('''
characters=pd.read_csv(OUT/'character_summary.csv')
display(characters.groupby('strategy').expected_kept.agg(['count','min','median','max']))
display(characters[characters.edf_name.eq('F_05_SE003_CBCol_Train04.edf')])
groups=pd.read_csv(OUT/'stimulus_group_summary.csv')
display(groups[groups.edf_name.eq('F_05_SE003_CBCol_Train04.edf') & groups.strategy.eq('blink_overlap')].head(15))
print('全部逐刺激集合计数：',OUT/'stimulus_group_summary.csv')
''')
md('''
## 7. 触发通道与策略分歧

PTP拒绝记录实际超上限通道；基础故障记录故障通道。眨眼重叠是独立事件关联，不应编造一个PTP触发通道；其supporting_channels可回查原候选表。

候选被判为疑似误报不意味着整段对应epoch一定应该保留，同样，人工确认眨眼也不意味着该epoch一定需要拒绝。本轮先保留三种策略供后续性能对照。
''')
code('''
triggers=pd.read_csv(OUT/'trigger_channels.csv')
display(triggers.groupby(['strategy','channel'])['count'].sum().sort_values(ascending=False).to_frame())
display(pd.read_csv(OUT/'rejection_reasons.csv').groupby(['strategy','reason'])['count'].sum().to_frame())
''')
md('''
## 8. 在两份完整人工检查区间内，具体保留了什么？

只看完全落在人工确认区间内、且共同基础条件通过的epoch：F_05 SE003 Train04 0–35秒、F_08 SE003 Train03 1–50秒；不确定区间排除。人工原始眨眼相交用作描述性标签，不参与三种策略判定。

“无人工眨眼重叠却被眨眼策略拒绝”**不能直接当作epoch级误报**：辅助因果分支的延迟、眨眼恢复尾波及边界差异，都可能令自动区间延伸到人工边界以外。反过来，即使每个人工事件都有候选重叠，自动策略也未必拒绝所有与人工原始边界相交的epoch。该表用于观察代价，不重新定义人工真值。
''')
code('''
review=pd.read_csv(OUT/'reviewed_epoch_summary.csv')
display(review)
print('overlaps_manual_blink=True：epoch与人工疑似眨眼时间段相交；False：完整检查区间内未相交。')
''')
md('''
## 9. 在线可用时间：眨眼重叠不能偷看未来

基础/PTP方案等完整0–800ms样本到齐后即可判定。眨眼候选需要观察恢复和合并等待；即使当前还没有候选，也不能立刻宣布“这个epoch不会与稍后确认的候选重叠”。

本轮为眨眼策略采用**统一保守确认窗口**：epoch样本终点 + 最大候选持续时间1.2秒 + max(恢复0.08秒,合并0.15秒)，各项按整数样本向上取整。256Hz下额外等待约1.355秒。它用于保证可重复的无未来信息对照，并非最短延迟或最终在线策略。

would_keep是回顾性策略结果；policy_decision_seconds才是这套保守政策的输出时间。若文件末尾没有足够数据，pending_at_eof单独标记，不悄悄当作生理故障。本批epoch均有足够记录尾部，因此pending为0。
''')
code('''
display(overall[['strategy','mean_extra_wait_seconds','pending_at_eof']])
display(events.loc[events.strategy.eq('blink_overlap'),['edf_name','event_index','epoch_decision_seconds','policy_decision_seconds','additional_wait_seconds','decision_data_available']].head(8))
''')
md('''
## 10. 三种处理后的平均ERP（描述性）

同一正式因果信号、baseline=None；仅平均各方案保留的完整有限epoch。每条曲线标出参与平均的数量，各通道/类别共享坐标。这里按epoch合并12份EDF，并非每被试等权均值；被试构成、类别数和重叠epoch都会影响曲线。

250µV与基础方案重合是因为本批没有额外拒绝。眨眼策略改变ERP形态不等于提高分类效果，也不证明其拒绝必要。不能根据曲线更“像P300”就选方案。
''')
code('''
display(pd.read_csv(OUT/'pooled_erp_counts.csv'))
display(Image(filename=str(OUT/'pooled_erp.png'),width=1300))
''')
md('''
## 11. 一致性检查与可重复运行

运行：`scripts/compare_epoch_policies.py`；统一配置：`configs/three_policy_comparison.json`；候选检测参数哈希与原8文件清单均检查未变。

结果文件包含全部9054个epoch×每个方案，不覆盖原表。不存在分类分数，后续不合格flash应“不提供分数”，不能填0。13项回归测试覆盖故障条件一致、标签不参与、策略差异、等待与EOF等行为；下面继续检查保存产物的一致性。
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
n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python (bigp3bci)','language':'python'}})
p=ROOT/'notebooks/10_bigp3bci_three_epoch_policy_comparison.ipynb'
nb.write(n,p);print(p)
