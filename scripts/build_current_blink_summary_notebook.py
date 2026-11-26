"""Build the current all-file blink detection and annotation summary notebook."""
from pathlib import Path
import textwrap
import nbformat as nb
from notebook_reports import write_report
ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(textwrap.dedent(s).strip()))
def code(s):cells.append(nb.v4.new_code_cell(textwrap.dedent(s).strip()))
md('''
# 08 · 最新人工眨眼标注与全部检测汇总

读取本轮更新后的全部人工CSV/FIF，并对已有检测结果作对照。原固定8文件pilot保持不变；对有人工标注、但不在固定pilot内的训练EDF，使用**完全相同的参数**补充检测，另存在 `outputs/blink_annotation_update/additional_edfs`。

这些补充文件由人工标注可用性选择，不能当作随机抽样或独立盲测。本轮不调任何阈值，不删除epoch，也不把候选当作已确认眨眼。旧05/06/07中的人工对照属于当时快照，本notebook是当前汇总入口。
''')
code('''
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display,Markdown
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'configs/expanded_pilot.json').exists())
OUT=ROOT/'outputs/blink_annotation_update'
summary=pd.read_csv(OUT/'annotation_update_summary.csv')
all_files=pd.read_csv(OUT/'all_processed_detection_summary.csv')
detail=pd.read_csv(OUT/'manual_candidate_details.csv')
pairs=pd.read_csv(OUT/'manual_candidate_pairs.csv')
record=json.loads((OUT/'analysis_record.json').read_text())
plt.rcParams.update({'figure.dpi':110,'axes.spines.top':False,'axes.spines.right':False})
display(pd.Series(record,name='current snapshot').to_frame())
''')
md('''
## 1. 六份已保存标注逐文件汇总

每份CSV与FIF时间/标签一致性均检查。`MANUAL_bad_span`按用户说明视为疑似眨眼时间段，不是永久坏通道。人工标注与检测候选可以多对多相交。

下表同时给出全部人工段数、启动期段数、非启动期段数和与非启动期候选相交的段数。相交只需有正时长重叠；这不是严格边界吻合，也不是正式召回率。
''')
code('''
cols=['edf_name','in_frozen_pilot','suspected_manual_spans','startup_spans','nonstartup_spans',
      'nonstartup_spans_with_overlap','nonstartup_spans_without_overlap','auto_candidates','source_consistency']
display(summary[cols])
print('非启动期已标段与候选重叠：',summary.nonstartup_spans_with_overlap.sum(),'/',summary.nonstartup_spans.sum())
fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
x=np.arange(len(summary));matched=summary.nonstartup_spans_with_overlap
ax.bar(x,matched,label='overlapping annotated spans',color='tab:green')
ax.bar(x,summary.nonstartup_spans_without_overlap,bottom=matched,label='no overlap: needs review',color='tab:orange')
ax.set(xticks=x,xticklabels=[f'{r.edf_name.split("_CBCol")[0]}\\n{r.edf_name.split("_CBCol_")[1].removesuffix(".edf")}' for r in summary.itertuples()],ylabel='non-startup annotated spans',title='Descriptive overlap; not validated recall')
ax.legend();plt.show()
''')
md('''
## 2. 尚未与候选重叠的人工片段

下面列出所有非启动期的未相交标注。它们值得逐段复核，但在稀疏、疑似眨眼标注条件下，不能直接宣布为确定漏检。辅助因果分支会改变波形时序，狭窄的人工区间也可能与候选边界错开。

max_overlapping_epoch_ptp_uv是与人工区间相交的epoch里、全通道最大PTP，不是眨眼波形自身幅值，不能直接用于重设眨眼检测门槛。
''')
code('''
unmatched=pd.read_csv(OUT/'unmatched_nonstartup_manual.csv')
display(unmatched[['edf_name','manual_id','start_sec','end_sec','overlap_epochs','max_overlapping_epoch_ptp_uv']])
print('未重叠详细记录：',OUT/'unmatched_nonstartup_manual.csv')
''')
md('''
## 3. 人工区间覆盖程度与边界差

某候选与人工区间只相交一点也会计为“有重叠”，因此补看人工区间有多少时长被候选覆盖，以及相交对的起止差。注意人工标注在原始信号域，候选在额外因果低通分支上，边界差混合了滤波相位和人工边界误差，不是已验证定位精度。

同一个候选可能跨多个人工标注，同一人工标注也可能与多个候选相交；下表按配对描述，不能把配对数当作独立事件数。
''')
code('''
d=detail[~detail.startup]
display(d.groupby('edf_name').manual_duration_covered_fraction.agg(['count','min','median','mean','max']))
display(pairs.groupby('edf_name')[['overlap_seconds','onset_difference_seconds','offset_difference_seconds']].agg(['count','median','min','max']))
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
axes[0].hist(d.manual_duration_covered_fraction,bins=np.linspace(0,1,21))
axes[0].set(xlabel='annotated duration fraction covered',ylabel='annotated spans',title='Partial overlap is not full coverage')
axes[1].hist(pairs.onset_difference_seconds,bins=30)
axes[1].set(xlabel='candidate onset - manual onset (s)',ylabel='overlap pairs',title='Different signal domains; descriptive only')
plt.show()
''')
md('''
## 4. 所有已处理文件：固定pilot与补充文件分开

这张表覆盖原8份文件和本次4份补充训练文件，不重复计算已在pilot内的已标注文件。manually_annotated=False表示没有找到当前人工标注，并不表示正常。没有候选也不代表该文件无眨眼。

仅汇总候选数、关联epoch数、启动epoch与基础跳变。站点全部仍为inferred，未按候选结果改动原固定文件清单。
''')
code('''
display(all_files[['edf_name','site','in_frozen_pilot','manually_annotated','flashes','auto_candidates',
                   'nonstartup_auto_candidates','epochs_with_candidate','filter_startup_epochs','jump_candidates','other_basic_candidates']])
for name,g in all_files.groupby('in_frozen_pilot'):
    print('原固定pilot' if name else '人工标注补充文件',':',len(g),'files;',g.flashes.sum(),'flashes;',g.auto_candidates.sum(),'candidates')
''')
md('''
## 5. 自动候选没有人工重叠，如何解释？

下表给出已标注文件内的自动候选数，以及没有与任何已保存标注相交的数量。这些候选可能是未标出的眨眼，也可能是其他额区变化；**不能称为误报，不能算精确率**。同样，人工有重叠的候选也只是疑似匹配，不是确诊眼动来源。

要评价漏检和误报，需要完整检查的连续区间及明确不确定片段。当前reviewed_intervals仍缺少，程序没有生成ground truth。
''')
code('''
display(summary[['edf_name','nonstartup_auto_candidates','nonstartup_auto_candidates_overlapping_manual',
                 'auto_candidates_without_manual_overlap','complete_reviewed_intervals','reviewed_event_recall','reviewed_candidate_match_fraction']])
print('注意auto_candidates_without_manual_overlap包含启动期候选；其他两列明确仅非启动期。')
''')
md('''
## 6. PTP方案模拟与眨眼候选分开

下表仍只模拟150/250µV上限与无上限方案的预计有效epoch数。基础条件相同；眨眼候选不参与拒绝。no_upper也仍保留启动、不完整、非有限、恢复、原始平线与过低PTP等条件。

新增标注只帮助检查检测器，不意味着需要调PTP上限。PTP不是眨眼来源特异指标，不能因希望覆盖所有眨眼而直接降低到大量拒绝的数值。
''')
code('''
display(all_files[['edf_name','flashes','all16_upper_150uv_expected_kept','all16_upper_250uv_expected_kept','all16_no_upper_expected_kept']])
params=json.loads((ROOT/'configs/expanded_pilot.json').read_text())
display(pd.Series(params['blink_detector'],name='unchanged candidate parameter').to_frame())
''')
md('''
## 7. 当前结论与一致性检查

当前不自动调整阈值。先看新增文件是否重复出现特定的未匹配波形或边界模式，再用完整检查的区间评估降低门槛的代价。已标正例的高重叠比例本身不足以确定参数最优。

同一F_05多个Session并非多个独立被试；六份标注集中在少数被试，跨被试泛化证据仍有限。补充文件因有人工标注而被选入，也不是独立盲测。

重复运行：`D:/Anaconda/envs/bigp3bci/python.exe scripts/check_blink_annotation_updates.py`。检测参数和原固定8文件manifest的hash在处理前后保持一致；旧检测结果可复用，新增文件的原始EDF处理前后hash同样保存。下列检查只验证产物一致性，不证明检测有效。
''')
code('''
assert summary.source_consistency.eq('CSV_FIF_agree').all()
assert summary.suspected_manual_spans.sum()==len(detail)
assert summary.nonstartup_spans.sum()==detail.startup.eq(False).sum()
assert summary.nonstartup_spans_with_overlap.sum()==((~detail.startup)&detail.has_nonstartup_candidate_overlap).sum()
assert len(all_files)==all_files.edf_name.nunique()
assert record['config_unchanged'] and record['frozen_manifest_unchanged']
unreviewed=summary.complete_reviewed_intervals.eq(0)
assert summary.loc[unreviewed,'reviewed_event_recall'].isna().all()
assert summary.loc[unreviewed,'reviewed_candidate_match_fraction'].isna().all()
print('PASS: annotation sources, counts, unique files, unchanged parameters and unavailable validation metrics')
''')
md("""## 8. 已完整复核的 F_08 SE003 Train03：约第 1–50 秒

该文件有一段完整人工复核区间，共 49 秒。前 10 秒是滤波启动期，不作评估；在 10–50 秒内，4 个人工眨眼与 4 个自动候选一一匹配，没有漏检，也没有多余候选。

49.844–50.852 秒的候选，其判定时刻 51.004 秒已超出复核区间终点，属于边界事件；对应的人工片段 49.720–50.055 秒同样按边界处理，二者都不计入评估。

1–10 秒处于滤波启动期：1.527–2.000 秒的候选被标为启动期；人工片段 2.316–2.630 秒和 3.395–3.826 秒没有候选覆盖，但同样位于启动期内，不计为漏检。

该区间内共有 4 个可评估的人工眨眼，全部被检出。
""")
code("""f08=OUT/'per_edf/F_08_SE003_CBCol_Train03'
display(pd.Series(json.loads((f08/'manual_validation_status.json').read_text()),name='reviewed F08 status').to_frame())
c=pd.read_csv(f08/'manual_candidate_comparison.csv')
display(c.loc[c.onset_seconds<51,['blink_candidate_id','onset_seconds','offset_seconds','decision_seconds','filter_startup','fully_reviewed','evaluation_status']])
m=pd.read_csv(f08/'manual_event_comparison.csv')
display(m.loc[m.start_sec<51,['manual_id','start_sec','end_sec','matching_candidate_ids','evaluation_status']])
assert c.loc[c.evaluation_eligible,'evaluation_status'].eq('matched').all()
""")
n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python (bigp3bci)','language':'python'}})
p=ROOT/'notebooks/08_bigp3bci_current_blink_detection_summary.ipynb'
write_report(n,p);print(p)
