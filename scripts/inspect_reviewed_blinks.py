"""Inspect unmatched blink candidates inside explicitly reviewed 0–35 seconds."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('_MNE_FAKE_HOME_DIR',str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat as nb
from bigp3bci_continuous_qc import read_bigp3bci_edf
STEM='F_05_SE003_CBCol_Train04'
OUT=ROOT/'outputs/blink_annotation_update/reviewed_0_35';OUT.mkdir(parents=True,exist_ok=True)
BASE=ROOT/'outputs/blink_annotation_update/additional_edfs'/STEM
b=pd.read_csv(BASE/'blink_candidates.csv')
m=pd.read_csv(ROOT/'outputs/artifact_qc/continuous_manual_qc/manual/manual_review_tables'/f'{STEM}_manual_review.csv')
rows=[]
for r in b[b.onset_seconds<35].itertuples():
 hits=m[(m.start_sec<r.offset_seconds)&(m.end_sec>r.onset_seconds)]
 rows.append(dict(blink_candidate_id=r.blink_candidate_id,onset_seconds=r.onset_seconds,peak_seconds=r.peak_seconds,
                  offset_seconds=r.offset_seconds,decision_seconds=r.decision_seconds,startup=r.filter_startup,
                  fully_within_review=r.onset_seconds>=0 and r.offset_seconds<=35,
                  decision_by_review_end=r.decision_seconds<=35,matched_manual_ids=';'.join(hits.manual_id),
                  unmatched_to_complete_annotation=not len(hits),frontal_amplitude_uv=r.frontal_amplitude_uv,
                  posterior_amplitude_uv=r.posterior_amplitude_uv,frontal_posterior_metric=r.frontal_posterior_metric))
t=pd.DataFrame(rows);t.to_csv(OUT/'reviewed_candidates.csv',index=False)
unmatched=t[t.unmatched_to_complete_annotation & t.fully_within_review]
unmatched.to_csv(OUT/'suspected_false_positive_candidates.csv',index=False)
with np.load(BASE/'continuous_eeg.npz') as d:y=d['eeg_v'];aux=d['auxiliary_eeg_v'];channels=d['channels'].tolist();fs=float(d['sfreq'])
raw=read_bigp3bci_edf(ROOT/'data/raw/bigP3BCI-data/StudyF/F_05/SE003/Train/CBCol'/f'{STEM}.edf')
x=raw.get_data(picks=channels);raw.close()
events=pd.read_csv(BASE/'epoch_candidates.csv')
for r in unmatched.itertuples():
 a=max(0,int((r.onset_seconds-1)*fs));end=min(y.shape[1],int((r.offset_seconds+1)*fs))
 fig,axes=plt.subplots(4,1,figsize=(12,9),sharex=True,layout='constrained')
 for ch in ['EEG_F3','EEG_Fz','EEG_F4']:
  axes[0].plot(np.arange(a,end)/fs,x[channels.index(ch),a:end]*1e6,label=ch)
  axes[1].plot(np.arange(a,end)/fs,y[channels.index(ch),a:end]*1e6,label=ch)
 for ch in ['EEG_Pz','EEG_PO7','EEG_PO8','EEG_Oz']:axes[2].plot(np.arange(a,end)/fs,y[channels.index(ch),a:end]*1e6,label=ch)
 axes[3].plot(np.arange(a,end)/fs,np.median(aux[[channels.index(ch) for ch in ['EEG_F3','EEG_Fz','EEG_F4']],a:end],axis=0)*1e6,label='auxiliary frontal median')
 for ax,label in zip(axes,['raw frontal','analysis frontal','analysis posterior','auxiliary slow']):
  for c in b[(b.onset_seconds<end/fs)&(b.offset_seconds>a/fs)].itertuples():ax.axvspan(c.onset_seconds,c.offset_seconds,color='orange',alpha=.16)
  for human in m[(m.start_sec<end/fs)&(m.end_sec>a/fs)].itertuples():ax.axvspan(human.start_sec,human.end_sec,color='green',alpha=.15)
  ax.axvline(r.peak_seconds,color='red',ls='--');ax.axvline(r.decision_seconds,color='purple',ls=':')
  for flash in events.loc[(events.onset_seconds>=a/fs)&(events.onset_seconds<end/fs),'onset_seconds']:ax.axvline(flash,color='gray',lw=.4,alpha=.5)
  ax.set(ylabel=label+' (µV)',xlim=(a/fs,end/fs));ax.legend(fontsize=7,loc='upper right')
 axes[-1].set_xlabel('EDF time (s)')
 fig.suptitle(f'{r.blink_candidate_id}: unmatched; green=manual, orange=automatic; startup={r.startup}')
 fig.savefig(OUT/(r.blink_candidate_id+'.png'),dpi=120);plt.close(fig)
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(s))
def code(s):cells.append(nb.v4.new_code_cell(s))
md('''# 09 · F_05 SE003 Train04：完整标注0–35秒的疑似误报核查

用户明确确认0–35秒已完整标注，已将该覆盖范围存入reviewed_intervals.csv；不是程序自行推断正常区间。只在此范围内对照。眨眼候选仍不自动拒绝epoch，检测参数不变。

10–35秒：12段人工疑似眨眼均有相交候选；15个自动候选中3个未匹配。相对人工参考的候选匹配比例为12/15=80%，未匹配比例为20%；**20%不是样本级假阳性率**。事件级人工覆盖为12/12，只反映这一已复核训练区间，不能推广到全文件或其他被试。匹配采用任意正时长相交，并非严格边界准确。

0–10秒另有1个未匹配候选，属于启动保护区，单独展示，不混入上述统计。无候选跨越35秒边界，也没有本区间候选需要等到35秒以后才能判定。
''')
code("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display,Image\nROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'configs/expanded_pilot.json').exists())\nOUT=ROOT/'outputs/blink_annotation_update/reviewed_0_35'\nt=pd.read_csv(OUT/'reviewed_candidates.csv')\ndisplay(t)\ndisplay(pd.read_csv(OUT/'suspected_false_positive_candidates.csv'))")
md('''## 未匹配片段波形

每图四行依次为原始额区、正式分析额区、正式分析后部、辅助慢分支。绿色=人工标注，橙色=自动候选，红虚线=此未匹配候选的峰，紫点线=其判定时间，灰竖线=flash。

这些是相对于完整人工标注的**疑似误报**，仍应复核是否是眨眼后的反向恢复波被拆成第二个候选、其他共同额区变化，或人工边界太窄。辅助因果分支增加的延迟可能造成边界错位；不能仅靠未相交就断定具体原因。
''')
for r in unmatched.itertuples():
 md(f'### {r.onset_seconds:.3f}–{r.offset_seconds:.3f}秒'+('（启动期，单独报告）' if r.startup else ''))
 code(f"display(Image(filename=str(OUT/'{r.blink_candidate_id}.png'),width=1100))")
md('''## 当前判断

存在未匹配候选，说明“覆盖已标眨眼很多”不代表没有误报。先针对这3段复核波形，再决定是否需要调整恢复、合并或慢波形态条件；不直接降低幅值阈值，也不为这35秒单独调出最优参数。其他未完整复核的区间仍属于未知。

重新计算并画图：`scripts/inspect_reviewed_blinks.py`。最新全文件人工对照：notebook08。''')
n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python (bigp3bci)','language':'python'}})
nb.write(n,ROOT/'notebooks/09_bigp3bci_reviewed_blink_false_positives.ipynb')
assert len(t[t.startup.eq(False)])==15 and len(unmatched[~unmatched.startup])==3
print(unmatched[['onset_seconds','offset_seconds','startup']].to_string(index=False))
