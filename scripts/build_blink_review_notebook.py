"""Build UTF-8 notebook 11: main-wave-only interactive candidate review."""
from pathlib import Path
import nbformat as nb
from notebook_reports import write_report
from blink_review_panel_cells import panel_cells
ROOT=Path(__file__).resolve().parents[1]
md=nb.v4.new_markdown_cell
code=nb.v4.new_code_cell
cells=[md('''# 11 · 主波眨眼候选复核

窗口显示橙色主波和本轮复核标记，并恢复flash事件竖线：红色为target，绿色为non-target，灰色为未知标签。旧候选区间、恢复区间和历史标注仍隐藏。使用 `BAD` 标记你认为多检的部分，再点击“保存窗口复核”。

原始EDF、自动检测参数和已有人工标注不变。主波边界仍是待复核估计，并不代表主波结束后信号一定干净。如果旧窗口还有未保存标记，请先使用下面的补存单元，不要先重启内核或运行全部单元。
'''),md('''## 已经用BAD画过标记：先补存

仅在原来的内核和窗口仍在时，单独运行下一格。它重新加载已修复的保存函数，尝试保存当前窗口中的 `BAD`、`bad`、`BAD_*` 或 `REVIEW_*` 标记。

如果提示没有活动窗口或没有标记，就没有保存成功；磁盘空文件不能恢复已经丢失的内存标记。补存成功后，再重新运行第3节面板。
'''),code('''from pathlib import Path
import sys,importlib
_rescue_root=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'configs/expanded_pilot.json').exists())
if str(_rescue_root/'src') not in sys.path:sys.path.insert(0,str(_rescue_root/'src'))
import bigp3bci_candidate_review as candidate_review
importlib.reload(candidate_review)
_existing_panel=globals().get('panel_state',{})
_existing_raw=_existing_panel.get('raw')
if _existing_raw is None:
    _existing_raw=globals().get('browser_raw')
if _existing_raw is None:
    print('未找到当前内核中的复核窗口；本单元没有写文件。首次使用可继续运行后面的单元。')
else:
    from collections import Counter
    print('当前窗口标签：',dict(Counter(map(str,_existing_raw.annotations.description))))
    _n=sum(candidate_review.normalize_review_label(label) is not None and duration>0 for label,duration in zip(_existing_raw.annotations.description,_existing_raw.annotations.duration))
    if not _n:
        print('当前对象没有可保存的BAD或REVIEW标记，未覆盖磁盘文件。')
    else:
        _edf=getattr(_existing_raw,'_blink_review_edf',None)
        _csv,_fif,_spans=candidate_review.save_review_spans(_rescue_root,_edf,_existing_raw)
        if _existing_panel.get('raw') is _existing_raw and '_span_signature' in globals():
            _existing_panel['saved_signature']=_span_signature(_existing_raw)
        print('成功补存',_n,'段：',_fif)
        print('候选复核表：',_csv)
        display(_spans)
'''),code('''from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'configs/expanded_pilot.json').exists())
if str(ROOT/'src') not in sys.path:sys.path.insert(0,str(ROOT/'src'))
from bigp3bci_candidate_review import load_candidate_review,open_candidate_review
INDEX=pd.read_csv(ROOT/'outputs/blink_candidate_review/file_index.csv')
display(INDEX[['edf_name','candidate_count','nonstartup_candidates','annotation_reload_verified']])
'''),md('''## 1. 选择文件和波形

这里只叠加新版主波。`raw` 为原始EEG，便于查看上升和回落；`analysis` 为正式因果滤波信号，`auxiliary` 为旧检测器的辅助分支。切换波形不会移动标注或修改参数。

下面先生成静态预览；交互窗口在第3节打开。
'''),code('''EDF_NAME='F_05_SE003_CBCol_Train04.edf'
MODE='raw'
START_SECONDS=10.0
DURATION_SECONDS=12.0
review_raw,flash_events,candidates,review_csv=load_candidate_review(ROOT,EDF_NAME,MODE,include_manual=False,main_only=True)
display(candidates[['blink_candidate_id','onset_seconds','peak_seconds','offset_seconds','filter_startup','localization_flags']].head(20))
print('候选复核表：',review_csv)
'''),md('''## 2. 主波静态预览

橙色区间为暂定主波；其他彩色区间仅为本轮已保存的复核标记。主波边界仍需人工检查，不表示其后恢复区间没有污染。
'''),code('''fs=float(review_raw.info['sfreq'])
a=round(START_SECONDS*fs);b=min(review_raw.n_times,a+round(DURATION_SECONDS*fs))
fig,axes=plt.subplots(2,1,figsize=(14,6),sharex=True,layout='constrained')
for ax,names in zip(axes,[['EEG_F3','EEG_Fz','EEG_F4'],['EEG_Pz','EEG_PO7','EEG_PO8','EEG_Oz']]):
    values=review_raw.get_data(picks=names,start=a,stop=b)*1e6
    for name,v in zip(names,values):ax.plot(np.arange(a,b)/fs,v,label=name)
    for on,duration,label in zip(review_raw.annotations.onset,review_raw.annotations.duration,review_raw.annotations.description):
        if on<b/fs and on+duration>a/fs and str(label).startswith(('CANDIDATE_blink_main','REVIEW_')):
            ax.axvspan(on,on+duration,alpha=.14,color='orange' if label=='CANDIDATE_blink_main' else 'red')
    for sample,_,event_code in flash_events:
        if a<=sample<b:
            ax.axvline(sample/fs,color={1:'green',2:'red',3:'gray'}.get(int(event_code),'gray'),lw=.6,alpha=.65)
    ax.set(ylabel='µV',xlim=(a/fs,b/fs));ax.legend(fontsize=8)
axes[-1].set_xlabel('EDF time (s)')
fig.suptitle(f'{EDF_NAME}: {MODE}; main wave only')
plt.show()
''')]
cells.extend(panel_cells())
cells.extend([md('''## 4. 保存位置

主波复核单独保存在：

- `*-human-main-review-annot.fif`：本轮人工复核时间段。
- `human_main_review_spans.csv`：复核时间段及对应的候选ID。
- `review_decisions.csv`：按候选ID记录复核决定，`review_region=main` 表示复核的是主波。

旧复核文件不会被覆盖。切换文件或关闭窗口前，请先点击“保存窗口复核”，并确认下方显示保存成功。
'''),code('''display(pd.read_csv(review_csv,keep_default_na=False).head(15))
review_raw.close()
assert INDEX.annotation_reload_verified.all()
print('主波复核入口已准备好，原始EDF及历史标注未修改。')
''')])
n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python (bigp3bci)','language':'python'}})
p=ROOT/'notebooks/11_bigp3bci_blink_candidate_review.ipynb'
write_report(n,p)
print(p)
