"""Notebook cells for the local Qt/matplotlib candidate review panel."""
import nbformat as nb


def panel_cells():
 return [nb.v4.new_markdown_cell('''## 3. 交互窗口：抽选文件并标记“多检”

运行下一格出现控制面板，选择EDF、波形模式和候选，点击 **打开/定位候选** 即弹出16通道可滚动窗口。程序使用已安装的PyQt5与MNE matplotlib浏览器，不需要额外安装mne-qt-browser。可先查看原始EEG，再切到analysis/auxiliary对照；检测参数不变。

**两种复核方法：**

- 对完整候选下结论：选择候选后点“多检/不是眨眼”“确认眨眼”或“不确定”。自动单独保存该ID的复核决定，并在内存中加红色REVIEW标记。
- 在窗口里划片段：按 `a` 进入annotation模式，新建标签 `REVIEW_not_blink`（多检），也可使用 `REVIEW_confirmed_blink`、`REVIEW_uncertain`。拖出区间后点面板的“保存窗口复核”。区间包含候选峰时，决定才会关联到该候选ID；没包含峰的片段仍保存，但需你核对对应关系。

**不要直接删除橙色主波来表示多检。** 可以直接使用默认的 `BAD` 标签标记多检，再点击“保存窗口复核”。显示橙色主波、本轮复核标记和flash竖线（红色target、绿色non-target、灰色未知）；旧区间、恢复区间和历史标注隐藏。主波复核单独保存，不覆盖EDF、自动候选或历史人工标注。再次打开会恢复已保存的主波复核标记。已有冲突决定保守记为uncertain，需在CSV中明确更正。

切换窗口前先保存；有未保存的REVIEW区间时程序会阻止切换。只检查选中候选不代表整段完整检查，不会自动增加reviewed_intervals。
'''),nb.v4.new_code_cell('''import ipywidgets as widgets
import importlib
import mne
import bigp3bci_candidate_review as candidate_review
importlib.reload(candidate_review)
from IPython.display import display,clear_output

panel_file=widgets.Dropdown(options=INDEX.edf_name.tolist(),value=EDF_NAME,description='EDF:',layout=widgets.Layout(width='560px'))
panel_mode=widgets.Dropdown(options=['raw','analysis','auxiliary'],value='raw',description='波形:')
panel_candidate=widgets.Dropdown(description='候选:',layout=widgets.Layout(width='560px'))
panel_notes=widgets.Text(value='',description='备注:',layout=widgets.Layout(width='650px'))
open_button=widgets.Button(description='打开/定位候选',button_style='primary')
false_button=widgets.Button(description='多检/不是眨眼',button_style='danger')
true_button=widgets.Button(description='确认眨眼',button_style='success')
uncertain_button=widgets.Button(description='不确定')
save_button=widgets.Button(description='保存窗口复核',button_style='info')
panel_output=widgets.Output()
panel_state={'raw':None,'browser':None,'edf':None,'saved_signature':None,'candidates':None}

def _span_signature(raw):
    if raw is None:return ()
    return tuple((round(float(a),6),round(float(d),6),str(label))
                 for a,d,label in zip(raw.annotations.onset,raw.annotations.duration,raw.annotations.description)
                 if candidate_review.normalize_review_label(label) is not None and d>0)

def _refresh_candidates(change=None):
    entry=INDEX[INDEX.edf_name.eq(panel_file.value)].iloc[0]
    c=pd.read_csv(ROOT/entry.candidate_details_csv)
    boundary=pd.read_csv(ROOT/'outputs/blink_boundary_localization_v1'/Path(panel_file.value).stem/'boundaries.csv')
    for target,source in [('onset_seconds','main_onset_seconds'),('peak_seconds','main_peak_seconds'),('offset_seconds','main_offset_seconds')]:c[target]=boundary[source].to_numpy(float)
    panel_state['candidates']=c
    panel_candidate.options=[(f'{r.blink_candidate_id.split("_B")[-1]} | {r.onset_seconds:.3f}–{r.offset_seconds:.3f}s'+(' 启动期' if r.filter_startup else ''),r.blink_candidate_id) for r in c.itertuples()]
    eligible=c[c.filter_startup.eq(False)]
    if len(eligible):panel_candidate.value=eligible.iloc[0].blink_candidate_id

panel_file.observe(_refresh_candidates,names='value')
_refresh_candidates()

def _selected_candidate():
    c=panel_state['candidates']
    selected=c[c.blink_candidate_id.eq(panel_candidate.value)]
    return None if selected.empty else selected.iloc[0]

def _open_window(button):
    with panel_output:
        clear_output(wait=True)
        try:
            current=panel_state['raw']
            if current is not None and _span_signature(current)!=panel_state['saved_signature']:
                raise RuntimeError('当前窗口有未保存REVIEW区间，请先点“保存窗口复核”。')
            get_ipython().run_line_magic('matplotlib','qt')
            mne.viz.set_browser_backend('matplotlib')
            if panel_state['browser'] is not None:
                plt.close(panel_state['browser'])
            if current is not None:current.close()
            selected=_selected_candidate()
            start=START_SECONDS if selected is None else max(0.,selected.onset_seconds-2.)
            raw,browser,_,_=candidate_review.open_candidate_review(ROOT,panel_file.value,mode=panel_mode.value,start=start,duration=12.,include_manual=False,main_only=True)
            panel_state.update(raw=raw,browser=browser,edf=panel_file.value,saved_signature=_span_signature(raw))
            print(f'已打开 {panel_file.value} / {panel_mode.value}，起点 {start:.3f}s。')
            print('按a进入标注；新建REVIEW_not_blink标多检，完成后点保存窗口复核。')
        except Exception as exc:print(type(exc).__name__+': '+str(exc))

def _save_window(button=None):
    raw=panel_state['raw']
    if raw is None:raise RuntimeError('请先打开一个文件窗口。')
    path,annotation,spans=candidate_review.save_review_spans(ROOT,panel_state['edf'],raw)
    panel_state['saved_signature']=_span_signature(raw)
    print('已保存：',path)
    print('本轮人工区间：',annotation)
    if len(spans):display(spans.tail(8))
    empty=spans[spans.blink_candidate_ids.eq('')]
    if len(empty):print('注意：',len(empty),'个区间未包含任何候选峰，已保留区间但未写候选决定。')

def _save_clicked(button):
    with panel_output:
        clear_output(wait=True)
        try:_save_window()
        except Exception as exc:print(type(exc).__name__+': '+str(exc))

def _decide(label):
    with panel_output:
        clear_output(wait=True)
        try:
            if panel_state['raw'] is None or panel_state['edf']!=panel_file.value:
                raise RuntimeError('请先打开当前选择的EDF。')
            selected=_selected_candidate()
            if selected is None:raise RuntimeError('当前文件没有候选。')
            raw=panel_state['raw']
            raw.set_annotations(raw.annotations+mne.Annotations([selected.onset_seconds],[selected.offset_seconds-selected.onset_seconds],[label],orig_time=raw.annotations.orig_time))
            _save_window()
            if panel_notes.value.strip():
                entry=INDEX[INDEX.edf_name.eq(panel_file.value)].iloc[0]
                path=ROOT/entry.review_decisions_csv
                d=pd.read_csv(path,keep_default_na=False)
                mask=d.blink_candidate_id.eq(selected.blink_candidate_id)
                d.loc[mask,'review_notes']=d.loc[mask,'review_notes'].astype(str)+'; '+panel_notes.value.strip()
                d.to_csv(path,index=False)
            print(selected.blink_candidate_id,'→',candidate_review.REVIEW_DECISIONS[label])
            print('可选择下一候选，点击“打开/定位候选”跳转。')
        except Exception as exc:print(type(exc).__name__+': '+str(exc))

open_button.on_click(_open_window)
save_button.on_click(_save_clicked)
false_button.on_click(lambda b:_decide('REVIEW_not_blink'))
true_button.on_click(lambda b:_decide('REVIEW_confirmed_blink'))
uncertain_button.on_click(lambda b:_decide('REVIEW_uncertain'))
display(widgets.VBox([panel_file,panel_mode,panel_candidate,panel_notes,
    widgets.HBox([open_button,save_button]),widgets.HBox([false_button,true_button,uncertain_button]),panel_output]))
''')]
