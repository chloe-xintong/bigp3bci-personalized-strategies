"""Anonymous single-reviewer MNE annotation browser and independent persistence."""
from pathlib import Path
import shutil
import time
import mne
import numpy as np
import pandas as pd

LABELS = {'BAD':'true_blink', 'BAD_':'true_blink', 'BAD_blink':'true_blink', 'BLINK':'true_blink',
          'RECOVERY':'reverse_recovery', 'OTHER':'other_artifact', 'UNCERTAIN':'uncertain'}


def signature(raw):
    return tuple(zip(map(float, raw.annotations.onset), map(float, raw.annotations.duration), map(str, raw.annotations.description)))


def focused_label(annotations, focus):
    positive = annotations[np.asarray(annotations.duration) > 0]
    invalid = set(positive.description) - set(LABELS)
    if invalid:
        raise ValueError('请使用 BAD / BLINK / RECOVERY / OTHER / UNCERTAIN 标签。')
    hits = {LABELS[str(label)] for on, duration, label in zip(positive.onset, positive.duration, positive.description)
            if on <= focus < on + duration}
    if len(hits) != 1:
        raise ValueError('请用一种标签拖选覆盖灰色定位线的区间；无标记或冲突标记不会写入结论。')
    return next(iter(hits))


def load_clip(root, packet, blind_id):
    root, packet = Path(root), Path(packet)
    if not isinstance(blind_id, str) or not blind_id.strip():
        raise ValueError('请先选择一个复核编号，再点击打开波形。')
    blind_id = blind_id.strip()
    key = pd.read_csv(packet/'coordinator_only/sampling_key.csv')
    hit = key[key.blind_id.eq(blind_id)]
    if len(hit) != 1:
        raise ValueError('复核编号无效。')
    row = hit.iloc[0]
    manifest = pd.read_csv(root/'outputs/blink_annotation_update/annotation_manifest.csv').set_index('edf_name')
    folder = root/manifest.loc[row.edf_name, 'results_path']
    with np.load(folder/'continuous_eeg.npz') as data:
        fs = float(data['sfreq']); center = int(round(row.peak_seconds*fs))
        start, stop = max(0, center-int(4*fs)), min(data['eeg_v'].shape[1], center+int(4*fs)+1)
        values = data['eeg_v'][:,start:stop].copy()
        channels = data['channels'].tolist()
    raw = mne.io.RawArray(values, mne.create_info(channels, fs, 'eeg'), verbose='ERROR')
    saved = packet/'reviewer/annotations'/(blind_id+'-annot.fif')
    draft = packet/'reviewer/annotations'/(blind_id+'-draft-annot.fif')
    if draft.exists() and (not saved.exists() or draft.stat().st_mtime_ns > saved.stat().st_mtime_ns):
        saved = draft
    if saved.exists():
        raw.set_annotations(mne.read_annotations(saved))
    raw._single_review_id = blind_id
    raw._single_review_packet = str(packet.resolve())
    raw._single_review_focus = (center-start)/fs
    return raw


def open_clip(root, packet, blind_id):
    raw = load_clip(root, packet, blind_id)
    return raw, plot_clip(raw, blind_id)


def plot_clip(raw, blind_id):
    focus = raw._single_review_focus
    # Neutral review focus only: no target/non-target events or prior annotations.
    events = np.array([[int(round(focus*raw.info['sfreq'])), 0, 1]])
    with mne.viz.use_browser_backend('matplotlib'):
        browser = raw.plot(duration=raw.n_times/raw.info['sfreq'], n_channels=16,
            events=events, event_id={'review focus':1}, event_color={1:'#888888'},
            scalings={'eeg':100e-6}, remove_dc=False, title=blind_id,
            show=True, block=False)
    return browser


def save_draft(packet, blind_id, raw):
    """Persist drawn intervals even when no valid final decision can be inferred."""
    packet = Path(packet)
    if getattr(raw, '_single_review_id', None) != blind_id or getattr(raw, '_single_review_packet', None) != str(packet.resolve()):
        raise ValueError('当前窗口与所选复核编号不一致；未保存草稿。')
    folder = packet/'reviewer/annotations';folder.mkdir(parents=True,exist_ok=True)
    path = folder/(blind_id+'-draft-annot.fif')
    if path.exists():
        backup = packet/'reviewer/history'/str(time.time_ns());backup.mkdir(parents=True)
        shutil.copy2(path,backup/path.name)
    raw.annotations.save(path,overwrite=True,verbose='ERROR')
    return path


def save_clip(packet, blind_id, raw, confidence='', notes='', no_blink=False):
    packet = Path(packet)
    if getattr(raw, '_single_review_id', None) != blind_id or getattr(raw, '_single_review_packet', None) != str(packet.resolve()):
        raise ValueError('当前窗口与所选复核编号不一致；未保存。')
    annotations = raw.annotations[np.asarray(raw.annotations.duration) > 0]
    if no_blink and len(annotations):
        raise ValueError('当前还有区间标注，未按“未见眨眼”保存。请保留并保存标注，或先明确删除区间。')
    if not len(annotations) and not no_blink:
        raise ValueError('没有画区间。若已检查且未见眨眼，请点击“未见眨眼，保存并下一个”。')
    if set(annotations.description) - set(LABELS):
        raise ValueError('区间标签请使用 BAD_ / BAD / BLINK / RECOVERY / OTHER / UNCERTAIN。')
    focus = raw._single_review_focus
    has_focus = any(on <= focus < on+duration for on,duration in zip(annotations.onset,annotations.duration))
    # Saving freely drawn blink spans does not require a candidate verdict.
    label = 'not_blink' if no_blink else (focused_label(annotations, focus) if has_focus else '')
    if confidence not in ('', 'high', 'medium', 'low'):
        raise ValueError('置信度无效。')
    form_path = packet/'reviewer/review_form.csv'
    form = pd.read_csv(form_path, keep_default_na=False)
    mask = form.blind_id.eq(blind_id)
    if mask.sum() != 1:
        raise ValueError('复核表编号重复或缺失。')
    # Retain revisions and keep existing EDF/manual/candidate annotations untouched.
    folder = packet/'reviewer/annotations'; folder.mkdir(parents=True, exist_ok=True)
    fif = folder/(blind_id+'-annot.fif')
    spans_path = folder/(blind_id+'_spans.csv')
    backup = packet/'reviewer/history'/str(time.time_ns()); backup.mkdir(parents=True)
    for path in (form_path, fif, spans_path):
        if path.exists(): shutil.copy2(path, backup/path.name)
    annotations.save(fif, overwrite=True, verbose='ERROR')
    loaded = mne.read_annotations(fif)
    np.testing.assert_allclose(loaded.onset, annotations.onset, atol=1/raw.info['sfreq'])
    np.testing.assert_allclose(loaded.duration, annotations.duration, atol=1/raw.info['sfreq'])
    pd.DataFrame(dict(blind_id=blind_id, start_seconds=annotations.onset,
        end_seconds=annotations.onset+annotations.duration,
        start_relative_to_focus=annotations.onset-raw._single_review_focus,
        end_relative_to_focus=annotations.onset+annotations.duration-raw._single_review_focus,
        annotation_label=annotations.description)).to_csv(spans_path,index=False,encoding='utf-8-sig')
    form.loc[mask, ['label','confidence','notes']] = [label,confidence,notes]
    if 'span_review_status' not in form:form['span_review_status']=''
    form.loc[mask,'span_review_status']='reviewed_no_blink' if no_blink else ('spans_saved_with_focus_decision' if label else 'spans_saved_no_focus_decision')
    temp = form_path.with_suffix('.tmp.csv')
    form.to_csv(temp,index=False,encoding='utf-8-sig'); temp.replace(form_path)
    return label or 'spans_saved_no_focus_decision'


def review_panel(root, packet, existing_state=None):
    import ipywidgets as w
    from IPython.display import display, clear_output
    packet = Path(packet)
    state = existing_state if existing_state and existing_state.get('raw') is not None else {
        'raw':None,'browser':None,'id':None,'saved':None,'phase':'pilot_30'}
    prior_controls = state.get('_controls', {})
    prior_confidence = prior_controls['confidence'].value if 'confidence' in prior_controls else ''
    prior_notes = prior_controls['notes'].value if 'notes' in prior_controls else ''
    phase = w.Dropdown(options=[('先做 30 个','pilot_30'),('其余 90 个','remaining_90')], description='阶段:')
    item = w.Dropdown(description='编号:',layout=w.Layout(width='450px'))
    confidence = w.Dropdown(options=[('未填写',''),('高','high'),('中','medium'),('低','low')],description='置信度:')
    notes = w.Textarea(description='备注:',layout=w.Layout(width='650px'))
    open_button=w.Button(description='打开波形',button_style='primary')
    save_button=w.Button(description='保存标注',button_style='success')
    draft_button=w.Button(description='先保存草稿',button_style='info')
    next_button=w.Button(description='保存并下一个')
    no_blink_button=w.Button(description='未见眨眼，保存并下一个',layout=w.Layout(width='220px'))
    output=w.Output()
    status=w.HTML(value='尚未保存当前区间。')
    if state['raw'] is not None:
        saved_form=pd.read_csv(packet/'reviewer/review_form.csv',keep_default_na=False).set_index('blind_id')
        current=saved_form.loc[state['id']]
        phase.value=current.phase
        # Old panels did not expose their controls. In that case recover the
        # saved values, while leaving all in-memory annotations untouched.
        confidence.value=prior_confidence if prior_controls else current.confidence
        notes.value=prior_notes if prior_controls else current.notes

    def dirty():
        if state['raw'] is None:return False
        return (signature(state['raw']),confidence.value,notes.value)!=state['saved']

    def choices(*_):
        table=pd.read_csv(packet/'reviewer/review_form.csv',keep_default_na=False)
        table=table[table.phase.eq(phase.value)]
        old=item.value
        item.options=[(f'{r.order:03d} | {r.blind_id}'+(' [已保存]' if r.label or getattr(r,'span_review_status','') else ''),r.blind_id) for r in table.itertuples()]
        # A Dropdown created without options retains value=None when options
        # arrive later; explicitly select a valid ID on initial load/phase change.
        item.value=old if old in table.blind_id.values else (table.blind_id.iloc[0] if len(table) else None)
        open_button.disabled=item.value is None
        next_button.disabled=item.value is None

    def open_current(_):
        with output:
            clear_output(wait=True)
            if item.value is None:print('本阶段没有可复核编号。');return
            if state['raw'] is not None and state['id']==item.value:
                # Reopening the active item must use the in-memory Raw, never
                # replace it with the disk copy or block access to its window.
                if state['browser'] is not None:
                    import matplotlib.pyplot as plt
                    plt.close(state['browser'])
                state['browser']=plot_clip(state['raw'],state['id'])
                print('已重新显示当前内存波形；标注保留。完成后点击“保存标注”，或先保存草稿。')
                return
            if dirty():
                save_draft(packet,state['id'],state['raw'])
                print('已自动保存前一候选的区间草稿；未自动生成复核结论。')
            if state['browser'] is not None:
                import matplotlib.pyplot as plt
                plt.close(state['browser'])
            raw,browser=open_clip(root,packet,item.value)
            state.update(raw=raw,browser=browser,id=item.value)
            row=pd.read_csv(packet/'reviewer/review_form.csv',keep_default_na=False).set_index('blind_id').loc[item.value]
            confidence.value=row.confidence;notes.value=row.notes
            state['saved']=(signature(raw),confidence.value,notes.value)
            print('按 a 打开 annotation；用默认 BAD_ 拖选眨眼区间，再点击保存。不必覆盖灰色定位线。')

    def save_current(_, no_blink=False):
        with output:
            clear_output(wait=True)
            if state['raw'] is None:print('请先打开波形。');return False
            save_draft(packet,state['id'],state['raw'])
            try:
                label=save_clip(packet,state['id'],state['raw'],confidence.value,notes.value,no_blink=no_blink)
            except ValueError as error:
                status.value='<b>区间草稿已保存，正式保存未完成。请查看下方提示。</b>'
                print('画出的区间已保存为草稿，正式结论尚未提交。',str(error));return False
            state['saved']=(signature(state['raw']),confidence.value,notes.value)
            choices()
            count=int((np.asarray(state['raw'].annotations.duration)>0).sum())
            message=(f'已正式保存 {state["id"]}：已检查当前片段，未见眨眼（not_blink）。' if no_blink else
                f'已正式保存 {state["id"]}，共 {count} 段区间。'+('未覆盖定位线，候选结论保持空白。' if label=='spans_saved_no_focus_decision' else f'候选结论：{label}'))
            status.value='<b>'+message+'</b>'
            print(message);return True

    def draft_current(_):
        with output:
            clear_output(wait=True)
            if state['raw'] is None:print('请先打开波形。');return
            save_draft(packet,state['id'],state['raw'])
            print('已保存',state['id'],'区间草稿；正式复核结论不变，可继续编辑或切换候选。')

    def next_current(_, no_blink=False):
        if not save_current(None,no_blink=no_blink):return
        ids=[v for _,v in item.options]
        current=ids.index(state['id'])
        if current+1<len(ids):item.value=ids[current+1];open_current(None)
        else:
            with output: print('本阶段已到最后一个。')

    def change_phase(change):
        if dirty():
            save_draft(packet,state['id'],state['raw'])
            with output:print('已保存当前区间草稿；正式复核结论未自动填写。')
        choices()
        state['phase']=phase.value

    choices()
    if state['id'] in [value for _,value in item.options]:item.value=state['id']
    phase.observe(change_phase,names='value')
    open_button.on_click(open_current);save_button.on_click(save_current);next_button.on_click(next_current)
    draft_button.on_click(draft_current)
    no_blink_button.on_click(lambda _:next_current(None,no_blink=True))
    panel=w.VBox([phase,item,w.HBox([open_button,save_button,next_button,draft_button]),no_blink_button,confidence,notes,status,output])
    state['_controls']={'confidence':confidence,'notes':notes}
    display(panel)
    if state['raw'] is not None:
        with output:print('已接回当前窗口和内存标注。无需重新打开波形，可直接点击“保存标注”。')
    return state
