"""Read-only candidate overlay viewers. No EDF or existing annotation writes."""
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from bigp3bci_continuous_qc import read_bigp3bci_edf,get_all_eeg_channels,stimulus_begin_events,flash_event_display_settings


def load_candidate_review(root,edf_name,mode='analysis',include_manual=True,main_only=False):
    root=Path(root)
    index=pd.read_csv(root/'outputs/blink_candidate_review/file_index.csv')
    row=index[index.edf_name.eq(edf_name)]
    if len(row)!=1:raise ValueError(f'Choose exactly one exported EDF: {edf_name}')
    row=row.iloc[0]
    raw=read_bigp3bci_edf(row.source_edf)
    if raw.first_samp!=0:raise ValueError('This overlay export expects the un-cropped EDF sample origin')
    channels=get_all_eeg_channels(raw)
    flashes=stimulus_begin_events(raw)
    if mode in ('analysis','auxiliary'):
        with np.load(root/row.processing_results/'continuous_eeg.npz') as data:
            if data['channels'].tolist()!=channels:raise ValueError('Saved signal channels differ')
            if data['eeg_v'].shape[1]!=raw.n_times or float(data['sfreq'])!=raw.info['sfreq']:raise ValueError('Saved sample grid differs')
            values=data['eeg_v'] if mode=='analysis' else data['auxiliary_eeg_v']
        info=raw.copy().pick(channels).info
        original=raw;raw=mne.io.RawArray(values,info,verbose='ERROR');original.close()
    elif mode=='raw':raw.pick(channels)
    else:raw.close();raise ValueError('mode must be raw, analysis or auxiliary')
    candidates=pd.read_csv(root/row.candidate_details_csv)
    if main_only:
        boundaries=pd.read_csv(root/'outputs/blink_boundary_localization_v1'/Path(edf_name).stem/'boundaries.csv')
        if candidates.blink_candidate_id.tolist()!=boundaries.blink_candidate_id.tolist():raise ValueError('Main boundary IDs differ')
        for target,source in [('onset_seconds','main_onset_seconds'),('peak_seconds','main_peak_seconds'),('offset_seconds','main_offset_seconds')]:
            candidates[target]=boundaries[source].to_numpy(float)
        candidates['localization_flags']=boundaries.localization_flags.fillna('').to_numpy()
        new=mne.Annotations(candidates.onset_seconds.to_numpy(float),candidates.offset_seconds.to_numpy(float)-candidates.onset_seconds.to_numpy(float),['CANDIDATE_blink_main']*len(candidates),orig_time=raw.annotations.orig_time)
    else:
        ann=mne.read_annotations(root/row.annotation_fif)
        new=mne.Annotations(ann.onset,ann.duration,ann.description,orig_time=raw.annotations.orig_time)
    raw.set_annotations(raw.annotations+new)
    manual=root/'outputs/artifact_qc/continuous_manual_qc/manual/manual_annotations'/(Path(edf_name).stem+'-manual-annot.fif')
    if include_manual and not main_only and manual.exists():
        saved=mne.read_annotations(manual)
        selected=np.array([str(label).startswith('MANUAL_') for label in saved.description])
        saved=saved[selected]
        raw.set_annotations(raw.annotations+mne.Annotations(saved.onset,saved.duration,saved.description,orig_time=raw.annotations.orig_time))
    review_copy=root/'outputs/blink_candidate_review'/Path(edf_name).stem/(Path(edf_name).stem+('-human-main-review-annot.fif' if main_only else '-human-review-annot.fif'))
    if review_copy.exists():
        reviewed=mne.read_annotations(review_copy)
        raw.set_annotations(raw.annotations+mne.Annotations(reviewed.onset,reviewed.duration,reviewed.description,orig_time=raw.annotations.orig_time))
    raw._blink_review_edf=edf_name
    raw._blink_review_main_only=main_only
    raw._blink_review_mode=mode
    return raw,flashes,candidates,root/row.review_decisions_csv


def open_candidate_review(root,edf_name,mode='analysis',start=0.,duration=15.,include_manual=True,main_only=False):
    raw,flashes,candidates,path=load_candidate_review(root,edf_name,mode,include_manual,main_only=main_only)
    event_id,colors=flash_event_display_settings(flashes)
    ann_colors={str(label):('#ef8a17' if str(label).startswith('CANDIDATE_blink') else ('#cc3344' if str(label).startswith('REVIEW_') else '#33a65c')) for label in raw.annotations.description if str(label).startswith(('CANDIDATE_','MANUAL_','REVIEW_'))}
    browser=raw.plot(start=max(0.,float(start)),duration=duration,picks='eeg',n_channels=16,events=flashes,
        event_id=event_id,event_color=colors,remove_dc=False,scalings={'eeg':100e-6},
        annotation_colors=ann_colors,annotation_regex=r'^(CANDIDATE_blink_main$|REVIEW_|(?i:BAD)(?:_|$))' if main_only else r'^(CANDIDATE_blink|MANUAL_|REVIEW_|(?i:BAD)(?:_|$))',
        title=f'{edf_name}: {mode}; main wave only' if main_only else f'{edf_name}: {mode}; orange=automatic, green=manual; display scale is not a threshold',
        show=True,block=False)
    return raw,browser,candidates,path


REVIEW_DECISIONS = {'REVIEW_not_blink':'not_blink', 'REVIEW_confirmed_blink':'confirmed_blink', 'REVIEW_uncertain':'uncertain'}

def normalize_review_label(label):
    label=str(label).strip()
    if label.upper()=='BAD' or label.upper().startswith('BAD_'):
        return 'REVIEW_not_blink'
    return label if label in REVIEW_DECISIONS else None


def save_review_spans(root,edf_name,raw):
    """Save only explicit REVIEW_* spans; preserve automatic/manual sources.

    Candidate matching uses its peak inside the marked interval to avoid
    accidentally labelling an adjacent candidate that only touches its edge.
    """
    root=Path(root)
    index=pd.read_csv(root/'outputs/blink_candidate_review/file_index.csv')
    selected=index[index.edf_name.eq(edf_name)]
    if len(selected)!=1:raise ValueError('Select exactly one EDF')
    entry=selected.iloc[0]
    source=load_review_source(entry,root)
    if raw.n_times!=source['n_samples'] or raw.info['sfreq']!=source['sfreq']:raise ValueError('Review raw does not match EDF sample grid')
    if getattr(raw,'_blink_review_edf',None)!=edf_name:raise ValueError('Review raw belongs to another EDF')
    masks=np.array([normalize_review_label(d) is not None for d in raw.annotations.description],dtype=bool)
    annotations=raw.annotations[masks]
    positive=np.asarray(annotations.duration)>0
    annotations=annotations[positive]
    out=root/'outputs/blink_candidate_review'/Path(edf_name).stem
    main_only=getattr(raw,'_blink_review_main_only',False)
    span_path=out/(Path(edf_name).stem+('-human-main-review-annot.fif' if main_only else '-human-review-annot.fif'))
    clean=mne.Annotations(annotations.onset,annotations.duration,[normalize_review_label(label) for label in annotations.description])
    # Save first; candidate decision table remains independent and recoverable.
    if len(clean)==0 and span_path.exists():
        prior=mne.read_annotations(span_path)
        if len(prior):
            raise ValueError('Current window has no recognized review spans; existing nonempty human review preserved')
    clean.save(span_path,overwrite=True,verbose='ERROR')
    candidates=pd.read_csv(root/entry.candidate_details_csv)
    if main_only:
        boundary=pd.read_csv(root/'outputs/blink_boundary_localization_v1'/Path(edf_name).stem/'boundaries.csv').set_index('blink_candidate_id')
        candidates['peak_seconds']=boundary.loc[candidates.blink_candidate_id,'main_peak_seconds'].to_numpy()
    decisions_path=root/entry.review_decisions_csv
    decisions=pd.read_csv(decisions_path,keep_default_na=False)
    if decisions.blink_candidate_id.tolist()!=candidates.blink_candidate_id.tolist():raise ValueError('Candidate review IDs differ from exported source')
    span_rows=[];proposals={}
    for on,duration,label in zip(clean.onset,clean.duration,clean.description):
        matched=candidates[(candidates.peak_seconds>=on)&(candidates.peak_seconds<on+duration)]
        span_rows.append(dict(edf_name=edf_name,start_sec=on,end_sec=on+duration,label=label,
                              blink_candidate_ids=';'.join(matched.blink_candidate_id)))
        for key in matched.blink_candidate_id:proposals.setdefault(key,set()).add(REVIEW_DECISIONS[str(label)])
    for key,values in proposals.items():
        mask=decisions.blink_candidate_id.eq(key)
        old=str(decisions.loc[mask,'review_decision'].iloc[0])
        proposed=next(iter(values)) if len(values)==1 else 'uncertain'
        if old and old!=proposed:
            proposed='uncertain'
            decisions.loc[mask,'review_notes']=decisions.loc[mask,'review_notes'].astype(str)+'; conflicting decisions: inspect annotations and correct CSV explicitly'
        decisions.loc[mask,'review_decision']=proposed
        decisions.loc[mask,'review_region']='main' if main_only else 'old_candidate'
    decisions.to_csv(decisions_path,index=False)
    spans=pd.DataFrame(span_rows,columns=['edf_name','start_sec','end_sec','label','blink_candidate_ids'])
    spans.to_csv(out/('human_main_review_spans.csv' if main_only else 'human_review_spans.csv'),index=False)
    return decisions_path,span_path,spans


def load_review_source(entry,root):
    import json
    return json.loads((root/entry.processing_results/'processing_record.json').read_text())
