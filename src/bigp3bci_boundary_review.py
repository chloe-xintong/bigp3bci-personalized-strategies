"""Separate boundary-domain review; never changes old blink review annotations."""
from pathlib import Path
import json
import textwrap
import numpy as np
import pandas as pd
import mne
from bigp3bci_candidate_review import load_candidate_review


def load_boundary_review(root,edf_name,mode='localization'):
    root=Path(root);out=root/'outputs/blink_boundary_localization_v1'/Path(edf_name).stem
    raw,flashes,old,review_csv=load_candidate_review(root,edf_name,'raw' if mode=='localization' else mode,include_manual=True)
    if mode=='localization':
        with np.load(out/'localization_branch.npz') as data:
            if data['channels'].tolist()!=raw.ch_names:raise ValueError('Channel order mismatch')
            values=data['eeg_v']
        original=raw
        raw=mne.io.RawArray(values,original.info.copy(),verbose='ERROR')
        raw.set_annotations(original.annotations.copy());original.close()
    boundaries=pd.read_csv(out/'boundaries.csv')
    for b in boundaries.itertuples():
        suffix='' if b.localization_usable_for_provisional_comparison else '_uncertain'
        raw.set_annotations(raw.annotations+mne.Annotations([b.main_onset_seconds,b.recovery_onset_seconds],
            [b.main_offset_seconds-b.main_onset_seconds,b.recovery_offset_seconds-b.recovery_onset_seconds],
            ['CANDIDATE_blink_main'+suffix,'CANDIDATE_blink_recovery'+(suffix or ('_unconfirmed' if b.recovery_status!='quiet_criterion_met_unvalidated' else ''))],orig_time=raw.annotations.orig_time))
    saved=root/'outputs/blink_boundary_localization_v1/human_boundary_reviews'/Path(edf_name).stem/(mode+'-boundary-review-annot.fif')
    if saved.exists():
        a=mne.read_annotations(saved)
        raw.set_annotations(raw.annotations+mne.Annotations(a.onset,a.duration,a.description,orig_time=raw.annotations.orig_time))
    raw._boundary_review_edf=edf_name;raw._boundary_review_mode=mode
    return raw,flashes,boundaries


def open_boundary_review(root,edf_name,mode='localization',start=10.,duration=12.):
    from bigp3bci_continuous_qc import flash_event_display_settings
    raw,events,b=load_boundary_review(root,edf_name,mode)
    event_id,event_colors=flash_event_display_settings(events)
    colors={'CANDIDATE_blink':'#888888','CANDIDATE_blink_main':'#ef9217','CANDIDATE_blink_recovery':'#9472bd',
            'CANDIDATE_blink_recovery_unconfirmed':'#665080','CANDIDATE_blink_main_uncertain':'#cc5555','CANDIDATE_blink_recovery_uncertain':'#bb8899'}
    for label in raw.annotations.description:
        if str(label).startswith(('MANUAL_','BOUNDARY_MANUAL_')):colors[str(label)]='#28a060'
    browser=raw.plot(start=start,duration=duration,picks='eeg',n_channels=16,remove_dc=False,scalings={'eeg':100e-6},
        events=events,event_id=event_id,event_color=event_colors,annotation_colors=colors,
        annotation_regex=r'^(CANDIDATE_blink|MANUAL_|BOUNDARY_MANUAL_)',
        title=f'{edf_name}: {mode}; gray=old, orange=main, purple=recovery, red=uncertain',show=True,block=False)
    return raw,browser,b


def save_domain_manual_boundaries(root,raw):
    root=Path(root);edf=raw._boundary_review_edf;mode=raw._boundary_review_mode
    selected=np.array([str(label) in ['BOUNDARY_MANUAL_main','BOUNDARY_MANUAL_recovery','BOUNDARY_MANUAL_uncertain'] for label in raw.annotations.description],bool)
    a=raw.annotations[selected]
    if not len(a):raise ValueError('No BOUNDARY_MANUAL_* marks; nothing saved')
    out=root/'outputs/blink_boundary_localization_v1/human_boundary_reviews'/Path(edf).stem;out.mkdir(parents=True,exist_ok=True)
    path=out/(mode+'-boundary-review-annot.fif')
    annotations=mne.Annotations(a.onset,a.duration,a.description);annotations.save(path,overwrite=True,verbose='ERROR')
    pd.DataFrame(dict(onset_seconds=a.onset,offset_seconds=a.onset+a.duration,label=a.description)).to_csv(out/(mode+'-boundary-review.csv'),index=False)
    (out/(mode+'-domain.json')).write_text(json.dumps(dict(edf_name=edf,review_signal_domain=mode,manual_boundary_labels=True,
                no_automatic_truth=True,not_an_exhaustive_review_interval=True),indent=2),encoding='utf8')
    return path


def render_boundary_examples(root,out):
    import matplotlib.pyplot as plt
    from bigp3bci_continuous_qc import read_bigp3bci_edf
    root=Path(root);out=Path(out);manifest=pd.read_csv(root/'outputs/blink_candidate_review/file_index.csv')
    examples=[]
    for edf in ['F_05_SE003_CBCol_Train04.edf','F_08_SE003_CBCol_Train03.edf']:
        r=manifest[manifest.edf_name.eq(edf)].iloc[0];folder=root/r.processing_results;dest=out/Path(edf).stem
        b=pd.read_csv(dest/'boundaries.csv');events=pd.read_csv(folder/'epoch_candidates.csv')
        eligible=b[~b.filter_startup];choices=[]
        if len(eligible):choices.append(eligible.iloc[0])
        warning=eligible[eligible.localization_flags.fillna('').ne('')]
        if len(warning):choices.append(warning.iloc[0])
        if edf.startswith('F_05'):
            example=b[b.blink_candidate_id.str.endswith('B0010')]
            if len(example):choices.append(example.iloc[0])
        with np.load(folder/'continuous_eeg.npz') as data:formal=data['eeg_v'];aux=data['auxiliary_eeg_v'];channels=data['channels'].tolist();fs=float(data['sfreq'])
        with np.load(dest/'localization_branch.npz') as data:local=data['eeg_v']
        raw=read_bigp3bci_edf(r.source_edf);x=raw.get_data(picks=channels);raw.close()
        manual=root/'outputs/artifact_qc/continuous_manual_qc/manual/manual_review_tables'/(Path(edf).stem+'_manual_review.csv')
        m=pd.read_csv(manual) if manual.exists() else pd.DataFrame(columns=['start_sec','end_sec'])
        seen=set()
        for v in choices:
            if v.blink_candidate_id in seen:continue
            seen.add(v.blink_candidate_id)
            left=max(0,int((min(v.main_onset_seconds,v.old_onset_seconds)-.4)*fs))
            right=min(x.shape[1],int((max(v.recovery_offset_seconds,v.old_offset_seconds,v.main_decision_seconds,v.decision_seconds)+.2)*fs))
            fig,axes=plt.subplots(4,1,figsize=(13,11),sharex=True,layout='constrained')
            for ax,data,name in zip(axes,[x,local,formal,aux],['original raw (legacy manual domain)','causal raw-domain localization branch (new main boundaries)','official 0.5–30Hz (inspect residual and tail)','old trigger branch: official + causal5Hz']):
                for ch in ['EEG_F3','EEG_Fz','EEG_F4']:ax.plot(np.arange(left,right)/fs,data[channels.index(ch),left:right]*1e6,label=ch,lw=.8)
                ax.axvspan(v.old_onset_seconds,v.old_offset_seconds,color='gray',alpha=.15,label='old interval')
                uncertain=not v.localization_usable_for_provisional_comparison
                ax.axvspan(v.main_onset_seconds,v.main_offset_seconds,color='red' if uncertain else 'orange',alpha=.15,label='main UNCERTAIN' if uncertain else 'provisional main')
                ax.axvspan(v.recovery_onset_seconds,v.recovery_offset_seconds,color='purple',alpha=.1,label='recovery provisional')
                ax.axvline(v.main_decision_seconds,color='red',ls='--',lw=.8,label='main decision')
                ax.axvline(v.decision_seconds,color='purple',ls=':',lw=.8,label='final decision')
                for row in m[(m.start_sec<right/fs)&(m.end_sec>left/fs)].itertuples():ax.axvspan(row.start_sec,row.end_sec,color='green',alpha=.1)
                for t in events.loc[(events.onset_seconds>=left/fs)&(events.onset_seconds<right/fs),'onset_seconds']:ax.axvline(t,color='gray',lw=.35,alpha=.35)
                ax.set(title=name,ylabel='µV',xlim=(left/fs,right/fs));ax.legend(fontsize=6,ncol=4)
            axes[-1].set_xlabel('EDF seconds; green=legacy raw manual, gray lines=flash; domains are not interchangeable')
            flags=str(v.localization_flags) if pd.notna(v.localization_flags) else 'none'
            title=f'{v.blink_candidate_id}; flags={flags}; recovery={v.recovery_status}'
            fig.suptitle(textwrap.fill(title,width=110),fontsize=10)
            path=dest/(v.blink_candidate_id+'-domains.png');fig.savefig(path,dpi=120);plt.close(fig)
            examples.append(dict(edf_name=edf,blink_candidate_id=v.blink_candidate_id,figure=str(path.relative_to(out)),
                                 selection='earliest nonstartup, first warning, and previously inspected B0010; no label-based selection',
                                 main_usable=v.localization_usable_for_provisional_comparison))
    pd.DataFrame(examples).to_csv(out/'plotted_examples.csv',index=False)
