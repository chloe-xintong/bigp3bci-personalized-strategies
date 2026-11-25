"""Expanded candidate-only pilot: raw event audit, causal blink branch and simulations.

No labels enter detection. All sample intervals are half-open. Output decisions
are availability timestamps, not retrospectively assigned onset timestamps.
"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
from bigp3bci_continuous_qc import read_bigp3bci_edf, get_all_eeg_channels, stimulus_begin_events
from bigp3bci_preprocessing import (design_filter, causal_filter, continuous_quality,
                                   event_metadata, acquisition_header, file_hash)


def write_csv(rows, path, columns=None):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, index=False)
    return frame


def runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def rising(values):
    b = np.asarray(values) > .5
    return np.flatnonzero(b & ~np.r_[False, b[:-1]])


def read_edf_states(path, names):
    """Decode unmodified EDF int16 state values and their physical scaling.

    Independently audit MNE's state decoding; require matching sample rates
    rather than resampling states. This reader intentionally supports EDF only.
    """
    with Path(path).open('rb') as f:
        fixed = f.read(256)
        ns, nrecords = int(fixed[252:256]), int(fixed[236:244])
        header_bytes = int(fixed[184:192])
        duration = float(fixed[244:252])
        fields = {}
        for key, width in [('label',16),('transducer',80),('unit',8),('physical_min',8),
                           ('physical_max',8),('digital_min',8),('digital_max',8),
                           ('prefilter',80),('samples_per_record',8),('reserved',32)]:
            data = f.read(ns * width)
            fields[key] = [data[i*width:(i+1)*width].decode('ascii', errors='replace').strip() for i in range(ns)]
    counts = np.array(fields['samples_per_record'], int)
    record_words = int(counts.sum())
    if nrecords < 0:
        nrecords = (Path(path).stat().st_size - header_bytes) // (record_words * 2)
    mapped = np.memmap(path, dtype='<i2', mode='r', offset=header_bytes, shape=(nrecords,record_words))
    offsets = np.r_[0, np.cumsum(counts)]
    states, headers = {}, []
    for name in names:
        i = fields['label'].index(name)
        dig = np.array(mapped[:,offsets[i]:offsets[i+1]]).ravel()
        pmin,pmax = float(fields['physical_min'][i]),float(fields['physical_max'][i])
        dmin,dmax = float(fields['digital_min'][i]),float(fields['digital_max'][i])
        physical = (dig.astype(float)-dmin)*(pmax-pmin)/(dmax-dmin)+pmin
        states[name] = (dig, physical)
        headers.append(dict(channel=name, physical_min=pmin,physical_max=pmax,digital_min=dmin,digital_max=dmax,
                            samples_per_record=int(counts[i]),record_seconds=duration,unit=fields['unit'][i]))
    del mapped
    return states, headers


def audit_events(raw, path, site, out):
    names = ['StimulusBegin','StimulusType','StimulusCode','CurrentTarget','SelectedTarget']
    values = dict(zip(names, raw.get_data(picks=names)))
    fs = float(raw.info['sfreq'])
    helper = stimulus_begin_events(raw)
    onsets = helper[:,0] - raw.first_samp
    independent = rising(values['StimulusBegin'])
    states, headers = read_edf_states(path,names)
    rows = []
    for name in names:
        dig, physical = states[name]
        if len(dig) != raw.n_times:
            raise ValueError(f'State {name} has another sample rate; no implicit resampling allowed')
        for representation, z in [('raw_edf_digital',dig),('edf_physical',physical),('mne_state',values[name])]:
            for scope, subset in [('all_samples',z),('flash_onsets',z[onsets])]:
                unique, counts = np.unique(subset,return_counts=True)
                for value,count in zip(unique,counts):
                    rows.append(dict(channel=name,representation=representation,scope=scope,value=float(value),sample_count=int(count)))
    write_csv(rows,out/'state_value_counts.csv')
    write_csv(headers,out/'state_edf_header.csv')
    metadata = event_metadata(raw,path,site)
    metadata['stimulus_begin_at_onset'] = values['StimulusBegin'][onsets]
    metadata['stimulus_begin_previous'] = values['StimulusBegin'][np.maximum(0,onsets-1)]
    metadata['stimulus_type_previous'] = values['StimulusType'][np.maximum(0,onsets-1)]
    metadata['stimulus_type_next'] = values['StimulusType'][np.minimum(raw.n_times-1,onsets+1)]
    metadata['stimulus_type_raw'] = values['StimulusType'][onsets]
    metadata['stimulus_type_digital'] = states['StimulusType'][0][onsets]
    metadata['stimulus_type_physical'] = states['StimulusType'][1][onsets]
    metadata['label_exact_value_valid'] = np.isin(metadata.stimulus_type_raw,[0.,1.])
    metadata['label'] = metadata.stimulus_type_raw.map({0.:'non-target',1.:'target'}).fillna('unknown')
    expected = np.where(values['StimulusType'][onsets]==0,1,np.where(values['StimulusType'][onsets]==1,2,3))
    metadata['helper_label_agrees_exact'] = expected == helper[:,2]
    # Never convert a monotone flash counter into an assumed row/column group.
    characters = [n for n in raw.ch_names if re.match(r'^.+_\d+_\d+$',n)]
    character_values = raw.get_data(picks=characters)[:,onsets]
    signatures, group_rows = [], {}
    for k in range(len(onsets)):
        active = sorted(np.asarray(characters)[character_values[:,k]>.5].tolist())
        signature = '|'.join(active)
        key = 'observed_' + hashlib.sha256(signature.encode()).hexdigest()[:12] if active else 'unknown'
        signatures.append(key)
        group_rows[key] = dict(stimulus_group_id=key,active_character_channels=json.dumps(active),n_active_characters=len(active),
                               provenance='observed character-state set at flash onset; no assumed row/column meaning')
    metadata['stimulus_group_id'] = signatures
    write_csv(list(group_rows.values()),out/'stimulus_group_dictionary.csv')
    boundaries = rising(values['SelectedTarget'])
    trials = []
    previous = -1
    for i,boundary in enumerate(boundaries,1):
        selected = metadata[(metadata.onset_sample>previous)&(metadata.onset_sample<boundary)]
        codes = selected.stimulus_code.to_numpy()
        trials.append(dict(character_trial_id=i,start_boundary_exclusive=int(previous),end_boundary_exclusive=int(boundary),
                           boundary_seconds=boundary/fs,n_flashes=len(selected),n_target=int(selected.label.eq('target').sum()),
                           n_nontarget=int(selected.label.eq('non-target').sum()),n_unknown=int(selected.label.eq('unknown').sum()),
                           current_target_values=json.dumps(sorted(selected.current_target.unique().tolist())),
                           unique_observed_stimulus_groups=int(selected.stimulus_group_id.nunique()),
                           code_first=float(codes[0]) if len(codes) else np.nan,code_last=float(codes[-1]) if len(codes) else np.nan,
                           code_adjacent_increment_one=bool(np.all(np.diff(codes)==1)),
                           note='counts observed, not enforced; strict flash > previous boundary and < SelectedTarget rising boundary'))
        previous = int(boundary)
    write_csv(trials,out/'character_event_audit.csv')
    checks = dict(edf_name=path.name,subject_id=path.parts[-5],session_id=path.parts[-4],site=site,site_status='inferred',
                  flash_count=len(onsets),independent_rising_count=len(independent),helper_onsets_match=bool(np.array_equal(onsets,independent)),
                  sample_zero_active=bool(values['StimulusBegin'][0]>.5),helper_exact_label_mismatch=int((expected!=helper[:,2]).sum()),
                  stimulus_type_unique=json.dumps(np.unique(values['StimulusType']).tolist()),
                  stimulus_type_unique_at_flash=json.dumps(np.unique(values['StimulusType'][onsets]).tolist()),
                  mne_matches_edf_physical_stimulus_type=bool(np.allclose(values['StimulusType'],states['StimulusType'][1],rtol=0,atol=1e-8)),
                  n_character_boundaries=len(boundaries),unassigned_flashes=int(metadata.character_trial_id.eq(-1).sum()),
                  flashes_on_boundary=int(np.isin(onsets,boundaries).sum()),
                  label_provenance='raw StimulusType exact values at helper rising onset; 0/1 semantics inherited, not established solely by counts',
                  stimulus_group_provenance='observed simultaneously active character-state sets; see per-EDF dictionary')
    (out/'event_audit.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
    write_csv(metadata,out/'events_audited.csv')
    if not checks['helper_onsets_match'] or checks['helper_exact_label_mismatch']:
        raise ValueError(f'Event parser discrepancy: inspect {out}; no counts coerced')
    return metadata, checks


BASIC_COLUMNS=['candidate_id','label','channel','start_sample','stop_sample_exclusive','onset_seconds','offset_seconds','decision_seconds','metric_uv']


def basic_quality_candidates(x,fs,channels,cfg):
    """Raw signal faults only; no biological-source interpretation."""
    rows=[]
    def add(kind,ch,a,b,decision,metric):
        rows.append(dict(candidate_id=f'Q{len(rows)+1:06d}',label=kind,channel=ch,start_sample=int(a),stop_sample_exclusive=int(b),
                         onset_seconds=a/fs,offset_seconds=b/fs,decision_seconds=decision/fs,metric_uv=float(metric)))
    w,step=round(cfg['flat_window_seconds']*fs),round(cfg['flat_step_seconds']*fs)
    for i,ch in enumerate(channels):
        z=x[i]*1e6
        for a,b in runs(~np.isfinite(z)):
            add('CANDIDATE_nonfinite',ch,a,b,a,np.nan)
        equal=np.isfinite(z[1:]) & np.isfinite(z[:-1]) & (np.diff(z)==0)
        for a,b in runs(equal):
            b+=1
            if b-a>=int(np.ceil(cfg['constant_min_seconds']*fs)):
                add('CANDIDATE_constant',ch,a,b,a+int(np.ceil(cfg['constant_min_seconds']*fs))-1,0.)
        for a,b in runs(np.isfinite(z)&(z==0)):
            if b-a>=int(np.ceil(cfg['zero_min_seconds']*fs)):
                add('CANDIDATE_zero',ch,a,b,a+int(np.ceil(cfg['zero_min_seconds']*fs))-1,0.)
        for a in range(0,max(0,len(z)-w+1),step):
            segment=z[a:a+w]
            if np.isfinite(segment).all() and np.ptp(segment)<=cfg['flat_ptp_uv'] and np.std(segment)<=cfg['flat_std_uv']:
                add('CANDIDATE_flat',ch,a,a+w,a+w-1,np.ptp(segment))
        jumps=np.flatnonzero(np.isfinite(z[:-1]) & np.isfinite(z[1:]) & (np.abs(np.diff(z))>=cfg['jump_delta_uv']))+1
        for t in jumps:
            add('CANDIDATE_jump',ch,t,t+1,t,abs(z[t]-z[t-1]))
    return pd.DataFrame(rows,columns=BASIC_COLUMNS)


BLINK_COLUMNS=['edf_name','blink_candidate_id','label','onset_sample','peak_sample','offset_sample_exclusive','decision_sample',
               'onset_seconds','peak_seconds','offset_seconds','decision_seconds','detection_delay_from_peak_seconds',
               'frontal_amplitude_uv','posterior_amplitude_uv','frontal_posterior_metric','frontal_dominance_support',
               'supporting_channels','min_support_correlation','slow_ptp_fraction','duration_seconds','filter_startup',
               'detector_parameters']


def detect_blinks(y,fs,channels,cfg,filter_cfg,edf_name,contaminated=None):
    """Hysteretic slow common-frontal excursion with causal recovery confirmation.

    Scan the auxiliary branch in temporal order. No event/target labels or
    whole-file estimated thresholds are accepted. Merge and recovery delays are
    explicit. Incomplete excursions at EOF cannot become confirmed candidates.
    """
    sos=signal.butter(cfg['auxiliary_order'],cfg['auxiliary_lowpass_hz'],fs=fs,output='sos')
    aux,_,_=causal_filter(y,sos,fs,filter_cfg)
    frontal=[channels.index(c) for c in cfg['frontal_channels']]
    posterior=[channels.index(c) for c in cfg['posterior_channels']]
    z=aux[frontal]*1e6
    common=np.median(z,axis=0)
    amplitude=np.abs(common)
    hold=max(1,int(np.ceil(cfg['recovery_seconds']*fs)))
    gap=int(np.ceil(cfg['merge_gap_seconds']*fs))
    segments=[]; active=None; below=None
    for t,a in enumerate(amplitude):
        if active is None:
            if a>=cfg['entry_uv']:
                active=t;below=None
        else:
            if a<=cfg['exit_uv']:
                if below is None: below=t
                if t-below+1>=hold:
                    segments.append(dict(start=active,stop=below,confirmed=t,incomplete=False))
                    active=None;below=None
            else:
                below=None
    if active is not None:
        segments.append(dict(start=active,stop=len(amplitude),confirmed=len(amplitude),incomplete=True))
    merged=[]
    for segment in segments:
        if merged and segment['start']-merged[-1]['stop']<=gap:
            merged[-1].update(stop=segment['stop'],confirmed=segment['confirmed'],incomplete=segment['incomplete'])
        else: merged.append(segment.copy())
    rows=[];logs=[];last_peak=-np.inf
    for seg in merged:
        a,b=seg['start'],seg['stop']
        decision=max(seg['confirmed'],b+gap)
        reasons=[]
        if seg['incomplete'] or decision>=y.shape[1]: reasons.append('unconfirmed_at_eof')
        if b<=a: continue
        peak=a+int(np.argmax(amplitude[a:b]))
        duration=(b-a)/fs
        if not cfg['min_duration_seconds']<=duration<=cfg['max_duration_seconds']: reasons.append('duration')
        sign=1 if common[peak]>=0 else -1
        peak_values=z[:,peak]
        amp=float(np.median(np.abs(peak_values)))
        post=float(np.median(np.abs(aux[posterior,peak]*1e6)))
        ratio=amp/max(post,cfg['posterior_floor_uv'])
        if amp<cfg['min_frontal_amplitude_uv']: reasons.append('amplitude')
        correlations=[]
        for trace in z:
            if b-a>=3 and np.std(trace[a:b])>0 and np.std(common[a:b])>0:
                correlations.append(float(np.corrcoef(trace[a:b],common[a:b])[0,1]))
            else: correlations.append(0.)
        support=[i for i in range(len(frontal)) if peak_values[i]*sign>=cfg['support_peak_fraction']*amp and correlations[i]>=cfg['min_correlation']]
        if len(support)<cfg['min_support_channels']: reasons.append('insufficient_common_frontal_support')
        slow_ptp=float(np.median(np.ptp(aux[frontal,a:b],axis=1)))
        official_ptp=float(np.median(np.ptp(y[frontal,a:b],axis=1)))
        slow_fraction=slow_ptp/max(official_ptp,1e-12)
        if slow_fraction<cfg['min_slow_ptp_fraction']: reasons.append('slow_shape')
        if cfg['require_frontal_dominance'] and ratio<cfg['frontal_posterior_support_ratio']: reasons.append('frontal_dominance')
        if contaminated is not None and contaminated[:,a:min(decision+1,y.shape[1])].any(): reasons.append('nonfinite_recovery')
        if (peak-last_peak)/fs<cfg['min_peak_distance_seconds']: reasons.append('peak_distance_from_previous_accepted')
        logs.append(dict(onset_sample=a,offset_sample_exclusive=b,decision_sample=decision,exclusion_reasons=';'.join(reasons)))
        if reasons: continue
        last_peak=peak
        rows.append(dict(edf_name=edf_name,blink_candidate_id=f'{Path(edf_name).stem}_B{len(rows)+1:04d}',label='CANDIDATE_blink',
                         onset_sample=a,peak_sample=peak,offset_sample_exclusive=b,decision_sample=decision,
                         onset_seconds=a/fs,peak_seconds=peak/fs,offset_seconds=b/fs,decision_seconds=decision/fs,
                         detection_delay_from_peak_seconds=(decision-peak)/fs,frontal_amplitude_uv=amp,posterior_amplitude_uv=post,
                         frontal_posterior_metric=ratio,frontal_dominance_support=ratio>=cfg['frontal_posterior_support_ratio'],
                         supporting_channels=';'.join(cfg['frontal_channels'][i] for i in support),
                         min_support_correlation=min(correlations[i] for i in support),slow_ptp_fraction=slow_fraction,
                         duration_seconds=duration,filter_startup=a<round(filter_cfg['startup_seconds']*fs),
                         detector_parameters=json.dumps(cfg,sort_keys=True)))
    return pd.DataFrame(rows,columns=BLINK_COLUMNS),aux,pd.DataFrame(logs,columns=['onset_sample','offset_sample_exclusive','decision_sample','exclusion_reasons']),sos


def associate_epochs(metadata,x,y,contaminated,quality,blinks,fs,channels,cfg):
    size=int(np.ceil(cfg['epoch']['duration_seconds']*fs))
    table=metadata.copy();ptp=np.full((len(table),len(channels)),np.nan)
    associations=[];quality_links=[];flags=[]
    flat_labels={'CANDIDATE_flat','CANDIDATE_constant','CANDIDATE_zero'}
    for i,r in table.iterrows():
        a=int(r.onset_sample);b=a+size;end=min(b,y.shape[1]);complete=b<=y.shape[1]
        if complete:ptp[i]=np.ptp(y[:,a:b],axis=1)*1e6
        q=quality[(quality.start_sample<b)&(quality.stop_sample_exclusive>a)]
        for _,v in q.iterrows():
            quality_links.append(dict(event_index=i,candidate_id=v.candidate_id,label=v.label,channel=v.channel,
                                      overlap_seconds=(min(b,v.stop_sample_exclusive)-max(a,v.start_sample))/fs,
                                      candidate_decision_seconds=v.decision_seconds,known_by_epoch_decision=v.decision_seconds<=max(b/fs,a/fs+.8)))
        overlaps=[]
        hits=blinks[(blinks.onset_sample<b)&(blinks.offset_sample_exclusive>a)]
        for _,v in hits.iterrows():
            start=max(a,int(v.onset_sample));stop=min(b,int(v.offset_sample_exclusive))
            overlaps.append((start,stop))
            associations.append(dict(event_index=i,blink_candidate_id=v.blink_candidate_id,overlap_start_sample=start,overlap_stop_sample_exclusive=stop,
                                     overlap_duration_seconds=(stop-start)/fs,blink_decision_seconds=v.decision_seconds,
                                     known_by_epoch_decision=v.decision_seconds<=max(b/fs,a/fs+.8),
                                     earliest_joint_decision_seconds=max(v.decision_seconds,b/fs,a/fs+.8)))
        # Use union duration, not a double-counted sum if a future detector allows overlapping candidates.
        union=[]
        for start,stop in sorted(overlaps):
            if union and start<=union[-1][1]:union[-1][1]=max(union[-1][1],stop)
            else:union.append([start,stop])
        flat=q[q.label.isin(flat_labels)]; jump=q[q.label.eq('CANDIDATE_jump')]
        raw_nonfinite=~np.isfinite(x[:,a:end]).all(axis=1)
        affected=sorted(set(q.channel))
        flags.append(dict(complete_epoch=complete,epoch_stop_sample_exclusive=b,epoch_decision_seconds=max(b/fs,a/fs+.8),
                          filter_startup=a<round(fs*cfg['filter']['startup_seconds']),
                          has_blink_candidate=bool(len(hits)),blink_candidate_ids=';'.join(hits.blink_candidate_id),
                          blink_overlap_duration=sum(stop-start for start,stop in union)/fs,
                          max_ptp_uv=float(np.max(ptp[i])) if complete and np.isfinite(ptp[i]).all() else np.nan,
                          min_ptp_uv=float(np.min(ptp[i])) if complete and np.isfinite(ptp[i]).all() else np.nan,
                          ptp_by_channel=json.dumps({ch:float(value) if np.isfinite(value) else None for ch,value in zip(channels,ptp[i])}),
                          flat_candidate=bool(len(flat)),flat_channels=';'.join(sorted(set(flat.channel))),
                          jump_candidate=bool(len(jump)),jump_channels=';'.join(sorted(set(jump.channel))),
                          nonfinite_flag=bool(raw_nonfinite.any()),nonfinite_channels=';'.join(np.asarray(channels)[raw_nonfinite]),
                          filter_recovery=bool(contaminated[:,a:end].any()),filter_recovery_channels=';'.join(np.asarray(channels)[contaminated[:,a:end].any(axis=1)]),raw_quality_channels=';'.join(affected),
                          association_policy='retrospective overlap; see decision timestamps; candidates do not determine rejection'))
    for col in flags[0] if flags else []:table[col]=[row[col] for row in flags]
    a=pd.DataFrame(associations,columns=['event_index','blink_candidate_id','overlap_start_sample','overlap_stop_sample_exclusive',
                                      'overlap_duration_seconds','blink_decision_seconds','known_by_epoch_decision','earliest_joint_decision_seconds'])
    q=pd.DataFrame(quality_links,columns=['event_index','candidate_id','label','channel','overlap_seconds','candidate_decision_seconds','known_by_epoch_decision'])
    return table,ptp,a,q


def simulate_thresholds(table,ptp,quality_links,channels,cfg,fs):
    rows=[]
    for channel_set,names in cfg['simulation']['qc_channel_sets'].items():
        idx=[channels.index(ch) for ch in names]
        for upper in cfg['simulation']['upper_ptp_uv']:
            scheme=f'{channel_set}_upper_{upper:g}uv' if upper is not None else f'{channel_set}_no_upper'
            for i,r in table.iterrows():
                reasons=[];triggers=set()
                q=quality_links[(quality_links.event_index==i)&quality_links.channel.isin(names)&quality_links.known_by_epoch_decision]
                flat=q[q.label.isin(['CANDIDATE_flat','CANDIDATE_constant','CANDIDATE_zero'])]
                raw_nonfinite=set(str(r.nonfinite_channels).split(';'))&set(names)
                if not r.complete_epoch:reasons.append('incomplete')
                if r.filter_startup:reasons.append('filter_startup')
                if raw_nonfinite:reasons.append('nonfinite');triggers.update(raw_nonfinite)
                recovery=set(str(r.filter_recovery_channels).split(';'))&set(names)
                if recovery:reasons.append('filter_recovery');triggers.update(recovery)
                if len(flat):reasons.append('raw_flat');triggers.update(flat.channel)
                values=ptp[i,idx]
                low=np.isfinite(values)&(values<cfg['simulation']['lower_ptp_uv'])
                high=np.isfinite(values)&(values>upper) if upper is not None else np.zeros(len(idx),bool)
                if low.any():reasons.append('ptp_low');triggers.update(np.array(names)[low])
                if high.any():reasons.append('ptp_high');triggers.update(np.array(names)[high])
                if r.complete_epoch and not np.isfinite(values).all():reasons.append('filtered_nonfinite')
                rows.append(dict(event_index=i,scheme=scheme,qc_channel_set=channel_set,upper_ptp_uv=upper,
                                 would_reject=bool(reasons),would_keep=not reasons,reasons=';'.join(reasons),trigger_channels=';'.join(sorted(triggers)),
                                 simulation_only=True))
    return pd.DataFrame(rows)


def manual_comparison(path,blinks,table,ptp,channels,fs,cfg,root,out):
    """Only exhaustive reviewed intervals can define matches/misses/unmatched.

    Sparse legacy annotations remain descriptive suspected events, not a truth
    set. No reviewed interval or negative annotation is manufactured here.
    """
    manual_root=root/cfg['manual_validation']['manual_root']
    csv=manual_root/'manual_review_tables'/f'{path.stem}_manual_review.csv'
    fif=manual_root/'manual_annotations'/f'{path.stem}-manual-annot.fif'
    manual=pd.DataFrame(columns=['manual_id','start_sec','end_sec','type'])
    sources=[];consistency='not_applicable'
    if csv.exists():
        manual=pd.read_csv(csv);sources.append(str(csv.relative_to(root)))
    if fif.exists():
        import mne
        ann=mne.read_annotations(fif)
        sources.append(str(fif.relative_to(root)))
        if len(manual):
            consistency='agree' if len(ann)==len(manual) and np.allclose(ann.onset,manual.start_sec,atol=1/fs) and np.allclose(ann.onset+ann.duration,manual.end_sec,atol=1/fs) and np.array_equal(ann.description,manual.type) else 'mismatch'
            if consistency=='mismatch':raise ValueError(f'Conflicting manual sources for {path.name}')
        else:
            manual=pd.DataFrame(dict(manual_id=[f'{path.stem}_M{i+1:04d}' for i in range(len(ann))],start_sec=ann.onset,end_sec=ann.onset+ann.duration,type=ann.description))
    legacy_allowed=path.name in cfg['manual_validation']['sparse_blink_files']
    blink_types=['MANUAL_blink','MANUAL_possible_blink','MANUAL_suspected_blink']
    if legacy_allowed:blink_types+=cfg['manual_validation']['legacy_blink_types']
    all_manual=manual.copy()
    manual=manual[manual.type.isin(blink_types)].copy()
    if len(manual) and not (manual.end_sec> manual.start_sec).all():raise ValueError('Manual intervals must have positive length')
    reviewed_file=root/cfg['manual_validation']['reviewed_intervals_path']
    reviewed=[]
    if reviewed_file.exists():
        review=pd.read_csv(reviewed_file)
        required=set(cfg['manual_validation']['reviewed_schema'])
        if not required<=set(review.columns):raise ValueError(f'Reviewed interval schema missing {required-set(review.columns)}')
        review=review[(review.edf_name==path.name)&(review.review_status==cfg['manual_validation']['complete_status'])]
        for _,r in review.sort_values('start_sec').iterrows():
            if not 0<=r.start_sec<r.end_sec<=table.attrs['duration_seconds']+1/fs:raise ValueError('Invalid reviewed interval')
            if reviewed and r.start_sec<=reviewed[-1][1]:reviewed[-1][1]=max(reviewed[-1][1],r.end_sec)
            else:reviewed.append([r.start_sec,r.end_sec])
    def fully_reviewed(a,b):return any(a>=start and b<=stop for start,stop in reviewed)
    manual['fully_reviewed']=np.array([fully_reviewed(r.start_sec,r.end_sec) for _,r in manual.iterrows()],dtype=bool)
    manual['filter_startup']=manual.start_sec<cfg['filter']['startup_seconds']
    manual['evaluation_eligible']=manual.fully_reviewed & ~manual.filter_startup
    # Unknown/uncertain annotations are not an exhaustive truth set: if present,
    # exclude the entire overlapping manual/candidate event from evaluated metrics.
    uncertain=[]
    if len(all_manual):
        uncertain=[(r.start_sec,r.end_sec) for _,r in all_manual[all_manual.type.str.contains('uncertain',case=False,na=False)].iterrows()]
    def uncertain_overlap(a,b):return any(a<stop and b>start for start,stop in uncertain)
    manual['evaluation_eligible'] &= np.array([not uncertain_overlap(r.start_sec,r.end_sec) for _,r in manual.iterrows()],dtype=bool)
    candidate=blinks.copy()
    candidate['fully_reviewed']=np.array([fully_reviewed(r.onset_seconds,r.offset_seconds) for _,r in candidate.iterrows()],dtype=bool)
    candidate['evaluation_eligible']=candidate.fully_reviewed & ~candidate.filter_startup
    candidate['evaluation_eligible'] &= np.array([not uncertain_overlap(r.onset_seconds,r.offset_seconds) for _,r in candidate.iterrows()],dtype=bool)
    pairs=[];manual_events=[]
    for _,m in manual.iterrows():
        hits=candidate[(candidate.onset_seconds<m.end_sec)&(candidate.offset_seconds>m.start_sec)]
        eligible_hits=hits[hits.evaluation_eligible]
        status=('matched' if len(eligible_hits) else 'missed') if m.evaluation_eligible else 'not_evaluable_sparse_or_boundary_or_startup'
        manual_events.append(dict(manual_id=m.manual_id,start_sec=m.start_sec,end_sec=m.end_sec,evaluation_eligible=m.evaluation_eligible,
                                  matching_candidate_ids=';'.join(hits.blink_candidate_id),n_overlap_candidates=len(hits),evaluation_status=status))
        for _,c in hits.iterrows():
            overlap=min(m.end_sec,c.offset_seconds)-max(m.start_sec,c.onset_seconds)
            pairs.append(dict(manual_id=m.manual_id,blink_candidate_id=c.blink_candidate_id,overlap_seconds=overlap,
                              intersection_over_union=overlap/(max(m.end_sec,c.offset_seconds)-min(m.start_sec,c.onset_seconds)),
                              onset_difference_seconds=c.onset_seconds-m.start_sec,offset_difference_seconds=c.offset_seconds-m.end_sec,
                              evaluation_eligible=bool(m.evaluation_eligible and c.evaluation_eligible),
                              interpretation='suspected blink overlap, never automatic epoch rejection'))
    for i,c in candidate.iterrows():
        m=manual[(manual.start_sec<c.offset_seconds)&(manual.end_sec>c.onset_seconds)]
        candidate.loc[i,'overlapping_manual_ids']=';'.join(m.manual_id)
        matched=bool(m.evaluation_eligible.any())
        candidate.loc[i,'evaluation_status']=('matched' if matched else 'unmatched_candidate') if c.evaluation_eligible else 'unreviewed_or_boundary_or_startup'
    write_csv(manual_events,out/'manual_event_comparison.csv',columns=['manual_id','start_sec','end_sec','evaluation_eligible','matching_candidate_ids','n_overlap_candidates','evaluation_status'])
    write_csv(pairs,out/'manual_candidate_overlaps.csv',columns=['manual_id','blink_candidate_id','overlap_seconds','intersection_over_union','onset_difference_seconds','offset_difference_seconds','evaluation_eligible','interpretation'])
    write_csv(candidate,out/'manual_candidate_comparison.csv')
    manual_epoch=[]
    for _,m in manual.iterrows():
        for i,r in table.iterrows():
            if r.onset_sample/fs<m.end_sec and r.epoch_stop_sample_exclusive/fs>m.start_sec:
                for ch,v in zip(channels,ptp[i]):
                    manual_epoch.append(dict(manual_id=m.manual_id,event_index=i,channel=ch,ptp_uv=v,filter_startup=r.filter_startup,
                                             manual_evaluation_eligible=m.evaluation_eligible))
    write_csv(manual_epoch,out/'manual_event_epoch_ptp.csv',columns=['manual_id','event_index','channel','ptp_uv','filter_startup','manual_evaluation_eligible'])
    nmanual=int(manual.evaluation_eligible.sum());ncandidate=int(candidate.evaluation_eligible.sum())
    matched_manual=sum(r['evaluation_status']=='matched' for r in manual_events)
    matched_candidates=int(candidate.get('evaluation_status',pd.Series(dtype=str)).eq('matched').sum())
    summary=dict(edf_name=path.name,manual_sources=json.dumps(sources),manual_source_consistency=consistency,
                 suspected_manual_blinks=len(manual),complete_reviewed_intervals=len(reviewed),reviewed_seconds=sum(b-a for a,b in reviewed),
                 evaluated_manual_events=nmanual,evaluated_candidates=ncandidate,matched_manual_events=matched_manual if nmanual else None,
                 unmatched_candidates=int(candidate.get('evaluation_status',pd.Series(dtype=str)).eq('unmatched_candidate').sum()) if reviewed else None,
                 missed_manual_events=nmanual-matched_manual if nmanual else None,
                 reviewed_event_recall=matched_manual/nmanual if nmanual else None,
                 reviewed_candidate_match_fraction=matched_candidates/ncandidate if ncandidate else None,
                 validation_status='reviewed_intervals_available' if reviewed else 'unavailable: no exhaustive reviewed_intervals; sparse overlaps descriptive only',
                 filter_startup_evaluated_as_blink=False,ground_truth_generated=False)
    (out/'manual_validation_status.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    return summary


def distribution_rows(x,y,ptp,channels):
    rows=[]
    for c,ch in enumerate(channels):
        for metric,values in [('raw_amplitude_uv',x[c]*1e6),('filtered_amplitude_uv',y[c]*1e6),('filtered_epoch_ptp_uv',ptp[:,c])]:
            v=values[np.isfinite(values)]
            row=dict(channel=ch,metric=metric,n_finite=len(v),n_nonfinite=int((~np.isfinite(values)).sum()))
            for name,q in [('min',0),('q01',.01),('q05',.05),('q25',.25),('q50',.5),('q75',.75),('q95',.95),('q99',.99),('max',1)]:
                row[name]=float(np.quantile(v,q)) if len(v) else np.nan
            rows.append(row)
    return rows


def process_expanded_file(path,site,cfg,out,root):
    out.mkdir(parents=True,exist_ok=True)
    before=file_hash(path)
    raw=read_bigp3bci_edf(path)
    try:
        channels=get_all_eeg_channels(raw);fs=float(raw.info['sfreq'])
        metadata,audit=audit_events(raw,path,site,out)
        x=raw.get_data(picks=channels)
        sos=design_filter(fs,cfg['filter'])
        y,contaminated,states=causal_filter(x,sos,fs,cfg['filter'])
        assert y.shape==x.shape
        windows=continuous_quality(x,fs,channels,cfg['continuous_qc'])
        write_csv(windows,out/'raw_continuous_windows.csv')
        channel_rows=[]
        for ch,g in windows.groupby('channel',sort=False):
            bad=g[g.reason.ne('')]
            channel_rows.append(dict(channel=ch,flat_window_fraction=float(g.reason.str.contains('constant|zero|low_variation').mean()),
                                     anomaly_window_fraction=float(g.reason.ne('').mean()),persistent_candidate=bool(g.persistent_as_of_window.any()),
                                     reasons=';'.join(sorted({reason for value in bad.reason for reason in value.split(';')})),
                                     strategy='report_only; no channel drop, interpolation, rereferencing or file exclusion'))
        write_csv(channel_rows,out/'persistent_channel_candidates.csv')
        quality=basic_quality_candidates(x,fs,channels,cfg['signal_quality'])
        write_csv(quality,out/'basic_quality_candidates.csv')
        blinks,aux,logs,aux_sos=detect_blinks(y,fs,channels,cfg['blink_detector'],cfg['filter'],path.name,contaminated)
        write_csv(blinks,out/'blink_candidates.csv')
        write_csv(logs,out/'blink_excursion_log.csv')
        table,ptp,links,qlinks=associate_epochs(metadata,x,y,contaminated,quality,blinks,fs,channels,cfg)
        table.attrs['duration_seconds']=x.shape[1]/fs
        write_csv(table,out/'epoch_candidates.csv')
        write_csv(links,out/'blink_epoch_links.csv')
        write_csv(qlinks,out/'raw_quality_epoch_links.csv')
        simulations=simulate_thresholds(table,ptp,qlinks,channels,cfg,fs)
        write_csv(simulations,out/'threshold_simulation_epochs.csv')
        long=pd.DataFrame(ptp,columns=channels).rename_axis('event_index').reset_index().melt(id_vars='event_index',var_name='channel',value_name='ptp_uv')
        write_csv(long,out/'epoch_ptp_long.csv')
        distributions=distribution_rows(x,y,ptp,channels)
        write_csv(distributions,out/'amplitude_ptp_distribution.csv')
        manual=manual_comparison(path,blinks,table,ptp,channels,fs,cfg,root,out)
        np.savez_compressed(out/'continuous_eeg.npz',eeg_v=y,auxiliary_eeg_v=aux,channels=channels,sfreq=fs,sos=sos,auxiliary_sos=aux_sos,final_state=states)
        size=int(np.ceil(fs*cfg['epoch']['duration_seconds']))
        epochs=np.full((len(metadata),16,size),np.nan)
        for i,r in metadata.iterrows():
            start=int(r.onset_sample);end=min(start+size,y.shape[1]);epochs[i,:,:end-start]=y[:,start:end]
        np.savez_compressed(out/'epochs_all.npz',eeg_v=epochs,channels=channels,sfreq=fs,event_index=metadata.event_index.to_numpy(),baseline='None')
        after=file_hash(path)
        if after!=before:raise RuntimeError('Source EDF changed')
        record=dict(source_edf=str(path.resolve()),source_sha256_before=before,source_sha256_after=after,original_edf_unchanged=True,
                    channels=channels,sfreq=fs,n_samples=x.shape[1],epoch_samples=size,epoch_endpoint='[onset, onset + ceil(0.8*sfreq))',baseline=None,
                    acquisition_prefilter=acquisition_header(path),added_analysis_sos=sos.tolist(),auxiliary_sos=aux_sos.tolist(),
                    filter_initialization=cfg['filter']['initialization'],state_policy='per-continuous-EDF, state retained across chunks, independent EDF reset',
                    startup_interval_samples=[0,round(fs*cfg['filter']['startup_seconds'])],events_unchanged=True,
                    signal_delay='frequency-dependent causal delay; no event shifts; auxiliary branch adds its own delay',
                    blink_timing='sample-based onset/offset on auxiliary signal; decision after recovery and merge gap, not raw blink truth boundary',
                    candidate_count=len(blinks),basic_quality_count=len(quality),n_epochs=len(table),
                    no_automatic_epoch_rejection=True,valid_epoch_created=False,config=cfg,
                    software=dict(numpy=np.__version__,pandas=pd.__version__,scipy=__import__('scipy').__version__,mne=__import__('mne').__version__))
        (out/'processing_record.json').write_text(json.dumps(record,indent=2),encoding='utf8')
        shared=dict(edf_name=path.name,subject_id=path.parts[-5],session_id=path.parts[-4],site=site,site_status='inferred')
        return dict(events=table.assign(**{k:v for k,v in shared.items() if k not in table}),ptp=long.assign(**shared),
                    simulations=simulations.merge(table[['event_index','label','character_trial_id','stimulus_group_id']],on='event_index',validate='many_to_one').assign(**shared),
                    blinks=blinks.assign(**{k:v for k,v in shared.items() if k not in blinks}),
                    distributions=pd.DataFrame(distributions).assign(**shared),audit=audit,manual=manual,
                    quality=quality.assign(**shared),persistent=pd.DataFrame(channel_rows).assign(**shared))
    finally:
        raw.close()
