"""Separate frozen blink detection from provisional main-wave localization.

Localize on an independent causal raw-domain lowpass branch. No detector labels,
manual annotations or full-record distribution estimates enter this function.
All retrospective boundaries retain their actual information-availability time.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy import signal
from bigp3bci_preprocessing import causal_filter


def quiet_runs(values,threshold,length):
    mask=np.asarray(values)<=threshold
    edges=np.diff(np.r_[False,mask,False].astype(int))
    return [(int(a),int(b)) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if b-a>=length]


def localization_signal(x,fs,cfg):
    b,a=signal.iirnotch(cfg['notch_hz'],cfg['notch_q'],fs=fs)
    sos=np.vstack([signal.tf2sos(b,a),signal.butter(cfg['lowpass_order'],cfg['lowpass_hz'],fs=fs,output='sos')])
    y,contaminated,state=causal_filter(x,sos,fs,cfg)
    return y,contaminated,sos,state


def localize_boundaries(x,official,localization,old,channels,fs,cfg,contaminated=None):
    frontal=[channels.index(ch) for ch in cfg['frontal_channels']]
    trace=np.median(localization[frontal],axis=0)*1e6
    formal=np.median(official[frontal],axis=0)*1e6
    count=x.shape[1];rows=[];lobes=[]
    n=lambda seconds:max(1,int(np.ceil(seconds*fs)))
    for r in old.itertuples():
        trigger=int(r.onset_sample)
        ref_a=max(0,trigger-n(cfg['baseline_start_before_trigger_seconds']))
        ref_b=max(ref_a+1,trigger-n(cfg['baseline_end_before_trigger_seconds']))
        ref_b=min(ref_b,count)
        baseline=float(np.median(trace[ref_a:ref_b]))
        formal_baseline=float(np.median(formal[ref_a:ref_b]))
        baseline_channels=np.median(localization[frontal,ref_a:ref_b],axis=1)*1e6
        search_a=max(0,trigger-n(cfg['peak_search_before_trigger_seconds']))
        search_b=min(count,trigger+n(cfg['peak_search_after_trigger_seconds'])+1)
        centered=trace-baseline
        peak=search_a+int(np.argmax(np.abs(centered[search_a:search_b])))
        sign=1 if centered[peak]>=0 else -1
        amplitude=float(abs(centered[peak]));signed=centered*sign
        local_flags=[]
        if ref_a==0 and trigger<n(cfg['baseline_start_before_trigger_seconds']):local_flags.append('baseline_left_censored')
        if search_b==count:local_flags.append('peak_search_right_censored')
        if peak in (search_a,search_b-1):local_flags.append('peak_at_search_edge')
        if amplitude<cfg['minimum_localization_peak_uv']:local_flags.append('weak_raw_domain_waveform')
        values=(localization[frontal,peak]*1e6-baseline_channels)*sign
        support=[ch for ch,value in zip(cfg['frontal_channels'],values) if value>=amplitude*cfg['support_peak_fraction']]
        if len(support)<cfg['minimum_same_sign_frontal_support']:local_flags.append('weak_frontal_support')
        onset_threshold=max(cfg['boundary_floor_uv'],amplitude*cfg['onset_fraction_of_peak'])
        offset_threshold=max(cfg['boundary_floor_uv'],amplitude*cfg['offset_fraction_of_peak'])
        left=max(0,peak-n(cfg['onset_lookback_seconds']))
        before=quiet_runs(signed[left:peak+1],onset_threshold,n(cfg['onset_quiet_seconds']))
        if before:
            # Last completed quiet run before the selected peak, then rising edge.
            onset=min(peak,left+before[-1][1])
        else:
            onset=left;local_flags.append('main_onset_left_censored')
        right=min(count,peak+n(cfg['main_end_search_after_peak_seconds'])+1)
        after=quiet_runs(signed[peak:right],offset_threshold,n(cfg['offset_quiet_seconds']))
        if after:
            offset=peak+after[0][0]
            main_confirmation=offset+n(cfg['offset_quiet_seconds'])-1
        else:
            offset=right;main_confirmation=right-1;local_flags.append('main_offset_unconfirmed')
        offset=min(count,max(onset+1,offset))
        duration=(offset-onset)/fs
        if not cfg['min_main_duration_seconds']<=duration<=cfg['max_main_duration_seconds']:local_flags.append('main_duration_outside_bounds')
        # Look for multiple same-polarity local lobes on the same comparable
        # branch; only flags, never silently split/merge accepted detector IDs.
        peaks,_=signal.find_peaks(signed[search_a:right],distance=n(cfg['multi_peak_audit']['minimum_distance_seconds']),
                                 prominence=cfg['multi_peak_audit']['prominence_uv'])
        peak_indices=[search_a+int(p) for p in peaks if signed[search_a+p]>=cfg['minimum_localization_peak_uv']]
        separated=[]
        for a,b in zip(peak_indices,peak_indices[1:]):
            valley=float(np.min(signed[a:b+1]));limit=cfg['multi_peak_audit']['valley_fraction']*min(signed[a],signed[b])
            if valley<=limit:separated.append((a,b))
        if separated:local_flags.append('possible_adjacent_main_lobes')
        for a,b in separated:
            lobes.append(dict(blink_candidate_id=r.blink_candidate_id,first_peak_sample=a,second_peak_sample=b,
                              first_peak_seconds=a/fs,second_peak_seconds=b/fs,action='flag only; no automatic merge or split'))
        main_decision=max(int(r.decision_sample),trigger+n(cfg['peak_search_after_trigger_seconds']),main_confirmation,peak+n(cfg['main_end_search_after_peak_seconds']))
        # Recovery is assessed separately on OFFICIAL EEG, never inferred merely
        # from the unipolar main lobe ending. This is not proof of usability.
        rec_cfg=cfg['recovery'];rec_start=min(offset,count)
        rec_limit=min(count,rec_start+n(rec_cfg['maximum_after_main_seconds']))
        formal_peak=float(np.max(np.abs(formal[onset:min(offset,count)]-formal_baseline))) if onset<count else 0.
        rec_threshold=max(rec_cfg['floor_uv'],rec_cfg['fraction_of_formal_peak']*formal_peak)
        quiet=quiet_runs(np.abs(formal[rec_start:rec_limit]-formal_baseline),rec_threshold,n(rec_cfg['quiet_seconds']))
        if quiet:
            rec_offset=rec_start+quiet[0][0]
            rec_confirmation=rec_offset+n(rec_cfg['quiet_seconds'])-1
            rec_status='quiet_criterion_met_unvalidated'
        else:
            rec_offset=rec_limit;rec_confirmation=rec_limit-1
            rec_status='not_confirmed_within_horizon' if rec_limit<count else 'unconfirmed_at_eof'
        decision=max(main_decision,rec_confirmation)
        if rec_status=='unconfirmed_at_eof':decision=max(decision,rec_start+n(rec_cfg['maximum_after_main_seconds'])-1)
        if main_decision>=count:local_flags.append('main_decision_not_available')
        if contaminated is not None and contaminated[:,ref_a:min(count,decision+1)].any():local_flags.append('nonfinite_recovery_in_localization')
        reliable=not local_flags
        tail_ptp=np.ptp(official[:,rec_start:rec_offset],axis=1)*1e6 if rec_offset>rec_start else np.zeros(len(channels))
        row=dict(edf_name=r.edf_name,blink_candidate_id=r.blink_candidate_id,
                 trigger_crossing_sample=trigger,trigger_crossing_seconds=trigger/fs,
                 frozen_detector_decision_seconds=r.decision_seconds,
                 old_onset_sample=r.onset_sample,old_offset_sample_exclusive=r.offset_sample_exclusive,
                 old_onset_seconds=r.onset_seconds,old_offset_seconds=r.offset_seconds,
                 main_onset_sample=onset,main_peak_sample=peak,main_offset_sample_exclusive=offset,
                 main_onset_seconds=onset/fs,main_peak_seconds=peak/fs,main_offset_seconds=offset/fs,
                 recovery_onset_sample=rec_start,recovery_offset_sample_exclusive=rec_offset,
                 recovery_onset_seconds=rec_start/fs,recovery_offset_seconds=rec_offset/fs,
                 main_decision_sample=main_decision,main_decision_seconds=main_decision/fs,
                 decision_sample=decision,decision_seconds=decision/fs,decision_available=decision<count,
                 actual_decision_seconds=decision/fs if decision<count else None,main_decision_available=main_decision<count,
                 main_duration_seconds=duration,recovery_duration_seconds=(rec_offset-rec_start)/fs,
                 recovery_status=rec_status,recovery_criterion_uv=rec_threshold,recovery_channels=';'.join(cfg['frontal_channels']),
                 localization_signal='independent raw EEG: causal 60Hz notch + 10Hz lowpass, no highpass',
                 main_polarity=sign,localization_baseline_uv=baseline,localization_peak_uv=amplitude,
                 supporting_channels=';'.join(support),onset_threshold_uv=onset_threshold,offset_threshold_uv=offset_threshold,
                 localization_usable_for_provisional_comparison=reliable,localization_flags=';'.join(local_flags),
                 recovery_proves_clean_signal=False,possible_merged_lobes=bool(separated),
                 filter_startup=onset<n(cfg['localization_branch']['startup_seconds']),
                 recovery_ptp_by_channel_uv=json.dumps({ch:float(v) for ch,v in zip(channels,tail_ptp)}),
                 policy_interval_source='provisional_main' if reliable else 'old_interval_fallback',
                 detector_parameters_unchanged=True,boundary_parameters=json.dumps(cfg,sort_keys=True))
        rows.append(row)
    result=pd.DataFrame(rows)
    if not len(result):
        result=pd.DataFrame(columns='edf_name blink_candidate_id main_onset_sample main_peak_sample main_offset_sample_exclusive main_onset_seconds main_peak_seconds main_offset_seconds recovery_onset_sample recovery_offset_sample_exclusive recovery_onset_seconds recovery_offset_seconds main_decision_seconds decision_seconds main_duration_seconds recovery_duration_seconds recovery_status localization_flags localization_usable_for_provisional_comparison possible_merged_lobes near_duplicate_localized_peak other_localized_mains_in_recovery policy_interval_source'.split())
    if len(result):
        # Keep distinct parent IDs even when two triggers re-anchor onto one
        # raw-domain lobe. Report for review, never collapse true blink events.
        result['near_duplicate_localized_peak']=False
        result['other_localized_mains_in_recovery']=0
        for i,r in result.iterrows():
            others=result.drop(index=i)
            result.loc[i,'near_duplicate_localized_peak']=bool((np.abs(others.main_peak_sample-r.main_peak_sample)<=n(cfg['multi_peak_audit']['near_duplicate_peak_seconds'])).any())
            result.loc[i,'other_localized_mains_in_recovery']=int(((others.main_onset_sample<r.recovery_offset_sample_exclusive)&(others.main_offset_sample_exclusive>r.recovery_onset_sample)).sum())
    return result,pd.DataFrame(lobes,columns=['blink_candidate_id','first_peak_sample','second_peak_sample','first_peak_seconds','second_peak_seconds','action'])


def union_duration(intervals):
    merged=[]
    for a,b in sorted(intervals):
        if b<=a:continue
        if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
        else:merged.append([a,b])
    return sum(b-a for a,b in merged)


def epoch_boundary_links(events,boundaries,fs,cfg):
    rows=[];flags=[]
    for e in events.itertuples():
        a=int(e.onset_sample);b=int(e.epoch_stop_sample_exclusive)
        intervals={key:[] for key in ['old','main','recovery','main_policy','main_recovery_policy']}
        ids={key:[] for key in intervals}
        decisions={key:[] for key in intervals}
        for r in boundaries.itertuples():
            reliable=r.localization_usable_for_provisional_comparison
            choices={'old':(r.old_onset_sample,r.old_offset_sample_exclusive,r.frozen_detector_decision_seconds),
                     'main':(r.main_onset_sample,r.main_offset_sample_exclusive,r.main_decision_seconds),
                     'recovery':(r.recovery_onset_sample,r.recovery_offset_sample_exclusive,r.decision_seconds),
                     'main_policy':(r.main_onset_sample,r.main_offset_sample_exclusive,r.main_decision_seconds) if reliable else (r.old_onset_sample,r.old_offset_sample_exclusive,r.frozen_detector_decision_seconds),
                     'main_recovery_policy':(r.main_onset_sample,r.recovery_offset_sample_exclusive,r.decision_seconds) if reliable else (r.old_onset_sample,r.old_offset_sample_exclusive,r.frozen_detector_decision_seconds)}
            for region,(start,stop,decision) in choices.items():
                left=max(a,int(start));right=min(b,int(stop))
                if right>left:
                    intervals[region].append((left,right));ids[region].append(r.blink_candidate_id);decisions[region].append(decision)
                    rows.append(dict(event_index=e.event_index,blink_candidate_id=r.blink_candidate_id,region=region,
                                     overlap_seconds=(right-left)/fs,decision_seconds=decision,
                                     known_by_epoch_decision=decision<=e.epoch_decision_seconds,
                                     earliest_joint_decision_seconds=max(decision,e.epoch_decision_seconds)))
        f=dict(event_index=e.event_index)
        for key in intervals:
            f[f'has_{key}_overlap']=bool(intervals[key]);f[f'{key}_overlap_seconds']=union_duration(intervals[key])/fs
            f[f'{key}_candidate_ids']=';'.join(ids[key])
        flags.append(f)
    return pd.DataFrame(flags),pd.DataFrame(rows,columns=['event_index','blink_candidate_id','region','overlap_seconds','decision_seconds','known_by_epoch_decision','earliest_joint_decision_seconds'])
