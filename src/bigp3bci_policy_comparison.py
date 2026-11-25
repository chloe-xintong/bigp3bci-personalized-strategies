"""Fixed-detector, three-family epoch handling comparison. Never drop samples."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def evaluate_policies(events,ptp,raw_links,blink_links,channels,fs,n_samples,detector_cfg,config):
    """Labels are carried through only; selection uses signal metrics/timestamps."""
    picks=[channels.index(ch) for ch in config['qc_channels']]
    schemes=[('faults_only',None),('ptp_250uv',config['ptp_primary_upper_uv']),('blink_overlap',None)]
    schemes += [(f'ptp_{upper:g}uv_sensitivity',upper) for upper in config['ptp_sensitivity_upper_uv']]
    guard=(int(np.ceil(detector_cfg['max_duration_seconds']*fs))+
           max(int(np.ceil(detector_cfg['recovery_seconds']*fs)),int(np.ceil(detector_cfg['merge_gap_seconds']*fs))))
    rows=[]
    for i,e in events.iterrows():
        base=[];fault_channels=set()
        if not e.complete_epoch:base.append('incomplete')
        if e.filter_startup:base.append('filter_startup')
        for flag,column,reason in [('nonfinite_flag','nonfinite_channels','nonfinite'),('filter_recovery','filter_recovery_channels','filter_recovery')]:
            names=set(filter(None,str(e[column]).split(';')))&set(config['qc_channels'])
            if e[flag] and names:base.append(reason);fault_channels.update(names)
        v=ptp[i,picks]
        if e.complete_epoch and not np.isfinite(v).all():
            base.append('filtered_nonfinite');fault_channels.update(np.array(config['qc_channels'])[~np.isfinite(v)])
        link=raw_links[(raw_links.event_index==e.event_index)&raw_links.channel.isin(config['qc_channels'])&raw_links.known_by_epoch_decision]
        for label,reason in [('CANDIDATE_constant','raw_constant'),('CANDIDATE_zero','raw_zero')]:
            names=link.loc[link.label.eq(label),'channel'].unique()
            if len(names):base.append(reason);fault_channels.update(names)
        bl=blink_links[blink_links.event_index==e.event_index]
        has_blink=bool(len(bl))
        if has_blink != bool(e.has_blink_candidate):raise ValueError('Blink link/epoch marker mismatch')
        for scheme,upper in schemes:
            reasons=base.copy();trigger=set(fault_channels)
            high=np.isfinite(v)&(v>upper) if upper is not None else np.zeros(len(picks),bool)
            if high.any():reasons.append('ptp_high');trigger.update(np.asarray(config['qc_channels'])[high])
            if scheme=='blink_overlap' and has_blink:reasons.append('blink_candidate_overlap')
            decision=e.epoch_decision_seconds
            if scheme=='blink_overlap':decision=max(decision,(e.epoch_stop_sample_exclusive+guard)/fs)
            if scheme=='blink_overlap' and len(bl):
                if decision+1e-9<bl.blink_decision_seconds.max():raise ValueError('Confirmation horizon insufficient')
            rows.append(dict(**e.to_dict(),strategy=scheme,ptp_upper_uv=upper,common_fault=bool(base),common_fault_reasons=';'.join(base),
                             would_keep=not reasons,would_reject=bool(reasons),rejection_reasons=';'.join(reasons),trigger_channels=';'.join(sorted(trigger)),
                             ptp_trigger_channels=';'.join(np.asarray(config['qc_channels'])[high]),
                             policy_decision_seconds=float(decision),additional_wait_seconds=float(decision-e.epoch_decision_seconds),
                             decision_data_available=bool(decision<=(n_samples-1)/fs),
                             pending_at_eof=bool(decision>(n_samples-1)/fs),
                             blink_known_at_epoch_decision=bool(len(bl)) and bool(bl.known_by_epoch_decision.all()),
                             policy_simulation_only=True))
    return pd.DataFrame(rows)


def summarize(frame,keys):
    return frame.groupby(['strategy']+keys,dropna=False).agg(total_flashes=('would_keep','size'),
            expected_kept=('would_keep','sum'),expected_retention=('would_keep','mean'),common_fault_epochs=('common_fault','sum'),
            pending_at_eof=('pending_at_eof','sum'),mean_extra_wait_seconds=('additional_wait_seconds','mean')).reset_index()
