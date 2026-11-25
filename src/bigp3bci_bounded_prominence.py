"""Finite-context peak prominence and conservative three-state confirmation."""
from __future__ import annotations
import numpy as np
from scipy import signal


def choose_reference(z,start,stop,fs,cfg,reverse=False):
    """Nearest locally mild-trend window, not necessarily perfectly flat."""
    w=max(3,int(np.ceil(cfg['reference_window_seconds']*fs)))
    step=max(1,int(np.ceil(cfg['reference_step_seconds']*fs)))
    positions=list(range(int(start),int(stop)-w+1,step))
    if reverse:positions.reverse()
    candidates=[]
    for a in positions:
        b=a+w;v=z[a:b]
        if len(v)<w or not np.isfinite(v).all():continue
        spread=float(np.quantile(v,.9)-np.quantile(v,.1))
        slope=float(np.polyfit(np.arange(w)/fs,v,1)[0])
        candidate=dict(start_sample=a,stop_sample_exclusive=b,level_uv=float(np.median(v)),spread_uv=spread,slope_uv_per_sec=slope)
        candidates.append(candidate)
        if spread<=cfg['reference_max_q90_q10_uv'] and abs(slope)<=cfg['reference_max_abs_slope_uv_per_sec']:
            return candidate,True
    return (candidates[0] if candidates else None),False


def bounded_prominence(signal_v,anchor,fs,cfg,neighbor_intervals=()):
    """Uses only a bounded prefix ending at required_decision_sample.

    Prominence is computed on the actual signed local extremum, never on a
    fixed pre-window mean or absolute voltage. NaN/Inf is not passed to SciPy.
    """
    z=np.asarray(signal_v,float)*1e6
    common=np.median(z,axis=0);n=z.shape[1]
    radius=max(1,int(np.ceil(cfg['anchor_peak_search_radius_seconds']*fs)))
    half=max(2,int(np.ceil(cfg['prominence_half_window_seconds']*fs)))
    outer=max(2,int(np.ceil(cfg['reference_search_outer_seconds']*fs)))
    extent=radius+max(half,outer)+max(1,int(np.ceil(cfg['channel_peak_tolerance_seconds']*fs)))
    a=anchor-extent;b=anchor+extent+1
    out=dict(anchor_sample=int(anchor),required_decision_sample=int(b-1),observation_start_sample=int(a),observation_stop_sample_exclusive=int(b),
             prominence_uv=np.nan,selected_peak_sample=int(anchor),selected_polarity=0,channel_prominences_uv=[np.nan]*z.shape[0],
             left_reference_reliable=False,right_reference_reliable=False,reference_neighbor_overlap=False,
             local_peak_found=False,context_complete=False,feature_status='uncertain',feature_reasons='')
    if a<0 or b>n:
        out['feature_reasons']='finite_context_incomplete';return out
    if not np.isfinite(z[:,a:b]).all():
        out['feature_reasons']='nonfinite_context';return out
    out['context_complete']=True
    candidates=[]
    for polarity in [1,-1]:
        peaks,_=signal.find_peaks(polarity*common[a:b])
        for p in peaks:
            sample=a+int(p)
            if abs(sample-anchor)<=radius:
                candidates.append((abs(sample-anchor),sample,polarity))
    # Choose the nearest extremum to the frozen anchor. Prominence is NOT used
    # to choose a larger, more favorable neighboring blink after inspection.
    if candidates:
        _,peak,polarity=min(candidates,key=lambda v:(v[0],v[1],-v[2]))
    else:
        peak=int(anchor);polarity=1
    inner=max(1,int(np.ceil(cfg['reference_search_inner_seconds']*fs)))
    left,ok_left=choose_reference(common,peak-outer,peak-inner,fs,cfg,reverse=True)
    right,ok_right=choose_reference(common,peak+inner,peak+outer,fs,cfg)
    out.update(selected_peak_sample=peak,selected_polarity=polarity,left_reference_reliable=ok_left,right_reference_reliable=ok_right,
               local_peak_found=bool(candidates))
    for name,ref in [('left',left),('right',right)]:
        if ref:
            for key,value in ref.items():out[name+'_reference_'+key]=value
    overlap=False
    for ref in [left,right]:
        if ref:
            overlap |= any(ref['start_sample']<stop and ref['stop_sample_exclusive']>start for start,stop in neighbor_intervals)
    out['reference_neighbor_overlap']=bool(overlap)
    reasons=[]
    if not ok_left:reasons.append('left_background_unreliable')
    if not ok_right:reasons.append('right_background_unreliable')
    if overlap:reasons.append('background_overlaps_another_main')
    if not candidates:
        # A stable constant plateau has no independent local lobe. If the
        # background is unreliable, lack of an extremum remains uncertain.
        out['feature_status']='uncertain' if reasons else 'no_independent_peak'
        out['feature_reasons']=';'.join(reasons+['no_local_extremum'])
        return out
    # Context extends past both possible anchor shifts. wlen limits each
    # prominence evaluation to the declared local radius around its own peak.
    local=polarity*common[a:b]
    prominence,lb,rb=signal.peak_prominences(local,[peak-a],wlen=2*half+1)
    out.update(prominence_uv=float(prominence[0]),left_base_sample=a+int(lb[0]),right_base_sample=a+int(rb[0]),
               contour_level_uv=float(common[peak]-polarity*prominence[0]))
    tolerance=max(1,int(np.ceil(cfg['channel_peak_tolerance_seconds']*fs)))
    values=[]
    for ch in z:
        ps,_=signal.find_peaks(polarity*ch[a:b])
        allowed=[a+int(p) for p in ps if abs(a+int(p)-peak)<=tolerance]
        if not allowed:values.append(0.);continue
        q=min(allowed,key=lambda sample:abs(sample-peak))
        values.append(float(signal.peak_prominences(polarity*ch[a:b],[q-a],wlen=2*half+1)[0][0]))
    out['channel_prominences_uv']=values
    out['feature_status']='uncertain' if reasons else 'evaluable'
    out['feature_reasons']=';'.join(reasons)
    return out


def confirmation_status(features,threshold,cfg,duplicate_anchor=False):
    """Uncertain candidates stay retained, including closely spaced blinks."""
    if duplicate_anchor:return 'uncertain','near_duplicate_anchor_keep_for_review'
    if features['feature_status']=='uncertain':return 'uncertain',features['feature_reasons']
    if features['feature_status']=='no_independent_peak':return 'not_supported','no_independent_peak_in_reliable_context'
    support=int(np.sum(np.asarray(features['channel_prominences_uv'])>=threshold))
    if features['prominence_uv']>=threshold and support>=cfg['minimum_support_channels']:
        return 'supported','bounded_prominence_and_frontal_support'
    return 'not_supported','weak_bounded_prominence_in_reliable_context'
