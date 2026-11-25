"""Local-platform morphology metrics, independent of blink/target labels."""
import numpy as np


def platform_excursion(signal_v,peak,fs,cfg):
    """Measure local rise and return on a pre-specified causal signal domain.

    A high absolute offset is insufficient: an independent lobe needs contrast
    with both adjacent platforms. Missing/unstable platforms remain uncertain.
    """
    z=np.asarray(signal_v,float)*1e6
    n=z.shape[1]
    ranges=[]
    for name in ['pre','post']:
        left,right=cfg[f'{name}_platform_window_relative_peak_seconds']
        a=peak+int(np.ceil(left*fs));b=peak+int(np.ceil(right*fs))
        ranges.append((a,b))
    result=dict(required_decision_sample=max(peak,ranges[1][1]-1),platform_context_complete=all(0<=a<b<=n for a,b in ranges))
    if not result['platform_context_complete'] or not np.isfinite(z[:,max(0,ranges[0][0]):min(n,ranges[1][1])]).all():
        result.update(pre_level_uv=None,post_level_uv=None,local_excursion_uv=None,meets_excursion_candidate=False,platforms_stable=False,support_channels=0,measurement_status='context_missing_or_nonfinite')
        return result
    pre=z[:,ranges[0][0]:ranges[0][1]];post=z[:,ranges[1][0]:ranges[1][1]]
    before=np.median(pre,axis=1);after=np.median(post,axis=1);height=z[:,peak]
    up_before=float(np.median(height-before));up_after=float(np.median(height-after))
    up=min(up_before,up_after);down=min(-up_before,-up_after)
    sign=1 if up>=down else -1
    rise=sign*up_before;fall=sign*up_after
    excursion=max(0.,min(rise,fall))
    q0,q1=cfg['platform_spread_quantiles']
    spreads=[];slopes=[]
    for segment in [pre,post]:
        common=np.median(segment,axis=0)
        spreads.append(float(np.quantile(common,q1)-np.quantile(common,q0)))
        t=np.arange(len(common))/fs
        slopes.append(float(np.polyfit(t,common,1)[0]))
    stable=max(spreads)<=cfg['candidate_max_platform_spread_uv'] and max(abs(v) for v in slopes)<=cfg['candidate_max_abs_platform_slope_uv_per_sec']
    supports=np.minimum(sign*(height-before),sign*(height-after))>=cfg['candidate_local_excursion_uv']
    result.update(pre_level_uv=float(np.median(before)),post_level_uv=float(np.median(after)),peak_level_uv=float(np.median(height)),
                  selected_polarity=sign,rise_from_pre_platform_uv=rise,fall_to_post_platform_uv=fall,
                  local_excursion_uv=excursion,pre_platform_spread_uv=spreads[0],post_platform_spread_uv=spreads[1],
                  pre_platform_slope_uv_per_sec=slopes[0],post_platform_slope_uv_per_sec=slopes[1],
                  platforms_stable=stable,support_channels=int(supports.sum()),
                  meets_excursion_candidate=excursion>=cfg['candidate_local_excursion_uv'] and int(supports.sum())>=cfg['minimum_support_channels'],
                  measurement_status='stable_platforms' if stable else 'unstable_platforms_do_not_force_decision')
    return result
