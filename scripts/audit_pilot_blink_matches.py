"""Audit timing, duplicate annotations and rejection logs without retuning."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/pilot_blink_misses'


def timing_summary(values):
    a=np.asarray(values,float)
    return dict(median=float(np.median(a)),q25=float(np.quantile(a,.25)),q75=float(np.quantile(a,.75)),
                median_absolute=float(np.median(np.abs(a))))


def main():
    manual=pd.read_csv(OUT/'manual_event_matches.csv')
    raw=pd.read_csv(OUT/'manual_span_mapping.csv')
    manifest=pd.read_csv(ROOT/'outputs/blink_annotation_update/annotation_manifest.csv').set_index('edf_name')
    cfg=json.loads((ROOT/'configs/expanded_pilot.json').read_text())['blink_detector']
    pairs=[];duplicate_rows=[]
    counts={name:dict(event_coverage=0,one_to_one_matches=0) for name in ['any_overlap','manual_coverage_0.5','manual_coverage_0.8','iou_0.3','iou_0.5']}
    for edf,events in manual.groupby('edf_name'):
        folder=ROOT/manifest.loc[edf,'results_path'];c=pd.read_csv(folder/'blink_candidates.csv')
        a=events.start_seconds.to_numpy()[:,None];b=events.end_seconds.to_numpy()[:,None]
        x=c.onset_seconds.to_numpy()[None,:];y=c.offset_seconds.to_numpy()[None,:]
        overlap=np.maximum(0,np.minimum(b,y)-np.maximum(a,x))
        iou=overlap/((b-a)+(y-x)-overlap);coverage=overlap/(b-a)
        rules={'any_overlap':overlap>0,'manual_coverage_0.5':coverage>=.5,'manual_coverage_0.8':coverage>=.8,'iou_0.3':iou>=.3,'iou_0.5':iou>=.5}
        for name,mask in rules.items():
            counts[name]['event_coverage']+=int(mask.any(axis=1).sum())
            counts[name]['one_to_one_matches']+=int((maximum_bipartite_matching(csr_matrix(mask),perm_type='column')>=0).sum())
        for i,event in enumerate(events.itertuples()):
            if not (overlap[i]>0).any():continue
            j=int(np.argmax(iou[i]));candidate=c.iloc[j]
            pairs.append(dict(manual_event_id=event.manual_event_id,edf_name=edf,blink_candidate_id=candidate.blink_candidate_id,
                manual_start=event.start_seconds,manual_end=event.end_seconds,candidate_start=candidate.onset_seconds,candidate_end=candidate.offset_seconds,
                overlap_seconds=float(overlap[i,j]),manual_coverage=float(coverage[i,j]),iou=float(iou[i,j]),
                onset_error_seconds=candidate.onset_seconds-event.start_seconds,
                offset_error_seconds=candidate.offset_seconds-event.end_seconds))
    for event in manual.itertuples():
        members=raw[(raw.edf_name.eq(event.edf_name))&(raw.start_seconds<event.end_seconds)&(raw.end_seconds>event.start_seconds)&raw.exclude_reason.isna()]
        assert len(members)==event.annotation_count
        for member in members.itertuples():
            duplicate_rows.append(dict(manual_event_id=event.manual_event_id,edf_name=event.edf_name,blind_id=member.blind_id,
                original_start=member.start_seconds,original_end=member.end_seconds,merged_start=event.start_seconds,merged_end=event.end_seconds,
                cluster_annotations=len(members),cluster_distinct_clips=members.blind_id.nunique()))
    duplicates=pd.DataFrame(duplicate_rows)
    repeated=duplicates[duplicates.cluster_annotations>1]
    pair_table=pd.DataFrame(pairs)
    log_rows=[];diagnoses=[];segment_rows=[]
    for missed in manual[~manual.matched].itertuples():
        folder=ROOT/manifest.loc[missed.edf_name,'results_path']
        record=json.loads((folder/'processing_record.json').read_text());fs=float(record['sfreq'])
        log=pd.read_csv(folder/'blink_excursion_log.csv',keep_default_na=False)
        log['onset_seconds']=log.onset_sample/fs;log['offset_seconds']=log.offset_sample_exclusive/fs
        hits=log[(log.onset_seconds<missed.end_seconds)&(log.offset_seconds>missed.start_seconds)].copy()
        hits['manual_event_id']=missed.manual_event_id;hits['duration_seconds']=hits.offset_seconds-hits.onset_seconds
        log_rows.extend(hits.to_dict('records'))
        with np.load(folder/'continuous_eeg.npz') as d:
            ch=d['channels'].tolist();z=d['auxiliary_eeg_v'][[ch.index(c) for c in cfg['frontal_channels']]]*1e6
            common=np.median(z,axis=0);amplitude=np.abs(common)
        # Replay the exact entry/exit hysteresis from saved auxiliary data to
        # expose pre-merge excursions; no new detector parameters are fitted.
        hold=max(1,int(np.ceil(cfg['recovery_seconds']*fs)));active=None;below=None;segments=[]
        for t,value in enumerate(amplitude):
            if active is None:
                if value>=cfg['entry_uv']:active=t;below=None
            elif value<=cfg['exit_uv']:
                if below is None:below=t
                if t-below+1>=hold:
                    segments.append((active,below,t));active=None;below=None
            else:below=None
        for hit in hits.itertuples():
            for start,stop,confirmed in segments:
                if start<hit.offset_sample_exclusive and stop>hit.onset_sample:
                    segment_rows.append(dict(manual_event_id=missed.manual_event_id,start_seconds=start/fs,end_seconds=stop/fs,
                                             duration_seconds=(stop-start)/fs,confirmed_seconds=confirmed/fs))
        a=int(np.floor(missed.start_seconds*fs));b=int(np.ceil(missed.end_seconds*fs))
        diagnoses.append(dict(manual_event_id=missed.manual_event_id,manual_duration=missed.end_seconds-missed.start_seconds,
                              auxiliary_common_max_uv_in_manual=float(amplitude[a:b].max()),entry_uv=cfg['entry_uv'],
                              min_duration=cfg['min_duration_seconds'],max_duration=cfg['max_duration_seconds'],
                              overlapping_excursion_logs=hits.to_dict('records')))
    for stats in counts.values():
        stats['denominator']=len(manual);stats['event_coverage_fraction']=stats['event_coverage']/len(manual)
        stats['one_to_one_fraction']=stats['one_to_one_matches']/len(manual)
    result=dict(scope='Candidate-centered reviewed-clip recall/coverage only; not global detector recall.',
        matching_thresholds_posthoc_descriptive=True,matching=counts,
        best_iou_pair_boundary_errors_seconds=dict(onset=timing_summary(pair_table.onset_error_seconds),offset=timing_summary(pair_table.offset_error_seconds)),
        best_pair_iou=timing_summary(pair_table.iou),best_pair_manual_coverage=timing_summary(pair_table.manual_coverage),
        deduplication=dict(original_spans=len(raw),unique_events=len(manual),extra_annotations=len(raw)-len(manual),
            repeated_event_clusters=int(repeated.manual_event_id.nunique()),
            merged_multiple_spans_from_same_clip=bool((repeated.groupby('manual_event_id').size()>repeated.groupby('manual_event_id').blind_id.nunique()).any()),
            rule='Sort by EDF and start time; merge only positive-duration-overlapping intervals, transitively. Touching intervals and different EDFs stay separate. Repeated cross-clip annotations are treated as the same event; this is an operational rule, not independently validated identity.'),
        missed_event_diagnoses=diagnoses,
        caveats=['Best-IoU boundary errors use one best candidate independently per matched manual event; they are descriptive and not the one-to-one assignment.',
                 'Stricter threshold failures combine timing/localization errors and missed events; do not label all as physiological blink false negatives.',
                 'Manual annotation duration is not auxiliary threshold-excursion duration. Exclusion logs refer to merged excursions before acceptance.',
                 'The 30 selected clips do not establish exhaustive review of full recordings. No population recall or confidence interval is estimated.'])
    pair_table.to_csv(OUT/'best_iou_pair_timing.csv',index=False,encoding='utf-8-sig')
    duplicates.to_csv(OUT/'deduplication_audit.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(log_rows).to_csv(OUT/'missed_event_excursion_logs.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(segment_rows).to_csv(OUT/'missed_event_premerge_segments.csv',index=False,encoding='utf-8-sig')
    (OUT/'matching_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
