"""Descriptive missed manual blink events in the 30 candidate-centered clips."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/pilot_blink_misses'
POSITIVE={'BAD','BAD_','BAD_blink','BLINK'}


def union_events(rows):
    """Deduplicate overlapping annotations in overlapping clips, within EDF only."""
    merged=[]
    for r in sorted(rows,key=lambda r:(r['edf_name'],r['start_seconds'],r['end_seconds'])):
        if merged and merged[-1]['edf_name']==r['edf_name'] and r['start_seconds']<merged[-1]['end_seconds']:
            merged[-1]['end_seconds']=max(merged[-1]['end_seconds'],r['end_seconds'])
            merged[-1]['source_blind_ids']=sorted(set(merged[-1]['source_blind_ids']+[r['blind_id']]))
            merged[-1]['annotation_count']+=1
        else:
            merged.append(dict(edf_name=r['edf_name'],subject_id=r['subject_id'],start_seconds=r['start_seconds'],
                               end_seconds=r['end_seconds'],source_blind_ids=[r['blind_id']],annotation_count=1))
    return merged


def main():
    packet=ROOT/'outputs/single_blind_review_v1'
    form_path=packet/'reviewer/review_form.csv'
    form=pd.read_csv(form_path,keep_default_na=False)
    key=pd.read_csv(packet/'coordinator_only/sampling_key.csv')
    pilot=form[form.phase.eq('pilot_30')].merge(key[['blind_id','edf_name','subject_id','peak_seconds']],on='blind_id',validate='one_to_one')
    assert len(pilot)==30 and pilot.span_review_status.ne('').all()
    manifest=pd.read_csv(ROOT/'outputs/blink_annotation_update/annotation_manifest.csv').set_index('edf_name')
    rows=[];clips=[];provenance=[];candidate_tables={}
    for r in pilot.itertuples():
        folder=ROOT/manifest.loc[r.edf_name,'results_path']
        record=json.loads((folder/'processing_record.json').read_text())
        fs=float(record['sfreq'])
        # Exactly reproduce load_clip's rounding and exclusive stop.
        center=int(round(r.peak_seconds*fs));a=max(0,center-int(4*fs));b=min(record['n_samples'],center+int(4*fs)+1)
        clip_start=a/fs;clip_end=b/fs
        path=packet/'reviewer/annotations'/(r.blind_id+'_spans.csv')
        annotations=pd.read_csv(path)
        for span in annotations.itertuples():
            if span.annotation_label not in POSITIVE:continue
            start=clip_start+span.start_seconds;end=clip_start+span.end_seconds
            reason='startup' if start<10 else ('clip_boundary' if span.start_seconds<=1/fs or span.end_seconds>=(b-a)/fs-1/fs else '')
            rows.append(dict(edf_name=r.edf_name,subject_id=r.subject_id,blind_id=r.blind_id,
                             start_seconds=start,end_seconds=end,exclude_reason=reason))
        clips.append(dict(blind_id=r.blind_id,edf_name=r.edf_name,start_seconds=clip_start,end_seconds=clip_end,
                          saved_status=r.span_review_status,positive_spans=int(annotations.annotation_label.isin(POSITIVE).sum())))
        provenance.append(dict(blind_id=r.blind_id,spans_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if r.edf_name not in candidate_tables:
            candidate_tables[r.edf_name]=pd.read_csv(folder/'blink_candidates.csv')
            assert record['config']['blink_detector']==json.loads((ROOT/'configs/expanded_pilot.json').read_text())['blink_detector']
    manual=union_events([r for r in rows if not r['exclude_reason']])
    events=[]
    for i,r in enumerate(manual,1):
        candidates=candidate_tables[r['edf_name']]
        hits=candidates[(candidates.onset_seconds<r['end_seconds'])&(candidates.offset_seconds>r['start_seconds'])]
        peak_hits=candidates[(candidates.peak_seconds>=r['start_seconds'])&(candidates.peak_seconds<r['end_seconds'])]
        distance=(candidates.onset_seconds-r['end_seconds']).clip(lower=0)+(r['start_seconds']-candidates.offset_seconds).clip(lower=0)
        nearest=candidates.loc[distance.idxmin()]
        events.append(dict(**r,manual_event_id=f'PILOT_M{i:04d}',
                           matched=bool(len(hits)),matched_candidate_ids=';'.join(hits.blink_candidate_id),
                           peak_contained=bool(len(peak_hits)),nearest_candidate_id=nearest.blink_candidate_id,
                           nearest_interval_gap_seconds=float(distance.min())))
    events=pd.DataFrame(events)
    events['source_blind_ids']=events.source_blind_ids.map(';'.join)
    summary=[]
    for subject,g in events.groupby('subject_id'):
        fn=int((~g.matched).sum());summary.append(dict(subject_id=subject,manual_events=len(g),matched=int(g.matched.sum()),
                                                      missed=fn,miss_fraction=fn/len(g)))
    # Subjects with no manually marked blinks have no FN denominator, not zero FN rate.
    for subject in sorted(set(pilot.subject_id)-set(events.subject_id)):
        summary.append(dict(subject_id=subject,manual_events=0,matched=0,missed=0,miss_fraction=None))
    fn=int((~events.matched).sum())
    result=dict(scope='Candidate-centered reviewed-clip recall only; not global detector recall.',
                clips=len(pilot),files=pilot.edf_name.nunique(),raw_manual_positive_spans=len(rows),
                excluded_startup_spans=sum(r['exclude_reason']=='startup' for r in rows),
                excluded_clip_boundary_spans=sum(r['exclude_reason']=='clip_boundary' for r in rows),
                deduplicated_manual_events=len(events),matched=int(events.matched.sum()),missed=fn,
                descriptive_miss_fraction=fn/len(events),descriptive_recall=float(events.matched.mean()),
                strict_peak_contained_events=int(events.peak_contained.sum()),
                sensitivity_with_100ms_interval_tolerance_missed=int(events.nearest_interval_gap_seconds.gt(.1).sum()),
                by_subject=summary,matching='Any positive-duration overlap with ANY old_candidate_overlap_v1 candidate in the same EDF, not just sampled candidates.',
                deduplication='Union overlapping manual intervals within EDF to avoid repeated events across overlapping clips; do not merge across EDFs.',
                limitations=['Assumes saved clip review covered the entire displayed approximately 8 seconds. No full-recording review coverage is created.',
                             'Sample centers were previously unmatched automatic candidates; this is not a random sample of true blinks or recording time. This miss fraction is not population sensitivity or whole-detector FNR.',
                             'Single reviewer, no independent adjudication; temporal nonoverlap can reflect detector/filter delay or tight manual boundaries.',
                             'No binomial confidence interval: overlapping clips, event clustering and the selected sampling frame invalidate an independent-event interpretation.',
                             'No label association analysis or detector parameter retuning performed.'],
                review_form_sha256=hashlib.sha256(form_path.read_bytes()).hexdigest(),provenance=provenance)
    OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT/'manual_span_mapping.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(clips).to_csv(OUT/'reviewed_clips.csv',index=False,encoding='utf-8-sig')
    events.to_csv(OUT/'manual_event_matches.csv',index=False,encoding='utf-8-sig')
    events[~events.matched].to_csv(OUT/'unmatched_manual_events.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(summary).sort_values('subject_id').to_csv(OUT/'by_subject.csv',index=False,encoding='utf-8-sig')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='provenance'},indent=2))


if __name__=='__main__':main()
