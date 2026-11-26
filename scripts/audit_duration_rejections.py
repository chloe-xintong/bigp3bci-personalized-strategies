"""All StudyF Train duration rejects against saved positive manual references."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from evaluate_pilot_blink_misses import union_events

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/duration_rejection_audit_v1'


def build_references():
    rows=[]
    config=json.loads((ROOT/'configs/expanded_pilot.json').read_text())
    folder=ROOT/config['manual_validation']['manual_root']/'manual_review_tables'
    for path in sorted(folder.glob('*_manual_review.csv')):
        table=pd.read_csv(path)
        assert table.type.isin(config['manual_validation']['legacy_blink_types']).all()
        for r in table.itertuples():
            rows.append(dict(edf_name=r.file,subject_id='_'.join(r.file.split('_')[:2]),
                             blind_id='legacy:'+r.manual_id,start_seconds=r.start_sec,end_seconds=r.end_sec))
    pilot=pd.read_csv(ROOT/'outputs/pilot_blink_misses/manual_span_mapping.csv')
    for r in pilot.itertuples():
        if pd.notna(r.exclude_reason):continue
        rows.append(dict(edf_name=r.edf_name,subject_id=r.subject_id,blind_id='pilot:'+r.blind_id,
                         start_seconds=r.start_seconds,end_seconds=r.end_seconds))
    eligible=[r for r in rows if r['start_seconds']>=10]
    refs=pd.DataFrame(union_events(eligible))
    refs['reference_id']=[f'REF{i:04d}' for i in range(1,len(refs)+1)]
    refs['source_blind_ids']=refs.source_blind_ids.map(';'.join)
    return refs,len(rows),len(eligible)


def main():
    manifest=pd.read_csv(ROOT/'outputs/studyf_all_preprocessing/input_manifest.csv')
    assert len(manifest)==90 and manifest.groupby(['subject_id','session_id']).ngroups==30
    cfg_path=ROOT/'configs/expanded_pilot.json';before=hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    cfg=json.loads(cfg_path.read_text())['blink_detector']
    refs,raw_ref_count,nonstartup_ref_count=build_references()
    manual_root=ROOT/json.loads(cfg_path.read_text())['manual_validation']['manual_root']/'manual_review_tables'
    reference_sources=[*sorted(manual_root.glob('*_manual_review.csv')),
                       ROOT/'outputs/pilot_blink_misses/manual_span_mapping.csv']
    reference_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in reference_sources}
    reject_rows=[];file_rows=[];reference_rows=[];provenance=[]
    for number,r in enumerate(manifest.itertuples(),1):
        folder=ROOT/r.results_path
        record=json.loads((folder/'processing_record.json').read_text())
        assert record['config']['blink_detector']==cfg
        fs=float(record['sfreq']);log_path=folder/'blink_excursion_log.csv'
        log=pd.read_csv(log_path,keep_default_na=False)
        accepted=pd.read_csv(folder/'blink_candidates.csv')
        accepted_logs=log[log.exclusion_reasons.eq('')]
        assert len(accepted_logs)==len(accepted)
        assert accepted_logs.onset_sample.tolist()==accepted.onset_sample.tolist()
        duration=log[log.exclusion_reasons.str.split(';').map(lambda v:'duration' in v)].copy()
        localrefs=refs[refs.edf_name.eq(r.edf_name)]
        if len(duration):
            with np.load(folder/'continuous_eeg.npz') as data:
                channels=data['channels'].tolist()
                z=data['auxiliary_eeg_v'][[channels.index(c) for c in cfg['frontal_channels']]]*1e6
            for i,event in duration.iterrows():
                a,b=int(event.onset_sample),int(event.offset_sample_exclusive)
                peak=a+int(np.argmax(np.abs(np.median(z[:,a:b],axis=0))))
                amp=float(np.median(np.abs(z[:,peak])))
                reasons=event.exclusion_reasons.split(';')
                assert (amp>=cfg['min_frontal_amplitude_uv'])==('amplitude' not in reasons)
                hits=localrefs[(localrefs.start_seconds<b/fs)&(localrefs.end_seconds>a/fs)]
                reject_rows.append(dict(edf_name=r.edf_name,subject_id=r.subject_id,session_id=r.session_id,
                    excursion_log_row=int(i)+1,onset_seconds=a/fs,offset_seconds=b/fs,duration_seconds=(b-a)/fs,
                    duration_side='too_long' if (b-a)/fs>cfg['max_duration_seconds'] else 'too_short',
                    filter_startup=a<round(record['config']['filter']['startup_seconds']*fs),
                    peak_frontal_amplitude_uv=amp,amplitude_pass=amp>=cfg['min_frontal_amplitude_uv'],
                    duration_only=reasons==['duration'],exclusion_reasons=event.exclusion_reasons,
                    overlapping_reference_ids=';'.join(hits.reference_id),overlaps_manual_blink=bool(len(hits))))
        for ref in localrefs.itertuples():
            hits=accepted[(accepted.onset_seconds<ref.end_seconds)&(accepted.offset_seconds>ref.start_seconds)]
            reject=[v for v in reject_rows if v['edf_name']==r.edf_name and v['onset_seconds']<ref.end_seconds and v['offset_seconds']>ref.start_seconds]
            amp_reject=[v for v in reject if v['amplitude_pass'] and not v['filter_startup']]
            only=[v for v in amp_reject if v['duration_only']]
            reference_rows.append(dict(**ref._asdict(),has_accepted_overlap=bool(len(hits)),
                accepted_candidate_ids=';'.join(hits.blink_candidate_id),has_duration_amplitude_reject_overlap=bool(amp_reject),
                has_duration_only_reject_overlap=bool(only),
                unmatched_with_duration_amplitude_reject=bool(amp_reject) and not len(hits),
                unmatched_with_duration_only_reject=bool(only) and not len(hits)))
        file_rows.append(dict(edf_name=r.edf_name,subject_id=r.subject_id,session_id=r.session_id,
            logged_excursions=len(log),accepted_candidates=len(accepted),duration_rejected=len(duration),
            manually_annotated_reference_events=len(localrefs)))
        provenance.append(dict(edf_name=r.edf_name,excursion_log_sha256=hashlib.sha256(log_path.read_bytes()).hexdigest()))
        if number%15==0:print(f'Processed {number}/90',flush=True)
    rejected=pd.DataFrame(reject_rows);references=pd.DataFrame(reference_rows).drop(columns=['Index'])
    def counts(frame):
        both=frame[frame.amplitude_pass & frame.overlaps_manual_blink]
        return dict(duration_rejected=len(frame),too_long=int(frame.duration_side.eq('too_long').sum()),
                    too_short=int(frame.duration_side.eq('too_short').sum()),
                    duration_only=int(frame.duration_only.sum()),
                    amplitude_pass=int(frame.amplitude_pass.sum()),amplitude_pass_manual_overlap=len(both),
                    amplitude_pass_manual_overlap_duration_only=int(both.duration_only.sum()),
                    manual_overlap_fraction_among_amplitude_pass=float(len(both)/frame.amplitude_pass.sum()) if frame.amplitude_pass.sum() else None)
    def reference_counts(frame):
        return dict(manual_events=len(frame),accepted_overlap=int(frame.has_accepted_overlap.sum()),
                    duration_amplitude_reject_overlap=int(frame.has_duration_amplitude_reject_overlap.sum()),
                    unmatched_with_duration_amplitude_reject=int(frame.unmatched_with_duration_amplitude_reject.sum()),
                    unmatched_with_duration_only_reject=int(frame.unmatched_with_duration_only_reject.sum()),
                    unmatched_with_duration_only_fraction=float(frame.unmatched_with_duration_only_reject.mean()))
    result=dict(scope='90 StudyF Train EDFs, 30 sessions; v1 unchanged; no final Test/Static-external evaluation performed.',
        all_logged_duration_rejections=counts(rejected),nonstartup_duration_rejections=counts(rejected[~rejected.filter_startup]),
        nonstartup_by_duration_side={side:counts(group) for side,group in rejected[~rejected.filter_startup].groupby('duration_side')},
        raw_positive_reference_spans=raw_ref_count,nonstartup_positive_reference_spans_before_dedup=nonstartup_ref_count,
        reference_events=reference_counts(references),
        by_subject_reference_events={s:reference_counts(g) for s,g in references.groupby('subject_id')},
        amplitude_threshold_uv=cfg['min_frontal_amplitude_uv'],entry_threshold_uv=cfg['entry_uv'],
        duration_limits_seconds=[cfg['min_duration_seconds'],cfg['max_duration_seconds']],
        notes=['Duration counts include all rejected merged threshold excursions, not accepted candidates.',
               'Too-short and too-long duration failures are reported separately. Entry threshold 20 uV and acceptance amplitude 25 uV are different gates.',
               'Manual reference combines legacy saved suspected-blink annotations and 30-clip positive annotations; positive intervals overlapping within EDF are union-deduplicated. Negative-only reviews do not create positive references.',
               'Unannotated duration rejections remain unknown. Manual-overlap fractions among all rejections are observed-reference fractions, not estimated prevalence of true blinks.',
               'Duration-only rejection overlapping a reference with no accepted overlap is evidence of loss under v1, not a counterfactual count of events that a modified detector would recover; changing acceptance can alter later peak-distance decisions.',
               'The reference-event fraction is descriptive for available positive annotations, not a global false-negative rate.'],
        config_sha256=before,reference_source_sha256=reference_hashes,provenance=provenance)
    assert hashlib.sha256(cfg_path.read_bytes()).hexdigest()==before
    assert all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==value for path,value in reference_hashes.items())
    OUT.mkdir(parents=True,exist_ok=True)
    rejected.to_csv(OUT/'duration_rejected_excursions.csv',index=False,encoding='utf-8-sig')
    rejected[rejected.amplitude_pass & rejected.overlaps_manual_blink].to_csv(OUT/'amplitude_pass_manual_overlap.csv',index=False,encoding='utf-8-sig')
    references.to_csv(OUT/'manual_reference_impact.csv',index=False,encoding='utf-8-sig')
    references[references.unmatched_with_duration_only_reject].to_csv(OUT/'manual_blinks_lost_duration_only.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(file_rows).to_csv(OUT/'by_file.csv',index=False,encoding='utf-8-sig')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='provenance'},indent=2))


if __name__=='__main__':main()
