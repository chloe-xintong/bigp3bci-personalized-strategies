"""Evaluate every saved blink annotation with fixed parameters, including supplements.

The original eight-file manifest is unchanged. Supplementary training EDFs are
selected only because the user saved annotations, and are not independent
held-out validation. Unannotated spans are never assumed clean.
"""
from pathlib import Path
import copy
import hashlib
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('_MNE_FAKE_HOME_DIR',str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import mne
from bigp3bci_expanded_pilot import process_expanded_file, manual_comparison
from bigp3bci_preprocessing import file_hash
OUT=ROOT/'outputs/blink_annotation_update'
SITE={'F_03':'ETSU','F_05':'Duke','F_06':'Duke','F_07':'Duke','F_08':'Duke','F_20':'ETSU','F_21':'ETSU','F_23':'ETSU','F_24':'ETSU','F_25':'ETSU'}
SCIENCE_KEYS=['filter','epoch','continuous_qc','signal_quality','blink_detector','simulation']


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg_path=ROOT/'configs/expanded_pilot.json'
    config_hash=file_hash(cfg_path)
    cfg=json.loads(cfg_path.read_text())
    frozen=ROOT/'outputs/expanded_pilot/selection_manifest.csv'
    manifest_hash=file_hash(frozen)
    manual=ROOT/cfg['manual_validation']['manual_root']
    sources=sorted((manual/'manual_review_tables').glob('*_manual_review.csv'))
    cfg=copy.deepcopy(cfg)
    # User explicitly identifies the updated bad_span annotations as blinks.
    cfg['manual_validation']['sparse_blink_files']=[p.name.removesuffix('_manual_review.csv')+'.edf' for p in sources]
    manifest=[]
    for path in sources:
        stem=path.name.removesuffix('_manual_review.csv')
        subject='_'.join(stem.split('_')[:2]);session=stem.split('_')[2]
        edf=ROOT/'data/raw/bigP3BCI-data/StudyF'/subject/session/'Train/CBCol'/(stem+'.edf')
        if not edf.exists():raise FileNotFoundError(edf)
        original=ROOT/'outputs/expanded_pilot'/stem
        base=original if (original/'processing_record.json').exists() else OUT/'additional_edfs'/stem
        manifest.append(dict(edf_name=edf.name,subject_id=subject,session_id=session,site=SITE.get(subject,'unknown'),site_status='inferred',
                             in_frozen_pilot=base==original,source_edf=str(edf.relative_to(ROOT)),annotation_csv=str(path.relative_to(ROOT)),
                             results_path=str(base.relative_to(ROOT)),selection='user annotated training EDF; fixed detector; supplementary evaluation if outside frozen pilot'))
    pd.DataFrame(manifest).to_csv(OUT/'annotation_manifest.csv',index=False)
    summary=[];details=[];pair_rows=[];candidate_rows=[];epoch_rows=[]
    for i,(item,source) in enumerate(zip(manifest,sources),1):
        edf=ROOT/item['source_edf'];base=ROOT/item['results_path']
        print(f'[{i}/{len(manifest)}] {edf.name}',flush=True)
        if (base/'processing_record.json').exists():
            r=json.loads((base/'processing_record.json').read_text())
            if any(r['config'][key]!=cfg[key] for key in SCIENCE_KEYS):raise ValueError(f'Cached detector/processing parameters differ: {base}')
            if r['source_sha256_after']!=file_hash(edf):raise ValueError(f'Source EDF differs from cached processing: {edf}')
        else:
            process_expanded_file(edf,item['site'],cfg,base,ROOT)
        record=json.loads((base/'processing_record.json').read_text())
        frame=pd.read_csv(source)
        fif=manual/'manual_annotations'/f'{edf.stem}-manual-annot.fif'
        consistent='FIF_missing'
        if fif.exists():
            ann=mne.read_annotations(fif)
            assert len(ann)==len(frame) and np.allclose(ann.onset,frame.start_sec,atol=1/record['sfreq']) and np.allclose(ann.onset+ann.duration,frame.end_sec,atol=1/record['sfreq'])
            assert np.array_equal(ann.description,frame.type)
            consistent='CSV_FIF_agree'
        frame=frame[frame.type.isin(['MANUAL_bad_span','MANUAL_blink','MANUAL_possible_blink','MANUAL_suspected_blink'])]
        b=pd.read_csv(base/'blink_candidates.csv')
        t=pd.read_csv(base/'epoch_candidates.csv');t.attrs['duration_seconds']=record['n_samples']/record['sfreq']
        long=pd.read_csv(base/'epoch_ptp_long.csv')
        ptp=long.pivot(index='event_index',columns='channel',values='ptp_uv').reindex(index=t.event_index,columns=record['channels']).to_numpy()
        validation_dir=OUT/'per_edf'/edf.stem;validation_dir.mkdir(parents=True,exist_ok=True)
        status=manual_comparison(edf,b,t,ptp,record['channels'],record['sfreq'],cfg,ROOT,validation_dir)
        pair=pd.read_csv(validation_dir/'manual_candidate_overlaps.csv');pair['edf_name']=edf.name;pair_rows.append(pair)
        startup=record['config']['filter']['startup_seconds'];local=[]
        for _,a in frame.iterrows():
            overlap=b[(b.onset_seconds<a.end_sec)&(b.offset_seconds>a.start_sec)]
            nonstartup_hits=overlap[overlap.filter_startup.eq(False)]
            epoch=t[(t.onset_seconds<a.end_sec)&(t.epoch_stop_sample_exclusive/record['sfreq']>a.start_sec)]
            intervals=sorted([(max(a.start_sec,c.onset_seconds),min(a.end_sec,c.offset_seconds)) for _,c in overlap.iterrows()])
            union=[]
            for left,right in intervals:
                if union and left<=union[-1][1]:union[-1][1]=max(union[-1][1],right)
                else:union.append([left,right])
            d=dict(edf_name=edf.name,manual_id=a.manual_id,start_sec=a.start_sec,end_sec=a.end_sec,startup=a.start_sec<startup,
                   overlapping_candidates=';'.join(overlap.blink_candidate_id),has_candidate_overlap=bool(len(overlap)),
                   has_nonstartup_candidate_overlap=bool(len(nonstartup_hits)),manual_duration_covered_fraction=sum(right-left for left,right in union)/(a.end_sec-a.start_sec),
                   overlap_epochs=len(epoch),nonstartup_overlap_epochs=int(epoch.filter_startup.eq(False).sum()),
                   epoch_indices=';'.join(epoch.event_index.astype(str)),max_overlapping_epoch_ptp_uv=epoch.max_ptp_uv.max() if len(epoch) else None,
                   interpretation='descriptive only; unmatched is not confirmed missed blink without exhaustive review')
            details.append(d);local.append(d)
        local=pd.DataFrame(local)
        after=local[~local.startup]
        candidate=pd.read_csv(validation_dir/'manual_candidate_comparison.csv')
        candidate['edf_name']=edf.name
        # Descriptive overlap uses the actual pair table, independent of reviewed status.
        matched_ids=set(pair.blink_candidate_id) if len(pair) else set()
        candidate['overlaps_saved_manual']=candidate.blink_candidate_id.isin(matched_ids)
        candidate_rows.append(candidate)
        t=t.copy();t['has_manual_blink_overlap']=False
        for a in local.itertuples():
            ids=[int(v) for v in a.epoch_indices.split(';') if v]
            t.loc[t.event_index.isin(ids),'has_manual_blink_overlap']=True
        epoch_rows.append(t)
        rawq=pd.read_csv(base/'basic_quality_candidates.csv')
        summary.append(dict(**item,suspected_manual_spans=len(frame),source_consistency=consistent,
                            annotation_sha256=file_hash(source),nonstartup_spans=len(after),
                            nonstartup_spans_with_overlap=int(after.has_nonstartup_candidate_overlap.sum()),
                            nonstartup_spans_without_overlap=int((~after.has_nonstartup_candidate_overlap).sum()),
                            startup_spans=int(local.startup.sum()),all_spans_with_any_overlap=int(local.has_candidate_overlap.sum()),
                            auto_candidates=len(b),nonstartup_auto_candidates=int(b.filter_startup.eq(False).sum()),
                            nonstartup_auto_candidates_overlapping_manual=int(candidate.loc[candidate.filter_startup.eq(False),'overlaps_saved_manual'].sum()),
                            auto_candidates_without_manual_overlap=int((~candidate.overlaps_saved_manual).sum()),
                            flashes=len(t),epochs_with_candidate=int(t.has_blink_candidate.sum()),epochs_with_manual_overlap=int(t.has_manual_blink_overlap.sum()),
                            jump_candidates=int(rawq.label.eq('CANDIDATE_jump').sum()),complete_reviewed_intervals=status['complete_reviewed_intervals'],
                            reviewed_event_recall=status['reviewed_event_recall'],reviewed_candidate_match_fraction=status['reviewed_candidate_match_fraction'],
                            thresholds_changed=False))
        print(f'  manual={len(frame)}; nonstartup overlap={int(after.has_nonstartup_candidate_overlap.sum())}/{len(after)}; auto={len(b)}',flush=True)
    summary=pd.DataFrame(summary);detail=pd.DataFrame(details)
    summary.to_csv(OUT/'annotation_update_summary.csv',index=False)
    detail.to_csv(OUT/'manual_candidate_details.csv',index=False)
    detail[~detail.startup & ~detail.has_nonstartup_candidate_overlap].to_csv(OUT/'unmatched_nonstartup_manual.csv',index=False)
    pairs=pd.concat(pair_rows,ignore_index=True);pairs.to_csv(OUT/'manual_candidate_pairs.csv',index=False)
    candidates=pd.concat(candidate_rows,ignore_index=True);candidates.to_csv(OUT/'all_annotated_file_candidates.csv',index=False)
    annotated_epochs=pd.concat(epoch_rows,ignore_index=True);annotated_epochs.to_csv(OUT/'all_annotated_file_epochs.csv',index=False)
    # Include all twelve processed files without double-counting annotated pilot files.
    original=pd.read_csv(ROOT/'outputs/expanded_pilot/selection_manifest.csv')
    all_files=[]
    for r in original.to_dict('records'):
        base=ROOT/'outputs/expanded_pilot'/Path(r['edf_name']).stem
        all_files.append((r['edf_name'],r['subject_id'],r['session_id'],r['site'],True,base))
    for r in manifest:
        if not r['in_frozen_pilot']:all_files.append((r['edf_name'],r['subject_id'],r['session_id'],r['site'],False,ROOT/r['results_path']))
    all_summary=[]
    for edf,subject,session,site,original,base in all_files:
        rec=json.loads((base/'processing_record.json').read_text())
        e=pd.read_csv(base/'epoch_candidates.csv');b=pd.read_csv(base/'blink_candidates.csv')
        q=pd.read_csv(base/'basic_quality_candidates.csv');sim=pd.read_csv(base/'threshold_simulation_epochs.csv')
        row=dict(edf_name=edf,subject_id=subject,session_id=session,site=site,site_status='inferred',in_frozen_pilot=original,
                 manually_annotated=edf in set(summary.edf_name),flashes=len(e),auto_candidates=len(b),
                 nonstartup_auto_candidates=int(b.filter_startup.eq(False).sum()),epochs_with_candidate=int(e.has_blink_candidate.sum()),
                 filter_startup_epochs=int(e.filter_startup.sum()),jump_candidates=int(q.label.eq('CANDIDATE_jump').sum()),
                 other_basic_candidates=int(q.label.ne('CANDIDATE_jump').sum()))
        for scheme in ['all16_upper_150uv','all16_upper_250uv','all16_no_upper']:
            values=sim[sim.scheme.eq(scheme)]
            row[scheme+'_expected_kept']=int(values.would_keep.sum())
        all_summary.append(row)
    pd.DataFrame(all_summary).to_csv(OUT/'all_processed_detection_summary.csv',index=False)
    assert file_hash(cfg_path)==config_hash and file_hash(frozen)==manifest_hash
    totals=dict(annotated_files=len(summary),manual_spans=int(summary.suspected_manual_spans.sum()),nonstartup_manual_spans=int(summary.nonstartup_spans.sum()),
                nonstartup_manual_spans_with_overlap=int(summary.nonstartup_spans_with_overlap.sum()),
                nonstartup_manual_spans_without_overlap=int(summary.nonstartup_spans_without_overlap.sum()),
                total_processed_files=len(all_summary),all_processed_blink_candidates=sum(r['auto_candidates'] for r in all_summary),
                all_processed_flashes=sum(r['flashes'] for r in all_summary),config_unchanged=True,frozen_manifest_unchanged=True,
                detector_config_sha256=config_hash,ground_truth_created=False,
                annotation_semantics='all saved bad_span files explicitly identified by user as suspected blinks; original labels unchanged',
                files_with_reviewed_intervals=int(summary.complete_reviewed_intervals.gt(0).sum()),
                metrics_unavailable=['overall recall across unreviewed recordings','sample-based false_positive_rate'],
                limitation='evaluate matching only within explicitly complete reviewed intervals; all outside-review candidates unknown')
    (OUT/'analysis_record.json').write_text(json.dumps(totals,indent=2),encoding='utf8')
    print(json.dumps(totals,indent=2),flush=True)

if __name__=='__main__':main()
