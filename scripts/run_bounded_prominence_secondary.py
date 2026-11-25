"""Fixed-parent bounded-prominence development experiment; no automatic rejection."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd
import scipy
from hashlib import sha256
from bigp3bci_bounded_prominence import bounded_prominence,confirmation_status
OUT=ROOT/'outputs/bounded_prominence_secondary_v1';OUT.mkdir(parents=True,exist_ok=True)


def main():
 cfg=json.loads((ROOT/'configs/bounded_prominence_secondary.json').read_text(encoding='utf-8-sig'))
 detector_path=ROOT/'configs/expanded_pilot.json';before=sha256(detector_path.read_bytes()).hexdigest()
 manifest=pd.read_csv(ROOT/'outputs/blink_candidate_review/file_index.csv')
 all_features=[];all_status=[];positive_rows=[];adjacent_rows=[]
 negatives=pd.read_csv(ROOT/'outputs/blink_false_positive_review_audit/reviewed_main_candidates.csv')
 negative_ids=set(negatives.blink_candidate_id)
 for file in manifest.itertuples():
  stem=Path(file.edf_name).stem
  base=ROOT/'outputs/blink_boundary_localization_v1'/stem
  boundaries=pd.read_csv(base/'boundaries.csv')
  source_hash=sha256((base/'boundaries.csv').read_bytes()).hexdigest()
  with np.load(base/'localization_branch.npz') as f:y=f['eeg_v'];channels=f['channels'].tolist();fs=float(f['sfreq'])
  picks=[channels.index(ch) for ch in cfg['frontal_channels']]
  statuses={t:{} for t in cfg['prominence_candidates_uv']}
  for row in boundaries.itertuples():
   anchor=int(row.main_peak_sample)
   radius=int(np.ceil(cfg['anchor_peak_search_radius_seconds']*fs));half=int(np.ceil(cfg['prominence_half_window_seconds']*fs))
   tol=int(np.ceil(cfg['channel_peak_tolerance_seconds']*fs))
   decision=max(row.frozen_detector_decision_seconds,row.main_decision_seconds,(anchor+radius+half+tol)/fs)
   # Only context about OTHER provisional events already available by the
   # current declared decision may qualify a reference as contaminated.
   neighbors=boundaries[(boundaries.blink_candidate_id!=row.blink_candidate_id)&(boundaries.decision_seconds<=decision)&boundaries.localization_usable_for_provisional_comparison]
   intervals=list(zip(neighbors.main_onset_sample.astype(int),neighbors.main_offset_sample_exclusive.astype(int)))
   f=bounded_prominence(y[picks],anchor,fs,cfg,intervals)
   decision=max(decision,f['required_decision_sample']/fs)
   duplicate=bool((np.abs(neighbors.main_peak_sample-anchor)<=int(np.ceil(cfg['near_duplicate_anchor_seconds']*fs))).any())
   other={k:v for k,v in f.items() if k!='channel_prominences_uv'}
   all_features.append(dict(edf_name=file.edf_name,blink_candidate_id=row.blink_candidate_id,anchor_seconds=anchor/fs,
      v1_main_flags=row.localization_flags,v1_main_usable=row.localization_usable_for_provisional_comparison,
      filter_startup=row.filter_startup,decision_seconds=decision,decision_available=decision<y.shape[1]/fs,
      near_duplicate_available_anchor=duplicate,
      channel_prominences_uv=json.dumps(dict(zip(cfg['frontal_channels'],f['channel_prominences_uv']))),**other))
   for threshold in cfg['prominence_candidates_uv']:
    status,reason=confirmation_status(f,threshold,cfg,duplicate)
    statuses[threshold][row.blink_candidate_id]=status
    all_status.append(dict(edf_name=file.edf_name,blink_candidate_id=row.blink_candidate_id,threshold_uv=threshold,
       secondary_status=status,reason=reason,would_retain_candidate=status!='not_supported',
       reference_credible=f['feature_status']!='uncertain',prominence_uv=f['prominence_uv'],
       support_channels=int(np.sum(np.asarray(f['channel_prominences_uv'])>=threshold)),
       filter_startup=row.filter_startup,decision_seconds=decision,experimental_only=True))
  manual=ROOT/'outputs/artifact_qc/continuous_manual_qc/manual/manual_review_tables'/(stem+'_manual_review.csv')
  if manual.exists():
   m=pd.read_csv(manual);m=m[m.type.isin(['MANUAL_bad_span','MANUAL_blink','MANUAL_possible_blink','MANUAL_suspected_blink'])]
   entries=[]
   for row in m.itertuples():
    # Preserve the original detector/manual overlap relation for comparison.
    parents=boundaries[(boundaries.old_onset_seconds<row.end_sec)&(boundaries.old_offset_seconds>row.start_sec)] if len(boundaries) else boundaries
    parent_ids=parents.blink_candidate_id.tolist()
    conflict=bool(negative_ids.intersection(parent_ids))
    for threshold,lookup in statuses.items():
     ss=[lookup[key] for key in parent_ids]
     verdict='supported' if 'supported' in ss else 'uncertain_retained' if 'uncertain' in ss else 'all_parents_not_supported' if ss else 'no_old_parent_overlap'
     entry=dict(edf_name=file.edf_name,manual_id=row.manual_id,start_sec=row.start_sec,end_sec=row.end_sec,
       startup=row.start_sec<10,threshold_uv=threshold,parent_ids=';'.join(parent_ids),positive_reference_status=verdict,
       would_retain_original_overlap=verdict in ['supported','uncertain_retained'],positive_negative_review_conflict=conflict,
       reference_kind='legacy raw-domain user blink examples; not proof of new-branch boundary accuracy')
     positive_rows.append(entry);entries.append(entry)
   for threshold in cfg['prominence_candidates_uv']:
    group=sorted([e for e in entries if e['threshold_uv']==threshold],key=lambda e:e['start_sec'])
    for a,b in zip(group,group[1:]):
     gap=b['start_sec']-a['end_sec']
     if 0<=gap<=cfg['adjacent_manual_pair_gap_seconds']:
      adjacent_rows.append(dict(edf_name=file.edf_name,first_manual_id=a['manual_id'],second_manual_id=b['manual_id'],gap_seconds=gap,
          threshold_uv=threshold,both_have_old_parent=bool(a['parent_ids'] and b['parent_ids']),
          both_retained=a['would_retain_original_overlap'] and b['would_retain_original_overlap'],
          first_status=a['positive_reference_status'],second_status=b['positive_reference_status'],
          either_startup=a['startup'] or b['startup'],either_review_conflict=a['positive_negative_review_conflict'] or b['positive_negative_review_conflict']))
  assert sha256((base/'boundaries.csv').read_bytes()).hexdigest()==source_hash
 features=pd.DataFrame(all_features);status=pd.DataFrame(all_status);positive=pd.DataFrame(positive_rows)
 features.to_csv(OUT/'candidate_features.csv',index=False);status.to_csv(OUT/'candidate_secondary_status.csv',index=False)
 positive.to_csv(OUT/'legacy_positive_example_comparison.csv',index=False)
 adjacent=pd.DataFrame(adjacent_rows,columns=['edf_name','first_manual_id','second_manual_id','gap_seconds','threshold_uv','both_have_old_parent','both_retained','first_status','second_status','either_startup','either_review_conflict'])
 adjacent.to_csv(OUT/'adjacent_blink_pair_audit.csv',index=False)
 negative=negatives.merge(status,on=['edf_name','blink_candidate_id'],validate='one_to_many')
 negative.to_csv(OUT/'marked_negative_comparison.csv',index=False)
 status.groupby(['threshold_uv','secondary_status']).size().rename('candidates').reset_index().to_csv(OUT/'all_candidate_summary.csv',index=False)
 negative.groupby(['threshold_uv','startup','secondary_status']).size().rename('marked_candidates').reset_index().to_csv(OUT/'marked_negative_summary.csv',index=False)
 positive.groupby(['threshold_uv','startup','positive_negative_review_conflict','positive_reference_status']).size().rename('manual_examples').reset_index().to_csv(OUT/'positive_example_summary.csv',index=False)
 assert sha256(detector_path.read_bytes()).hexdigest()==before
 (OUT/'config_snapshot.json').write_text(json.dumps(cfg,indent=2),encoding='utf8')
 (OUT/'run_record.json').write_text(json.dumps(dict(parent_candidates=len(features),detector_parameters_unchanged=True,
   source_config_sha256=before,old_boundaries_unchanged=True,notebook11_unchanged=True,
   actual_suppression=False,best_threshold_selected=False,thresholds_inherit_old_25uv=False,
   positive_examples=len(positive)//len(cfg['prominence_candidates_uv']),negative_marked_candidates=len(negatives),
   scipy_version=scipy.__version__,source_reference=cfg['scipy_reference'],
   limitation='development labels overlap and are selected; conflicts separated; no unbiased precision/recall, no held-out validation or training',config=cfg),indent=2),encoding='utf8')
 # Refresh per-human-file tables whenever source annotations are re-evaluated.
 import runpy
 runpy.run_path(str(ROOT/'scripts/summarize_manual_prominence_groups.py'),run_name='__main__')
 print(status.groupby(['threshold_uv','secondary_status']).size().unstack(fill_value=0).to_string())
 print('NONSTARTUP MARKED NEGATIVES:')
 print(negative[~negative.startup].groupby(['threshold_uv','secondary_status']).size().unstack(fill_value=0).to_string())
 print('NONSTARTUP POSITIVES, NO NEGATIVE-REVIEW CONFLICT:')
 good=positive[~positive.startup & ~positive.positive_negative_review_conflict]
 print(good.groupby(['threshold_uv','positive_reference_status']).size().unstack(fill_value=0).to_string())

if __name__=='__main__':main()
