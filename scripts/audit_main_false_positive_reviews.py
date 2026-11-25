"""Audit saved human false-positive main-wave reviews without changing parameters."""
from pathlib import Path
import json,os,sys
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('_MNE_FAKE_HOME_DIR',str(ROOT))
import pandas as pd,numpy as np,mne
from hashlib import sha256
OUT=ROOT/'outputs/blink_false_positive_review_audit';OUT.mkdir(parents=True,exist_ok=True)
summary=[];candidate_rows=[];mappings=[];legacy=[]
config=ROOT/'configs/expanded_pilot.json';before=sha256(config.read_bytes()).hexdigest()
for span_path in sorted((ROOT/'outputs/blink_candidate_review').glob('*/human_main_review_spans.csv')):
 s=pd.read_csv(span_path,keep_default_na=False)
 if not len(s):continue
 folder=span_path.parent;stem=folder.name
 fif=folder/(stem+'-human-main-review-annot.fif');a=mne.read_annotations(fif)
 b=pd.read_csv(ROOT/'outputs/blink_boundary_localization_v1'/stem/'boundaries.csv').set_index('blink_candidate_id')
 original=pd.read_csv(folder/'candidate_details.csv').set_index('blink_candidate_id')
 assert len(a)==len(s)
 np.testing.assert_allclose(a.onset,s.start_sec,atol=1/256)
 np.testing.assert_allclose(a.onset+a.duration,s.end_sec,atol=1/256)
 assert np.array_equal(a.description,s.label)
 manual_path=ROOT/'outputs/artifact_qc/continuous_manual_qc/manual/manual_review_tables'/(stem+'_manual_review.csv')
 old=pd.read_csv(manual_path) if manual_path.exists() else pd.DataFrame(columns=['start_sec','end_sec','manual_id'])
 ids=set()
 for i,r in s.iterrows():
  hits=b[(b.main_peak_seconds>=r.start_sec)&(b.main_peak_seconds<r.end_sec)]
  expected=set(filter(None,r.blink_candidate_ids.split(';')))
  assert expected==set(hits.index),'Saved peak mapping differs from main boundary snapshot'
  ids.update(hits.index)
  for key,v in hits.iterrows():mappings.append(dict(edf_name=stem+'.edf',span_index=i+1,start_sec=r.start_sec,end_sec=r.end_sec,blink_candidate_id=key,main_peak_seconds=v.main_peak_seconds))
 for key in sorted(ids):
  v=b.loc[key]
  raw_positive=old[(old.start_sec<=v.main_peak_seconds)&(old.end_sec>v.main_peak_seconds)]
  candidate_rows.append(dict(edf_name=stem+'.edf',blink_candidate_id=key,main_start_sec=v.main_onset_seconds,main_peak_sec=v.main_peak_seconds,
    main_end_sec=v.main_offset_seconds,old_start_sec=v.old_onset_seconds,old_end_sec=v.old_offset_seconds,
    startup=bool(original.loc[key,'filter_startup']),review_decision='not_blink',review_region='main',
    localization_internal_checks_passed=bool(v.localization_usable_for_provisional_comparison),localization_flags=v.localization_flags if pd.notna(v.localization_flags) else '',
    near_duplicate_main_peak=bool(v.near_duplicate_localized_peak),legacy_positive_at_main_peak=bool(len(raw_positive)),
    legacy_positive_ids=';'.join(raw_positive.manual_id),interpretation='human-selected suspected false positive on new main interval; not automatically a false frozen trigger'))
 decisions=pd.read_csv(folder/'review_decisions.csv',keep_default_na=False)
 prior=decisions[decisions.review_decision.ne('') & ~decisions.blink_candidate_id.isin(ids)]
 if len(prior):legacy.append(prior.assign(note='Existing decision outside currently saved main span mapping; preserve separately'))
 selected=pd.DataFrame([r for r in candidate_rows if r['edf_name']==stem+'.edf'])
 summary.append(dict(edf_name=stem+'.edf',saved_main_spans=len(s),unique_marked_main_candidates=len(ids),
    startup_marked=int(selected.startup.sum()),nonstartup_marked=int((~selected.startup).sum()),
    internally_uncertain=int((~selected.localization_internal_checks_passed).sum()),near_duplicate_peak_candidates=int(selected.near_duplicate_main_peak.sum()),
    legacy_positive_conflicts=int(selected.legacy_positive_at_main_peak.sum()),historical_other_decisions=len(prior),
    csv_fif_consistent=True,reviewed_complete_scope='not established by these negative-only annotations',
    span_csv_sha256=sha256(span_path.read_bytes()).hexdigest()))
pd.DataFrame(summary).to_csv(OUT/'file_summary.csv',index=False)
pd.DataFrame(candidate_rows).to_csv(OUT/'reviewed_main_candidates.csv',index=False)
pd.DataFrame(mappings).to_csv(OUT/'main_span_candidate_mapping.csv',index=False)
if legacy:pd.concat(legacy,ignore_index=True).to_csv(OUT/'historical_other_decisions.csv',index=False)
assert sha256(config.read_bytes()).hexdigest()==before
(OUT/'audit_record.json').write_text(json.dumps(dict(files=len(summary),marked_spans=sum(r['saved_main_spans'] for r in summary),marked_candidates=len(candidate_rows),
    detector_config_sha256=before,parameters_changed=False,original_annotations_changed=False,
    scope='main-wave negative review only; do not conflate with old trigger detection, missed events or boundary accuracy',
    false_positive_rate=None,precision=None,limitation='no exhaustive candidate-review coverage confirmed; unmatched/unmarked candidates remain unknown'),indent=2),encoding='utf8')
print(pd.DataFrame(summary).to_string(index=False))
