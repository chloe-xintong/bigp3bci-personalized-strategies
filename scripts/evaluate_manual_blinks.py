"""Compare saved manual blink spans with PTP candidates; never assume unmarked=clean."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
STEM='F_05_SE001_CBCol_Train02'
OUT=ROOT/'outputs/blink_validation'
OUT.mkdir(parents=True,exist_ok=True)
source=ROOT/'outputs/artifact_qc/continuous_manual_qc/manual/manual_review_tables'/f'{STEM}_manual_review.csv'
base=ROOT/'outputs/causal_pilot'/STEM
annotations=pd.read_csv(source)
record=json.loads((base/'processing_record.json').read_text())
fs=record['sfreq']; n=record['epoch_samples']
t=pd.read_csv(base/'epoch_metadata_quality.csv')
p=pd.read_csv(base/'epoch_ptp_uv.csv').set_index('event_index').loc[t.event_index]
assert np.array_equal(p.index,t.event_index)
assert annotations.file.eq(STEM+'.edf').all()
assert (annotations.end_sec>annotations.start_sec).all()
starts=np.ceil(annotations.start_sec.to_numpy()*fs).astype(int)
stops=np.ceil(annotations.end_sec.to_numpy()*fs).astype(int)
overlap=(t.onset_sample.to_numpy()[:,None]<stops)&((t.onset_sample.to_numpy()+n)[:,None]>starts)
positive=overlap.any(axis=1)
eligible=t.complete_epoch.to_numpy() & (t.onset_sample.to_numpy()>=round(record['config']['filter']['startup_seconds']*fs)) & ~t.channel_policy_pending.to_numpy()
# This scope excludes startup, incomplete and persistent channel-policy cases.
# It is an evaluation mask, not an annotation-dependent online rejection rule.
t['manual_blink_overlap']=positive
t['annotation_ids']=[';'.join(annotations.manual_id[hit]) for hit in overlap]
t['evaluation_eligible']=eligible
t['reference_status']=np.where(positive,'annotated_blink','unknown_unreviewed_coverage')
t['max_ptp_all_uv']=p.max(axis=1).to_numpy()
frontal=['EEG_F3','EEG_Fz','EEG_F4']
t['max_ptp_frontal_uv']=p[frontal].max(axis=1).to_numpy()
t.to_csv(OUT/'epoch_blink_overlap.csv',index=False)
rows=[]
for group,values in [('all16',t.max_ptp_all_uv.to_numpy()),('frontal3_exploratory',t.max_ptp_frontal_uv.to_numpy())]:
 for upper in [75,100,125,150,200,250,350]:
  rejected=(values>upper)&eligible
  known=positive&eligible
  affected_intervals=overlap[eligible].any(axis=0)
  hits=(overlap&rejected[:,None]).any(axis=0)
  rows.append(dict(channel_group=group,upper_uv=upper,eligible_epochs=int(eligible.sum()),
                   annotated_blink_epochs=int(known.sum()),detected_blink_epochs=int((rejected&known).sum()),
                   missed_blink_epochs=int((known&~rejected).sum()),
                   annotated_epoch_detection_fraction=float((rejected&known).sum()/known.sum()) if known.any() else np.nan,
                   annotated_intervals_with_eligible_epochs=int(affected_intervals.sum()),intervals_with_any_rejection=int((hits&affected_intervals).sum()),
                   rejected_total=int(rejected.sum()),rejected_outside_annotation_unknown=int((rejected&~positive).sum()),
                   precision=np.nan,false_positive_rate=np.nan,
                   note='positive-only annotation; outside spans unknown; same-file exploration, not held-out validation'))
pd.DataFrame(rows).to_csv(OUT/'threshold_blink_comparison.csv',index=False)
intervals=annotations.copy()
intervals['overlap_epochs']=overlap.sum(axis=0)
intervals['eligible_overlap_epochs']=overlap[eligible].sum(axis=0)
for upper in [75,100,125,150,250,350]:
 rejected=(t.max_ptp_all_uv.to_numpy()>upper)&eligible
 intervals[f'rejected_epochs_upper_{upper}_uv']=(overlap&rejected[:,None]).sum(axis=0)
intervals.to_csv(OUT/'manual_interval_detection.csv',index=False)
# Sensitivity analysis only. Padding is not assumed ground truth or filter delay correction.
sensitivity=[]
for tail in [0,.25,.5]:
 hit=(t.onset_sample.to_numpy()[:,None]<stops+round(tail*fs))&((t.onset_sample.to_numpy()+n)[:,None]>starts)
 known=hit.any(axis=1)&eligible
 for upper in [75,100,125,150,250]:
  reject=(t.max_ptp_all_uv.to_numpy()>upper)&eligible
  sensitivity.append(dict(post_annotation_padding_seconds=tail,upper_uv=upper,positive_epochs=int(known.sum()),detected=int((known&reject).sum()),recall_on_provisional_spans=float((known&reject).sum()/known.sum())))
pd.DataFrame(sensitivity).to_csv(OUT/'padding_sensitivity.csv',index=False)
(OUT/'analysis_record.json').write_text(json.dumps(dict(annotation_source=str(source.relative_to(ROOT)),semantic_source='User states saved bad spans are blinks; original files unchanged',n_annotations=len(annotations),reviewed_coverage='User confirms sparse examples in each viewing window; many blinks remain unmarked; no exhaustively reviewed negative intervals',label_usage='none in detection',thresholds_changed=False,training_only=True,interval_overlap='half-open integer sample ranges, ceil annotation endpoints',scope='blink artifact evaluation; existing data-integrity/startup checks retained separately',frontal_channels=frontal),indent=2),encoding='utf8')
print(pd.DataFrame(rows).query("channel_group=='all16'").to_string(index=False))

# Suggested review blocks: a deliberately enriched sample, not a representative test set.
requests=[]
for site,stem in [('Duke',STEM),('ETSU','F_03_SE001_CBCol_Train01')]:
 data=pd.read_csv(ROOT/'outputs/causal_pilot'/stem/'epoch_metadata_quality.csv')
 local_record=json.loads((ROOT/'outputs/causal_pilot'/stem/'processing_record.json').read_text())
 usable=data[data.complete_epoch & (data.onset_seconds>=10)].copy()
 selected=[]
 def add(center,reason):
  start=max(10.,float(center)-5.)
  stop=min(local_record['n_samples']/local_record['sfreq'],start+10.)
  if any(start < b and stop > a for a,b in selected): return
  selected.append((start,stop))
  requests.append(dict(site_inferred=site,edf_name=stem+'.edf',start_sec=round(start,3),end_sec=round(stop,3),selection_reason=reason,
                       review_status='not_reviewed',instruction='Review entire interval; annotate every blink and uncertain span; explicitly confirm blink-free remainder only after exhaustive review'))
 for _,row in usable.sort_values('max_ptp_uv',ascending=False).head(20).iterrows():
  if len(selected)>=2: break
  add(row.onset_seconds,'high PTP: blink or another transient?')
 borderline=usable.iloc[(usable.max_ptp_uv-100).abs().argsort()]
 for _,row in borderline.iterrows():
  if len(selected)>=4: break
  add(row.onset_seconds,'near exploratory 100uV cutoff')
 rng=np.random.default_rng(20261124)
 for idx in rng.permutation(len(usable)):
  if len(selected)>=6: break
  add(usable.iloc[idx].onset_seconds,'random epoch-centered block, review full interval')
pd.DataFrame(requests).to_csv(OUT/'suggested_review_intervals.csv',index=False)
