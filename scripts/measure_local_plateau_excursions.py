"""Measure fixed local-platform contrasts without retuning old trigger parameters."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd
from bigp3bci_plateau_metrics import platform_excursion
from hashlib import sha256
out=ROOT/'outputs/local_plateau_metrics_v1';out.mkdir(parents=True,exist_ok=True)
cfg=json.loads((ROOT/'configs/local_plateau_metrics.json').read_text(encoding='utf-8-sig'))
source=ROOT/'configs/expanded_pilot.json';source_hash=sha256(source.read_bytes()).hexdigest()
files=pd.read_csv(ROOT/'outputs/blink_candidate_review/file_index.csv');rows=[]
for r in files.itertuples():
 folder=ROOT/'outputs/blink_boundary_localization_v1'/Path(r.edf_name).stem
 b=pd.read_csv(folder/'boundaries.csv')
 with np.load(folder/'localization_branch.npz') as f:y=f['eeg_v'];channels=f['channels'].tolist();fs=float(f['sfreq'])
 picks=[channels.index(ch) for ch in cfg['frontal_channels']]
 for v in b.itertuples():
  m=platform_excursion(y[picks],int(v.main_peak_sample),fs,cfg)
  decision=max(v.frozen_detector_decision_seconds,m['required_decision_sample']/fs)
  rows.append(dict(edf_name=r.edf_name,blink_candidate_id=v.blink_candidate_id,main_peak_seconds=v.main_peak_seconds,
                   old_local_baseline_uv=v.localization_baseline_uv,old_localization_peak_uv=v.localization_peak_uv,
                   localization_flags=v.localization_flags,filter_startup=v.filter_startup,
                   decision_seconds=decision,decision_available=decision<y.shape[1]/fs,**m))
result=pd.DataFrame(rows)
result.to_csv(out/'all_candidate_platform_metrics.csv',index=False)
negative=pd.read_csv(ROOT/'outputs/blink_false_positive_review_audit/reviewed_main_candidates.csv')
comparison=negative.merge(result,on=['edf_name','blink_candidate_id'],validate='one_to_one',suffixes=('_review',''))
comparison.to_csv(out/'human_marked_negative_comparison.csv',index=False)
summary=result.groupby('edf_name').agg(candidates=('blink_candidate_id','size'),meets_excursion_candidate=('meets_excursion_candidate','sum'),stable_platforms=('platforms_stable','sum')).reset_index()
summary.to_csv(out/'file_summary.csv',index=False)
assert sha256(source.read_bytes()).hexdigest()==source_hash
(out/'analysis_record.json').write_text(json.dumps(dict(candidates=len(result),human_marked_negative_candidates=len(comparison),
 detector_parameters_unchanged=True,source_config_sha256=source_hash,old_boundaries_unchanged=True,new_rejections_applied=False,
 limitation='exploratory fixed windows and candidate cutoffs; no platform truth or independent validation; unstable references unresolved',config=cfg),indent=2),encoding='utf8')
print(comparison[['edf_name','blink_candidate_id','old_localization_peak_uv','rise_from_pre_platform_uv','fall_to_post_platform_uv','local_excursion_uv','platforms_stable','meets_excursion_candidate']].to_string(index=False))
