"""Per-manual-file comparison and explicitly exploratory criterion ablations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/bounded_prominence_secondary_v1'
p=pd.read_csv(OUT/'legacy_positive_example_comparison.csv')
n=pd.read_csv(OUT/'marked_negative_comparison.csv')
f=pd.read_csv(OUT/'candidate_features.csv').set_index('blink_candidate_id')
s=pd.read_csv(OUT/'candidate_secondary_status.csv')
pp=p[~p.startup & ~p.positive_negative_review_conflict]
nn=n[~n.startup]
# Preserve conflict and startup records separately instead of relabelling them.
p[p.startup | p.positive_negative_review_conflict].to_csv(OUT/'manual_positive_startup_or_conflicts.csv',index=False)
n[n.startup | n.legacy_positive_at_main_peak].to_csv(OUT/'manual_negative_startup_or_conflicts.csv',index=False)
pos=pp.groupby(['edf_name','threshold_uv','positive_reference_status']).size().unstack(fill_value=0)
for key in ['supported','uncertain_retained','all_parents_not_supported','no_old_parent_overlap']:
 if key not in pos:pos[key]=0
pos['manual_examples']=pos.sum(axis=1)
pos['retained_original_overlap']=pos.supported+pos.uncertain_retained
pos.reset_index().to_csv(OUT/'manual_positive_by_file.csv',index=False)
neg=nn.groupby(['edf_name','threshold_uv','secondary_status']).size().unstack(fill_value=0)
for key in ['not_supported','supported','uncertain']:
 if key not in neg:neg[key]=0
neg['marked_candidates']=neg.sum(axis=1)
neg.reset_index().to_csv(OUT/'manual_negative_by_file.csv',index=False)
rows=[];flagged_positives=[]
for threshold,g in s.groupby('threshold_uv'):
 full=g.set_index('blink_candidate_id').secondary_status.to_dict()
 lookups={}
 for criterion in ['prominence_only','prominence_and_frontal_support','full_background_guard']:
  lookup={}
  for key,row in f.iterrows():
   if criterion=='full_background_guard':lookup[key]=full[key];continue
   if not row.context_complete:lookup[key]='uncertain';continue
   if not row.local_peak_found:lookup[key]='not_supported';continue
   values=np.asarray(list(json.loads(row.channel_prominences_uv).values()),float)
   supported=row.prominence_uv>=threshold and (criterion=='prominence_only' or int((values>=threshold).sum())>=2)
   lookup[key]='supported' if supported else 'not_supported'
  lookups[criterion]=lookup
  positives=pp[pp.threshold_uv==threshold]
  flags=uncertain=no_parent=retained=0
  for v in positives.itertuples():
   ids=[key for key in str(v.parent_ids).split(';') if key and key!='nan']
   states=[lookup[key] for key in ids]
   if not states:no_parent+=1
   elif 'supported' in states:retained+=1
   elif 'uncertain' in states:retained+=1;uncertain+=1
   else:
    flags+=1
    flagged_positives.append(dict(edf_name=v.edf_name,manual_id=v.manual_id,start_sec=v.start_sec,end_sec=v.end_sec,threshold_uv=threshold,criterion=criterion,parent_ids=';'.join(ids),status='would be unsupported if this ablation were applied; not actually deleted'))
  negatives=nn[nn.threshold_uv==threshold]
  negative_status=[lookup[key] for key in negatives.blink_candidate_id]
  rows.append(dict(threshold_uv=threshold,criterion=criterion,positive_manual_examples=len(positives),positive_original_parent_absent=no_parent,
    positive_examples_flagged=flags,positive_examples_retained=retained,positive_examples_retained_via_uncertainty=uncertain,
    marked_negative_candidates=len(negatives),negative_not_supported=negative_status.count('not_supported'),
    negative_supported=negative_status.count('supported'),negative_uncertain=negative_status.count('uncertain'),
    note='development ablation; removing background guard is an unsafe control, not a recommended rule'))
pd.DataFrame(rows).to_csv(OUT/'criterion_ablation_on_manual_groups.csv',index=False)
pd.DataFrame(flagged_positives).to_csv(OUT/'ablation_flagged_positive_examples.csv',index=False)
clean=nn[~nn.legacy_positive_at_main_peak]
clean.groupby(['edf_name','threshold_uv','secondary_status']).size().unstack(fill_value=0).reset_index().to_csv(OUT/'manual_negative_by_file_no_conflict.csv',index=False)
print('POSITIVE BY FILE, all full-rule thresholds preserve the same event totals:')
print(pos.reset_index().query('threshold_uv==12')[['edf_name','manual_examples','supported','uncertain_retained','all_parents_not_supported','no_old_parent_overlap']].to_string(index=False))
print('NEGATIVE BY FILE:')
print(neg.reset_index().to_string(index=False))
print('ABLATION:')
print(pd.DataFrame(rows).to_string(index=False))
