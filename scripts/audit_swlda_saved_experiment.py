"""Independently audit saved train-only scaling, coefficients and score masks."""
from pathlib import Path
import argparse,json,os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
 os.environ[name]='1'
import numpy as np
import pandas as pd
from scipy import stats

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('output',type=Path)
 parser.add_argument('--reference',type=Path)
 args=parser.parse_args()
 out=args.output
 models=pd.read_csv(out/'model_summary.csv')
 scores=pd.read_csv(out/'epoch_scores.csv',low_memory=False)
 with np.load(out/'features_and_membership.npz') as d:x=d['features_uv']
 rows=[]
 for r in models.itertuples():
  q=scores.loc[scores.model_id.eq(r.model_id)&scores.policy.eq(r.policy)]
  train=q.loc[q.set.eq('train')&q.would_keep]
  a=x[train.feature_row.to_numpy(int)]
  finite=np.isfinite(a).all(axis=1)
  train=train.loc[finite];a=a[finite]
  y=train.label.map({'non-target':0,'target':1}).to_numpy(float)
  with np.load(out/'models'/f'{r.model_id}_{r.policy}.npz') as m:
   np.testing.assert_allclose(m['train_mean'],a.mean(axis=0),rtol=1e-10,atol=1e-10)
   expected=a.std(axis=0);expected=np.where(expected>1e-12,expected,1.)
   np.testing.assert_allclose(m['train_scale'],expected,rtol=1e-10,atol=1e-10)
   good=q.loc[q.has_score]
   np.testing.assert_allclose(good.target_score,x[good.feature_row.to_numpy(int)]@m['weights_uv']+m['intercept'],rtol=1e-9,atol=1e-10)
   indices=m['selected_features']
   design=np.column_stack([np.ones(len(a)),((a-m['train_mean'])/m['train_scale'])[:,indices]])
   beta=np.linalg.lstsq(design,y,rcond=1e-10)[0]
   residual=y-design@beta;df=len(y)-design.shape[1]
   variance=np.diag(np.linalg.pinv(design.T@design))*float(residual@residual)/df
   p=2*stats.t.sf(abs(beta[1:]/np.sqrt(variance[1:])),df)
   max_p=float(p.max()) if len(p) else np.nan
   if len(p) and max_p>.15001:raise AssertionError('Final feature removal criterion violated')
   rows.append(dict(model_id=r.model_id,policy=r.policy,training_epochs=len(a),n_selected=len(indices),max_final_selected_p=max_p,train_only_scaling_verified=True,saved_scores_verified=True))
 assert scores.loc[~scores.would_keep,'target_score'].isna().all()
 pred=pd.read_csv(out/'character_predictions.csv')
 assert pred.groupby('character_key').policy.nunique().eq(3).all()
 assert pred.loc[~pred.output_available,'prediction_available_seconds'].isna().all()
 pd.DataFrame(rows).to_csv(out/'model_training_only_audit.csv',index=False)
 if args.reference:
  ref=pd.read_csv(args.reference/'model_summary.csv')
  ref_pred=pd.read_csv(args.reference/'character_predictions.csv',keep_default_na=False)
  now_pred=pd.read_csv(out/'character_predictions.csv',keep_default_na=False)
  consistency=[]
  for r in ref.itertuples():
   filename=f'{r.model_id}_{r.policy}.npz'
   with np.load(out/'models'/filename) as a,np.load(args.reference/'models'/filename) as b:
    for key in ['weights_uv','intercept','train_mean','train_scale','selected_features']:
     np.testing.assert_allclose(a[key],b[key],rtol=1e-12,atol=1e-12)
   cols=['character_key','predicted_character','correct','output_available','valid_flashes']
   a=now_pred.loc[now_pred.model_id.eq(r.model_id)&now_pred.policy.eq(r.policy),cols].sort_values('character_key').reset_index(drop=True)
   b=ref_pred.loc[ref_pred.model_id.eq(r.model_id)&ref_pred.policy.eq(r.policy),cols].sort_values('character_key').reset_index(drop=True)
   pd.testing.assert_frame_equal(a,b)
   consistency.append(dict(model_id=r.model_id,policy=r.policy,reference_weights_and_predictions_unchanged=True))
  pd.DataFrame(consistency).to_csv(out/'previous_six_subject_consistency.csv',index=False)
  print(f'PASS: {len(consistency)} reference models and predictions unchanged after adding subjects')
 print(f'PASS: {len(rows)} train-only models, original saved scores, final feature significance, missing rejected scores and matched validation characters')

if __name__=='__main__':main()
