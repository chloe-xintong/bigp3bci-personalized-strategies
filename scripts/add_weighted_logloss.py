"""Append balanced flash log-loss using training-only grouped probability calibration."""
from pathlib import Path
import os
for _name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[_name]='1'
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bigp3bci_probability import fit_grouped_training_platt,predict_platt,balanced_log_loss
from bigp3bci_budget_plot import plot_budget_results,plot_budget_small_multiples


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def run(experiment):
    base=Path(experiment);out=base/'weighted_logloss'
    if out.exists():raise FileExistsError('Weighted log-loss already exists; preserve it and use a new experiment directory')
    cfg=json.loads((base/'config_snapshot.json').read_text())
    completion=json.loads((base/'completion.json').read_text())
    if completion.get('used_test',True):raise ValueError('Expected completed Train-only validation')
    settings=dict(folds=3,l2=.001,probability_clip=1e-7,
        scoring='0.5 * mean target negative log p + 0.5 * mean nontarget negative log (1-p)',
        weighting='Equal class contribution on the fixed validation set, not an assumed 1:5 ratio',
        calibration='Unweighted Platt sigmoid on training-only character-grouped out-of-fold SWLDA scores; natural training prior',
        single_character_fallback='Training in-sample scores only, flagged; never use validation data',
        note='ROC-AUC and character predictions remain the original SWLDA outputs; probability calibration is only for log-loss')
    preserved={name:digest(base/name) for name in ['overall_results.csv','subject_results.csv','flash_predictions.csv','character_predictions.csv']}
    source=ROOT/cfg['source_experiment']
    epochs=pd.read_csv(source/'epoch_scores.csv',usecols=['policy','character_key','feature_row','would_keep','label','split'],low_memory=False)
    epochs=epochs[epochs.policy.eq(cfg['policy'])&epochs.would_keep].sort_values('feature_row')
    if not epochs.split.eq('Train').all():raise ValueError('Test data forbidden')
    membership=pd.read_csv(base/'training_character_membership.csv')
    validation=set(pd.read_csv(base/'fixed_validation_characters.csv').character_key)
    if set(membership.character_key)&validation:raise ValueError('Validation leaked into training membership')
    with np.load(source/'features_and_membership.npz') as data:x=data['features_uv']
    model_rows=pd.read_csv(base/'subject_results.csv')
    flashes=pd.read_csv(base/'flash_predictions.csv')
    flashes['target_probability']=np.nan
    out.mkdir();(out/'calibrators').mkdir()
    (out/'config_snapshot.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
    cached={};metrics=[];calibrator_records=[]
    for number,row in enumerate(model_rows.itertuples(),1):
        mask=flashes.subject_id.eq(row.subject_id)&flashes.scheme.eq(row.scheme)&flashes.k.eq(row.k)
        valid=mask&flashes.scored
        if not row.model_available:
            metrics.append(dict(subject_id=row.subject_id,scheme=row.scheme,k=row.k,weighted_log_loss=np.nan,
                                probability_calibration='unavailable_no_model',scored_flashes=0))
            continue
        keys=membership[(membership.subject_id.eq(row.subject_id))&membership.scheme.eq(row.scheme)&membership.k.eq(row.k)].character_key
        if set(keys)&validation:raise ValueError('Validation characters cannot fit probabilities')
        training=epochs[epochs.character_key.isin(keys)]
        ids=training.feature_row.to_numpy(int);labels=training.label.map({'non-target':0,'target':1}).to_numpy(int)
        cache_key=tuple(ids.tolist())
        if cache_key not in cached:
            with np.load(base/'models'/f'{row.subject_id}_{row.scheme}_k{row.k:02d}.npz') as data:
                final=dict(weights=data['weights'],intercept=float(data['intercept']))
            cal=fit_grouped_training_platt(x[ids],labels,training.character_key.to_numpy(),cfg['swlda'],final,
                                         n_folds=settings['folds'],l2=settings['l2'])
            cached[cache_key]=cal
        cal=cached[cache_key]
        probability=predict_platt(cal,flashes.loc[valid,'target_score'].to_numpy(float))
        flashes.loc[valid,'target_probability']=probability
        loss=balanced_log_loss(flashes.loc[valid,'label'].to_numpy(int),probability,eps=settings['probability_clip'])
        metrics.append(dict(subject_id=row.subject_id,scheme=row.scheme,k=row.k,weighted_log_loss=loss,
                            probability_calibration=cal['method'],scored_flashes=int(valid.sum())))
        record=dict(subject_id=row.subject_id,scheme=row.scheme,k=row.k,**cal)
        calibrator_records.append(record)
        (out/'calibrators'/f'{row.subject_id}_{row.scheme}_k{row.k:02d}.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        if number%25==0:print(f'Probability calibration {number}/{len(model_rows)}',flush=True)
    subjects=pd.DataFrame(metrics)
    aggregate=subjects.groupby(['scheme','k'],sort=False).agg(macro_weighted_log_loss=('weighted_log_loss','mean'),
        sd_subject_weighted_log_loss=('weighted_log_loss','std'),available_calibrators=('weighted_log_loss','count')).reset_index()
    overall=pd.read_csv(base/'overall_results.csv').merge(aggregate,on=['scheme','k'],validate='one_to_one')
    subjects.to_csv(out/'subject_metrics.csv',index=False)
    overall.to_csv(out/'overall_metrics.csv',index=False)
    flashes.to_csv(out/'flash_probabilities.csv',index=False)
    plot_budget_results(overall,out/'weighted_logloss_curves.png',cfg['k_values'],metric='macro_weighted_log_loss')
    full_subjects=model_rows.merge(subjects,on=['subject_id','scheme','k'],validate='one_to_one')
    plot_budget_small_multiples(overall,full_subjects,out/'logloss_small_multiples.png',cfg['k_values'],metric='macro_weighted_log_loss')
    plot_budget_small_multiples(overall,full_subjects,out/'logloss_small_multiples_zoom.png',cfg['k_values'],metric='macro_weighted_log_loss',zoom=True)
    assert all(digest(base/name)==value for name,value in preserved.items())
    (out/'completion.json').write_text(json.dumps(dict(models=len(calibrator_records),unique_training_fits=len(cached),
        base_predictions_unchanged=True,validation_used_to_fit_probabilities=False,
        fallback_models=sum(r['method']=='training_in_sample_single_character' for r in calibrator_records),
        base_sha256=preserved),indent=2),encoding='utf-8')
    print(overall[overall.scheme.isin(['history_only','current_max_endpoint','all_available_reference'])][['scheme','k','macro_weighted_log_loss']].to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',default='outputs/swlda_calibration_budget_train_validation_v1')
    args=parser.parse_args();run(ROOT/args.experiment)
