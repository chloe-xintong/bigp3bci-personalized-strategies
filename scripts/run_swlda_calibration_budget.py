"""SWLDA budget curves; last six SE003 Train characters are fixed validation."""
from pathlib import Path
import os
for _thread_var in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[_thread_var]='1'
import argparse
import json
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('_MNE_FAKE_HOME_DIR',str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
from bigp3bci_swlda import fit_swlda,score_swlda,decode_character,auc_rank
from bigp3bci_calibration_budget import make_plan,training_keys
from bigp3bci_preprocessing import file_hash
from bigp3bci_budget_plot import plot_budget_results,plot_budget_small_multiples


def model_jobs(cfg,plan):
    jobs=[(scheme,k) for scheme in cfg['curves'] for k in cfg['k_values']]
    jobs += [(scheme,0 if scheme.startswith('history_') else len(plan['calibration'])) for scheme in cfg['endpoints']]
    return jobs


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/swlda_calibration_budget.json')
    parser.add_argument('--output',default='outputs/swlda_calibration_budget_train_validation_v1')
    parser.add_argument('--subjects',nargs='+',help='Optional explicit subset; use a separate output directory')
    parser.add_argument('--k-values',nargs='+',type=int,help='Optional smoke-check budgets; endpoints still use full allowed k')
    args=parser.parse_args()
    cfg=json.loads((ROOT/args.config).read_text())
    if cfg['validation_source']!='SE003/Train/last6_characters' or cfg['policy']!='A':
        raise ValueError('Only frozen policy A and last-six Train validation are allowed')
    if args.k_values is not None:cfg['k_values']=sorted(set(args.k_values))
    cfg['subject_subset']=args.subjects
    out=ROOT/args.output
    if out.exists():raise FileExistsError('Choose a new output directory to preserve frozen results')
    source=ROOT/cfg['source_experiment']
    source_cfg=json.loads((source/'config_snapshot.json').read_text())
    implementation=json.loads((source/'implementation_record.json').read_text())
    if implementation['used_final_test_data']:raise ValueError('Source experiment is not Train-only')
    manifest=pd.read_csv(ROOT/source_cfg['input_manifest'])
    for r in manifest.itertuples():
        rec=json.loads((ROOT/r.results_path/'processing_record.json').read_text())
        if Path(rec['source_edf']).parts[-3]!='Train':raise ValueError('No Test sources permitted')
        if file_hash(ROOT/r.results_path/'processing_record.json')!=r.source_record_sha256:
            raise ValueError('Cached preprocessing record changed')
    trials=pd.read_csv(source/'character_split.csv')
    cols=['edf_name','event_index','feature_row','character_key','subject_id','session_id','split','policy',
          'would_keep','decision_data_available','label','onset_sample','epoch_stop_sample_exclusive','policy_decision_seconds']
    epochs=pd.read_csv(source/'epoch_scores.csv',usecols=cols,low_memory=False)
    epochs=epochs[epochs.policy.eq(cfg['policy'])].copy()
    if not epochs.split.eq('Train').all() or not epochs.feature_row.is_unique:
        raise ValueError('Expected unique Train epochs for the predefined policy')
    with np.load(source/'features_and_membership.npz') as cache:
        x=cache['features_uv'];flashed=cache['flashed'];characters=cache['characters'].tolist()
    if x.shape[1]!=256 or len(source_cfg['features']['channels'])!=16:
        raise ValueError('Expected the fixed 16-channel, 256-feature representation')
    if len(x)!=len(epochs) or set(epochs.feature_row)!=set(range(len(x))):
        raise ValueError('Feature rows differ from the saved epoch inventory')
    epoch_ids=pd.read_csv(source/'epoch_split.csv')
    check=epochs.merge(epoch_ids[['edf_name','event_index','feature_row','character_key']],
                       on=['edf_name','event_index'],suffixes=('','_frozen'),validate='one_to_one')
    if not check.feature_row.eq(check.feature_row_frozen).all() or not check.character_key.eq(check.character_key_frozen).all():
        raise ValueError('Feature/character provenance mismatch')
    subjects=sorted(args.subjects or trials.subject_id.unique().tolist())
    if not set(subjects)<=set(trials.subject_id):raise ValueError('Unknown subject')
    out.mkdir(parents=True);(out/'models').mkdir()
    resolved=dict(**cfg,features=source_cfg['features'],swlda=source_cfg['swlda'],character_evidence=source_cfg['character_evidence'])
    (out/'config_snapshot.json').write_text(json.dumps(resolved,indent=2),encoding='utf-8')
    provenance={name:file_hash(source/name) for name in ['config_snapshot.json','character_split.csv','epoch_scores.csv','epoch_split.csv','features_and_membership.npz','input_provenance.csv']}
    plans={s:make_plan(trials[trials.subject_id.eq(s)],cfg['history_sessions'],cfg['current_session']) for s in subjects}
    for plan in plans.values():
        if len(plan['calibration'])!=12 or len(plan['validation'])!=6:raise ValueError('Expected 12 calibration and 6 validation characters')
        for k in cfg['k_values']:training_keys(plan,'current_only',k)
    (out/'provenance.json').write_text(json.dumps(dict(source_files=provenance,
        implementation_sha256=file_hash(ROOT/'src/bigp3bci_swlda.py'),
        budget_implementation_sha256=file_hash(ROOT/'src/bigp3bci_calibration_budget.py'),
        channels_selected_from_data=False,used_test=False),indent=2),encoding='utf-8')
    # Freeze budget membership before fitting; validation labels never determine it.
    roles=[];membership=[];validation=[]
    for subject,plan in plans.items():
        for role in ['history','calibration','validation']:
            for rank,key in enumerate(plan[role]):roles.append(dict(subject_id=subject,role=role,order=rank,character_key=key))
        jobs=model_jobs(cfg,plan)
        for scheme,k in jobs:
            for key in training_keys(plan,scheme,k):membership.append(dict(subject_id=subject,scheme=scheme,k=k,character_key=key))
        validation.extend(dict(subject_id=subject,character_key=key) for key in plan['validation'])
    pd.DataFrame(roles).to_csv(out/'character_roles.csv',index=False)
    pd.DataFrame(membership).to_csv(out/'training_character_membership.csv',index=False)
    pd.DataFrame(validation).to_csv(out/'fixed_validation_characters.csv',index=False)
    predictions=[];summaries=[];endpoints=[];fit_audit=[];flash_predictions=[]
    for subject,plan in plans.items():
        e=epochs[epochs.subject_id.eq(subject)].sort_values('feature_row')
        val=e[e.character_key.isin(plan['validation'])]
        train_pool=e[e.character_key.isin(plan['calibration'])]
        # Sample overlap and policy lookahead checks are within EDF clocks only.
        for edf,v in val.groupby('edf_name'):
            c=train_pool[train_pool.edf_name.eq(edf)]
            if len(c):
                if c.epoch_stop_sample_exclusive.max()>v.onset_sample.min():raise ValueError('Shared calibration/validation samples')
                fs=float(trials[trials.edf_name.eq(edf)].sampling_hz.iloc[0])
                if c.policy_decision_seconds.max()>=v.onset_sample.min()/fs:raise ValueError('Training policy consumes validation samples')
        jobs=model_jobs(cfg,plan)
        fitted={}
        for scheme,k in jobs:
            keys=training_keys(plan,scheme,k)
            train=e[e.character_key.isin(keys)&e.would_keep]
            if not train.split.eq('Train').all():raise ValueError('Test must never enter fitting')
            ids=train.feature_row.to_numpy(int)
            y=train.label.map({'non-target':0,'target':1}).to_numpy(float)
            if len(ids) and (not np.isfinite(x[ids]).all() or not np.isfinite(y).all()):raise ValueError('Nonfinite retained training input')
            if set(train.character_key)&set(plan['validation']):raise ValueError('Held-out outcomes in fitting')
            start=time.perf_counter();model=None
            if not keys:status='unavailable_no_current_calibration'
            else:
                try:model=fit_swlda(x[ids],y,source_cfg['swlda']);status=model['status']
                except ValueError as error:status='unavailable: '+str(error)
            fit_seconds=time.perf_counter()-start
            scores=np.full(len(val),np.nan);valid=val.would_keep.to_numpy(bool)&np.isfinite(x[val.feature_row.to_numpy(int)]).all(axis=1)
            if model is not None:scores[valid]=score_swlda(model,x[val.loc[valid,'feature_row'].to_numpy(int)])
            v=val.copy();v['score']=scores
            val_labels=val.label.map({'non-target':0,'target':1}).to_numpy(float)
            flash_auc=auc_rank(val_labels,scores) if model is not None else np.nan
            for event,score,label in zip(val.itertuples(),scores,val_labels):
                flash_predictions.append(dict(subject_id=subject,scheme=scheme,k=k,edf_name=event.edf_name,
                    event_index=event.event_index,character_key=event.character_key,label=int(label),
                    retained=bool(event.would_keep),scored=bool(np.isfinite(score)),target_score=score))
            per_trial=[]
            for key in plan['validation']:
                g=v[v.character_key.eq(key)]
                trial=trials[trials.character_key.eq(key)].iloc[0]
                result=decode_character(g.score.to_numpy(float),flashed[g.feature_row.to_numpy(int)],source_cfg['character_evidence'])
                predicted=result['predicted_index'] if g.decision_data_available.all() else None
                per_trial.append(dict(subject_id=subject,scheme=scheme,k=k,character_key=key,
                    true_character_index=int(trial.true_character_index),predicted_character_index=predicted,
                    correct=bool(predicted is not None and predicted==trial.true_character_index),
                    output_available=predicted is not None,fit_status=status,n_scored=result['n_scored']))
            predictions.extend(per_trial)
            correct=sum(r['correct'] for r in per_trial)
            summaries.append(dict(subject_id=subject,scheme=scheme,k=k,model_available=model is not None,
                fit_status=status,history_characters=sum(key in set(plan['history']) for key in keys),
                current_calibration_characters=sum(key in set(plan['calibration']) for key in keys),
                training_characters=len(keys),retained_training_epochs=len(ids),training_target_epochs=int(y.sum()),
                validation_characters=len(per_trial),correct_characters=correct,
                accuracy=correct/len(per_trial) if model is not None else np.nan,
                flash_roc_auc=flash_auc,validation_flashes=len(val),scored_validation_flashes=int(np.isfinite(scores).sum()),
                scored_target_flashes=int(((val_labels==1)&np.isfinite(scores)).sum()),
                scored_nontarget_flashes=int(((val_labels==0)&np.isfinite(scores)).sum()),
                no_output_characters=sum(not r['output_available'] for r in per_trial),
                selected_features=len(model['selected_features']) if model else 0,fit_seconds=fit_seconds))
            fit_audit.append(dict(subject_id=subject,scheme=scheme,k=k,validation_in_fit=False,
                selected_history_characters=sum(key in set(plan['history']) for key in keys),
                selected_current_characters=sum(key in set(plan['calibration']) for key in keys),
                current_budget_obeyed=(scheme not in cfg['curves'] or sum(key in set(plan['calibration']) for key in keys)==k),
                no_data_driven_channel_selection=True))
            fitted[(scheme,k)]=(model,[r['predicted_character_index'] for r in per_trial])
            if model:
                prefix=out/'models'/f'{subject}_{scheme}_k{k:02d}'
                np.savez_compressed(str(prefix)+'.npz',weights=model['weights'],intercept=model['intercept'],
                    selected_features=model['selected_features'],train_mean=model['train_mean'],train_scale=model['train_scale'])
                serial={key:value for key,value in model.items() if key not in ['weights','train_mean','train_scale']}
                Path(str(prefix)+'.json').write_text(json.dumps(serial,indent=2),encoding='utf-8')
            print(subject,scheme,'k='+str(k),status,f'{correct}/{len(per_trial)}' if model else 'no model',flush=True)
        checks=[(('history_only',0),('history_plus_current',0)),
                (('history_se001_only',0),('history_se001_plus_current',0)),
                (('history_se002_only',0),('history_se002_plus_current',0)),
                (('current_max_endpoint',len(plan['calibration'])),('current_only',len(plan['calibration']))),
                (('all_available_reference',len(plan['calibration'])),('history_plus_current',len(plan['calibration'])))]
        for endpoint,curve in checks:
            if curve not in fitted:continue
            a,pa=fitted[endpoint];b,pb=fitted[curve]
            if (a is None)!=(b is None) or pa!=pb:raise AssertionError('Endpoint predictions differ')
            if a is not None:
                np.testing.assert_allclose(a['weights'],b['weights'],rtol=1e-10,atol=1e-10)
                np.testing.assert_allclose(a['intercept'],b['intercept'],rtol=1e-10,atol=1e-10)
            endpoints.append(dict(subject_id=subject,endpoint=endpoint[0],curve=curve[0],k=curve[1],independent_refit_agrees=True))
    summary=pd.DataFrame(summaries)
    summary.to_csv(out/'subject_results.csv',index=False)
    pd.DataFrame(predictions).to_csv(out/'character_predictions.csv',index=False)
    pd.DataFrame(flash_predictions).to_csv(out/'flash_predictions.csv',index=False)
    pd.DataFrame(fit_audit).to_csv(out/'budget_leakage_audit.csv',index=False)
    pd.DataFrame(endpoints).to_csv(out/'endpoint_checks.csv',index=False)
    overall=summary.groupby(['scheme','k'],sort=False).agg(subjects=('subject_id','size'),available_models=('model_available','sum'),
        mean_subject_accuracy=('accuracy','mean'),sd_subject_accuracy=('accuracy','std'),
        macro_flash_roc_auc=('flash_roc_auc','mean'),sd_subject_flash_roc_auc=('flash_roc_auc','std'),
        validation_flashes=('validation_flashes','sum'),scored_validation_flashes=('scored_validation_flashes','sum'),
        correct_characters=('correct_characters','sum'),validation_characters=('validation_characters','sum')).reset_index()
    overall['pooled_accuracy']=np.where(overall.available_models.eq(overall.subjects),overall.correct_characters/overall.validation_characters,np.nan)
    overall.to_csv(out/'overall_results.csv',index=False)
    plot_budget_results(overall,out/'calibration_curves.png',cfg['k_values'],metric='macro_flash_roc_auc')
    plot_budget_results(overall,out/'character_accuracy_curves.png',cfg['k_values'],metric='mean_subject_accuracy')
    plot_budget_small_multiples(overall,summary,out/'auc_small_multiples.png',cfg['k_values'])
    plot_budget_small_multiples(overall,summary,out/'character_accuracy_small_multiples.png',cfg['k_values'],metric='mean_subject_accuracy')
    assert all(file_hash(source/name)==digest for name,digest in provenance.items())
    (out/'completion.json').write_text(json.dumps(dict(subjects=subjects,k_values=cfg['k_values'],
        models_fitted=int(summary.model_available.sum()),endpoint_checks=len(endpoints),
        used_test=False,primary_metric='macro_subject_flash_roc_auc',used_gpu=False,source_inputs_unchanged=True,
        current_validation_characters_never_used_for_fitting=True),indent=2),encoding='utf-8')
    print(overall.to_string(index=False))
    if cfg.get('additional_metric')=='macro_subject_class_balanced_flash_log_loss':
        from add_weighted_logloss import run as add_logloss
        add_logloss(out)


if __name__=='__main__':main()
