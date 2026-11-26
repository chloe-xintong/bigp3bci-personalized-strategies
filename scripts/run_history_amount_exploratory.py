"""Frozen exploratory SWLDA enumeration; resumable, Train-only, no retuning."""
from pathlib import Path
import os
for _name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[_name]='1'
import argparse
import hashlib
import io
import json
import sys
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bigp3bci_swlda import fit_swlda,score_swlda,auc_rank,decode_character
from bigp3bci_calibration_budget import make_plan
from bigp3bci_history_amount import enumerate_subsets,summarize_history


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def freeze_json(value,path):
    if path.exists() and json.loads(path.read_text())!=value:raise ValueError('Frozen inputs changed: '+str(path))
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def freeze_csv(frame,path):
    text=frame.to_csv(index=False,lineterminator='\n')
    if path.exists():
        actual=pd.read_csv(path,keep_default_na=False)
        expected=pd.read_csv(io.StringIO(text),keep_default_na=False)
        if not actual.equals(expected):raise ValueError('Frozen manifest changed: '+str(path))
        return
    path.write_text(text,encoding='utf-8',newline='\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/history_amount_exploratory.json')
    parser.add_argument('--output',default='outputs/history_amount_exploratory_v1')
    args=parser.parse_args();cfg=json.loads((ROOT/args.config).read_text());out=ROOT/args.output
    source_budget=ROOT/cfg['source_budget_experiment'];base=json.loads((source_budget/'config_snapshot.json').read_text())
    if base['validation_source']!='SE003/Train/last6_characters' or base['policy']!=cfg['policy']:
        raise ValueError('Expected the same Train-only validation and quality policy as the main experiment')
    source=ROOT/base['source_experiment'];source_cfg=json.loads((source/'config_snapshot.json').read_text())
    trials=pd.read_csv(source/'character_split.csv');subjects=sorted(trials.subject_id.unique())
    if len(subjects)!=10:raise ValueError('Expected all 10 subjects')
    preprocessing=pd.read_csv(ROOT/source_cfg['input_manifest'])
    for r in preprocessing.itertuples():
        record_path=ROOT/r.results_path/'processing_record.json'
        rec=json.loads(record_path.read_text())
        if Path(rec['source_edf']).parts[-3]!='Train':raise ValueError('No Test sources permitted')
        if digest(record_path)!=r.source_record_sha256:raise ValueError('Cached Train processing changed')
    paths=[source_budget/'config_snapshot.json',source_budget/'fixed_validation_characters.csv',
           source_budget/'subject_results.csv',source/'character_split.csv',source/'epoch_scores.csv',source/'epoch_split.csv',
           source/'features_and_membership.npz',ROOT/'src/bigp3bci_swlda.py',ROOT/'src/bigp3bci_history_amount.py',
           ROOT/'docs/history_amount_exploratory_protocol.md']
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}
    out.mkdir(parents=True,exist_ok=True)
    frozen=dict(protocol=cfg,features=base['features'],swlda=base['swlda'],character_evidence=base['character_evidence'])
    freeze_json(frozen,out/'config_snapshot.json');freeze_json(hashes,out/'source_hashes.json')
    cols=['edf_name','event_index','feature_row','character_key','subject_id','session_id','split','policy',
          'would_keep','decision_data_available','label','onset_sample','epoch_stop_sample_exclusive']
    epochs=pd.read_csv(source/'epoch_scores.csv',usecols=cols,low_memory=False)
    epochs=epochs[epochs.policy.eq(cfg['policy'])].sort_values('feature_row')
    if not epochs.split.eq('Train').all() or not epochs.feature_row.is_unique:raise ValueError('Invalid Train feature mapping')
    with np.load(source/'features_and_membership.npz') as d:x=d['features_uv'];flashed=d['flashed']
    if len(epochs)!=len(x) or x.shape[1]!=256 or len(base['features']['channels'])!=16:raise ValueError('Feature/channel mismatch')
    ids=pd.read_csv(source/'epoch_split.csv')
    matched=epochs.merge(ids[['edf_name','event_index','feature_row','character_key']],on=['edf_name','event_index'],suffixes=('','_frozen'),validate='one_to_one')
    if not matched.feature_row.eq(matched.feature_row_frozen).all() or not matched.character_key.eq(matched.character_key_frozen).all():raise ValueError('Cached feature identity changed')
    plans={s:make_plan(trials[trials.subject_id.eq(s)],cfg['history_sessions'],cfg['current_session']) for s in subjects}
    pool=[];jobs=[];roles=[];eegnet=[]
    for subject,plan in plans.items():
        if len(plan['calibration'])!=12 or len(plan['validation'])!=6:raise ValueError('Wrong current partition')
        history=trials[trials.subject_id.eq(subject)&trials.session_id.isin(cfg['history_sessions'])]
        files=history[['session_id','edf_name']].drop_duplicates().sort_values(['session_id','edf_name']).reset_index(drop=True)
        if len(files)!=6 or files.session_id.value_counts().to_dict()!={'SE001':3,'SE002':3}:raise ValueError('Wrong historical inventory')
        for slot,r in enumerate(files.itertuples()):
            count=int(history.edf_name.eq(r.edf_name).sum())
            if count!=6:raise ValueError('Historical files must contain six whole characters')
            pool.append(dict(subject_id=subject,slot=slot,session_id=r.session_id,edf_name=r.edf_name,characters=count))
        for role in ['history','calibration','validation']:
            for order,key in enumerate(plan[role]):roles.append(dict(subject_id=subject,role=role,order=order,character_key=key))
        for h,slots in enumerate_subsets():
            subset=''.join('1' if i in slots else '0' for i in range(6))
            names=files.iloc[list(slots)].edf_name.tolist() if slots else []
            for k in cfg['k_values']:
                jobs.append(dict(subject_id=subject,H=h,k=k,subset_id=subset,history_files=';'.join(names),
                    history_slots=';'.join(map(str,slots)),n_s1=sum(i<3 for i in slots),n_s2=sum(i>=3 for i in slots),
                    history_characters=6*h,current_characters=k,defined=bool(h or k)))
        for h in cfg['history_file_counts']:
            for seed in cfg['eegnet_plan']['subset_seeds']:
                slots=sorted(np.random.default_rng(seed).choice(6,h,replace=False).tolist())
                eegnet.append(dict(subject_id=subject,H=h,seed=seed,subset_id=''.join('1' if i in slots else '0' for i in range(6)),
                    history_slots=';'.join(map(str,slots)),n_s1=sum(i<3 for i in slots),n_s2=sum(i>=3 for i in slots),
                    k_values=';'.join(map(str,cfg['k_values'])),status='plan_only_no_training'))
    pool=pd.DataFrame(pool);jobs=pd.DataFrame(jobs)
    freeze_csv(pool,out/'history_file_pool.csv');freeze_csv(jobs,out/'enumeration_manifest.csv')
    freeze_csv(pd.DataFrame(roles),out/'character_roles.csv');freeze_csv(pd.DataFrame(eegnet),out/'eegnet_subset_plan.csv')
    keys=set(pd.DataFrame(roles).query("role == 'validation'").character_key)
    if keys!=set(pd.read_csv(source_budget/'fixed_validation_characters.csv').character_key):raise ValueError('Validation differs from main experiment')
    val_manifest=epochs[epochs.character_key.isin(keys)].copy()
    freeze_csv(val_manifest,out/'validation_flash_manifest.csv')
    if len(val_manifest)!=7560 or int(val_manifest.would_keep.sum())!=7340:raise ValueError('Quality mask differs from main experiment')
    truth=trials.set_index('character_key').true_character_index.to_dict();result_rows=[]
    for subject in subjects:
        subject_epochs=epochs[epochs.subject_id.eq(subject)]
        val=subject_epochs[subject_epochs.character_key.isin(plans[subject]['validation'])].reset_index(drop=True)
        val_ids=val.feature_row.to_numpy(int);val_labels=val.label.map({'non-target':0,'target':1}).to_numpy(int)
        val_mask=val.would_keep.to_numpy(bool)&np.isfinite(x[val_ids]).all(axis=1)
        folder=out/'jobs'/subject;folder.mkdir(parents=True,exist_ok=True)
        local_jobs=jobs[jobs.subject_id.eq(subject)]
        for number,job in enumerate(local_jobs.itertuples(),1):
            stem=f'H{job.H}_k{job.k:02d}_{job.subset_id}';marker=folder/(stem+'.json');npz=folder/(stem+'.npz')
            if marker.exists():
                record=json.loads(marker.read_text())
                if record['model_available'] and (not npz.exists() or digest(npz)!=record['npz_sha256']):raise ValueError('Checkpoint changed')
                result_rows.append(record);continue
            names=job.history_files.split(';') if job.history_files else []
            chosen=subject_epochs.edf_name.isin(names)|subject_epochs.character_key.isin(plans[subject]['calibration'][:job.k])
            candidate=subject_epochs[chosen]
            if candidate.character_key.nunique()!=6*job.H+job.k:raise ValueError('Wrong character budget')
            if set(candidate.character_key)&keys:raise ValueError('Validation in fitting')
            train=candidate[candidate.would_keep];train_ids=train.feature_row.to_numpy(int)
            labels=train.label.map({'non-target':0,'target':1}).to_numpy(int)
            scores=np.full(len(val),np.nan);model=None;start=time.perf_counter()
            if job.defined:model=fit_swlda(x[train_ids],labels,base['swlda']);scores[val_mask]=score_swlda(model,x[val_ids[val_mask]])
            predictions=[]
            if model is not None:
                for key in plans[subject]['validation']:
                    g=val[val.character_key.eq(key)];idx=g.index.to_numpy(int)
                    d=decode_character(scores[idx],flashed[g.feature_row.to_numpy(int)],base['character_evidence'])
                    predicted=d['predicted_index'] if g.decision_data_available.all() else None
                    predictions.append(dict(character_key=key,true_index=int(truth[key]),predicted_index=predicted,
                                             correct=bool(predicted is not None and predicted==truth[key])))
                temporary=folder/(stem+'.tmp.npz')
                np.savez_compressed(temporary,weights=model['weights'],intercept=model['intercept'],train_mean=model['train_mean'],
                    train_scale=model['train_scale'],selected_features=model['selected_features'],training_feature_rows=train_ids,
                    validation_feature_rows=val_ids,validation_scores=scores)
                temporary.replace(npz)
            record=dict(subject_id=subject,H=int(job.H),k=int(job.k),subset_id=job.subset_id,history_files=job.history_files,
                history_slots=job.history_slots,n_s1=int(job.n_s1),n_s2=int(job.n_s2),history_characters=int(6*job.H),
                current_characters=int(job.k),model_available=model is not None,fit_status=model['status'] if model else 'undefined_H0_k0',
                training_characters=int(candidate.character_key.nunique()),retained_training_flashes=len(train_ids),
                validation_characters=6,validation_flashes=len(val),scored_validation_flashes=int(val_mask.sum()) if model else 0,
                flash_roc_auc=auc_rank(val_labels,scores) if model else None,
                character_accuracy=float(np.mean([p['correct'] for p in predictions])) if model else None,
                fit_seconds=time.perf_counter()-start,selected_features=len(model['selected_features']) if model else 0,
                training_character_keys=sorted(candidate.character_key.unique().tolist()),validation_in_fit=False,
                predictions=predictions,npz_sha256=digest(npz) if model else None)
            temporary=marker.with_suffix('.tmp.json');temporary.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8');temporary.replace(marker)
            result_rows.append(record)
            if number%32==0:print(subject,f'{number}/256 subsets/budgets completed',flush=True)
        print(subject,'complete; all 256 combinations/budgets retained',flush=True)
    raw=pd.DataFrame([{key:value for key,value in row.items() if key not in ['predictions','training_character_keys']} for row in result_rows])
    raw.to_csv(out/'subset_results.csv',index=False)
    prediction_rows=[dict(subject_id=r['subject_id'],H=r['H'],k=r['k'],subset_id=r['subset_id'],**p) for r in result_rows for p in r['predictions']]
    pd.DataFrame(prediction_rows).to_csv(out/'character_predictions.csv',index=False)
    per_subject,macro,sessions,session_macro=summarize_history(raw,subjects,cfg['k_values'],cfg['bootstrap']['replicates'],cfg['bootstrap']['seed'])
    for frame,name in [(per_subject,'subject_summary'),(macro,'macro_summary'),(sessions,'session_subject_contrasts'),(session_macro,'session_macro_contrasts')]:frame.to_csv(out/(name+'.csv'),index=False)
    macro[['H','k','history_characters','mean_auc','ci_low','ci_high','current_only_auc','delta_vs_current','delta_ci_low','delta_ci_high']].to_csv(out/'paired_history_vs_current.csv',index=False)
    main=pd.read_csv(source_budget/'subject_results.csv');checks=[]
    for r in raw.itertuples():
        scheme=None
        if r.H==0 and r.k>0:scheme='current_only'
        elif r.H==6:scheme='history_plus_current'
        elif r.H==3 and r.n_s1==3:scheme='history_se001_plus_current'
        elif r.H==3 and r.n_s2==3:scheme='history_se002_plus_current'
        if scheme:
            ref=main[main.subject_id.eq(r.subject_id)&main.scheme.eq(scheme)&main.k.eq(r.k)].iloc[0]
            np.testing.assert_allclose(r.flash_roc_auc,ref.flash_roc_auc,atol=1e-12)
            np.testing.assert_allclose(r.character_accuracy,ref.accuracy,atol=1e-12)
            checks.append(dict(subject_id=r.subject_id,H=r.H,k=r.k,scheme=scheme,main_result_reproduced=True))
    pd.DataFrame(checks).to_csv(out/'main_endpoint_checks.csv',index=False)
    from plot_history_amount import plot_all
    plot_all(out)
    assert all(digest(ROOT/path)==value for path,value in hashes.items())
    freeze_json(dict(total_jobs=len(raw),fitted_models=int(raw.model_available.sum()),subjects=subjects,
        enumeration_complete=True,main_endpoint_checks=len(checks),validation_flashes=7560,retained_validation_flashes=7340,
        bootstrap_replicates=10000,bootstrap_unit='subject',used_test=False,eegnet_trained=False,
        main_source_files_unchanged=True),out/'completion.json')
    print(macro.to_string(index=False));print('DONE',out,flush=True)


if __name__=='__main__':main()
