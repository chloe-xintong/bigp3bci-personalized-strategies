"""Exploratory session-wise Euclidean alignment (EA) sensitivity check with SWLDA.

Train validation only; never reads Test. Reference covariances use only data the
session may use: all SE001/SE002 Train characters, and the first k SE003
calibration characters. Unaligned conditions must reproduce the frozen history
amount results exactly; the main design and reported results are not changed.
"""
from pathlib import Path
import os
for _name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[_name]='1'
import argparse
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from bigp3bci_swlda import fit_swlda,score_swlda,auc_rank,decode_character,window_features
from bigp3bci_data_policy import require_train_edf,require_train_record
from bigp3bci_euclidean_alignment import (trial_covariances,shrink,inverse_sqrt,align_epochs,align_features,
                                          log_distance,bootstrap_deltas)

CONDITIONS=[('A',True,False),('B',True,True),('current_only',False,False),('C',False,True)]


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/ea_sensitivity.json')
    parser.add_argument('--output',default='outputs/ea_sensitivity_v1')
    args=parser.parse_args();cfg=json.loads((ROOT/args.config).read_text());out=ROOT/args.output
    if (out/'completion.json').exists():raise FileExistsError('Completed output exists; choose a new directory')
    history_out=ROOT/cfg['source_history_experiment']
    frozen=json.loads((history_out/'config_snapshot.json').read_text())
    hcfg=frozen['protocol'];features_cfg=frozen['features'];swlda_cfg=frozen['swlda'];evidence_cfg=frozen['character_evidence']
    if hcfg['policy']!=cfg['policy'] or not json.loads((history_out/'completion.json').read_text())['enumeration_complete']:
        raise ValueError('History-amount source incomplete or uses a different policy')
    budget=ROOT/hcfg['source_budget_experiment'];base=json.loads((budget/'config_snapshot.json').read_text())
    source=ROOT/base['source_experiment'];source_cfg=json.loads((source/'config_snapshot.json').read_text())
    channels=features_cfg['channels'];n_ch=len(channels);n_win=int(features_cfg['windows_per_channel'])
    shrinkage=float(cfg['regularization']['trace_identity_shrinkage'])
    protected=[history_out/'subset_results.csv',history_out/'macro_summary.csv',budget/'subject_results.csv',
               source/'features_and_membership.npz',source/'epoch_scores.csv']
    protected_hashes={str(p.relative_to(ROOT)):digest(p) for p in protected}
    out.mkdir(parents=True,exist_ok=True)
    (out/'config_snapshot.json').write_text(json.dumps(dict(protocol=cfg,features=features_cfg,swlda=swlda_cfg,
        character_evidence=evidence_cfg),indent=2),encoding='utf-8')
    (out/'source_hashes.json').write_text(json.dumps(protected_hashes,indent=2),encoding='utf-8')

    cols=['edf_name','event_index','feature_row','character_key','subject_id','session_id','split','policy',
          'would_keep','decision_data_available','label']
    epochs=pd.read_csv(source/'epoch_scores.csv',usecols=cols,low_memory=False)
    epochs=epochs[epochs.policy.eq(cfg['policy'])].sort_values('feature_row').reset_index(drop=True)
    if not epochs.split.eq('Train').all():raise ValueError('No Test flashes permitted')
    with np.load(source/'features_and_membership.npz') as d:x=d['features_uv'];flashed=d['flashed']
    if x.shape!=(len(epochs),n_ch*n_win):raise ValueError('Feature layout mismatch')
    roles=pd.read_csv(history_out/'character_roles.csv').sort_values(['subject_id','role','order'])
    trials=pd.read_csv(source/'character_split.csv');truth=trials.set_index('character_key').true_character_index.to_dict()
    manifest=pd.read_csv(ROOT/source_cfg['input_manifest'])
    val_manifest=pd.read_csv(history_out/'validation_flash_manifest.csv')
    subjects=sorted(roles.subject_id.unique())
    if len(subjects)!=10:raise ValueError('Expected 10 subjects')
    results=[];diagnostics=[];equivalence=[];reference_rows=[]
    for subject in subjects:
        r=roles[roles.subject_id.eq(subject)]
        history=r[r.role.eq('history')].character_key.tolist()
        calibration=r[r.role.eq('calibration')].sort_values('order').character_key.tolist()
        validation=r[r.role.eq('validation')].sort_values('order').character_key.tolist()
        if len(history)!=36 or len(calibration)!=12 or len(validation)!=6:raise ValueError('Wrong character roles')
        sub=epochs[epochs.subject_id.eq(subject)]
        usable=sub.would_keep.to_numpy(bool)&np.isfinite(x[sub.feature_row.to_numpy(int)]).all(axis=1)
        sub=sub.assign(usable=usable)
        # Load raw filtered epochs (V) for this subject and map them to feature rows.
        raw={}
        for item in manifest[manifest.subject_id.eq(subject)].itertuples():
            folder=ROOT/item.results_path;record_path=folder/'processing_record.json'
            record=json.loads(record_path.read_text())
            require_train_record(record)
            if digest(record_path)!=item.source_record_sha256:raise ValueError('Changed input')
            rows=sub[sub.edf_name.eq(item.edf_name)].sort_values('event_index')
            with np.load(folder/'epochs_all.npz') as data:
                if data['channels'].tolist()!=channels:raise ValueError('Epoch channel order differs')
                if not np.array_equal(data['event_index'],rows.event_index.to_numpy()):raise ValueError('Epoch/feature identity mismatch')
                eeg=data['eeg_v'];fs=float(data['sfreq'])
            ids=rows.feature_row.to_numpy(int)
            recomputed,_=window_features(eeg,fs,features_cfg);finite=np.isfinite(recomputed).all(axis=1)
            if not np.allclose(recomputed[finite],x[ids][finite],rtol=0,atol=1e-9):raise ValueError('Cached features differ from raw epochs')
            raw[item.edf_name]=(eeg,ids,fs)
        def epochs_for(keys):
            chosen=sub[sub.character_key.isin(keys)&sub.usable]
            blocks=[]
            for edf,g in chosen.groupby('edf_name'):
                eeg,ids,_=raw[edf];pos=pd.Index(ids).get_indexer(g.feature_row)
                if (pos<0).any():raise ValueError('Missing raw epoch')
                blocks.append(eeg[pos]*1e6)
            return np.concatenate(blocks),chosen
        def reference(keys):
            data,chosen=epochs_for(keys)
            if set(chosen.character_key)&set(validation):raise ValueError('Validation characters in a reference covariance')
            mean=trial_covariances(data).mean(axis=0);ref=shrink(mean,shrinkage)
            return ref,inverse_sqrt(ref),len(data),chosen.character_key.nunique()
        refs={}
        for session in cfg['history_sessions']:
            keys=sub[sub.session_id.eq(session)].character_key.unique().tolist()
            if not set(keys)<=set(history) or len(keys)!=18:raise ValueError('Historical reference must use all 18 Train characters')
            refs[session]=reference(keys)
        for k in cfg['k_values']:refs[('SE003',k)]=reference(calibration[:k])
        for key,(ref,w,n,chars) in refs.items():
            session,k=(key,None) if isinstance(key,str) else key
            eig=np.linalg.eigvalsh(ref)
            reference_rows.append(dict(subject_id=subject,session_id=session,k=k,reference_flashes=n,reference_characters=chars,
                min_eigenvalue_uv2=float(eig.min()),condition_number=float(eig.max()/eig.min())))
        # Exactness check: features of aligned raw epochs == aligned cached features.
        edf=sorted(raw)[0];eeg,ids,fs=raw[edf];w=refs['SE001'][1]
        direct,_=window_features(align_epochs(eeg*1e6,w)/1e6,fs,features_cfg);finite=np.isfinite(direct).all(axis=1)
        error=float(np.max(np.abs(direct[finite]-align_features(x[ids][finite],w,n_ch,n_win))))
        if error>1e-8:raise ValueError('Feature-space EA is not equivalent to epoch-space EA')
        equivalence.append(dict(subject_id=subject,edf_name=edf,max_abs_error_uv=error))
        val=sub[sub.character_key.isin(validation)].reset_index(drop=True)
        expected=val_manifest[val_manifest.subject_id.eq(subject)].sort_values('feature_row')
        if not np.array_equal(val.feature_row.to_numpy(),expected.feature_row.to_numpy()):raise ValueError('Validation flashes differ')
        val_ids=val.feature_row.to_numpy(int);val_mask=val.usable.to_numpy(bool)
        val_labels=val.label.map({'non-target':0,'target':1}).to_numpy(int)
        for k in cfg['k_values']:
            w3=refs[('SE003',k)][1]
            r12=refs[('SE003',12)][0];rk=refs[('SE003',k)][0]
            raw_d,norm_d=log_distance(rk,r12)
            d1=log_distance(refs['SE001'][0],rk)[1];d2=log_distance(refs['SE002'][0],rk)[1]
            diagnostics.append(dict(subject_id=subject,k=k,s3_reference_flashes=refs[('SE003',k)][2],
                s3k_to_s3_12_normalized=norm_d,s1_to_s3k_normalized=d1,s2_to_s3k_normalized=d2))
            for condition,with_history,aligned in CONDITIONS:
                keys=(history if with_history else [])+calibration[:k]
                train=sub[sub.character_key.isin(keys)&sub.usable]
                if set(train.character_key)&set(validation) or train.character_key.nunique()!=len(keys):raise ValueError('Wrong training characters')
                train_ids=train.feature_row.to_numpy(int);features=x[train_ids].copy()
                val_features=x[val_ids[val_mask]].copy()
                if aligned:
                    for session in train.session_id.unique():
                        m=(train.session_id==session).to_numpy()
                        w=w3 if session==cfg['current_session'] else refs[session][1]
                        features[m]=align_features(features[m],w,n_ch,n_win)
                    val_features=align_features(val_features,w3,n_ch,n_win)
                labels=train.label.map({'non-target':0,'target':1}).to_numpy(int)
                start=time.perf_counter();model=fit_swlda(features,labels,swlda_cfg)
                scores=np.full(len(val),np.nan);scores[val_mask]=score_swlda(model,val_features)
                correct=[]
                for key in validation:
                    g=val[val.character_key.eq(key)];idx=g.index.to_numpy(int)
                    dec=decode_character(scores[idx],flashed[g.feature_row.to_numpy(int)],evidence_cfg)
                    predicted=dec['predicted_index'] if g.decision_data_available.all() else None
                    correct.append(predicted is not None and predicted==truth[key])
                results.append(dict(subject_id=subject,k=k,condition=condition,with_history=with_history,aligned=aligned,
                    training_characters=len(keys),training_flashes=len(train_ids),scored_validation_flashes=int(val_mask.sum()),
                    selected_features=len(model['selected_features']),fit_status=model['status'],
                    flash_roc_auc=auc_rank(val_labels,scores),character_accuracy=float(np.mean(correct)),
                    fit_seconds=time.perf_counter()-start))
        print(subject,'done',flush=True)
    res=pd.DataFrame(results);res.to_csv(out/'condition_results.csv',index=False)
    pd.DataFrame(reference_rows).to_csv(out/'reference_covariances.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'reference_stability.csv',index=False)
    pd.DataFrame(equivalence).to_csv(out/'feature_ea_equivalence.csv',index=False)
    # Unaligned conditions must reproduce the frozen history-amount results.
    subset=pd.read_csv(history_out/'subset_results.csv',dtype={'subset_id':str});checks=[]
    for condition,h in [('A',6),('current_only',0)]:
        for r in res[res.condition.eq(condition)].itertuples():
            ref=subset[subset.subject_id.eq(r.subject_id)&subset.H.eq(h)&subset.k.eq(r.k)].iloc[0]
            np.testing.assert_allclose(r.flash_roc_auc,ref.flash_roc_auc,atol=1e-12)
            np.testing.assert_allclose(r.character_accuracy,ref.character_accuracy,atol=1e-12)
            checks.append(dict(subject_id=r.subject_id,k=r.k,condition=condition,history_amount_H=h,reproduced=True))
    pd.DataFrame(checks).to_csv(out/'unaligned_reproduction_checks.csv',index=False)
    wide=res.pivot_table(index=['subject_id','k'],columns='condition',values='flash_roc_auc').reset_index()
    wide['ea_with_history']=wide.B-wide.A;wide['ea_current_only']=wide.C-wide.current_only
    wide['highlight']=wide.subject_id.isin(cfg['highlight_subjects'])
    wide.columns.name=None;wide.to_csv(out/'subject_deltas.csv',index=False)
    rules=cfg['decision_rules'];boot=cfg['bootstrap']
    macro=pd.DataFrame(bootstrap_deltas(wide,'ea_with_history',subjects,cfg['k_values'],boot['replicates'],boot['seed'],rules['direction_minimum_same_sign'])
                       +bootstrap_deltas(wide,'ea_current_only',subjects,cfg['k_values'],boot['replicates'],boot['seed'],rules['direction_minimum_same_sign']))
    means=res.groupby(['k','condition']).flash_roc_auc.mean().unstack()
    macro['mean_auc_aligned']=[means.loc[r.k,'B' if r.comparison=='ea_with_history' else 'C'] for r in macro.itertuples()]
    macro['mean_auc_unaligned']=[means.loc[r.k,'A' if r.comparison=='ea_with_history' else 'current_only'] for r in macro.itertuples()]
    macro.to_csv(out/'macro_deltas.csv',index=False)
    b=macro[macro.comparison.eq('ea_with_history')]
    clear=b[b.ci_excludes_zero&b.direction.eq('consistent')&(b.mean_delta.abs()>=rules['clear_change_min_abs_auc'])]
    from plot_ea_sensitivity import plot_deltas
    plot_deltas(out,cfg)
    for path,value in protected_hashes.items():
        if digest(ROOT/path)!=value:raise ValueError('Protected main/history output changed: '+path)
    (out/'completion.json').write_text(json.dumps(dict(subjects=subjects,k_values=cfg['k_values'],fits=len(res),
        unaligned_reproduced=len(checks),eegnet_followup_triggered=bool(len(clear)),clear_change_k=clear.k.tolist(),
        directions={f'{r.comparison}_k{r.k}':r.direction for r in macro.itertuples()},
        used_test=False,used_s3_validation_for_reference=False,main_results_unchanged=True,eegnet_trained=False),indent=2),encoding='utf-8')
    print(macro.to_string(index=False));print('DONE',out,flush=True)


if __name__=='__main__':main()
