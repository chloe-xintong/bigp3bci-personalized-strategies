"""Frozen supplementary cross-subject SWLDA; reads existing Train caches only."""
from pathlib import Path
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import sys,json,hashlib,subprocess,argparse
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bigp3bci_data_policy import require_train_record
from bigp3bci_swlda import fit_swlda,score_swlda,auc_rank,decode_character
from analysis.cross_subject_split import make_splits,condition_keys,assert_control_equal,bootstrap_mean


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['controls','cross'],required=True)
    args=parser.parse_args();cfg=json.loads((ROOT/'configs/cross_subject_swlda_exploratory.json').read_text())
    out=ROOT/cfg['output'];source=ROOT/cfg['source_experiment'];control=ROOT/cfg['control_experiment']
    source_cfg=json.loads((source/'config_snapshot.json').read_text());reference_cfg=json.loads((control/'config_snapshot.json').read_text())
    for key in ['swlda','features','character_evidence']:
        if source_cfg[key]!=reference_cfg[key]:raise ValueError('Source and control settings differ')
    if any(source_cfg['swlda'][k]!=v for k,v in cfg['swlda'].items()):raise ValueError('Frozen SWLDA mismatch')
    if json.loads((source/'implementation_record.json').read_text())['used_final_test_data']:raise ValueError('Non-Train source')
    manifest=pd.read_csv(ROOT/source_cfg['input_manifest'])
    for row in manifest.itertuples():
        record=ROOT/row.results_path/'processing_record.json';require_train_record(json.loads(record.read_text()))
        if sha(record)!=row.source_record_sha256:raise ValueError('Changed Train processing record')
    inputs=[source/name for name in ['config_snapshot.json','character_split.csv','epoch_scores.csv','epoch_split.csv','features_and_membership.npz']]
    inputs += [control/name for name in ['config_snapshot.json','subject_results.csv','flash_predictions.csv']]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    trials=pd.read_csv(source/'character_split.csv');splits=make_splits(trials,cfg)
    epochs=pd.read_csv(source/'epoch_scores.csv',low_memory=False);epochs=epochs[epochs.policy.eq(cfg['policy'])].sort_values('feature_row')
    if not epochs.split.eq('Train').all() or not epochs.feature_row.is_unique:raise ValueError('Invalid Train epochs')
    with np.load(source/'features_and_membership.npz') as z:x=z['features_uv'];flashed=z['flashed']
    if x.shape!=(len(epochs),cfg['n_features']) or not np.array_equal(epochs.feature_row,np.arange(len(x))):raise ValueError('Feature row mismatch')
    ids=pd.read_csv(source/'epoch_split.csv')
    check=epochs.merge(ids[['edf_name','event_index','feature_row','character_key']],on=['edf_name','event_index'],validate='one_to_one',suffixes=('','_original'))
    if not check.feature_row.eq(check.feature_row_original).all() or not check.character_key.eq(check.character_key_original).all():raise ValueError('Changed feature membership')
    # Per-EDF clocks only; no calibration interval/policy can reach validation.
    for s,split in splits.items():
        v=epochs[epochs.character_key.isin(split['validation'])];c=epochs[epochs.character_key.isin(split['calibration'])]
        if len(v)-int(v.would_keep.sum())!=22:raise ValueError('Unexpected validation exclusions')
        for edf,g in v.groupby('edf_name'):
            t=c[c.edf_name.eq(edf)]
            if len(t):
                fs=float(trials[trials.edf_name.eq(edf)].sampling_hz.iloc[0])
                if t.epoch_stop_sample_exclusive.max()>g.onset_sample.min() or t.policy_decision_seconds.max()>=g.onset_sample.min()/fs:raise ValueError('Calibration consumes validation samples')
    resolved=dict(cfg,features=source_cfg['features'],swlda=source_cfg['swlda'],character_evidence=source_cfg['character_evidence'],input_sha256=hashes)
    if args.stage=='controls':
        if out.exists():raise FileExistsError('Preserve existing analysis; choose a new preregistered version')
        out.mkdir(parents=True)
        (out/'config_snapshot.json').write_text(json.dumps(resolved,indent=2),encoding='utf-8')
        (out/'commit.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()+'\n')
        rows=[]
        indexed=trials.set_index('character_key')
        for target,split in splits.items():
            for role in ['donors','calibration','validation']:
                for order,key in enumerate(split[role]):
                    r=indexed.loc[key];rows.append(dict(analysis=cfg['analysis'],target_subject=target,role=role,order=order,character_key=key,source_subject=r.subject_id,session_id=r.session_id,edf_name=r.edf_name,split='Train'))
        pd.DataFrame(rows).to_csv(out/'split_manifest.csv',index=False)
    else:
        if not (out/'controls_verified.json').exists():raise ValueError('Verify C before fitting X')
        if (out/'completion.json').exists():raise FileExistsError('Already complete')
        if json.loads((out/'config_snapshot.json').read_text())!=resolved:raise ValueError('Config/inputs changed between stages')
    expected=pd.read_csv(control/'subject_results.csv');expected_flash=pd.read_csv(control/'flash_predictions.csv') if args.stage=='controls' else None
    results=[];predictions=[];checks=[]
    # Complete all 30 C fits and their comparisons before any of the 40 X fits.
    condition='C' if args.stage=='controls' else 'X'
    for s,split in splits.items():
        val=epochs[epochs.character_key.isin(split['validation'])].sort_values('feature_row');vid=val.feature_row.to_numpy(int)
        for k in cfg['conditions'][condition]:
            keys=condition_keys(split,condition,k);train=epochs[epochs.character_key.isin(keys)&epochs.would_keep].sort_values('feature_row')
            y=train.label.map({'non-target':0,'target':1}).to_numpy(float)
            model=fit_swlda(x[train.feature_row.to_numpy(int)],y,source_cfg['swlda'])
            scores=np.full(len(val),np.nan);valid=val.would_keep.to_numpy(bool)&np.isfinite(x[vid]).all(axis=1)
            scores[valid]=score_swlda(model,x[vid[valid]])
            labels=val.label.map({'non-target':0,'target':1}).to_numpy(float);auc=auc_rank(labels,scores)
            correct=0
            for key in split['validation']:
                mask=val.character_key.eq(key).to_numpy();g=val[mask]
                decoded=decode_character(scores[mask],flashed[vid[mask]],source_cfg['character_evidence'])
                predicted=decoded['predicted_index'] if g.decision_data_available.all() else None
                correct+=int(predicted is not None and predicted==int(trials[trials.character_key.eq(key)].true_character_index.iloc[0]))
            accuracy=correct/6
            if condition=='C':
                ref=expected[expected.subject_id.eq(s)&expected.scheme.eq('current_only')&expected.k.eq(k)]
                if len(ref)!=1:raise ValueError('Missing control reference')
                assert_control_equal([auc,accuracy],ref[['flash_roc_auc','accuracy']].iloc[0].to_numpy(),cfg['control_atol'])
                refscore=expected_flash[expected_flash.subject_id.eq(s)&expected_flash.scheme.eq('current_only')&expected_flash.k.eq(k)]
                scorecheck=val[['edf_name','event_index']].merge(refscore[['edf_name','event_index','target_score']],on=['edf_name','event_index'],validate='one_to_one',how='left')
                assert_control_equal(scores,scorecheck.target_score.to_numpy(),cfg['control_atol'])
                checks.append(dict(subject_id=s,k=k,auc_difference=auc-float(ref.flash_roc_auc.iloc[0]),accuracy_difference=accuracy-float(ref.accuracy.iloc[0]),max_flash_score_difference=float(np.nanmax(np.abs(scores-scorecheck.target_score))),passed=True))
            results.append(dict(analysis=cfg['analysis'],subject_id=s,condition=condition,condition_label=f'{condition}{k}',k=k,flash_roc_auc=auc,accuracy=accuracy,correct_characters=correct,validation_characters=6,validation_flashes=len(val),scored_validation_flashes=int(np.isfinite(scores).sum()),training_characters=len(keys),donor_characters=324 if condition=='X' else 0,retained_training_epochs=len(train),selected_features=len(model['selected_features']),fit_status=model['status']))
            for event,score,label in zip(val.itertuples(),scores,labels):predictions.append(dict(subject_id=s,condition=condition,k=k,edf_name=event.edf_name,event_index=event.event_index,label=int(label),score=score))
            print(s,f'{condition}{k}',f'AUC={auc:.6f}',f'characters={correct}/6',flush=True)
    frame=pd.DataFrame(results)
    if args.stage=='controls':
        frame.to_csv(out/'control_results.csv',index=False);pd.DataFrame(checks).to_csv(out/'control_checks.csv',index=False)
        pd.DataFrame(predictions).to_csv(out/'control_flash_scores.csv',index=False)
        (out/'controls_verified.json').write_text(json.dumps(dict(models=len(frame),all_passed=True,atol=cfg['control_atol'])))
        return
    frame=pd.concat([pd.read_csv(out/'control_results.csv'),frame],ignore_index=True)
    frame.to_csv(out/'subject_results.csv',index=False);pd.DataFrame(predictions).to_csv(out/'cross_flash_scores.csv',index=False)
    indices=np.random.default_rng(cfg['bootstrap']['seed']).integers(0,10,size=(cfg['bootstrap']['replicates'],10))
    summaries=[];paired=[]
    for (cond,k),g in frame.groupby(['condition','k']):
        g=g.set_index('subject_id').loc[cfg['subjects']]
        mean,low,high=bootstrap_mean(g.flash_roc_auc.to_numpy(),indices)
        summaries.append(dict(analysis=cfg['analysis'],condition=cond,k=k,macro_flash_roc_auc=mean,ci_low=low,ci_high=high,macro_character_accuracy=float(g.accuracy.mean()),subjects=10))
    for k in cfg['conditions']['C']:
        g=frame[frame.k.eq(k)].pivot(index='subject_id',columns='condition',values='flash_roc_auc').loc[cfg['subjects']]
        delta=g.X-g.C;mean,low,high=bootstrap_mean(delta.to_numpy(),indices)
        for s in cfg['subjects']:paired.append(dict(analysis=cfg['analysis'],subject_id=s,k=k,x_auc=g.loc[s,'X'],c_auc=g.loc[s,'C'],auc_difference=delta.loc[s],macro_difference=mean,ci_low=low,ci_high=high,bootstrap_replicates=10000))
    pd.DataFrame(summaries).to_csv(out/'macro_results.csv',index=False);pd.DataFrame(paired).to_csv(out/'paired_differences.csv',index=False)
    if any(sha(ROOT/p)!=h for p,h in hashes.items()):raise ValueError('Existing source/control inputs changed')
    for cond in ['C','X']:
        g=frame[frame.condition.eq(cond)&frame.k.eq(3)]
        if (int(g.validation_flashes.sum()),int(g.scored_validation_flashes.sum()))!=(7560,7340):raise ValueError('Unexpected flash totals')
    from plot_cross_subject_swlda import plot
    plot(out)
    (out/'completion.json').write_text(json.dumps(dict(models=70,subjects=10,control_models_verified=30,test_file_reads=0,raw_edf_reads=0,main_results_unchanged=True,analysis=cfg['analysis']),indent=2))


if __name__=='__main__':main()
