"""Independent row-budget, AUC and subject-bootstrap checks of the saved analysis."""
from pathlib import Path
import argparse
import hashlib
import json
from math import comb
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def check_session_intervals(raw,subjects,index,out):
    table=pd.read_csv(out/'session_macro_contrasts.csv')
    for row in table.itertuples():
        difference=[]
        for subject in subjects:
            g=raw[raw.subject_id.eq(subject)&raw.H.eq(row.H)&raw.k.eq(row.k)]
            s1=g[g.n_s1.eq(row.H)].flash_roc_auc.mean()
            s2=g[g.n_s2.eq(row.H)].flash_roc_auc.mean()
            difference.append(s2-s1)
        difference=np.asarray(difference)
        lo,hi=np.quantile(difference[index].mean(axis=1),[.025,.975])
        np.testing.assert_allclose([difference.mean(),lo,hi],[row.delta_s2_minus_s1,row.ci_low,row.ci_high],atol=1e-12)
    return len(table)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='outputs/history_amount_exploratory_v1')
    args=parser.parse_args();out=ROOT/args.output
    completed=json.loads((out/'completion.json').read_text())
    assert completed['fitted_models']==2550 and not completed['used_test'] and not completed['eegnet_trained']
    cfg=json.loads((out/'config_snapshot.json').read_text())['protocol']
    budget=ROOT/cfg['source_budget_experiment'];base=json.loads((budget/'config_snapshot.json').read_text());source=ROOT/base['source_experiment']
    raw=pd.read_csv(out/'subset_results.csv',dtype={'subset_id':str});roles=pd.read_csv(out/'character_roles.csv')
    assert len(raw)==2560 and raw.model_available.sum()==2550
    assert raw.groupby(['subject_id','H','k']).size().eq([comb(6,int(h)) for _,h,_ in raw.groupby(['subject_id','H','k']).size().index]).all()
    epochs=pd.read_csv(source/'epoch_scores.csv',usecols=['policy','would_keep','character_key','edf_name','feature_row','label'],low_memory=False)
    epochs=epochs[epochs.policy.eq(cfg['policy'])].sort_values('feature_row')
    per_file=epochs[['edf_name','character_key']].drop_duplicates().groupby('edf_name').character_key.apply(set).to_dict()
    retained_rows=epochs[epochs.would_keep].groupby('character_key').feature_row.apply(lambda x:x.to_numpy(int)).to_dict()
    validation=pd.read_csv(out/'validation_flash_manifest.csv')
    fixed=set(validation.feature_row)
    audit_count=0
    for record_path in sorted((out/'jobs').glob('*/*.json')):
        record=json.loads(record_path.read_text())
        subject=record['subject_id'];k=record['k']
        calibration=roles[roles.subject_id.eq(subject)&roles.role.eq('calibration')].sort_values('order').character_key.tolist()
        allowed=set(calibration[:k])
        for file in filter(None,record['history_files'].split(';')):
            assert '_Train' in file
            allowed.update(per_file[file])
        assert allowed==set(record['training_character_keys'])
        assert len(allowed)==6*record['H']+k
        if not record['model_available']:continue
        expected=np.sort(np.concatenate([retained_rows[key] for key in allowed]))
        path=record_path.with_suffix('.npz')
        with path.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==record['npz_sha256']
        with np.load(path) as saved:
            np.testing.assert_array_equal(saved['training_feature_rows'],expected)
            assert not set(expected)&fixed
            v=validation[validation.subject_id.eq(subject)].sort_values('feature_row')
            np.testing.assert_array_equal(saved['validation_feature_rows'],v.feature_row.to_numpy(int))
            score=saved['validation_scores'];y=v.label.map({'non-target':0,'target':1}).to_numpy(int)
            good=np.isfinite(score);assert good.sum()==734
            a=score[good&(y==1)];b=score[good&(y==0)]
            independent_auc=float(((a[:,None]>b).sum()+.5*(a[:,None]==b).sum())/(len(a)*len(b)))
            np.testing.assert_allclose(independent_auc,record['flash_roc_auc'],atol=1e-12)
        audit_count+=1
    subjects=sorted(raw.subject_id.unique());index=np.random.default_rng(cfg['bootstrap']['seed']).integers(0,10,(10000,10))
    means=raw.groupby(['subject_id','H','k']).flash_roc_auc.mean()
    macro=pd.read_csv(out/'macro_summary.csv')
    for row in macro.itertuples():
        if row.H==0 and row.k==0:
            assert pd.isna(row.mean_auc);continue
        x=np.array([means.loc[(s,row.H,row.k)] for s in subjects])
        lo,hi=np.quantile(x[index].mean(axis=1),[.025,.975])
        np.testing.assert_allclose([x.mean(),lo,hi],[row.mean_auc,row.ci_low,row.ci_high],atol=1e-12)
        if row.k>0:
            current=np.array([means.loc[(s,0,row.k)] for s in subjects]);difference=x-current
            dlo,dhi=np.quantile(difference[index].mean(axis=1),[.025,.975])
            np.testing.assert_allclose([difference.mean(),dlo,dhi],[row.delta_vs_current,row.delta_ci_low,row.delta_ci_high],atol=1e-12)
        else:assert pd.isna(row.delta_vs_current)
    session_checks=check_session_intervals(raw,subjects,index,out)
    seed_plan=pd.read_csv(out/'eegnet_subset_plan.csv')
    assert len(seed_plan)==350 and seed_plan.status.eq('plan_only_no_training').all()
    for path,value in json.loads((out/'source_hashes.json').read_text()).items():
        with (ROOT/path).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==value
    result=dict(models_checked=audit_count,all_training_row_budgets_verified=True,all_auc_values_independently_verified=True,
        all_macro_and_paired_current_bootstrap_intervals_verified=True,no_test_inputs=True,
        session_contrasts_verified=session_checks,frozen_source_hashes_unchanged=True,eegnet_manifest_only=True)
    (out/'independent_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
