"""Descriptive mean epoch covariance/session amplitude check; no alignment or Test."""
from pathlib import Path
import os
for _name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[_name]='1'
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
import sys;sys.path.insert(0,str(ROOT/'src'))
from bigp3bci_data_policy import require_train_edf,require_train_record


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def spd_log(matrix):
    values,vectors=np.linalg.eigh((matrix+matrix.T)/2)
    if not np.isfinite(values).all() or values.min()<=0:raise ValueError('Covariance must be positive definite')
    return (vectors*np.log(values))@vectors.T


def log_distance(a,b):
    value=float(np.linalg.norm(spd_log(a)-spd_log(b),'fro'))
    return value,value/np.sqrt(a.shape[0])


def amplitude_plot(frame,out,metric,ylabel,iqr=False):
    people=sorted(frame.subject_id.unique());sessions=['SE001','SE002','SE003']
    colors=['#1f77b4','#ff7f0e','#2ca02c'];fig,axes=plt.subplots(2,5,figsize=(20,8),layout='constrained',sharey=True)
    channels=frame.channel.drop_duplicates().tolist();x=np.arange(len(channels))
    for subject,ax in zip(people,axes.flat):
        for session,color in zip(sessions,colors):
            g=frame[frame.subject_id.eq(subject)&frame.session_id.eq(session)].set_index('channel').reindex(channels)
            ax.plot(x,g[metric],'-o',ms=2.5,lw=1.2,label=session,color=color)
            if iqr:ax.fill_between(x,g[metric+'_q25'],g[metric+'_q75'],color=color,alpha=.10)
        ax.set(title=subject,xticks=x,xticklabels=[c.removeprefix('EEG_') for c in channels]);ax.tick_params(axis='x',rotation=65,labelsize=7)
        ax.set_ylabel(ylabel,fontsize=8);ax.grid(axis='y',alpha=.2)
        if metric=='signed_mean_uv':ax.axhline(0,color='gray',ls=':',lw=.8)
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='outside lower center',ncol=3)
    fig.suptitle('Train session amplitude: '+ylabel+(' | shading: epoch IQR, not CI' if iqr else ''),fontsize=14)
    fig.savefig(out/(metric+'.png'),dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='outputs/session_drift_descriptive_v1')
    args=parser.parse_args();out=ROOT/args.output
    if out.exists():raise FileExistsError('Choose a new output directory; preserve diagnostic results')
    cfg=json.loads((ROOT/'configs/session_drift_descriptive.json').read_text())
    base=ROOT/cfg['source_budget_experiment'];settings=json.loads((base/'config_snapshot.json').read_text())
    source=ROOT/settings['source_experiment'];source_cfg=json.loads((source/'config_snapshot.json').read_text())
    manifest=pd.read_csv(ROOT/source_cfg['input_manifest']);channels=settings['features']['channels']
    if len(channels)!=16:raise ValueError('Expected the predefined 16 channels')
    roles=pd.read_csv(base/'character_roles.csv')
    allowed=set(roles[roles.role.isin(['history','calibration'])].character_key)
    validation=set(roles[roles.role.eq('validation')].character_key)
    epochs=pd.read_csv(source/'epoch_scores.csv',usecols=['policy','character_key','edf_name','event_index','subject_id','session_id','would_keep','split'])
    epochs=epochs[epochs.policy.eq(cfg['policy'])&epochs.character_key.isin(allowed)&epochs.would_keep]
    if not epochs.split.eq('Train').all() or set(epochs.character_key)&validation:raise ValueError('Validation/Test in diagnostic pool')
    if not epochs[epochs.session_id.eq('SE003')].character_key.isin(set(roles[roles.role.eq('calibration')].character_key)).all():raise ValueError('S3 outside calibration pool')
    pool=epochs.groupby(['subject_id','session_id']).character_key.nunique()
    if not pool[pool.index.get_level_values('session_id').isin(['SE001','SE002'])].eq(18).all() or not pool[pool.index.get_level_values('session_id')=='SE003'].eq(12).all():
        raise ValueError('Unexpected session training-character counts')
    out.mkdir(parents=True)
    (out/'config_snapshot.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    protected={str(p.relative_to(ROOT)):digest(p) for p in [base/'subject_results.csv',base/'overall_results.csv',base/'character_predictions.csv',base/'config_snapshot.json']}
    sums={};amplitudes={};counts={};provenance=[]
    for item in manifest.itertuples():
        selected=epochs[epochs.edf_name.eq(item.edf_name)].sort_values('event_index')
        if selected.empty:continue
        folder=ROOT/item.results_path;record_path=folder/'processing_record.json';record=json.loads(record_path.read_text())
        require_train_record(record)
        if digest(record_path)!=item.source_record_sha256:raise ValueError('Changed input')
        if record['channels']!=channels:raise ValueError('Channel order differs')
        with np.load(folder/'epochs_all.npz') as data:
            if data['channels'].tolist()!=channels:raise ValueError('Epoch channels differ')
            indices=pd.Index(data['event_index']).get_indexer(selected.event_index)
            if (indices<0).any():raise ValueError('Missing epoch indices')
            x=data['eeg_v'][indices]*1e6
        if not np.isfinite(x).all() or x.shape[1]!=16:raise ValueError('Invalid retained epoch')
        centered=x-x.mean(axis=2,keepdims=True)
        covariance=np.einsum('nct,ndt->ncd',centered,centered)/(x.shape[2]-1)
        key=(item.subject_id,item.session_id)
        sums[key]=sums.get(key,np.zeros((16,16)))+covariance.sum(axis=0)
        counts[key]=counts.get(key,0)+len(x)
        amplitudes.setdefault(key,[]).append(dict(signed_mean_uv=x.mean(axis=2),mean_abs_uv=np.abs(x).mean(axis=2),rms_uv=np.sqrt((x*x).mean(axis=2))))
        provenance.append(dict(edf_name=item.edf_name,epochs=len(x),epoch_cache_sha256=digest(folder/'epochs_all.npz'),processing_record_sha256=item.source_record_sha256))
    covariance={};summaries=[];regularization=cfg['regularization']['trace_identity_shrinkage']
    for key,total in sums.items():
        mean=total/counts[key];scale=float(np.trace(mean)/16)
        if scale<=0:raise ValueError('Zero covariance')
        covariance[key]=(1-regularization)*mean+regularization*scale*np.eye(16)
        metrics={name:np.concatenate([r[name] for r in amplitudes[key]]) for name in amplitudes[key][0]}
        for i,channel in enumerate(channels):
            row=dict(subject_id=key[0],session_id=key[1],channel=channel,n_epochs=counts[key],training_characters=int(pool.loc[key]))
            for name,values in metrics.items():
                row[name]=float(values[:,i].mean());row[name+'_q25']=float(np.quantile(values[:,i],.25));row[name+'_q50']=float(np.median(values[:,i]));row[name+'_q75']=float(np.quantile(values[:,i],.75))
            summaries.append(row)
    distances=[]
    for subject in sorted(epochs.subject_id.unique()):
        row=dict(subject_id=subject)
        for a,b,label in [('SE001','SE003','s1_s3'),('SE002','SE003','s2_s3'),('SE001','SE002','s1_s2')]:
            raw,normalized=log_distance(covariance[(subject,a)],covariance[(subject,b)])
            row[label+'_distance']=raw;row[label+'_normalized']=normalized
        row['s2_s3_over_s1_s3']=row['s2_s3_distance']/row['s1_s3_distance'] if row['s1_s3_distance']>0 else None
        row['s2_at_least_20pct_closer']=bool(row['s2_s3_distance']<=cfg['s2_closer_rule']['distance_ratio_max']*row['s1_s3_distance'] and row['s1_s3_distance']>0)
        row['all_pairs_small']=all(row[name+'_normalized']<=cfg['small_distance_threshold'] for name in ['s1_s3','s2_s3','s1_s2'])
        distances.append(row)
    distances=pd.DataFrame(distances);amplitude=pd.DataFrame(summaries)
    all_small=bool(distances.all_pairs_small.all());closer=int(distances.s2_at_least_20pct_closer.sum())
    conclusion=('所有被试的三个 session 对均低于预设实用小差异阈值，未发现明显的 session 漂移；本次不加对齐。' if all_small else
        ('达到预设描述性判据：S2 比 S1 更接近 S3；存在 session 协方差差异，EA 仅作为待决定的后续对照。' if closer>=cfg['s2_closer_rule']['minimum_subjects'] else
         '存在超出预设小差异阈值的 session 协方差差异，但未达到“S2 明显更接近 S3”的预设判据；暂不作统一来源解释，本次不加对齐。'))
    distances.to_csv(out/'session_distances.csv',index=False,encoding='utf-8-sig');amplitude.to_csv(out/'channel_amplitude_summary.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(out/'mean_covariances.npz',**{s+'_'+session:matrix for (s,session),matrix in covariance.items()},channels=channels)
    amplitude_plot(amplitude,out,'mean_abs_uv','Mean absolute amplitude (uV)',iqr=True)
    amplitude_plot(amplitude,out,'signed_mean_uv','Signed temporal mean (uV)')
    for path,value in protected.items():
        if digest(ROOT/path)!=value:raise ValueError('Main results changed')
    result=dict(conclusion=conclusion,s2_closer_subjects=closer,subjects=int(len(distances)),all_pairs_small=all_small,
        threshold=cfg['small_distance_threshold'],ea_run=False,used_test=False,used_current_validation=False,
        main_results_unchanged=True,training_epochs={s+'_'+session:n for (s,session),n in counts.items()},
        caveats=['Operational thresholds are descriptive, not universally validated.',
                 'Covariance mixes spatial structure and amplitude scale; blink/artifact prevalence and label mixture can also differ across sessions.',
                 'S1/S2 use 18 characters each; S3 calibration uses 12. Overlapping epochs are not independent observations.',
                 'Signed mean reflects offset; absolute mean/RMS reflect magnitude. IQR is a distribution summary, not a confidence interval.'],
        provenance=provenance,protected_main_sha256=protected)
    (out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(distances[['subject_id','s1_s3_normalized','s2_s3_normalized','s1_s2_normalized','s2_s3_over_s1_s3']].to_string(index=False))
    print(conclusion)


if __name__=='__main__':main()
