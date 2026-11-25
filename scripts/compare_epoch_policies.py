"""Compare fixed signal-fault, PTP and blink-overlap strategies on existing EDF outputs."""
from pathlib import Path
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('_MNE_FAKE_HOME_DIR',str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bigp3bci_policy_comparison import evaluate_policies,summarize
from bigp3bci_preprocessing import file_hash

OUT=ROOT/'outputs/three_policy_comparison'
PRIMARY=['faults_only','ptp_250uv','blink_overlap']


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    config_path=ROOT/'configs/three_policy_comparison.json';config=json.loads(config_path.read_text())
    source_path=ROOT/config['source_config'];source=json.loads(source_path.read_text());detector_hash=file_hash(source_path)
    frozen=ROOT/'outputs/expanded_pilot/selection_manifest.csv';frozen_hash=file_hash(frozen)
    manifest=pd.read_csv(ROOT/config['source_manifest'])
    supplement=pd.read_csv(ROOT/'outputs/blink_annotation_update/annotation_manifest.csv').set_index('edf_name')
    records=[];frames=[];review_rows=[];provenance=[]
    reviewed=pd.read_csv(ROOT/config['reviewed_intervals']);reviewed=reviewed[reviewed.review_status.eq('complete')]
    manual_root=ROOT/source['manual_validation']['manual_root']/'manual_review_tables'
    sums={};counts={}
    for r in manifest.itertuples():
        folder=ROOT/'outputs/expanded_pilot'/Path(r.edf_name).stem if r.in_frozen_pilot else ROOT/supplement.loc[r.edf_name,'results_path']
        rec=json.loads((folder/'processing_record.json').read_text())
        if rec['config']['blink_detector']!=source['blink_detector'] or rec['config']['filter']!=source['filter']:raise ValueError('Cached detector/filter differs from frozen config')
        e=pd.read_csv(folder/'epoch_candidates.csv',keep_default_na=False)
        q=pd.read_csv(folder/'raw_quality_epoch_links.csv',keep_default_na=False)
        b=pd.read_csv(folder/'blink_epoch_links.csv',keep_default_na=False)
        long=pd.read_csv(folder/'epoch_ptp_long.csv')
        ptp=long.pivot(index='event_index',columns='channel',values='ptp_uv').reindex(index=e.event_index,columns=rec['channels']).to_numpy()
        # Empty string is not a missing physiological number. Explicit numeric
        # fields used here originate from complete saved epoch arrays.
        output=evaluate_policies(e,ptp,q,b,rec['channels'],rec['sfreq'],rec['n_samples'],source['blink_detector'],config)
        output['in_frozen_pilot']=r.in_frozen_pilot
        frames.append(output)
        provenance.append(dict(edf_name=r.edf_name,results_path=str(folder.relative_to(ROOT)),source_record_sha256=file_hash(folder/'processing_record.json'),
                               epoch_table_sha256=file_hash(folder/'epoch_candidates.csv'),blink_links_sha256=file_hash(folder/'blink_epoch_links.csv'),
                               subject_id=r.subject_id,session_id=r.session_id,site=r.site,in_frozen_pilot=r.in_frozen_pilot))
        # Descriptive ERP: pool epochs, no baseline, same channels and x/y scales.
        with np.load(folder/'epochs_all.npz') as data:
            epochs=data['eeg_v']
            for strategy,g in output.groupby('strategy',sort=False):
                for label in ['target','non-target']:
                    mask=(g.would_keep & g.label.eq(label)).to_numpy(copy=True)
                    finite=np.isfinite(epochs).all(axis=(1,2));mask &= finite
                    key=(strategy,label)
                    if key not in sums:sums[key]=np.zeros(epochs.shape[1:]);counts[key]=0
                    sums[key]+=epochs[mask].sum(axis=0);counts[key]+=int(mask.sum())
        review=reviewed[reviewed.edf_name.eq(r.edf_name)]
        if len(review):
            m=pd.read_csv(manual_root/(Path(r.edf_name).stem+'_manual_review.csv'))
            blink=m[m.type.isin(['MANUAL_bad_span','MANUAL_blink','MANUAL_possible_blink','MANUAL_suspected_blink'])]
            uncertain=m[m.type.str.contains('uncertain',case=False,na=False)]
            for a in review.itertuples():
                mask=(output.onset_seconds>=a.start_sec)&(output.epoch_stop_sample_exclusive/rec['sfreq']<=a.end_sec)&~output.common_fault
                subset=output[mask].copy()
                subset['overlaps_manual_blink']=[bool(((blink.start_sec<row.epoch_stop_sample_exclusive/rec['sfreq'])&(blink.end_sec>row.onset_seconds)).any()) for row in subset.itertuples()]
                subset['uncertain']=[bool(((uncertain.start_sec<row.epoch_stop_sample_exclusive/rec['sfreq'])&(uncertain.end_sec>row.onset_seconds)).any()) for row in subset.itertuples()]
                subset=subset[~subset.uncertain]
                review_rows.append(subset.assign(review_start_sec=a.start_sec,review_end_sec=a.end_sec))
        print(r.edf_name, 'policy rows',len(output),flush=True)
    result=pd.concat(frames,ignore_index=True)
    result.to_csv(OUT/'all_epoch_policy_decisions.csv',index=False)
    pd.DataFrame(provenance).to_csv(OUT/'input_manifest.csv',index=False)
    for name,keys in [('overall',[]),('cohort',['in_frozen_pilot']),('edf',['edf_name']),('subject',['subject_id']),
                      ('session',['subject_id','session_id']),('label',['edf_name','label']),('character',['edf_name','character_trial_id']),
                      ('stimulus_group',['edf_name','character_trial_id','stimulus_group_id'])]:
        summarize(result,keys).to_csv(OUT/f'{name}_summary.csv',index=False)
    triggers=[];reasons=[]
    for row in result.itertuples():
        for ch in filter(None,row.trigger_channels.split(';')):triggers.append(dict(strategy=row.strategy,edf_name=row.edf_name,channel=ch))
        for reason in filter(None,row.rejection_reasons.split(';')):reasons.append(dict(strategy=row.strategy,edf_name=row.edf_name,reason=reason))
    for name,rows,key in [('trigger_channels',triggers,'channel'),('rejection_reasons',reasons,'reason')]:
        pd.DataFrame(rows,columns=['strategy','edf_name',key]).groupby(['strategy','edf_name',key]).size().rename('count').reset_index().to_csv(OUT/(name+'.csv'),index=False)
    joined=result.pivot(index=['edf_name','event_index'],columns='strategy',values='would_keep')
    comparisons=[]
    for strategy in joined.columns:
        if not (joined[strategy]<=joined.faults_only).all():raise AssertionError('Additional strategy kept a base-fault epoch')
    primary_ptp=~joined.ptp_250uv & joined.faults_only
    blink=~joined.blink_overlap & joined.faults_only
    for name,mask in [('both',primary_ptp&blink),('ptp_only',primary_ptp&~blink),('blink_only',~primary_ptp&blink),('neither',~primary_ptp&~blink&joined.faults_only)]:
        comparisons.append(dict(category=name,epochs=int(mask.sum())))
    pd.DataFrame(comparisons).to_csv(OUT/'additional_rejection_overlap.csv',index=False)
    if review_rows:
        review_result=pd.concat(review_rows,ignore_index=True)
        review_result.to_csv(OUT/'reviewed_epoch_policy_decisions.csv',index=False)
        review_result.groupby(['edf_name','strategy','overlaps_manual_blink']).agg(total=('would_keep','size'),kept=('would_keep','sum'),rejected=('would_reject','sum')).reset_index().to_csv(OUT/'reviewed_epoch_summary.csv',index=False)
    channels=rec['channels'];time=np.arange(rec['epoch_samples'])/rec['sfreq']*1000
    erp_meta=[];erp_data={}
    for key,total in sums.items():
        strategy,label=key;erp_data[strategy+'_'+label]=total/max(counts[key],1)
        erp_meta.append(dict(strategy=strategy,label=label,n_epochs=counts[key]))
    np.savez_compressed(OUT/'pooled_erp.npz',**erp_data,channels=channels,time_ms=time)
    pd.DataFrame(erp_meta).to_csv(OUT/'pooled_erp_counts.csv',index=False)
    fig,axes=plt.subplots(2,4,figsize=(16,7),sharex=True,sharey=True,layout='constrained')
    for row,label in enumerate(['target','non-target']):
        for col,ch in enumerate(['EEG_Fz','EEG_Cz','EEG_Pz','EEG_Oz']):
            ax=axes[row,col]
            for strategy in PRIMARY:
                ax.plot(time,erp_data[strategy+'_'+label][channels.index(ch)]*1e6,label=f'{strategy} n={counts[(strategy,label)]}')
            ax.set(title=f'{ch}: {label}',xlabel='ms',ylabel='µV');ax.legend(fontsize=6)
    fig.savefig(OUT/'pooled_erp.png',dpi=120);plt.close(fig)
    summary=summarize(result,['edf_name'])
    order=manifest.edf_name.tolist()
    fig,ax=plt.subplots(figsize=(14,5),layout='constrained')
    for strategy in result.strategy.unique():
        y=summary[summary.strategy.eq(strategy)].set_index('edf_name').reindex(order).expected_retention*100
        ax.plot(range(1,len(order)+1),y,marker='o',label=strategy)
    ax.set(xticks=range(1,len(order)+1),xlabel='input manifest index',ylabel='expected retention (%)',ylim=(0,102));ax.legend()
    fig.savefig(OUT/'retention_by_file.png',dpi=120);plt.close(fig)
    assert file_hash(source_path)==detector_hash and file_hash(frozen)==frozen_hash
    (OUT/'config_snapshot.json').write_text(json.dumps(config,indent=2),encoding='utf8')
    (OUT/'detector_config_snapshot.json').write_text(json.dumps(source['blink_detector'],indent=2),encoding='utf8')
    record=dict(files=len(manifest),total_events=int(result[result.strategy.eq('faults_only')].shape[0]),
                detector_parameters_unchanged=True,detector_config_sha256=detector_hash,original_manifest_unchanged=True,
                base_faults=config['hard_faults'],jump_flat_candidates='report only; not confirmed faults',baseline=None,
                ptp_primary_upper_uv=config['ptp_primary_upper_uv'],simulation_only=True,actual_epochs_removed=0,
                timing='blink policy adds uniform conservative confirmation horizon; pending_at_eof is separate from retrospective would_keep',
                reviewed_definition='full epoch contained in explicitly reviewed interval, common faults excluded; no manual labels enter strategy',
                limitation='supplementary annotation-selected files are not held-out validation; retention/ERP changes are not classifier performance')
    (OUT/'comparison_record.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    print(summarize(result,[]).to_string(index=False),flush=True)

if __name__=='__main__':main()
