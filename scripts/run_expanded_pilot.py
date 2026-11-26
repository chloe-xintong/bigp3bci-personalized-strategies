"""Run frozen training-only expanded pilot. Candidate outputs never drop epochs."""
from pathlib import Path
import argparse
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('_MNE_FAKE_HOME_DIR',str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
from bigp3bci_data_policy import require_train_edf,require_train_record
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from bigp3bci_expanded_pilot import process_expanded_file,write_csv


def selection_manifest(root,cfg):
    rows=[]
    for subject,site in cfg['selection']['subjects'].items():
        sessions=sorted(p for p in (root/subject).glob('SE*') if list(p.glob('Train/CBCol/*.edf')))
        if len(sessions)<cfg['selection']['sessions_per_subject']:
            raise ValueError(f'{subject}: insufficient sessions for frozen rule')
        for session in sessions[:cfg['selection']['sessions_per_subject']]:
            path=sorted(session.glob('Train/CBCol/*.edf'))[0]
            rows.append(dict(subject_id=subject,session_id=session.name,site=site,site_status='inferred',edf_name=path.name,
                             relative_path=str(path.relative_to(root)),selection_reason=cfg['selection']['rule']))
    return pd.DataFrame(rows)


def aggregate(results,out):
    frames={name:pd.concat([r[name] for r in results],ignore_index=True) for name in ['events','ptp','simulations','blinks','distributions','quality','persistent']}
    for name,frame in frames.items():write_csv(frame,out/f'all_{name}.csv')
    write_csv([r['audit'] for r in results],out/'event_audit.csv')
    write_csv([r['manual'] for r in results],out/'manual_validation_status.csv')
    sims=frames['simulations']
    for name,keys in [('edf',['edf_name']),('subject',['subject_id']),('session',['subject_id','session_id']),
                      ('label',['edf_name','label']),('character',['edf_name','character_trial_id']),('stimulus_group',['edf_name','character_trial_id','stimulus_group_id'])]:
        summary=sims.groupby(['scheme']+keys,dropna=False).agg(total_flashes=('would_keep','size'),expected_valid_flashes=('would_keep','sum'),
                                                                          expected_retention=('would_keep','mean')).reset_index()
        write_csv(summary,out/f'simulation_{name}_summary.csv')
    trigger_rows=[]
    reason_rows=[]
    for _,r in sims.iterrows():
        for ch in filter(None,r.trigger_channels.split(';')):trigger_rows.append(dict(edf_name=r.edf_name,scheme=r.scheme,channel=ch))
        for reason in filter(None,r.reasons.split(';')):reason_rows.append(dict(edf_name=r.edf_name,scheme=r.scheme,reason=reason))
    for rows,name,key in [(trigger_rows,'trigger_channels','channel'),(reason_rows,'reasons','reason')]:
        counts=pd.DataFrame(rows,columns=['edf_name','scheme',key]).groupby(['edf_name','scheme',key]).size().rename('count').reset_index()
        write_csv(counts,out/f'simulation_{name}.csv')
    # Subject/session distributions: actual epoch values, not averages of per-EDF quantiles.
    for name,keys in [('subject',['subject_id','channel']),('session',['subject_id','session_id','channel'])]:
        q=frames['ptp'].groupby(keys).ptp_uv.quantile([.05,.25,.5,.75,.95,.99]).unstack().reset_index()
        q.columns=keys+['q05','q25','q50','q75','q95','q99']
        write_csv(q,out/f'ptp_{name}_quantiles.csv')
    return frames


def plots(frames,manifest,out):
    figures=out/'figures';figures.mkdir(exist_ok=True)
    ptp=frames['ptp'];edfs=manifest.edf_name.tolist();channels=list(ptp.channel.drop_duplicates())
    fig,axes=plt.subplots(4,4,figsize=(17,13),sharey=True,layout='constrained')
    for ax,ch in zip(axes.flat,channels):
        groups=[ptp.loc[(ptp.channel==ch)&(ptp.edf_name==edf),'ptp_uv'].dropna().to_numpy() for edf in edfs]
        ax.boxplot(groups,showfliers=False)
        ax.set_xticks(range(1,len(edfs)+1),[str(i+1) for i in range(len(edfs))])
        ax.set(title=ch,xlabel='manifest file index',ylabel='filtered epoch PTP (µV)')
    fig.savefig(figures/'ptp_by_file_channel.png',dpi=120);plt.close(fig)
    q=frames['distributions']
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained',sharey=True)
    for ax,metric in zip(axes,['raw_amplitude_uv','filtered_amplitude_uv']):
        for index,edf in enumerate(edfs):
            g=q[(q.edf_name==edf)&(q.metric==metric)].set_index('channel').reindex(channels)
            ax.plot(range(16),(g.q95-g.q05).to_numpy(),marker='.',label=str(index+1))
        ax.set(title=f'{metric}: q95 - q05',xticks=range(16),xticklabels=[c.replace('EEG_','') for c in channels],ylabel='µV')
    axes[1].legend(title='file index',ncol=2);fig.savefig(figures/'amplitude_distribution.png',dpi=120);plt.close(fig)
    summary=pd.read_csv(out/'simulation_edf_summary.csv')
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
    for scheme,g in summary[summary.scheme.str.startswith('all16')].groupby('scheme'):
        ax.plot(range(1,len(edfs)+1),g.set_index('edf_name').loc[edfs].expected_retention*100,marker='o',label=scheme)
    ax.set(xticks=range(1,len(edfs)+1),xlabel='manifest file index',ylabel='expected retention (%)',ylim=(0,102));ax.legend()
    fig.savefig(figures/'threshold_retention.png',dpi=120);plt.close(fig)
    # Four deterministic examples only: earliest non-startup candidate and earliest
    # full 4-second interval without a candidate, for one fixed subject per site.
    examples=[]
    for site in ['Duke','ETSU']:
        selected=manifest[manifest.site==site].iloc[0]
        stem=Path(selected.edf_name).stem
        events=frames['events'][frames['events'].edf_name==selected.edf_name]
        blinks=frames['blinks'][frames['blinks'].edf_name==selected.edf_name]
        with np.load(out/stem/'continuous_eeg.npz') as data:
            y=data['eeg_v'];aux=data['auxiliary_eeg_v'];channels=data['channels'].tolist();fs=float(data['sfreq'])
        clean_blinks=blinks[blinks.filter_startup.eq(False)]
        choices=[]
        if len(clean_blinks):choices.append(('candidate',float(clean_blinks.iloc[0].peak_seconds)))
        for center in np.arange(12.,y.shape[1]/fs-2,4.):
            if not ((blinks.onset_seconds<center+2)&(blinks.offset_seconds>center-2)).any():
                choices.append(('untriggered_unknown',center));break
        for kind,center in choices:
            a=max(0,round((center-2)*fs));b=min(y.shape[1],a+round(4*fs))
            fig,axes=plt.subplots(3,1,figsize=(12,7),sharex=True,layout='constrained')
            for ch in ['EEG_F3','EEG_Fz','EEG_F4']:axes[0].plot(np.arange(a,b)/fs,y[channels.index(ch),a:b]*1e6,label=ch)
            for ch in ['EEG_Pz','EEG_PO7','EEG_PO8','EEG_Oz']:axes[1].plot(np.arange(a,b)/fs,y[channels.index(ch),a:b]*1e6,label=ch)
            axes[2].plot(np.arange(a,b)/fs,np.median(aux[[channels.index(ch) for ch in ['EEG_F3','EEG_Fz','EEG_F4']],a:b],axis=0)*1e6,label='slow frontal median')
            nearby=blinks[(blinks.onset_seconds<b/fs)&(blinks.offset_seconds>a/fs)]
            for _,v in nearby.iterrows():
                for ax in axes:ax.axvspan(v.onset_seconds,v.offset_seconds,color='orange',alpha=.18)
                axes[2].axvline(v.peak_seconds,color='red',ls='--',label='peak')
                axes[2].axvline(v.decision_seconds,color='purple',ls=':',label='decision')
            for t in events.loc[(events.onset_seconds>=a/fs)&(events.onset_seconds<b/fs),'onset_seconds']:
                for ax in axes:ax.axvline(t,color='gray',lw=.5,alpha=.5)
            for ax in axes:ax.set_ylabel('µV');ax.legend(loc='upper right',fontsize=7)
            axes[-1].set_xlabel('EDF time (s)')
            fig.suptitle(f'{stem}: {kind}; gray = flash; orange = candidate; untriggered is not confirmed clean')
            filename=f'{stem}_{kind}.png';fig.savefig(figures/filename,dpi=120);plt.close(fig)
            examples.append(dict(edf_name=selected.edf_name,kind=kind,start_seconds=a/fs,stop_seconds=b/fs,figure=filename,selection='earliest eligible candidate / earliest fixed-grid nontriggered interval; no labels'))
    write_csv(examples,out/'plotted_examples.csv')
    # Actual responses of both causal branches (auxiliary includes official cascade).
    first=Path(manifest.iloc[0].edf_name).stem
    with np.load(out/first/'continuous_eeg.npz') as data:
        sos=data['sos'];aux_sos=data['auxiliary_sos'];fs=float(data['sfreq'])
    frequency=np.linspace(.05,fs/2-.01,10000)
    response=[]
    fig,axes=plt.subplots(2,1,figsize=(10,7),layout='constrained')
    for name,cascade in [('analysis',sos),('auxiliary_total',np.vstack([sos,aux_sos]))]:
        f,h=signal.sosfreqz(cascade,worN=frequency,fs=fs);gain=20*np.log10(np.maximum(abs(h),1e-30));delay=-np.gradient(np.unwrap(np.angle(h)),2*np.pi*f)
        axes[0].plot(f,gain,label=name);axes[1].plot(f,delay*1000,label=name)
        response.append(pd.DataFrame(dict(branch=name,frequency_hz=f,gain_db=gain,group_delay_seconds=delay)))
    axes[0].set(xlim=(0,125),ylim=(-120,5),ylabel='dB');axes[1].set(xlim=(.5,30),ylabel='group delay ms',xlabel='Hz')
    for ax in axes:ax.legend()
    fig.savefig(figures/'causal_branch_responses.png',dpi=120);plt.close(fig)
    write_csv(pd.concat(response),out/'causal_branch_response.csv')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/expanded_pilot.json')
    parser.add_argument('--data-root',type=Path,default=ROOT/'data/raw/bigP3BCI-data/StudyF')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/expanded_pilot')
    args=parser.parse_args();cfg=json.loads(args.config.read_text(encoding='utf8'))
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    manifest=selection_manifest(args.data_root,cfg)
    manifest_path=out/'selection_manifest.csv'
    if manifest_path.exists():
        previous=pd.read_csv(manifest_path)
        if not previous.equals(manifest):raise ValueError('Frozen manifest differs; choose a new output directory')
    # Freeze before any EDF signal or detection is read.
    write_csv(manifest,manifest_path)
    (out/'config_snapshot.json').write_text(json.dumps(cfg,indent=2),encoding='utf8')
    results=[]
    for i,r in manifest.iterrows():
        path=args.data_root/r.relative_path
        require_train_edf(path)
        print(f'[{i+1}/{len(manifest)}] {path.name} ({r.site}, inferred)',flush=True)
        result=process_expanded_file(path,r.site,cfg,out/path.stem,ROOT)
        results.append(result)
        print(f'  {len(result["events"])} flashes; {len(result["blinks"])} blink candidates; {len(result["quality"])} raw quality candidates',flush=True)
    frames=aggregate(results,out);plots(frames,manifest,out)
    summary=dict(n_files=len(manifest),subjects=manifest.subject_id.unique().tolist(),sessions_per_subject=manifest.groupby('subject_id').session_id.nunique().to_dict(),
                 n_events=len(frames['events']),n_blink_candidates=len(frames['blinks']),n_basic_candidates=len(frames['quality']),
                 event_counts=frames['events'].groupby('edf_name').size().to_dict(),thresholds_validated=False,automatic_epoch_rejection=False)
    (out/'run_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
