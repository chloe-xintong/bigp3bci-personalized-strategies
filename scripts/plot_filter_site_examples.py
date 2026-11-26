"""Fixed first-Train examples from each site; illustrative, not a site comparison."""
from pathlib import Path
import sys,json,os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('_MNE_FAKE_HOME_DIR',str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
from bigp3bci_preprocessing import design_filter,causal_filter
from bigp3bci_continuous_qc import read_bigp3bci_edf


def attach_filter_cells(notebook):
    import nbformat as nb
    import base64
    notebook.cells=[c for c in notebook.cells if not c.metadata.get('filter_site_example')]
    md=nb.v4.new_markdown_cell('## 两家机构的连续 EEG 滤波示例\n\n每家机构按文件名字典序取第一份 Train EDF；展示 30–40 秒的全部 16 通道，左右波形使用相同尺度。功率谱取启动保护期后整段信号，先逐通道 Welch，再在线性功率上平均。EDF 已保存信号可能含采集端滤波；本图仅示意额外的 60 Hz notch → 0.5–30 Hz 因果滤波，不用于推断机构差异。')
    cell=nb.v4.new_code_cell("from pathlib import Path\nfrom IPython.display import display, Image\n_filter_root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'configs').is_dir())\ndisplay(Image(filename=str(_filter_root / 'outputs/filter_site_examples/filter_by_site.png')))\n")
    picture=ROOT/'outputs/filter_site_examples/filter_by_site.png'
    if picture.exists():
        cell.outputs=[nb.v4.new_output('display_data',data={'image/png':base64.b64encode(picture.read_bytes()).decode(),'text/plain':'<Filter comparison by site>'})]
        cell.execution_count=1
    for c in [md,cell]:c.metadata['filter_site_example']=True
    notebook.cells[1:1]=[md,cell]
    return notebook


def main():
    out=ROOT/'outputs/filter_site_examples';out.mkdir(exist_ok=True)
    manifest=pd.read_csv(ROOT/'outputs/studyf_all_preprocessing/input_manifest.csv')
    chosen=manifest.sort_values('edf_name').groupby('site',sort=True).head(1).sort_values('site')
    fig,axes=plt.subplots(2,3,figsize=(19,10),layout='constrained')
    provenance=[]
    for row,entry in enumerate(chosen.itertuples()):
        record=json.loads((ROOT/entry.results_path/'processing_record.json').read_text())
        path=Path(record['source_edf']);assert path.parts[-3]=='Train'
        raw=read_bigp3bci_edf(path);channels=record['channels'];fs=float(raw.info['sfreq'])
        x=raw.get_data(picks=channels);cfg=record['config']['filter']
        y,invalid,_=causal_filter(x,design_filter(fs,cfg),fs,cfg)
        start,stop=30.,40.;a,b=round(start*fs),round(stop*fs)
        assert x.shape[1]>=b and start>=cfg['startup_seconds'] and not invalid[:,a:b].any()
        # Identical offsets and scale for raw/filtered traces within each site.
        spacing=max(25.,float(np.quantile(np.abs(np.concatenate([x[:,a:b],y[:,a:b]],axis=1))*1e6,.99))*2.3)
        offset=np.arange(16)[::-1]*spacing;t=np.arange(a,b)/fs
        for ax,values,title in zip(axes[row,:2],[x,y],['EDF stored signal','After project causal filter']):
            ax.plot(t,values[:,a:b].T*1e6+offset,lw=.55,color='#27648c')
            ax.set(yticks=offset,yticklabels=[c.removeprefix('EEG_') for c in channels],
                   xlabel='Time in file (s)',title=f'{entry.site}: {title}')
            ax.set_ylim(-spacing,offset[0]+spacing);ax.tick_params(axis='y',labelsize=7)
            ax.plot([39.5,39.5],[-spacing*.6,-spacing*.6+25],color='black',lw=2)
            ax.text(39.4,-spacing*.6,'25 uV',ha='right',fontsize=8)
        keep=slice(round(cfg['startup_seconds']*fs),None)
        spectra={}
        for values,label,color in [(x,'EDF stored','#777777'),(y,'Project filtered','#0072B2')]:
            f,power=welch(values[:,keep]*1e6,fs=fs,nperseg=round(4*fs),axis=1)
            spectra[label]=power
            axes[row,2].plot(f,10*np.log10(np.maximum(power.mean(axis=0),1e-20)),label=label,color=color)
        axes[row,2].set(xlim=(0,100),xlabel='Frequency (Hz)',ylabel='Mean-channel PSD (dB re 1 uV²/Hz)',
                        title=f'{entry.site}: Welch PSD, post-startup Train')
        axes[row,2].axvline(30,color='gray',ls=':',lw=.8);axes[row,2].axvline(60,color='gray',ls=':',lw=.8)
        axes[row,2].legend();axes[row,2].grid(alpha=.2)
        axes[row,0].set_ylabel(entry.edf_name+'\nStacked channels (shared scale within site)',fontsize=9)
        provenance.append(dict(site=entry.site,edf_name=entry.edf_name,filter=cfg,display_seconds=[start,stop],
                               psd_start_seconds=cfg['startup_seconds'],acquisition_prefilter=record['acquisition_prefilter']))
        pd.DataFrame({'frequency_hz':f,**{label:power.mean(axis=0) for label,power in spectra.items()}}).to_csv(out/(entry.site+'_psd.csv'),index=False)
        raw.close()
    fig.suptitle('Train EDF examples by site | 60 Hz notch + 0.5–30 Hz causal bandpass\n'
                 'EDF stored signals may already include acquisition filters; examples are not site-level estimates',fontsize=14)
    fig.savefig(out/'filter_by_site.png',dpi=160);plt.close(fig)
    (out/'examples.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    import nbformat as nb
    notebook=ROOT/'notebooks/02_preprocessing.ipynb'
    n=attach_filter_cells(nb.read(notebook,4));nb.validate(n);nb.write(n,notebook)
    print(out/'filter_by_site.png')


if __name__=='__main__':main()
