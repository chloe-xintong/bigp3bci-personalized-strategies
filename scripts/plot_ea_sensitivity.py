"""EA sensitivity figures.

1. ea_condition_auc: AUC by k for A (blue), B (vermillion), C (gray dashed);
   current only is a gray reference labelled in place. 10 subjects + macro.
2. ea_subject_deltas: B-A and C-current only per subject; k uses the ordinal
   purple ramp shared with the history-amount figures.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bigp3bci_figure_style import (K_COLORS,K_MARKERS,EA_CONDITIONS,EA_REFERENCE,BAND_ALPHA,AUC_LIMITS,
                                   REFERENCE_TEXT,label_references)

PANELS=[('ea_with_history','B - A: EA effect with history + current'),
        ('ea_current_only','C - current only: EA effect without history')]


def plot_conditions(out,cfg):
    res=pd.read_csv(out/'condition_results.csv');subjects=sorted(res.subject_id.unique());k_values=cfg['k_values']
    if res.flash_roc_auc.min()<AUC_LIMITS[0] or res.flash_roc_auc.max()>AUC_LIMITS[1]:raise ValueError('AUC outside fixed axis')
    boot=cfg['bootstrap'];index=np.random.default_rng(boot['seed']).integers(0,len(subjects),(boot['replicates'],len(subjects)))
    fig=plt.figure(figsize=(20,12),layout='constrained');grid=fig.add_gridspec(3,5,height_ratios=[1,1,1.15])
    axes=[fig.add_subplot(grid[i//5,i%5]) for i in range(len(subjects))]+[fig.add_subplot(grid[2,:])]
    for i,ax in enumerate(axes):
        macro=i==len(subjects)
        table=res if macro else res[res.subject_id.eq(subjects[i])]
        mean=table.groupby(['condition','k']).flash_roc_auc.mean()
        reference=mean.loc['current_only'].reindex(k_values)
        ax.plot(k_values,reference.to_numpy(),**EA_REFERENCE,zorder=1)
        for condition,style in EA_CONDITIONS.items():
            values=mean.loc[condition].reindex(k_values).to_numpy()
            if macro and condition in ('A','B'):
                lo=[];hi=[]
                for k in k_values:
                    v=res[res.condition.eq(condition)&res.k.eq(k)].set_index('subject_id').flash_roc_auc.reindex(subjects).to_numpy(float)
                    a,b=np.quantile(v[index].mean(axis=1),[.025,.975]);lo.append(a);hi.append(b)
                if condition=='A':ax.fill_between(k_values,lo,hi,color=style['color'],alpha=BAND_ALPHA,lw=0,zorder=2)
                else:
                    for edge in (lo,hi):ax.plot(k_values,edge,color=style['color'],lw=.9,ls=(0,(2,2)),zorder=2)
            ax.plot(k_values,values,ls=style['ls'],lw=style['lw']*(1 if macro else .8),marker=style['marker'],
                    ms=6 if macro else 4.5,color=style['color'],zorder=style['zorder'],label=style['label'])
        ax.set(ylim=AUC_LIMITS,xticks=k_values,xlim=(k_values[0]-.8,k_values[-1]+.8),
               xlabel='Current calibration characters (k)')
        label_references(ax,[(float(reference.iloc[-1]),'current only')],color=REFERENCE_TEXT)
        title=(f'Macro mean: {len(subjects)} subjects | subject-bootstrap 95% CI for A and B' if macro else
               subjects[i]+(' (largest drift asymmetry)' if subjects[i] in cfg['highlight_subjects'] else ''))
        ax.set_title(title,fontsize=12 if macro else 11)
        if i%5==0 or macro:ax.set_ylabel('Validation flash ROC-AUC')
        ax.tick_params(labelsize=8);ax.grid(axis='y',alpha=.2)
    handles=[Line2D([0],[0],color=s['color'],ls=s['ls'],lw=s['lw'],marker=s['marker'],label=s['label']) for s in EA_CONDITIONS.values()]
    handles.append(Patch(color=EA_CONDITIONS['A']['color'],alpha=.3,label='Macro 95% CI: A (band)'))
    handles.append(Line2D([0],[0],color=EA_CONDITIONS['B']['color'],lw=.9,ls=(0,(2,2)),label='Macro 95% CI: B (thin dashed edges)'))
    fig.legend(handles=handles,loc='outside lower center',ncol=len(handles),fontsize=10,frameon=False)
    fig.suptitle('Session-wise EA sensitivity (SWLDA, all 36 historical characters) | exploratory Train validation | k=0 undefined for EA\n'
                 'Gray dotted: current only without alignment (reference, labelled at right)',fontsize=13)
    fig.savefig(out/'ea_condition_auc.png',dpi=160);fig.savefig(out/'ea_condition_auc.pdf');plt.close(fig)


def plot_deltas(out,cfg=None):
    out=Path(out);cfg=cfg or json.loads((out/'config_snapshot.json').read_text())['protocol']
    deltas=pd.read_csv(out/'subject_deltas.csv');macro=pd.read_csv(out/'macro_deltas.csv')
    subjects=sorted(deltas.subject_id.unique());rows=subjects+['Macro mean']
    k_values=cfg['k_values'];offset={k:(i-(len(k_values)-1)/2)*.22 for i,k in enumerate(k_values)}
    values=pd.concat([deltas.ea_with_history,deltas.ea_current_only,macro.ci_low,macro.ci_high])
    span=max(abs(values.min()),abs(values.max()))*1.12
    fig,axes=plt.subplots(1,2,figsize=(12,6.8),sharey=True,sharex=True,layout='constrained')
    for ax,(column,title) in zip(axes,PANELS):
        for i,subject in enumerate(subjects):
            if subject in cfg['highlight_subjects']:ax.axhspan(i-.45,i+.45,color='#F0F0F0',zorder=0)
            for k in k_values:
                v=deltas[deltas.subject_id.eq(subject)&deltas.k.eq(k)][column].iloc[0]
                ax.plot(v,i+offset[k],K_MARKERS[k],color=K_COLORS[k],ms=6,mec='#54278F' if k==3 else K_COLORS[k],mew=.6,zorder=3)
        y=len(subjects)
        for r in macro[macro.comparison.eq(column)].itertuples():
            ax.errorbar(r.mean_delta,y+offset[r.k],xerr=[[r.mean_delta-r.ci_low],[r.ci_high-r.mean_delta]],
                        fmt=K_MARKERS[r.k],color=K_COLORS[r.k],mec='#54278F',mew=.6,ms=7,capsize=3,lw=1.4,zorder=3)
        ax.axhline(len(subjects)-.5,color='#BFBFBF',lw=.8)
        ax.axvline(0,color='#7F7F7F',lw=.9,zorder=1)
        ax.set(title=title,xlim=(-span,span),xlabel='Validation flash ROC-AUC difference (aligned - unaligned)')
        ax.grid(axis='x',alpha=.2);ax.tick_params(labelsize=9)
    axes[0].set_yticks(range(len(rows)),[s+(' *' if s in cfg['highlight_subjects'] else '') for s in rows])
    axes[0].invert_yaxis()
    handles=[Line2D([0],[0],ls='',marker=K_MARKERS[k],color=K_COLORS[k],mec='#54278F',mew=.6,label=f'k={k}') for k in k_values]
    handles.append(Line2D([0],[0],color='#595959',lw=1.4,label='Macro: subject-bootstrap 95% CI'))
    fig.legend(handles=handles,loc='outside lower center',ncol=len(handles),fontsize=9,frameon=False)
    fig.suptitle('Session-wise EA sensitivity: paired differences per subject | exploratory Train validation\n'
                 '* shaded rows: largest session drift asymmetry (F_07, F_21); k=0 not defined for EA',fontsize=11)
    fig.savefig(out/'ea_subject_deltas.png',dpi=180);fig.savefig(out/'ea_subject_deltas.pdf');plt.close(fig)
    plot_conditions(out,cfg)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='outputs/ea_sensitivity_v1')
    plot_deltas(ROOT/parser.parse_args().output)
