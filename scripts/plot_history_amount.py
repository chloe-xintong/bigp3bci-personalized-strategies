"""Shared-axis subject small multiples and subject-bootstrap macro panels."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT/'src'))
# k=0 is history only (a different scheme): black. k=3/6/12: one ordered purple
# ramp, kept apart from the blue/orange/green scheme colors used in notebook 07.
from bigp3bci_figure_style import (K_COLORS as COLORS,K_MARKERS as MARKERS,K_REFERENCE as REFERENCE,
                                   K_BAND_COLORS as BAND_COLORS,AUC_LIMITS,BAND_ALPHA,label_references)
CEILING_AUC=.95


def ceiling_subjects(subject):
    """Subjects whose k=12 current-only AUC is already near the ceiling."""
    ref=subject[subject.k.eq(12)].drop_duplicates('subject_id').set_index('subject_id').current_only_auc
    return set(ref[ref>=CEILING_AUC].index)


def reference_lines(ax,table,k_values):
    """Light gray current-only references, labelled only with their k at the right edge."""
    items=[(float(table[table.k.eq(k)].current_only_auc.iloc[0]),f'k={k}') for k in k_values if k>0]
    for value,_ in items:ax.axhline(value,**REFERENCE,zorder=1)
    label_references(ax,items,color='#7F7F7F')


def draw(out,raw,subject,macro,kind,cfg):
    auc_style=kind in ('auc','character')
    subjects=sorted(subject.subject_id.unique())
    fig=plt.figure(figsize=(20,12),layout='constrained')
    grid=fig.add_gridspec(3,5,height_ratios=[1,1,1.15])
    axes=[fig.add_subplot(grid[i//5,i%5]) for i in range(len(subjects))]
    axes.append(fig.add_subplot(grid[2,:]))
    rng=np.random.default_rng(cfg['display']['jitter_seed'])
    k_values=cfg['k_values'] if kind!='delta' else [k for k in cfg['k_values'] if k>0]
    if kind=='session':
        values=subject.delta_s2_minus_s1.to_numpy();x_ticks=[6,12,18]
    elif kind=='delta':
        baseline=subject[['subject_id','k','current_only_auc']].drop_duplicates(['subject_id','k'])
        scatter=raw.merge(baseline,on=['subject_id','k'],validate='many_to_one')
        scatter['value']=scatter.flash_roc_auc-scatter.current_only_auc
        values=scatter.value.dropna().to_numpy();x_ticks=list(range(0,37,6))
    else:
        values=raw.flash_roc_auc.dropna().to_numpy();x_ticks=list(range(0,37,6))
    minimum=float(np.min(values));maximum=float(np.max(values));pad=max(.015,.07*(maximum-minimum))
    limits=(0,102) if kind=='character' else (AUC_LIMITS if kind=='auc' else (minimum-pad,maximum+pad))
    if kind=='auc' and (minimum<AUC_LIMITS[0] or maximum>AUC_LIMITS[1]):raise ValueError('AUC outside the fixed shared axis')
    ceiling=ceiling_subjects(subject) if kind in ('auc','delta') else set()
    for i,ax in enumerate(axes):
        is_macro=i==len(subjects)
        name=subjects[i] if not is_macro else (f'Macro mean: {len(subjects)} subjects; descriptive only' if kind=='character' else f'Macro mean: {len(subjects)} subjects; subject-bootstrap 95% CI')
        table=macro if is_macro else subject[subject.subject_id.eq(name)]
        for k in k_values:
            g=table[table.k.eq(k)].sort_values('H');color=COLORS[k]
            y='mean_auc' if auc_style else ('delta_vs_current' if kind=='delta' else 'delta_s2_minus_s1')
            ax.plot(g.history_characters,g[y],'-',marker=MARKERS[k],color=color,lw=1.7,ms=4.5,label=f'k={k}',zorder=3)
            if is_macro and kind!='character' and k in BAND_COLORS:
                low='ci_low' if kind!='delta' else 'delta_ci_low';high='ci_high' if kind!='delta' else 'delta_ci_high'
                ax.fill_between(g.history_characters,g[low],g[high],color=BAND_COLORS[k],alpha=BAND_ALPHA,lw=0,zorder=2)
                for edge in (low,high):ax.plot(g.history_characters,g[edge],color=BAND_COLORS[k],lw=.6,alpha=.7,zorder=2)
            elif not is_macro and kind!='session':
                points=(raw if auc_style else scatter)
                points=points[points.subject_id.eq(name)&points.k.eq(k)]
                jitter=rng.uniform(-cfg['display']['jitter_characters'],cfg['display']['jitter_characters'],len(points))
                ax.scatter(points.H.to_numpy()*6+jitter,points.flash_roc_auc if auc_style else points.value,
                           s=10,alpha=.3,color=color,edgecolors='none',zorder=2)
        if auc_style:ax.set_ylim(limits);reference_lines(ax,table,k_values)
        if not auc_style:ax.axhline(0,color='#7F7F7F',ls='-',lw=.8)
        if name in ceiling:name+=' (baseline near ceiling)'
        ax.set(title=name,ylim=limits,xticks=x_ticks,xlim=(4.5,19.5) if kind=='session' else (-1.5,37.5))
        ax.set_xlabel('Historical characters (6 per file)',fontsize=9)
        ax.set_ylabel('Character accuracy (%)' if kind=='character' else ('Validation flash ROC-AUC' if kind=='auc' else ('AUC difference vs current only' if kind=='delta' else 'AUC: S2 only - S1 only')),fontsize=9)
        ax.tick_params(labelsize=8);ax.grid(axis='y',alpha=.2)
    handles=[Line2D([0],[0],color=COLORS[k],marker=MARKERS[k],label='k=0 (history only)' if k==0 else f'k={k}') for k in k_values]
    if kind!='character':handles.append(Patch(color=BAND_COLORS[12],alpha=.3,label='Macro band: 95% CI, k=3 and k=12 only'))
    fig.legend(handles=handles,loc='outside lower center',ncol=len(handles),fontsize=10)
    title={'auc':'Historical data amount: all SWLDA file subsets','delta':'Historical data benefit vs current-only calibration',
           'session':'Historical Session contrast at equal file counts','character':'Character accuracy by historical amount (descriptive only)'}[kind]
    fig.suptitle(title+' | Exploratory Train validation; 6 characters per subject'
                 +('\nGray dashed: current-only reference, labelled with its k at right' if auc_style else ''),fontsize=14)
    name={'auc':'history_amount_small_multiples','delta':'history_vs_current_small_multiples','session':'session_contrast_small_multiples','character':'character_accuracy_small_multiples'}[kind]
    fig.savefig(out/(name+'.png'),dpi=160)
    fig.savefig(out/(name+'.pdf'))
    plt.close(fig)
    if kind=='auc':
        appendix=out/'subjects';appendix.mkdir(exist_ok=True)
        for sid in subjects:
            fig,ax=plt.subplots(figsize=(6,4.5),layout='constrained')
            for k in k_values:
                g=subject[subject.subject_id.eq(sid)&subject.k.eq(k)].sort_values('H')
                points=raw[raw.subject_id.eq(sid)&raw.k.eq(k)]
                jitter=rng.uniform(-cfg['display']['jitter_characters'],cfg['display']['jitter_characters'],len(points))
                ax.scatter(points.H*6+jitter,points.flash_roc_auc,s=13,alpha=.3,color=COLORS[k],edgecolors='none')
                ax.plot(g.history_characters,g.mean_auc,'-',marker=MARKERS[k],color=COLORS[k],lw=1.6,ms=4.5,
                        label='k=0 (history only)' if k==0 else f'k={k}')
            ax.set_ylim(limits);reference_lines(ax,subject[subject.subject_id.eq(sid)],k_values)
            ax.set(title=sid+(' (baseline near ceiling)' if sid in ceiling else '')+' | gray dashed: current only',xlabel='Historical characters (6 per file)',
                   ylabel='Validation flash ROC-AUC',ylim=limits,xticks=x_ticks,xlim=(-1.5,37.5))
            ax.grid(axis='y',alpha=.2)
            fig.legend(*ax.get_legend_handles_labels(),loc='outside lower center',ncol=4,fontsize=9)
            fig.savefig(appendix/(sid+'.png'),dpi=180);plt.close(fig)


def draw_macro(out,macro,cfg):
    """Enlarged macro panel for the main results; subject panels go to the appendix."""
    fig,ax=plt.subplots(figsize=(8.5,5.6),layout='constrained')
    for k in cfg['k_values']:
        g=macro[macro.k.eq(k)].sort_values('H')
        if k in BAND_COLORS:
            ax.fill_between(g.history_characters,g.ci_low,g.ci_high,color=BAND_COLORS[k],alpha=BAND_ALPHA,lw=0,zorder=2)
            for edge in ('ci_low','ci_high'):ax.plot(g.history_characters,g[edge],color=BAND_COLORS[k],lw=.7,alpha=.7,zorder=2)
        ax.plot(g.history_characters,g.mean_auc,'-',marker=MARKERS[k],color=COLORS[k],lw=2,ms=6,zorder=3,
                label='k=0 (history only)' if k==0 else f'k={k}')
    ax.set_ylim(AUC_LIMITS);reference_lines(ax,macro,cfg['k_values'])
    ax.set(ylim=AUC_LIMITS,xlim=(-1.5,37.5),xticks=list(range(0,37,6)),xlabel='Historical characters (6 per file)',
           ylabel='Validation flash ROC-AUC (macro mean, 10 subjects)',
           title='Historical data amount: macro mean with subject-bootstrap 95% CI')
    ax.grid(axis='y',alpha=.2)
    handles,labels=ax.get_legend_handles_labels()
    handles+=[Patch(color=BAND_COLORS[12],alpha=.3)]
    labels+=['95% CI, k=3 and k=12 only']
    ax.text(0,1.005,'Gray dashed: current only, labelled with its k at right',transform=ax.transAxes,fontsize=8,color='#595959',va='bottom')
    ax.legend(handles,labels,loc='lower right',fontsize=9,frameon=False)
    fig.savefig(out/'history_amount_macro.png',dpi=200);fig.savefig(out/'history_amount_macro.pdf');plt.close(fig)


def plot_all(out):
    out=Path(out);cfg=json.loads((out/'config_snapshot.json').read_text())['protocol']
    raw=pd.read_csv(out/'subset_results.csv');subject=pd.read_csv(out/'subject_summary.csv');macro=pd.read_csv(out/'macro_summary.csv')
    draw(out,raw,subject,macro,'auc',cfg);draw_macro(out,macro,cfg);draw(out,raw,subject,macro,'delta',cfg)
    draw(out,raw,pd.read_csv(out/'session_subject_contrasts.csv'),pd.read_csv(out/'session_macro_contrasts.csv'),'session',cfg)
    char_raw=raw.copy();char_raw['flash_roc_auc']=char_raw.character_accuracy*100
    char_subject=subject.copy();char_subject['mean_auc']=char_subject.mean_character_accuracy*100
    baseline=char_subject[char_subject.H.eq(0)].set_index(['subject_id','k']).mean_auc.to_dict()
    char_subject['current_only_auc']=[baseline[(r.subject_id,r.k)] for r in char_subject.itertuples()]
    char_macro=macro.copy();char_macro['mean_auc']=char_macro.mean_character_accuracy*100
    macro_baseline=char_macro[char_macro.H.eq(0)].set_index('k').mean_auc.to_dict()
    char_macro['current_only_auc']=char_macro.k.map(macro_baseline)
    draw(out,char_raw,char_subject,char_macro,'character',cfg)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='outputs/history_amount_exploratory_v1')
    args=parser.parse_args();plot_all(ROOT/args.output)
