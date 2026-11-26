"""Shared rendering of calibration curves and fixed historical baselines.

Colors come from bigp3bci_figure_style: scheme colors are Okabe-Ito, the main
design is the thickest line, and reference lines are gray, labelled in place
and kept out of the legend.
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import math
import numpy as np
from bigp3bci_figure_style import (SCHEMES,SCHEME_REFERENCE,SCHEME_REFERENCE_STYLES,BAND_ALPHA,AUC_LIMITS,
                                   label_references)

REFERENCE_NAMES={'history_only':'S1+S2 only (k=0)','history_se001_only':'S1 only (k=0)',
                 'history_se002_only':'S2 only (k=0)','all_available_reference':'All training data'}
BAND_SCHEMES=('history_plus_current','current_only')


def _curve(ax,g,scheme,values,small=False):
    style=SCHEMES[scheme]
    ax.plot(g.k,values,'-',marker=style['marker'],color=style['color'],lw=style['lw']*(.85 if small else 1),
            ms=4 if small else 5,zorder=style['zorder'],label=style['label'])


def _references(ax,table,column,scale,schemes,loss):
    items=[]
    for scheme in schemes:
        g=table[table.scheme.eq(scheme)]
        if not len(g) or not np.isfinite(g[column].iloc[0]):continue
        value=float(g[column].iloc[0])*scale
        ax.axhline(value,ls=SCHEME_REFERENCE_STYLES[scheme],**SCHEME_REFERENCE,zorder=1)
        items.append((value,REFERENCE_NAMES[scheme]))
    if loss:
        ax.axhline(math.log(2),ls=(0,(5,3)),**SCHEME_REFERENCE,zorder=1);items.append((math.log(2),'p=0.5 (ln 2)'))
    return items


def _legend(fig,band):
    handles=[Line2D([0],[0],color=s['color'],lw=s['lw'],marker=s['marker'],label=s['label']) for s in SCHEMES.values()]
    if band:handles.append(Patch(color=SCHEMES['history_plus_current']['color'],alpha=.3,
                                 label='Macro 95% CI (main design and current only)'))
    fig.legend(handles=handles,loc='outside lower center',ncol=len(handles),fontsize=9,frameon=False)


def plot_budget_results(overall,path,k_values,metric='macro_flash_roc_auc'):
    fig,axes=plt.subplots(1,2,figsize=(15,6.4),gridspec_kw={'width_ratios':[1,1.65]},layout='constrained')
    auc=metric=='macro_flash_roc_auc'
    logloss=metric=='macro_weighted_log_loss'
    scale=1 if auc or logloss else 100
    full_limits=(0,1.02) if auc else ((0,max(1.,float(overall[metric].max())*1.08)) if logloss else (0,102))
    cluster=overall.loc[~overall.scheme.eq('current_only'),metric].dropna()*scale
    lower,upper=float(cluster.min()),float(cluster.max())
    step=.01 if auc or logloss else 5.
    padding=max(.005 if auc or logloss else 1.,(upper-lower)*.1)
    zoom_limits=(max(0,math.floor((lower-padding)/step)*step),math.ceil((upper+padding)/step)*step)
    points=overall[overall.scheme.isin(list(SCHEMES))][metric].dropna()*scale
    outside=int(((points<zoom_limits[0])|(points>zoom_limits[1])).sum())
    references=['history_only','all_available_reference']
    references+=[s for s,c in [('history_se001_only','history_se001_plus_current'),('history_se002_only','history_se002_plus_current')]
                 if not overall.scheme.eq(c).any()]
    for i,ax in enumerate(axes):
        for scheme in SCHEMES:
            g=overall[overall.scheme.eq(scheme)].sort_values('k')
            if len(g):_curve(ax,g,scheme,g[metric]*scale)
        items=_references(ax,overall,metric,scale,references,logloss)
        ax.set(xlabel='Current-session calibration characters (k)',
           ylabel='Validation flash ROC-AUC (subject macro mean)' if auc else ('Class-balanced flash log-loss (lower is better)' if logloss else 'Train holdout character accuracy (%)'),
           ylim=full_limits if i==0 else zoom_limits,xticks=k_values)
        label_references(ax,[(y,t) for y,t in items if ax.get_ylim()[0]<=y<=ax.get_ylim()[1]])
        ax.set_title('Full-range overview' if i==0 else f'Zoomed comparison ({outside} points outside range; see left)',fontsize=11)
        ax.grid(axis='y',alpha=.25)
    _legend(fig,band=False)
    fig.savefig(path,dpi=150);plt.close(fig)


def plot_budget_small_multiples(overall,subjects,path,k_values,metric='macro_flash_roc_auc',zoom=False):
    """Ten shared-axis subject panels followed by an explicit macro panel."""
    subject_metric={'macro_flash_roc_auc':'flash_roc_auc','macro_weighted_log_loss':'weighted_log_loss',
                    'mean_subject_accuracy':'accuracy'}[metric]
    people=sorted(subjects.subject_id.unique())
    if len(people)!=10:raise ValueError('Expected ten subjects for the fixed 2-by-5 layout')
    auc=metric=='macro_flash_roc_auc';loss=metric=='macro_weighted_log_loss'
    scale=1 if auc or loss else 100
    cluster=subjects[~subjects.scheme.eq('current_only')] if zoom else subjects
    data=cluster[subject_metric].dropna().to_numpy()*scale
    if auc:
        lower,upper=AUC_LIMITS
        if zoom:lower=max(0,float(data.min())-.025);upper=min(1.01,float(data.max())+.025)
    elif loss:lower=0;upper=max(1.,float(data.max())*1.06)
    else:lower=0;upper=102
    fig=plt.figure(figsize=(20,12),layout='constrained');grid=fig.add_gridspec(3,5,height_ratios=[1,1,1.1])
    axes=[fig.add_subplot(grid[i//5,i%5]) for i in range(10)]+[fig.add_subplot(grid[2,:])]
    index=np.random.default_rng(20261126).integers(0,10,(10000,10))
    for i,ax in enumerate(axes):
        macro=i==10;table=overall if macro else subjects[subjects.subject_id.eq(people[i])]
        column=metric if macro else subject_metric
        for scheme in SCHEMES:
            g=table[table.scheme.eq(scheme)].sort_values('k')
            if not len(g):continue
            _curve(ax,g,scheme,g[column]*scale,small=not macro)
            if macro and (auc or loss) and scheme in BAND_SCHEMES:
                lo=[];hi=[]
                for k in g.k:
                    values=subjects[subjects.scheme.eq(scheme)&subjects.k.eq(k)].set_index('subject_id')[subject_metric].reindex(people).to_numpy(float)
                    if np.isfinite(values).all():
                        limits=np.quantile(values[index].mean(axis=1),[.025,.975])*scale
                        lo.append(limits[0]);hi.append(limits[1])
                    else:lo.append(np.nan);hi.append(np.nan)
                ax.fill_between(g.k,lo,hi,color=SCHEMES[scheme]['color'],alpha=BAND_ALPHA,lw=0,zorder=2)
        items=_references(ax,table,column,scale,['history_only','all_available_reference'],loss)
        outside=table[table.scheme.isin(list(SCHEMES))][column].dropna()*scale
        count=int(((outside<lower)|(outside>upper)).sum())
        if auc and not zoom and count:raise ValueError('AUC outside the fixed shared axis')
        title=(f'Macro mean: 10 subjects'+(' | subject-bootstrap 95% CI' if auc or loss else ' | descriptive only')) if macro else people[i]
        if zoom and count:title+=f' ({count} points off scale)'
        ax.set(title=title,ylim=(lower,upper),xticks=k_values,xlabel='Current calibration characters (k)')
        label_references(ax,[(y,t) for y,t in items if lower<=y<=upper])
        if i%5==0 or macro:ax.set_ylabel('Flash ROC-AUC' if auc else ('Weighted flash log-loss' if loss else 'Character accuracy (%)'))
        ax.tick_params(labelsize=8);ax.grid(axis='y',alpha=.2)
    _legend(fig,band=auc or loss)
    title='Flash ROC-AUC' if auc else ('Class-balanced flash log-loss' if loss else 'Character accuracy (descriptive)')
    fig.suptitle(title+' | 10 subject panels + macro mean | shared y range | gray lines: references labelled at right'
                 +(' | zoom: full-range figure retains all points and intervals' if zoom else ''),fontsize=13)
    fig.savefig(path,dpi=160);plt.close(fig)
