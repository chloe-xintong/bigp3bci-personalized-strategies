"""Separate supplementary figures; no main-result changes."""
from pathlib import Path
import sys,argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bigp3bci_figure_style import SCHEMES

TITLE='Cross-subject | exploratory | supplementary analysis'
STYLES={'X':dict(color=SCHEMES['history_plus_current']['color'],label='Other subjects + current (X)',lw=2.4),
        'C':dict(color=SCHEMES['current_only']['color'],label='Current only (C)',lw=1.4)}


def plot(out):
    subjects=pd.read_csv(out/'subject_results.csv');macro=pd.read_csv(out/'macro_results.csv')
    for metric,filename,ylabel in [('flash_roc_auc','figure_per_subject.png','Validation flash ROC-AUC'),('accuracy','figure_character_per_subject.png','Validation character accuracy (descriptive)')]:
        fig,axes=plt.subplots(2,5,figsize=(19,8),sharey=True,layout='constrained')
        for s,ax in zip(sorted(subjects.subject_id.unique()),axes.flat):
            for cond,style in STYLES.items():
                g=subjects[subjects.subject_id.eq(s)&subjects.condition.eq(cond)].sort_values('k')
                ax.plot(g.k,g[metric],'-o',**style)
            ax.set(title=s,xticks=[0,3,6,12],ylim=(0,1.02),xlabel='Target calibration characters (k)',ylabel=ylabel);ax.grid(alpha=.2)
        fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='outside lower center',ncol=2)
        fig.suptitle(TITLE);fig.savefig(out/filename,dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
    for cond,style in STYLES.items():
        g=macro[macro.condition.eq(cond)].sort_values('k')
        ax.plot(g.k,g.macro_flash_roc_auc,'-o',**style)
        ax.fill_between(g.k,g.ci_low,g.ci_high,color=style['color'],alpha=.15)
    ax.set(title=TITLE+'\nMacro mean; 95% subject-bootstrap intervals (10,000 resamples)',xticks=[0,3,6,12],ylim=(0,1.02),xlabel='Target calibration characters (k)',ylabel='Validation flash ROC-AUC (10-subject mean)')
    ax.grid(alpha=.2);ax.legend();fig.savefig(out/'figure_macro.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/cross_subject_swlda_exploratory_v1');plot(ROOT/p.parse_args().output)
