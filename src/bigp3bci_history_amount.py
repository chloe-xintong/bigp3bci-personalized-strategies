"""Enumerated file subsets and subject-paired bootstrap summaries."""
from itertools import combinations
from math import comb
import numpy as np
import pandas as pd


def enumerate_subsets(n_files=6):
    return [(h,tuple(indices)) for h in range(n_files+1) for indices in combinations(range(n_files),h)]


def mean_ci(values,indices):
    x=np.asarray(values,float)
    if not np.isfinite(x).all():return np.nan,np.nan,np.nan
    low,high=np.quantile(x[indices].mean(axis=1),[.025,.975])
    return float(x.mean()),float(low),float(high)


def summarize_history(raw,subjects,k_values,replicates=10000,seed=20261126):
    subjects=sorted(subjects)
    indices=np.random.default_rng(seed).integers(0,len(subjects),size=(replicates,len(subjects)))
    subject_rows=[]
    for (subject,h,k),g in raw.groupby(['subject_id','H','k'],sort=True):
        if len(g)!=comb(6,int(h)):raise ValueError('Incomplete subset enumeration')
        expected_available=not (h==0 and k==0)
        if expected_available and not g.model_available.all():raise ValueError('Unavailable subset must not be silently omitted')
        subject_rows.append(dict(subject_id=subject,H=h,k=k,history_characters=6*h,subsets=len(g),
            mean_auc=float(g.flash_roc_auc.mean()) if expected_available else np.nan,
            min_auc=float(g.flash_roc_auc.min()) if expected_available else np.nan,
            max_auc=float(g.flash_roc_auc.max()) if expected_available else np.nan,
            mean_character_accuracy=float(g.character_accuracy.mean()) if expected_available else np.nan))
    per_subject=pd.DataFrame(subject_rows)
    current=per_subject[per_subject.H.eq(0)][['subject_id','k','mean_auc']].rename(columns={'mean_auc':'current_only_auc'})
    per_subject=per_subject.merge(current,on=['subject_id','k'],validate='many_to_one')
    per_subject['delta_vs_current']=per_subject.mean_auc-per_subject.current_only_auc
    macro=[]
    for (h,k),g in per_subject.groupby(['H','k'],sort=True):
        g=g.set_index('subject_id').reindex(subjects)
        mean,lo,hi=mean_ci(g.mean_auc,indices)
        delta,dlo,dhi=mean_ci(g.delta_vs_current,indices)
        base,blo,bhi=mean_ci(g.current_only_auc,indices)
        macro.append(dict(H=h,k=k,history_characters=6*h,subjects=len(subjects),subsets_per_subject=comb(6,int(h)),
                          mean_auc=mean,ci_low=lo,ci_high=hi,current_only_auc=base,current_ci_low=blo,current_ci_high=bhi,
                          delta_vs_current=delta,delta_ci_low=dlo,delta_ci_high=dhi,
                          mean_character_accuracy=float(g.mean_character_accuracy.mean()) if not (h==0 and k==0) else np.nan))
    session_rows=[]
    for (subject,h,k),g in raw[raw.H.between(1,3)].groupby(['subject_id','H','k']):
        s1=g[g.n_s1.eq(h)];s2=g[g.n_s2.eq(h)]
        if len(s1)!=comb(3,int(h)) or len(s2)!=comb(3,int(h)):raise ValueError('Missing pure-session subsets')
        a=float(s1.flash_roc_auc.mean());b=float(s2.flash_roc_auc.mean())
        session_rows.append(dict(subject_id=subject,H=h,k=k,history_characters=6*h,
                                 s1_mean_auc=a,s2_mean_auc=b,delta_s2_minus_s1=b-a))
    sessions=pd.DataFrame(session_rows);session_macro=[]
    for (h,k),g in sessions.groupby(['H','k']):
        g=g.set_index('subject_id').reindex(subjects)
        delta,low,high=mean_ci(g.delta_s2_minus_s1,indices)
        session_macro.append(dict(H=h,k=k,history_characters=6*h,subjects=len(subjects),
            s1_mean_auc=float(g.s1_mean_auc.mean()),s2_mean_auc=float(g.s2_mean_auc.mean()),
            delta_s2_minus_s1=delta,ci_low=low,ci_high=high))
    return per_subject,pd.DataFrame(macro),sessions,pd.DataFrame(session_macro)
