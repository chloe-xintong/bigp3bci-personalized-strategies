"""Classic SWLDA via forward/backward OLS stepwise binary regression.

Project implementation: p-to-enter partial F test, p-to-remove coefficient F
(t-squared) test, train-only scaling and a capped selected feature count.
This is not sklearn LDA, regularized LDA, nor a bit-exact BCI2000 port.
"""
from __future__ import annotations
import numpy as np
from scipy import stats


def fit_swlda(x,y,cfg):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if x.ndim!=2 or len(x)!=len(y) or len(x)<4 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('SWLDA needs finite training rows and matching labels')
    if not np.array_equal(np.unique(y),[0.,1.]):raise ValueError('SWLDA needs both binary classes')
    if not 0<cfg['p_enter']<cfg['p_remove']<1:raise ValueError('Require p_enter < p_remove')
    n,p=x.shape;mean=x.mean(axis=0);std=x.std(axis=0)
    usable=std>1e-12;scale=np.where(usable,std,1.)
    z=(x-mean)/scale;z[:,~usable]=0
    center=y-y.mean();gram=z.T@z;cross=z.T@center;yss=float(center@center)
    selected=[];history=[];seen=set();status='converged'
    max_features=min(int(cfg['max_features']),n-3,int(usable.sum()))
    tol=cfg['rank_tolerance']
    def fit(indices):
        if not indices:return np.empty(0),np.empty((0,0)),yss
        g=gram[np.ix_(indices,indices)]
        inverse=np.linalg.pinv(g,rcond=tol,hermitian=True)
        beta=inverse@cross[indices]
        rss=max(float(yss-cross[indices]@beta),np.finfo(float).eps*max(yss,1.))
        return beta,inverse,rss
    for step in range(int(cfg['max_steps'])):
        state=tuple(selected)
        if state in seen:
            status='cycle_detected';break
        seen.add(state);changed=False
        beta,inverse,rss=fit(selected)
        if len(selected)<max_features:
            excluded=[j for j in range(p) if usable[j] and j not in selected]
            if excluded:
                if selected:
                    g=gram[np.ix_(excluded,selected)]
                    variance=np.diag(gram)[excluded]-np.einsum('ij,jk,ik->i',g,inverse,g)
                    residual_cross=cross[excluded]-g@beta
                else:variance=np.diag(gram)[excluded];residual_cross=cross[excluded]
                valid=variance>tol*np.maximum(np.diag(gram)[excluded],1.)
                reduction=np.zeros(len(excluded));reduction[valid]=residual_cross[valid]**2/variance[valid]
                residual=np.maximum(rss-reduction,np.finfo(float).eps*max(yss,1.))
                df=n-len(selected)-2
                fvalue=reduction/(residual/df)
                probability=np.where(valid,stats.f.sf(fvalue,1,df),1.)
                best=int(np.argmin(probability))
                if probability[best]<cfg['p_enter']:
                    feature=excluded[best];selected.append(feature);changed=True
                    history.append(dict(step=step,action='enter',feature=feature,p_value=float(probability[best]),n_selected=len(selected)))
        while selected:
            beta,inverse,rss=fit(selected)
            variance_beta=(rss/(n-len(selected)-1))*np.diag(inverse)
            fvalue=beta**2/np.maximum(variance_beta,np.finfo(float).tiny)
            probability=stats.f.sf(fvalue,1,n-len(selected)-1)
            worst=int(np.argmax(probability))
            if probability[worst]<=cfg['p_remove']:break
            feature=selected.pop(worst);changed=True
            history.append(dict(step=step,action='remove',feature=feature,p_value=float(probability[worst]),n_selected=len(selected)))
        if not changed:break
        if len(selected)>=max_features:
            status='max_features_reached';break
    else:status='max_steps_reached'
    weights=np.zeros(p)
    if selected:
        design=np.column_stack([np.ones(n),z[:,selected]])
        coefficients,_,rank,_=np.linalg.lstsq(design,y,rcond=tol)
        weights[selected]=coefficients[1:]/scale[selected]
        intercept=float(coefficients[0]-mean@weights)
    else:
        rank=1;intercept=float(y.mean());status='intercept_only_no_selected_features'
    return dict(weights=weights,intercept=intercept,train_mean=mean,train_scale=scale,selected_features=selected,
                status=status,history=history,n_train=n,n_target=int(y.sum()),rank=int(rank),settings=cfg)


def score_swlda(model,x):
    return np.asarray(x,float)@model['weights']+model['intercept']


def window_features(epochs,fs,cfg):
    a=np.asarray(epochs,float)
    edges=np.ceil(np.arange(cfg['windows_per_channel']+1)*cfg['window_seconds']*fs-1e-10).astype(int)
    if a.ndim!=3 or a.shape[1]!=len(cfg['channels']) or edges[0]!=0 or edges[-1]!=a.shape[2] or np.any(np.diff(edges)<=0):
        raise ValueError('Epoch shape/sample grid does not match requested 16x16 features')
    means=np.stack([a[:,:,start:stop].mean(axis=2) for start,stop in zip(edges[:-1],edges[1:])],axis=2)*1e6
    return means.reshape(len(a),-1),edges


def auc_rank(y,score):
    y=np.asarray(y);score=np.asarray(score,float)
    known=np.isin(y,[0,1])&np.isfinite(score);y=y[known];score=score[known]
    n1=int((y==1).sum());n0=int((y==0).sum())
    if not n1 or not n0:return np.nan
    rank=stats.rankdata(score)
    return float((rank[y==1].sum()-n1*(n1+1)/2)/(n1*n0))


def decode_character(scores,flashed,cfg):
    """Prediction uses scores and stimulus membership only, never target labels."""
    scores=np.asarray(scores,float);flashed=np.asarray(flashed,bool)
    keep=np.isfinite(scores)
    score=scores[keep];active=flashed[keep]
    n_active=active.sum(axis=0);n_inactive=len(score)-n_active
    evidence=np.full(flashed.shape[1],np.nan)
    comparable=(n_active>=cfg['minimum_active_flashes_per_character'])&(n_inactive>=cfg['minimum_inactive_flashes_per_character'])
    for j in np.flatnonzero(comparable):
        evidence[j]=float(score[active[:,j]].mean()-score[~active[:,j]].mean())
    result=dict(predicted_index=None,status='insufficient_evidence',n_scored=len(score),n_active=n_active,n_inactive=n_inactive,evidence=evidence)
    if cfg['require_coverage_all_36_characters'] and not comparable.all():return result
    if not np.isfinite(evidence).any():return result
    best=float(np.nanmax(evidence));ties=np.flatnonzero(np.isclose(evidence,best,rtol=0,atol=cfg['tie_atol']))
    if len(ties)!=1:result['status']='evidence_tie';return result
    result.update(predicted_index=int(ties[0]),status='predicted')
    return result
