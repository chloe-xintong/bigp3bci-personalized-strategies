"""Session-wise Euclidean alignment (EA) for the exploratory sensitivity check.

EA maps every trial X (channels x samples) to R_s^(-1/2) X so that the
reference (mean trial) covariance of session s becomes the identity.  The 16x16
window features are per-channel window means, and a channel-space linear map
commutes with a temporal mean: mean_t(W X) = W mean_t(X).  Applying W to the
cached 16-channel vector of each window is therefore *exactly* equal to
recomputing window features from aligned epochs; callers verify this on raw
epochs at run time.
"""
from __future__ import annotations
import numpy as np


def trial_covariances(epochs_uv):
    """Per-trial covariance, temporal channel demeaning, ddof=1 (same as drift check)."""
    x=np.asarray(epochs_uv,float)
    if x.ndim!=3 or x.shape[2]<2 or not np.isfinite(x).all():raise ValueError('Need finite epochs shaped (n, channels, samples)')
    centered=x-x.mean(axis=2,keepdims=True)
    return np.einsum('nct,ndt->ncd',centered,centered)/(x.shape[2]-1)


def shrink(mean_covariance,shrinkage):
    """(1-a) R + a tr(R)/C I with a fixed a, guaranteeing a positive definite reference."""
    r=np.asarray(mean_covariance,float);c=len(r)
    if not 0<shrinkage<1:raise ValueError('Shrinkage must be fixed in (0, 1)')
    r=(r+r.T)/2;scale=np.trace(r)/c
    if not np.isfinite(scale) or scale<=0:raise ValueError('Reference covariance has no power')
    return (1-shrinkage)*r+shrinkage*scale*np.eye(c)


def reference_covariance(epochs_uv,shrinkage):
    """Arithmetic mean trial covariance plus fixed trace-identity shrinkage."""
    covariances=trial_covariances(epochs_uv)
    if not len(covariances):raise ValueError('No trials available for the reference covariance')
    return shrink(covariances.mean(axis=0),shrinkage)


def inverse_sqrt(reference):
    values,vectors=np.linalg.eigh((reference+reference.T)/2)
    if values.min()<=0:raise ValueError('Reference covariance is not positive definite')
    return (vectors*values**-.5)@vectors.T


def align_epochs(epochs_uv,whitener):
    return np.einsum('cd,ndt->nct',whitener,np.asarray(epochs_uv,float))


def align_features(features,whitener,n_channels=16,n_windows=16):
    """Apply W to channel-major (channel, window) features; exact for window means."""
    x=np.asarray(features,float)
    if x.ndim!=2 or x.shape[1]!=n_channels*n_windows or whitener.shape!=(n_channels,n_channels):
        raise ValueError('Feature layout must be channel-major channels x windows')
    blocks=x.reshape(len(x),n_channels,n_windows)
    return np.einsum('cd,ndw->ncw',whitener,blocks).reshape(len(x),-1)


def log_distance(a,b):
    """Normalized log-Euclidean distance, identical to the session drift check."""
    def logm(m):
        values,vectors=np.linalg.eigh((m+m.T)/2);return (vectors*np.log(values))@vectors.T
    raw=float(np.linalg.norm(logm(a)-logm(b),'fro'))
    return raw,raw/np.sqrt(len(a))


def direction_rule(deltas,minimum_same_sign):
    """Predeclared consistency label for one comparison at one k."""
    d=np.asarray(deltas,float)
    if not np.isfinite(d).all():raise ValueError('Every subject needs a paired difference')
    positive=int((d>0).sum());negative=int((d<0).sum());mean=float(d.mean())
    leading=positive if mean>0 else negative if mean<0 else 0
    return dict(n_positive=positive,n_negative=negative,n_zero=int(len(d)-positive-negative),
                direction='consistent' if leading>=minimum_same_sign else 'inconsistent_report_per_subject')


def bootstrap_deltas(table,value,subjects,k_values,replicates,seed,minimum_same_sign):
    """Subject-paired percentile bootstrap of the macro mean difference per k."""
    subjects=sorted(subjects)
    indices=np.random.default_rng(seed).integers(0,len(subjects),size=(replicates,len(subjects)))
    rows=[]
    for k in k_values:
        g=table[table.k.eq(k)].set_index('subject_id').reindex(subjects)[value].to_numpy(float)
        if not np.isfinite(g).all():raise ValueError(f'Missing paired difference for {value} at k={k}')
        low,high=np.quantile(g[indices].mean(axis=1),[.025,.975])
        rows.append(dict(k=k,comparison=value,subjects=len(subjects),mean_delta=float(g.mean()),
                         ci_low=float(low),ci_high=float(high),ci_excludes_zero=bool(low>0 or high<0),
                         **direction_rule(g,minimum_same_sign)))
    return rows
