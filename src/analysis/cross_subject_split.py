"""Label-independent leave-one-subject donor/calibration/validation plans."""
from pathlib import PureWindowsPath
import numpy as np
from bigp3bci_calibration_budget import make_plan,training_keys
from bigp3bci_data_policy import FORBIDDEN_PARTS


def make_splits(trials,cfg):
    if set(trials.subject_id)!=set(cfg['subjects']) or not trials.character_key.is_unique:
        raise ValueError('Expected all ten subjects and unique character keys')
    for name in trials.edf_name:
        parts=PureWindowsPath(name).parts
        if FORBIDDEN_PARTS.intersection(parts) or '_Train' not in name or not name.endswith('.edf'):
            raise ValueError('Only Train character records are allowed')
    plans={s:make_plan(trials[trials.subject_id.eq(s)],cfg['history_sessions'],cfg['current_session']) for s in cfg['subjects']}
    for s,p in plans.items():
        if (len(p['history']),len(p['calibration']),len(p['validation']))!=(36,12,6):
            raise ValueError('Unexpected character counts')
        ordered=trials[trials.subject_id.eq(s)&trials.session_id.eq(cfg['current_session'])].sort_values(['edf_name','character_trial_id'])
        if ordered.character_key.tolist()!=p['calibration']+p['validation']:
            raise ValueError('Current split is not chronological first12/last6')
    result={}
    indexed=trials.set_index('character_key')
    for s,p in plans.items():
        donors=[key for other in cfg['subjects'] if other!=s for key in plans[other]['history']]
        calibration=training_keys(p,'current_only',12);validation=p['validation']
        if len(donors)!=324 or indexed.loc[donors].subject_id.eq(s).any():
            raise ValueError('Target subject in donor training pool')
        if not indexed.loc[donors].session_id.isin(cfg['history_sessions']).all():
            raise ValueError('Donor current session in training')
        groups=[set(donors),set(calibration),set(validation)]
        if any(groups[i]&groups[j] for i in range(3) for j in range(i)):
            raise ValueError('Overlapping donor/calibration/validation keys')
        result[s]=dict(donors=donors,calibration=calibration,validation=validation,plan=p)
    return result


def condition_keys(split,condition,k):
    if condition not in ('X','C') or k not in (0,3,6,12) or (condition=='C' and k==0):
        raise ValueError('Unknown frozen condition')
    current=training_keys(split['plan'],'current_only',k)
    return (split['donors'] if condition=='X' else [])+current


def assert_control_equal(actual,expected,atol=1e-10):
    a=np.asarray(actual,float);b=np.asarray(expected,float)
    if a.shape!=b.shape or not np.allclose(a,b,atol=atol,rtol=0,equal_nan=True):
        raise ValueError('Current-only control differs from the frozen budget experiment')


def bootstrap_mean(values,indices):
    values=np.asarray(values,float)
    if values.shape!=(10,) or not np.isfinite(values).all():
        raise ValueError('All ten subjects must have finite metrics')
    low,high=np.quantile(values[indices].mean(axis=1),[.025,.975])
    return float(values.mean()),float(low),float(high)
