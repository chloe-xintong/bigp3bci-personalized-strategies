"""Character-budget plans; no labels, features or held-out outcomes are used."""


def make_plan(trials, history_sessions, current_session):
    if current_session in history_sessions:
        raise ValueError('Current session cannot also be historical')
    if trials.subject_id.nunique() != 1:
        raise ValueError('Build a separate plan for each subject')
    ordered=trials.sort_values(['session_id','edf_name','character_trial_id'])
    if not ordered.character_key.is_unique:
        raise ValueError('Duplicate character IDs')
    historical=ordered[ordered.session_id.isin(history_sessions)]
    current=ordered[ordered.session_id.eq(current_session)]
    if set(historical.session_id)!=set(history_sessions):
        raise ValueError('A historical session is missing')
    calibration=current[current['set'].eq('train')]
    validation=current[current['set'].eq('validation')]
    if len(calibration)+len(validation)!=len(current) or not len(calibration) or not len(validation):
        raise ValueError('Current session needs a fixed calibration/validation partition')
    if calibration.acquisition_order.max()>=validation.acquisition_order.min():
        raise ValueError('Calibration must precede validation')
    return dict(history=historical.character_key.tolist(),
                history_by_session={s:historical[historical.session_id.eq(s)].character_key.tolist() for s in history_sessions},
                calibration=calibration.character_key.tolist(),
                validation=validation.character_key.tolist())


def training_keys(plan, scheme, k):
    if k<0 or k>len(plan['calibration']):
        raise ValueError('Current-character budget out of range')
    if scheme=='current_only':
        selected=plan['calibration'][:k]
    elif scheme=='history_plus_current':
        selected=plan['history']+plan['calibration'][:k]
    elif scheme in ('history_se001_plus_current','history_se002_plus_current'):
        session='SE001' if scheme=='history_se001_plus_current' else 'SE002'
        selected=plan['history_by_session'][session]+plan['calibration'][:k]
    elif scheme=='history_only':
        if k!=0:raise ValueError('History-only endpoint has k=0')
        selected=plan['history']
    elif scheme in ('history_se001_only','history_se002_only'):
        if k!=0:raise ValueError('Single-session history baseline has k=0')
        session='SE001' if scheme=='history_se001_only' else 'SE002'
        selected=plan['history_by_session'][session]
    elif scheme=='current_max_endpoint':
        if k!=len(plan['calibration']):raise ValueError('Current endpoint must use max k')
        selected=plan['calibration']
    elif scheme=='all_available_reference':
        if k!=len(plan['calibration']):raise ValueError('Reference uses all allowed calibration characters')
        selected=plan['history']+plan['calibration']
    else:
        raise ValueError('Unknown strategy')
    if set(selected)&set(plan['validation']):
        raise ValueError('Validation character leakage into training')
    return list(selected)
