"""All Train flashes: candidate overlap counts and subject-cluster bootstrap OR."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/blink_label_association'


def summarize(frame, keys):
    rows = []
    for key, group in frame.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key))
        row['total_flashes'] = len(group)
        for label, prefix in [('target', 'target'), ('non-target', 'nontarget')]:
            subset = group[group.label.eq(label)]
            row[prefix + '_flashes'] = len(subset)
            row[prefix + '_blink_flashes'] = int(subset.has_blink_candidate.sum())
            row[prefix + '_blink_rate'] = float(subset.has_blink_candidate.mean())
        row['blink_flashes'] = int(group.has_blink_candidate.sum())
        row['blink_rate'] = float(group.has_blink_candidate.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    manifest = pd.read_csv(ROOT / 'outputs/studyf_all_preprocessing/input_manifest.csv')
    assert len(manifest) == 90
    frames, candidates, provenance = [], [], []
    for r in manifest.itertuples():
        folder = ROOT / r.results_path
        path = folder / 'epoch_candidates.csv'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == r.epoch_table_sha256
        e = pd.read_csv(path)
        b = pd.read_csv(folder / 'blink_candidates.csv')
        links = pd.read_csv(folder / 'blink_epoch_links.csv')
        assert e.event_index.is_unique
        assert e.label.isin(['target', 'non-target']).all()
        assert e.has_blink_candidate.eq(e.event_index.isin(links.event_index)).all()
        assert e.subject_id.eq(r.subject_id).all() and e.session_id.eq(r.session_id).all()
        frames.append(e)
        candidates.append(dict(subject_id=r.subject_id, session_id=r.session_id,
                               edf_name=r.edf_name, unique_candidate_events=len(b)))
        provenance.append(dict(edf_name=r.edf_name, epoch_sha256=r.epoch_table_sha256))
    flashes = pd.concat(frames, ignore_index=True)
    assert not flashes.duplicated(['edf_name', 'event_index']).any()
    assert flashes.groupby(['subject_id', 'session_id']).ngroups == 30
    subject = summarize(flashes, ['subject_id'])
    session = summarize(flashes, ['subject_id', 'session_id'])
    events = pd.DataFrame(candidates)
    subject = subject.merge(events.groupby('subject_id').unique_candidate_events.sum(), on='subject_id')
    session = session.merge(events.groupby(['subject_id', 'session_id']).unique_candidate_events.sum(), on=['subject_id', 'session_id'])
    # Mantel-Haenszel OR conditions on subject; bootstrap resamples whole subjects,
    # preserving all within-subject dependence from overlapping flash windows.
    a = subject.target_blink_flashes.to_numpy(float)
    b = subject.target_flashes.to_numpy(float) - a
    c = subject.nontarget_blink_flashes.to_numpy(float)
    d = subject.nontarget_flashes.to_numpy(float) - c
    n = a + b + c + d
    numerator, denominator = a * d / n, b * c / n
    rng = np.random.default_rng(20261125)
    ix = rng.integers(0, len(subject), size=(50000, len(subject)))
    den = denominator[ix].sum(axis=1)
    num = numerator[ix].sum(axis=1)
    valid = (den > 0) & (num > 0)
    ci = np.quantile(num[valid] / den[valid], [.025, .975])
    result = dict(total_flashes=len(flashes), subjects=len(subject), sessions=30,
                  target_flashes=int((flashes.label == 'target').sum()),
                  nontarget_flashes=int((flashes.label == 'non-target').sum()),
                  blink_flashes=int(flashes.has_blink_candidate.sum()),
                  unique_candidate_events=int(events.unique_candidate_events.sum()),
                  subject_stratified_mh_odds_ratio=float(numerator.sum()/denominator.sum()),
                  subject_cluster_bootstrap_percentile_ci95=ci.tolist(),
                  bootstrap_replicates=50000, finite_positive_replicates=int(valid.sum()), seed=20261125,
                  definition='Any old_candidate_overlap_v1 interval overlap with [flash onset, onset+ceil(0.8*fs)); all flashes, no QC rejection.',
                  limitations='Engineering candidates, not confirmed blinks. Only 10 subject clusters; sparse events and overlapping windows limit precision. Descriptive Train analysis, not a causal label effect or equivalence test.',
                  provenance=provenance)
    OUT.mkdir(parents=True, exist_ok=True)
    subject.to_csv(OUT / 'by_subject.csv', index=False)
    session.to_csv(OUT / 'by_subject_session.csv', index=False)
    events.to_csv(OUT / 'candidate_events_by_edf.csv', index=False)
    (OUT / 'association.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(subject.to_string(index=False))
    print(json.dumps({k:v for k,v in result.items() if k != 'provenance'}, indent=2))


if __name__ == '__main__':
    main()
