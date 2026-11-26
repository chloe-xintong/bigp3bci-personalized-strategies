"""Build two independently ordered, anonymous review packets; private key separate."""
from pathlib import Path
import argparse
import csv
import json
import hashlib
import shutil
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['true_blink', 'reverse_recovery', 'other_artifact', 'uncertain']


def allocate(counts, total):
    """Largest-remainder proportional allocation with finite stratum capacity."""
    raw = counts / counts.sum() * total
    result = np.floor(raw).astype(int)
    for key in (raw-result).sort_values(ascending=False, kind='stable').index[:total-int(result.sum())]:
        result.loc[key] += 1
    assert result.sum() == total and (result <= counts).all()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/double_blind_review_v1')
    args = parser.parse_args()
    out = args.output
    if out.exists():
        raise FileExistsError('Choose a new output directory; never overwrite reviewer answers.')
    rng = np.random.default_rng(20261125)
    source = ROOT/'outputs/unmatched_blink_context/all_unmatched_candidates.csv'
    all_rows = pd.read_csv(source).drop_duplicates(['edf_name', 'blink_candidate_id'])
    manifest = pd.read_csv(ROOT/'outputs/blink_annotation_update/annotation_manifest.csv').set_index('edf_name')
    inventories = []
    for edf, row in manifest.iterrows():
        b = pd.read_csv(ROOT/row.results_path/'blink_candidates.csv').sort_values(['onset_seconds', 'blink_candidate_id'])
        b['previous_automatic_candidate_id'] = b.blink_candidate_id.shift(1)
        b['previous_automatic_end_seconds'] = b.offset_seconds.shift(1)
        b['gap_from_previous_automatic_seconds'] = b.onset_seconds - b.previous_automatic_end_seconds
        inventories.append(b[['blink_candidate_id', 'previous_automatic_candidate_id', 'previous_automatic_end_seconds', 'gap_from_previous_automatic_seconds']])
    all_rows = all_rows.merge(pd.concat(inventories), on='blink_candidate_id', validate='one_to_one')
    pool = all_rows[~all_rows.filter_startup & ~all_rows.fully_within_complete_review & all_rows.saved_candidate_review_decision.isna()].copy()
    pool['session_id'] = pool.edf_name.str.split('_').str[2]
    pool['short_gap'] = pool.gap_from_previous_automatic_seconds.lt(.3)
    pool['stratum'] = pool.subject_id + '/' + pool.session_id + '/' + pool.short_gap.map({True:'short', False:'other'})
    assert len(pool) == 340 and pool.short_gap.sum() == 68
    # All short-gap and F_08 candidates are certainty strata. Remaining candidates
    # are sampled randomly within subject/session strata, with known inclusion rates.
    quotas = {'F_05': 91, 'F_25': 25, 'F_08': 4}
    sampled, strata = [], []
    for subject, quota in quotas.items():
        group = pool[pool.subject_id.eq(subject)]
        forced = group[group.short_gap | group.subject_id.eq('F_08')]
        rest = group.drop(forced.index)
        allocation = allocate(rest.groupby('stratum').size(), quota-len(forced)) if len(rest) else pd.Series(dtype=int)
        selected = forced.copy()
        for key, n in allocation.items():
            candidates = rest[rest.stratum.eq(key)]
            selected = pd.concat([selected, candidates.iloc[rng.choice(len(candidates), int(n), replace=False)]])
        for key, candidates in group.groupby('stratum', sort=True):
            count = int(selected.stratum.eq(key).sum())
            assert count > 0
            strata.append(dict(stratum=key, population_n=len(candidates), sampled_n=count,
                               inclusion_probability=count/len(candidates), analysis_weight=len(candidates)/count))
        sampled.append(selected)
    selected = pd.concat(sampled).sort_values(['edf_name','onset_seconds']).copy()
    selected['blind_id'] = [f'R{x:06d}' for x in rng.choice(np.arange(100000, 999999), len(selected), replace=False)]
    pilot_ids = set(rng.choice(selected.blind_id.to_numpy(), 30, replace=False))
    selected['phase'] = selected.blind_id.map(lambda x: 'pilot_30' if x in pilot_ids else 'remaining_90')
    selected = selected.merge(pd.DataFrame(strata), on='stratum', validate='many_to_one')
    selected['pilot_inclusion_probability'] = selected.inclusion_probability * .25
    assert len(selected) == 120 and selected.short_gap.sum() == 68
    private = out/'coordinator_only'; private.mkdir(parents=True)
    selected.to_csv(private/'sampling_key.csv', index=False, encoding='utf-8-sig')
    pool.to_csv(private/'eligible_population.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(strata).to_csv(private/'sampling_strata.csv', index=False)
    image_dir = private/'clips'; image_dir.mkdir()
    channels = ['EEG_F3','EEG_Fz','EEG_F4','EEG_Pz','EEG_PO7','EEG_PO8','EEG_Oz']
    for edf, group in selected.groupby('edf_name'):
        folder = ROOT/manifest.loc[edf,'results_path']
        with np.load(folder/'continuous_eeg.npz') as d:
            y = d['eeg_v']; names = d['channels'].tolist(); fs = float(d['sfreq'])
            for r in group.itertuples():
                center = int(round(r.peak_seconds*fs))
                start, stop = max(0, center-int(4*fs)), min(y.shape[1], center+int(4*fs)+1)
                time = (np.arange(start,stop)-center)/fs
                fig, ax = plt.subplots(figsize=(12,6), layout='constrained')
                for i, ch in enumerate(channels):
                    values = y[names.index(ch),start:stop]*1e6
                    ax.plot(time, values-np.median(values)+(len(channels)-i-1)*120, color='#17354a', lw=.8)
                ax.set_yticks(np.arange(len(channels))*120, [c.removeprefix('EEG_') for c in channels[::-1]])
                ax.axvline(0, color='#777777', lw=.8)
                ax.plot([3.5,3.5],[-80,-30],color='black',lw=2)
                ax.text(3.4,-55,'50 uV',ha='right',va='center')
                ax.set(xlim=(-4,4), xlabel='Time relative to review focus (s)', title=r.blind_id)
                ax.spines[['top','right']].set_visible(False)
                fig.savefig(image_dir/(r.blind_id+'.png'),dpi=110); plt.close(fig)
        print('Rendered',len(group),'anonymous clips',flush=True)
    instructions = '''Independent blink review

Review the event at time zero using the surrounding +/-4 seconds of EEG.
Traces show the existing causal analysis EEG, median-centered for display only.
All clips use the same channel order and scale. The vertical line is a review
focus, not a human annotation or a detector boundary. No event labels, prior
decisions, detector features, file identifiers or absolute times are shown.

Complete pilot_30 first. Do not discuss cases or see the other reviewer's
answers before both independent forms have been locked by the coordinator.
Use exactly one label:
true_blink: morphology and frontal distribution support a separate blink.
reverse_recovery: continuation/reversal of a preceding event rather than a separate blink.
other_artifact: visible change attributable to a different artifact morphology.
uncertain: the clip does not support a reliable distinction; do not guess.
These are provisional visual categories, not physiological ground truth.
Confidence: high / medium / low. Record a short morphological reason in notes.
Keep uncertainty as a valid category. Escalate insufficient-context cases to the
coordinator for a new anonymous clip, without exposing identity or prior labels.
Do not edit review IDs or order. Fill reviewer_code consistently.

After 30 paired reviews: compute four-category agreement and Cohen's kappa
before adjudication. Refine definitions if needed, then have both reviewers
independently re-review the pilot under the same final standard before pooling.
Continue to remaining_90. Adjudication is a separate later step and must not
overwrite either independent decision. Do not infer whole-detector precision
from this sample of unmatched, previously unreviewed candidates.
'''
    for reviewer in ['A','B']:
        dest = out/f'reviewer_{reviewer}'; (dest/'clips').mkdir(parents=True)
        order = pd.concat([selected[selected.phase.eq(phase)].iloc[rng.permutation(int(selected.phase.eq(phase).sum()))]
                           for phase in ['pilot_30','remaining_90']])
        form = pd.DataFrame(dict(order=np.arange(1,121),phase=order.phase.to_numpy(),blind_id=order.blind_id.to_numpy()))
        for column in ['reviewer_code','label','confidence','notes']: form[column]=''
        form.to_csv(dest/'review_form.csv', index=False, encoding='utf-8-sig')
        (dest/'instructions.txt').write_text(instructions, encoding='utf-8')
        html = ['<!doctype html><meta charset="utf-8"><title>Independent review</title><style>body{font:16px sans-serif;max-width:1250px;margin:auto}img{width:100%}article{border-bottom:1px solid #aaa;padding:20px}nav{position:sticky;top:0;background:white;padding:12px}</style><nav>Independent review: enter labels in review_form.csv. Review pilot_30 first.</nav>']
        for r in form.itertuples():
            shutil.copyfile(image_dir/(r.blind_id+'.png'), dest/'clips'/(r.blind_id+'.png'))
            html.append(f'<article><h2>{r.order}. {r.blind_id} / {r.phase}</h2><img src="clips/{r.blind_id}.png" alt="Anonymous EEG clip"></article>')
        (dest/'index.html').write_text('\n'.join(html), encoding='utf-8')
        with zipfile.ZipFile(out/f'reviewer_{reviewer}.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in dest.rglob('*'):
                if p.is_file(): z.write(p,p.relative_to(dest))
    record = dict(seed=20261125, population_n=len(pool),sampled_n=len(selected),pilot_n=30,
                  subject_quotas=quotas,short_gap_population=68,short_gap_selected=68,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),labels=LABELS,
                  scope='Unmatched nonstartup candidates with no saved decision, outside explicit complete review; three subjects only.',
                  analysis='Use stratum N/n weights for label proportions. Pilot is simple random 30 of 120. Kappa is descriptive for the enriched sample; unweighted kappa is not population-wide reliability.',
                  privacy='Share reviewer_A.zip and reviewer_B.zip separately; never share coordinator_only or the original candidate CSV with reviewers. Separate packets do not enforce access control on this shared filesystem.')
    (private/'sampling_record.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__': main()
