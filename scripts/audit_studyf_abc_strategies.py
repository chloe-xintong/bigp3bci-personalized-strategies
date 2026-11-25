"""Independent audit of the StudyF A/B/C rejection-strategy experiment (v3).

Reads raw EDFs (read-only, hash-checked) and saved v3/preprocessing outputs; writes ONLY to
outputs/audit_studyf_abc_v1. Re-derives the split, re-fits every SWLDA model from training
rows, re-decodes every validation character, and runs paired / subject-cluster statistics and
the train-only / validation-only rule decomposition.
"""
from pathlib import Path
import os, sys, json, re, inspect
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('_MNE_FAKE_HOME_DIR', str(ROOT)); sys.path.insert(0, str(ROOT / 'src'))
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ[k] = '1'
import numpy as np
import pandas as pd
from scipy import stats
from bigp3bci_swlda import fit_swlda, score_swlda, decode_character
from bigp3bci_preprocessing import causal_filter, design_filter, file_hash
from bigp3bci_expanded_pilot import detect_blinks
import bigp3bci_policy_comparison, bigp3bci_swlda
from bigp3bci_continuous_qc import read_bigp3bci_edf

V3 = ROOT / 'outputs/swlda_all_studyf_v3'
PRE = ROOT / 'outputs/studyf_all_preprocessing'
OUT = ROOT / 'outputs/audit_studyf_abc_v1'
OUT.mkdir(parents=True, exist_ok=True)
cfg = json.loads((V3 / 'config_snapshot.json').read_text())
pre_cfg = json.loads((ROOT / 'configs/expanded_pilot.json').read_text())
pol_cfg = json.loads((ROOT / 'configs/three_policy_comparison.json').read_text())
POL = ['A', 'B', 'C']
checks = []


def check(section, item, status, evidence):
    checks.append(dict(section=section, item=item, status=status, evidence=evidence))
    print(f'[{status}] {section} {item}: {evidence}', flush=True)


# ------------------------------------------------------------------ load
split = pd.read_csv(V3 / 'character_split.csv')
esplit = pd.read_csv(V3 / 'epoch_split.csv')
scores = pd.read_csv(V3 / 'epoch_scores.csv', low_memory=False)
pred_saved = pd.read_csv(V3 / 'character_predictions.csv')
manifest = pd.read_csv(PRE / 'input_manifest.csv')
with np.load(V3 / 'features_and_membership.npz') as z:
    X = z['features_uv']; FLASHED = z['flashed']; CHARS = z['characters'].tolist()
src_hash_before = {r.edf_name: file_hash(Path(json.loads((ROOT / r.results_path / 'processing_record.json').read_text())['source_edf'])) for r in manifest.itertuples()}

# ================================================================== I. split
s = split.copy()
s['train_no'] = s.edf_name.str.extract(r'Train(\d+)')[0].astype(int)
re_rows = []
for mid, g in s.groupby('model_id'):
    g = g.sort_values(['edf_name', 'character_trial_id'])
    n = len(g); ntr = int(np.floor(n * cfg['split']['train_fraction']))
    re_rows.append(pd.Series(['train'] * ntr + ['validation'] * (n - ntr), index=g.index))
s['set_recomputed'] = pd.concat(re_rows)
lex_eq_num = all((g.drop_duplicates('edf_name').sort_values('edf_name').train_no.diff().dropna() > 0).all() for _, g in s.groupby('model_id'))
per_file = s.groupby(['model_id', 'edf_name']).set.nunique()
val_pos = s[s.set.eq('validation')].groupby('model_id').edf_name.nunique()
check('I.1', 'split rule', 'pass' if (s.set == s.set_recomputed).all() else 'problem',
      'Chronological, not random: scripts/run_swlda_rejection_experiment.py:100-110 sorts by (edf_name, character_trial_id) and takes '
      'the first floor(2n/3) characters as train. Recomputed split equals saved character_split.csv for all 540 characters. '
      f'Every session has 18 characters -> 12 train / 6 validation; no EDF is split ({int((per_file > 1).sum())} mixed files); '
      f'validation = last EDF of each session ({int((val_pos == 1).sum())}/30). Lexical order == Train number order: {lex_eq_num}.')
dates = pd.read_csv(Path(os.environ.get('MEAS_DATES', OUT / 'edf_meas_dates.csv'))) if (OUT / 'edf_meas_dates.csv').exists() else None
if dates is None:
    import mne
    rows = []
    for r in pd.read_csv(PRE / 'selection_manifest.csv').itertuples():
        raw = mne.io.read_raw_edf(r.path, preload=False, verbose='error')
        rows.append(dict(edf_name=r.edf_name, meas_date=str(raw.info['meas_date'])))
    dates = pd.DataFrame(rows); dates.to_csv(OUT / 'edf_meas_dates.csv', index=False)
check('I.1', 'chronology of EDF order', 'unconfirmed',
      f'EDF header meas_date has {dates.meas_date.nunique()} distinct value(s) ({dates.meas_date.iloc[0]}): anonymised. '
      'That Train01<Train02<Train03 is acquisition order is inferred from file numbering only; needs acquisition logs/BCI2000 .dat timestamps.')

# I.2 every flash of a character on one side
e = esplit.merge(split[['character_key', 'set', 'flashes']].rename(columns={'set': 'set_char'}), on='character_key', validate='many_to_one')
sides = e.groupby('character_key').set.nunique().max()
counts_ok = (e.groupby('character_key').size() == split.set_index('character_key').flashes).all()
same_file = (e.groupby('character_key').edf_name.nunique() == 1).all()
check('I.2', 'flashes of a character not split', 'pass' if sides == 1 and (e.set == e.set_char).all() and counts_ok and same_file else 'problem',
      f'epoch_split.csv: max sides per character = {sides}; flash set == character set for all {len(e)} flashes; '
      f'flash count per character matches split ({counts_ok}); all flashes of a character in one EDF ({same_file}). '
      'Enforced in code at run_swlda_rejection_experiment.py:112 (many_to_one merge) and :131.')

# I.3 final Test data
paths = [json.loads((ROOT / r.results_path / 'processing_record.json').read_text())['source_edf'] for r in manifest.itertuples()]
n_train_path = sum(Path(p).parts[-3] == 'Train' for p in paths)
raw_insp = ROOT / 'outputs/raw_inspection'
eeg_stats_test = sum('Test' in open(f, encoding='utf-8-sig').read() for f in raw_insp.glob('StudyF_Train_CBCol_EEG_*.csv'))
check('I.3', 'final Test data', 'pass' if n_train_path == 90 and eeg_stats_test == 0 else 'problem',
      f'{n_train_path}/90 source EDFs are under .../Train/ (guard at run_swlda_rejection_experiment.py:41). '
      'Preprocessing, detector, PTP threshold and SWLDA configs are literal constants. Test files appear only in notebook 02 '
      '(outputs/raw_inspection: StimulusCode / character-count / feedback-channel event audits); EEG signal statistics there are '
      f'Train-only ({eeg_stats_test} Test rows). No step selects anything from Test EEG.')

# I.4 tables
fl = esplit.groupby(['character_key']).size().rename('collected_flashes')
kept = scores.groupby(['character_key', 'policy']).would_keep.sum().unstack().add_prefix('kept_flashes_')
chars = split.set_index('character_key').join(fl).join(kept).reset_index()
chars['label'] = 'T' + chars.character_trial_id.astype(str).str.zfill(2) + ':' + chars.true_character.str.split('_').str[0]
sess = []
for mid, g in chars.groupby('model_id'):
    t = g[g.set.eq('train')]; v = g[g.set.eq('validation')]
    sess.append(dict(model_id=mid, subject_id=g.subject_id.iloc[0], session_id=g.session_id.iloc[0],
                     train_edfs=';'.join(sorted(t.edf_name.unique())), validation_edfs=';'.join(sorted(v.edf_name.unique())),
                     n_train_characters=len(t), n_validation_characters=len(v),
                     train_characters=' '.join(t.edf_name.str.extract(r'(Train\d+)')[0] + '/' + t.label),
                     validation_characters=' '.join(v.edf_name.str.extract(r'(Train\d+)')[0] + '/' + v.label),
                     train_flashes=int(t.collected_flashes.sum()), validation_flashes=int(v.collected_flashes.sum()),
                     **{f'train_kept_{p}': int(t[f'kept_flashes_{p}'].sum()) for p in POL},
                     **{f'validation_kept_{p}': int(v[f'kept_flashes_{p}'].sum()) for p in POL}))
sess = pd.DataFrame(sess); sess.to_csv(OUT / 'I4_session_split.csv', index=False)
chars.drop(columns=['label']).to_csv(OUT / 'I4_character_split_detail.csv', index=False)
subj_val = chars[chars.set.eq('validation')].groupby('subject_id').size().rename('validation_characters').reset_index()
subj_val.to_csv(OUT / 'I4_subject_validation_counts.csv', index=False)
check('I.4', 'validation characters per subject', 'pass' if (subj_val.validation_characters == 18).all() and len(subj_val) == 10 else 'problem',
      f'{len(subj_val)} subjects, counts = {sorted(set(subj_val.validation_characters))}; see I4_session_split.csv / I4_subject_validation_counts.csv')

# ================================================================== II. leakage
# Refit every model from training rows only and compare with saved models / scores.
lab = scores.drop_duplicates(['edf_name', 'event_index']).set_index('feature_row')
y_all = lab.label.map({'non-target': 0, 'target': 1}).reindex(np.arange(len(X))).to_numpy(float)
finite = np.isfinite(X).all(axis=1)
keep = {p: scores[scores.policy.eq(p)].set_index('feature_row').would_keep.reindex(np.arange(len(X))).to_numpy(bool) for p in POL}
avail = {p: scores[scores.policy.eq(p)].set_index('feature_row').decision_data_available.reindex(np.arange(len(X))).to_numpy(bool) for p in POL}
setv = lab.set.reindex(np.arange(len(X))).to_numpy()
mid_v = lab.model_id.reindex(np.arange(len(X))).to_numpy()
models = {}; std_rows = []; sel_rows = []
for mid in sorted(set(mid_v)):
    rows = np.flatnonzero(mid_v == mid)
    for p in POL:
        tr = rows[(setv[rows] == 'train') & keep[p][rows] & finite[rows]]
        m = fit_swlda(X[tr], y_all[tr], cfg['swlda']); models[(mid, p)] = m
        with np.load(V3 / 'models' / f'{mid}_{p}.npz') as sv:
            allrows = rows[finite[rows] & keep[p][rows]]
            std_rows.append(dict(model_id=mid, policy=p, n_train_rows=len(tr),
                                 saved_mean_equals_train_mean=bool(np.allclose(sv['train_mean'], X[tr].mean(0), rtol=0, atol=1e-9)),
                                 saved_scale_equals_train_std=bool(np.allclose(sv['train_scale'], np.where(X[tr].std(0) > 1e-12, X[tr].std(0), 1), rtol=0, atol=1e-9)),
                                 max_abs_diff_vs_train_plus_validation_mean=float(np.abs(sv['train_mean'] - X[allrows].mean(0)).max())))
            sel_rows.append(dict(model_id=mid, policy=p, n_selected=len(m['selected_features']), status=m['status'],
                                 refit_selection_identical=sv['selected_features'].tolist() == m['selected_features'],
                                 refit_weights_identical=bool(np.allclose(sv['weights_uv'], m['weights'], rtol=1e-9, atol=1e-12)),
                                 n_selected_le_60=len(m['selected_features']) <= 60))
std_t = pd.DataFrame(std_rows); sel_t = pd.DataFrame(sel_rows)
std_t.to_csv(OUT / 'II1_standardization_check.csv', index=False); sel_t.to_csv(OUT / 'II2_feature_selection_check.csv', index=False)
check('II.1', 'standardization', 'pass' if std_t.saved_mean_equals_train_mean.all() and std_t.saved_scale_equals_train_std.all() else 'problem',
      'src/bigp3bci_swlda.py:18-20 mean/std of the rows passed to fit_swlda; caller passes only set==train & would_keep & finite rows '
      '(run_swlda_rejection_experiment.py:170-177). All 90 saved train_mean/train_scale equal training-row statistics '
      f'(median |diff| vs train+validation mean = {std_t.max_abs_diff_vs_train_plus_validation_mean.median():.3f} uV, i.e. validation not included).')
check('II.2', 'SWLDA feature selection', 'pass' if sel_t.refit_selection_identical.all() and sel_t.refit_weights_identical.all() and sel_t.n_selected_le_60.all() else 'problem',
      f'Stepwise p_enter={cfg["swlda"]["p_enter"]}, p_remove={cfg["swlda"]["p_remove"]}, max_features={cfg["swlda"]["max_features"]} are config constants '
      '(src/bigp3bci_swlda.py:53,62,66). Independent refit on training rows reproduces selected features and weights for '
      f'{int(sel_t.refit_selection_identical.sum())}/90 models; max selected = {sel_t.n_selected.max()}.')

# II.3 B threshold
ep = pd.concat([pd.read_csv(ROOT / r.results_path / 'epoch_candidates.csv', usecols=['event_index', 'max_ptp_uv', 'onset_sample', 'filter_startup', 'has_blink_candidate', 'complete_epoch']).assign(edf_name=r.edf_name) for r in manifest.itertuples()])
w = scores.pivot_table(index=['edf_name', 'event_index'], columns='policy', values='would_keep', aggfunc='first').reset_index().merge(ep, on=['edf_name', 'event_index'])
b_ok = (w.B == (w.A & ~(w.max_ptp_uv > 150.0))).all()
check('II.3', 'B 150 uV threshold', 'pass' if b_ok and pol_cfg['ptp_sensitivity_upper_uv'] == [150.0] else 'problem',
      'Literal 150.0 in configs/three_policy_comparison.json ("ptp_sensitivity_upper_uv"), applied in src/bigp3bci_policy_comparison.py:35 '
      'as max over 16 channels of filtered epoch PTP > 150 uV. Not estimated from data. Recomputed B == A & ~(max_ptp>150) for all '
      f'{len(w)} epochs: {b_ok}.')

# II.4 C detector: constants, overlap recomputation, truncation (no whole-record estimation)
recs = [json.loads((ROOT / r.results_path / 'processing_record.json').read_text()) for r in manifest.itertuples()]
same_det = all(r['config']['blink_detector'] == cfg['blink_detector'] for r in recs)
size = int(np.ceil(0.8 * 256))
rec_mismatch = 0
for r in manifest.itertuples():
    bc = pd.read_csv(ROOT / r.results_path / 'blink_candidates.csv')
    g = w[w.edf_name.eq(r.edf_name)]
    hit = np.array([((bc.onset_sample < o + size) & (bc.offset_sample_exclusive > o)).any() for o in g.onset_sample])
    rec_mismatch += int((hit != g.has_blink_candidate.to_numpy()).sum())
c_ok = (w.C == (w.A & ~w.has_blink_candidate)).all()
# truncation experiment on one validation EDF: candidates confirmed before the cut must not change
val_edf = 'F_24_SE003_CBCol_Train03.edf'
mrow = manifest.set_index('edf_name').loc[val_edf]
rec = json.loads((ROOT / mrow.results_path / 'processing_record.json').read_text())
raw = read_bigp3bci_edf(Path(rec['source_edf'])); xraw = raw.get_data(picks=rec['channels']); raw.close()
sos = design_filter(rec['sfreq'], pre_cfg['filter'])
y_full, cont_full, _ = causal_filter(xraw, sos, rec['sfreq'], pre_cfg['filter'])
with np.load(ROOT / mrow.results_path / 'continuous_eeg.npz') as zz: saved_y = zz['eeg_v']
cut = xraw.shape[1] // 2
y_half, _, _ = causal_filter(xraw[:, :cut], sos, rec['sfreq'], pre_cfg['filter'])
filt_prefix_equal = bool(np.array_equal(y_full[:, :cut], y_half))
filt_reproduces = bool(np.allclose(y_full, saved_y, rtol=0, atol=1e-15))
bl_full = detect_blinks(y_full, rec['sfreq'], rec['channels'], pre_cfg['blink_detector'], pre_cfg['filter'], val_edf, cont_full)[0]
bl_half = detect_blinks(y_half, rec['sfreq'], rec['channels'], pre_cfg['blink_detector'], pre_cfg['filter'], val_edf, cont_full[:, :cut])[0]
cols = ['onset_sample', 'peak_sample', 'offset_sample_exclusive', 'decision_sample']
early = bl_full[bl_full.decision_sample < cut - 1][cols].reset_index(drop=True)
trunc_equal = early.equals(bl_half[bl_half.decision_sample < cut - 1][cols].reset_index(drop=True))
check('II.4', 'C blink detector thresholds', 'pass' if same_det and c_ok and rec_mismatch == 0 and trunc_equal else 'problem',
      'All thresholds are literal constants (configs/expanded_pilot.json "blink_detector"; src/bigp3bci_expanded_pilot.py:189-266); per-sample median '
      'over F3/Fz/F4 and segment-local correlation only, no whole-record percentile/STD. Identical detector config in all 90 processing records: '
      f'{same_det}. Recomputed epoch/candidate overlap mismatches: {rec_mismatch}. C == A & ~has_blink_candidate: {c_ok}. '
      f'Truncation test on {val_edf}: {len(early)} candidates confirmed before the cut are identical when the file is cut in half: {trunc_equal}.')
check('II.4b', 'detector development reused current validation EDFs', 'problem (non-label, limited)',
      'Manual blink annotations (outputs/artifact_qc/continuous_manual_qc/manual/manual_annotations) and the 12-file detector evaluation '
      '(outputs/three_policy_comparison, outputs/blink_annotation_update) include F_05_SE003_CBCol_Train04 and F_08_SE003_CBCol_Train03, '
      'which are now validation EDFs (12/180 validation characters). Annotations are of blinks, not targets, and '
      'outputs/blink_validation/analysis_record.json records thresholds_changed=false, but parameter choices were made after viewing these '
      'files. Subjects F_03/F_05/F_06/F_08/F_20/F_25 were all used in development; F_07/F_21/F_23/F_24 were not. See V_sensitivity.csv.')

# II.5 startup
st_ok = (ep.filter_startup == (ep.onset_sample < round(256 * pre_cfg['filter']['startup_seconds']))).all()
check('II.5', 'startup exclusion', 'pass' if st_ok else 'problem',
      'Fixed 10 s from the start of each EDF (src/bigp3bci_expanded_pilot.py:301, config filter.startup_seconds=10); recomputed '
      f'filter_startup == onset_sample < 2560 for all epochs: {st_ok}. Uses no labels and no validation information; identical for A/B/C.')

# II.6 filter
check('II.6', 'causal filtering', 'pass' if filt_prefix_equal and filt_reproduces else 'problem',
      'iirnotch 60 Hz Q=30 + Butterworth order-4 0.5-30 Hz bandpass (src/bigp3bci_preprocessing.py:16-22) applied with scipy.signal.sosfilt '
      'forward-only, state carried across 1 s chunks (:47-49), initial state = steady state of first sample; no filtfilt anywhere in the pipeline. '
      f'On {val_edf}: refiltering reproduces saved signal ({filt_reproduces}); filtering only the first half gives a bit-identical prefix '
      f'({filt_prefix_equal}). Filter state is reset per EDF and train/validation are always different EDFs, so filtering never crosses the boundary.')

# II.7 decoding
sig = list(inspect.signature(decode_character).parameters)
# Dynamic test: permute every truth column of each validation EDF and re-run the policy rule.
truth_cols = ['label', 'current_target', 'stimulus_type_raw', 'stimulus_type_digital', 'stimulus_type_physical', 'stimulus_code', 'event_code']
perm_diff = 0; n_files = 0
for r in manifest[manifest.edf_name.isin(set(split.loc[split.set.eq('validation'), 'edf_name']))].itertuples():
    folder = ROOT / r.results_path; rc = json.loads((folder / 'processing_record.json').read_text())
    ev = pd.read_csv(folder / 'epoch_candidates.csv', keep_default_na=False)
    ql = pd.read_csv(folder / 'raw_quality_epoch_links.csv', keep_default_na=False); bl = pd.read_csv(folder / 'blink_epoch_links.csv', keep_default_na=False)
    lg = pd.read_csv(folder / 'epoch_ptp_long.csv')
    ptp = lg.pivot(index='event_index', columns='channel', values='ptp_uv').reindex(index=ev.event_index, columns=rc['channels']).to_numpy()
    args = (rc['channels'], rc['sfreq'], rc['n_samples'], rc['config']['blink_detector'], pol_cfg)
    a0 = bigp3bci_policy_comparison.evaluate_policies(ev, ptp, ql, bl, *args)
    evp = ev.copy(); prng = np.random.default_rng(n_files)
    for c in truth_cols:
        if c in evp: evp[c] = prng.permutation(evp[c].to_numpy())
    a1 = bigp3bci_policy_comparison.evaluate_policies(evp, ptp, ql, bl, *args)
    perm_diff += int((a0.would_keep.to_numpy() != a1.would_keep.to_numpy()).sum() + (a0.policy_decision_seconds.to_numpy() != a1.policy_decision_seconds.to_numpy()).sum())
    n_files += 1
check('II.7', 'decoder / retention use no validation truth', 'pass' if sig == ['scores', 'flashed', 'cfg'] and perm_diff == 0 else 'problem',
      'decode_character(scores, flashed, cfg) (src/bigp3bci_swlda.py:103-119) sees only SWLDA scores and character-flash membership. '
      'would_keep is fixed during preprocessing by evaluate_policies (src/bigp3bci_policy_comparison.py:8-50) before any model is fit; '
      f'permuting {truth_cols} in all {n_files} validation EDFs changes {perm_diff} keep/decision values. The SWLDA script only reads '
      'would_keep (run_swlda_rejection_experiment.py:28); no accuracy feeds back into retention. Re-decoding reproduces saved predictions (check V).')

# ================================================================== III. C version
new_mods = ['bigp3bci_blink_boundaries', 'bigp3bci_bounded_prominence', 'bigp3bci_plateau_metrics', 'bigp3bci_candidate_review', 'bigp3bci_boundary_review']
chain = ['scripts/run_all_studyf_preprocessing.py', 'scripts/run_swlda_rejection_experiment.py', 'src/bigp3bci_expanded_pilot.py',
         'src/bigp3bci_policy_comparison.py', 'src/bigp3bci_preprocessing.py', 'src/bigp3bci_continuous_qc.py', 'src/bigp3bci_swlda.py']
imports = {f: [m for m in new_mods if m in (ROOT / f).read_text(encoding='utf8')] for f in chain}
check('III.1', 'C = old_candidate_overlap_v1', 'pass' if cfg['blink_interval_version'] == 'old_candidate_overlap_v1' and c_ok else 'problem',
      'config_snapshot blink_interval_version=old_candidate_overlap_v1; definition in III_C_definition.json.')
check('III.2', 'new main-wave / prominence logic not mixed in', 'pass' if not any(imports.values()) and c_ok and rec_mismatch == 0 else 'problem',
      f'None of {new_mods} is imported by the v3 chain {chain}. C keep mask equals A & ~overlap(blink_candidates.csv from detect_blinks) exactly.')
guard = int(np.ceil(1.2 * 256)) + max(int(np.ceil(0.08 * 256)), int(np.ceil(0.15 * 256)))
(OUT / 'III_C_definition.json').write_text(json.dumps(dict(
    version='old_candidate_overlap_v1', detector='src/bigp3bci_expanded_pilot.py:189-266 detect_blinks', detector_parameters=cfg['blink_detector'],
    detector_input='causally filtered EEG (60 Hz notch + 0.5-30 Hz), then causal Butterworth order-2 5 Hz lowpass auxiliary branch',
    algorithm=['common = per-sample median of F3/Fz/F4 auxiliary signal (uV)',
               'hysteresis: enter when |common|>=20 uV, exit after |common|<=10 uV for recovery_seconds (0.08 s)',
               'merge segments separated by <=0.15 s', 'reject if duration outside [0.08,1.2] s, median |frontal peak|<25 uV, '
               '<2 frontal channels with same-sign peak>=0.5*amp and corr>=0.6, slow/official PTP fraction<0.35, overlaps non-finite '
               'recovery, or peak <0.35 s after previous accepted peak, or unconfirmed at EOF', 'frontal-dominance ratio recorded but not required'],
    interval='[onset_sample, offset_sample_exclusive) of the auxiliary-signal excursion (no padding)',
    epoch_rule='C rejects an epoch iff A keeps it AND any positive sample overlap with a candidate interval in [onset, onset+205)',
    online_timing=f'policy decision = max(epoch decision, epoch stop + {guard} samples ({guard / 256:.4f} s)); src/bigp3bci_policy_comparison.py:13-14,39',
    not_used=['main-wave boundary localisation (blink_boundary_localization_v1)', 'bounded prominence secondary confirmation', 'manual annotations']), indent=2), encoding='utf8')

# ================================================================== IV/V. re-decode all combinations
val_chars = split[split.set.eq('validation')].sort_values(['model_id', 'edf_name', 'character_trial_id'])
onset = lab.onset_sample.reindex(np.arange(len(X))).to_numpy()
ck = lab.character_key.reindex(np.arange(len(X))).to_numpy()
rows_by_char = {k: np.flatnonzero(ck == k) for k in val_chars.character_key}
score_cache = {(mid, p): score_swlda(m, np.nan_to_num(X)) for (mid, p), m in models.items()} if False else {}
res = []
for (mid, ptrain), m in models.items():
    for pval in POL:
        for t in val_chars[val_chars.model_id.eq(mid)].itertuples():
            r = rows_by_char[t.character_key]; r = r[np.argsort(onset[r], kind='stable')]
            sc = np.full(len(r), np.nan); ok = keep[pval][r] & finite[r]
            sc[ok] = score_swlda(m, X[r[ok]])
            d = decode_character(sc, FLASHED[r], cfg['character_evidence'])
            idx = d['predicted_index'] if avail[pval][r].all() else None
            res.append(dict(train_rule=ptrain, validation_rule=pval, model_id=mid, subject_id=t.subject_id, session_id=t.session_id,
                            edf_name=t.edf_name, character_key=t.character_key, character_trial_id=t.character_trial_id,
                            true_character=t.true_character, predicted_character=CHARS[idx] if idx is not None else '',
                            no_output=idx is None, correct=idx is not None and idx == t.true_character_index,
                            valid_flashes=int(keep[pval][r].sum())))
res = pd.DataFrame(res)
same = res[res.train_rule == res.validation_rule].rename(columns={'train_rule': 'policy'})
cmp_ = same.merge(pred_saved[['policy', 'character_key', 'predicted_character', 'correct']].rename(columns={'predicted_character': 'saved_pred', 'correct': 'saved_correct'}), on=['policy', 'character_key'])
cmp_['saved_pred'] = cmp_.saved_pred.fillna('')
repro = bool((cmp_.predicted_character == cmp_.saved_pred).all() and (cmp_.correct == cmp_.saved_correct).all() and len(cmp_) == 540)
check('V', 'reproduction of saved predictions', 'pass' if repro else 'problem',
      f'Independent refit + re-decode reproduces all {len(cmp_)} saved (policy, character) predictions: {repro}.')

# IV.1 per-character table
pc = same[['policy', 'subject_id', 'session_id', 'edf_name', 'character_trial_id', 'character_key', 'true_character', 'predicted_character', 'no_output', 'correct', 'valid_flashes']].copy()
pc['true_character'] = pc.true_character.str.split('_').str[0]; pc['predicted_character'] = pc.predicted_character.str.split('_').str[0]
pc.sort_values(['policy', 'subject_id', 'session_id', 'edf_name', 'character_trial_id']).to_csv(OUT / 'IV1_per_character_results.csv', index=False)


def summarize(df, by):
    g = df.groupby(by)
    out = g.agg(characters=('correct', 'size'), correct=('correct', 'sum'), no_output=('no_output', 'sum')).reset_index()
    out['accuracy_all_characters'] = out.correct / out.characters
    out['output_characters'] = out.characters - out.no_output
    out['accuracy_among_output_characters'] = out.correct / out.output_characters
    return out


summ = summarize(same, ['policy']); summ.to_csv(OUT / 'IV2_policy_summary.csv', index=False)


def mcnemar(a, b):
    n01 = int((a & ~b).sum()); n10 = int((~a & b).sum()); n = n01 + n10
    chi = (abs(n01 - n10) - 1) ** 2 / n if n else 0.0
    return dict(first_only_correct=n01, second_only_correct=n10, discordant=n, mcnemar_chi2_cc=chi,
                p_chi2_cc=float(stats.chi2.sf(chi, 1)) if n else 1.0,
                p_exact=float(stats.binomtest(n10, n, .5).pvalue) if n else 1.0)


wide = same.pivot(index='character_key', columns='policy', values='correct')
mc = pd.DataFrame([dict(first=a, second=b, n=len(wide), accuracy_first=wide[a].mean(), accuracy_second=wide[b].mean(),
                        difference_second_minus_first=wide[b].mean() - wide[a].mean(), **mcnemar(wide[a].astype(bool), wide[b].astype(bool)))
                   for a, b in [('A', 'B'), ('A', 'C'), ('B', 'C')]])
mc.to_csv(OUT / 'IV4_mcnemar.csv', index=False)

# IV.5 subject-cluster bootstrap
subj_acc = same.groupby(['subject_id', 'policy']).correct.agg(['sum', 'size']).unstack()
subjects = subj_acc.index.to_numpy(); rng = np.random.default_rng(20261125); NB = 20000
corr = subj_acc['sum'].to_numpy(float); tot = subj_acc['size'].to_numpy(float)
draw = rng.integers(0, len(subjects), (NB, len(subjects)))
boot = np.stack([corr[draw, j].sum(1) / tot[draw, j].sum(1) for j in range(3)], 1)
point = corr.sum(0) / tot.sum(0)
bt = []
for j, p in enumerate(POL):
    bt.append(dict(quantity=f'accuracy_{p}', estimate=point[j], ci_low=np.percentile(boot[:, j], 2.5), ci_high=np.percentile(boot[:, j], 97.5), boot_p_le_0=np.nan))
for a, b in [('A', 'B'), ('A', 'C'), ('B', 'C')]:
    i, j = POL.index(a), POL.index(b); d = boot[:, j] - boot[:, i]
    bt.append(dict(quantity=f'{b}_minus_{a}', estimate=point[j] - point[i], ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5),
                   boot_p_le_0=float(min(1, 2 * min((d >= 0).mean(), (d <= 0).mean())))))
bt = pd.DataFrame(bt); bt['resamples'] = NB; bt['unit'] = 'subject (10), with replacement'
bt.to_csv(OUT / 'IV5_subject_bootstrap.csv', index=False)

# IV.6 per-subject
ps = same.pivot_table(index='subject_id', columns='policy', values='correct', aggfunc='mean')
ps.columns = [f'accuracy_{c}' for c in ps.columns]
nout = same.pivot_table(index='subject_id', columns='policy', values='no_output', aggfunc='sum'); nout.columns = [f'no_output_{c}' for c in nout.columns]
ps = ps.join(nout); ps['C_minus_A'] = ps.accuracy_C - ps.accuracy_A; ps['B_minus_A'] = ps.accuracy_B - ps.accuracy_A
ps['C_minus_A_characters'] = (ps.C_minus_A * 18).round().astype(int)
dev_subj = {'F_03', 'F_05', 'F_06', 'F_08', 'F_20', 'F_25'}
ps['used_in_detector_development'] = ps.index.isin(dev_subj)
ps.reset_index().to_csv(OUT / 'IV6_per_subject_accuracy.csv', index=False)
loss = -ps.C_minus_A_characters
sign = stats.binomtest(int((ps.C_minus_A < 0).sum()), int((ps.C_minus_A != 0).sum()), .5).pvalue if (ps.C_minus_A != 0).any() else 1.0
conc = dict(subjects_C_lower=int((ps.C_minus_A < 0).sum()), subjects_C_equal=int((ps.C_minus_A == 0).sum()), subjects_C_higher=int((ps.C_minus_A > 0).sum()),
            net_character_loss=int(loss.sum()), loss_from_top2_subjects=int(loss.sort_values(ascending=False).head(2).sum()),
            top2_subjects=loss.sort_values(ascending=False).head(2).index.tolist(), subject_sign_test_p=float(sign))
(OUT / 'IV6_concentration.json').write_text(json.dumps(conc, indent=2))

# V. decomposition
dec = summarize(res, ['train_rule', 'validation_rule'])
base = res[(res.train_rule == 'A') & (res.validation_rule == 'A')].set_index('character_key').correct
lab_map = {('A', 'A'): 'A baseline', ('B', 'A'): 'B train only', ('A', 'B'): 'B validation only', ('B', 'B'): 'B both (reported)',
           ('C', 'A'): 'C train only', ('A', 'C'): 'C validation only', ('C', 'C'): 'C both (reported)'}
dec = dec[[k in lab_map for k in zip(dec.train_rule, dec.validation_rule)]].copy()
dec['setting'] = [lab_map[k] for k in zip(dec.train_rule, dec.validation_rule)]
extra = []
for t in dec.itertuples():
    cur = res[(res.train_rule == t.train_rule) & (res.validation_rule == t.validation_rule)].set_index('character_key').correct.reindex(base.index)
    m_ = mcnemar(base.astype(bool), cur.astype(bool))
    sa = res[(res.train_rule == t.train_rule) & (res.validation_rule == t.validation_rule)].groupby('subject_id').correct.mean().reindex(subjects).to_numpy()
    sb = res[(res.train_rule == 'A') & (res.validation_rule == 'A')].groupby('subject_id').correct.mean().reindex(subjects).to_numpy()
    d = (sa[draw] - sb[draw]).mean(1)
    extra.append(dict(diff_vs_A=cur.mean() - base.mean(), ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5),
                      mcnemar_p_exact_vs_A=m_['p_exact'], A_only_correct=m_['first_only_correct'], setting_only_correct=m_['second_only_correct']))
dec = pd.concat([dec.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
tr_ep = pd.DataFrame([dict(policy=p, training_epochs=int(((setv == 'train') & keep[p] & finite).sum()),
                           validation_kept_flashes=int(((setv == 'validation') & keep[p]).sum())) for p in POL])
dec = dec.merge(tr_ep.rename(columns={'policy': 'train_rule', 'validation_kept_flashes': 'drop'})[['train_rule', 'training_epochs']], on='train_rule') \
         .merge(tr_ep.rename(columns={'policy': 'validation_rule'})[['validation_rule', 'validation_kept_flashes']], on='validation_rule')
dec = dec.set_index('setting').loc[list(lab_map.values())].reset_index()
dec.to_csv(OUT / 'V_decomposition.csv', index=False)
dec_subj = res.assign(setting=[lab_map.get(k, '') for k in zip(res.train_rule, res.validation_rule)])
dec_subj = dec_subj[dec_subj.setting.ne('')].pivot_table(index='subject_id', columns='setting', values='correct', aggfunc='mean')[list(lab_map.values())]
dec_subj.reset_index().to_csv(OUT / 'V_decomposition_per_subject.csv', index=False)

# sensitivity: development reuse
dev_val = {'F_05_SE003_CBCol_Train04.edf', 'F_08_SE003_CBCol_Train03.edf'}
sens = []
for name, mask in [('all subjects', same.edf_name.notna()), ('exclude 2 validation EDFs used in detector development', ~same.edf_name.isin(dev_val)),
                   ('only subjects never used in development (F_07,F_21,F_23,F_24)', ~same.subject_id.isin(dev_subj)),
                   ('only development subjects', same.subject_id.isin(dev_subj))]:
    sm = summarize(same[mask], ['policy']); sm.insert(0, 'subset', name)
    wv = same[mask].pivot(index='character_key', columns='policy', values='correct')
    sm['mcnemar_p_exact_vs_A'] = [1.0 if p == 'A' else mcnemar(wv.A.astype(bool), wv[p].astype(bool))['p_exact'] for p in sm.policy]
    sens.append(sm)
pd.concat(sens).to_csv(OUT / 'V_sensitivity.csv', index=False)

# raw data unchanged
after = {r.edf_name: file_hash(Path(json.loads((ROOT / r.results_path / 'processing_record.json').read_text())['source_edf'])) for r in manifest.itertuples()}
check('-', 'raw EDFs unchanged', 'pass' if after == src_hash_before else 'problem', f'SHA256 of all {len(after)} EDFs identical before/after audit.')
pd.DataFrame(checks).to_csv(OUT / 'checks.csv', index=False)
print(summ.to_string(index=False)); print(mc.to_string(index=False)); print(bt.to_string(index=False))
print(ps.to_string()); print(conc); print(dec.to_string(index=False)); print(pd.concat(sens).to_string(index=False))
