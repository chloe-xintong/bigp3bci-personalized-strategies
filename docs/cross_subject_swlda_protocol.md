# Cross-subject SWLDA: exploratory supplementary analysis

Rules frozen before execution on branch `analysis/cross-subject-swlda`.
This analysis follows previous development and test-result exposure documented
in the experiment log. It is supplementary to the within-subject main design;
it does not enter the main comparison tables or alter reported conclusions.

## Data and splits

Keep all ten subjects listed in `configs/cross_subject_swlda_exploratory.json`.
For each held-out target, donors are the other nine subjects, using their S1/S2
Train characters only (36 each, 324 total). Donor S3 is excluded from fitting.
Target S1/S2 is excluded. Target S3 Train first 12 chronological characters form
the calibration pool; last six are report-only validation. X0 uses donors alone;
X3/X6/X12 concatenate donor epochs with the first k target calibration characters.
C3/C6/C12 use only those target calibration characters. There is no C0.

Training, calibration and validation character membership must be checked before
fitting, including subject/session ownership, Train-only provenance, unique keys,
chronology and no shared calibration/validation samples within an EDF. The
target is absent from the donor training pool; target calibration is intentionally
included in X when k>0. Never open Test, Static, Dyn or DynBigram files. Read only
existing verified Train feature caches and processing records; no raw EDF reads.
The aggregate cache contains all subjects, but donor S3 rows must never enter
fitting or target evaluation. Validation cannot select parameters or conditions.

## Fixed model and controls

Reuse `bigp3bci_swlda.fit_swlda`, `score_swlda`, `auc_rank` and
`decode_character`, and `bigp3bci_calibration_budget.make_plan`/`training_keys`
from the current calibration-budget experiment (now notebook 07, formerly 11).
Match its complete SWLDA, feature and character-evidence configurations:
p_enter=.10, p_remove=.15, maximum 60 selected features, maximum 500 steps,
rank tolerance 1e-10, training-only population z-score, predefined 16 channels
and 256 features. Use policy A and the existing retained-epoch masks.
Concatenate all allowed training epochs without donor balancing or reweighting.
No alignment, tuning, donor subsets or subject exclusions.

Fit C first and require each subject/k AUC and character accuracy, as well as
saved flash scores, to agree with the existing current-only results within
absolute tolerance 1e-10 (no relative tolerance). Abort on a mismatch before X.

## Metrics and statistics

Report validation flash ROC-AUC per subject, then unweighted macro mean over
all ten. Exactly 22 startup flashes per subject (220 total) are unscored:
7,560 validation flashes and 7,340 scored. Character accuracy over six validation
characters is descriptive, without significance claims.

Compute paired Xk-Ck differences for k=3,6,12 per subject. Bootstrap ten whole
subjects with replacement 10,000 times, seed 20261126; use percentile 2.5/97.5%
intervals for mean AUCs and paired mean differences. Average subjects, not flashes.
Retain all subjects/conditions, including unfavorable outcomes; abort and report
unavailable metrics instead of silently dropping subjects. Plot shared-axis 2x5
subject panels and a separate macro curve with intervals, using report colors
(current-only gray, pooled cross-subject blue), clearly labelled as exploratory
supplementary cross-subject results. No outcome-dependent changes to this protocol.

## Outputs and order

1. Commit this protocol and configuration before running the analysis.
2. Run split/unit tests, then fit and verify C before any X fit.
3. Fit X0/X3/X6/X12; generate the prescribed tables and figures.
4. Record date, runtime commit and zero test-file reads in the experiment log.

Use only the new directory `outputs/cross_subject_swlda_exploratory_v1/`.
Preserve all existing configs/results. Save full resolved config, runtime commit,
subject results, paired subject differences and bootstrap intervals, auditable
split manifest, per-subject and macro figures. Additional provenance, scores,
control checks and completion metadata support verification. `outputs/` stays
untracked. The PR must state: exploratory supplementary analysis; no test reads,
no tuning and no subject exclusions. If a rule conflicts, raise it in the PR
before proceeding; do not change the design autonomously.
