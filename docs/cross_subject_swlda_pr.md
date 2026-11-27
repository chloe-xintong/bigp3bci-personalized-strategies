# Cross-subject SWLDA: exploratory supplementary analysis

Base: `data-preprocessing`; branch: `analysis/cross-subject-swlda`.

Adds a separate leave-one-subject SWLDA analysis asking whether the other nine
subjects' S1/S2 Train data help target S3 calibration at k=0,3,6,12. This is an
exploratory supplementary analysis, outside the within-subject main comparison.
The protocol and config were committed before execution (`762ea1e`); the actual
run used implementation commit `88dd0e3`.

- No test-set files were read (0); only verified Train caches were used.
- No hyperparameters were tuned; main SWLDA settings and preprocessing were reused.
- No subjects were excluded: all ten subjects and all seven conditions remain.
- No donor S3 or target S1/S2 rows entered fitting. Target last-six S3 Train
  characters were report-only validation, never used to select settings.
- Existing configurations/results are unchanged. Data and `outputs/` are untracked.

Thirty recomputed current-only models match notebook 11 (now 07), including AUC,
character accuracy and every validation flash score (maximum absolute difference
0; tolerance 1e-10). Forty cross-subject fits completed. X−C macro AUC differences
at k=3/6/12 are −0.042472/−0.080191/−0.091591, with subject-bootstrap 95% intervals
[−0.073878, −0.005008], [−0.121428, −0.035858], [−0.130375, −0.043919].
These exploratory intervals are unadjusted for multiple comparisons. Direct
unweighted pooling did not improve mean calibration performance in this setup;
this does not change the main design or conclusions.

Validation: full unittest suite; real-data control-equivalence test; independent
split, flash exclusion and bootstrap checks. Separate subject panels, macro plot,
all subject differences and provenance are under
`outputs/cross_subject_swlda_exploratory_v1/` locally.

Reproduce after the existing Train preprocessing and calibration-budget run:

```powershell
python -m unittest discover -s tests -v
python scripts/run_cross_subject_swlda.py --stage controls
python -m unittest discover -s tests -p test_cross_subject_split.py -v
python scripts/run_cross_subject_swlda.py --stage cross
python scripts/plot_cross_subject_swlda.py
```

The runner rejects an existing controls output directory, verifies source hashes
between stages, and requires verified controls before any X fit.
