# Toward Personalized P300 BCI Decoding

Development pipeline for the bigP3BCI dataset (StudyF, row/column P300 speller). It covers causal
preprocessing, epoch-level quality control and blink-candidate detection. It then compares how three
epoch-rejection strategies affect character accuracy with a project-owned SWLDA classifier.

Current experiments use Train EDFs only. The calibration-budget experiment
reserves the last six SE003 Train characters for validation and uses flash-level
ROC-AUC as its primary metric. The discarded Static experiment and its outputs
were removed; no Test EDFs are opened by the current pipeline.

![Character accuracy by rejection strategy](docs/figures/character_accuracy_by_strategy.png)

*SWLDA on all StudyF Train data: 10 subjects × 3 Sessions, with a chronological 2/3 → 1/3 split of
complete characters within each Session. A = basic faults only, B = + 150 µV PTP ceiling, C = + blink
overlap. Validation accuracy: A 78.9%, B 76.7%, C 68.9% (180 characters).*

## Repository layout

```
configs/    frozen JSON configurations, one per experiment
src/        reusable modules (preprocessing, QC, blink detection, policies, SWLDA)
scripts/    run_*  -> run an experiment and write outputs/<name>/
            build_*_notebook.py -> regenerate the matching report notebook
notebooks/  report notebooks 01–09 (10 = EEGNet, planned) and appendix A1, read from outputs/
learning/   MNE tutorial notebook (not part of the analysis)
tests/      unittest suite
docs/       protocols, experiment log, detector freeze note, figures shown in this README
data/       (not tracked) raw EDFs
outputs/    (not tracked) all generated results; recreate with scripts/
```

## Setup

```bash
conda create -n bigp3bci python=3.12
conda activate bigp3bci
pip install -r requirements.txt
python -m ipykernel install --user --name bigp3bci   # kernel used by the notebooks
```

Download the bigP3BCI dataset from PhysioNet (https://physionet.org/content/bigp3bci/). Place it so
that the StudyF EDFs sit at `data/raw/bigP3BCI-data/StudyF/<subject>/SE00x/Train/<run>/*.edf`.
In restricted Windows environments, set `$env:_MNE_FAKE_HOME_DIR=$PWD.Path` before running scripts.

Notebooks 01 and 04 are development notebooks and are committed without outputs. Register the
output filter once per clone (`.gitattributes` applies it to those two files only):

```bash
git config filter.nbstripout.clean "python -m nbstripout"
git config filter.nbstripout.smudge cat
git config filter.nbstripout.required true
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

## Test-set policy

Analyses read Train EDFs only (`<subject>/<session>/Train/CBCol/*.edf`). The Test split (Static, Dyn,
DynBigram) is held out for the final evaluation. Scripts check every EDF and cached processing record
with `src/bigp3bci_data_policy.py`; `tests/test_data_policy.py` fails if code names a Test-split folder.
Two earlier exceptions are recorded in `docs/experiment_log.md`: the raw-data inspection (notebook 01)
read Test EDF headers and state channels to count characters, and a Static experiment read Test/Static
data; its outputs were deleted. The history-amount and EA analyses were run after those results had
been seen and are labelled exploratory.

## Reproduction order

```bash
python scripts/run_causal_pilot.py                      # notebook 02
python scripts/run_expanded_pilot.py                    # notebook 02; preprocessing cache used below
python scripts/compare_epoch_policies.py                # notebook 05
python scripts/refine_blink_boundaries.py               # notebook 04
python scripts/run_bounded_prominence_secondary.py      # notebook 04
python scripts/measure_local_plateau_excursions.py
python scripts/run_all_studyf_preprocessing.py          # notebook 06 (main result, below)
python scripts/run_swlda_rejection_experiment.py --config configs/swlda_all_studyf.json --output outputs/swlda_all_studyf_v3
python scripts/run_swlda_calibration_budget.py          # notebook 07 (also adds weighted log-loss)
python scripts/check_session_drift.py
python scripts/run_history_amount_exploratory.py        # notebook 08
python scripts/audit_history_amount_exploratory.py
python scripts/run_ea_sensitivity.py                    # notebook 09
```

Each run writes a new directory under `outputs/` and records configuration and input hashes. Rebuild a
report with its `build_*_notebook.py` script and execute it with the `bigp3bci` kernel.

## Reproducing the main result (all StudyF, v3)

```bash
python scripts/run_all_studyf_preprocessing.py
python scripts/run_swlda_rejection_experiment.py --config configs/swlda_all_studyf.json --output outputs/swlda_all_studyf_v3
python scripts/audit_swlda_saved_experiment.py outputs/swlda_all_studyf_v3
python scripts/build_three_session_experiment_notebook.py --all-studyf
python scripts/plot_swlda_all_studyf_accuracy.py
```

Then execute `notebooks/06_swlda_comparison.ipynb` with the `bigp3bci` kernel.
`run_all_studyf_preprocessing.py` reuses cached per-EDF results and can be restarted after an interruption.

## Notebook index

| Notebook | Topic | Script / config |
|---|---|---|
| [01](notebooks/01_data_inventory.ipynb) | Data inventory and raw inspection, EDF inspection, artifact QC (no outputs) | interactive |
| [02](notebooks/02_preprocessing.ipynb) | Preprocessing and candidate detection: expanded pilot, then causal pilot | `build_expanded_pilot_notebook.py`, `build_causal_pilot_notebook.py` |
| [03](notebooks/03_blink_review.ipynb) | Blink detection summary, candidate review, false positives, threshold validation | builders update their own chapter |
| [04](notebooks/04_blink_boundaries_secondary.ipynb) | Blink main-wave / recovery localization and bounded-prominence confirmation (no outputs) | `refine_blink_boundaries.py`, `run_bounded_prominence_secondary.py` |
| [05](notebooks/05_epoch_policy_comparison.ipynb) | Three epoch policies A/B/C | `compare_epoch_policies.py`, `configs/three_policy_comparison.json` |
| [06](notebooks/06_swlda_comparison.ipynb) | SWLDA main result (all StudyF, v3); v2 and v1 in the appendix | `build_three_session_experiment_notebook.py --all-studyf` |
| [07](notebooks/07_swlda_calibration_budget.ipynb) | SWLDA calibration budget, last-six SE003 Train validation, flash ROC-AUC | `run_swlda_calibration_budget.py`, `configs/swlda_calibration_budget.json` |
| [08](notebooks/08_history_amount_exploratory.ipynb) | Historical-file amount H versus current budget k (exploratory) | `run_history_amount_exploratory.py`, `configs/history_amount_exploratory.json` |
| [09](notebooks/09_ea_sensitivity.ipynb) | Session-wise Euclidean alignment sensitivity (exploratory) | `run_ea_sensitivity.py`, `configs/ea_sensitivity.json` |
| 10 | EEGNet (planned) | `docs/eegnet_protocol.md` |
| [A1](notebooks/A1_continuous_eeg_manual_qc.ipynb) | Appendix: continuous EEG manual QC record | interactive (Qt) |

Related reports share one notebook, with the latest coverage first and historical experiments in separate chapters. Original builders update only their own chapter and retain other chapters and saved outputs. Restart the kernel before running a selected chapter from its initialization cell; interactive review chapters retain their original instructions. Rebuilding a chapter clears that chapter's execution outputs until it is executed again. Data scopes and detector versions remain separate.

`python scripts/notebook_reports.py` consolidates legacy report files if present and validates that code and saved outputs survive the merge. It can also be rerun on the consolidated reports.

## Experiment log

### Causal preprocessing pilot (notebook 02)

新增预处理 pilot 可通过以下命令重复运行（项目已有 `bigp3bci` conda 环境）：

```powershell
python scripts/run_causal_pilot.py
python -m unittest discover -s tests -v
```

统一配置为 `configs/causal_pilot.json`，实现位于 `src/bigp3bci_preprocessing.py`。CLI 支持 `--config`、`--data-root`、`--output`。默认按稳定排序为 F_05 / F_03 各选择一份 Train/CBCol EDF；Duke/ETSU 映射沿用 notebook 01 的推断，未提升为已证实的站点事实。处理函数也可用于其他 EDF，不进行训练或阈值拟合；正式测试配置须在验证与 calibration budget 确定后单独冻结。

仅处理已核对的 16 个 EEG 通道；依次执行 60 Hz notch（Q=30，二阶）与 Butterworth 0.5–30 Hz bandpass（原型四阶，带通总八阶），使用 SOS 因果实现。选择依据为平坦通带与适中的工程复杂度；这些均是可调整候选，**不代表精确复现 BCI2000**。每份 EDF 以首个样本对应的稳态初始化，按 1 秒块保留内部状态；不跨 EDF 传递状态。初始 10 秒为保守开发启动保护区，涉及 epoch 无效，尚非已验证的最短稳定时间。采集滤波字符串按 EDF 原始 header 逐通道保存，与新增滤波分开。输出前后校验源文件 SHA256。事件通道不被滤波，事件不移动，采样率及连续样本数不变。

Epoch 使用 `[onset, onset + ceil(0.8*sfreq))` 半开样本区间；256 Hz 时为 205 点，最后一点 796.875 ms。在线等待 onset 后完整 800 ms，再使用该时点已完成的连续 QC 窗口；重叠允许，末尾不足仍保留并标记。Baseline=None。复用 `stimulus_begin_events`：StimulusType=0 为 non-target，1 为 target。按 notebook 01 的 SelectedTarget 上升沿结束 trial；缺边界事件保留 trial=-1，不能根据目标字符值猜测边界。

连续 QC 采用固定阈值的 2 秒窗口、1 秒步长，尾部附加一个截至文件末尾的窗口，不按整份 EDF 分布估阈值。检查原始非有限、全零、常值、低 PTP/STD、绝对大幅与大波动；保存每窗可用样本、历史异常占比、平线比例和持续异常候选理由。连续平线检测的时间尺度为窗口长度，短于窗口的孤立平线可能漏检。持续候选使用至少 10 秒历史及 20% 异常窗口比率；此规则也是开发候选。窗口级与 epoch 级严重平线标记均保留。非有限输入仅为防止递归滤波永久污染而以前一有限输入维持状态；对应点及后续 10 秒恢复区无效。

持续异常通道不会被删除、插值或改变输入维度。达到持续候选条件后，`valid_epoch` 为缺失值、`epoch_qc_status=pending_channel_policy`，与普通拒绝区分。文件记录 `file_needs_channel_policy=true`，整份文件的 `classification_eligible=false`。其余开发诊断仍保留，包括阈值触发原因。**下游必须同时检查 valid_epoch 与 classification_eligible**；NPZ 额外保存 validity_pending，不应仅据布尔数组判断。未出现持续异常时，valid_epoch 直接反映 QC 通过与否。候选阈值比较表将 pending 视为未保留；细分 CSV 中 size 是全部数量，sum 是已知有效数量，出现 pending 时 mean 仅对已判定记录计算。

PTP 在滤波后以 V 计算，配置与表格为 µV。三个候选上下限为 (0.5,150)、(0.5,250)、(0.2,350) µV，默认第二组仅用于展示，不是最终阈值。它们是用于探索宽松程度的固定工程候选，未从标签或全训练字符估计，不使用测试分布，也没有实施 calibration budget 的估计器。任何后续阈值选择须遵守原有验证划分及对应字符预算。人工 annotation 未用于此次阈值验证；未标注区间没有被当作正常。

每份 EDF 的 `outputs/causal_pilot/<edf_stem>/report.html` 汇集 PSD（逐通道及线性功率平均，0–125 Hz 同坐标，含 60 Hz）、低频 PSD、同时间同轴波形（启动、固定片段、最大原始样本跳变附近）、实际频响及群延迟、PTP 分布、拒绝前后 ERP。最大跳变片段是开发可视化选择，不参与在线 QC。ERP 仅对完整窗口作均值，图例给出各类别计数，拒绝前包含启动期；它不是未滤波 ERP。群延迟依频率变化，不可用单一偏移修正全部信号，原始事件未移位。

同目录保存全部连续 EEG（V）、全部 epochs（包括无效/末尾补 NaN）、每条事件 metadata 与 QC、逐通道逐 epoch PTP、连续窗口指标、通道候选、三个阈值比较及各类/各通道触发统计、每个字符有效 flash 数、处理记录。每事件可有多个拒绝原因；顶部目录再提供 EDF/subject/session 汇总。NPZ 通道顺序以各自 channels 数组为准。当前配置 QC 通道与 EEG 顺序相同；不合格 flash 不产生任何分类分数。字符仅在有效 flash 为零时标记证据不足，其他最小证据量及评分/停止规则尚未确定。

2026-11-24 pilot：F_05_SE001_CBCol_Train02 与 F_03_SE001_CBCol_Train01 各 756 个 flash、6 个字符；默认候选各保留 734（97.09%），22 个均因启动保护区无效。target 保留 81/84，non-target 保留 653/672；首字符保留 104/126，其余各 126/126。两份均未检出持续异常候选。更严格 150 µV 上限在 Duke 保留 729，在 ETSU 保留 734；这两份不能支持确定最终阈值。采集 header 分别为 Duke HP:0.1Hz LP:60Hz N:60Hz、ETSU HP:0.5Hz LP:30Hz NHP:58Hz NLP:62Hz。新增滤波仍在两者执行。

已验证分块一致性、未来数据独立性、标签与 QC 独立性、重复目标字符的边界、不完整 epoch 保留、持续坏通道待处理及非有限值恢复。未加入 ICA、伪迹来源分类、坏通道插值、模型训练、动态停止或 baseline correction。

<!-- swlda-rejection-experiment-v1 -->

## SWLDA epoch rejection comparison (Train development only)

Run `python scripts/run_swlda_rejection_experiment.py` from the repository root. Configuration: `configs/swlda_rejection_experiment.json`. In restricted Windows environments, set `$env:_MNE_FAKE_HOME_DIR=$PWD.Path` before running. View the fully executed `notebooks/06_swlda_comparison.ipynb` for figures and step-by-step explanations. The notebook builder is `scripts/build_swlda_experiment_notebook.py`; rebuilding creates an unexecuted notebook.

The comparison uses 12 existing Train EDFs from 6 subjects, grouped into 11 subject/session models per policy: 48 complete calibration characters and the same 24 chronological validation characters. Final Test EDFs are excluded. Source and configuration hashes, fixed split manifests, mapping checks, 256-feature definitions, selected features, SWLDA models, missing scores for rejected flashes, character predictions, per-stimulus counts, AUC and paired comparisons are saved under `outputs/swlda_rejection_experiment_v1/`.

A uses basic hard-fault/technical checks; B adds a 150 microvolt PTP ceiling; C adds blink overlap using only `old_candidate_overlap_v1` (the original 25 microvolt frontal-amplitude candidate rule and its original interval boundaries, not the newer main-wave intervals). All policies share startup and basic quality rules. SWLDA is a project-owned forward/backward stepwise OLS implementation, with entry p<0.10, removal p>0.15 and at most 60 features; feature selection and standardization use each policy's retained training epochs only. It is not an ordinary/regularized LDA substitute or a claim of bit-exact BCI2000 reproduction.

Observed validation accuracy A/B/C: 18/24 (75.0%), 17/24 (70.8%), 12/24 (50.0%). Validation epoch retention: 100.0%, 92.49%, 73.28%. C has four no-output characters, included in its denominator, and an additional 1.34375 s sample-clock decision wait after character completion. Timing includes data availability and confirmation, but excludes unmeasured device communication and online compute latency. Rejected scores remain missing; candidate evidence is the mean score when lit minus the mean score when unlit, requiring both groups for all 36 candidates.

These are small, descriptive development results on records already used in preprocessing development. They do not establish generalization, validate a final blink threshold, or evaluate the newer main-wave boundaries. No detector parameters were retuned from this experiment. Tests include an independent direct-OLS SWLDA reference and feature-bin/decoding checks (`python -m unittest discover -s tests -q`).

<!-- /swlda-rejection-experiment-v1 -->


<!-- full-three-session-experiment -->
## Full three-Session SWLDA comparison

The expanded development comparison includes all 54 Train/CBCol EDFs for F_03, F_05, F_06, F_08, F_20 and F_25: three Sessions and three EDFs per Session. The source selection is frozen before detection; final Test files are excluded. Each subject/Session is an independent model group with a chronological 2/3 character calibration split. Parameters and the original 25-microvolt blink interval version remain unchanged from v1.

```powershell
python scripts/run_full_three_session_preprocessing.py
python scripts/run_swlda_rejection_experiment.py --config configs/swlda_three_sessions.json --output outputs/swlda_three_sessions_v2
python scripts/audit_swlda_saved_experiment.py outputs/swlda_three_sessions_v2
python scripts/build_three_session_experiment_notebook.py
```

Then execute `notebooks/06_swlda_comparison.ipynb`. Outputs are separate from v1. The combined six-subject figure shows the mean and sample SD over all three Sessions; character denominators and no-output counts remain visible in tables. Independent EDFs use separate sample clocks and filter state, so cross-file elapsed time is not fabricated. Expanded coverage also increases the per-Session calibration budget relative to the small pilot; differences from v1 cannot be attributed to coverage alone.
<!-- /full-three-session-experiment -->


<!-- all-studyf-experiment -->
## All StudyF subjects (v3)

Scope: all 10 local StudyF subjects, 30 Sessions and 90 Train/CBCol EDFs. StudyB and final Test files are excluded. Selection is checked against the complete local StudyF Train inventory before any detection or fitting. Reuse identical cached preprocessing; preserve all v1/v2 results. Filter, old_candidate_overlap_v1 blink intervals, PTP 150 microvolt comparison, fixed 256 features and SWLDA settings remain unchanged.

```powershell
python scripts/run_all_studyf_preprocessing.py
python scripts/run_swlda_rejection_experiment.py --config configs/swlda_all_studyf.json --output outputs/swlda_all_studyf_v3
python scripts/audit_swlda_saved_experiment.py outputs/swlda_all_studyf_v3
python scripts/build_three_session_experiment_notebook.py --all-studyf
```

Execute `notebooks/06_swlda_comparison.ipynb` for the full report and combined ten-subject error-bar plot (three Session means +/- sample SD). Detailed scores, selected features, models, fixed character splits, subject/Session results and paired comparisons live in `outputs/swlda_all_studyf_v3`. These remain development results, without detector retuning or final Test evaluation.
<!-- /all-studyf-experiment -->

## SWLDA calibration budgets with Train holdout validation

```powershell
python scripts/run_swlda_calibration_budget.py
```

Configuration: `configs/swlda_calibration_budget.json`. Output:
`outputs/swlda_calibration_budget_train_validation_v1/`. Existing outputs are
preserved; use a new `--output` directory for a rerun. A short check can use
`--subjects F_05 --k-values 0 1 12 --output outputs/<new-name>`.

View the executed [notebook 07](notebooks/07_swlda_calibration_budget.ipynb).
Rebuild with `python scripts/build_swlda_budget_notebook.py`; its cells only
read saved results, without training.

Each subject has a separate model. History is all 36 SE001/SE002 Train characters.
SE003 Train has 18 characters: the first 12 form the calibration pool and the
last six are fixed validation, so k ranges from 0 through 12. Four curves compare
current only, both historical Sessions plus current, SE001 history plus current,
and SE002 history plus current. Test/Static, Dyn and DynBigram are excluded.

The primary metric is per-subject validation flash ROC-AUC, macro-averaged across
subjects. Raw scores from separate subject models are not pooled into one AUC.
Character accuracy is secondary. Flash scores and QC flags are exported to
`flash_predictions.csv`; overlapping flash epochs are not independent observations.
SWLDA raw scores are not probabilities. Weighted flash log-loss is calculated
after a training-only Platt sigmoid fitted on whole-character out-of-fold scores
(up to three folds); current-only k=1 explicitly falls back to in-sample training
scores because there is only one character. Validation data never fit the sigmoid.
The sigmoid retains the natural training prior. The evaluation loss averages the
two class-specific mean negative log-likelihoods with equal weight, then averages
subjects. No fixed 1:5 label ratio is assumed. Current-only k=0 has no model.

`python scripts/add_weighted_logloss.py` appends this metric to an existing run
under `weighted_logloss/`, preserving original predictions and AUC. New complete
runs execute this step automatically. The extension includes per-flash probabilities,
calibrator parameters, fallback flags and weighted log-loss curves.

All models use the predefined 16 channels, existing 256-dimensional fixed causal
features and policy A. Scaling and SWLDA feature selection use only the retained
training epochs allowed by that model's budget. No validation character is used
for fitting. Endpoint refits verify predictions and weights; the data-rich reference
uses 36 historical plus 12 current calibration characters and is not budget-matched
to smaller k. It is not a mathematical upper bound.

No EEGNet training loop currently exists. `src/bigp3bci_flash_early_stopping.py`
provides flash-AUC and class-balanced flash-log-loss checkpoint interfaces for a
future EEGNet loop. Predeclare one metric: save weights when AUC increases or
weighted log-loss decreases, stop after the configured patience, and restore the
best checkpoint. Training class weights can be calculated from the allowed training
labels with `bigp3bci_probability.binary_class_weights`. Adding an evaluation metric
alone does not reweight SWLDA training. Character accuracy must not choose the epoch.
These are development validation results, not an independent final test.

## Exploratory historical data amount

The frozen [protocol](docs/history_amount_exploratory_protocol.md) enumerates every
subset of each subject's six SE001/SE002 Train EDFs, with H=0..6 and k in
{0,3,6,12}. Current calibration uses the first 12 SE003 Train characters; the
last six remain validation. H=0,k=0 is undefined, leaving 2,550 fitted models.
No Test files are read, no parameters are tuned per H, and no unfavorable subsets
are removed. This is explicitly exploratory work after prior test results were seen.

```powershell
python scripts/run_history_amount_exploratory.py
python scripts/build_history_amount_notebook.py
```

Outputs are in `outputs/history_amount_exploratory_v1/`. The run checkpoints each
model and resumes with unchanged configuration and input hashes. Notebook 08
shows shared-axis 2-by-5 subject panels followed by a macro panel, all subset
points, same-k current-only references, and 10,000-replicate subject-bootstrap
intervals. Tables include per-subject subset means, paired differences from
current-only, and pure-SE002 minus pure-SE001 contrasts at H=1,2,3. Character
accuracy is descriptive only. EEGNet has only a seeded subset manifest: no
network results are reported, and a fixed epoch count must be set before future
training; this exploratory validation set must not choose its checkpoint.

## Exploratory Euclidean alignment sensitivity

The frozen [protocol](docs/ea_sensitivity_protocol.md) asks whether session-wise
Euclidean alignment (EA) changes history-plus-current SWLDA performance. Each
session's reference covariance uses only its allowed data: all SE001/SE002 Train
characters, and only the first k SE003 calibration characters (never the six
validation characters or Test), with fixed shrinkage 0.001. EA is applied after
preprocessing and before the 16x16 window features; because window means commute
with channel-space linear maps it is applied to cached features, verified against
raw epochs. Only k=3, 6, 12 are reported (k=0 has no S3 data for R). Conditions:
A (history+current, no EA; reproduces notebook 08 H=6), B (history+current, EA),
current only, and C (current only, EA).

```powershell
python scripts/run_ea_sensitivity.py
python scripts/build_ea_sensitivity_notebook.py
```

Result (exploratory, report only): macro B-A is +0.001/+0.005/+0.007 AUC at
k=3/6/12 with all subject-bootstrap CIs including 0, and subject directions are
inconsistent (5-6 of 10 positive), so results are reported per subject. F_07, the
subject with the largest drift asymmetry, improves by +0.04 to +0.07. The
predeclared EEGNet follow-up rule was not met, and the main design is unchanged.

## Blink candidate association with flash labels

Run `python scripts/summarize_blink_label_association.py` after all-StudyF preprocessing.
The script verifies cached epoch-table hashes and summarizes all 68,004 Train flashes,
without quality-based rejection, by subject and subject/Session under
`outputs/blink_label_association/`. Candidate-positive flashes overlap at least one
`old_candidate_overlap_v1` interval within the 0–800 ms epoch; these counts differ
from unique candidate events because flash windows overlap.

The subject-stratified Mantel–Haenszel target/non-target odds ratio is 1.52, with a
95% percentile interval of 0.85–3.58 from 50,000 whole-subject bootstrap samples
(seed 20261125). Only ten subject clusters are available, and this wide interval
does not establish equivalence or absence of association. F_05 has candidate
overlap in 685/756 target and 3,470/6,048 non-target flashes; F_25 has 275/756 and
2,109/6,048, respectively. These are engineering candidates rather than confirmed
blinks, and this descriptive analysis does not change the preprocessing policy.

## Reference

### Scripts not covered above

| Script | Purpose |
|---|---|
| `run_expanded_pilot.py` | Expanded Train pilot; writes `outputs/expanded_pilot`, part of the preprocessing input manifest |
| `check_session_drift.py` | Session covariance and amplitude drift (`configs/session_drift_descriptive.json`), used to choose EA subjects |
| `measure_local_plateau_excursions.py` | Local plateau metrics (`configs/local_plateau_metrics.json`) |
| `audit_history_amount_exploratory.py` | Independent audit of the history-amount analysis, shown in notebook 08 |
| `audit_studyf_abc_strategies.py` | Independent audit of the v3 A/B/C experiment |
| `plot_swlda_calibration_budget.py` | Redraw notebook 07 figures from saved CSVs without refitting |
| `build_double_blind_review.py`, `score_double_blind_review.py` | Double-blind blink review: anonymous packets and comparison of two independent forms |
| `export_blink_candidates.py`, `export_unmatched_blink_context.py` | Export candidate annotations and unmatched-candidate context for manual review |
| `audit_main_false_positive_reviews.py` | Audit saved main-wave false-positive reviews |

### Modules (`src/bigp3bci_*.py`)

| Purpose | Modules |
|---|---|
| Data access | `data_policy` |
| Preprocessing and QC | `preprocessing`, `continuous_qc`, `policy_comparison` |
| Blink detection and review | `expanded_pilot`, `candidate_review`, `blink_boundaries`, `boundary_review`, `blind_review`, `bounded_prominence`, `plateau_metrics` |
| SWLDA and evaluation | `swlda`, `probability`, `flash_early_stopping` |
| Exploratory analyses | `calibration_budget`, `history_amount`, `euclidean_alignment` |
| Figures | `budget_plot`, `figure_style` |

### Configurations

| Config | Entry script | Output |
|---|---|---|
| `causal_pilot.json` | `run_causal_pilot.py` | `outputs/causal_pilot` |
| `expanded_pilot.json` | `run_expanded_pilot.py` | `outputs/expanded_pilot` |
| `three_policy_comparison.json` | `compare_epoch_policies.py` | `outputs/three_policy_comparison` |
| `blink_boundary_localization.json` | `refine_blink_boundaries.py` | `outputs/blink_boundary_localization_v1` |
| `bounded_prominence_secondary.json` | `run_bounded_prominence_secondary.py` | `outputs/bounded_prominence_secondary_v1` |
| `local_plateau_metrics.json` | `measure_local_plateau_excursions.py` | `outputs/local_plateau_metrics_v1` |
| `swlda_rejection_experiment.json` | `run_swlda_rejection_experiment.py` | `outputs/swlda_rejection_experiment_v1` |
| `swlda_three_sessions.json` | read by notebook 06 builder | `outputs/swlda_three_sessions_v2` |
| `swlda_all_studyf.json` | read by notebook 06 builder | `outputs/swlda_all_studyf_v3` |
| `swlda_calibration_budget.json` | `run_swlda_calibration_budget.py` | `outputs/swlda_calibration_budget_train_validation_v1` |
| `session_drift_descriptive.json` | `check_session_drift.py` | `outputs/session_drift_descriptive_v1` |
| `history_amount_exploratory.json` | `run_history_amount_exploratory.py` | `outputs/history_amount_exploratory_v1` |
| `ea_sensitivity.json` | `run_ea_sensitivity.py` | `outputs/ea_sensitivity_v1` |

### Documents

- `docs/history_amount_exploratory_protocol.md`, `docs/ea_sensitivity_protocol.md`: rules fixed before each exploratory run.
- `docs/eegnet_protocol.md`: EEGNet rules; open items are filled in before the first training run.
- `docs/blink_detector_freeze.md`: frozen detector version, parameters and development files.
- `docs/experiment_log.md`: date, commit and Test access of every run.
- `docs/reorganization_checklist.md`: repository reorganization plan.
