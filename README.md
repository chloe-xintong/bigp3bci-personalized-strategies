# Toward Personalized P300 BCI Decoding

Development pipeline for the bigP3BCI dataset (StudyF, row/column P300 speller). It covers causal
preprocessing, epoch-level quality control and blink-candidate detection. It then compares how three
epoch-rejection strategies affect character accuracy with a project-owned SWLDA classifier.

All results so far are **development results on Train EDFs only**. Final Test files have not been used.

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
notebooks/  executed report notebooks (00–16), read from outputs/
tests/      unittest suite
docs/       curated figures shown in this README
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

Run the tests:

```bash
python -m unittest discover -s tests -v
```

## Reproducing the main result (all StudyF, v3)

```bash
python scripts/run_all_studyf_preprocessing.py
python scripts/run_swlda_rejection_experiment.py --config configs/swlda_all_studyf.json --output outputs/swlda_all_studyf_v3
python scripts/audit_swlda_saved_experiment.py outputs/swlda_all_studyf_v3
python scripts/build_three_session_experiment_notebook.py --all-studyf
python scripts/plot_swlda_all_studyf_accuracy.py
```

Then execute `notebooks/16_bigp3bci_swlda_all_studyf.ipynb` with the `bigp3bci` kernel.
`run_all_studyf_preprocessing.py` reuses cached per-EDF results and can be restarted after an interruption.

## Notebook index

| Notebook | Topic | Script / config / output |
|---|---|---|
| 00–04 | MNE tutorial, EDF inspection, raw inspection, artifact QC, continuous manual QC | interactive |
| 05 | Causal preprocessing pilot | `run_causal_pilot.py`, `configs/causal_pilot.json`, `outputs/causal_pilot` |
| 06 | Manual blink threshold validation | `evaluate_manual_blinks.py`, `outputs/blink_validation` |
| 07 | Expanded candidate pilot | `run_expanded_pilot.py`, `configs/expanded_pilot.json`, `outputs/expanded_pilot` |
| 08 | Current blink detection summary | `check_blink_annotation_updates.py`, `outputs/blink_annotation_update` |
| 09 | Reviewed blink false positives | `inspect_reviewed_blinks.py`, `audit_main_false_positive_reviews.py`, `outputs/blink_false_positive_review_audit` |
| 10 | Three epoch-policy comparison | `compare_epoch_policies.py`, `configs/three_policy_comparison.json`, `outputs/three_policy_comparison` |
| 11 | Blink candidate review | `export_blink_candidates.py`, `outputs/blink_candidate_review` |
| 12 | Blink main-wave / recovery localization | `refine_blink_boundaries.py`, `configs/blink_boundary_localization.json`, `outputs/blink_boundary_localization_v1` |
| 13 | Bounded-prominence secondary analysis | `run_bounded_prominence_secondary.py`, `configs/bounded_prominence_secondary.json`, `outputs/bounded_prominence_secondary_v1` |
| 14 | SWLDA rejection experiment v1 (12 EDFs) | `run_swlda_rejection_experiment.py`, `configs/swlda_rejection_experiment.json` |
| 15 | SWLDA, full three Sessions v2 (54 EDFs) | `configs/swlda_three_sessions.json`, `outputs/swlda_three_sessions_v2` |
| 16 | SWLDA, all StudyF v3 (90 EDFs) | `configs/swlda_all_studyf.json`, `outputs/swlda_all_studyf_v3` |

## Experiment log

### Causal preprocessing pilot (notebook 05)

新增预处理 pilot 可通过以下命令重复运行（项目已有 `bigp3bci` conda 环境）：

```powershell
python scripts/run_causal_pilot.py
python -m unittest discover -s tests -v
```

统一配置为 `configs/causal_pilot.json`，实现位于 `src/bigp3bci_preprocessing.py`。CLI 支持 `--config`、`--data-root`、`--output`。默认按稳定排序为 F_05 / F_03 各选择一份 Train/CBCol EDF；Duke/ETSU 映射沿用 notebook 02 的推断，未提升为已证实的站点事实。处理函数也可用于其他 EDF，不进行训练或阈值拟合；正式测试配置须在验证与 calibration budget 确定后单独冻结。

仅处理已核对的 16 个 EEG 通道；依次执行 60 Hz notch（Q=30，二阶）与 Butterworth 0.5–30 Hz bandpass（原型四阶，带通总八阶），使用 SOS 因果实现。选择依据为平坦通带与适中的工程复杂度；这些均是可调整候选，**不代表精确复现 BCI2000**。每份 EDF 以首个样本对应的稳态初始化，按 1 秒块保留内部状态；不跨 EDF 传递状态。初始 10 秒为保守开发启动保护区，涉及 epoch 无效，尚非已验证的最短稳定时间。采集滤波字符串按 EDF 原始 header 逐通道保存，与新增滤波分开。输出前后校验源文件 SHA256。事件通道不被滤波，事件不移动，采样率及连续样本数不变。

Epoch 使用 `[onset, onset + ceil(0.8*sfreq))` 半开样本区间；256 Hz 时为 205 点，最后一点 796.875 ms。在线等待 onset 后完整 800 ms，再使用该时点已完成的连续 QC 窗口；重叠允许，末尾不足仍保留并标记。Baseline=None。复用 `stimulus_begin_events`：StimulusType=0 为 non-target，1 为 target。按 notebook 02 的 SelectedTarget 上升沿结束 trial；缺边界事件保留 trial=-1，不能根据目标字符值猜测边界。

连续 QC 采用固定阈值的 2 秒窗口、1 秒步长，尾部附加一个截至文件末尾的窗口，不按整份 EDF 分布估阈值。检查原始非有限、全零、常值、低 PTP/STD、绝对大幅与大波动；保存每窗可用样本、历史异常占比、平线比例和持续异常候选理由。连续平线检测的时间尺度为窗口长度，短于窗口的孤立平线可能漏检。持续候选使用至少 10 秒历史及 20% 异常窗口比率；此规则也是开发候选。窗口级与 epoch 级严重平线标记均保留。非有限输入仅为防止递归滤波永久污染而以前一有限输入维持状态；对应点及后续 10 秒恢复区无效。

持续异常通道不会被删除、插值或改变输入维度。达到持续候选条件后，`valid_epoch` 为缺失值、`epoch_qc_status=pending_channel_policy`，与普通拒绝区分。文件记录 `file_needs_channel_policy=true`，整份文件的 `classification_eligible=false`。其余开发诊断仍保留，包括阈值触发原因。**下游必须同时检查 valid_epoch 与 classification_eligible**；NPZ 额外保存 validity_pending，不应仅据布尔数组判断。未出现持续异常时，valid_epoch 直接反映 QC 通过与否。候选阈值比较表将 pending 视为未保留；细分 CSV 中 size 是全部数量，sum 是已知有效数量，出现 pending 时 mean 仅对已判定记录计算。

PTP 在滤波后以 V 计算，配置与表格为 µV。三个候选上下限为 (0.5,150)、(0.5,250)、(0.2,350) µV，默认第二组仅用于展示，不是最终阈值。它们是用于探索宽松程度的固定工程候选，未从标签或全训练字符估计，不使用测试分布，也没有实施 calibration budget 的估计器。任何后续阈值选择须遵守原有验证划分及对应字符预算。人工 annotation 未用于此次阈值验证；未标注区间没有被当作正常。

每份 EDF 的 `outputs/causal_pilot/<edf_stem>/report.html` 汇集 PSD（逐通道及线性功率平均，0–125 Hz 同坐标，含 60 Hz）、低频 PSD、同时间同轴波形（启动、固定片段、最大原始样本跳变附近）、实际频响及群延迟、PTP 分布、拒绝前后 ERP。最大跳变片段是开发可视化选择，不参与在线 QC。ERP 仅对完整窗口作均值，图例给出各类别计数，拒绝前包含启动期；它不是未滤波 ERP。群延迟依频率变化，不可用单一偏移修正全部信号，原始事件未移位。

同目录保存全部连续 EEG（V）、全部 epochs（包括无效/末尾补 NaN）、每条事件 metadata 与 QC、逐通道逐 epoch PTP、连续窗口指标、通道候选、三个阈值比较及各类/各通道触发统计、每个字符有效 flash 数、处理记录。每事件可有多个拒绝原因；顶部目录再提供 EDF/subject/session 汇总。NPZ 通道顺序以各自 channels 数组为准。当前配置 QC 通道与 EEG 顺序相同；不合格 flash 不产生任何分类分数。字符仅在有效 flash 为零时标记证据不足，其他最小证据量及评分/停止规则尚未确定。

2026-11-24 pilot：F_05_SE001_CBCol_Train02 与 F_03_SE001_CBCol_Train01 各 756 个 flash、6 个字符；默认候选各保留 734（97.09%），22 个均因启动保护区无效。target 保留 81/84，non-target 保留 653/672；首字符保留 104/126，其余各 126/126。两份均未检出持续异常候选。更严格 150 µV 上限在 Duke 保留 729，在 ETSU 保留 734；这两份不能支持确定最终阈值。采集 header 分别为 Duke HP:0.1Hz LP:60Hz N:60Hz、ETSU HP:0.5Hz LP:30Hz NHP:58Hz NLP:62Hz。新增滤波仍在两者执行。

已验证分块一致性、未来数据独立性、标签与 QC 独立性、重复目标字符的边界、不完整 epoch 保留、持续坏通道待处理及非有限值恢复。未加入 ICA、伪迹来源分类、坏通道插值、模型训练、动态停止或 baseline correction。

<!-- swlda-rejection-experiment-v1 -->

## SWLDA epoch rejection comparison (Train development only)

Run `python scripts/run_swlda_rejection_experiment.py` from the repository root. Configuration: `configs/swlda_rejection_experiment.json`. In restricted Windows environments, set `$env:_MNE_FAKE_HOME_DIR=$PWD.Path` before running. View the fully executed `notebooks/14_bigp3bci_swlda_rejection_experiment.ipynb` for figures and step-by-step explanations. The notebook builder is `scripts/build_swlda_experiment_notebook.py`; rebuilding creates an unexecuted notebook.

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

Then execute `notebooks/15_bigp3bci_swlda_full_three_sessions.ipynb`. Outputs are separate from v1. The combined six-subject figure shows the mean and sample SD over all three Sessions; character denominators and no-output counts remain visible in tables. Independent EDFs use separate sample clocks and filter state, so cross-file elapsed time is not fabricated. Expanded coverage also increases the per-Session calibration budget relative to the small pilot; differences from v1 cannot be attributed to coverage alone.
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

Execute `notebooks/16_bigp3bci_swlda_all_studyf.ipynb` for the full report and combined ten-subject error-bar plot (three Session means +/- sample SD). Detailed scores, selected features, models, fixed character splits, subject/Session results and paired comparisons live in `outputs/swlda_all_studyf_v3`. These remain development results, without detector retuning or final Test evaluation.
<!-- /all-studyf-experiment -->
