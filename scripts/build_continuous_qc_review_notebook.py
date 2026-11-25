"""Build the per-EDF manual continuous-EEG QC notebook.

This is a mechanical notebook generator: candidate detection is implemented in
``src/bigp3bci_continuous_qc.py`` and its saved tables determine which EDFs get
review cells.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "artifact_qc" / "continuous_manual_qc"
FULL_OUTPUT = OUTPUT_ROOT / "full"
NOTEBOOK_PATH = (
    PROJECT_ROOT / "notebooks" / "04_bigp3bci_continuous_eeg_manual_qc.ipynb"
)
ARTIFACT_NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "03_bigp3bci_artifact_qc.ipynb"

PILOT_FILES = [
    "F_05_SE001_CBCol_Train02.edf",
    "F_03_SE001_CBCol_Train01.edf",
    "F_25_SE001_CBCol_Train01.edf",
]

DUKE_SUBJECTS = ["F_05", "F_06", "F_07", "F_08"]
ETSU_SUBJECTS = ["F_03", "F_20", "F_21", "F_23", "F_24", "F_25"]


def markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def review_cell(file_name: str, edf_path: str) -> dict:
    return code(
        f"""# {file_name}
review_raw, review_browser, review_table = open_edf_for_manual_review(
    Path({edf_path!r}), merged_review, duration=20.0
)
display(review_table)
"""
    )


def manual_annotation_cell(file_name: str, edf_path: str) -> dict:
    return code(
        f"""# Manual-only window for {file_name}
manual_file_name = {file_name!r}
manual_raw, manual_browser, manual_annotation_file = open_edf_for_manual_annotation(
    Path({edf_path!r}), MANUAL_OUTPUT, duration=20.0
)
print('Manual annotation file:', manual_annotation_file)
"""
    )


def manual_save_cell(file_name: str) -> dict:
    return code(
        f"""# Run after annotating {file_name} in the manual-only Qt window
if manual_file_name != {file_name!r}:
    raise RuntimeError('manual_raw belongs to another EDF; rerun the manual-only cell above first.')
manual_annotation_file, manual_review_file, manual_review_table = save_manual_annotation_session(
    manual_raw, manual_file_name, MANUAL_OUTPUT
)
display(manual_review_table)
"""
    )


def append_edf_review_group(
    cells: list[dict], file_name: str, edf_path: str
) -> None:
    cells.append(markdown("##### 自动候选窗口与候选对照表"))
    cells.append(review_cell(file_name, edf_path))
    cells.append(
        markdown(
            """##### 纯人工标注窗口（不加载自动候选）

此 cell 会重新读取 EDF，只加载此前保存的 `MANUAL_...` annotation，并保留分类型的 flash 事件线：绿色为 non-target，红色为 target，灰色为无法识别的 `StimulusType`。按 `a` 进入 MNE annotation 模式。建议人工描述以 `MANUAL_` 开头；若使用 MNE 默认的 `BAD_` 或普通名称，保存时会自动转换成 `MANUAL_...`。候选表只作为上方参考，不会进入这份 raw。"""
        )
    )
    cells.append(manual_annotation_cell(file_name, edf_path))
    cells.append(
        markdown(
            """##### 保存该 EDF 的人工 annotation 与复核表

完成当前文件标注后运行。它保存 `MANUAL_...` annotation，并自动规范化 MNE 默认的 `BAD_...` 名称；复核表中的 `affected_channels`、`certainty`、`rationale` 可随后填写。重新保存相同区间时会保留已有文字记录。"""
        )
    )
    cells.append(manual_save_cell(file_name))


def remove_manual_qc_tables_from_artifact_notebook() -> None:
    """Keep notebook 03 focused on visual artifact overview only."""
    notebook = json.loads(ARTIFACT_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    remove_markers = (
        "## 8. QC 记录模板",
        "qc_template=pd.DataFrame",
        "## 9. 单个 Train/CBCol EDF：连续 EEG 质量筛查",
        "pilot_name, pilot_raw = next(iter(raw_train.items()))",
        "## 10. 单个 EDF：短窗峰峰值异常与候选 EOG",
        "pilot_qc_screening = pd.DataFrame",
    )
    kept = []
    removed = []
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if any(marker in source for marker in remove_markers):
            removed.append(source.splitlines()[0] if source.splitlines() else "empty")
        else:
            kept.append(cell)
    notebook["cells"] = kept
    ARTIFACT_NOTEBOOK_PATH.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"Removed {len(removed)} manual-QC/table cells from notebook 03")


def main() -> None:
    remove_manual_qc_tables_from_artifact_notebook()
    review_csv = FULL_OUTPUT / "manual_review_candidates.csv"
    if not review_csv.exists():
        raise FileNotFoundError(
            f"Run the full candidate scan before building the notebook: {review_csv}"
        )
    review = pd.read_csv(review_csv)
    files = (
        review[["subject", "session", "file", "edf_path"]]
        .drop_duplicates()
        .sort_values(["subject", "session", "file"])
    )
    path_by_name = dict(zip(files["file"], files["edf_path"]))
    missing_pilots = [name for name in PILOT_FILES if name not in path_by_name]
    if missing_pilots:
        raise RuntimeError(f"Pilot files have no candidates: {missing_pilots}")

    cells = [
        code("%matplotlib qt\n"),
        markdown(
            """# bigP3BCI Study F：连续 EEG 人工质量复核

本 notebook 仅处理 `Train/CBCol`，每次运行一个 EDF cell。自动检测只生成候选，不修改 EDF、不设置 `raw.info['bads']`，也不生成 `BAD_` annotation。

候选颜色：大幅异常为红色、平直片段为橙色、疑似眨眼为蓝色、多类型重叠为紫色。混合 annotation 会直接写出组成，例如 `CANDIDATE_MIXED__large_amplitude+blink`。flash 事件由 `StimulusBegin` 上升沿定位，并由同一采样点的 `StimulusType` 分类：绿色竖线为 non-target（0），红色竖线为 target（1），灰色竖线表示意外取值。事件线只用于判断波形是否与刺激锁定，不属于异常 annotation。自动候选中的大幅异常也是红色，但表现为有宽度的 annotation 区间，可与 target 细竖线区分。MNE 1.12.1 支持 `annotation_colors`；若更换到不支持该参数的旧版本，仍可通过 `CANDIDATE_...` 名称和下方对照表识别。"""
        ),
        code(
            """from pathlib import Path
import importlib
import sys
import numpy as np
import pandas as pd
import mne

PROJECT_ROOT = Path.cwd()
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / 'README.md').exists() or (PROJECT_ROOT / '.git').exists()):
    PROJECT_ROOT = PROJECT_ROOT.parent
if not ((PROJECT_ROOT / 'README.md').exists() or (PROJECT_ROOT / '.git').exists()):
    raise FileNotFoundError('请在项目目录或子目录启动 Jupyter。')

SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import bigp3bci_continuous_qc as continuous_qc
importlib.reload(continuous_qc)

from bigp3bci_continuous_qc import (
    ANNOTATION_COLORS,
    DEFAULT_CONFIG,
    discover_train_cbcol,
    open_edf_for_manual_review,
    open_edf_for_manual_annotation,
    save_manual_annotation_session,
    scan_many_edfs,
)

STUDY_F_ROOT = PROJECT_ROOT / 'data' / 'raw' / 'bigP3BCI-data' / 'StudyF'
QC_ROOT = PROJECT_ROOT / 'outputs' / 'artifact_qc' / 'continuous_manual_qc'
FULL_OUTPUT = QC_ROOT / 'full'
PILOT_OUTPUT = QC_ROOT / 'pilot'
MANUAL_OUTPUT = QC_ROOT / 'manual'
TRAIN_FILES = discover_train_cbcol(STUDY_F_ROOT)

print('MNE:', mne.__version__)
print('Train/CBCol EDF files:', len(TRAIN_FILES))
print('Candidate annotation colors:', ANNOTATION_COLORS)
print('Automatic candidates are read-only references.')
"""
        ),
        markdown(
            """## 自动筛查参数与重新计算

峰峰值、标准差和参考值均以 µV 计算。默认使用 2 秒窗口、1 秒步长、`robust z >= 6`；平直候选阈值为峰峰值不超过 0.5 µV 或标准差不超过 0.1 µV。`EEG_Fz` 的眨眼候选由 `find_eog_events` 产生，并使用 100 µV 阈值和 0.75 秒最小间隔以减少刺激节律造成的误报。

通常直接读取已经计算并验证过的表格。只有修改参数后，才把 `RECOMPUTE_ALL` 改为 `True`。"""
        ),
        code(
            """QC_CONFIG = {
    **DEFAULT_CONFIG,
    'window_sec': 2.0,
    'step_sec': 1.0,
    'large_robust_z': 6.0,
    'flat_ptp_uv': 0.5,
    'flat_std_uv': 0.1,
    'persistent_ratio': 0.20,
    'blink_threshold_uv': 100.0,
    'blink_min_separation_sec': 0.75,
    'merge_gap_sec': 0.25,
}

RECOMPUTE_ALL = False
if RECOMPUTE_ALL:
    candidate_details, merged_review, channel_summary, file_summary, annotation_verification = scan_many_edfs(
        TRAIN_FILES, FULL_OUTPUT, QC_CONFIG
    )
else:
    candidate_details = pd.read_csv(FULL_OUTPUT / 'candidate_details.csv')
    merged_review = pd.read_csv(FULL_OUTPUT / 'manual_review_candidates.csv', keep_default_na=False)
    channel_summary = pd.read_csv(FULL_OUTPUT / 'channel_summary.csv')
    file_summary = pd.read_csv(FULL_OUTPUT / 'file_summary.csv', keep_default_na=False)
    annotation_verification = pd.read_csv(FULL_OUTPUT / 'candidate_annotation_verification.csv')

INSTITUTION_BY_SUBJECT = {
    **{subject: 'Duke University' for subject in ['F_05', 'F_06', 'F_07', 'F_08']},
    **{subject: 'East Tennessee State University' for subject in ['F_03', 'F_20', 'F_21', 'F_23', 'F_24', 'F_25']},
}
file_summary.insert(0, 'institution', file_summary['subject'].map(INSTITUTION_BY_SUBJECT))

print('Files with candidates:', merged_review['file'].nunique())
print('Raw candidate details:', len(candidate_details))
print('Merged review segments:', len(merged_review))
print('All candidate annotations reload correctly:', bool(annotation_verification['reload_ok'].all()))
display(file_summary)
"""
        ),
        markdown(
            """## 自动候选与人工结果严格分离

上方自动候选表只作为只读参考，不在其中填写或保存人工结论。每个 EDF 的人工标注必须在其“纯人工标注窗口”中完成，并保存到独立的 `manual` 输出目录。绿色 non-target 与红色 target 事件线只用于判断波形是否与刺激锁定。"""
        ),
        markdown(
            """## 三个试点 EDF

请先运行下面三个 cell，确认 Qt 窗口可以滚动、16 个 EEG 通道均显示、annotation 与对照表一致。试点也按机构分组；每次只运行一个 cell，关闭前一个窗口后再运行下一个。"""
        ),
    ]

    cells.append(markdown("### Duke University 试点"))
    for file_name in [name for name in PILOT_FILES if name.startswith("F_05_")]:
        cells.append(markdown(f"#### 试点：`{file_name}`"))
        append_edf_review_group(cells, file_name, path_by_name[file_name])

    cells.append(markdown("### East Tennessee State University（ETSU）试点"))
    for file_name in [name for name in PILOT_FILES if not name.startswith("F_05_")]:
        cells.append(markdown(f"#### 试点：`{file_name}`"))
        append_edf_review_group(cells, file_name, path_by_name[file_name])

    cells.append(
        markdown(
            """## 其余有候选的 Train/CBCol EDF

以下先按机构分组，再按 subject、session、文件名排序。三个试点文件不重复列出。每个 EDF 单独占一个可运行 cell。"""
        )
    )
    institution_groups = [
        ("Duke University", DUKE_SUBJECTS),
        ("East Tennessee State University（ETSU）", ETSU_SUBJECTS),
    ]
    for institution_name, subjects in institution_groups:
        cells.append(markdown(f"### {institution_name}"))
        institution_files = files[files["subject"].isin(subjects)]
        for row in institution_files.itertuples(index=False):
            if row.file in PILOT_FILES:
                continue
            cells.append(markdown(f"#### `{row.subject}` / `{row.session}` / `{row.file}`"))
            append_edf_review_group(cells, row.file, row.edf_path)

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "bigp3bci",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"Wrote {NOTEBOOK_PATH}")
    print(f"Candidate EDF review cells: {len(files)}")


if __name__ == "__main__":
    main()
