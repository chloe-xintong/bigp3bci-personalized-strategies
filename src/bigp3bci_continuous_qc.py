"""Candidate generation and manual review helpers for bigP3BCI Study F.

The functions in this module never modify EDF files, set ``raw.info['bads']``,
or create ``BAD_`` annotations. Automatic detections are candidates only.
"""

from __future__ import annotations

from pathlib import Path
import re

import mne
import numpy as np
import pandas as pd


EVENT_CHANNELS = {
    "StimulusBegin",
    "StimulusType",
    "StimulusCode",
    "CurrentTarget",
    "FakeFeedback",
    "DisplayResults",
    "DisplayResult",
    "SelectedTarget",
    "SelectedRow",
    "SelectedColumn",
    "PhaseInSequence",
}

ANNOTATION_COLORS = {
    "CANDIDATE_large_amplitude": "#d62728",
    "CANDIDATE_flat": "#ff7f0e",
    "CANDIDATE_blink": "#1f77b4",
    "CANDIDATE_MIXED": "#9467bd",
}

MANUAL_REVIEW_OPTIONS = (
    "confirmed_bad_span",
    "bad_channel",
    "possible_blink",
    "normal",
    "uncertain",
)

DEFAULT_CONFIG = {
    "window_sec": 2.0,
    "step_sec": 1.0,
    "large_robust_z": 6.0,
    "robust_scale_floor_uv": 0.05,
    "flat_ptp_uv": 0.5,
    "flat_std_uv": 0.1,
    "persistent_ratio": 0.20,
    "blink_pre_sec": 0.5,
    "blink_post_sec": 0.8,
    "blink_threshold_uv": 100.0,
    "blink_min_separation_sec": 0.75,
    "merge_gap_sec": 0.25,
}

DETAIL_COLUMNS = [
    "candidate_detail_id",
    "subject",
    "session",
    "file",
    "edf_path",
    "start_sec",
    "end_sec",
    "duration_sec",
    "trigger_channel",
    "candidate_type",
    "ptp_uv",
    "std_uv",
    "reference_ptp_uv",
    "robust_scale_uv",
    "robust_z",
    "detection_value",
]


def assign_bigp3bci_channel_types(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Apply the channel-type rules already used in the project notebooks."""
    channel_types: dict[str, str] = {}
    for name in raw.ch_names:
        if name in EVENT_CHANNELS:
            channel_types[name] = "stim"
        elif re.match(r"^.+_\d+_\d+$", name):
            channel_types[name] = "misc"
        elif name.upper().startswith("EOG"):
            channel_types[name] = "eog"
    if channel_types:
        raw.set_channel_types(
            channel_types, on_unit_change="ignore", verbose=False
        )
    return raw


def read_bigp3bci_edf(
    edf_path: str | Path, *, preload: bool = False
) -> mne.io.BaseRaw:
    """Read an EDF with the same channel typing used elsewhere in the project."""
    raw = mne.io.read_raw_edf(
        Path(edf_path),
        preload=preload,
        infer_types=True,
        verbose="ERROR",
    )
    return assign_bigp3bci_channel_types(raw)


def get_all_eeg_channels(raw: mne.io.BaseRaw) -> list[str]:
    """Return only published EEG_* channels; task-state channels are excluded."""
    channel_types = raw.get_channel_types()
    eeg_channels = [
        name
        for name, channel_type in zip(raw.ch_names, channel_types)
        if channel_type == "eeg" and name.startswith("EEG_")
    ]
    if len(eeg_channels) != 16:
        raise ValueError(
            f"Expected 16 EEG_* channels, found {len(eeg_channels)}: {eeg_channels}"
        )
    return eeg_channels


def discover_train_cbcol(study_f_root: str | Path) -> list[Path]:
    """Find all Study F Train/CBCol EDF files in a stable order."""
    root = Path(study_f_root)
    files = sorted(root.glob("F_*/SE*/Train/CBCol/*.edf"))
    return files


def _subject_session(edf_path: Path) -> tuple[str, str]:
    return edf_path.parts[-5], edf_path.parts[-4]


def _empty_detail_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=DETAIL_COLUMNS)


def scan_edf_candidates(
    edf_path: str | Path,
    config: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate amplitude, flat-line and blink candidates for one EDF.

    EEG values returned by MNE are converted from volts to microvolts before
    statistics are calculated or written to tables.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    edf_path = Path(edf_path).resolve()
    subject, session = _subject_session(edf_path)
    raw = read_bigp3bci_edf(edf_path, preload=False)
    eeg_channels = get_all_eeg_channels(raw)
    sfreq = float(raw.info["sfreq"])
    duration_sec = raw.n_times / sfreq
    data_uv = raw.get_data(
        picks=eeg_channels,
        reject_by_annotation=None,
        verbose=False,
    ) * 1e6

    window_samples = max(1, int(round(cfg["window_sec"] * sfreq)))
    step_samples = max(1, int(round(cfg["step_sec"] * sfreq)))
    starts = np.arange(0, data_uv.shape[1] - window_samples + 1, step_samples)
    if len(starts) == 0:
        raw.close()
        raise ValueError(f"Recording is shorter than one QC window: {edf_path}")

    ptp_uv = np.empty((len(starts), len(eeg_channels)), dtype=float)
    std_uv = np.empty_like(ptp_uv)
    for row_index, start in enumerate(starts):
        window = data_uv[:, start : start + window_samples]
        ptp_uv[row_index] = np.ptp(window, axis=1)
        std_uv[row_index] = np.std(window, axis=1)

    reference_ptp_uv = np.median(ptp_uv, axis=0)
    mad_ptp_uv = np.median(
        np.abs(ptp_uv - reference_ptp_uv[np.newaxis, :]), axis=0
    )
    robust_scale_uv = np.maximum(
        1.4826 * mad_ptp_uv,
        float(cfg["robust_scale_floor_uv"]),
    )
    robust_z = (
        ptp_uv - reference_ptp_uv[np.newaxis, :]
    ) / robust_scale_uv[np.newaxis, :]

    large_mask = robust_z >= float(cfg["large_robust_z"])
    flat_mask = (ptp_uv <= float(cfg["flat_ptp_uv"])) | (
        std_uv <= float(cfg["flat_std_uv"])
    )

    detail_rows: list[dict] = []
    detail_counter = 0

    def add_window_candidate(row_i: int, ch_i: int, candidate_type: str) -> None:
        nonlocal detail_counter
        detail_counter += 1
        start_sec = starts[row_i] / sfreq
        end_sec = min(duration_sec, start_sec + float(cfg["window_sec"]))
        if candidate_type == "CANDIDATE_large_amplitude":
            value = (
                f"ptp={ptp_uv[row_i, ch_i]:.3f} µV; "
                f"reference={reference_ptp_uv[ch_i]:.3f} µV; "
                f"robust_z={robust_z[row_i, ch_i]:.2f}"
            )
        else:
            value = (
                f"ptp={ptp_uv[row_i, ch_i]:.3f} µV; "
                f"std={std_uv[row_i, ch_i]:.3f} µV"
            )
        detail_rows.append(
            {
                "candidate_detail_id": f"{edf_path.stem}_D{detail_counter:05d}",
                "subject": subject,
                "session": session,
                "file": edf_path.name,
                "edf_path": str(edf_path),
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "duration_sec": float(end_sec - start_sec),
                "trigger_channel": eeg_channels[ch_i],
                "candidate_type": candidate_type,
                "ptp_uv": float(ptp_uv[row_i, ch_i]),
                "std_uv": float(std_uv[row_i, ch_i]),
                "reference_ptp_uv": float(reference_ptp_uv[ch_i]),
                "robust_scale_uv": float(robust_scale_uv[ch_i]),
                "robust_z": float(robust_z[row_i, ch_i]),
                "detection_value": value,
            }
        )

    for row_i, ch_i in zip(*np.where(large_mask)):
        add_window_candidate(row_i, ch_i, "CANDIDATE_large_amplitude")
    for row_i, ch_i in zip(*np.where(flat_mask)):
        add_window_candidate(row_i, ch_i, "CANDIDATE_flat")

    try:
        eog_events = mne.preprocessing.find_eog_events(
            raw,
            ch_name="EEG_Fz",
            reject_by_annotation=False,
            thresh=float(cfg["blink_threshold_uv"]) * 1e-6,
            verbose=False,
        )
    except Exception as exc:  # keep batch scanning alive and record the limitation
        eog_events = np.empty((0, 3), dtype=int)
        eog_error = repr(exc)
    else:
        eog_error = ""

    retained_eog_events = []
    previous_event_sec = -np.inf
    for event in eog_events:
        event_sec = (int(event[0]) - int(raw.first_samp)) / sfreq
        if (
            event_sec - previous_event_sec
            < float(cfg["blink_min_separation_sec"])
        ):
            continue
        retained_eog_events.append(event)
        previous_event_sec = event_sec

    for event in retained_eog_events:
        event_sec = (int(event[0]) - int(raw.first_samp)) / sfreq
        start_sec = max(0.0, event_sec - float(cfg["blink_pre_sec"]))
        end_sec = min(duration_sec, event_sec + float(cfg["blink_post_sec"]))
        detail_counter += 1
        detail_rows.append(
            {
                "candidate_detail_id": f"{edf_path.stem}_D{detail_counter:05d}",
                "subject": subject,
                "session": session,
                "file": edf_path.name,
                "edf_path": str(edf_path),
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "duration_sec": float(end_sec - start_sec),
                "trigger_channel": "EEG_Fz",
                "candidate_type": "CANDIDATE_blink",
                "ptp_uv": np.nan,
                "std_uv": np.nan,
                "reference_ptp_uv": np.nan,
                "robust_scale_uv": np.nan,
                "robust_z": np.nan,
                "detection_value": (
                    f"find_eog_events peak at {event_sec:.3f} s; "
                    f"threshold={float(cfg['blink_threshold_uv']):.1f} µV"
                ),
            }
        )

    channel_rows = []
    for ch_i, channel in enumerate(eeg_channels):
        large_count = int(large_mask[:, ch_i].sum())
        flat_count = int(flat_mask[:, ch_i].sum())
        any_count = int((large_mask[:, ch_i] | flat_mask[:, ch_i]).sum())
        anomaly_ratio = any_count / len(starts)
        channel_rows.append(
            {
                "subject": subject,
                "session": session,
                "file": edf_path.name,
                "edf_path": str(edf_path),
                "channel": channel,
                "n_windows": int(len(starts)),
                "large_amplitude_windows": large_count,
                "flat_windows": flat_count,
                "anomaly_windows": any_count,
                "anomaly_window_ratio": float(anomaly_ratio),
                "persistent_channel_candidate": bool(
                    anomaly_ratio >= float(cfg["persistent_ratio"])
                ),
                "median_ptp_uv": float(reference_ptp_uv[ch_i]),
                "robust_scale_uv": float(robust_scale_uv[ch_i]),
                "median_std_uv": float(np.median(std_uv[:, ch_i])),
                "eog_detection_error": eog_error if channel == "EEG_Fz" else "",
            }
        )

    raw.close()
    detail = (
        pd.DataFrame(detail_rows, columns=DETAIL_COLUMNS)
        if detail_rows
        else _empty_detail_frame()
    )
    channel_summary = pd.DataFrame(channel_rows)
    return detail, channel_summary


def merge_candidate_intervals(
    detail: pd.DataFrame,
    *,
    merge_gap_sec: float = DEFAULT_CONFIG["merge_gap_sec"],
) -> pd.DataFrame:
    """Merge overlapping/adjacent candidate intervals within each EDF."""
    review_columns = [
        "candidate_id",
        "subject",
        "session",
        "file",
        "edf_path",
        "start_sec",
        "end_sec",
        "duration_sec",
        "trigger_channels",
        "candidate_types",
        "detection_values",
        "source_candidate_ids",
        "annotation_description",
        "manual_review",
        "manual_notes",
        "reviewed_start_sec",
        "reviewed_end_sec",
    ]
    if detail.empty:
        return pd.DataFrame(columns=review_columns)

    merged_rows: list[dict] = []
    for (_, _, file_name), group in detail.groupby(
        ["subject", "session", "file"], sort=True
    ):
        group = group.sort_values(["start_sec", "end_sec", "candidate_detail_id"])
        current: list[pd.Series] = []
        current_end = -np.inf
        groups: list[list[pd.Series]] = []
        for _, row in group.iterrows():
            if current and float(row["start_sec"]) > current_end + merge_gap_sec:
                groups.append(current)
                current = []
                current_end = -np.inf
            current.append(row)
            current_end = max(current_end, float(row["end_sec"]))
        if current:
            groups.append(current)

        for index, rows in enumerate(groups, start=1):
            types = sorted({str(row["candidate_type"]) for row in rows})
            channels = sorted({str(row["trigger_channel"]) for row in rows})
            start_sec = min(float(row["start_sec"]) for row in rows)
            end_sec = max(float(row["end_sec"]) for row in rows)
            if len(types) == 1:
                annotation_description = types[0]
            else:
                mixed_labels = [
                    candidate_type.removeprefix("CANDIDATE_")
                    for candidate_type in types
                ]
                annotation_description = (
                    "CANDIDATE_MIXED__" + "+".join(mixed_labels)
                )
            first = rows[0]
            merged_rows.append(
                {
                    "candidate_id": f"{Path(file_name).stem}_C{index:04d}",
                    "subject": first["subject"],
                    "session": first["session"],
                    "file": file_name,
                    "edf_path": first["edf_path"],
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "duration_sec": end_sec - start_sec,
                    "trigger_channels": ";".join(channels),
                    "candidate_types": ";".join(types),
                    "detection_values": " | ".join(
                        str(row["detection_value"]) for row in rows
                    ),
                    "source_candidate_ids": ";".join(
                        str(row["candidate_detail_id"]) for row in rows
                    ),
                    "annotation_description": annotation_description,
                    "manual_review": "",
                    "manual_notes": "",
                    "reviewed_start_sec": np.nan,
                    "reviewed_end_sec": np.nan,
                }
            )
    return pd.DataFrame(merged_rows, columns=review_columns)


def build_file_summary(
    detail: pd.DataFrame,
    merged: pd.DataFrame,
    channel_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize candidate counts and channel anomaly ratios per EDF."""
    file_keys = channel_summary[
        ["subject", "session", "file", "edf_path"]
    ].drop_duplicates()
    rows = []
    for _, key in file_keys.iterrows():
        file_detail = detail[detail["file"] == key["file"]]
        file_merged = merged[merged["file"] == key["file"]]
        file_channels = channel_summary[channel_summary["file"] == key["file"]]
        counts = file_detail["candidate_type"].value_counts()
        persistent = file_channels.loc[
            file_channels["persistent_channel_candidate"], "channel"
        ].tolist()
        rows.append(
            {
                **key.to_dict(),
                "n_large_amplitude_details": int(
                    counts.get("CANDIDATE_large_amplitude", 0)
                ),
                "n_flat_details": int(counts.get("CANDIDATE_flat", 0)),
                "n_blink_details": int(counts.get("CANDIDATE_blink", 0)),
                "n_raw_candidate_details": int(len(file_detail)),
                "n_merged_review_segments": int(len(file_merged)),
                "persistent_candidate_channels": ";".join(persistent),
                "max_channel_anomaly_ratio": float(
                    file_channels["anomaly_window_ratio"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def annotations_from_review(
    review: pd.DataFrame,
    *,
    orig_time=None,
) -> mne.Annotations:
    """Create non-BAD candidate annotations from a merged review table."""
    if review.empty:
        return mne.Annotations([], [], [], orig_time=orig_time)
    return mne.Annotations(
        onset=review["start_sec"].astype(float).to_numpy(),
        duration=(
            review["end_sec"].astype(float) - review["start_sec"].astype(float)
        ).to_numpy(),
        description=review["annotation_description"].astype(str).to_numpy(),
        orig_time=orig_time,
    )


def candidate_annotation_path(output_dir: str | Path, file_name: str) -> Path:
    return Path(output_dir) / "candidate_annotations" / (
        f"{Path(file_name).stem}-candidate-annot.fif"
    )


def stimulus_begin_events(raw: mne.io.BaseRaw) -> np.ndarray:
    """Return flash onsets coded by StimulusType for browser display.

    Event code 1 is a non-target flash (StimulusType=0), event code 2 is a
    target flash (StimulusType=1), and event code 3 is an unexpected value.
    """
    if "StimulusBegin" not in raw.ch_names:
        return np.empty((0, 3), dtype=int)
    picks = ["StimulusBegin"]
    if "StimulusType" in raw.ch_names:
        picks.append("StimulusType")
    data = raw.get_data(
        picks=picks,
        reject_by_annotation=None,
        verbose=False,
    )
    values = data[0]
    rising_samples = np.flatnonzero(
        (values[:-1] <= 0.5) & (values[1:] > 0.5)
    ) + 1
    if len(picks) == 1:
        event_codes = np.full(len(rising_samples), 3, dtype=int)
    else:
        stimulus_type = np.rint(data[1, rising_samples]).astype(int)
        event_codes = np.full(len(rising_samples), 3, dtype=int)
        event_codes[stimulus_type == 0] = 1
        event_codes[stimulus_type == 1] = 2
    return np.column_stack(
        [
            rising_samples + int(raw.first_samp),
            np.zeros(len(rising_samples), dtype=int),
            event_codes,
        ]
    ).astype(int)


def flash_event_display_settings(events: np.ndarray) -> tuple[dict, dict]:
    """Return event labels and colors for available flash event codes."""
    present = {int(code) for code in events[:, 2]} if len(events) else set()
    labels = {
        1: "non-target_flash",
        2: "target_flash",
        3: "unknown_flash_type",
    }
    colors = {
        1: "#00aa44",
        2: "#ff0000",
        3: "#666666",
    }
    event_id = {labels[code]: code for code in sorted(present)}
    event_color = {code: colors[code] for code in sorted(present)}
    return event_id, event_color


def save_candidate_annotations(
    merged: pd.DataFrame,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Save and reload one candidate annotation file per EDF."""
    output_dir = Path(output_dir)
    annotation_dir = output_dir / "candidate_annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    verification_rows = []
    for file_name, review in merged.groupby("file", sort=True):
        annotations = annotations_from_review(review)
        path = candidate_annotation_path(output_dir, file_name)
        annotations.save(path, overwrite=True, verbose=False)
        reloaded = mne.read_annotations(path)
        verification_rows.append(
            {
                "file": file_name,
                "annotation_path": str(path.resolve()),
                "saved_count": len(annotations),
                "reloaded_count": len(reloaded),
                "reload_ok": len(annotations) == len(reloaded),
            }
        )
    return pd.DataFrame(verification_rows)


def scan_many_edfs(
    edf_paths: list[str | Path],
    output_dir: str | Path,
    config: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scan EDFs, save candidate tables/annotations, and return all outputs."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    details = []
    channel_summaries = []
    for index, edf_path in enumerate(sorted(map(Path, edf_paths)), start=1):
        print(f"[{index}/{len(edf_paths)}] {edf_path.name}")
        detail, channel_summary = scan_edf_candidates(edf_path, cfg)
        details.append(detail)
        channel_summaries.append(channel_summary)
    detail_all = (
        pd.concat(details, ignore_index=True) if details else _empty_detail_frame()
    )
    channel_all = (
        pd.concat(channel_summaries, ignore_index=True)
        if channel_summaries
        else pd.DataFrame()
    )
    merged = merge_candidate_intervals(
        detail_all, merge_gap_sec=float(cfg["merge_gap_sec"])
    )
    file_summary = build_file_summary(detail_all, merged, channel_all)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_all.to_csv(
        output_dir / "candidate_details.csv", index=False, encoding="utf-8-sig"
    )
    merged.to_csv(
        output_dir / "manual_review_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    channel_all.to_csv(
        output_dir / "channel_summary.csv", index=False, encoding="utf-8-sig"
    )
    file_summary.to_csv(
        output_dir / "file_summary.csv", index=False, encoding="utf-8-sig"
    )
    per_edf_dir = output_dir / "per_edf"
    per_edf_dir.mkdir(parents=True, exist_ok=True)
    for file_name in channel_all["file"].drop_duplicates().sort_values():
        stem = Path(file_name).stem
        detail_all[detail_all["file"] == file_name].to_csv(
            per_edf_dir / f"{stem}_candidate_details.csv",
            index=False,
            encoding="utf-8-sig",
        )
        merged[merged["file"] == file_name].to_csv(
            per_edf_dir / f"{stem}_manual_review_candidates.csv",
            index=False,
            encoding="utf-8-sig",
        )
        channel_all[channel_all["file"] == file_name].to_csv(
            per_edf_dir / f"{stem}_channel_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        file_summary[file_summary["file"] == file_name].to_csv(
            per_edf_dir / f"{stem}_file_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
    verification = save_candidate_annotations(merged, output_dir)
    verification.to_csv(
        output_dir / "candidate_annotation_verification.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return detail_all, merged, channel_all, file_summary, verification


def open_edf_for_manual_review(
    edf_path: str | Path,
    review_table: pd.DataFrame,
    *,
    duration: float = 20.0,
    show: bool = True,
) -> tuple[mne.io.BaseRaw, object, pd.DataFrame]:
    """Open one EDF with candidate annotations in the interactive MNE browser."""
    edf_path = Path(edf_path).resolve()
    rows = review_table[review_table["file"] == edf_path.name].copy()
    if rows.empty:
        raise ValueError(f"No candidates found for {edf_path.name}")
    raw = read_bigp3bci_edf(edf_path, preload=False)
    annotations = annotations_from_review(
        rows, orig_time=raw.annotations.orig_time
    )
    raw.set_annotations(raw.annotations + annotations)
    flash_events = stimulus_begin_events(raw)
    flash_event_id, flash_event_colors = flash_event_display_settings(
        flash_events
    )
    start = max(0.0, float(rows["start_sec"].min()) - 5.0)
    present_descriptions = set(rows["annotation_description"])
    colors = {}
    for description in present_descriptions:
        if description.startswith("CANDIDATE_MIXED__"):
            colors[description] = ANNOTATION_COLORS["CANDIDATE_MIXED"]
        elif description in ANNOTATION_COLORS:
            colors[description] = ANNOTATION_COLORS[description]
    browser = raw.plot(
        events=flash_events,
        event_id=flash_event_id,
        event_color=flash_event_colors,
        start=start,
        duration=duration,
        picks="eeg",
        n_channels=16,
        remove_dc=False,
        scalings="auto",
        show_scrollbars=True,
        annotation_colors=colors,
        annotation_regex=r"^CANDIDATE_",
        title=f"{edf_path.name}: continuous EEG candidate review",
        show=show,
        block=False,
    )
    target_count = int(np.sum(flash_events[:, 2] == 2))
    non_target_count = int(np.sum(flash_events[:, 2] == 1))
    unknown_count = int(np.sum(flash_events[:, 2] == 3))
    print(
        f"Flash markers: total={len(flash_events)}, "
        f"target={target_count}, non-target={non_target_count}, "
        f"unknown={unknown_count} (display only)"
    )
    columns = [
        "candidate_id",
        "file",
        "start_sec",
        "end_sec",
        "trigger_channels",
        "candidate_types",
        "annotation_description",
        "detection_values",
        "manual_review",
        "manual_notes",
        "reviewed_start_sec",
        "reviewed_end_sec",
    ]
    return raw, browser, rows[columns]


def manual_annotation_path(output_dir: str | Path, file_name: str) -> Path:
    """Return the independent manual-annotation path for one EDF."""
    return Path(output_dir) / "manual_annotations" / (
        f"{Path(file_name).stem}-manual-annot.fif"
    )


def _copy_annotations_with_orig_time(
    annotations: mne.Annotations,
    orig_time,
) -> mne.Annotations:
    """Copy annotations while aligning them to an EDF's measurement time."""
    return mne.Annotations(
        onset=np.asarray(annotations.onset, dtype=float),
        duration=np.asarray(annotations.duration, dtype=float),
        description=np.asarray(annotations.description, dtype=str),
        orig_time=orig_time,
        ch_names=list(annotations.ch_names),
    )


def open_edf_for_manual_annotation(
    edf_path: str | Path,
    output_dir: str | Path,
    *,
    duration: float = 20.0,
    show: bool = True,
) -> tuple[mne.io.BaseRaw, object, Path]:
    """Open a clean EDF copy for manual annotation only.

    Automatic ``CANDIDATE_`` annotations are never loaded. If a manual file
    already exists, only its ``MANUAL_`` annotations are restored.
    """
    edf_path = Path(edf_path).resolve()
    raw = read_bigp3bci_edf(edf_path, preload=False)
    if any(
        str(description).startswith("CANDIDATE_")
        for description in raw.annotations.description
    ):
        raise RuntimeError("Fresh EDF unexpectedly contains CANDIDATE_ annotations")

    path = manual_annotation_path(output_dir, edf_path.name)
    loaded_count = 0
    if path.exists():
        stored = mne.read_annotations(path)
        manual_mask = np.array(
            [
                str(description).startswith("MANUAL_")
                for description in stored.description
            ],
            dtype=bool,
        )
        stored = stored[manual_mask]
        stored = _copy_annotations_with_orig_time(
            stored, raw.annotations.orig_time
        )
        raw.set_annotations(raw.annotations + stored)
        loaded_count = len(stored)

    flash_events = stimulus_begin_events(raw)
    flash_event_id, flash_event_colors = flash_event_display_settings(
        flash_events
    )
    browser = raw.plot(
        events=flash_events,
        event_id=flash_event_id,
        event_color=flash_event_colors,
        start=0.0,
        duration=duration,
        picks="eeg",
        n_channels=16,
        remove_dc=False,
        scalings="auto",
        show_scrollbars=True,
        annotation_regex=r"^(MANUAL_|Begin recording)",
        title=f"{edf_path.name}: MANUAL annotation only (no candidates)",
        show=show,
        block=False,
    )
    target_count = int(np.sum(flash_events[:, 2] == 2))
    non_target_count = int(np.sum(flash_events[:, 2] == 1))
    unknown_count = int(np.sum(flash_events[:, 2] == 3))
    print(
        f"Manual-only window: candidate annotations=0; "
        f"loaded manual annotations={loaded_count}; "
        f"flash markers total={len(flash_events)}, target={target_count}, "
        f"non-target={non_target_count}, unknown={unknown_count}"
    )
    print("Use annotation descriptions beginning with MANUAL_ before saving.")
    return raw, browser, path


def save_manual_annotation_session(
    raw: mne.io.BaseRaw,
    file_name: str,
    output_dir: str | Path,
) -> tuple[Path, Path, pd.DataFrame]:
    """Save manual annotations and a separate review table.

    MNE's annotation editor commonly creates ``BAD_`` labels by default.
    Because these annotations are still only review decisions at this stage,
    they are normalized to ``MANUAL_...`` before saving. EDF/system and
    automatic candidate annotations are excluded.
    """
    output_dir = Path(output_dir)
    annotation_path = manual_annotation_path(output_dir, file_name)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    descriptions = np.asarray(raw.annotations.description, dtype=str)
    system_descriptions = {"Begin recording"}
    manual_mask = np.array(
        [
            description not in system_descriptions
            and not description.startswith("CANDIDATE_")
            and not description.startswith("EDGE_")
            for description in descriptions
        ],
        dtype=bool,
    )
    if not manual_mask.any():
        raise ValueError(
            "No manual time-span annotations found. Press 'a' in the manual-only "
            "MNE window, draw at least one interval, then run this cell again. "
            "Existing manual file was not overwritten."
        )
    selected = raw.annotations[manual_mask]
    normalized_descriptions = []
    renamed = []
    for description in selected.description:
        description = str(description).strip()
        if description.startswith("MANUAL_"):
            normalized = description
        elif description.startswith("BAD_"):
            suffix = description[4:].strip().strip("_") or "bad_span"
            normalized = f"MANUAL_{suffix}"
        elif description == "BAD":
            normalized = "MANUAL_bad_span"
        else:
            suffix = description.replace(" ", "_").strip("_") or "annotation"
            normalized = f"MANUAL_{suffix}"
        normalized_descriptions.append(normalized)
        if normalized != description:
            renamed.append((description, normalized))
    manual_annotations = mne.Annotations(
        onset=np.asarray(selected.onset, dtype=float),
        duration=np.asarray(selected.duration, dtype=float),
        description=np.asarray(normalized_descriptions, dtype=str),
        orig_time=selected.orig_time,
        ch_names=list(selected.ch_names),
    )
    if any(
        str(description).startswith("CANDIDATE_")
        for description in manual_annotations.description
    ):
        raise RuntimeError("Candidate annotations must not enter the manual file")
    manual_annotations.save(annotation_path, overwrite=True, verbose=False)
    reloaded = mne.read_annotations(annotation_path)
    if len(reloaded) != len(manual_annotations):
        raise RuntimeError("Manual annotation save/reload count mismatch")
    if renamed:
        print("Normalized manual annotation labels:")
        for old, new in sorted(set(renamed)):
            print(f"  {old!r} -> {new!r}")

    review_dir = output_dir / "manual_review_tables"
    review_dir.mkdir(parents=True, exist_ok=True)
    review_path = review_dir / f"{Path(file_name).stem}_manual_review.csv"
    previous = (
        pd.read_csv(review_path, keep_default_na=False)
        if review_path.exists()
        else pd.DataFrame()
    )
    previous_by_key = {}
    if not previous.empty:
        for _, row in previous.iterrows():
            key = (
                round(float(row["start_sec"]), 6),
                round(float(row["end_sec"]), 6),
                str(row["type"]),
            )
            previous_by_key[key] = row

    rows = []
    for index, (onset, duration_sec, description, channels) in enumerate(
        zip(
            manual_annotations.onset,
            manual_annotations.duration,
            manual_annotations.description,
            manual_annotations.ch_names,
        ),
        start=1,
    ):
        end_sec = float(onset + duration_sec)
        key = (round(float(onset), 6), round(end_sec, 6), str(description))
        old = previous_by_key.get(key)
        rows.append(
            {
                "manual_id": f"{Path(file_name).stem}_M{index:04d}",
                "file": file_name,
                "start_sec": float(onset),
                "end_sec": end_sec,
                "duration_sec": float(duration_sec),
                "type": str(description),
                "affected_channels": (
                    ";".join(channels)
                    if channels
                    else (str(old["affected_channels"]) if old is not None else "")
                ),
                "certainty": str(old["certainty"]) if old is not None else "",
                "rationale": str(old["rationale"]) if old is not None else "",
            }
        )
    review_table = pd.DataFrame(rows)
    review_table.to_csv(review_path, index=False, encoding="utf-8-sig")
    print("Saved manual annotations:", annotation_path)
    print("Saved manual review table:", review_path)
    return annotation_path, review_path, review_table


def save_reviewed_results(
    reviewed_table: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, pd.DataFrame]:
    """Save manual decisions separately from automatic candidate outputs."""
    output_dir = Path(output_dir)
    reviewed_dir = output_dir / "reviewed_annotations"
    reviewed_dir.mkdir(parents=True, exist_ok=True)
    reviewed_table = reviewed_table.copy()
    invalid = sorted(
        set(reviewed_table["manual_review"].dropna().astype(str))
        - set(MANUAL_REVIEW_OPTIONS)
        - {""}
    )
    if invalid:
        raise ValueError(
            f"Unknown manual_review values: {invalid}; expected {MANUAL_REVIEW_OPTIONS}"
        )
    reviewed_csv = output_dir / "manual_review_results.csv"
    reviewed_table.to_csv(reviewed_csv, index=False, encoding="utf-8-sig")

    verification_rows = []
    completed = reviewed_table[
        reviewed_table["manual_review"].fillna("").astype(str).ne("")
    ]
    for file_name, rows in completed.groupby("file", sort=True):
        reviewed_starts = pd.to_numeric(
            rows["reviewed_start_sec"], errors="coerce"
        )
        reviewed_ends = pd.to_numeric(
            rows["reviewed_end_sec"], errors="coerce"
        )
        starts = reviewed_starts.fillna(
            pd.to_numeric(rows["start_sec"], errors="raise")
        )
        ends = reviewed_ends.fillna(
            pd.to_numeric(rows["end_sec"], errors="raise")
        )
        descriptions = "REVIEWED_" + rows["manual_review"].astype(str)
        annotations = mne.Annotations(
            onset=starts.astype(float).to_numpy(),
            duration=(ends.astype(float) - starts.astype(float)).to_numpy(),
            description=descriptions.to_numpy(),
        )
        path = reviewed_dir / f"{Path(file_name).stem}-reviewed-annot.fif"
        annotations.save(path, overwrite=True, verbose=False)
        reloaded = mne.read_annotations(path)
        verification_rows.append(
            {
                "file": file_name,
                "reviewed_annotation_path": str(path.resolve()),
                "saved_count": len(annotations),
                "reloaded_count": len(reloaded),
                "reload_ok": len(annotations) == len(reloaded),
            }
        )
    return reviewed_csv, pd.DataFrame(verification_rows)
