"""Export the CHB-MIT EDF files to windowed NumPy arrays."""

import argparse
from pathlib import Path

import mne
import numpy as np
from numpy.lib.format import open_memmap

from utils import parse_summary


def export_all_subjects(data_dir: Path, output_dir: Path, window_sec: float = 1.0) -> None:
    """Export all subjects in the CHB-MIT dataset to windowed NumPy arrays."""
    for subject_dir in sorted(data_dir.glob("chb*")):
        if not subject_dir.is_dir():
            continue
        subject_id = subject_dir.name
        print(f"Exporting {subject_id}...")
        export_dataset(subject_dir, output_dir / subject_id, window_sec)


def export_dataset(data_dir: Path, output_dir: Path, window_sec: float = 1.0) -> None:
    pattern = "*.edf" if data_dir.name.startswith("chb") else "chb*/**/*.edf"
    edf_files = sorted(data_dir.glob(pattern))
    if not edf_files:
        raise FileNotFoundError(f"No EDF files found below {data_dir}")
    if window_sec <= 0:
        raise ValueError("window_sec must be greater than zero")

    # Pass 1: read headers only to find common channels and count windows.
    first_raw = mne.io.read_raw_edf(edf_files[0], preload=False, verbose="ERROR")
    channels = list(first_raw.ch_names)
    sfreq = first_raw.info["sfreq"]
    window_samples = int(window_sec * sfreq)
    if window_samples < 1:
        raise ValueError("window_sec is shorter than one sample")

    n_windows = 0
    for edf_path in edf_files:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
        if raw.info["sfreq"] != sfreq:
            raise ValueError(
                f"{edf_path.name} has sfreq {raw.info['sfreq']}, expected {sfreq}"
            )
        available = set(raw.ch_names)
        channels = [channel for channel in channels if channel in available]
        n_windows += raw.n_times // window_samples
    if not channels:
        raise ValueError("No EEG channels are common to all EDF files")

    # Pass 2: load one recording at a time and write windows straight to disk.
    output_dir.mkdir(parents=True, exist_ok=True)
    X = open_memmap(
        output_dir / "X.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n_windows, len(channels), window_samples),
    )
    y = np.zeros(n_windows, dtype=np.int8)

    i = 0
    for edf_path in edf_files:
        subject_dir = edf_path.parent
        subject_id = subject_dir.name
        summary = parse_summary(subject_dir / f"{subject_id}-summary.txt")
        intervals = summary.get(edf_path.name, [])

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
        raw.pick(channels)
        data = raw.get_data().astype(np.float32)

        for start in range(0, data.shape[1] - window_samples + 1, window_samples):
            end = start + window_samples
            start_sec, end_sec = start / sfreq, end / sfreq
            label = any(
                start_sec < seizure_end and end_sec > seizure_start
                for seizure_start, seizure_end in intervals
            )
            X[i] = data[:, start:end]
            y[i] = int(label)
            i += 1

        del raw, data
        print(f"{edf_path.relative_to(data_dir)}: {i}/{n_windows} windows")

    assert i == n_windows, f"expected {n_windows} windows, wrote {i}"
    X.flush()
    del X
    np.save(output_dir / "y.npy", y)
    print(
        f"saved X.npy ({n_windows}, {len(channels)}, {window_samples}) "
        f"and y.npy ({n_windows},)"
    )
    print(f"channels ({len(channels)}): {', '.join(channels)}")