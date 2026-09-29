from __future__ import annotations

import numpy as np
import mne
import os
import re
from glob import glob

import matplotlib.pyplot as plt
import pywt


def plot(signal:np.array[float], title:str = "", minF:float = 0, maxF:float = 100, contrastMode:bool = True) -> None:
    # signal: 1-D EEG, fs: sampling rate in Hz
    fs = 256
    dt = 1 / fs

    # choose scales -> these map to frequencies
    # use a continuous wavelet: 'morl' (Morlet), 'cmor' (complex Morlet), 'mexh'
    wavelet = 'shan'   # complex Morlet: bandwidth 1.5, center freq 1.0

    # pick the frequencies you care about (e.g. 1-40 Hz for EEG) and convert to scales
    freqs = np.arange(minF, maxF)                          # 1..40 Hz
    scales = pywt.frequency2scale(wavelet, freqs / fs)  # note: normalized freq

    coefs, freqs_out = pywt.cwt(signal, scales, wavelet, sampling_period=dt)

    # coefs is complex for cmor -> power is |coefs|^2
    power = np.abs(coefs) ** 2
    pLabel = "Power"
    if contrastMode:
        power = 10 * np.log10(power + 1e-12)
        pLabel = "contrasted Power"


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 4), dpi=100, sharex=True,
                                    gridspec_kw={'height_ratios': [1, 2]})

    t = np.arange(len(signal)) * dt

    # Top: signal
    ax1.plot(t, signal)
    ax1.set_ylabel('Amplitude')

    if title != "":
        ax1.set_title(title)

    # Bottom: spectrogram
    pcm = ax2.pcolormesh(t, freqs_out, power, shading='gouraud')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (s)')
    fig.colorbar(pcm, ax=[ax1, ax2], label=pLabel)
    plt.show()

def parse_summary(summary_path):
    """
    Parse a chbXX-summary.txt file to extract seizure intervals per EDF file.
    Returns: dict {edf_filename: [(start_sec, end_sec), ...]}
    """
    seizures = {}
    current_file = None
    with open(summary_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith('File Name:'):
            current_file = line.split(':')[1].strip()
            seizures[current_file] = []

        elif 'Number of Seizures in File:' in line:
            n = int(line.split(':')[1].strip())
            # Next lines contain start/end times for each seizure
            for _ in range(n):
                i += 1
                start_line = lines[i].strip()
                i += 1
                end_line = lines[i].strip()
                # Extract integers (seconds) from lines like "Seizure Start Time: 2996 seconds"
                start = int(re.search(r'(\d+)\s*seconds', start_line).group(1))
                end = int(re.search(r'(\d+)\s*seconds', end_line).group(1))
                seizures[current_file].append((start, end))
        i += 1

    return seizures


def load_edf_windows(edf_path:str, seizure_intervals, window_sec:float=1.0, overlap:float=0.0):
    """
    Load one EDF, split into fixed windows, label each window.
    A window is labeled 1 (seizure) if it overlaps any seizure interval, else 0.

    Returns:
        X: np.array (n_windows, n_channels, n_samples)
        y: np.array (n_windows,)
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='ERROR')
    data = raw.get_data()          # shape (n_channels, n_total_samples)
    sfreq = raw.info['sfreq']

    win_samples = int(window_sec * sfreq)
    step = int(win_samples * (1 - overlap))
    n_channels, n_total = data.shape

    X, y = [], []
    for start_idx in range(0, n_total - win_samples + 1, step):
        end_idx = start_idx + win_samples
        window = data[:, start_idx:end_idx]

        # Time bounds of this window in seconds
        t_start = start_idx / sfreq
        t_end = end_idx / sfreq

        # Label: does window overlap any seizure?
        label = 0
        for (s_start, s_end) in seizure_intervals:
            if t_start < s_end and t_end > s_start:
                label = 1
                break

        X.append(window)
        y.append(label)

    return np.array(X), np.array(y)


def load_chbmit_subject(subject_dir, window_sec:float=1.0, overlap:float=0.0,
                        common_channels=None):
    """
    Load all EDF files for one subject (e.g., chb01/) into arrays.

    subject_dir: path to folder containing chbXX_YY.edf and chbXX-summary.txt
    common_channels: optional list of channel names to keep (ensures consistent shape)
    """
    subject_id = os.path.basename(subject_dir.rstrip('/'))
    summary_path = os.path.join(subject_dir, f'{subject_id}-summary.txt')
    seizures = parse_summary(summary_path)

    X_all, y_all = [], []

    edf_files = sorted(glob(os.path.join(subject_dir, '*.edf')))
    for edf_path in edf_files:
        fname = os.path.basename(edf_path)
        intervals = seizures.get(fname, [])

        # Optionally restrict to a fixed channel set for shape consistency
        if common_channels is not None:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='ERROR')
            available = [ch for ch in common_channels if ch in raw.ch_names]
            if len(available) != len(common_channels):
                print(f"Skipping {fname}: missing channels")
                continue

        X, y = load_edf_windows(edf_path, intervals, window_sec, overlap)

        # If restricting channels, re-select (simplest: reload with picks)
        if common_channels is not None:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='ERROR')
            raw.pick(common_channels)
            data = raw.get_data()
            sfreq = raw.info['sfreq']
            win_samples = int(window_sec * sfreq)
            step = int(win_samples * (1 - overlap))
            X = []
            for s in range(0, data.shape[1] - win_samples + 1, step):
                X.append(data[:, s:s + win_samples])
            X = np.array(X)  # y already correct length

        X_all.append(X)
        y_all.append(y)
        print(f"{fname}: {X.shape[0]} windows, {int(np.sum(y))} seizure")

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    return X_all, y_all
