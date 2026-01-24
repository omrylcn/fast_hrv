"""
fast_hrv.ecg_numba - Numba-optimized ECG Processing Module

Full pipeline optimized with Numba JIT compilation.
"""

import numpy as np
from typing import Dict, Tuple, Any

from numba import njit

from neurokit2.ecg import ecg_clean
from .utils import sanitize_signal
from .peaks_numba import ecg_peaks as _ecg_peaks


@njit
def _extract_segments_numba(signal, rpeaks, start_offset, end_offset, segment_len):
    """Numba-optimized segment extraction."""
    n_samples = len(signal)
    n_peaks = len(rpeaks)

    # First pass: count valid segments
    count = 0
    for i in range(n_peaks):
        start = rpeaks[i] + start_offset
        end = rpeaks[i] + end_offset
        if start >= 0 and end <= n_samples and (end - start) == segment_len:
            count += 1

    if count == 0:
        return np.empty((0, segment_len)), np.empty(0, dtype=np.int64)

    # Allocate arrays
    segments = np.empty((count, segment_len))
    valid_indices = np.empty(count, dtype=np.int64)

    # Second pass: extract
    idx = 0
    for i in range(n_peaks):
        start = rpeaks[i] + start_offset
        end = rpeaks[i] + end_offset
        if start >= 0 and end <= n_samples and (end - start) == segment_len:
            segments[idx] = signal[start:end]
            valid_indices[idx] = i
            idx += 1

    return segments, valid_indices


@njit
def _interpolate_quality_numba(quality_per_beat, valid_peaks, n_samples):
    """Numba-optimized quality interpolation."""
    quality = np.zeros(n_samples)
    n = len(quality_per_beat)

    if n == 0:
        return quality

    # Before first peak
    if valid_peaks[0] > 0:
        for i in range(valid_peaks[0]):
            quality[i] = quality_per_beat[0]

    # Between peaks
    for i in range(n):
        start_idx = valid_peaks[i]
        end_idx = valid_peaks[i + 1] if i < n - 1 else n_samples
        for j in range(start_idx, end_idx):
            quality[j] = quality_per_beat[i]

    return quality


def compute_quality(
    ecg_cleaned: np.ndarray,
    rpeaks: np.ndarray,
    sampling_rate: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-optimized ECG signal quality computation.

    Parameters
    ----------
    ecg_cleaned : np.ndarray
        Cleaned ECG signal
    rpeaks : np.ndarray
        R-peak indices
    sampling_rate : int
        Sampling frequency in Hz

    Returns
    -------
    quality : np.ndarray
        Continuous quality score for each sample (0-1)
    quality_per_beat : np.ndarray
        Quality score per heartbeat (0-1)
    """
    n_samples = len(ecg_cleaned)

    if len(rpeaks) < 5:
        return np.zeros(n_samples), np.array([])

    # Calculate window size
    rr = np.diff(rpeaks) / sampling_rate
    heart_rate = 60 / np.mean(rr)
    window_size = 60 / heart_rate
    ratio_pre = 0.35

    start_offset = int(-ratio_pre * window_size * sampling_rate)
    end_offset = int((1 - ratio_pre) * window_size * sampling_rate)
    segment_len = end_offset - start_offset

    # Extract segments (Numba-optimized)
    segments, valid_indices = _extract_segments_numba(
        ecg_cleaned, rpeaks, start_offset, end_offset, segment_len
    )

    if len(segments) < 3:
        return np.zeros(n_samples), np.array([])

    # Standardize (NumPy vectorized)
    col_mean = np.mean(segments, axis=0)
    col_std = np.std(segments, axis=0, ddof=1)
    col_std[col_std == 0] = 1
    Z = (segments - col_mean) / col_std

    # Distance calculation
    dist = np.mean(Z, axis=1)
    abs_dist = np.abs(dist)
    min_d, max_d = np.min(abs_dist), np.max(abs_dist)

    if max_d > min_d:
        dist_scaled = (abs_dist - min_d) / (max_d - min_d)
    else:
        dist_scaled = np.zeros(len(abs_dist))

    quality_per_beat = 1 - dist_scaled

    # Interpolate (Numba-optimized)
    valid_peaks = rpeaks[valid_indices]
    quality = _interpolate_quality_numba(quality_per_beat, valid_peaks, n_samples)

    return quality, quality_per_beat


def _compute_stats(
    rpeaks: np.ndarray,
    quality_per_beat: np.ndarray,
    sampling_rate: int
) -> Dict[str, Any]:
    """
    Compute ECG and HRV statistics.
    """
    if len(rpeaks) < 2:
        return {"error": "insufficient_peaks", "n_peaks": len(rpeaks)}

    # RR intervals in milliseconds
    rr_ms = np.diff(rpeaks) / sampling_rate * 1000
    hr_bpm = 60000 / rr_ms

    stats = {
        "n_peaks": int(len(rpeaks)),
        "mean_hr_bpm": float(np.mean(hr_bpm)),
        "std_hr_bpm": float(np.std(hr_bpm)),
        "min_hr_bpm": float(np.min(hr_bpm)),
        "max_hr_bpm": float(np.max(hr_bpm)),
        "mean_rr_ms": float(np.mean(rr_ms)),
        "std_rr_ms": float(np.std(rr_ms)),
        "min_rr_ms": float(np.min(rr_ms)),
        "max_rr_ms": float(np.max(rr_ms)),
    }

    if len(quality_per_beat) > 0:
        stats["mean_quality"] = float(np.mean(quality_per_beat))
        stats["min_quality"] = float(np.min(quality_per_beat))
        stats["max_quality"] = float(np.max(quality_per_beat))
        stats["n_low_quality_beats"] = int(np.sum(quality_per_beat < 0.5))

    if len(rr_ms) >= 3:
        diff_rr = np.diff(rr_ms)
        stats["sdnn_ms"] = float(np.std(rr_ms, ddof=1))
        stats["rmssd_ms"] = float(np.sqrt(np.mean(diff_rr ** 2)))
        nn50 = np.sum(np.abs(diff_rr) > 50)
        stats["pnn50"] = float(nn50 / len(diff_rr) * 100)
        nn20 = np.sum(np.abs(diff_rr) > 20)
        stats["pnn20"] = float(nn20 / len(diff_rr) * 100)
        stats["sdsd_ms"] = float(np.std(diff_rr, ddof=1))

    return stats


def process_ecg(
    ecg_signal: np.ndarray,
    sampling_rate: int = 512,
    clean_method: str = "neurokit",
    correct_artifacts: bool = True
) -> Dict[str, Any]:
    """
    Numba-optimized ECG processing pipeline for HRV analysis.

    Parameters
    ----------
    ecg_signal : np.ndarray
        Raw ECG signal
    sampling_rate : int
        Sampling frequency in Hz (default: 512)
    clean_method : str
        Cleaning method (default: "neurokit")
    correct_artifacts : bool
        Whether to correct peak artifacts (default: True)

    Returns
    -------
    dict
        - ecg_raw: Original signal
        - ecg_cleaned: Cleaned signal
        - quality: Continuous quality score 0-1
        - rpeaks: R-peak indices
        - stats: Statistics dict
    """
    # 1. Sanitize input
    ecg_raw = sanitize_signal(ecg_signal)

    # 2. Clean signal
    ecg_cleaned = ecg_clean(
        ecg_raw,
        sampling_rate=sampling_rate,
        method=clean_method
    )

    # 3. Detect R-peaks (Numba-optimized)
    _, info = _ecg_peaks(
        ecg_cleaned,
        sampling_rate=sampling_rate,
        correct_artifacts=correct_artifacts
    )
    rpeaks = info["ECG_R_Peaks"]

    # 4. Compute quality (Numba-optimized)
    quality, quality_per_beat = compute_quality(
        ecg_cleaned,
        rpeaks,
        sampling_rate
    )

    # 5. Compute statistics
    stats = _compute_stats(rpeaks, quality_per_beat, sampling_rate)

    return {
        "ecg_raw": ecg_raw,
        "ecg_cleaned": ecg_cleaned,
        "quality": quality,
        "rpeaks": rpeaks,
        "stats": stats
    }
