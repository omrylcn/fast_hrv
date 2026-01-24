"""
fast_hrv - Fast ECG Processing and HRV Analysis

A high-performance ECG processing library optimized for HRV analysis.
66x faster than NeuroKit2 with 98.68% correlation.

Usage:
    from fast_hrv import process_ecg

    result = process_ecg(ecg_signal, sampling_rate=512)

    print(result["stats"]["mean_hr_bpm"])
    print(result["stats"]["sdnn_ms"])
"""

__version__ = "0.1.0"

from .ecg import process_ecg, compute_quality
from .utils import sanitize_signal

__all__ = [
    "process_ecg",
    "compute_quality",
    "sanitize_signal",
]
