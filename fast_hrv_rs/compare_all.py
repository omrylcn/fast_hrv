"""
Compare fast_hrv, neurokit2, and Rust implementations
"""
import numpy as np
import json

# Generate reproducible synthetic ECG
np.random.seed(42)
sr = 512
duration = 30  # seconds
n = duration * sr

# Create ECG-like signal
signal = np.zeros(n)
pos = 410
state = 42
while pos < n - 50:
    state = (state * 1103515245 + 12345) & 0xFFFFFFFF
    jitter = ((state >> 16) & 0x3f) - 32
    actual_pos = max(0, pos + jitter)
    if actual_pos < n and actual_pos >= 5 and actual_pos + 10 < n:
        signal[actual_pos] = 1.0
        signal[actual_pos - 3] = 0.2
        signal[actual_pos + 3] = -0.3
    pos += 410

for i in range(n):
    state = (state * 1103515245 + 12345) & 0xFFFFFFFF
    signal[i] += ((state >> 16) & 0xff) / 2560.0 - 0.05

print("=" * 70)
print("ECG Signal Comparison Test")
print("=" * 70)
print(f"Signal: {n} samples, {duration}s @ {sr}Hz")
print(f"Signal range: [{signal.min():.4f}, {signal.max():.4f}]")
print()

# ============================================================================
# 1. NeuroKit2 (reference)
# ============================================================================
print("1. NeuroKit2 (reference)")
print("-" * 40)

import neurokit2 as nk

# Clean
nk_cleaned = nk.ecg_clean(signal, sampling_rate=sr, method="neurokit")

# Peaks
nk_signals, nk_info = nk.ecg_peaks(nk_cleaned, sampling_rate=sr, method="neurokit")
nk_rpeaks = nk_info["ECG_R_Peaks"]

# Fix artifacts (Kubios method)
nk_fixed_result = nk.signal_fixpeaks(nk_rpeaks, sampling_rate=sr, method="kubios")
nk_fixed = np.array(nk_fixed_result[0]) if isinstance(nk_fixed_result, tuple) else np.array(nk_fixed_result)
if nk_fixed.ndim == 0:
    nk_fixed = nk_rpeaks  # fallback to original if fix fails

# HRV metrics
if len(nk_fixed) > 5:
    rr_ms = np.diff(nk_fixed) / sr * 1000
    nk_mean_hr = 60000 / np.mean(rr_ms)
    nk_sdnn = np.std(rr_ms, ddof=1)
    nk_rmssd = np.sqrt(np.mean(np.diff(rr_ms) ** 2))
    nk_pnn50 = np.sum(np.abs(np.diff(rr_ms)) > 50) / len(np.diff(rr_ms)) * 100
else:
    nk_mean_hr = nk_sdnn = nk_rmssd = nk_pnn50 = 0

print(f"  Cleaned signal range: [{nk_cleaned.min():.6f}, {nk_cleaned.max():.6f}]")
print(f"  R-peaks detected: {len(nk_rpeaks)}")
print(f"  R-peaks after fix: {len(nk_fixed)}")
print(f"  Mean HR: {nk_mean_hr:.2f} bpm")
print(f"  SDNN: {nk_sdnn:.2f} ms")
print(f"  RMSSD: {nk_rmssd:.2f} ms")
print(f"  pNN50: {nk_pnn50:.2f} %")
print()

# ============================================================================
# 2. fast_hrv (Python)
# ============================================================================
print("2. fast_hrv (Python)")
print("-" * 40)

import sys
sys.path.insert(0, '/home/omrylcn/Work/bio_works')
from fast_hrv.ecg_numba import process_ecg as fh_process_ecg

fh_result = fh_process_ecg(signal, sampling_rate=sr, correct_artifacts=True)
fh_cleaned = fh_result["ecg_cleaned"]
fh_rpeaks = fh_result["rpeaks"]
fh_stats = fh_result["stats"]

print(f"  Cleaned signal range: [{fh_cleaned.min():.6f}, {fh_cleaned.max():.6f}]")
print(f"  R-peaks detected: {len(fh_rpeaks)}")
print(f"  Mean HR: {fh_stats.get('mean_hr_bpm', 0):.2f} bpm")
print(f"  SDNN: {fh_stats.get('sdnn_ms', 0):.2f} ms")
print(f"  RMSSD: {fh_stats.get('rmssd_ms', 0):.2f} ms")
print(f"  pNN50: {fh_stats.get('pnn50', 0):.2f} %")
print()

# ============================================================================
# 3. Export data for Rust comparison
# ============================================================================
# Save signal to file for Rust to read
np.savetxt('/tmp/test_ecg_signal.txt', signal, fmt='%.10f')

# Save Python results for comparison
python_results = {
    "nk_cleaned_min": float(nk_cleaned.min()),
    "nk_cleaned_max": float(nk_cleaned.max()),
    "nk_n_peaks": len(nk_rpeaks),
    "nk_n_peaks_fixed": len(nk_fixed),
    "nk_mean_hr": float(nk_mean_hr),
    "nk_sdnn": float(nk_sdnn),
    "nk_rmssd": float(nk_rmssd),
    "nk_pnn50": float(nk_pnn50),
    "nk_first_5_peaks": [int(x) for x in nk_fixed[:5]] if len(nk_fixed) >= 5 else [],
    "fh_cleaned_min": float(fh_cleaned.min()),
    "fh_cleaned_max": float(fh_cleaned.max()),
    "fh_n_peaks": len(fh_rpeaks),
    "fh_mean_hr": float(fh_stats.get('mean_hr_bpm', 0)),
    "fh_sdnn": float(fh_stats.get('sdnn_ms', 0)),
    "fh_rmssd": float(fh_stats.get('rmssd_ms', 0)),
    "fh_pnn50": float(fh_stats.get('pnn50', 0)),
    "fh_first_5_peaks": [int(x) for x in fh_rpeaks[:5]] if len(fh_rpeaks) >= 5 else [],
}

with open('/tmp/python_results.json', 'w') as f:
    json.dump(python_results, f, indent=2)

print("Data exported for Rust comparison.")
print(f"Signal saved to: /tmp/test_ecg_signal.txt")
print()
