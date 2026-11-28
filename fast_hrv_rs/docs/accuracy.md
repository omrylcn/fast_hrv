# Accuracy and speed

[← README](../README.md)

## Accuracy

Checked in `cargo test` against reference values generated at development time with SciPy and NeuroKit2
(`tools/*.py`; the versions are recorded in each JSON file under `tests/data/`). All test data is synthetic
(formulas or `nk.ecg_simulate`); no real recordings are used. Current references: NeuroKit2 0.3.0,
SciPy 1.18.1, NumPy 2.5.5, pandas 3.1.1.

| Component | Reference | Result |
|---|---|---|
| Butterworth SOS coefficients (20 designs: low/high/band-pass/band-stop, orders 1–8) | `scipy.signal.butter(output="sos")` | max error 2.2e-16 |
| `sosfiltfilt`, `filtfilt`, `iirnotch` | SciPy | ≤ 5e-13 relative |
| `ecg_clean` on 512 Hz simulated ECG (clean, noisy, powerline + baseline) | `nk.ecg_clean(method="neurokit")` | ≤ 2.7e-13 relative |
| R-peaks on the same signals | `nk.ecg_peaks` | 69/69 identical |
| Elgendi 8–20 Hz filter (5 simulated 30 s signals: clean, noisy, powerline + baseline, 140 bpm, inverted low amplitude) | `nk.ecg_clean(method="elgendi2010")` | ≤ 6.4e-14 relative |
| Elgendi peaks before refinement, same signals | `nk.ecg_findpeaks(method="elgendi2010")` / `"elgendi2013"` | 201/201 and 202/202 identical (default config = elgendi2013 peaks as well) |
| Kubios detection and correction (26 series with injected ectopic/missed/extra/long-short beats, 30–350 beats) | `nk.signal_fixpeaks(method="Kubios")` | 26/26 identical |
| Kubios with `alpha` 5.2 / 6 and `threshold_floor_s` 0 / 30 / 40 / 50 ms on sub-sample positions (11 synthetic series, 3282 beats) | Python research port of fast-hrv Kubios (`tools/exclusion_reference.py`) | 55/55 identical (flags and corrected positions, bit for bit) |
| Exclusion mode (single / two-stage, with / without disagreement, 3 parameter sets, same series) | Python research reference (`tools/exclusion_reference.py`) | 132/132 identical (positions, classes, excluded NN) |
| MeanNN, SDNN, RMSSD, SDSD, Median, MAD, IQR, Min, Max, CVNN, CVSD, HTI, SD1, SD2, SD1/SD2, CSI, CVI, SampEn, ApEn, GI, PI, SD1d/a, C1d/a | `nk.hrv_time` / `hrv_nonlinear` (same R-peaks) | ≤ 3.3e-14 relative |
| pNN50 / pNN20 | NeuroKit divides by the number of RR intervals, fast-hrv by the number of differences (Task Force) | identical after rescaling |
| LF, HF, LF/HF, LFnu (records ≥ 5 min) | `nk.hrv_frequency(normalize=False)` | ≤ 24 % observed, test tolerance 30 % (different interpolation and Welch settings) |
| lnHF | `nk.hrv_frequency` `HRV_LnHF` | ≤ 2.0 % observed (same method difference as HF, on the log scale) |
| PIP, IALS, PSS, PAS | `nk.hrv_nonlinear` | ≤ 1.4e-16 (identical definitions) |
| SDRMSSD, Prc20NN, Prc80NN, S, CSI modified | `nk.hrv_time` / `hrv_nonlinear` | ≤ 2.2e-15 |
| Baevsky stress index | hand-computed (no NeuroKit2 equivalent; Kubios definitions without VLF detrending) | exact |
| Breathing rate from ECG (2 × 120 s ECGSYN, 15/min) | `nk.ecg_simulate(method="ecgsyn")` respiratory component (`tools/respiration_reference.py`) | 5/6 windows valid, ≤ 0.7/min error; 1 rejected by the AM/FM agreement rule |
| Lomb–Scargle periodogram (4 series, 120–600 s, with 0–10 % dropped intervals) | `astropy.timeseries.LombScargle(fit_mean=False, normalization="psd")` 8.0.3 (`tools/lomb_reference.py`) | ≤ 1e-9 of the peak |
| Smoothness-priors detrending (λ 10 and 500) | `nk.signal_detrend(method="tarvainen2002")` | ≤ 7e-10 |
| VLF, TINN, DFA α1/α2 | — | not comparable (different definitions; see `tests/neurokit_parity.rs`) |

Real ECG is used only outside `cargo test`: [`tools/mitdb_timing.py`](../tools/mitdb_timing.py) downloads
MIT-BIH Arrhythmia records from PhysioNet (ODC-By 1.0) and measures live-path detection (F1 at ±50 / ±150 ms,
timing offset, beat-to-beat jitter, RMSSD error against the annotations); results for the fiducial options
are in [detection.md](detection.md#fiducial-point-which-point-of-the-qrs-is-reported).

Regenerate the references with `python tools/neurokit_reference.py` (and `filter_reference.py`,
`sqi_reference.py`, `exclusion_reference.py`); the parity tests fail if a reference file is missing (`FAST_HRV_SKIP_PARITY=1` skips them).

## Speed

`cargo bench --bench throughput` runs a reproducible, dependency-free benchmark on a deterministic
synthetic 5-minute ECG at 512 Hz (median of 21 runs). Example (Intel i7-9700K, Rust 1.100.0):

| Step | Time |
|---|---|
| `ecg_clean` (zero-phase, NeuroKit-compatible) | ~6.5 ms |
| `EcgStreamFilter`, 32-sample packets, with notch | ~1.0 ms |
| `find_peaks_default` | ~1.0 ms |
| `fix_peaks_default` (Kubios) | ~0.2 ms |
| `hrv()` (all metrics, 360 RR) | ~2.0 ms |
| `elgendi_peaks` (8–20 Hz filter + detection) | ~1.8 ms |
| `refine_peaks` (±50 ms, parabolic) | ~0.15 ms |
| `elgendi_refined_windowed` (10 s windows + 2 × 2 s margin) | ~2.8 ms |
| `dual_detect` on 30 × 10 s windows (both detectors + bSQI) | ~3.2 ms |

`cargo bench --bench capacity` simulates the planned per-user live pipeline (stream filter, 10 s dual
detection + SQI, 60 s Kubios + time domain, 5 min full `hrv()`) for 500 interleaved synthetic sessions
and reports CPU per user, users per core, per-second load spikes and memory; results and method in
[CAPACITY.md](../CAPACITY.md) (Turkish). `-- --stream` runs the detection stage through `StreamDetector`
(same cost within noise: 50.3 vs 50.5 µs per user per second of signal; memory per session unchanged).
