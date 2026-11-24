# fast-hrv

Fast, dependency-light **heart rate variability (HRV)** analysis from ECG, in Rust.

- **Signal:** Butterworth filter design (SciPy-equivalent), NeuroKit2-compatible zero-phase ECG cleaning,
  a causal **streaming** filter for packet-by-packet processing, R-peak detection (NeuroKit2 method),
  Kubios artifact correction (NeuroKit2-equivalent), gap-aware RR intervals
- **Signal quality (`sqi`):** absolute indices: flat line, saturation, kSQI, pSQI, basSQI (Zhao & Zhang 2018),
  bSQI (detector agreement), template correlation, RR plausibility, zhao2018 fusion; plus the relative
  per-beat `compute_quality` (NeuroKit2 averageQRS)
- **HRV metrics:** time domain (17), frequency domain (9), nonlinear (9), asymmetry (6)
- **No heavy dependencies:** filters, FFT and spline are implemented in the crate; plain scalar `f64` code,
  no CPU-specific instructions (no AVX assumed, no `target-cpu=native`)
- Optional: `serde` serialization, Python bindings (PyO3)

```rust
let rri = fast_hrv::rri_from_peaks(&peaks, 512);   // R-peak indices → NN intervals (ms)
let r = fast_hrv::hrv(&rri);
println!("SDNN {:.1} ms  RMSSD {:.1} ms  LF/HF {:.2}  SampEn {:.2}",
         r.time.sdnn, r.time.rmssd, r.frequency.lf_hf, r.nonlinear.sampen);
```

## ECG cleaning

| Function | Use | What it does |
|---|---|---|
| `ecg_clean`, `ecg_clean_neurokit` | offline | `nk.ecg_clean(method="neurokit")`: 0.5 Hz high-pass Butterworth order 5, zero-phase (`sosfiltfilt` with SciPy's padding), then NeuroKit's powerline moving average of `int(fs/50)` samples (`filtfilt`) |
| `EcgStreamFilter`, `ecg_clean_causal` | live / offline | causal 0.5 Hz high-pass (order 2) + 40 Hz low-pass (order 4), optional notch (`EcgFilterConfig::with_notch(50.0, q)`); state carried across packets |
| `ecg_clean_zero_phase` | offline | the same configurable band, zero-phase with SciPy-style padding |
| `filter::{butter_sos, iirnotch, sosfilt, sosfiltfilt, filtfilt_ba, SosFilter}` | building blocks | SciPy `butter(..., output="sos")`, `iirnotch`, `sosfilt`, `sosfiltfilt`, `filtfilt` |

Live and offline processing give the same result: feeding a signal to `EcgStreamFilter` in packets of any
size (e.g. 32 samples) is **bit-identical** to `ecg_clean_causal` on the whole signal (`tests/streaming.rs`).

```rust
use fast_hrv::ecg::{EcgFilterConfig, EcgStreamFilter};
let mut f = EcgStreamFilter::new(512.0, &EcgFilterConfig::default().with_notch(50.0, 30.0))?;
for packet in packets { let clean = f.process(packet); /* … */ }
```

## Gaps

`gaps::split_at_gaps` / `rri_with_gaps` take R-peak sample indices plus `Gap { start, end }` markers
(or `split_peak_times` with timestamps in seconds) and return segments: no RR interval spans a gap, and
`fix_peaks_segmented` runs Kubios per segment so no beat is inserted into a gap. Compute metrics per segment.

## Metrics

| Group | Metrics |
|---|---|
| Time domain | MeanNN, SDNN, RMSSD, SDSD, pNN50, pNN20, MedianNN, MadNN, IQRNN, MinNN, MaxNN, CVNN, CVSD, MeanHR, SDHR, HTI, TINN |
| Frequency domain (Welch, 4 Hz cubic-spline resampling) | VLF, LF, HF, total power, LF/HF, LFnu, HFnu, LF peak, HF peak |
| Nonlinear | SD1, SD2, SD1/SD2, CSI, CVI, SampEn, ApEn, DFA α1, DFA α2 |
| Asymmetry | GI, PI, SD1d, SD1a, C1d, C1a |

Minimum data (otherwise the metric is `NaN`): time domain ≥ 3 intervals; frequency domain per band
(sum of RR): HF ≥ 60 s, LF ≥ 120 s, VLF ≥ 300 s (configurable, `FrequencyOptions`); entropy/DFA ≥ 16
intervals. A constant RR series gives zero band power and `NaN` LF/HF. Inputs containing `NaN` give `NaN`
results; invalid inputs never panic. Bands follow the Task Force (1996) definitions.

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
| Kubios detection and correction (26 series with injected ectopic/missed/extra/long-short beats, 30–350 beats) | `nk.signal_fixpeaks(method="Kubios")` | 26/26 identical |
| MeanNN, SDNN, RMSSD, SDSD, Median, MAD, IQR, Min, Max, CVNN, CVSD, HTI, SD1, SD2, SD1/SD2, CSI, CVI, SampEn, ApEn, GI, PI, SD1d/a, C1d/a | `nk.hrv_time` / `hrv_nonlinear` (same R-peaks) | ≤ 3.3e-14 relative |
| pNN50 / pNN20 | NeuroKit divides by the number of RR intervals, fast-hrv by the number of differences (Task Force) | identical after rescaling |
| LF, HF, LF/HF, LFnu (records ≥ 5 min) | `nk.hrv_frequency(normalize=False)` | ≤ 24 % observed, test tolerance 30 % (different interpolation and Welch settings) |
| VLF, TINN, DFA α1/α2 | — | not comparable (different definitions; see `tests/neurokit_parity.rs`) |

Regenerate the references with `python tools/neurokit_reference.py` (and `filter_reference.py`,
`sqi_reference.py`); the parity tests fail if a reference file is missing (`FAST_HRV_SKIP_PARITY=1` skips them).

## Benchmark

`cargo bench --bench throughput` runs a reproducible, dependency-free benchmark on a deterministic
synthetic 5-minute ECG at 512 Hz (median of 21 runs). Example (Intel i7-9700K, Rust 1.100.0):

| Step | Time |
|---|---|
| `ecg_clean` (zero-phase, NeuroKit-compatible) | ~6.5 ms |
| `EcgStreamFilter`, 32-sample packets, with notch | ~1.0 ms |
| `find_peaks_default` | ~1.0 ms |
| `fix_peaks_default` (Kubios) | ~0.2 ms |
| `hrv()` (all metrics, 360 RR) | ~2.0 ms |

## Features

| Feature | Effect |
|---|---|
| `serde` | `Serialize`/`Deserialize` for report structs |
| `parallel` | reserved, currently no effect |
| `simd` | portable unrolled helpers (`simd` module); no CPU-specific instructions; off by default |
| `python` | PyO3 module `fast_hrv` (`maturin develop --release --features python`, then `import fast_hrv`) |

Toolchain: Rust 1.100.0 (`rust-toolchain.toml`), MSRV 1.85, edition 2024.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.

Based on `fast_hrv_rs` from [omrylcn/fast_hrv](https://github.com/omrylcn/fast_hrv).
