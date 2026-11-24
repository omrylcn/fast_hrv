# fast-hrv

Fast, dependency-light **heart rate variability (HRV)** analysis from ECG, in Rust.

- **Signal:** Butterworth filter design (SciPy-equivalent), NeuroKit2-compatible zero-phase ECG cleaning,
  a causal **streaming** filter for packet-by-packet processing, R-peak detection (NeuroKit2 method and
  Elgendi two-moving-average detector), sub-sample R-peak refinement, a dual-detector API with bSQI,
  windowed processing of long recordings, Kubios artifact correction (NeuroKit2-equivalent),
  gap-aware RR intervals
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

## R-peak detection

| Function | What it does |
|---|---|
| `find_peaks_default` | NeuroKit2's default `neurokit` method (smoothed gradient, QRS regions, 0.3 s minimum delay) |
| `detect::elgendi_peaks` + `ElgendiConfig` | Elgendi two-moving-average detector: 8–20 Hz Butterworth (order 2, causal as NeuroKit2), squared signal, 97 ms / 611 ms moving averages, threshold offset β = 0.08 · mean, minimum block = QRS window (Elgendi 2013), 0.3 s refractory. Presets `ElgendiConfig::neurokit2013()` / `neurokit2010()` reproduce NeuroKit2's `elgendi2013` / `elgendi2010` exactly |
| `detect::refine_peaks` + `RefineConfig` | moves candidate peaks (any detector) to the R extremum of the clean signal within ±50 ms (configurable), polarity `Auto` (one sign per window, majority vote), `Positive`, `Negative` or `Absolute`; 3-point parabolic interpolation gives sub-sample positions |
| `rri_from_positions`, `rri_from_times` | RR intervals (ms) from sub-sample positions or times in seconds (the integer `rri_from_peaks` is unchanged) |
| `detect::dual_detect` + `DualConfig` | primary (Elgendi + refinement) and secondary (NeuroKit2 method) peaks on the same window and their bSQI (tolerance 0.15 s by default, 0.05 s variant documented); results only, no gate |
| `detect::detect_windowed`, `elgendi_refined_windowed` + `WindowConfig` | long recordings in fixed windows (default 10 s + 2 s margin each side) anchored at absolute sample indices; the detector restarts in every window and overlap duplicates are removed deterministically, so the output does not depend on where the recording starts |

The default Elgendi detector (causal 8–20 Hz filter) reports beats ~37 ms after the R peak; the
refinement removes this delay (synthetic signals with known R: every refined peak within 0.4 samples).
NeuroKit2's `elgendi2010` is ~80 ms late and needs a refinement window of at least 0.09 s.
Run long recordings through `detect_windowed`: the Elgendi offset (β · mean energy) is computed over the
analysed input, so a single pass over a long, artifact-laden recording can miss many beats.

```rust
use fast_hrv::detect::{DualConfig, RefineConfig, ElgendiConfig, dual_detect, elgendi_peaks, refine_peaks};
let clean = fast_hrv::ecg_clean_zero_phase(&raw, 512.0, &fast_hrv::EcgFilterConfig::default())?; // 0.5–40 Hz
let cand = elgendi_peaks(&clean, 512.0, &ElgendiConfig::default())?;
let r = refine_peaks(&clean, 512.0, &cand, &RefineConfig::default())?;
let rri = fast_hrv::rri_from_positions(&r.positions, 512.0);          // sub-sample RR (ms)
let d = dual_detect(&clean[..5120], 512.0, &DualConfig::default())?;   // one 10 s window
println!("bSQI {:?}", d.bsqi.map(|b| b.bsqi));
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
| Elgendi 8–20 Hz filter (5 simulated 30 s signals: clean, noisy, powerline + baseline, 140 bpm, inverted low amplitude) | `nk.ecg_clean(method="elgendi2010")` | ≤ 6.4e-14 relative |
| Elgendi peaks before refinement, same signals | `nk.ecg_findpeaks(method="elgendi2010")` / `"elgendi2013"` | 201/201 and 202/202 identical (default config = elgendi2013 peaks as well) |
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
| `elgendi_peaks` (8–20 Hz filter + detection) | ~1.8 ms |
| `refine_peaks` (±50 ms, parabolic) | ~0.15 ms |
| `elgendi_refined_windowed` (10 s windows + 2 × 2 s margin) | ~2.8 ms |
| `dual_detect` on 30 × 10 s windows (both detectors + bSQI) | ~3.2 ms |

`cargo bench --bench capacity` simulates the planned per-user live pipeline (stream filter, 10 s dual
detection + SQI, 60 s Kubios + time domain, 5 min full `hrv()`) for 500 interleaved synthetic sessions
and reports CPU per user, users per core, per-second load spikes and memory; results and method in
[CAPACITY.md](CAPACITY.md) (Turkish).

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
