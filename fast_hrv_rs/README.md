# fast-hrv

Fast, dependency-light **heart rate variability (HRV)** analysis from ECG, in Rust.

- **Signal:** ECG cleaning (zero-phase Butterworth), R-peak detection (NeuroKit2 method),
  Kubios artifact correction, per-beat quality (template matching)
- **39 HRV metrics:** time domain (17), frequency domain (9), nonlinear (9), asymmetry (6)
- **No heavy dependencies:** FFT, spline and filters are implemented in the crate
- Optional: `serde` serialization, `rayon` parallelism, Python bindings (PyO3)

```rust
let rri = fast_hrv::rri_from_peaks(&peaks, 512);   // R-peak indices → NN intervals (ms)
let r = fast_hrv::hrv(&rri);
println!("SDNN {:.1} ms  RMSSD {:.1} ms  LF/HF {:.2}  SampEn {:.2}",
         r.time.sdnn, r.time.rmssd, r.frequency.lf_hf, r.nonlinear.sampen);
```

## Metrics

| Group | Metrics |
|---|---|
| Time domain | MeanNN, SDNN, RMSSD, SDSD, pNN50, pNN20, MedianNN, MadNN, IQRNN, MinNN, MaxNN, CVNN, CVSD, MeanHR, SDHR, HTI, TINN |
| Frequency domain (Welch, 4 Hz cubic-spline resampling) | VLF, LF, HF, total power, LF/HF, LFnu, HFnu, LF peak, HF peak |
| Nonlinear | SD1, SD2, SD1/SD2, CSI, CVI, SampEn, ApEn, DFA α1, DFA α2 |
| Asymmetry | GI, PI, SD1d, SD1a, C1d, C1a |

Minimum data: time domain ≥ 3 intervals, frequency domain ≥ 30 s, entropy/DFA ≥ 16 intervals
(otherwise the metric is `NaN`). Bands follow the Task Force (1996) definitions.

## Accuracy

Each metric is tested against reference values (NumPy/SciPy/NeuroKit2). A cross-check against NeuroKit2
on real recordings and a reproducible benchmark are part of the test suite (`tests/`, `benches/`).

## Features

| Feature | Effect |
|---|---|
| `serde` | `Serialize`/`Deserialize` for all report structs |
| `parallel` | rayon-based batch processing |
| `simd` | SIMD helpers |
| `python` | PyO3 module (`maturin develop --release`) |

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.

Based on `fast_hrv_rs` from [omrylcn/fast_hrv](https://github.com/omrylcn/fast_hrv).

