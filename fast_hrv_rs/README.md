# fast-hrv

Fast, dependency-light **heart rate variability (HRV)** analysis from ECG, in Rust.

- **Signal:** Butterworth filter design (SciPy-equivalent), NeuroKit2-compatible zero-phase ECG cleaning,
  a causal **streaming** filter for packet-by-packet processing, R-peak detection (NeuroKit2 method and
  Elgendi two-moving-average detector), sub-sample R-peak refinement, a dual-detector API with bSQI,
  windowed processing of long recordings and a **streaming** detector with exactly the same output,
  Kubios artifact correction (NeuroKit2-equivalent; per-beat changed/inserted/removed result, also on sub-sample positions),
  an **exclusion mode** as the alternative to correction (ectopic / suspect beat classes, NN validity mask,
  artifact ratio, RMSSD from adjacent valid NN pairs), gap-aware RR intervals
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
| `detect::dual_detect_windowed` | both detectors of `dual_detect` over a long recording in the same windows (primary and secondary positions, each de-duplicated) |
| `stream::StreamDetector` + `StreamConfig` | the windowed pipeline packet by packet (see below): same peaks as `elgendi_refined_windowed` / `dual_detect_windowed`, bit for bit |

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

### Streaming detection

`StreamDetector` takes clean samples (e.g. from `EcgStreamFilter`) in packets of any size and processes
each 10 s window as soon as its 2 s right margin has arrived. It emits confirmed peaks (absolute sample
index + sub-sample position) of the primary detector and, by default, of the secondary detector. For a
segment starting at absolute index `s0`, the emitted peaks are **bit-identical** to
`elgendi_refined_windowed` / `dual_detect_windowed` on the same samples with `start_sample = s0`, for any
packetisation (tested with 1, 32, 33, 512, 7168 and random packet sizes, several start offsets, gaps and a
30 min run with artifacts: `tests/stream_detect.rs`).

- **Latency** (defaults): a peak is emitted 2.3–12 s after the R peak (window + margin; ~7 s on average);
  a peak in the last ~0.3 s of a window may be a duplicate of the next window's and waits for it, worst
  case window + margin + 2 · dedup ≈ 12.3 s (`max_latency_samples()`).
- **State**: one buffer of at most window + 2 · margin samples (14 s, 56 KiB at 512 Hz) plus pending peaks:
  `state_bytes()` = 59.9 KiB per stream at 512 Hz (500 streams ≈ 29 MiB), constant over time. The buffer
  is the chunk the detectors read; `push_with` gives each processed window (samples, its own detections,
  bSQI) to a callback, e.g. for signal quality on the core, so no second sample ring is needed.
- **Gaps**: `reset_at(next_index)` ends the segment (its last windows are processed with clipped chunks,
  as the batch path does at the end of a recording) and returns the `Gap` for `gaps::split_at_gaps`.
  `flush()` ends the recording.

```rust
use fast_hrv::stream::{StreamConfig, StreamDetector, StreamPeaks};
let mut filt = fast_hrv::EcgStreamFilter::new(512.0, &fast_hrv::EcgFilterConfig::default())?;
let mut det = StreamDetector::new(512.0, session_start_index, StreamConfig::default())?;
let mut out = StreamPeaks::default();
for packet in packets {                     // 32 raw samples
    det.push(&filt.process(packet), &mut out)?;
    // out.primary / out.secondary: confirmed peaks (index: u64, position: f64)
}
// A gap (connection lost, timestamps jump): close the segment and restart the filter state.
let gap = det.reset_at(det.next_index() + gap_samples, &mut out)?;
filt.reset();
// ... keep pushing packets of the next segment ...
det.flush(&mut out)?;
// RR per segment from sub-sample positions (never across `gap`):
let rri = fast_hrv::rri_from_positions(&positions_of_one_segment, 512.0);
```

A complete runnable version (two segments with a gap, a `push_with` callback, RR and RMSSD per segment)
is in [`examples/streaming.rs`](examples/streaming.rs): `cargo run --release --example streaming`.

**Reprocessing history must use the same causal path** (`EcgStreamFilter`, reset at every gap, then
`StreamDetector` or `dual_detect_windowed` on that output) to reproduce the live peaks. Cleaning history
with the zero-phase filter shifts peaks by ~8–10 ms; on noisy real recordings 16–26 % of them then no
longer match the live peaks within 50 ms.

## Artifact correction (Kubios)

`fix_peaks_with(peaks, fs, &KubiosParams::default())` runs the Lipponen–Tarvainen (2019) detection and
correction as NeuroKit2 `signal_fixpeaks(method="Kubios")` does (same output on the fixtures), iterating
while the number of flags decreases. `KubiosOutput` holds the corrected peaks, the artifact flags
(`artifacts`, cumulative over passes, and `passes`) and **`corrections`**: what really changed, decided by
comparing the output with the input.

- `corrections.status[j]` per output beat: `Unchanged` (same position as an input beat, also when it was
  flagged and ended where it started), `Moved` (ectopic or long/short correction), `Inserted` (missed beat);
  `corrections.removed` counts deleted (extra) input beats, which have no output beat;
  `input_to_output` / `output_to_input()` map between the two.
- **Gate on `changed_ratio()`** = (moved + inserted) / output beats, with removed beats reported separately
  (`removed_count()`). Do **not** use `artifacts.total_count() / n`: it counts flags of every pass,
  including flags that change nothing; on real wearable data it was 2–5× the share of beats actually changed.
- Segmented recordings: `gaps::segment_changed_counts`, `total_changed`, `total_removed`,
  `overall_changed_ratio` over the outputs of `fix_peaks_segmented`.
- Kubios has known blind spots that a ratio cannot show (a compensated premature beat moves two beats; at
  ~15 % artifacts the quartile thresholds inflate and it reports almost nothing): combine the ratio with a
  residual RR check and signal-quality gates.

```rust
let out = fast_hrv::fix_peaks_with(&peaks, 512.0, &fast_hrv::KubiosParams::default())?;
let ratio = out.changed_ratio();            // share of output beats actually changed
let removed = out.corrections.removed_count();
```

**Sub-sample positions.** `fix_positions_with(positions, fs, params)` runs the same algorithm on `f64`
positions (RR from the positions), so refined peaks need not be rounded: unchanged beats keep their
sub-sample positions bit for bit, inserted and moved beats go to the exact midpoint of their neighbours,
and the result (`KubiosPositions`) carries the same `corrections`. With `KubiosMidpoint::Floor`
(`fix_positions_with_midpoint`) integer-valued positions give results bit-identical to `fix_peaks_with`
(and NeuroKit2). Shortcuts: `RefinedPeaks::fix_kubios`, `stream::fix_stream_peaks` (one gap-free segment
of `StreamPeak`s), `gaps::fix_positions_segmented`, and `KubiosPositions::rri_ms(fs)` for the RR.
Integer RR series contain exact ties at some decision boundaries, so a sub-sample series can differ from
its rounded version in a few borderline flags (6 of 26 fixture series with ±0.45-sample offsets, same total
number of changes).

```rust
let r = refine_peaks(&clean, 512.0, &cand, &RefineConfig::default())?;
let fixed = r.fix_kubios(512.0, &fast_hrv::KubiosParams::default())?;
let rri = fixed.rri_ms(512.0);              // corrected sub-sample RR (ms)
let gate_ok = fixed.changed_ratio() <= 0.02;   // e.g. a 60 s window; the threshold is yours
```

**Parameters.** `KubiosParams` defaults are the published values (`alpha` 5.2, `c1` 0.13, `c2` 0.17,
91-beat quartile window, 11-beat median) and give NeuroKit2 parity. `alpha` scales both time-varying
thresholds (larger = fewer flags). `threshold_floor_s` (default `0.0` = off, bit-identical to the
published algorithm) is an absolute lower bound in seconds on both thresholds (Th1 for dRR, Th2 for mRR):
on very regular, low-HRV series the quartile thresholds shrink to a few milliseconds and small fiducial
jitter gets flagged and moved; a floor stops that. Th2 is also the missed/extra tolerance, which the floor
widens too.

```rust
let p = fast_hrv::KubiosParams { alpha: 6.0, threshold_floor_s: 0.040, ..Default::default() };
```

## Exclusion instead of correction

Kubios correction moves flagged beats to the midpoint of their neighbours. For a **compensated premature
beat** that is wrong: the beat is real, and moving it (often together with a neighbour) leaves 50–100 ms
steps in the NN series. On low-HRV recordings this multiplies RMSSD (synthetic test below: 3 % ectopy, median
RMSSD error 59 %, 146 % at RMSSD ≈ 10 ms). The exclusion mode classifies beats with the same detection but
**moves nothing**: the two NN intervals around an ectopic or suspect beat are excluded, and RMSSD uses only
successive differences of two adjacent valid NN intervals.

```rust
use fast_hrv::{ExclusionParams, classify_beats};

let out = classify_beats(&positions, 512.0, &ExclusionParams::default(), None)?; // f64 sub-sample positions
let td = out.time_domain();          // RMSSD / SDSD / pNNx from adjacent valid pairs, MeanNN / SDNN on valid NN
let rmssd = out.rmssd();             // Option<f64>: None with < 2 valid pairs
let kept = out.kept_count();         // valid NN; also usable_count(), kept_share()
let art = out.artifact_ratio();      // artifact events / beats (ectopic beats not counted)
let ect = out.ectopy_ratio();        // ectopic beats / beats (record it; not an artifact)
let w = out.summary_between(t0, t1); // the same counts for one window (positions in [t0, t1))
```

How it works (`exclusion` module docs have the details):

1. **Artifact passes:** Kubios with only the missed (insert at the midpoint) and extra (remove) corrections,
   iterated; ectopic and long/short flags never move a beat.
2. **Detection:** a beat is **ectopic** if the Kubios `ectopic` flag maps to it (the beat between the short
   and the long interval) or if it matches the premature + compensatory pattern
   `RR[b] ≤ 0.85·m` and `RR[b+1] − m ≥ 0.5·(m − RR[b])` (`m` = centred 11-beat median RR; both factors are
   parameters). Every other flag makes its beat a **suspect** (long/short or residual missed/extra).
3. **Two-stage** (default on): detection runs again on a copy with each ectopic beat at its neighbours'
   midpoint, so the ectopic dRR no longer inflate the thresholds; flags away from the ectopic beats become
   suspects. The copy is only used for detection.
4. **NN mask:** an interval is invalid if it is adjacent to an ectopic, suspect or caller-marked
   disagreement beat, touches an inserted beat, spans a removed beat (only with `exclude_inserted_removed`, default off since 0.6.1)
   or crosses a physiological split (`physiological_rr_s`, default off). Gap-free segments are processed
   separately (`classify_segments`, `concat_nn`).

Per beat: `out.beats[j]` (output beats: `Normal`, `Ectopic`, `Suspect`, `Disagreement`, `Inserted`, with
the Kubios / pattern source, the suspect stage and the midpoint move a suspect would need) and
`out.input_classes[i]` (input beats: also `Missed` = a beat was missing after it, `Extra`,
`PhysiologicalLimit`). The **artifact ratio** counts inserted, removed, physiological events, caller-given
primary/secondary disagreements (optional `&[bool]` per input beat) and suspects whose midpoint move
exceeds `suspect_move_s` (default 30 ms); ectopic beats are excluded from it. `classify_peaks` takes integer
indices. `time_domain_masked(nn_ms, valid)` works on any NN series with a validity mask (a `NaN` entry also
breaks the series); with every interval valid it equals `time_domain` bit for bit.

Accuracy on synthetic series (300 beats, RMSSD ≈ 10 / 20 / 40 ms, `tests/exclusion.rs`):

| Injected | Exclusion (two-stage) RMSSD error | Kubios interpolation RMSSD error |
|---|---|---|
| 3 % ectopy + 1 % missed/extra/jitter, 60 series | median 1.0 %, max 4.5 % | median 59 %, p90 150 % |
| 8 % ectopy + 2 % artifacts, 60 series | median 1.7 %, p90 6.1 % (single-stage: median 143 %) | — |

At 8 % ectopy single-stage detection is blind to most extra beats and jitter (acted on 219/360 artifacts;
two-stage 336/360). With Kubios flags only (`premature_factor = 0`) single-stage excludes 827/1440 ectopic
beats, two-stage 1175/1440.

> **Example only — product-side settings, not library defaults.** One product measured on its own data
> (single subject, wearable ECG) and chose: `KubiosParams { alpha: 6.0, ..Default::default() }` (c1/c2
> default, no threshold floor), two-stage exclusion, `suspect_move_s = 0.030`, primary/secondary
> disagreement (> 50 ms) passed as `disagreement`, physiological limits `(0.3, 2.0)`; per 60 s window:
> artifact ratio ≤ 2 %, ≥ 40 kept NN and kept share ≥ 80 %, residual check on the kept NN (none more than
> 30 % from the window median); per 300 s: ≥ 200 kept NN; ectopy ratio recorded, not gated. These are that
> product's thresholds; validate your own.

```rust
// the example settings above, in code
let params = fast_hrv::ExclusionParams {
    kubios: fast_hrv::KubiosParams { alpha: 6.0, ..Default::default() },
    physiological_rr_s: Some((0.3, 2.0)),
    ..Default::default()
};
let out = fast_hrv::classify_beats(&positions, 512.0, &params, Some(&disagreement))?;
let w = out.summary_between(t_end - 60.0 * 512.0, t_end);
let accept = w.artifact_ratio() <= 0.02 && w.kept_nn >= 40 && w.kept_share() >= 0.80;
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

**These per-band minimums are technical lower limits, not recommendations.** Below them a band cannot be
estimated at all (about 10 cycles of its lowest frequency, Task Force 1996). For interpretable values use
longer records: LF and LF/HF ≥ 180 s at the very least (the shortest window with validated agreement against
5 min records, Shaffer et al. 2019/2020), and ≥ 300 s (the Task Force short-term standard) when they are
reported or compared. LF/HF is not a sympathovagal-balance or stress index (Billman 2013; Quigley et al.
2024). Raise the limits per application with `FrequencyOptions`, e.g. `min_duration_lf_s: 300.0`.

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
| VLF, TINN, DFA α1/α2 | — | not comparable (different definitions; see `tests/neurokit_parity.rs`) |

Regenerate the references with `python tools/neurokit_reference.py` (and `filter_reference.py`,
`sqi_reference.py`, `exclusion_reference.py`); the parity tests fail if a reference file is missing (`FAST_HRV_SKIP_PARITY=1` skips them).

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
[CAPACITY.md](CAPACITY.md) (Turkish). `-- --stream` runs the detection stage through `StreamDetector`
(same cost within noise: 50.3 vs 50.5 µs per user per second of signal; memory per session unchanged).

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
