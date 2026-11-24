# Changelog

All notable changes to this crate. The format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.7.0] - unreleased

### Added
- `frequency_domain_lomb(times_s, nn_ms, &FrequencyOptions)` and `lomb_scargle(t, y, freqs)`:
  Lomb–Scargle frequency domain for unevenly spaced NN (intervals removed by the exclusion mode or short
  gaps are left out, nothing is interpolated). Matches astropy `LombScargle(fit_mean=False,
  normalization="psd")` to 1e-9 of the peak; density scaled like Welch (a sine of amplitude A → A²/2).
- Heart rate fragmentation (Costa 2017): `fragmentation(nn_ms) -> Fragmentation { pip, ials, pss, pas }`,
  also in `HrvReport::fragmentation`. NeuroKit2 definitions (identical to 1.4e-16).
- `TimeDomain::ln_rmssd` (also in `time_domain_masked`) and `FrequencyDomain::lnhf` (NeuroKit2 `LnHF`).
- `metrics::METRICS` / `metric_info(name)`: shortest and recommended resting recording length per metric,
  with the source, so applications read their window rules from one place.

### Changed (breaking for struct literals)
- New public fields: `TimeDomain::ln_rmssd`, `FrequencyDomain::lnhf`, `HrvReport::fragmentation`.

## [0.6.1] - unreleased

### Documentation
- Per-band frequency-domain minimums (HF ≥ 60 s, LF ≥ 120 s, VLF ≥ 300 s) are documented as technical
  lower limits; recommended: LF and LF/HF ≥ 180 s, ≥ 300 s when reported or compared. No behaviour change.

### Changed
- `ExclusionParams::exclude_inserted_removed` now defaults to `false`. Measured on a real wearable
  recording set and on injected synthetic data: coverage identical, and `true` never improved RMSSD
  accuracy (slightly worse with 1–2 % artifacts, e.g. 60 s p90 7.9 % vs 7.1 %), because the interval
  spanning a removed extra beat is the true interval and an inserted beat sits at the midpoint with a
  small error. Set it to `true` explicitly to get the 0.6.0 behaviour.

## [0.6.0] - unreleased

### Added
- `KubiosParams::threshold_floor_s` (default `0.0` = off): absolute lower bound in seconds on both
  time-varying Lipponen–Tarvainen thresholds (Th1 for dRR, Th2 for mRR). With the default the floor is not
  applied at all, so the output is bit-identical to 0.5.0 and to NeuroKit2. Th2 is also the missed/extra
  tolerance, which a floor widens. `alpha` (default 5.2) documented as the threshold multiplier.
- `exclusion` module, an alternative to Kubios correction/interpolation that moves no beat:
  `classify_beats` (f64 positions), `classify_peaks` (integer indices), `classify_segments`, `concat_nn`;
  `ExclusionParams { kubios, premature_factor (0.85), compensatory_factor (0.5), two_stage (true),
  suspect_move_s (0.030), exclude_inserted_removed (true), physiological_rr_s (None) }`.
  Artifact passes insert missed / remove extra beats only; one detection pass classifies **ectopic** beats
  (Kubios `ectopic` flag mapped to the beat between the short and the long interval, or the premature +
  compensatory pattern) and **suspects** (other flags); optional two-stage detection on a copy with the
  ectopic beats at their neighbours' midpoint (detection only). Result `BeatExclusion`: output positions,
  per output beat `BeatFlags` (`BeatClass::{Normal, Ectopic, Suspect, Disagreement, Inserted}`, Kubios /
  pattern source, suspect stage and required midpoint move, disagreement), per input beat classes (also
  `Missed`, `Extra`, `PhysiologicalLimit`), `input_to_output`, removed beats, physiological splits,
  `nn_valid` / `nn_usable` masks; `kept_count`, `usable_count`, `kept_share`, `ectopic_count`,
  `ectopy_ratio`, `artifact_counts` / `artifact_counts_with` / `artifact_ratio` (`ArtifactCounts`:
  inserted, removed, physiological, far suspects, caller-given primary/secondary disagreement; ectopic
  beats excluded), `summary_between` (`ExclusionSummary` for a window), `nn_ms`, `time_domain`, `rmssd`.
- `time_domain::time_domain_masked(nn_ms, valid)` → `MaskedTimeDomain { time, n_nn, n_pairs }`: RMSSD,
  SDSD, pNN50, pNN20, CVSD only from successive differences of adjacent valid (and finite) NN intervals;
  distribution metrics on the valid NN; fewer than `MIN_MASKED_PAIRS` (2) pairs → NaN; all valid →
  bit-identical to `time_domain`. Also `masked_differences`, `rmssd_masked` (`Option`).
- `tools/exclusion_reference.py` and `tests/data/kubios_floor_reference.json`,
  `tests/data/exclusion_reference.json`: synthetic references from the Python research port (floor / alpha
  variants and the exclusion mode, compared bit for bit); `tests/kubios_floor.rs`, `tests/exclusion.rs`.

### Changed
- `KubiosParams` has a new public field `threshold_floor_s` (breaking only for code that builds it with all
  fields and no `..Default::default()`); `KubiosParams` and `ExclusionParams` derive `Serialize` /
  `Deserialize` with the `serde` feature.
- The correction APIs (`fix_peaks*`, `fix_positions_with*`, `detect_artifacts*`) are unchanged; NeuroKit2
  Kubios fixtures still 26/26 identical.

## [0.5.0] - unreleased

### Added
- Per-beat Kubios result: `KubiosOutput::corrections` (`Corrections`) with `status: Vec<BeatStatus>`
  (`Unchanged` / `Moved` / `Inserted` for every output beat), `removed` (deleted extra beats),
  `input_to_output`, and `n_output`, `n_input`, `moved_count`, `inserted_count`, `removed_count`,
  `changed_count`, `changed_ratio`, `changed_mask`, `output_to_input`. Decided by comparing the final
  peaks with the input: a beat flagged and then restored is `Unchanged`. `KubiosOutput::changed_count()`
  / `changed_ratio()` shortcuts. Ratio definition: (moved + inserted) / output beats; removed separately.
- `gaps::segment_changed_counts`, `total_changed`, `total_removed`, `overall_changed_ratio` (generic over
  `AsRef<Corrections>`).
- Kubios on sub-sample positions: `fix_positions_with(positions: &[f64], fs, params)` →
  `KubiosPositions { positions, artifacts, passes, corrections }` (`changed_count`, `changed_ratio`,
  `rri_ms`). Same algorithm with RR from the `f64` positions; unchanged beats keep their input positions
  bit for bit; inserted / moved beats at the exact neighbour midpoint. `fix_positions_with_midpoint` with
  `KubiosMidpoint::{Exact, Floor}`: `Floor` on integer-valued positions is bit-identical to
  `fix_peaks_with` (tested on the 26 NeuroKit2 fixtures). `detect_artifacts_positions` (one pass).
- Convenience paths: `RefinedPeaks::fix_kubios`, `stream::peak_positions`, `stream::fix_stream_peaks`,
  `gaps::fix_positions_segmented`.
- Tests: per-beat corrections (unit + fixtures), integer-valued / constant-offset / per-beat sub-sample
  positions against the integer path, refined synthetic ECG with a missed beat through Kubios.

### Changed
- Docs: `Artifacts::total_count()` (and `gaps::total_artifacts`) count artifact **flags** over all passes,
  including flags that change nothing; they are not a corrected-beat ratio. Use `changed_ratio()`.
- `KubiosOutput` has a new public field `corrections` (breaking only for code that builds or
  destructures `KubiosOutput` with all fields).

## [0.4.0] - unreleased

### Added
- `stream` module: `StreamDetector` with `StreamConfig`, `StreamPeak`, `StreamPeaks`, `StreamWindow`.
  Live, packet-by-packet form of the windowed detection pipeline: clean samples in packets of any size,
  each 10 s window (plus 2 s margins) processed as soon as its right margin is complete, primary
  (Elgendi + refinement) and optionally secondary (`dual_detect`) peaks emitted as absolute indices plus
  sub-sample positions. Output bit-identical to `elgendi_refined_windowed` / `dual_detect_windowed` for
  any packetisation. `push` / `push_with` (per-window callback with the chunk, its detections and bSQI),
  `flush`, `reset_at(next_index)` (gap; returns the `gaps::Gap`), `state_bytes`, `max_latency_samples`,
  `next_window_ready_at`. Latency 2.3–12 s (worst case ~12.3 s) with the defaults; state 59.9 KiB per
  stream at 512 Hz, allocated once.
- `detect::dual_detect_windowed` → `WindowedDual { primary, secondary }`: both detectors of `dual_detect`
  over a long recording on the absolute window grid (batch reference of `StreamDetector`).
- `tests/stream_detect.rs`: raw → `EcgStreamFilter` → `StreamDetector` vs the batch path for 1/32/33/512/
  7168/random packet sizes and seven start offsets, other grids and sampling rates, the window callback,
  gaps / `reset_at`, a 30 min synthetic run with artifacts (latency and state size printed); a
  de-duplication stress test with a dense fake detector (unit test).
- `benches/capacity.rs -- --stream`: stage 2a through `StreamDetector` (SQI in the window callback, no
  clean-sample ring per session).

## [0.3.0] - unreleased

### Added
- `detect` module:
  - `elgendi_peaks` / `elgendi_filter` / `elgendi_detect` with `ElgendiConfig`: Elgendi two-moving-average
    QRS detector (8–20 Hz Butterworth order 2, squared signal, 97 ms / 611 ms moving averages,
    β = 0.08, minimum block = QRS window, 0.3 s refractory). Presets `neurokit2013()` and
    `neurokit2010()` reproduce NeuroKit2 0.3.0 `elgendi2013` / `elgendi2010` peaks exactly;
    `DetectorFilter::ZeroPhase` option; `moving_average_causal` (bit-identical to NeuroKit2's MWA).
  - `refine_peaks` with `RefineConfig` / `Polarity` (`Auto`, `Positive`, `Negative`, `Absolute`) →
    `RefinedPeaks { indices, positions, sign }`: local extremum within ±50 ms on the clean signal plus
    3-point parabolic interpolation (sub-sample positions); `parabolic_offset`, `auto_polarity`.
  - `dual_detect` with `DualConfig` → `DualDetection`: Elgendi + refinement (primary) and the NeuroKit2
    method (secondary) on the same window, plus `sqi::Bsqi` (default tolerance 0.15 s). No gate.
  - `detect_windowed` / `elgendi_refined_windowed` with `WindowConfig`: fixed windows anchored at absolute
    sample indices (default 10 s, 2 s margin, 0.15 s de-duplication), detector restarted per window.
- `rri_from_positions`, `rri_from_times`, `try_rri_from_times`: RR intervals from sub-sample peaks.
- NeuroKit2 Elgendi fixtures (`tests/data/neurokit_elgendi.json`, `tools/detect_reference.py`) and
  `tests/detect.rs` (parity, refinement accuracy on synthetic ECG with known R, dual detector, 30 min
  windowed recording, start-index invariance); detector rows in `benches/throughput.rs`.
- `benches/capacity.rs` and `CAPACITY.md`: per-user CPU and memory of the planned live pipeline
  (500 interleaved synthetic sessions, aligned vs staggered schedules, per-second load spikes).

### Changed (breaking)
- `HrvError` has a new variant `InvalidParameter(&'static str)` (exhaustive matches must handle it).

## [0.2.0] - unreleased

First release of the `fast-hrv` crate, based on `fast_hrv_rs` (omrylcn/fast_hrv, commit `e969cb7`).

### Added
- `time_domain`, `frequency_domain` and `report::hrv()`: 41 HRV metrics in one call
  (17 time domain, 9 frequency domain, 9 nonlinear, 6 asymmetry).
- `filter` module: SciPy-equivalent Butterworth design (`butter_sos`, low/high/band-pass/band-stop),
  `iirnotch`, `sosfilt`, `sosfilt_zi`, `sosfiltfilt` and `filtfilt_ba` with SciPy's default padding,
  and `SosFilter`, a causal cascade that keeps its state between calls.
- ECG cleaning: `ecg_clean_neurokit` (`nk.ecg_clean(method="neurokit")` equivalent), `EcgStreamFilter` /
  `EcgFilterConfig` / `ecg_clean_causal` for live packet-by-packet processing (default 0.5–40 Hz,
  optional notch), `ecg_clean_zero_phase`.
- `sqi` module: absolute ECG signal quality indices (flat line, saturation, kSQI, pSQI, basSQI, bSQI,
  template correlation, RR plausibility, zhao2018 fusion) with NeuroKit2-compatible PSD mode.
- `gaps` module: gap-aware RR intervals (`split_at_gaps`, `split_peak_times`, `rri_with_gaps`,
  `fix_peaks_segmented`).
- Kubios: `detect_artifacts`, `KubiosParams`, `fix_peaks_with` → `KubiosOutput` (per-pass artifacts),
  pandas-compatible `rolling_quantile_centered` / `rolling_median_centered`.
- `FrequencyOptions` / `frequency_domain_with`: per-band minimum durations.
- `HrvError` and validating APIs (`try_rri_from_peaks`, `butter_sos`, `fix_peaks_with`, …).
- NeuroKit2 / SciPy parity tests with committed synthetic references (`tests/data`, generated by
  `tools/*.py` at development time), streaming and robustness tests, `benches/throughput.rs`.
- `rust-toolchain.toml` (Rust 1.100.0), LICENSE-MIT, LICENSE-APACHE.

### Changed (breaking behaviour)
- `ecg_clean` is now NeuroKit2-compatible (0.5 Hz HP order 5 zero-phase + 50 Hz moving average) instead of
  an approximate 0.5–40 Hz band-pass; output values change.
- `design_butterworth_bandpass` returns real Butterworth sections (SciPy order semantics); invalid
  parameters return an empty vector.
- `fix_peaks` / `fix_peaks_default` return artifacts cumulative over all correction passes; unsorted
  input is sorted and de-duplicated instead of panicking.
- `frequency_domain`: band edges integrated exactly (no double-counted trapezoids); HF/LF/VLF need
  60/120/300 s of RR data, otherwise `NaN`; constant RR gives zero power and `NaN` ratios.
- HTI/TINN histogram on the 0-based 7.8125 ms grid (as NeuroKit2); TINN is `NaN` when the RR range
  exceeds 2048 bins (16 s).
- `rri_from_peaks`: non-increasing peak pairs give `NaN` instead of overflowing.
- Python module renamed to `fast_hrv` (was `fast_hrv_rs`, which did not match the library name).
- `parallel` feature is a no-op (rayon removed); `serde_json` is a dev-dependency only.

### Fixed
- Kubios: removed an extra loop increment that skipped the beat after every long/short check; rolling
  thresholds and median now match pandas `rolling(center=True, min_periods=1)` (also for < 45 beats);
  `np.pad(..., "reflect")` subspaces.
- Panics on empty input (`compute_poincare`, `compute_asymmetry`), `NaN` input (`time_domain`,
  `frequency_domain`), unsorted peaks, non-contiguous / negative arrays in the Python bindings.
- TINN cost: O(B²) separable search with a bin cap (a 69 s RR took ~0.8 s, now ~2 ms).
