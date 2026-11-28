# Metrics, frequency methods, breathing and window rules

[← README](../README.md)

## Metrics

| Group | Metrics |
|---|---|
| Time domain | MeanNN, SDNN, RMSSD, lnRMSSD, SDSD, SDRMSSD, pNN50, pNN20, MedianNN, MadNN, IQRNN, Prc20NN, Prc80NN, MinNN, MaxNN, CVNN, CVSD, MeanHR, SDHR, HTI, TINN, Baevsky stress index (and √SI) |
| Frequency domain: Welch on 4 Hz cubic-spline resampling (`frequency_domain`), Lomb–Scargle on the uneven beat times (`frequency_domain_lomb`), or NeuroKit2-compatible (`frequency_domain_neurokit`) | VLF, LF, HF, lnHF, total power, LF/HF, LFnu, HFnu, LF peak, HF peak |
| Nonlinear | SD1, SD2, SD1/SD2, S (ellipse area), CSI, CSI modified, CVI, SampEn, ApEn, DFA α1, DFA α2 |
| Asymmetry | GI, PI, SD1d, SD1a, C1d, C1a |
| Fragmentation (Costa 2017, NeuroKit2 definitions) | PIP, IALS, PSS, PAS |

**Lomb–Scargle** (`frequency_domain_lomb(times_s, nn_ms, &opts)`) needs no interpolation: pass only the
valid intervals at their beat times (e.g. after the exclusion mode, or with short gaps) and nothing is
filled in. Classical periodogram, density scaled so that a sine of amplitude A gives A²/2 (as Welch);
grid step 1 / (5 · T) up to 0.5 Hz; same bands, minimum durations and outputs as Welch. The two estimators
agree within ~15 % on gap-free series; compare values only within one method. About 0.7 ms per 5 min window.

**NeuroKit2-compatible frequency domain** (`frequency_domain_neurokit(rri)`) reproduces `nk.hrv_frequency`
with its defaults: 100 Hz quadratic spline, Welch with Hann windows of half the series (FFT length
2·nperseg), in-band trapezoid. LF/HF equals NeuroKit2's to ≤ 3e-13 relative (test on simulated ECG, 2–30
min). Band powers stay in ms² (NeuroKit2 normalises its PSD to the maximum, so only its ratios compare) and
no minimum durations apply. The default `frequency_domain` (Task Force / Kubios practice: 4 Hz cubic spline,
64 s segments) and this mode differ mainly through the window length: LF/HF moves by a few percent on long
records and up to ~25 % on short ones. Pick one method and compare LF/HF only within it.

**Breathing rate (ECG-derived respiration):** `r_amplitudes(&clean, &peaks, fs)` and
`respiratory_rate(&beat_times_s, &rr_ms, &r_amp, &RespirationConfig::default())` estimate breaths/min in a
30–60 s window from the R-amplitude and RR modulations (Lomb–Scargle peak in 0.1–0.5 Hz each; valid only
when both agree within 4/min, Karlen et al. 2013). Below 9/min the respiratory component falls into the LF
band; `hf_band_for(rate, 0.1, 0.04)` gives an HF band centred on the breathing rate. Checked on synthetic
ECG (9–22/min, error ≤ 0.01/min) and NeuroKit2 ECGSYN (15/min, 5 of 6 windows valid, error ≤ 0.7/min); not
yet validated against a reference respiration signal.

**Window rules per metric:** `METRICS` / `metric_info("lf_hf")` give, for every field, the shortest
(`min_s`) and the recommended (`recommended_s`) resting recording length with the source of the numbers, so
an application can read its window rules from the library instead of copying them.

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

## Detrending (smoothness priors)

`detrend::smoothness_priors(rri, lambda)` removes slow trends from an RR series (Tarvainen et al. 2002,
the method Kubios uses): a banded solver for `(I + λ² D₂ᵀD₂)`, O(n). Equal to NeuroKit2
`signal_detrend(method="tarvainen2002")` to ≤ 7e-10 (`tests/neurokit_detrend.rs`).
`smoothness_priors_cutoff_hz(lambda, fs)` gives the −3 dB cut-off: λ = 500 on a 4 Hz series ≈ 0.035 Hz,
below the LF band.

## Presets and provenance

Different tools give different LF/HF for the same data mainly because of different processing choices.
`hrv_with_preset(rri, preset)` runs one named set of choices and returns a `Provenance` record with the
result, so a stored value says how it was computed:

| `Preset` | Detrending | Frequency method |
|---|---|---|
| `TaskForce` (default) | none | Welch, 4 Hz cubic spline, 64 s segments |
| `KubiosLike` | smoothness priors, λ = 500 (mean level kept) | Welch, 4 Hz cubic spline, 300 s segments |
| `NeuroKit` | none | `frequency_domain_neurokit` (NeuroKit2 defaults) |

`KubiosLike` follows Kubios' published defaults; it is not checked against Kubios output.

## Per-window report: what may be shown

`window_report(&report, &context)` (or `window_report_rri(rri, corrected, breathing_bpm)`) gives every
metric of a window a verdict (`Valid`, `Warning`, `Suppressed`) with reason codes, so the rules live in the
library and the computed value stays separate from what is displayed:

| Reason | Verdict | Rule |
|---|---|---|
| `not_computed` | suppressed | value is NaN |
| `too_short` | suppressed | window shorter than the metric's `min_s` |
| `too_many_corrections` | suppressed | corrected / excluded beats above `max_artifact_ratio()` (DFA α1 3 %, others 5 %) |
| `breathing_in_lf_band` | suppressed (LF/HF, LFnu, HFnu) / warning (LF, HF) | breathing < 9 /min moves the respiratory peak into LF |
| `breathing_above_hf_band` | warning (HF) | breathing > 24 /min |
| `below_strict_length` | warning | shorter than `strict_min_s()` (no validated agreement with 5 min records) |
| `consecutive_corrections` | warning (beat-to-beat metrics) | two or more corrected beats in a row |

Output of [`examples/window_report.rs`](../examples/window_report.rs) (60 s, two corrected beats in a row,
breathing 7 /min):

```text
window 60 s, artifact ratio 2.8 %
mean_hr     70.79  valid
rmssd       19.41  warning    consecutive_corrections
sdnn        31.97  valid
hf           2.90  warning    below_strict_length, breathing_in_lf_band
lf_hf         NaN  suppressed not_computed, too_short, breathing_in_lf_band
sampen       0.18  suppressed too_short
```

## Breathing-centred HF

`respiratory_hf(rri, breathing_bpm, &opts)` integrates the spectrum in a band centred on the measured
breathing rate (`hf_band_for`) instead of the fixed 0.15–0.4 Hz, and flags `in_lf_band` / `above_hf_band`.
`band_power_rri(rri, band, &opts)` gives the power of any band.

## Accelerometer context

`motion::motion_context(xyz_g, reference, &MotionConfig::default())` summarises a window of accelerometer
samples: magnitude mean and SD, ENMO (van Hees et al. 2013), an activity class (rest / light / active), the
gravity direction, tilt against a reference posture and a posture-change flag (first vs second half of the
window). The default thresholds (0.02 g / 0.06 g SD, 30°) are conventions for a chest sensor at ~1 Hz, not
validated cut-offs; gating rules belong to the application.
