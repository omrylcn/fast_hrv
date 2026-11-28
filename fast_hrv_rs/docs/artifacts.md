# Artifacts: Kubios correction, exclusion, gaps

[← README](../README.md)

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
