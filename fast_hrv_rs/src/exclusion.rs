//! Exclusion mode: classify beats with the Kubios (Lipponen–Tarvainen 2019) detection and **exclude**
//! the NN intervals around ectopic and suspect beats instead of moving beats (correction /
//! interpolation).
//!
//! Why: a compensated premature beat is a real beat at a real time. Kubios correction moves it (and
//! often a neighbour) to the midpoint of its neighbours, which leaves 50–100 ms steps in the NN
//! series; on low-HRV recordings that multiplies RMSSD. Dropping the two NN intervals adjacent to the
//! premature beat and computing RMSSD only from successive differences of two adjacent kept NN
//! intervals (`time_domain::time_domain_masked`) keeps RMSSD close to the sinus value. The
//! correction APIs ([`crate::peaks::fix_peaks_with`], [`crate::peaks::fix_positions_with`]) are
//! unchanged; this is an alternative.
//!
//! # Algorithm ([`classify_beats`])
//!
//! Notation: output positions `p[k]`, `RR[k] = p[k] − p[k−1]` (the interval ending at beat `k`),
//! `m[k]` = Kubios' centred rolling median RR (`medfilt_order` beats, 11 by default).
//!
//! 0. *(optional, [`ExclusionParams::physiological_rr_s`])* physiological limits: a beat closer than
//!    `min` to the previous kept beat is dropped ([`BeatClass::PhysiologicalLimit`]); an interval
//!    longer than `max` splits the series (each run is processed alone, the interval is neither
//!    usable nor valid). Each drop and each split is one physiological event.
//! 1. **Artifact passes:** Kubios detection with only the missed (insert a beat at the midpoint) and
//!    extra (remove the beat) corrections, repeated while the number of those flags decreases
//!    (`kubios.iterative`). Ectopic and long/short flags never move a beat.
//! 2. **One Kubios detection pass** on the result. A beat `b` is **ectopic** if
//!    - (a) Kubios flags `ectopic` at index `j` and `b` is the beat of `{j−1, j}` whose two adjacent
//!      RR differ most (`|RR[b+1] − RR[b]|` largest, `j−1` on a tie), i.e. the beat between the short
//!      and the long interval (the Lipponen–Tarvainen flag of a compensated premature beat usually
//!      lands on the beat after it); or
//!    - (b) the premature + compensatory pattern: `RR[b] ≤ premature_factor · m[b]` and
//!      `RR[b+1] − m[b] ≥ compensatory_factor · (m[b] − RR[b])`.
//!
//!    A long/short (or residual missed / extra) flag at `b` or `b+1` of an ectopic beat belongs to
//!    that ectopic event; every other one makes its beat a **suspect** (stage 1).
//! 3. *(two-stage, default)* when there are ectopic beats, the detection runs again on a **copy** in
//!    which each ectopic beat is placed at the midpoint of its neighbours. The copy only serves the
//!    detection: ectopic dRR no longer inflate the quartile thresholds, so jitter and extra beats
//!    that they hid become visible. Flags of this pass farther than one beat from an ectopic beat
//!    make their beat a suspect (stage 2). **No output beat moves.**
//! 4. **NN validity:** the NN interval between beats `k` and `k+1` is invalid if either beat is
//!    ectopic, suspect, or marked by the caller's primary/secondary disagreement, if it crosses a
//!    physiological split, and (with [`ExclusionParams::exclude_inserted_removed`], default) if
//!    either beat was inserted or a removed beat lies inside it. Gaps between segments are handled by
//!    processing each gap-free segment separately ([`classify_segments`], [`concat_nn`]).
//!
//! # Artifact ratio ([`BeatExclusion::artifact_counts`])
//!
//! Only **artifact events** count: inserted beats, removed (extra) beats, physiological events,
//! disagreement beats (when the caller passes them) and suspects whose required Kubios midpoint move
//! exceeds [`ExclusionParams::suspect_move_s`] (a residual missed/extra flag or an edge beat always
//! counts). Ectopic beats do **not** count: they are physiology, recorded by
//! [`BeatExclusion::ectopy_ratio`]. The ratio divides by the number of output beats.
//!
//! The defaults are neutral or come from the measurement that motivated the mode; product thresholds
//! (the gate on the artifact ratio, the minimum kept NN, `alpha`) are left to the caller.

use crate::error::HrvError;
use crate::peaks::{KubiosParams, detect_artifacts_positions, detect_detail, fix_core};
use crate::report::rri_from_positions;

/// Parameters of the exclusion mode ([`classify_beats`]).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExclusionParams {
    /// Kubios detection parameters (alpha, c1, c2, windows, `iterative` for the artifact passes,
    /// `threshold_floor_s`). Default: [`KubiosParams::default`] (published values).
    pub kubios: KubiosParams,
    /// Premature beat: `RR[b] ≤ premature_factor · m[b]`. Default 0.85 (≥ 15 % premature, a common
    /// prematurity criterion). Set to 0 to disable the pattern rule (Kubios flags only).
    pub premature_factor: f64,
    /// Compensatory pause: `RR[b+1] − m[b] ≥ compensatory_factor · (m[b] − RR[b])`. Default 0.5 (at
    /// least half of the prematurity is compensated).
    pub compensatory_factor: f64,
    /// Second detection pass on a copy with the ectopic beats at their neighbours' midpoint (see the
    /// module docs). Default `true`.
    pub two_stage: bool,
    /// A suspect beat counts as an artifact event only if Kubios would move it by more than this
    /// (seconds) to the midpoint of its neighbours. Default 0.030 s ("fiducial far from R": 1.5× a
    /// 20 ms fiducial offset and below a 50 ms primary/secondary tolerance).
    pub suspect_move_s: f64,
    /// Invalidate the NN intervals touching an inserted beat or spanning a removed beat. Default
    /// `true`. `false` reproduces the research reference, which kept them.
    pub exclude_inserted_removed: bool,
    /// Physiological RR limits `(min, max)` in seconds (e.g. `(0.3, 2.0)`); `None` (default) = no
    /// limit.
    pub physiological_rr_s: Option<(f64, f64)>,
}

impl Default for ExclusionParams {
    fn default() -> Self {
        Self {
            kubios: KubiosParams::default(),
            premature_factor: 0.85,
            compensatory_factor: 0.5,
            two_stage: true,
            suspect_move_s: 0.030,
            exclude_inserted_removed: true,
            physiological_rr_s: None,
        }
    }
}

/// Class of a beat in the exclusion mode.
///
/// Output beats ([`BeatExclusion::beats`]) are `Normal`, `Ectopic`, `Suspect`, `Disagreement` or
/// `Inserted`. Input beats ([`BeatExclusion::input_classes`]) are `Normal`, `Ectopic`, `Suspect`,
/// `Disagreement`, `Missed`, `Extra` or `PhysiologicalLimit`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BeatClass {
    /// Sinus beat; its NN intervals are valid unless a neighbour is excluded.
    Normal,
    /// Ectopic (Kubios ectopic flag mapped to the beat between the short and the long interval, or
    /// the premature + compensatory pattern). Both adjacent NN intervals are excluded. Not an
    /// artifact.
    Ectopic,
    /// Long/short suspect: any other Kubios flag (long/short, residual missed / extra, or a stage-2
    /// flag). Both adjacent NN intervals are excluded; an artifact event only if its required move
    /// exceeds [`ExclusionParams::suspect_move_s`].
    Suspect,
    /// The caller marked this beat as a primary/secondary detector disagreement. Both adjacent NN
    /// intervals are excluded; an artifact event.
    Disagreement,
    /// Output only: a beat inserted at the midpoint of a missed-beat gap.
    Inserted,
    /// Input only: a beat was missed right after this beat (an inserted beat follows it).
    Missed,
    /// Input only: removed as an extra beat by the artifact passes.
    Extra,
    /// Input only: dropped by the physiological minimum RR.
    PhysiologicalLimit,
}

/// Why an input beat has no output beat.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RemovalReason {
    /// Kubios extra-beat correction.
    Extra,
    /// Closer than the physiological minimum RR to the previous kept beat.
    PhysiologicalLimit,
}

/// An input beat that is not in the output.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RemovedBeat {
    pub input_index: usize,
    /// Its input position.
    pub position: f64,
    pub reason: RemovalReason,
}

/// What the exclusion mode found for one output beat.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BeatFlags {
    /// Summary class (priority: `Inserted` > `Ectopic` > `Suspect` > `Disagreement` > `Normal`).
    pub class: BeatClass,
    /// Input beat this output beat is (`None` for an inserted beat).
    pub input_index: Option<usize>,
    /// Inserted by a missed-beat correction.
    pub inserted: bool,
    /// Ectopic by the mapped Kubios `ectopic` flag (rule a).
    pub ectopic_kubios: bool,
    /// Ectopic by the premature + compensatory pattern (rule b).
    pub ectopic_pattern: bool,
    /// 0 = not a suspect, 1 = flagged by the first detection pass, 2 = by the two-stage pass.
    pub suspect_stage: u8,
    /// For a suspect: distance (seconds) Kubios would move it to its neighbours' midpoint (on the
    /// detection copy for stage 2); `+∞` for a residual missed / extra flag or an edge beat. `NaN`
    /// for a beat that is not a suspect.
    pub suspect_move_s: f64,
    /// Primary/secondary disagreement given by the caller.
    pub disagreement: bool,
}

impl BeatFlags {
    fn new(input_index: Option<usize>) -> Self {
        Self {
            class: BeatClass::Normal,
            input_index,
            inserted: input_index.is_none(),
            ectopic_kubios: false,
            ectopic_pattern: false,
            suspect_stage: 0,
            suspect_move_s: f64::NAN,
            disagreement: false,
        }
    }

    /// Ectopic by either rule.
    pub fn is_ectopic(&self) -> bool {
        self.ectopic_kubios || self.ectopic_pattern
    }

    /// Flagged as a suspect (stage 1 or 2).
    pub fn is_suspect(&self) -> bool {
        self.suspect_stage > 0
    }

    /// A suspect whose required move exceeds `move_s` (an artifact event).
    pub fn is_far_suspect(&self, move_s: f64) -> bool {
        self.is_suspect() && self.suspect_move_s > move_s
    }
}

/// Artifact events (see the module docs). A beat can appear in two categories (e.g. an inserted beat
/// that is also a far suspect); [`ArtifactCounts::total`] sums the categories.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ArtifactCounts {
    pub inserted: usize,
    /// Extra beats removed.
    pub removed: usize,
    /// Physiological-limit drops plus splits.
    pub physiological: usize,
    /// Suspects whose required move exceeds the limit.
    pub suspect_far: usize,
    /// Disagreement beats (0 unless the caller passed disagreement flags).
    pub disagreement: usize,
}

impl ArtifactCounts {
    pub fn total(&self) -> usize {
        self.inserted + self.removed + self.physiological + self.suspect_far + self.disagreement
    }
}

/// Counts of [`BeatExclusion::summary_between`] for one time window.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExclusionSummary {
    /// Output beats in the window.
    pub beats: usize,
    /// Ectopic beats in the window.
    pub ectopic: usize,
    /// Valid NN intervals with both beats in the window.
    pub kept_nn: usize,
    /// Usable NN intervals (not crossing a split) with both beats in the window.
    pub usable_nn: usize,
    /// Artifact events in the window.
    pub artifacts: ArtifactCounts,
}

impl ExclusionSummary {
    /// `kept_nn / usable_nn` (`NaN` without usable NN).
    pub fn kept_share(&self) -> f64 {
        ratio(self.kept_nn, self.usable_nn)
    }
    /// Artifact events / beats (`NaN` without beats).
    pub fn artifact_ratio(&self) -> f64 {
        ratio(self.artifacts.total(), self.beats)
    }
    /// Ectopic beats / beats (`NaN` without beats).
    pub fn ectopy_ratio(&self) -> f64 {
        ratio(self.ectopic, self.beats)
    }
}

fn ratio(a: usize, b: usize) -> f64 {
    if b == 0 {
        f64::NAN
    } else {
        a as f64 / b as f64
    }
}

/// Result of [`classify_beats`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BeatExclusion {
    /// Output beat positions: the input beats minus the removed ones plus the inserted ones. Kept
    /// input beats keep their input value bit for bit; no beat is moved.
    pub positions: Vec<f64>,
    /// Flags of each output beat.
    pub beats: Vec<BeatFlags>,
    /// Class of each input beat.
    pub input_classes: Vec<BeatClass>,
    /// Output index of each input beat (`None` = removed).
    pub input_to_output: Vec<Option<usize>>,
    /// Removed input beats (extra / physiological limit), in input order.
    pub removed: Vec<RemovedBeat>,
    /// Output indices `k` whose NN interval `k` (between beats `k` and `k+1`) crosses a
    /// physiological split.
    pub splits: Vec<usize>,
    /// Validity of each NN interval (`len = positions.len() − 1`, interval `k` between beats `k` and
    /// `k+1`).
    pub nn_valid: Vec<bool>,
    /// Usable NN intervals (all but the splits).
    pub nn_usable: Vec<bool>,
    /// Sampling rate the positions are in.
    pub sampling_rate: f64,
    /// Parameters used.
    pub params: ExclusionParams,
}

impl BeatExclusion {
    /// NN intervals (ms) of the output positions; pair with [`BeatExclusion::nn_valid`].
    pub fn nn_ms(&self) -> Vec<f64> {
        rri_from_positions(&self.positions, self.sampling_rate)
    }

    /// Valid NN intervals.
    pub fn kept_count(&self) -> usize {
        self.nn_valid.iter().filter(|&&v| v).count()
    }

    /// Usable NN intervals (not crossing a physiological split).
    pub fn usable_count(&self) -> usize {
        self.nn_usable.iter().filter(|&&v| v).count()
    }

    /// `kept_count / usable_count` (`NaN` without usable NN).
    pub fn kept_share(&self) -> f64 {
        ratio(self.kept_count(), self.usable_count())
    }

    /// Ectopic output beats.
    pub fn ectopic_count(&self) -> usize {
        self.beats.iter().filter(|b| b.is_ectopic()).count()
    }

    /// Ectopic beats / output beats (to record; not an artifact measure). `NaN` without beats.
    pub fn ectopy_ratio(&self) -> f64 {
        ratio(self.ectopic_count(), self.positions.len())
    }

    /// Artifact events with [`ExclusionParams::suspect_move_s`].
    pub fn artifact_counts(&self) -> ArtifactCounts {
        self.artifact_counts_with(self.params.suspect_move_s)
    }

    /// Artifact events with another suspect move limit (seconds).
    pub fn artifact_counts_with(&self, suspect_move_s: f64) -> ArtifactCounts {
        self.summary_impl(f64::NEG_INFINITY, f64::INFINITY, suspect_move_s)
            .artifacts
    }

    /// `artifact_counts().total() / output beats` (`NaN` without beats).
    pub fn artifact_ratio(&self) -> f64 {
        ratio(self.artifact_counts().total(), self.positions.len())
    }

    /// Counts for the window `[start, end)` (same unit as the positions). A beat or event is in the
    /// window if its position is (removed beats and drops at their input position, a split at the
    /// first beat after it); an NN interval if both its beats are.
    pub fn summary_between(&self, start: f64, end: f64) -> ExclusionSummary {
        self.summary_impl(start, end, self.params.suspect_move_s)
    }

    fn summary_impl(&self, start: f64, end: f64, move_s: f64) -> ExclusionSummary {
        let inw = |x: f64| x >= start && x < end;
        let mut s = ExclusionSummary::default();
        for (j, (b, &p)) in self.beats.iter().zip(&self.positions).enumerate() {
            if !inw(p) {
                continue;
            }
            s.beats += 1;
            s.ectopic += b.is_ectopic() as usize;
            s.artifacts.inserted += b.inserted as usize;
            s.artifacts.suspect_far += b.is_far_suspect(move_s) as usize;
            s.artifacts.disagreement += b.disagreement as usize;
            if j + 1 < self.positions.len() && inw(self.positions[j + 1]) {
                s.usable_nn += self.nn_usable[j] as usize;
                s.kept_nn += self.nn_valid[j] as usize;
            }
        }
        for r in self.removed.iter().filter(|r| inw(r.position)) {
            match r.reason {
                RemovalReason::Extra => s.artifacts.removed += 1,
                RemovalReason::PhysiologicalLimit => s.artifacts.physiological += 1,
            }
        }
        s.artifacts.physiological += self
            .splits
            .iter()
            .filter(|&&k| inw(self.positions[k + 1]))
            .count();
        s
    }
}

fn check_positions(positions: &[f64], sampling_rate: f64) -> Result<(), HrvError> {
    if positions.iter().any(|p| !p.is_finite()) {
        return Err(HrvError::NonFinite);
    }
    if let Some(i) = positions.windows(2).position(|w| w[1] <= w[0]) {
        return Err(HrvError::UnsortedPeaks { index: i + 1 });
    }
    if !(sampling_rate.is_finite() && sampling_rate > 0.0) {
        return Err(HrvError::InvalidSamplingRate(sampling_rate));
    }
    Ok(())
}

fn check_params(p: &ExclusionParams) -> Result<(), HrvError> {
    if !(p.premature_factor.is_finite() && p.premature_factor >= 0.0) {
        return Err(HrvError::InvalidParameter("premature_factor"));
    }
    if !p.compensatory_factor.is_finite() {
        return Err(HrvError::InvalidParameter("compensatory_factor"));
    }
    if p.suspect_move_s.is_nan() {
        return Err(HrvError::InvalidParameter("suspect_move_s"));
    }
    if let Some((lo, hi)) = p.physiological_rr_s {
        if !(lo.is_finite() && hi.is_finite() && lo >= 0.0 && hi > lo) {
            return Err(HrvError::InvalidParameter("physiological_rr_s"));
        }
    }
    Ok(())
}

/// Exclusion-mode classification of strictly increasing `f64` beat positions (samples, or any unit
/// that `sampling_rate` converts to seconds) from one gap-free recording. See the module docs.
///
/// `disagreement`: optional per **input** beat flags (e.g. no secondary detection within 50 ms);
/// such beats have their NN excluded and count as artifact events. Its length must equal
/// `positions.len()` ([`HrvError::InvalidParameter`]). Inserted beats go to the exact midpoint.
pub fn classify_beats(
    positions: &[f64],
    sampling_rate: f64,
    params: &ExclusionParams,
    disagreement: Option<&[bool]>,
) -> Result<BeatExclusion, HrvError> {
    classify_impl(positions, sampling_rate, params, disagreement, |a, b| {
        a + (b - a) / 2.0
    })
}

/// [`classify_beats`] on integer sample indices. Inserted beats go to `prev + floor((next − prev) /
/// 2)` (the integer Kubios rule), so every output position is integer-valued.
pub fn classify_peaks(
    peaks: &[usize],
    sampling_rate: f64,
    params: &ExclusionParams,
    disagreement: Option<&[bool]>,
) -> Result<BeatExclusion, HrvError> {
    if let Some(i) = peaks.windows(2).position(|w| w[1] <= w[0]) {
        return Err(HrvError::UnsortedPeaks { index: i + 1 });
    }
    let pos: Vec<f64> = peaks.iter().map(|&p| p as f64).collect();
    classify_impl(&pos, sampling_rate, params, disagreement, |a, b| {
        a + ((b - a) / 2.0).floor()
    })
}

/// [`classify_beats`] on each gap-free segment (e.g. from [`crate::gaps::split_peak_times`]);
/// `disagreement`, if given, holds one flag vector per segment.
pub fn classify_segments(
    segments: &[Vec<f64>],
    sampling_rate: f64,
    params: &ExclusionParams,
    disagreement: Option<&[Vec<bool>]>,
) -> Result<Vec<BeatExclusion>, HrvError> {
    if disagreement.is_some_and(|d| d.len() != segments.len()) {
        return Err(HrvError::InvalidParameter(
            "disagreement (one vector per segment)",
        ));
    }
    segments
        .iter()
        .enumerate()
        .map(|(i, s)| {
            classify_beats(
                s,
                sampling_rate,
                params,
                disagreement.map(|d| d[i].as_slice()),
            )
        })
        .collect()
}

/// NN intervals (ms) and validity of several segments concatenated, with one `NaN` / invalid entry
/// between consecutive segments, so that no successive difference spans a gap. Feed the result to
/// `time_domain::time_domain_masked`.
pub fn concat_nn(outputs: &[BeatExclusion]) -> (Vec<f64>, Vec<bool>) {
    let mut nn = Vec::new();
    let mut valid = Vec::new();
    for (i, o) in outputs.iter().enumerate() {
        if i > 0 {
            nn.push(f64::NAN);
            valid.push(false);
        }
        nn.extend(o.nn_ms());
        valid.extend_from_slice(&o.nn_valid);
    }
    (nn, valid)
}

/// Result of one run (gap- and split-free stretch of beats).
struct RunOut {
    positions: Vec<f64>,
    beats: Vec<BeatFlags>,
    /// run-local input index of each removed (extra) beat
    removed: Vec<usize>,
}

fn classify_impl(
    positions: &[f64],
    fs: f64,
    params: &ExclusionParams,
    disagreement: Option<&[bool]>,
    mid: fn(f64, f64) -> f64,
) -> Result<BeatExclusion, HrvError> {
    check_positions(positions, fs)?;
    check_params(params)?;
    if disagreement.is_some_and(|d| d.len() != positions.len()) {
        return Err(HrvError::InvalidParameter(
            "disagreement (one flag per input beat)",
        ));
    }
    let n_in = positions.len();

    // 0. physiological limits: drops and runs (input indices)
    let mut removed: Vec<RemovedBeat> = Vec::new();
    let mut runs: Vec<Vec<usize>> = vec![Vec::new()];
    let mut last: Option<usize> = None;
    for i in 0..n_in {
        if let (Some((lo, hi)), Some(l)) = (params.physiological_rr_s, last) {
            let rr = (positions[i] - positions[l]) / fs;
            if rr < lo {
                removed.push(RemovedBeat {
                    input_index: i,
                    position: positions[i],
                    reason: RemovalReason::PhysiologicalLimit,
                });
                continue;
            }
            if rr > hi {
                runs.push(Vec::new());
            }
        }
        runs.last_mut().unwrap().push(i);
        last = Some(i);
    }

    // 1.–3. per run
    let mut out_pos: Vec<f64> = Vec::with_capacity(n_in);
    let mut beats: Vec<BeatFlags> = Vec::with_capacity(n_in);
    let mut splits = Vec::new();
    for (ri, run) in runs.iter().enumerate() {
        if run.is_empty() {
            continue;
        }
        if ri > 0 && !out_pos.is_empty() {
            splits.push(out_pos.len() - 1);
        }
        let pos: Vec<f64> = run.iter().map(|&i| positions[i]).collect();
        let dis: Option<Vec<bool>> = disagreement.map(|d| run.iter().map(|&i| d[i]).collect());
        let r = classify_run(&pos, fs, params, dis.as_deref(), mid);
        for &k in &r.removed {
            removed.push(RemovedBeat {
                input_index: run[k],
                position: pos[k],
                reason: RemovalReason::Extra,
            });
        }
        for mut b in r.beats {
            b.input_index = b.input_index.map(|k| run[k]);
            beats.push(b);
        }
        out_pos.extend(r.positions);
    }
    removed.sort_by_key(|r| r.input_index);

    // classes of the output beats
    for b in &mut beats {
        b.class = if b.inserted {
            BeatClass::Inserted
        } else if b.is_ectopic() {
            BeatClass::Ectopic
        } else if b.is_suspect() {
            BeatClass::Suspect
        } else if b.disagreement {
            BeatClass::Disagreement
        } else {
            BeatClass::Normal
        };
    }

    // 4. NN validity
    let n = out_pos.len();
    let n_nn = n.saturating_sub(1);
    let mut nn_usable = vec![true; n_nn];
    for &k in &splits {
        nn_usable[k] = false;
    }
    let excl_ir = params.exclude_inserted_removed;
    let flagged: Vec<bool> = beats
        .iter()
        .map(|b| b.is_ectopic() || b.is_suspect() || b.disagreement || (excl_ir && b.inserted))
        .collect();
    let mut nn_valid: Vec<bool> = (0..n_nn)
        .map(|k| nn_usable[k] && !flagged[k] && !flagged[k + 1])
        .collect();
    if excl_ir {
        for r in &removed {
            let k = out_pos.partition_point(|&p| p < r.position);
            // removed beat strictly inside interval k-1 (between beats k-1 and k)
            if k >= 1 && k < n && out_pos[k] > r.position {
                nn_valid[k - 1] = false;
            }
        }
    }

    // input view
    let mut input_to_output = vec![None; n_in];
    for (j, b) in beats.iter().enumerate() {
        if let Some(i) = b.input_index {
            input_to_output[i] = Some(j);
        }
    }
    let mut input_classes = vec![BeatClass::Normal; n_in];
    for r in &removed {
        input_classes[r.input_index] = match r.reason {
            RemovalReason::Extra => BeatClass::Extra,
            RemovalReason::PhysiologicalLimit => BeatClass::PhysiologicalLimit,
        };
    }
    for (i, o) in input_to_output.iter().enumerate() {
        let Some(j) = *o else { continue };
        let b = &beats[j];
        let missed_after = j + 1 < n && beats[j + 1].inserted;
        input_classes[i] = if b.is_ectopic() {
            BeatClass::Ectopic
        } else if b.is_suspect() {
            BeatClass::Suspect
        } else if missed_after {
            BeatClass::Missed
        } else if b.disagreement {
            BeatClass::Disagreement
        } else {
            BeatClass::Normal
        };
    }

    Ok(BeatExclusion {
        positions: out_pos,
        beats,
        input_classes,
        input_to_output,
        removed,
        splits,
        nn_valid,
        nn_usable,
        sampling_rate: fs,
        params: *params,
    })
}

/// Steps 1–3 on one run of validated positions. Input indices in the result are run-local.
fn classify_run(
    input: &[f64],
    fs: f64,
    params: &ExclusionParams,
    disagreement: Option<&[bool]>,
    mid: fn(f64, f64) -> f64,
) -> RunOut {
    let kp = &params.kubios;
    // 1. artifact passes (insert missed / remove extra only)
    let (pos, _, _, corr) = fix_core(input, fs, kp, mid, true);
    let origin = corr.output_to_input();
    let removed: Vec<usize> = (0..input.len())
        .filter(|&i| corr.input_to_output[i].is_none())
        .collect();
    let n = pos.len();
    let mut beats: Vec<BeatFlags> = origin.iter().map(|&o| BeatFlags::new(o)).collect();
    if let Some(d) = disagreement {
        for b in &mut beats {
            b.disagreement = b.input_index.is_some_and(|i| d[i]);
        }
    }
    if n < 3 {
        return RunOut {
            positions: pos,
            beats,
            removed,
        };
    }

    // 2. one detection pass
    let (a, medrr) = detect_detail(&pos, fs, kp);
    // rr[k] = pos[k] - pos[k-1] (samples), rr[0] unused; m in samples
    let rr = |k: usize| pos[k] - pos[k - 1];
    let m = |k: usize| medrr[k] * fs;
    let swing = |b: usize| (rr(b + 1) - rr(b)).abs();
    let inner = |b: usize| b >= 1 && b + 1 < n;
    for &j in &a.ectopic {
        let lo = j.checked_sub(1).filter(|&b| inner(b));
        let hi = Some(j).filter(|&b| inner(b));
        let pick = match (lo, hi) {
            (Some(l), Some(h)) => Some(if swing(h) > swing(l) { h } else { l }),
            (l, h) => l.or(h),
        };
        if let Some(b) = pick {
            beats[b].ectopic_kubios = true;
        }
    }
    for b in 1..n - 1 {
        let (r0, r1, mb) = (rr(b), rr(b + 1), m(b));
        beats[b].ectopic_pattern =
            r0 <= params.premature_factor * mb && r1 - mb >= params.compensatory_factor * (mb - r0);
    }
    let ev: Vec<usize> = (0..n).filter(|&b| beats[b].is_ectopic()).collect();
    let mut absorbed = vec![false; n + 1];
    for &b in &ev {
        absorbed[b] = true;
        absorbed[b + 1] = true;
    }
    // required midpoint move (seconds); +inf at the edges
    let move_s = |p: &[f64], j: usize| {
        if inner(j) {
            (p[j] - (p[j - 1] + p[j + 1]) / 2.0).abs() / fs
        } else {
            f64::INFINITY
        }
    };
    for &j in a.longshort.iter().chain(&a.missed).chain(&a.extra) {
        if j >= n || absorbed[j] || beats[j].is_suspect() {
            continue;
        }
        beats[j].suspect_stage = 1;
        beats[j].suspect_move_s = move_s(&pos, j);
    }
    for &j in a.missed.iter().chain(&a.extra) {
        if j < n && beats[j].suspect_stage == 1 {
            beats[j].suspect_move_s = f64::INFINITY;
        }
    }

    // 3. two-stage: detection on a copy with the ectopic beats at their neighbours' midpoint
    if params.two_stage && !ev.is_empty() {
        let mut cp = pos.clone();
        for &b in &ev {
            if inner(b) {
                cp[b] = (cp[b - 1] + cp[b + 1]) / 2.0;
            }
        }
        let a2 = detect_artifacts_positions(&cp, fs, kp);
        let mut near = vec![false; n + 1];
        for &b in &ev {
            near[b] = true;
            near[b + 1] = true;
            if b >= 1 {
                near[b - 1] = true;
            }
        }
        let kinds: [(&[usize], bool); 4] = [
            (&a2.ectopic, false),
            (&a2.missed, true),
            (&a2.extra, true),
            (&a2.longshort, false),
        ];
        for (list, residual) in kinds {
            for &j in list {
                if j >= n || near[j] || beats[j].is_suspect() {
                    continue;
                }
                beats[j].suspect_stage = 2;
                beats[j].suspect_move_s = if residual {
                    f64::INFINITY
                } else {
                    move_s(&cp, j)
                };
            }
        }
    }
    RunOut {
        positions: pos,
        beats,
        removed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn regular(n: usize, step: f64) -> Vec<f64> {
        (0..n)
            .map(|i| 100.0 + i as f64 * step + (i as f64 * 0.7).sin() * 4.0)
            .collect()
    }

    #[test]
    fn clean_series_is_all_normal_and_valid() {
        let pos = regular(200, 410.0);
        let out = classify_beats(&pos, 512.0, &ExclusionParams::default(), None).unwrap();
        assert_eq!(out.positions, pos);
        assert!(out.beats.iter().all(|b| b.class == BeatClass::Normal));
        assert!(out.input_classes.iter().all(|&c| c == BeatClass::Normal));
        assert_eq!(out.kept_count(), 199);
        assert_eq!(out.usable_count(), 199);
        assert_eq!(out.kept_share(), 1.0);
        assert_eq!(out.artifact_counts().total(), 0);
        assert_eq!(out.artifact_ratio(), 0.0);
        assert_eq!(out.ectopy_ratio(), 0.0);
    }

    #[test]
    fn premature_beat_is_ectopic_and_not_moved() {
        let mut pos = regular(200, 410.0);
        // premature by 30 % of the RR with full compensation
        pos[100] -= 0.3 * 410.0;
        let out = classify_beats(&pos, 512.0, &ExclusionParams::default(), None).unwrap();
        assert_eq!(out.positions, pos, "no beat moves");
        assert_eq!(out.input_classes[100], BeatClass::Ectopic);
        assert!(out.beats[100].ectopic_pattern);
        assert!(!out.nn_valid[99] && !out.nn_valid[100]);
        assert_eq!(out.kept_count(), 197);
        assert_eq!(
            out.artifact_counts().total(),
            0,
            "ectopy is not an artifact"
        );
        assert_eq!(out.ectopic_count(), 1);
    }

    #[test]
    fn missed_and_extra_beats() {
        let mut pos = regular(200, 410.0);
        pos.remove(60); // missed
        let extra = (pos[150] + pos[151]) / 2.0 + 30.0;
        pos.insert(151, extra); // extra
        let out = classify_beats(&pos, 512.0, &ExclusionParams::default(), None).unwrap();
        assert_eq!(out.input_classes[59], BeatClass::Missed);
        assert_eq!(out.input_classes[151], BeatClass::Extra);
        let c = out.artifact_counts();
        assert_eq!((c.inserted, c.removed), (1, 1));
        let ins = out.beats.iter().position(|b| b.inserted).unwrap();
        assert_eq!(out.beats[ins].class, BeatClass::Inserted);
        assert!(!out.nn_valid[ins - 1] && !out.nn_valid[ins]);
        // the interval spanning the removed beat is excluded
        let k = out.input_to_output[150].unwrap();
        assert_eq!(out.input_to_output[152], Some(k + 1));
        assert!(!out.nn_valid[k]);
        // without exclude_inserted_removed they stay valid
        let p = ExclusionParams {
            exclude_inserted_removed: false,
            ..Default::default()
        };
        let out2 = classify_beats(&pos, 512.0, &p, None).unwrap();
        assert!(out2.nn_valid[k] && out2.nn_valid[ins - 1]);
    }

    #[test]
    fn integer_peaks_insert_integer_positions() {
        let mut peaks: Vec<usize> = regular(200, 411.0)
            .iter()
            .map(|p| p.round() as usize)
            .collect();
        peaks.remove(80);
        let out = classify_peaks(&peaks, 512.0, &ExclusionParams::default(), None).unwrap();
        assert!(out.positions.iter().all(|p| p.fract() == 0.0));
        assert_eq!(out.artifact_counts().inserted, 1);
    }

    #[test]
    fn disagreement_and_physiological_limits() {
        let mut pos = regular(200, 410.0);
        pos.insert(51, pos[50] + 50.0); // 98 ms after beat 50: below 300 ms
        let gap_at = 120;
        for p in &mut pos[gap_at..] {
            *p += 3.0 * 512.0; // 3 s pause: above 2 s
        }
        let mut dis = vec![false; pos.len()];
        dis[30] = true;
        let p = ExclusionParams {
            physiological_rr_s: Some((0.3, 2.0)),
            ..Default::default()
        };
        let out = classify_beats(&pos, 512.0, &p, Some(&dis)).unwrap();
        assert_eq!(out.input_classes[51], BeatClass::PhysiologicalLimit);
        assert_eq!(out.input_classes[30], BeatClass::Disagreement);
        assert_eq!(out.splits.len(), 1);
        let k = out.splits[0];
        assert!(!out.nn_usable[k] && !out.nn_valid[k]);
        let c = out.artifact_counts();
        assert_eq!((c.physiological, c.disagreement), (2, 1), "{c:?}");
        assert_eq!(out.usable_count(), out.positions.len() - 2);
        // window summary covering everything equals the totals
        let s = out.summary_between(f64::NEG_INFINITY, f64::INFINITY);
        assert_eq!(s.artifacts, c);
        assert_eq!(
            (s.kept_nn, s.usable_nn),
            (out.kept_count(), out.usable_count())
        );
    }

    #[test]
    fn invalid_inputs() {
        let p = ExclusionParams::default();
        assert!(classify_beats(&[1.0, 0.5], 512.0, &p, None).is_err());
        assert!(classify_beats(&[1.0, f64::NAN], 512.0, &p, None).is_err());
        assert!(classify_beats(&[1.0, 2.0], 0.0, &p, None).is_err());
        assert!(classify_beats(&[1.0, 2.0], 512.0, &p, Some(&[false])).is_err());
        let bad = ExclusionParams {
            physiological_rr_s: Some((2.0, 0.3)),
            ..Default::default()
        };
        assert!(classify_beats(&[1.0, 2.0], 512.0, &bad, None).is_err());
        let out = classify_beats(&[1.0, 2.0], 512.0, &p, None).unwrap();
        assert_eq!(out.kept_count(), 1);
        let empty = classify_beats(&[], 512.0, &p, None).unwrap();
        assert!(empty.kept_share().is_nan() && empty.artifact_ratio().is_nan());
    }
}
