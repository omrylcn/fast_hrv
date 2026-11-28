//! Per-window decision: which metrics of an [`HrvReport`] may be shown, and why not.
//!
//! An HRV value is only as good as the window it comes from. [`window_report`] checks every metric
//! of the [`crate::METRICS`] catalogue against the window and returns a [`Verdict`] with reason codes,
//! so that applications do not re-implement the rules (and computed values stay separate from what
//! may be displayed):
//!
//! | Reason | Verdict | Rule |
//! |---|---|---|
//! | `not_computed` | suppressed | the value is NaN (too few intervals, band below its technical minimum…) |
//! | `too_short` | suppressed | window shorter than the metric's `min_s` |
//! | `too_many_corrections` | suppressed | share of corrected / excluded beats above `max_artifact_ratio()` |
//! | `breathing_in_lf_band` | suppressed (LF/HF, LFnu, HFnu) / warning (LF, HF) | breathing < 9 /min (0.15 Hz): the respiratory peak moves into LF |
//! | `breathing_above_hf_band` | warning (HF) | breathing > 24 /min (0.4 Hz): part of the respiratory power leaves HF |
//! | `below_strict_length` | warning | between `min_s` and `strict_min_s()` (no validated agreement with 5 min) |
//! | `consecutive_corrections` | warning (beat-to-beat metrics) | two or more corrected beats in a row |
//!
//! Sources: metric lengths and artifact limits in [`crate::metrics`]; respiration bands after
//! Quigley et al. 2024 (Psychophysiology 61:e14604) and the Task Force 1996 band limits; one ectopic
//! beat in a 60 s window is already 1–1.7 % of the beats (Choi & Shin 2018 found RMSSD biased from
//! 0.1 % uncorrected artifacts).

use crate::metrics::{METRICS, MetricInfo};
use crate::report::{HrvReport, hrv};

/// Decision for one metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum Verdict {
    /// May be shown and compared.
    Valid,
    /// May be shown, with its reasons.
    Warning,
    /// Not to be shown.
    Suppressed,
}

/// One metric of the window.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct MetricVerdict {
    pub name: &'static str,
    pub value: f64,
    pub verdict: Verdict,
    /// Reason codes (see the module documentation); empty when valid.
    pub reasons: Vec<&'static str>,
}

/// What the report needs to know about the window besides the metric values.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WindowContext {
    /// Length of the window (s); for an NN series, the sum of the intervals.
    pub duration_s: f64,
    /// Beats in the window.
    pub n_beats: usize,
    /// Beats that were corrected (Kubios) or excluded (exclusion mode).
    pub n_corrected: usize,
    /// Longest run of consecutive corrected / excluded beats.
    pub longest_corrected_run: usize,
    /// Breathing rate (breaths/min), e.g. from [`crate::respiratory_rate`]; `None` if unknown.
    pub respiration_bpm: Option<f64>,
}

impl WindowContext {
    /// From the intervals (ms) and a per-interval flag of correction / exclusion.
    pub fn from_flags(rri_ms: &[f64], corrected: &[bool], respiration_bpm: Option<f64>) -> Self {
        let (mut run, mut longest) = (0usize, 0usize);
        for &c in corrected {
            run = if c { run + 1 } else { 0 };
            longest = longest.max(run);
        }
        Self {
            duration_s: rri_ms.iter().filter(|v| v.is_finite()).sum::<f64>() / 1000.0,
            n_beats: rri_ms.len(),
            n_corrected: corrected.iter().filter(|&&c| c).count(),
            longest_corrected_run: longest,
            respiration_bpm,
        }
    }

    /// Share of corrected / excluded beats (0..1).
    pub fn artifact_ratio(&self) -> f64 {
        if self.n_beats == 0 {
            0.0
        } else {
            self.n_corrected as f64 / self.n_beats as f64
        }
    }
}

/// The decision for every metric of one window.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct WindowReport {
    pub context: WindowContext,
    pub artifact_ratio: f64,
    pub metrics: Vec<MetricVerdict>,
}

impl WindowReport {
    pub fn get(&self, name: &str) -> Option<&MetricVerdict> {
        self.metrics.iter().find(|m| m.name == name)
    }

    /// The value if it may be shown (valid or warning), else `None`.
    pub fn shown(&self, name: &str) -> Option<f64> {
        self.get(name)
            .filter(|m| m.verdict != Verdict::Suppressed)
            .map(|m| m.value)
    }
}

/// Metrics built from successive differences: a run of corrected beats replaces several differences.
const BEAT_TO_BEAT: &[&str] = &[
    "rmssd", "ln_rmssd", "sdsd", "cvsd", "pnn50", "pnn20", "sd1", "sdrmssd",
];
/// Ratios whose meaning breaks when breathing is in LF.
const LF_HF_RATIOS: &[&str] = &["lf_hf", "lfnu", "hfnu"];
const LF_FAMILY: &[&str] = &["lf", "lf_peak"];
const HF_FAMILY: &[&str] = &["hf", "lnhf", "hf_peak"];

fn decide(info: &MetricInfo, value: f64, ctx: &WindowContext) -> MetricVerdict {
    let mut reasons = Vec::new();
    let mut verdict = Verdict::Valid;
    let mut raise = |v: Verdict, r: &'static str, reasons: &mut Vec<&'static str>| {
        verdict = verdict.max(v);
        reasons.push(r);
    };
    let name = info.name;
    if !value.is_finite() {
        raise(Verdict::Suppressed, "not_computed", &mut reasons);
    }
    if ctx.duration_s < info.min_s {
        raise(Verdict::Suppressed, "too_short", &mut reasons);
    } else if ctx.duration_s < info.strict_min_s() {
        raise(Verdict::Warning, "below_strict_length", &mut reasons);
    }
    if ctx.artifact_ratio() > info.max_artifact_ratio() {
        raise(Verdict::Suppressed, "too_many_corrections", &mut reasons);
    }
    if ctx.longest_corrected_run >= 2 && BEAT_TO_BEAT.contains(&name) {
        raise(Verdict::Warning, "consecutive_corrections", &mut reasons);
    }
    if let Some(bpm) = ctx.respiration_bpm.filter(|b| b.is_finite()) {
        if bpm < 9.0 {
            if LF_HF_RATIOS.contains(&name) {
                raise(Verdict::Suppressed, "breathing_in_lf_band", &mut reasons);
            } else if LF_FAMILY.contains(&name) || HF_FAMILY.contains(&name) {
                raise(Verdict::Warning, "breathing_in_lf_band", &mut reasons);
            }
        } else if bpm > 24.0 && (HF_FAMILY.contains(&name) || LF_HF_RATIOS.contains(&name)) {
            raise(Verdict::Warning, "breathing_above_hf_band", &mut reasons);
        }
    }
    MetricVerdict {
        name,
        value,
        verdict,
        reasons,
    }
}

/// Decisions for every metric of `report` in the window described by `ctx`.
pub fn window_report(report: &HrvReport, ctx: &WindowContext) -> WindowReport {
    WindowReport {
        context: *ctx,
        artifact_ratio: ctx.artifact_ratio(),
        metrics: METRICS
            .iter()
            .map(|m| decide(m, report.value(m.name).unwrap_or(f64::NAN), ctx))
            .collect(),
    }
}

/// [`hrv`] on the intervals (ms) plus [`window_report`]; `corrected[i]` marks interval `i` as
/// corrected or excluded (same length as `rri_ms`, or empty for none).
pub fn window_report_rri(
    rri_ms: &[f64],
    corrected: &[bool],
    respiration_bpm: Option<f64>,
) -> WindowReport {
    let ctx = WindowContext::from_flags(rri_ms, corrected, respiration_bpm);
    window_report(&hrv(rri_ms), &ctx)
}
