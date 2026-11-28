//! Named analysis presets and the provenance record that travels with every result.
//!
//! The same RR series gives different frequency-domain values depending on resampling, window,
//! detrending and band integration (LF/HF can differ by several percent, more on short records).
//! A preset names one complete set of these choices so that results are reproducible and comparable
//! across tools, and [`Provenance`] records which one was used (the methods section of a report, or a
//! database column next to the values).
//!
//! | Preset | Detrending | Frequency domain | Reference |
//! |---|---|---|---|
//! | [`Preset::TaskForce`] (default) | none | 4 Hz cubic spline, Welch 64 s, exact band integration | Task Force 1996 |
//! | [`Preset::KubiosLike`] | smoothness priors, λ = 500 | 4 Hz cubic spline, Welch 300 s | Kubios HRV defaults as published; λ not checked against a Kubios export |
//! | [`Preset::NeuroKit`] | none | `nk.hrv_frequency` defaults ([`crate::frequency_domain_neurokit`]) | NeuroKit2 |
//!
//! Time-domain and nonlinear metrics are computed on the (detrended, where the preset says so) RR
//! series the same way in every preset.

use crate::detrend::smoothness_priors;
use crate::frequency::{FrequencyOptions, frequency_domain_with};
use crate::frequency_nk::frequency_domain_neurokit;
use crate::report::{HrvReport, hrv};

/// A named set of analysis choices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Preset {
    /// fast-hrv's default (Task Force 1996 practice).
    #[default]
    TaskForce,
    /// Kubios HRV's published defaults: smoothness-priors detrending (λ = 500), Welch 300 s.
    KubiosLike,
    /// NeuroKit2 `hrv()` defaults (frequency domain as `nk.hrv_frequency`).
    NeuroKit,
}

impl Preset {
    pub fn name(self) -> &'static str {
        match self {
            Preset::TaskForce => "task_force",
            Preset::KubiosLike => "kubios_like",
            Preset::NeuroKit => "neurokit",
        }
    }

    /// Smoothness-priors λ applied to the RR series before analysis (`None`: no detrending).
    pub fn detrend_lambda(self) -> Option<f64> {
        match self {
            Preset::KubiosLike => Some(500.0),
            _ => None,
        }
    }

    pub fn frequency_method(self) -> &'static str {
        match self {
            Preset::TaskForce => "welch_64s_cubic_4hz",
            Preset::KubiosLike => "welch_300s_cubic_4hz",
            Preset::NeuroKit => "neurokit_welch_half_series_quadratic_100hz",
        }
    }
}

/// How a result was computed. Serialisable (feature `serde`) so it can be stored next to the values.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Provenance {
    pub library: &'static str,
    pub version: &'static str,
    pub preset: &'static str,
    pub detrend_lambda: Option<f64>,
    pub frequency_method: &'static str,
    /// Intervals and their total length (s) that went into the result.
    pub n_intervals: usize,
    pub duration_s: f64,
}

/// [`hrv`] with a preset: detrending (if any) before the time, frequency and nonlinear metrics, the
/// preset's frequency method, and the provenance record.
pub fn hrv_with_preset(rri_ms: &[f64], preset: Preset) -> (HrvReport, Provenance) {
    let series: Vec<f64> = match preset.detrend_lambda() {
        // detrend the variation but keep the mean level (MeanNN, HR stay in ms / bpm)
        Some(lambda) => match smoothness_priors(rri_ms, lambda) {
            Ok(d) => {
                let mean = rri_ms.iter().sum::<f64>() / rri_ms.len().max(1) as f64;
                d.iter().map(|v| v + mean).collect()
            }
            Err(_) => rri_ms.to_vec(),
        },
        None => rri_ms.to_vec(),
    };
    let mut report = hrv(&series);
    // MeanNN / HR from the original series (detrending must not move the level)
    let level = hrv(rri_ms);
    report.time.mean_nn = level.time.mean_nn;
    report.time.mean_hr = level.time.mean_hr;
    report.frequency = match preset {
        Preset::TaskForce => report.frequency,
        Preset::KubiosLike => frequency_domain_with(
            &series,
            &FrequencyOptions {
                welch_segment_s: 300.0,
                ..FrequencyOptions::default()
            },
        ),
        Preset::NeuroKit => frequency_domain_neurokit(rri_ms),
    };
    let prov = Provenance {
        library: "fast-hrv",
        version: env!("CARGO_PKG_VERSION"),
        preset: preset.name(),
        detrend_lambda: preset.detrend_lambda(),
        frequency_method: preset.frequency_method(),
        n_intervals: rri_ms.len(),
        duration_s: rri_ms.iter().filter(|v| v.is_finite()).sum::<f64>() / 1000.0,
    };
    (report, prov)
}
