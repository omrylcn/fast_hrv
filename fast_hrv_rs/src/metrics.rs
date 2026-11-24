//! Shortest and recommended recording length per metric, so that applications read the window rules
//! from one place instead of copying them into their own settings.
//!
//! `min_s`: below this the metric is not estimated reliably at rest (clean, ectopy-free NN).
//! `recommended_s`: the length to use when values are compared or reported. Values computed on
//! different lengths must not be compared (SDNN and the frequency bands grow with the length).
//! `basis` names the source of the numbers; "convention" means no direct validation study.
//!
//! Main sources: Shaffer, Meehan & Zerr 2020 (Front Neurosci 14:594880, table 4: limits of agreement
//! against 5 min, ECG); Munoz et al. 2015 (PLoS ONE 10:e0138921); Task Force 1996 (Circulation
//! 93:1043); Quigley et al. 2024 (Psychophysiology 61:e14604, recording ≥ 10 cycles of the band's
//! lowest frequency); Rogers et al. 2021 (DFA α1 in 2 min windows during exercise); Costa et al. 2017
//! (fragmentation, mostly long recordings).
//!
//! These are lengths for **resting** data; during exercise only HR and DFA α1 are meaningful.

/// Window rule of one metric.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct MetricInfo {
    /// Field name as in [`crate::HrvReport`] (`time.rmssd` → `"rmssd"`).
    pub name: &'static str,
    /// `"time"` | `"frequency"` | `"nonlinear"` | `"asymmetry"` | `"fragmentation"`.
    pub domain: &'static str,
    pub unit: &'static str,
    pub min_s: f64,
    pub recommended_s: f64,
    pub basis: &'static str,
}

const fn m(
    name: &'static str,
    domain: &'static str,
    unit: &'static str,
    min_s: f64,
    recommended_s: f64,
    basis: &'static str,
) -> MetricInfo {
    MetricInfo {
        name,
        domain,
        unit,
        min_s,
        recommended_s,
        basis,
    }
}

/// All metrics of [`crate::HrvReport`].
pub const METRICS: &[MetricInfo] = &[
    m(
        "mean_nn",
        "time",
        "ms",
        10.0,
        60.0,
        "Shaffer 2020; Munoz 2015",
    ),
    m(
        "mean_hr",
        "time",
        "bpm",
        10.0,
        60.0,
        "Shaffer 2020; Munoz 2015",
    ),
    m("sd_hr", "time", "bpm", 60.0, 120.0, "convention"),
    m(
        "rmssd",
        "time",
        "ms",
        30.0,
        60.0,
        "Munoz 2015 (10 s); Shaffer 2020 (60 s)",
    ),
    m("ln_rmssd", "time", "ln(ms)", 30.0, 60.0, "as rmssd"),
    m("sdsd", "time", "ms", 30.0, 60.0, "as rmssd"),
    m("cvsd", "time", "", 30.0, 60.0, "as rmssd"),
    m(
        "sdnn",
        "time",
        "ms",
        60.0,
        300.0,
        "Shaffer 2020 (60 s); Task Force 1996 (5 min standard); length-dependent",
    ),
    m("cvnn", "time", "", 60.0, 300.0, "as sdnn"),
    m("pnn50", "time", "%", 60.0, 120.0, "Shaffer 2020"),
    m("pnn20", "time", "%", 60.0, 120.0, "convention (as pnn50)"),
    m("median_nn", "time", "ms", 60.0, 120.0, "convention"),
    m("mad_nn", "time", "ms", 60.0, 120.0, "convention"),
    m("iqr_nn", "time", "ms", 60.0, 120.0, "convention"),
    m("min_nn", "time", "ms", 60.0, 120.0, "convention"),
    m("max_nn", "time", "ms", 60.0, 120.0, "convention"),
    m(
        "hti",
        "time",
        "",
        300.0,
        1200.0,
        "Task Force 1996 (geometric methods: ≥ 20 min)",
    ),
    m(
        "tinn",
        "time",
        "ms",
        300.0,
        1200.0,
        "Task Force 1996 (geometric methods: ≥ 20 min)",
    ),
    m(
        "hf",
        "frequency",
        "ms²",
        60.0,
        300.0,
        "Quigley 2024 (≥ 1 min); Shaffer 2020 (180 s)",
    ),
    m("lnhf", "frequency", "ln(ms²)", 60.0, 300.0, "as hf"),
    m(
        "lf",
        "frequency",
        "ms²",
        120.0,
        300.0,
        "Quigley 2024 (≥ 2 min); Task Force 1996",
    ),
    m(
        "lf_hf",
        "frequency",
        "",
        180.0,
        300.0,
        "Shaffer 2020 (180 s); not a sympathovagal index (Billman 2013; Quigley 2024)",
    ),
    m("lfnu", "frequency", "%", 180.0, 300.0, "as lf_hf"),
    m("hfnu", "frequency", "%", 180.0, 300.0, "as lf_hf"),
    m("lf_peak", "frequency", "Hz", 120.0, 300.0, "as lf"),
    m("hf_peak", "frequency", "Hz", 60.0, 300.0, "as hf"),
    m(
        "vlf",
        "frequency",
        "ms²",
        300.0,
        86400.0,
        "Task Force 1996 (≥ 5 min; best 24 h)",
    ),
    m(
        "total_power",
        "frequency",
        "ms²",
        300.0,
        300.0,
        "Task Force 1996",
    ),
    m("sd1", "nonlinear", "ms", 30.0, 60.0, "= RMSSD / √2"),
    m("sd2", "nonlinear", "ms", 90.0, 120.0, "Shaffer 2020"),
    m("sd1_sd2", "nonlinear", "", 90.0, 120.0, "as sd2"),
    m("csi", "nonlinear", "", 90.0, 120.0, "as sd2"),
    m("cvi", "nonlinear", "", 90.0, 120.0, "as sd2"),
    m(
        "sampen",
        "nonlinear",
        "",
        180.0,
        300.0,
        "Shaffer 2020 (180 s)",
    ),
    m(
        "apen",
        "nonlinear",
        "",
        180.0,
        300.0,
        "convention (as sampen)",
    ),
    m(
        "dfa_alpha1",
        "nonlinear",
        "",
        120.0,
        120.0,
        "Shaffer 2020 (rest); Rogers 2021 (2 min windows in exercise)",
    ),
    m(
        "dfa_alpha2",
        "nonlinear",
        "",
        300.0,
        1200.0,
        "convention (scales 16–64 beats)",
    ),
    m("gi", "asymmetry", "%", 120.0, 300.0, "convention"),
    m("pi", "asymmetry", "%", 120.0, 300.0, "convention"),
    m("sd1d", "asymmetry", "ms", 120.0, 300.0, "convention"),
    m("sd1a", "asymmetry", "ms", 120.0, 300.0, "convention"),
    m("c1d", "asymmetry", "", 120.0, 300.0, "convention"),
    m("c1a", "asymmetry", "", 120.0, 300.0, "convention"),
    m(
        "pip",
        "fragmentation",
        "",
        300.0,
        300.0,
        "Costa 2017 (validated mostly on long recordings)",
    ),
    m("ials", "fragmentation", "", 300.0, 300.0, "as pip"),
    m("pss", "fragmentation", "", 300.0, 300.0, "as pip"),
    m("pas", "fragmentation", "", 300.0, 300.0, "as pip"),
];

/// Window rule of a metric by field name (`None` if unknown).
pub fn metric_info(name: &str) -> Option<&'static MetricInfo> {
    METRICS.iter().find(|m| m.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_consistent() {
        let mut names: Vec<&str> = METRICS.iter().map(|m| m.name).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), METRICS.len(), "duplicate name");
        for m in METRICS {
            assert!(m.min_s > 0.0 && m.recommended_s >= m.min_s, "{m:?}");
        }
        assert_eq!(metric_info("lf_hf").unwrap().min_s, 180.0);
        assert!(metric_info("nope").is_none());
    }
}
