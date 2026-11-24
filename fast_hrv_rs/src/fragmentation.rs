//! Heart rate fragmentation (Costa, Davis & Goldberger 2017, "Heart Rate Fragmentation: A New
//! Approach to the Analysis of Cardiac Interbeat Interval Dynamics", Front Physiol 8:255).
//!
//! The indices describe how often the sign of the successive RR differences changes (a sinus rhythm
//! that accelerates and decelerates in short alternating runs is "fragmented"). Definitions follow
//! NeuroKit2 0.3 (`hrv_nonlinear`, `_hrv_nonlinear_fragmentation`) so the values can be cross-checked:
//!
//! * `pip`  — inflection points (sign changes of ΔRR, zeros counted as a sign) / (number of ΔRR + 1).
//! * `ials` — 1 / mean length (in ΔRR) of the acceleration and deceleration segments (as in
//!   NeuroKit2, an empty acceleration or deceleration list counts as one run of length 0).
//! * `pss`  — share of ΔRR in segments shorter than 3.
//! * `pas`  — share of the alternation runs (consecutive inflection points) with length ≥ 4.
//!   NeuroKit2 counts runs here, while Costa (2017) counts NN intervals; we follow NeuroKit2.
//!
//! Ectopic beats and artifacts inflate every index: use clean NN (exclusion mode or corrected).

/// Heart rate fragmentation indices (0–1; NaN with fewer than 3 intervals).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Fragmentation {
    pub pip: f64,
    pub ials: f64,
    pub pss: f64,
    pub pas: f64,
}

impl Fragmentation {
    pub fn nan() -> Self {
        let n = f64::NAN;
        Self {
            pip: n,
            ials: n,
            pss: n,
            pas: n,
        }
    }
}

/// Runs of consecutive integers (NumPy `np.split(x, np.where(np.diff(x) != 1)[0] + 1)`): an empty
/// input gives one empty run, as in NeuroKit2.
fn consecutive_runs(x: &[usize]) -> Vec<usize> {
    if x.is_empty() {
        return vec![0];
    }
    let mut runs = Vec::new();
    let mut len = 1;
    for w in x.windows(2) {
        if w[1] == w[0] + 1 {
            len += 1;
        } else {
            runs.push(len);
            len = 1;
        }
    }
    runs.push(len);
    runs
}

/// Fragmentation indices of an NN series (ms) without gaps.
pub fn fragmentation(nn_ms: &[f64]) -> Fragmentation {
    if nn_ms.len() < 3 || nn_ms.iter().any(|x| !x.is_finite()) {
        return Fragmentation::nan();
    }
    let d: Vec<f64> = nn_ms.windows(2).map(|w| w[1] - w[0]).collect();
    // NeuroKit2 signal_zerocrossings: indices where sign(d) changes (0 is its own sign)
    let sign = |x: f64| -> i8 {
        if x > 0.0 {
            1
        } else if x < 0.0 {
            -1
        } else {
            0
        }
    };
    let zc: Vec<usize> = (0..d.len().saturating_sub(1))
        .filter(|&i| sign(d[i + 1]) != sign(d[i]))
        .collect();
    let n = (d.len() + 1) as f64;
    let pip = zc.len() as f64 / n;
    let acc: Vec<usize> = (0..d.len()).filter(|&i| d[i] > 0.0).collect();
    let dec: Vec<usize> = (0..d.len()).filter(|&i| d[i] < 0.0).collect();
    let mut lens = consecutive_runs(&acc);
    lens.extend(consecutive_runs(&dec));
    let total: usize = lens.iter().sum();
    let mean_len = total as f64 / lens.len() as f64;
    let ials = 1.0 / mean_len;
    let pss = lens.iter().filter(|&&l| l < 3).sum::<usize>() as f64 / total as f64;
    let alt = consecutive_runs(&zc);
    let pas = alt.iter().filter(|&&l| l >= 4).count() as f64 / alt.len() as f64;
    Fragmentation {
        pip,
        ials,
        pss,
        pas,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alternating_series_is_fully_fragmented() {
        let nn: Vec<f64> = (0..20)
            .map(|i| if i % 2 == 0 { 800.0 } else { 840.0 })
            .collect();
        let f = fragmentation(&nn);
        assert!((f.pip - 18.0 / 20.0).abs() < 1e-12);
        assert!((f.ials - 1.0).abs() < 1e-12);
        assert!((f.pss - 1.0).abs() < 1e-12);
        assert!((f.pas - 1.0).abs() < 1e-12);
    }

    #[test]
    fn smooth_series_is_not_fragmented() {
        let nn: Vec<f64> = (0..20).map(|i| 800.0 + 10.0 * i as f64).collect();
        let f = fragmentation(&nn);
        assert_eq!(f.pip, 0.0);
        // NeuroKit2 counts an empty deceleration list as one run of length 0: mean (19 + 0) / 2
        assert!((f.ials - 1.0 / 9.5).abs() < 1e-12);
        assert_eq!(f.pss, 0.0);
        assert!(fragmentation(&[800.0, 810.0]).pip.is_nan());
    }
}
