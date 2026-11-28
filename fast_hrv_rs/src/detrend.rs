//! Smoothness-priors detrending (Tarvainen, Ranta-Aho & Karjalainen 2002, IEEE TBME 49:172).
//!
//! The trend `z` of a series `x` (e.g. the RR series, one value per beat) is the solution of
//! `(I + λ²·D₂ᵀD₂) z = x`, where `D₂` is the second-order difference matrix; the detrended series is
//! `x − z`. It acts as a time-varying high-pass filter: larger `λ` keeps more of the slow variation
//! (lower cut-off). Kubios applies it to the RR series before frequency and nonlinear analysis and
//! before the stress index; NeuroKit2 offers it as `signal_detrend(method="tarvainen2002")`.
//!
//! The matrix is symmetric, positive definite and pentadiagonal, so the system is solved by a banded
//! LDLᵀ factorisation in O(N) time and memory (no dense or general sparse solver).
//!
//! Cut-off: for a series sampled at a constant rate `fs`, the frequency response of the trend filter
//! is `1 / (1 + λ²·(2 − 2·cos ω)²)`; [`smoothness_priors_cutoff_hz`] gives its −3 dB point.
//! For an RR series (one sample per beat) use the mean heart rate as `fs`.

use crate::error::HrvError;

/// Detrended series `x − trend` (same length). `lambda > 0`; series with fewer than 3 values are
/// returned with only their mean removed.
pub fn smoothness_priors(x: &[f64], lambda: f64) -> Result<Vec<f64>, HrvError> {
    let trend = smoothness_priors_trend(x, lambda)?;
    Ok(x.iter().zip(&trend).map(|(a, b)| a - b).collect())
}

/// The trend `z` of [`smoothness_priors`].
pub fn smoothness_priors_trend(x: &[f64], lambda: f64) -> Result<Vec<f64>, HrvError> {
    if !(lambda.is_finite() && lambda > 0.0) {
        return Err(HrvError::InvalidParameter("lambda"));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(HrvError::NonFinite);
    }
    let n = x.len();
    if n < 3 {
        let mean = if n == 0 {
            0.0
        } else {
            x.iter().sum::<f64>() / n as f64
        };
        return Ok(vec![mean; n]);
    }
    // bands of A = I + λ²·D₂ᵀD₂ (symmetric): d0 (main), d1 (first off), d2 (second off)
    let l2 = lambda * lambda;
    let mut d0 = vec![1.0; n];
    let mut d1 = vec![0.0; n - 1];
    let mut d2 = vec![0.0; n - 2];
    // row r of D₂ is [1, −2, 1] at columns r, r+1, r+2; add its outer product
    let c = [1.0, -2.0, 1.0];
    for r in 0..n - 2 {
        for a in 0..3 {
            d0[r + a] += l2 * c[a] * c[a];
            if a < 2 {
                d1[r + a] += l2 * c[a] * c[a + 1];
            }
        }
        d2[r] += l2 * c[0] * c[2];
    }
    // banded LDLᵀ: L unit lower with two sub-diagonals (l1, l2b), D diagonal
    let mut dd = vec![0.0; n];
    let mut l1 = vec![0.0; n]; // L[i][i-1]
    let mut lb = vec![0.0; n]; // L[i][i-2]
    for i in 0..n {
        if i >= 2 {
            lb[i] = d2[i - 2] / dd[i - 2];
        }
        if i >= 1 {
            let mut v = d1[i - 1];
            if i >= 2 {
                v -= lb[i] * l1[i - 1] * dd[i - 2];
            }
            l1[i] = v / dd[i - 1];
        }
        let mut v = d0[i];
        if i >= 1 {
            v -= l1[i] * l1[i] * dd[i - 1];
        }
        if i >= 2 {
            v -= lb[i] * lb[i] * dd[i - 2];
        }
        dd[i] = v;
    }
    // solve L y = x, D w = y, Lᵀ z = w
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut v = x[i];
        if i >= 1 {
            v -= l1[i] * y[i - 1];
        }
        if i >= 2 {
            v -= lb[i] * y[i - 2];
        }
        y[i] = v;
    }
    for i in 0..n {
        y[i] /= dd[i];
    }
    let mut z = vec![0.0; n];
    for i in (0..n).rev() {
        let mut v = y[i];
        if i + 1 < n {
            v -= l1[i + 1] * z[i + 1];
        }
        if i + 2 < n {
            v -= lb[i + 2] * z[i + 2];
        }
        z[i] = v;
    }
    Ok(z)
}

/// −3 dB cut-off (Hz) of the high-pass that [`smoothness_priors`] applies to a series sampled at
/// `fs` (Hz; the mean heart rate in beats/s for an RR series), for a long series.
pub fn smoothness_priors_cutoff_hz(lambda: f64, fs: f64) -> f64 {
    // detrended (high-pass) response H = λ²u² / (1 + λ²u²), u = 2 − 2cos ω; −3 dB where H = 1/√2
    //  →  λ²u² = 1/(√2 − 1)  →  u = 1.5538 / λ
    let u = (1.0 / (std::f64::consts::SQRT_2 - 1.0)).sqrt() / lambda;
    let c = 1.0 - u / 2.0;
    if !(-1.0..=1.0).contains(&c) {
        return f64::NAN;
    }
    c.acos() * fs / (2.0 * std::f64::consts::PI)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_a_linear_trend_and_keeps_fast_variation() {
        // second differences of a line are zero: the trend is the line itself
        let x: Vec<f64> = (0..200).map(|i| 800.0 + 0.5 * i as f64).collect();
        let d = smoothness_priors(&x, 500.0).unwrap();
        assert!(d.iter().all(|v| v.abs() < 1e-6), "{:?}", &d[..5]);
        // a fast oscillation passes almost unchanged
        let y: Vec<f64> = (0..400).map(|i| 30.0 * (i as f64 * 1.3).sin()).collect();
        let d = smoothness_priors(&y, 500.0).unwrap();
        // away from the two ends (the trend is pulled by the edges over tens of samples)
        let err = y[100..300]
            .iter()
            .zip(&d[100..300])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 0.5, "{err}");
    }

    #[test]
    fn cutoff_matches_the_frequency_response() {
        // λ = 500 on a 4 Hz series: −3 dB at ~0.035 Hz (the value quoted for Kubios' default)
        let fc = smoothness_priors_cutoff_hz(500.0, 4.0);
        assert!((fc - 0.0355).abs() < 0.0005, "{fc}");
        let w = 2.0 * std::f64::consts::PI * fc / 4.0;
        let u = 2.0 - 2.0 * w.cos();
        let h = 500.0f64.powi(2) * u * u / (1.0 + 500.0f64.powi(2) * u * u);
        assert!((h - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-9, "{h}");
        assert!(smoothness_priors_cutoff_hz(10.0, 4.0) > smoothness_priors_cutoff_hz(500.0, 4.0));
        assert!(smoothness_priors(&[1.0, 2.0], 0.0).is_err());
    }
}
