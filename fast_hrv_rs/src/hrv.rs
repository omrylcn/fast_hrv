//! HRV (Heart Rate Variability) metrics
//!
//! Time Domain, Poincaré, Asymmetry, Complexity (SampEn, DFA)

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/// Calculate mean of a slice
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Calculate standard deviation with ddof (degrees of freedom)
pub fn std_ddof(data: &[f64], ddof: usize) -> f64 {
    let n = data.len();
    if n <= ddof {
        return f64::NAN;
    }
    let mean_val = mean(data);
    let variance: f64 =
        data.iter().map(|&x| (x - mean_val).powi(2)).sum::<f64>() / (n - ddof) as f64;
    variance.sqrt()
}

// =============================================================================
// TIME DOMAIN
// =============================================================================

/// RMSSD - Root Mean Square of Successive Differences
pub fn compute_rmssd(rri: &[f64]) -> f64 {
    if rri.len() < 2 {
        return f64::NAN;
    }
    let n = rri.len() - 1;
    let sum_sq: f64 = rri.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
    (sum_sq / n as f64).sqrt()
}

/// SDNN - Standard Deviation of NN intervals
pub fn compute_sdnn(rri: &[f64]) -> f64 {
    std_ddof(rri, 1)
}

/// pNN50 - Percentage of successive differences > 50ms
pub fn compute_pnn50(rri: &[f64]) -> f64 {
    compute_pnnx(rri, 50.0)
}

/// pNN20 - Percentage of successive differences > 20ms
pub fn compute_pnn20(rri: &[f64]) -> f64 {
    compute_pnnx(rri, 20.0)
}

/// Generic pNNx function
fn compute_pnnx(rri: &[f64], threshold: f64) -> f64 {
    if rri.len() < 2 {
        return f64::NAN;
    }
    let n_diff = rri.len() - 1;
    let count: usize = rri
        .windows(2)
        .filter(|w| (w[1] - w[0]).abs() > threshold)
        .count();
    (count as f64 / n_diff as f64) * 100.0
}

/// SDSD - Standard deviation of successive differences
pub fn compute_sdsd(rri: &[f64]) -> f64 {
    if rri.len() < 2 {
        return f64::NAN;
    }
    let diffs: Vec<f64> = rri.windows(2).map(|w| w[1] - w[0]).collect();
    std_ddof(&diffs, 1)
}

// =============================================================================
// POINCARÉ
// =============================================================================

/// Poincaré plot results
#[derive(Debug, Clone)]
pub struct PoincareResult {
    pub sd1: f64,
    pub sd2: f64,
    pub csi: f64,
    pub cvi: f64,
}

/// Compute Poincaré plot metrics (SD1, SD2, CSI, CVI)
pub fn compute_poincare(rri: &[f64]) -> PoincareResult {
    let sqrt2 = std::f64::consts::SQRT_2;
    let n = rri.len() - 1;

    if n < 2 {
        return PoincareResult {
            sd1: f64::NAN,
            sd2: f64::NAN,
            csi: f64::NAN,
            cvi: f64::NAN,
        };
    }

    let (x1_vals, x2_vals): (Vec<f64>, Vec<f64>) = rri
        .windows(2)
        .map(|w| {
            let x1 = (w[0] - w[1]) / sqrt2;
            let x2 = (w[0] + w[1]) / sqrt2;
            (x1, x2)
        })
        .unzip();

    let sd1 = std_ddof(&x1_vals, 1);
    let sd2 = std_ddof(&x2_vals, 1);

    let t = 4.0 * sd1;
    let l = 4.0 * sd2;

    let csi = if t > 0.0 { l / t } else { f64::NAN };
    let cvi = if l > 0.0 && t > 0.0 {
        (l * t).log10()
    } else {
        f64::NAN
    };

    PoincareResult { sd1, sd2, csi, cvi }
}

// =============================================================================
// HEART RATE ASYMMETRY
// =============================================================================

/// Heart Rate Asymmetry results
#[derive(Debug, Clone)]
pub struct AsymmetryResult {
    pub gi: f64,
    pub pi: f64,
    pub sd1d: f64,
    pub sd1a: f64,
    pub c1d: f64,
    pub c1a: f64,
}

/// Compute Heart Rate Asymmetry metrics
pub fn compute_asymmetry(rri: &[f64]) -> AsymmetryResult {
    let sqrt2 = std::f64::consts::SQRT_2;
    let n = rri.len() - 1;

    if n < 2 {
        return AsymmetryResult {
            gi: f64::NAN,
            pi: f64::NAN,
            sd1d: f64::NAN,
            sd1a: f64::NAN,
            c1d: f64::NAN,
            c1a: f64::NAN,
        };
    }

    let mut decel_dist_sq_sum = 0.0;
    let mut accel_dist_sq_sum = 0.0;
    let mut decel_dist_sum = 0.0;
    let mut total_dist_sum = 0.0;
    let mut accel_count = 0usize;
    let mut nochange_count = 0usize;

    for w in rri.windows(2) {
        let diff = w[1] - w[0];
        let dist = diff.abs() / sqrt2;
        total_dist_sum += dist;

        if diff > 0.0 {
            decel_dist_sum += dist;
            decel_dist_sq_sum += dist * dist;
        } else if diff < 0.0 {
            accel_count += 1;
            accel_dist_sq_sum += dist * dist;
        } else {
            nochange_count += 1;
        }
    }

    let gi = if total_dist_sum > 0.0 {
        (decel_dist_sum / total_dist_sum) * 100.0
    } else {
        50.0
    };

    let m = n - nochange_count;
    let pi = if m > 0 {
        (accel_count as f64 / m as f64) * 100.0
    } else {
        50.0
    };

    let sd1d = (decel_dist_sq_sum / (n - 1) as f64).sqrt();
    let sd1a = (accel_dist_sq_sum / (n - 1) as f64).sqrt();

    let sd1i_sq = sd1d * sd1d + sd1a * sd1a;
    let (c1d, c1a) = if sd1i_sq > 0.0 {
        ((sd1d * sd1d) / sd1i_sq, (sd1a * sd1a) / sd1i_sq)
    } else {
        (0.5, 0.5)
    };

    AsymmetryResult {
        gi,
        pi,
        sd1d,
        sd1a,
        c1d,
        c1a,
    }
}

// =============================================================================
// SAMPLE ENTROPY
// =============================================================================

fn create_embedding(data: &[f64], dim: usize) -> Vec<Vec<f64>> {
    let n = data.len() - dim + 1;
    (0..n).map(|i| data[i..i + dim].to_vec()).collect()
}

fn count_neighbors_chebyshev(embedded: &[Vec<f64>], tolerance: f64) -> Vec<usize> {
    let n = embedded.len();
    let mut counts = vec![0usize; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let dist = embedded[i]
                .iter()
                .zip(&embedded[j])
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            if dist <= tolerance {
                counts[i] += 1;
            }
        }
    }
    counts
}

/// Compute Sample Entropy (SampEn)
pub fn compute_sampen(rri: &[f64], dim: usize, tolerance: Option<f64>) -> f64 {
    let tol = tolerance.unwrap_or_else(|| 0.2 * std_ddof(rri, 1));
    let n = rri.len();

    if n < dim + 2 {
        return f64::NAN;
    }

    let embedded_m = create_embedding(rri, dim);
    let embedded_m1 = create_embedding(rri, dim + 1);

    let count_m = count_neighbors_chebyshev(&embedded_m[..embedded_m.len() - 1], tol);
    let count_m1 = count_neighbors_chebyshev(&embedded_m1, tol);

    let n_m = (embedded_m.len() - 1) as f64;
    let n_m1 = embedded_m1.len() as f64;

    let b: f64 = count_m.iter().map(|&c| c as f64).sum::<f64>() / (n_m * (n_m - 1.0));
    let a: f64 = count_m1.iter().map(|&c| c as f64).sum::<f64>() / (n_m1 * (n_m1 - 1.0));

    if a == 0.0 || b == 0.0 {
        return f64::INFINITY;
    }
    -(a / b).ln()
}

/// Count neighbors including self-match (for ApEn)
fn count_neighbors_chebyshev_with_self(embedded: &[Vec<f64>], tolerance: f64) -> Vec<usize> {
    let n = embedded.len();
    let mut counts = vec![0usize; n];

    for i in 0..n {
        for j in 0..n {
            let dist = embedded[i]
                .iter()
                .zip(&embedded[j])
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            if dist <= tolerance {
                counts[i] += 1;
            }
        }
    }
    counts
}

/// Compute Approximate Entropy (ApEn)
pub fn compute_apen(rri: &[f64], dim: usize, tolerance: Option<f64>) -> f64 {
    let tol = tolerance.unwrap_or_else(|| 0.2 * std_ddof(rri, 1));
    let n = rri.len();

    if n < dim + 2 {
        return f64::NAN;
    }

    let embedded_m = create_embedding(rri, dim);
    let embedded_m1 = create_embedding(rri, dim + 1);

    // ApEn includes self-matches
    let count_m = count_neighbors_chebyshev_with_self(&embedded_m, tol);
    let count_m1 = count_neighbors_chebyshev_with_self(&embedded_m1, tol);

    let n_m = embedded_m.len() as f64;
    let n_m1 = embedded_m1.len() as f64;

    // Phi values
    let phi_m: f64 = count_m.iter().map(|&c| (c as f64 / n_m).ln()).sum::<f64>() / n_m;

    let phi_m1: f64 = count_m1
        .iter()
        .map(|&c| (c as f64 / n_m1).ln())
        .sum::<f64>()
        / n_m1;

    (phi_m - phi_m1).abs()
}

// =============================================================================
// DFA (Detrended Fluctuation Analysis)
// =============================================================================

/// DFA results
#[derive(Debug, Clone)]
pub struct DfaResult {
    pub alpha1: f64,
    pub alpha2: f64,
}

fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y).map(|(a, b)| a * b).sum();
    let sum_xx: f64 = x.iter().map(|a| a * a).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-10 {
        return (f64::NAN, f64::NAN);
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;
    (slope, intercept)
}

fn compute_dfa_alpha(integrated: &[f64], min_scale: usize, max_scale: usize) -> f64 {
    let n = integrated.len();
    let mut log_scales = Vec::new();
    let mut log_fluctuations = Vec::new();

    for scale in min_scale..=max_scale {
        let n_segments = n / scale;
        if n_segments < 2 {
            continue;
        }

        let mut local_rms_sum = 0.0;
        for seg_idx in 0..n_segments {
            let start = seg_idx * scale;
            let segment: Vec<f64> = integrated[start..start + scale].to_vec();
            let x: Vec<f64> = (0..scale).map(|i| i as f64).collect();

            let (slope, intercept) = linear_regression(&x, &segment);

            let mut sum_sq = 0.0;
            for (i, &val) in segment.iter().enumerate() {
                let trend = slope * i as f64 + intercept;
                sum_sq += (val - trend).powi(2);
            }
            local_rms_sum += (sum_sq / scale as f64).sqrt();
        }

        let avg_fluctuation = local_rms_sum / n_segments as f64;
        if avg_fluctuation > 0.0 {
            log_scales.push((scale as f64).ln());
            log_fluctuations.push(avg_fluctuation.ln());
        }
    }

    if log_scales.len() < 2 {
        return f64::NAN;
    }
    let (alpha, _) = linear_regression(&log_scales, &log_fluctuations);
    alpha
}

/// Compute DFA alpha1 and alpha2
pub fn compute_dfa(
    rri: &[f64],
    short_range: (usize, usize),
    long_range: (usize, Option<usize>),
) -> DfaResult {
    let n = rri.len();
    if n < 16 {
        return DfaResult {
            alpha1: f64::NAN,
            alpha2: f64::NAN,
        };
    }

    let mean_rri = mean(rri);
    let mut integrated = Vec::with_capacity(n);
    let mut cumsum = 0.0;
    for &val in rri {
        cumsum += val - mean_rri;
        integrated.push(cumsum);
    }

    let alpha1 = compute_dfa_alpha(&integrated, short_range.0, short_range.1);
    let max_scale = long_range.1.unwrap_or(n / 10);
    let alpha2 = if max_scale > long_range.0 {
        compute_dfa_alpha(&integrated, long_range.0, max_scale)
    } else {
        f64::NAN
    };

    DfaResult { alpha1, alpha2 }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const TEST_RRI: [f64; 10] = [
        800.0, 850.0, 790.0, 820.0, 880.0, 750.0, 810.0, 860.0, 800.0, 830.0,
    ];

    // Expected values from Python
    const EXPECTED_MEAN: f64 = 819.0;
    const EXPECTED_STD: f64 = 37.844711945293255;
    const EXPECTED_RMSSD: f64 = 65.06407098647712;
    const EXPECTED_PNN50: f64 = 55.55555555555556;
    const EXPECTED_PNN20: f64 = 100.0;
    const EXPECTED_SDSD: f64 = 68.92024376045111;
    const EXPECTED_SD1: f64 = 48.73397172404481;
    const EXPECTED_SD2: f64 = 27.938424357067053;
    const EXPECTED_CSI: f64 = 0.5732843716343878;
    const EXPECTED_GI: f64 = 52.830188679245296;
    const EXPECTED_PI: f64 = 33.33333333333333;
    const EXPECTED_SD1D: f64 = 29.580398915498076;
    const EXPECTED_SD1A: f64 = 38.810436740650054;
    const EXPECTED_C1D: f64 = 0.3674540682414698;
    const EXPECTED_C1A: f64 = 0.6325459317585304;

    const TEST_RRI_LARGE: [f64; 100] = [
        824.8357076505616,
        793.0867849414408,
        832.3844269050346,
        876.1514928204012,
        788.2923312638331,
        788.293152152541,
        878.9606407753696,
        838.3717364576454,
        776.5262807032524,
        827.1280021792983,
        776.8291153593768,
        776.7135123214872,
        812.0981135783018,
        704.33598776711,
        713.7541083743483,
        771.8856235379513,
        749.3584439832788,
        815.7123666297637,
        754.5987962239394,
        729.3848149332355,
        873.2824384460777,
        788.7111849756732,
        803.3764102343962,
        728.7625906893271,
        772.7808637737409,
        805.5461294854933,
        742.4503211288849,
        818.7849009172836,
        769.9680655040597,
        785.4153125103362,
        769.9146693885301,
        892.6139092254468,
        799.3251387631033,
        747.114453552205,
        841.1272456051595,
        738.9578175014489,
        810.4431797502377,
        702.0164938060112,
        733.5906975550785,
        809.8430617934562,
        836.9233289997705,
        808.5684140594985,
        794.217585880588,
        784.9448152205356,
        726.0739004816286,
        764.0077895802646,
        776.9680614520106,
        852.8561113109458,
        817.1809144784231,
        711.8479922318633,
        816.2041984697397,
        780.7458859791842,
        766.1538999847021,
        830.5838144420434,
        851.5499761247976,
        846.5640059558099,
        758.039123838868,
        784.5393812074393,
        816.5631715701782,
        848.777256356118,
        776.0412881077355,
        790.7170511668091,
        744.6832512996986,
        740.1896687959664,
        840.6262911197099,
        867.8120014285412,
        796.3994939209833,
        850.1766448946012,
        818.0818012523818,
        767.7440122697437,
        818.0697802754207,
        876.9018283232984,
        798.2086980445024,
        878.2321827907003,
        669.0127447955128,
        841.0951252187612,
        804.3523534119086,
        785.0496324767066,
        804.5880388267751,
        700.6215542699554,
        789.0164056081244,
        817.8556285755873,
        873.8947022370758,
        774.0864890863177,
        759.5753198553406,
        774.9121478207732,
        845.7701058851037,
        816.4375554829842,
        773.511989811648,
        825.6633716556678,
        804.8538774674021,
        848.4322495266445,
        764.8973453061324,
        783.6168926701116,
        780.3945923433921,
        726.824252593394,
        814.8060138532288,
        813.0527636089945,
        800.2556728321231,
        788.2706433312427,
    ];
    const EXPECTED_SAMPEN: f64 = 1.8632184332102002;
    const EXPECTED_APEN: f64 = 0.5357760466372254;
    const EXPECTED_DFA_ALPHA1: f64 = 0.5534084680171664;

    #[test]
    fn test_mean() {
        assert_relative_eq!(mean(&TEST_RRI), EXPECTED_MEAN, epsilon = 1e-10);
    }

    #[test]
    fn test_std() {
        assert_relative_eq!(std_ddof(&TEST_RRI, 1), EXPECTED_STD, epsilon = 1e-10);
    }

    #[test]
    fn test_rmssd() {
        assert_relative_eq!(compute_rmssd(&TEST_RRI), EXPECTED_RMSSD, epsilon = 1e-10);
    }

    #[test]
    fn test_sdnn() {
        assert_relative_eq!(compute_sdnn(&TEST_RRI), EXPECTED_STD, epsilon = 1e-10);
    }

    #[test]
    fn test_pnn50() {
        assert_relative_eq!(compute_pnn50(&TEST_RRI), EXPECTED_PNN50, epsilon = 1e-10);
    }

    #[test]
    fn test_pnn20() {
        assert_relative_eq!(compute_pnn20(&TEST_RRI), EXPECTED_PNN20, epsilon = 1e-10);
    }

    #[test]
    fn test_sdsd() {
        assert_relative_eq!(compute_sdsd(&TEST_RRI), EXPECTED_SDSD, epsilon = 1e-10);
    }

    #[test]
    fn test_poincare_sd1() {
        assert_relative_eq!(
            compute_poincare(&TEST_RRI).sd1,
            EXPECTED_SD1,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_poincare_sd2() {
        assert_relative_eq!(
            compute_poincare(&TEST_RRI).sd2,
            EXPECTED_SD2,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_poincare_csi() {
        assert_relative_eq!(
            compute_poincare(&TEST_RRI).csi,
            EXPECTED_CSI,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_asymmetry_gi() {
        assert_relative_eq!(
            compute_asymmetry(&TEST_RRI).gi,
            EXPECTED_GI,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_asymmetry_pi() {
        assert_relative_eq!(
            compute_asymmetry(&TEST_RRI).pi,
            EXPECTED_PI,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_asymmetry_sd1d() {
        assert_relative_eq!(
            compute_asymmetry(&TEST_RRI).sd1d,
            EXPECTED_SD1D,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_asymmetry_sd1a() {
        assert_relative_eq!(
            compute_asymmetry(&TEST_RRI).sd1a,
            EXPECTED_SD1A,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_asymmetry_c1d() {
        assert_relative_eq!(
            compute_asymmetry(&TEST_RRI).c1d,
            EXPECTED_C1D,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_asymmetry_c1a() {
        assert_relative_eq!(
            compute_asymmetry(&TEST_RRI).c1a,
            EXPECTED_C1A,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_sampen() {
        assert_relative_eq!(
            compute_sampen(&TEST_RRI_LARGE, 2, None),
            EXPECTED_SAMPEN,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_apen() {
        assert_relative_eq!(
            compute_apen(&TEST_RRI_LARGE, 2, None),
            EXPECTED_APEN,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_dfa_alpha1() {
        assert_relative_eq!(
            compute_dfa(&TEST_RRI_LARGE, (4, 11), (12, None)).alpha1,
            EXPECTED_DFA_ALPHA1,
            epsilon = 1e-6
        );
    }
}
