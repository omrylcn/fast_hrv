//! SIMD-optimized operations for HRV analysis
//!
//! Enable with: cargo build --features simd
//!
//! Uses compiler auto-vectorization with explicit hints for AVX/SSE.

/// SIMD-optimized mean calculation
/// Uses loop unrolling for better auto-vectorization
#[inline]
pub fn mean_simd(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let n = data.len();
    let chunks = n / 4;

    // Process 4 elements at a time (helps auto-vectorization)
    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;

    for i in 0..chunks {
        let base = i * 4;
        sum0 += data[base];
        sum1 += data[base + 1];
        sum2 += data[base + 2];
        sum3 += data[base + 3];
    }

    // Handle remainder
    let mut sum_rem = 0.0;
    for i in (chunks * 4)..n {
        sum_rem += data[i];
    }

    (sum0 + sum1 + sum2 + sum3 + sum_rem) / n as f64
}

/// SIMD-optimized sum of squares (for variance)
#[inline]
pub fn sum_sq_diff_simd(data: &[f64], mean: f64) -> f64 {
    let n = data.len();
    let chunks = n / 4;

    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;

    for i in 0..chunks {
        let base = i * 4;
        let d0 = data[base] - mean;
        let d1 = data[base + 1] - mean;
        let d2 = data[base + 2] - mean;
        let d3 = data[base + 3] - mean;
        sum0 += d0 * d0;
        sum1 += d1 * d1;
        sum2 += d2 * d2;
        sum3 += d3 * d3;
    }

    let mut sum_rem = 0.0;
    for i in (chunks * 4)..n {
        let d = data[i] - mean;
        sum_rem += d * d;
    }

    sum0 + sum1 + sum2 + sum3 + sum_rem
}

/// SIMD-optimized standard deviation
#[inline]
pub fn std_simd(data: &[f64], ddof: usize) -> f64 {
    if data.len() <= ddof {
        return 0.0;
    }
    let mean = mean_simd(data);
    let sum_sq = sum_sq_diff_simd(data, mean);
    (sum_sq / (data.len() - ddof) as f64).sqrt()
}

/// SIMD-optimized RMSSD calculation
#[inline]
pub fn rmssd_simd(rri: &[f64]) -> f64 {
    if rri.len() < 2 {
        return 0.0;
    }

    let n = rri.len() - 1;
    let chunks = n / 4;

    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;

    for i in 0..chunks {
        let base = i * 4;
        let d0 = rri[base + 1] - rri[base];
        let d1 = rri[base + 2] - rri[base + 1];
        let d2 = rri[base + 3] - rri[base + 2];
        let d3 = rri[base + 4] - rri[base + 3];
        sum0 += d0 * d0;
        sum1 += d1 * d1;
        sum2 += d2 * d2;
        sum3 += d3 * d3;
    }

    let mut sum_rem = 0.0;
    for i in (chunks * 4)..n {
        let d = rri[i + 1] - rri[i];
        sum_rem += d * d;
    }

    ((sum0 + sum1 + sum2 + sum3 + sum_rem) / n as f64).sqrt()
}

/// SIMD-optimized gradient (finite differences)
#[inline]
pub fn gradient_simd(signal: &[f64], output: &mut [f64]) {
    let n = signal.len();
    if n < 2 {
        output.iter_mut().for_each(|x| *x = 0.0);
        return;
    }

    // First point
    output[0] = signal[1] - signal[0];

    // Middle points - process 4 at a time
    let middle_len = n - 2;
    let chunks = middle_len / 4;

    for i in 0..chunks {
        let base = 1 + i * 4;
        output[base] = (signal[base + 1] - signal[base - 1]) * 0.5;
        output[base + 1] = (signal[base + 2] - signal[base]) * 0.5;
        output[base + 2] = (signal[base + 3] - signal[base + 1]) * 0.5;
        output[base + 3] = (signal[base + 4] - signal[base + 2]) * 0.5;
    }

    // Remainder
    for i in (1 + chunks * 4)..(n - 1) {
        output[i] = (signal[i + 1] - signal[i - 1]) * 0.5;
    }

    // Last point
    output[n - 1] = signal[n - 1] - signal[n - 2];
}

/// SIMD-optimized dot product
#[inline]
pub fn dot_product_simd(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let chunks = n / 4;

    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;

    for i in 0..chunks {
        let base = i * 4;
        sum0 += a[base] * b[base];
        sum1 += a[base + 1] * b[base + 1];
        sum2 += a[base + 2] * b[base + 2];
        sum3 += a[base + 3] * b[base + 3];
    }

    let mut sum_rem = 0.0;
    for i in (chunks * 4)..n {
        sum_rem += a[i] * b[i];
    }

    sum0 + sum1 + sum2 + sum3 + sum_rem
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_simd() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = mean_simd(&data);
        assert!((result - 5.5).abs() < 1e-10);
    }

    #[test]
    fn test_std_simd() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let result = std_simd(&data, 0);
        assert!((result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_rmssd_simd() {
        let rri = vec![800.0, 810.0, 790.0, 805.0, 795.0];
        let result = rmssd_simd(&rri);
        // Manual: diffs = [10, -20, 15, -10], sq = [100, 400, 225, 100], mean = 206.25, sqrt = 14.36
        assert!((result - 14.361406616345072).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_simd() {
        let signal = vec![1.0, 4.0, 9.0, 16.0, 25.0];
        let mut output = vec![0.0; 5];
        gradient_simd(&signal, &mut output);
        // Expected: [3.0, 4.0, 6.0, 8.0, 9.0]
        assert!((output[0] - 3.0).abs() < 1e-10);
        assert!((output[2] - 6.0).abs() < 1e-10);
        assert!((output[4] - 9.0).abs() < 1e-10);
    }
}
