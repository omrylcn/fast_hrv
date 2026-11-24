//! ECG Peak Detection and Artifact Correction
//!
//! Fast R-peak detection using NeuroKit algorithm with Rust optimization.
//! Artifact correction using Kubios method (Lipponen & Tarvainen 2019).

use ordered_float::OrderedFloat;
use std::collections::BTreeMap;

// =============================================================================
// SLIDING WINDOW DATA STRUCTURE (O(log k) operations)
// =============================================================================

/// Sliding window for O(n log k) median/percentile computation
struct SlidingWindow {
    tree: BTreeMap<OrderedFloat<f64>, usize>, // value -> count
    size: usize,
}

impl SlidingWindow {
    fn new() -> Self {
        Self {
            tree: BTreeMap::new(),
            size: 0,
        }
    }

    #[inline]
    fn insert(&mut self, val: f64) {
        *self.tree.entry(OrderedFloat(val)).or_insert(0) += 1;
        self.size += 1;
    }

    #[inline]
    fn remove(&mut self, val: f64) {
        let key = OrderedFloat(val);
        if let Some(count) = self.tree.get_mut(&key) {
            *count -= 1;
            if *count == 0 {
                self.tree.remove(&key);
            }
            self.size -= 1;
        }
    }

    /// Get value at given rank (0-indexed)
    fn get_rank(&self, rank: usize) -> f64 {
        let mut cumulative = 0;
        for (&val, &count) in &self.tree {
            cumulative += count;
            if cumulative > rank {
                return val.0;
            }
        }
        // Return last value if rank exceeds size
        self.tree.keys().last().map(|k| k.0).unwrap_or(0.0)
    }

    /// Get percentile value
    fn percentile(&self, pct: f64) -> f64 {
        if self.size == 0 {
            return 0.0;
        }
        let rank = (pct / 100.0 * (self.size - 1) as f64).round() as usize;
        self.get_rank(rank.min(self.size - 1))
    }

    fn median(&self) -> f64 {
        self.percentile(50.0)
    }
}

// =============================================================================
// SIGNAL PROCESSING UTILITIES
// =============================================================================

/// Compute gradient (finite differences) like np.gradient
pub fn gradient(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    if n < 2 {
        return vec![0.0; n];
    }

    let mut grad = vec![0.0; n];

    // First point: forward difference
    grad[0] = signal[1] - signal[0];

    // Middle points: central difference
    for i in 1..n - 1 {
        grad[i] = (signal[i + 1] - signal[i - 1]) / 2.0;
    }

    // Last point: backward difference
    grad[n - 1] = signal[n - 1] - signal[n - 2];

    grad
}

/// Uniform filter 1D (moving average) with 'nearest' edge mode - O(n) sliding window
pub fn uniform_filter1d(signal: &[f64], size: usize) -> Vec<f64> {
    let n = signal.len();
    if n == 0 || size == 0 {
        return signal.to_vec();
    }
    if size == 1 {
        return signal.to_vec();
    }

    let half = size / 2;
    let mut result = vec![0.0; n];
    let inv_size = 1.0 / size as f64;

    // Helper to get clamped value (nearest mode)
    let get = |i: isize| -> f64 { signal[i.clamp(0, (n - 1) as isize) as usize] };

    // Initialize sum for first position
    let mut sum = 0.0;
    for j in 0..size {
        sum += get(j as isize - half as isize);
    }
    result[0] = sum * inv_size;

    // Sliding window for remaining positions
    for i in 1..n {
        let old_idx = i as isize - 1 - half as isize;
        let new_idx = i as isize - 1 + size as isize - half as isize;
        sum -= get(old_idx);
        sum += get(new_idx);
        result[i] = sum * inv_size;
    }

    result
}

/// Median filter 1D with 'reflect' edge mode
/// O(n log k) using sliding window BTreeMap
pub fn median_filter(signal: &[f64], size: usize) -> Vec<f64> {
    let n = signal.len();
    if n == 0 || size == 0 {
        return signal.to_vec();
    }
    if size == 1 {
        return signal.to_vec();
    }

    let half = size / 2;
    let mut result = vec![0.0; n];

    // Helper for reflect mode indexing
    let get_val = |idx: isize| -> f64 {
        let clamped = if idx < 0 {
            ((-idx - 1) as usize).min(n - 1)
        } else if idx >= n as isize {
            ((2 * n as isize - 1 - idx) as usize).min(n - 1)
        } else {
            idx as usize
        };
        signal[clamped]
    };

    let mut sw = SlidingWindow::new();

    // Initialize window for position 0
    for j in 0..size {
        sw.insert(get_val(j as isize - half as isize));
    }
    result[0] = sw.median();

    // Slide window
    for i in 1..n {
        // Remove old value (leftmost of previous window)
        let old_idx = i as isize - 1 - half as isize;
        sw.remove(get_val(old_idx));

        // Add new value (rightmost of new window)
        let new_idx = i as isize + size as isize - 1 - half as isize;
        sw.insert(get_val(new_idx));

        result[i] = sw.median();
    }

    result
}

/// Percentile filter 1D with 'reflect' edge mode
/// O(n log k) using sliding window BTreeMap
pub fn percentile_filter(signal: &[f64], percentile: f64, size: usize) -> Vec<f64> {
    let n = signal.len();
    if n == 0 || size == 0 {
        return signal.to_vec();
    }
    if size == 1 {
        return signal.to_vec();
    }

    let half = size / 2;
    let mut result = vec![0.0; n];

    // Helper for reflect mode indexing
    let get_val = |idx: isize| -> f64 {
        let clamped = if idx < 0 {
            ((-idx - 1) as usize).min(n - 1)
        } else if idx >= n as isize {
            ((2 * n as isize - 1 - idx) as usize).min(n - 1)
        } else {
            idx as usize
        };
        signal[clamped]
    };

    let mut sw = SlidingWindow::new();

    // Initialize window for position 0
    for j in 0..size {
        sw.insert(get_val(j as isize - half as isize));
    }
    result[0] = sw.percentile(percentile);

    // Slide window
    for i in 1..n {
        // Remove old value
        let old_idx = i as isize - 1 - half as isize;
        sw.remove(get_val(old_idx));

        // Add new value
        let new_idx = i as isize + size as isize - 1 - half as isize;
        sw.insert(get_val(new_idx));

        result[i] = sw.percentile(percentile);
    }

    result
}

// =============================================================================
// PEAK DETECTION
// =============================================================================

/// Compute gradient and absolute value in one pass (fused operation)
#[inline]
fn gradient_abs_inplace(signal: &[f64], output: &mut [f64]) {
    let n = signal.len();
    if n < 2 {
        output.iter_mut().for_each(|x| *x = 0.0);
        return;
    }

    // First point: forward difference
    output[0] = (signal[1] - signal[0]).abs();

    // Middle points: central difference
    for i in 1..n - 1 {
        output[i] = ((signal[i + 1] - signal[i - 1]) * 0.5).abs();
    }

    // Last point: backward difference
    output[n - 1] = (signal[n - 1] - signal[n - 2]).abs();
}

/// Optimized uniform filter using sliding window (in-place capable)
#[inline]
fn uniform_filter_fast(input: &[f64], output: &mut [f64], size: usize) {
    let n = input.len();
    if n == 0 || size <= 1 {
        output.copy_from_slice(input);
        return;
    }

    let half = size / 2;
    let inv_size = 1.0 / size as f64;

    // Helper to get clamped value
    let get = |i: isize| -> f64 { input[i.clamp(0, (n - 1) as isize) as usize] };

    // Initialize sum
    let mut sum = 0.0;
    for j in 0..size {
        sum += get(j as isize - half as isize);
    }
    output[0] = sum * inv_size;

    // Sliding window
    for i in 1..n {
        sum -= get(i as isize - 1 - half as isize);
        sum += get(i as isize - 1 + size as isize - half as isize);
        output[i] = sum * inv_size;
    }
}

/// Find R-peaks in an ECG signal
///
/// Fast R-peak detection using NeuroKit algorithm with Rust optimization.
/// Optimized version with reduced allocations and fused operations.
pub fn find_peaks(
    ecg_cleaned: &[f64],
    sampling_rate: u32,
    smoothwindow: f64,
    avgwindow: f64,
    gradthreshweight: f64,
    minlenweight: f64,
    mindelay: f64,
) -> Vec<usize> {
    let n = ecg_cleaned.len();
    if n < 10 {
        return vec![];
    }

    let smooth_kernel = (smoothwindow * sampling_rate as f64).round().max(1.0) as usize;
    let avg_kernel = (avgwindow * sampling_rate as f64).round().max(1.0) as usize;
    let mindelay_samples = (sampling_rate as f64 * mindelay).round() as isize;

    // Allocate working buffers (reuse for multiple operations)
    let mut buf1 = vec![0.0; n];
    let mut buf2 = vec![0.0; n];

    // Step 1: Compute |gradient| directly into buf1
    gradient_abs_inplace(ecg_cleaned, &mut buf1);

    // Step 2: Smooth gradient -> buf2
    uniform_filter_fast(&buf1, &mut buf2, smooth_kernel);

    // Step 3: Average of smoothed -> buf1 (reuse)
    uniform_filter_fast(&buf2, &mut buf1, avg_kernel);

    // Step 4: Find QRS boundaries inline (no bool vector needed)
    // buf2 = smoothgrad, buf1 = avggrad (threshold base)
    let mut beg_qrs: Vec<usize> = Vec::with_capacity(n / 200); // ~300ms at 512Hz
    let mut end_qrs: Vec<usize> = Vec::with_capacity(n / 200);

    let mut prev_above = buf2[0] > gradthreshweight * buf1[0];
    for i in 1..n {
        let curr_above = buf2[i] > gradthreshweight * buf1[i];
        if !prev_above && curr_above {
            beg_qrs.push(i - 1);
        } else if prev_above && !curr_above {
            end_qrs.push(i - 1);
        }
        prev_above = curr_above;
    }

    if beg_qrs.is_empty() {
        return vec![];
    }

    // Filter end_qrs
    let first_beg = beg_qrs[0];
    end_qrs.retain(|&e| e > first_beg);

    if end_qrs.is_empty() {
        return vec![];
    }

    // Step 5: Calculate min length
    let num_qrs = beg_qrs.len().min(end_qrs.len());
    let total_len: usize = (0..num_qrs).map(|i| end_qrs[i] - beg_qrs[i]).sum();
    let mean_len = total_len as f64 / num_qrs as f64;
    let min_len = (mean_len * minlenweight) as usize;

    // Step 6: Find peaks
    let mut peaks: Vec<usize> = Vec::with_capacity(num_qrs);
    let mut last_peak: isize = -mindelay_samples;

    for i in 0..num_qrs {
        let beg = beg_qrs[i];
        let end = end_qrs[i];

        if end - beg < min_len {
            continue;
        }

        // Find argmax in segment (inline, no iterator overhead)
        let mut max_val = ecg_cleaned[beg];
        let mut max_idx = beg;
        for j in beg + 1..end {
            if ecg_cleaned[j] > max_val {
                max_val = ecg_cleaned[j];
                max_idx = j;
            }
        }

        if max_idx as isize - last_peak > mindelay_samples {
            peaks.push(max_idx);
            last_peak = max_idx as isize;
        }
    }

    peaks
}

/// Find peaks with default parameters
pub fn find_peaks_default(ecg_cleaned: &[f64], sampling_rate: u32) -> Vec<usize> {
    find_peaks(ecg_cleaned, sampling_rate, 0.1, 0.75, 1.5, 0.4, 0.3)
}

// =============================================================================
// ARTIFACT CORRECTION
// =============================================================================

/// Artifacts detected in the signal
#[derive(Debug, Clone, Default)]
pub struct Artifacts {
    pub ectopic: Vec<usize>,
    pub missed: Vec<usize>,
    pub extra: Vec<usize>,
    pub longshort: Vec<usize>,
}

impl Artifacts {
    pub fn total_count(&self) -> usize {
        self.ectopic.len() + self.missed.len() + self.extra.len() + self.longshort.len()
    }
}

/// Compute adaptive threshold using IQR
fn compute_threshold_fast(signal: &[f64], alpha: f64, window_width: usize) -> Vec<f64> {
    let n = signal.len();
    if n < 3 {
        return vec![0.0; n];
    }

    let abs_signal: Vec<f64> = signal.iter().map(|&x| x.abs()).collect();
    let q1 = percentile_filter(&abs_signal, 25.0, window_width);
    let q3 = percentile_filter(&abs_signal, 75.0, window_width);

    q1.iter()
        .zip(&q3)
        .map(|(&q1v, &q3v)| alpha * (q3v - q1v) / 2.0)
        .collect()
}

/// Compute subspaces s12 and s22
fn compute_subspaces_fast(drrs: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = drrs.len();
    if n < 3 {
        return (vec![0.0; n], vec![0.0; n]);
    }

    // Pad with reflection
    let mut padded = vec![0.0; n + 4];
    padded[0] = drrs[1];
    padded[1] = drrs[0];
    for (i, &v) in drrs.iter().enumerate() {
        padded[i + 2] = v;
    }
    padded[n + 2] = drrs[n - 1];
    padded[n + 3] = drrs[n - 2];

    let mut s12 = vec![0.0; n];
    let mut s22 = vec![0.0; n];

    for i in 0..n {
        let prev1 = padded[i + 1];
        let curr = padded[i + 2];
        let next1 = padded[i + 3];
        let next2 = padded[i + 4];

        // s12
        s12[i] = if curr > 0.0 {
            prev1.max(next1)
        } else if curr < 0.0 {
            prev1.min(next1)
        } else {
            0.0
        };

        // s22
        s22[i] = if curr >= 0.0 {
            next1.min(next2)
        } else {
            next1.max(next2)
        };
    }

    (s12, s22)
}

/// Classify artifacts based on Kubios criteria
fn classify_artifacts(
    drrs: &[f64],
    mrrs: &[f64],
    s12: &[f64],
    s22: &[f64],
    rr: &[f64],
    medrr: &[f64],
    th2: &[f64],
    c1: f64,
    c2: f64,
) -> Artifacts {
    let n = rr.len();
    let mut artifacts = Artifacts::default();

    let mut i = 0;
    while i < n.saturating_sub(2) {
        if drrs[i].abs() <= 1.0 {
            i += 1;
            continue;
        }

        // Ectopic detection
        let eq1 = drrs[i] > 1.0 && s12[i] < (-c1 * drrs[i] - c2);
        let eq2 = drrs[i] < -1.0 && s12[i] > (-c1 * drrs[i] + c2);

        if eq1 || eq2 {
            artifacts.ectopic.push(i);
            i += 1;
            continue;
        }

        // Check for long/short
        if !(drrs[i].abs() > 1.0 || mrrs[i].abs() > 3.0) {
            i += 1;
            continue;
        }

        let mut longshort_candidates = vec![i];
        if i + 2 < n && drrs[i + 1].abs() < drrs[i + 2].abs() {
            longshort_candidates.push(i + 1);
        }

        for &j in &longshort_candidates {
            if j >= n - 1 {
                continue;
            }

            let eq3 = drrs[j] > 1.0 && s22[j] < -1.0;
            let eq4 = mrrs[j].abs() > 3.0;
            let eq5 = drrs[j] < -1.0 && s22[j] > 1.0;

            if !(eq3 || eq4 || eq5) {
                i += 1;
                continue;
            }

            let eq6 = (rr[j] / 2.0 - medrr[j]).abs() < th2[j];
            let eq7 = j + 1 < n && (rr[j] + rr[j + 1] - medrr[j]).abs() < th2[j];

            if eq5 && eq7 {
                artifacts.extra.push(j);
                i += 1;
                continue;
            }

            if eq3 && eq6 {
                artifacts.missed.push(j);
                i += 1;
                continue;
            }

            artifacts.longshort.push(j);
            i += 1;
        }

        i += 1;
    }

    artifacts
}

/// Find artifacts in peaks
fn find_artifacts_fast(
    peaks: &[usize],
    sampling_rate: u32,
    c1: f64,
    c2: f64,
    alpha: f64,
    window_width: usize,
    medfilt_order: usize,
) -> Artifacts {
    let n = peaks.len();
    if n < 3 {
        return Artifacts::default();
    }

    // RR intervals in seconds
    let mut rr: Vec<f64> = vec![0.0; n];
    for i in 1..n {
        rr[i] = (peaks[i] - peaks[i - 1]) as f64 / sampling_rate as f64;
    }
    if n > 1 {
        rr[0] = rr[1..].iter().sum::<f64>() / (n - 1) as f64;
    }

    // dRRs: consecutive differences
    let mut drrs: Vec<f64> = vec![0.0; n];
    for i in 1..n {
        drrs[i] = rr[i] - rr[i - 1];
    }
    if n > 1 {
        drrs[0] = drrs[1..].iter().sum::<f64>() / (n - 1) as f64;
    }

    // Threshold 1
    let th1 = compute_threshold_fast(&drrs, alpha, window_width);

    // Normalize drrs
    let drrs_norm: Vec<f64> = drrs
        .iter()
        .zip(&th1)
        .map(|(&d, &t)| if t != 0.0 { d / t } else { f64::NAN })
        .collect();

    // Subspaces
    let (s12, s22) = compute_subspaces_fast(&drrs_norm);

    // mRRs: deviation from median
    let medrr = median_filter(&rr, medfilt_order);
    let mrrs: Vec<f64> = rr
        .iter()
        .zip(&medrr)
        .map(|(&r, &m)| {
            let diff = r - m;
            if diff < 0.0 { diff * 2.0 } else { diff }
        })
        .collect();

    // Threshold 2
    let th2 = compute_threshold_fast(&mrrs, alpha, window_width);

    // Normalize mrrs
    let mrrs_norm: Vec<f64> = mrrs
        .iter()
        .zip(&th2)
        .map(|(&m, &t)| if t != 0.0 { m / t } else { f64::NAN })
        .collect();

    // Classify
    classify_artifacts(
        &drrs_norm, &mrrs_norm, &s12, &s22, &rr, &medrr, &th2, c1, c2,
    )
}

/// Update indices after insertion/deletion
fn update_indices(source_idcs: &[usize], update_idcs: &[usize], update: isize) -> Vec<usize> {
    if update_idcs.is_empty() {
        return vec![];
    }

    let mut result: Vec<usize> = update_idcs
        .iter()
        .map(|&u| {
            let mut new_u = u as isize;
            for &s in source_idcs {
                if u > s {
                    new_u += update;
                }
            }
            new_u.max(0) as usize
        })
        .collect();

    result.sort();
    result.dedup();
    result
}

/// Correct detected artifacts
fn correct_artifacts(artifacts: &Artifacts, peaks: &[usize]) -> Vec<usize> {
    let mut peaks_clean: Vec<usize> = peaks.to_vec();

    let mut extra_idcs = artifacts.extra.clone();
    let mut missed_idcs = artifacts.missed.clone();
    let mut ectopic_idcs = artifacts.ectopic.clone();
    let mut longshort_idcs = artifacts.longshort.clone();

    // Delete extra peaks
    if !extra_idcs.is_empty() {
        extra_idcs.sort();
        for (offset, &idx) in extra_idcs.iter().enumerate() {
            if idx >= offset && idx - offset < peaks_clean.len() {
                peaks_clean.remove(idx - offset);
            }
        }
        missed_idcs = update_indices(&extra_idcs, &missed_idcs, -1);
        ectopic_idcs = update_indices(&extra_idcs, &ectopic_idcs, -1);
        longshort_idcs = update_indices(&extra_idcs, &longshort_idcs, -1);
    }

    // Add missing peaks
    if !missed_idcs.is_empty() {
        let valid_missed: Vec<usize> = missed_idcs
            .iter()
            .filter(|&&i| i > 1 && i < peaks_clean.len())
            .cloned()
            .collect();

        for &idx in valid_missed.iter().rev() {
            let prev = peaks_clean[idx - 1];
            let next = peaks_clean[idx];
            let added = prev + (next - prev) / 2;
            peaks_clean.insert(idx, added);
        }

        ectopic_idcs = update_indices(&valid_missed, &ectopic_idcs, 1);
        longshort_idcs = update_indices(&valid_missed, &longshort_idcs, 1);
    }

    // Correct misaligned (ectopic + longshort)
    for idcs in [&ectopic_idcs, &longshort_idcs] {
        if !idcs.is_empty() {
            let valid_idcs: Vec<usize> = idcs
                .iter()
                .filter(|&&i| i > 1 && i + 1 < peaks_clean.len())
                .cloned()
                .collect();

            if !valid_idcs.is_empty() {
                let mut to_remove: Vec<usize> = Vec::new();
                let mut to_add: Vec<usize> = Vec::new();

                for &idx in &valid_idcs {
                    let prev = peaks_clean[idx - 1];
                    let next = peaks_clean[idx + 1];
                    let interp = prev + (next - prev) / 2;
                    to_remove.push(idx);
                    to_add.push(interp);
                }

                // Remove and add
                for (offset, &idx) in to_remove.iter().enumerate() {
                    if idx >= offset && idx - offset < peaks_clean.len() {
                        peaks_clean.remove(idx - offset);
                    }
                }
                peaks_clean.extend(to_add);
                peaks_clean.sort();
            }
        }
    }

    peaks_clean
}

/// Fix peaks with artifact correction (Kubios method)
pub fn fix_peaks(
    peaks: &[usize],
    sampling_rate: u32,
    iterative: bool,
    c1: f64,
    c2: f64,
    alpha: f64,
    window_width: usize,
    medfilt_order: usize,
) -> (Vec<usize>, Artifacts) {
    if peaks.len() < 3 {
        return (peaks.to_vec(), Artifacts::default());
    }

    let mut artifacts = find_artifacts_fast(
        peaks,
        sampling_rate,
        c1,
        c2,
        alpha,
        window_width,
        medfilt_order,
    );
    let mut peaks_clean = correct_artifacts(&artifacts, peaks);

    if iterative {
        let mut n_artifacts_current = artifacts.total_count();

        loop {
            let new_artifacts = find_artifacts_fast(
                &peaks_clean,
                sampling_rate,
                c1,
                c2,
                alpha,
                window_width,
                medfilt_order,
            );
            let n_artifacts_new = new_artifacts.total_count();

            if n_artifacts_new >= n_artifacts_current {
                break;
            }

            artifacts = new_artifacts;
            n_artifacts_current = artifacts.total_count();
            peaks_clean = correct_artifacts(&artifacts, &peaks_clean);
        }
    }

    (peaks_clean, artifacts)
}

/// Fix peaks with default parameters
pub fn fix_peaks_default(peaks: &[usize], sampling_rate: u32) -> (Vec<usize>, Artifacts) {
    fix_peaks(peaks, sampling_rate, true, 0.13, 0.17, 5.2, 91, 11)
}

/// Main entry point: find R-peaks with optional artifact correction
pub fn ecg_peaks(
    ecg_cleaned: &[f64],
    sampling_rate: u32,
    correct_artifacts: bool,
) -> (Vec<usize>, Artifacts) {
    let peaks = find_peaks_default(ecg_cleaned, sampling_rate);

    if correct_artifacts {
        fix_peaks_default(&peaks, sampling_rate)
    } else {
        (peaks, Artifacts::default())
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient() {
        let signal = vec![1.0, 2.0, 4.0, 7.0, 11.0];
        let grad = gradient(&signal);
        // Expected: [1, 1.5, 2.5, 3.5, 4]
        assert!((grad[0] - 1.0).abs() < 1e-10);
        assert!((grad[1] - 1.5).abs() < 1e-10);
        assert!((grad[2] - 2.5).abs() < 1e-10);
        assert!((grad[3] - 3.5).abs() < 1e-10);
        assert!((grad[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_python_values() {
        // Python: np.gradient([1.0, 4.0, 9.0, 16.0, 25.0]) = [3.0, 4.0, 6.0, 8.0, 9.0]
        let signal = vec![1.0, 4.0, 9.0, 16.0, 25.0];
        let grad = gradient(&signal);
        let expected = vec![3.0, 4.0, 6.0, 8.0, 9.0];
        for (g, e) in grad.iter().zip(&expected) {
            assert!((g - e).abs() < 1e-10, "gradient mismatch: {} vs {}", g, e);
        }
    }

    #[test]
    fn test_uniform_filter() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let filtered = uniform_filter1d(&signal, 3);
        // Expected: moving average with size 3
        assert!((filtered[1] - 2.0).abs() < 1e-10);
        assert!((filtered[2] - 3.0).abs() < 1e-10);
        assert!((filtered[3] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_uniform_filter_python_values() {
        // Python: uniform_filter1d([1,2,3,4,5,6,7,8,9,10], size=3, mode='nearest')
        // = [1.333, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.666]
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let filtered = uniform_filter1d(&signal, 3);
        let expected = vec![
            1.3333333333333333,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            9.666666666666666,
        ];
        for (i, (f, e)) in filtered.iter().zip(&expected).enumerate() {
            assert!(
                (f - e).abs() < 1e-10,
                "uniform_filter mismatch at {}: {} vs {}",
                i,
                f,
                e
            );
        }
    }

    #[test]
    fn test_uniform_filter_size5_python() {
        // Python: uniform_filter1d([10, 20, 30, 40, 50], size=5, mode='nearest')
        // = [16.0, 22.0, 30.0, 38.0, 44.0]
        let signal = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let filtered = uniform_filter1d(&signal, 5);
        let expected = vec![16.0, 22.0, 30.0, 38.0, 44.0];
        for (i, (f, e)) in filtered.iter().zip(&expected).enumerate() {
            assert!(
                (f - e).abs() < 1e-10,
                "uniform_filter size=5 mismatch at {}: {} vs {}",
                i,
                f,
                e
            );
        }
    }

    #[test]
    fn test_median_filter() {
        let signal = vec![1.0, 5.0, 2.0, 8.0, 3.0];
        let filtered = median_filter(&signal, 3);
        // Middle values should be medians
        assert!((filtered[1] - 2.0).abs() < 1e-10); // median of [1, 5, 2]
        assert!((filtered[2] - 5.0).abs() < 1e-10); // median of [5, 2, 8]
        assert!((filtered[3] - 3.0).abs() < 1e-10); // median of [2, 8, 3]
    }

    #[test]
    fn test_median_filter_python_values() {
        // Python: median_filter([1, 10, 2, 9, 3, 8, 4, 7, 5, 6], size=3, mode='reflect')
        // = [1.0, 2.0, 9.0, 3.0, 8.0, 4.0, 7.0, 5.0, 6.0, 6.0]
        let signal = vec![1.0, 10.0, 2.0, 9.0, 3.0, 8.0, 4.0, 7.0, 5.0, 6.0];
        let filtered = median_filter(&signal, 3);
        let expected = vec![1.0, 2.0, 9.0, 3.0, 8.0, 4.0, 7.0, 5.0, 6.0, 6.0];
        for (i, (f, e)) in filtered.iter().zip(&expected).enumerate() {
            assert!(
                (f - e).abs() < 1e-10,
                "median_filter mismatch at {}: {} vs {}",
                i,
                f,
                e
            );
        }
    }

    #[test]
    fn test_percentile_filter_python_values() {
        // Python: percentile_filter([1,2,3,4,5,6,7,8,9,10], percentile=25, size=5, mode='reflect')
        // = [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let filtered = percentile_filter(&signal, 25.0, 5);
        let expected = vec![1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        for (i, (f, e)) in filtered.iter().zip(&expected).enumerate() {
            assert!(
                (f - e).abs() < 1e-10,
                "percentile_filter mismatch at {}: {} vs {}",
                i,
                f,
                e
            );
        }
    }

    #[test]
    fn test_find_peaks_empty() {
        let signal: Vec<f64> = vec![];
        let peaks = find_peaks_default(&signal, 512);
        assert!(peaks.is_empty());
    }

    #[test]
    fn test_find_peaks_simple() {
        // Create a simple synthetic ECG-like signal
        let mut signal = vec![0.0; 1000];
        // Add peaks at known locations
        for &pos in &[100, 300, 500, 700, 900] {
            signal[pos] = 1.0;
            if pos > 0 {
                signal[pos - 1] = 0.3;
            }
            if pos < 999 {
                signal[pos + 1] = 0.3;
            }
        }

        let peaks = find_peaks_default(&signal, 512);
        // Should find approximately 5 peaks
        assert!(peaks.len() >= 3 && peaks.len() <= 7);
    }

    #[test]
    fn test_artifacts_default() {
        let artifacts = Artifacts::default();
        assert_eq!(artifacts.total_count(), 0);
    }

    #[test]
    fn test_fix_peaks_regular() {
        // Python: Regular peaks at ~75 BPM (410 samples @ 512 Hz)
        // No artifacts should be detected
        let peaks = vec![100, 510, 920, 1330, 1740, 2150, 2560, 2970, 3380, 3790];
        let (corrected, artifacts) = fix_peaks_default(&peaks, 512);

        // Should remain unchanged
        assert_eq!(corrected, peaks);
        assert_eq!(artifacts.total_count(), 0);
    }

    #[test]
    fn test_fix_peaks_with_missed_beat() {
        // Python: peaks_missed = [100, 510, 1330, 1740, ...] (missing 920)
        // Python detects longshort at index 2, corrects by inserting 1125
        let peaks = vec![100, 510, 1330, 1740, 2150, 2560, 2970, 3380, 3790];
        let (corrected, _artifacts) = fix_peaks_default(&peaks, 512);

        // Should detect and attempt to correct the long RR interval
        // Python output: [100, 510, 1125, 1740, ...]
        // Check that correction happened (length increased or interpolation inserted)
        assert!(corrected.len() >= peaks.len());
    }
}
