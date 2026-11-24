//! ECG Peak Detection and Artifact Correction
//!
//! Fast R-peak detection using NeuroKit algorithm with Rust optimization.
//! Artifact correction using Kubios method (Lipponen & Tarvainen 2019).

use crate::error::HrvError;
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
// ARTIFACT CORRECTION (Kubios / Lipponen & Tarvainen 2019, NeuroKit2 `signal_fixpeaks`)
// =============================================================================

/// Artifact indices found by the Kubios method.
///
/// From a single detection pass ([`detect_artifacts`]) the indices refer to the input peaks. From
/// [`fix_peaks`] / [`fix_peaks_with`] the lists are **cumulative over all correction passes**:
/// each index refers to the peak array as it was at the start of the pass that found it
/// (see [`KubiosOutput::passes`] for the per-pass split).
///
/// These are **flags**, not corrected beats: see [`Artifacts::total_count`]. For the share of beats
/// that the correction actually changed use [`Corrections`] ([`KubiosOutput::corrections`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Artifacts {
    pub ectopic: Vec<usize>,
    pub missed: Vec<usize>,
    pub extra: Vec<usize>,
    pub longshort: Vec<usize>,
}

impl Artifacts {
    /// Number of artifact **flags** (ectopic + missed + extra + long/short).
    ///
    /// On the cumulative artifacts of [`fix_peaks_with`] this sums the flags of **every** correction
    /// pass, including flags whose correction changed nothing (a beat moved to where it already was,
    /// or moved in one pass and back in the next) and the same beat flagged again in a later pass.
    /// It is therefore **not** a corrected-beat count, and `total_count() / n_beats` is **not** a
    /// corrected-beat ratio (on real wearable data it was 2–5× the share of beats actually changed).
    /// For gating on correction use [`Corrections::changed_count`] / [`Corrections::changed_ratio`]
    /// (also [`KubiosOutput::changed_ratio`]).
    pub fn total_count(&self) -> usize {
        self.ectopic.len() + self.missed.len() + self.extra.len() + self.longshort.len()
    }

    fn append(&mut self, other: &Artifacts) {
        self.ectopic.extend_from_slice(&other.ectopic);
        self.missed.extend_from_slice(&other.missed);
        self.extra.extend_from_slice(&other.extra);
        self.longshort.extend_from_slice(&other.longshort);
    }
}

/// What the Kubios correction did to one **output** beat, decided by comparing the final output
/// with the input (not from the flags).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BeatStatus {
    /// Same position as an input beat (bit-identical). A beat that was flagged but ends where it
    /// started (e.g. moved in one pass and back in a later one) is `Unchanged`.
    Unchanged,
    /// An input beat shifted by an ectopic or long/short correction.
    Moved,
    /// A beat added by a missed-beat correction (no input beat).
    Inserted,
}

/// Per-beat result of a Kubios correction: what really changed between input and output.
///
/// How it is decided: an output beat whose position equals an input position exactly is
/// [`BeatStatus::Unchanged`] (input and output are both sorted; equal positions are paired in
/// order). Each remaining output beat is [`BeatStatus::Moved`] if it descends from an input beat
/// (tracked through all passes) that is not already paired, otherwise [`BeatStatus::Inserted`].
/// Input beats left without an output beat are **removed** (extra-beat corrections).
///
/// Ratio definition (for gating): [`Corrections::changed_ratio`] = (moved + inserted) / number of
/// **output** beats. Removed beats have no output beat and are reported separately
/// ([`Corrections::removed`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Corrections {
    /// Status of each output beat (same length as the corrected peaks).
    pub status: Vec<BeatStatus>,
    /// For each input beat, the index of its output beat, or `None` if it was removed.
    pub input_to_output: Vec<Option<usize>>,
    /// Number of input beats removed (extra beats); they have no output beat.
    pub removed: usize,
}

impl Corrections {
    /// No correction: every one of `n` beats unchanged.
    fn identity(n: usize) -> Self {
        Self {
            status: vec![BeatStatus::Unchanged; n],
            input_to_output: (0..n).map(Some).collect(),
            removed: 0,
        }
    }

    fn count(&self, s: BeatStatus) -> usize {
        self.status.iter().filter(|&&x| x == s).count()
    }

    /// Number of output beats.
    pub fn n_output(&self) -> usize {
        self.status.len()
    }

    /// Number of input beats.
    pub fn n_input(&self) -> usize {
        self.input_to_output.len()
    }

    /// Output beats moved by an ectopic or long/short correction.
    pub fn moved_count(&self) -> usize {
        self.count(BeatStatus::Moved)
    }

    /// Output beats inserted by a missed-beat correction.
    pub fn inserted_count(&self) -> usize {
        self.count(BeatStatus::Inserted)
    }

    /// Input beats removed by an extra-beat correction (same as [`Corrections::removed`]).
    pub fn removed_count(&self) -> usize {
        self.removed
    }

    /// Output beats actually changed: moved + inserted (removed beats are not included, see
    /// [`Corrections::removed_count`]).
    pub fn changed_count(&self) -> usize {
        self.status
            .iter()
            .filter(|&&s| s != BeatStatus::Unchanged)
            .count()
    }

    /// Share of output beats actually changed: `(moved + inserted) / n_output`, in `[0, 1]`.
    ///
    /// This is the corrected-beat ratio to gate on (not [`Artifacts::total_count`], which counts
    /// flags over all passes). Removed beats are reported separately ([`Corrections::removed_count`]).
    /// `NaN` when there is no output beat.
    pub fn changed_ratio(&self) -> f64 {
        if self.status.is_empty() {
            f64::NAN
        } else {
            self.changed_count() as f64 / self.status.len() as f64
        }
    }

    /// `true` for each output beat that is moved or inserted.
    pub fn changed_mask(&self) -> Vec<bool> {
        self.status
            .iter()
            .map(|&s| s != BeatStatus::Unchanged)
            .collect()
    }

    /// For each output beat, the index of the input beat it is (unchanged) or comes from (moved);
    /// `None` for inserted beats. Inverse of [`Corrections::input_to_output`].
    pub fn output_to_input(&self) -> Vec<Option<usize>> {
        let mut out = vec![None; self.status.len()];
        for (i, o) in self.input_to_output.iter().enumerate() {
            if let Some(j) = *o {
                out[j] = Some(i);
            }
        }
        out
    }
}

/// Parameters of the Kubios artifact detection (defaults as in Lipponen & Tarvainen 2019 / NeuroKit2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KubiosParams {
    pub c1: f64,
    pub c2: f64,
    /// Threshold multiplier of the quartile deviation.
    pub alpha: f64,
    /// Width (beats) of the rolling quartile window.
    pub window_width: usize,
    /// Width (beats) of the rolling median.
    pub medfilt_order: usize,
    /// Repeat detection + correction while the number of artifacts decreases.
    pub iterative: bool,
}

impl Default for KubiosParams {
    fn default() -> Self {
        Self {
            c1: 0.13,
            c2: 0.17,
            alpha: 5.2,
            window_width: 91,
            medfilt_order: 11,
            iterative: true,
        }
    }
}

/// Result of [`fix_peaks_with`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct KubiosOutput {
    /// Corrected peak indices.
    pub peaks: Vec<usize>,
    /// Artifact flags of all applied passes, concatenated. `artifacts.total_count()` counts flags,
    /// not corrected beats; see [`KubiosOutput::corrections`].
    pub artifacts: Artifacts,
    /// Artifacts of each applied pass (indices refer to the peaks at the start of that pass).
    pub passes: Vec<Artifacts>,
    /// Per-beat result (unchanged / moved / inserted, removed count, input → output mapping).
    pub corrections: Corrections,
}

impl KubiosOutput {
    /// Output beats actually changed (moved + inserted); see [`Corrections::changed_count`].
    pub fn changed_count(&self) -> usize {
        self.corrections.changed_count()
    }

    /// Share of output beats actually changed; see [`Corrections::changed_ratio`].
    pub fn changed_ratio(&self) -> f64 {
        self.corrections.changed_ratio()
    }
}

impl AsRef<Corrections> for KubiosOutput {
    fn as_ref(&self) -> &Corrections {
        &self.corrections
    }
}

/// Where [`fix_positions_with_midpoint`] places inserted (missed) and moved (ectopic, long/short)
/// beats: at the midpoint of the neighbours, `prev + (next − prev) / 2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KubiosMidpoint {
    /// Exact midpoint, no rounding (the default of [`fix_positions_with`]).
    #[default]
    Exact,
    /// `prev + floor((next − prev) / 2)`, the integer rule of [`fix_peaks_with`] and NeuroKit2.
    /// On integer-valued positions the result is bit-identical to [`fix_peaks_with`].
    Floor,
}

/// Result of [`fix_positions_with`]: like [`KubiosOutput`] but with `f64` sample positions.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct KubiosPositions {
    /// Corrected positions (samples, or whatever unit the input was in). Unchanged beats keep their
    /// input value bit for bit.
    pub positions: Vec<f64>,
    /// Artifact flags of all applied passes (flags, not corrected beats).
    pub artifacts: Artifacts,
    /// Artifacts of each applied pass.
    pub passes: Vec<Artifacts>,
    /// Per-beat result (unchanged / moved / inserted, removed count, input → output mapping).
    pub corrections: Corrections,
}

impl KubiosPositions {
    /// Output beats actually changed (moved + inserted); see [`Corrections::changed_count`].
    pub fn changed_count(&self) -> usize {
        self.corrections.changed_count()
    }

    /// Share of output beats actually changed; see [`Corrections::changed_ratio`].
    pub fn changed_ratio(&self) -> f64 {
        self.corrections.changed_ratio()
    }

    /// NN intervals (ms) of the corrected positions: [`crate::report::rri_from_positions`].
    pub fn rri_ms(&self, sampling_rate: f64) -> Vec<f64> {
        crate::report::rri_from_positions(&self.positions, sampling_rate)
    }
}

impl AsRef<Corrections> for KubiosPositions {
    fn as_ref(&self) -> &Corrections {
        &self.corrections
    }
}

/// NumPy's pairwise summation (`np.sum` / `np.mean` on float64), for bit-level parity.
pub(crate) fn numpy_sum(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 8 {
        let mut s = 0.0;
        for &v in x {
            s += v;
        }
        s
    } else if n <= 128 {
        let mut r = [0.0; 8];
        r.copy_from_slice(&x[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for k in 0..8 {
                r[k] += x[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += x[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        numpy_sum(&x[..n2]) + numpy_sum(&x[n2..])
    }
}

/// Window bounds of pandas `rolling(window, center=True)` at position `i`, clipped to `[0, n)`.
fn centered_bounds(i: usize, n: usize, window: usize) -> (usize, usize) {
    let offset = (window.saturating_sub(1)) / 2;
    let end = (i + 1 + offset).min(n);
    let start = (i + 1 + offset).saturating_sub(window).min(n);
    (start, end)
}

/// Sliding sorted window over the non-NaN values of `x`, evaluated with `stat` at every position,
/// for pandas `rolling(window, center=True, min_periods=1)` semantics (shrinking windows at the
/// edges; NaN values are skipped; NaN if a window has no valid value).
fn rolling_centered(x: &[f64], window: usize, stat: impl Fn(&[f64]) -> f64) -> Vec<f64> {
    let n = x.len();
    let window = window.max(1);
    let mut sorted: Vec<f64> = Vec::with_capacity(window);
    let (mut lo, mut hi) = (0usize, 0usize);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let (s, e) = centered_bounds(i, n, window);
        while hi < e {
            let v = x[hi];
            if !v.is_nan() {
                let pos = sorted.partition_point(|&w| w < v);
                sorted.insert(pos, v);
            }
            hi += 1;
        }
        while lo < s {
            let v = x[lo];
            if !v.is_nan() {
                let pos = sorted.partition_point(|&w| w < v);
                sorted.remove(pos);
            }
            lo += 1;
        }
        out.push(if sorted.is_empty() {
            f64::NAN
        } else {
            stat(&sorted)
        });
    }
    out
}

/// pandas `Series.rolling(window, center=True, min_periods=1).quantile(q)` (linear interpolation).
pub fn rolling_quantile_centered(x: &[f64], window: usize, q: f64) -> Vec<f64> {
    rolling_centered(x, window, |s| {
        let nobs = s.len();
        if nobs == 1 {
            return s[0];
        }
        let pos = q * (nobs - 1) as f64;
        let idx = pos as usize;
        if idx as f64 == pos {
            s[idx]
        } else {
            let (vlow, vhigh) = (s[idx], s[idx + 1]);
            vlow + (vhigh - vlow) * (pos - idx as f64)
        }
    })
}

/// pandas `Series.rolling(window, center=True, min_periods=1).median()`.
pub fn rolling_median_centered(x: &[f64], window: usize) -> Vec<f64> {
    rolling_centered(x, window, |s| {
        let nobs = s.len();
        let mid = nobs / 2;
        if nobs % 2 == 1 {
            s[mid]
        } else {
            (s[mid] + s[mid - 1]) / 2.0
        }
    })
}

/// NeuroKit `_compute_threshold`: `alpha * (Q3 - Q1) / 2` of `|signal|` over a centred rolling window.
fn compute_threshold(signal: &[f64], alpha: f64, window_width: usize) -> Vec<f64> {
    let abs_signal: Vec<f64> = signal.iter().map(|x| x.abs()).collect();
    let q1 = rolling_quantile_centered(&abs_signal, window_width, 0.25);
    let q3 = rolling_quantile_centered(&abs_signal, window_width, 0.75);
    q1.iter()
        .zip(&q3)
        .map(|(&a, &b)| alpha * ((b - a) / 2.0))
        .collect()
}

/// Index into `np.pad(x, pad, "reflect")` (reflection without repeating the edge sample).
fn reflect_index(i: isize, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let period = 2 * (n as isize - 1);
    let mut k = i.rem_euclid(period);
    if k >= n as isize {
        k = period - k;
    }
    k as usize
}

/// `np.max` / `np.min` of two values (NaN-propagating, unlike `f64::max`).
fn nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}
fn nan_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

/// Subspaces s12 and s22 (NeuroKit: `np.pad(drrs, 2, "reflect")`).
fn compute_subspaces(drrs: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = drrs.len();
    let at = |i: isize| drrs[reflect_index(i, n)];
    let mut s12 = vec![0.0; n];
    let mut s22 = vec![0.0; n];
    for d in 0..n {
        let i = d as isize;
        let cur = drrs[d];
        if cur > 0.0 {
            s12[d] = nan_max(at(i - 1), at(i + 1));
        } else if cur < 0.0 {
            s12[d] = nan_min(at(i - 1), at(i + 1));
        }
        if cur >= 0.0 {
            s22[d] = nan_min(at(i + 1), at(i + 2));
        } else if cur < 0.0 {
            s22[d] = nan_max(at(i + 1), at(i + 2));
        }
    }
    (s12, s22)
}

/// Classify artifacts (decision flow of Lipponen & Tarvainen 2019, Figure 1, as in NeuroKit2).
#[allow(clippy::too_many_arguments)]
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
    while i + 2 < n {
        if drrs[i].abs() <= 1.0 {
            i += 1;
            continue;
        }

        // Ectopic
        let eq1 = drrs[i] > 1.0 && s12[i] < (-c1 * drrs[i] - c2);
        let eq2 = drrs[i] < -1.0 && s12[i] > (-c1 * drrs[i] + c2);
        if eq1 || eq2 {
            artifacts.ectopic.push(i);
            i += 1;
            continue;
        }

        if !(drrs[i].abs() > 1.0 || mrrs[i].abs() > 3.0) {
            i += 1;
            continue;
        }

        // Long/short candidates: i, and i+1 if |dRR[i+1]| < |dRR[i+2]|
        let mut candidates = [i, usize::MAX];
        if drrs[i + 1].abs() < drrs[i + 2].abs() {
            candidates[1] = i + 1;
        }

        // Every candidate advances `i` by exactly one (no extra increment after the loop).
        for &j in candidates.iter().filter(|&&j| j != usize::MAX) {
            let eq3 = drrs[j] > 1.0 && s22[j] < -1.0;
            let eq4 = mrrs[j].abs() > 3.0;
            let eq5 = drrs[j] < -1.0 && s22[j] > 1.0;
            if !(eq3 || eq4 || eq5) {
                i += 1;
                continue;
            }
            let eq6 = (rr[j] / 2.0 - medrr[j]).abs() < th2[j];
            let eq7 = (rr[j] + rr[j + 1] - medrr[j]).abs() < th2[j];
            if eq5 && eq7 {
                artifacts.extra.push(j);
            } else if eq3 && eq6 {
                artifacts.missed.push(j);
            } else {
                artifacts.longshort.push(j);
            }
            i += 1;
        }
    }

    artifacts
}

/// Peak position type of the generic Kubios core: integer sample indices (`usize`) or `f64`
/// positions. Only RR (`as_f64` differences) and ordering are needed; the midpoint rule is passed
/// separately.
trait PeakPos: Copy {
    fn as_f64(self) -> f64;
    fn cmp_pos(&self, other: &Self) -> std::cmp::Ordering;
}

impl PeakPos for usize {
    #[inline]
    fn as_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn cmp_pos(&self, other: &Self) -> std::cmp::Ordering {
        self.cmp(other)
    }
}

impl PeakPos for f64 {
    #[inline]
    fn as_f64(self) -> f64 {
        self
    }
    #[inline]
    fn cmp_pos(&self, other: &Self) -> std::cmp::Ordering {
        self.total_cmp(other)
    }
}

/// One Kubios detection pass over strictly increasing `peaks` (NeuroKit2 `_find_artifacts`).
///
/// Rolling quartiles and median follow pandas `rolling(..., center=True, min_periods=1)` exactly
/// (shrinking windows at the series edges, linear interpolation), so short series (< 45 beats)
/// behave like NeuroKit2.
pub fn detect_artifacts(peaks: &[usize], sampling_rate: f64, params: &KubiosParams) -> Artifacts {
    detect_core(peaks, sampling_rate, params)
}

/// [`detect_artifacts`] on strictly increasing `f64` positions (samples; RR = difference / fs).
/// On integer-valued positions the result is identical to [`detect_artifacts`].
pub fn detect_artifacts_positions(
    positions: &[f64],
    sampling_rate: f64,
    params: &KubiosParams,
) -> Artifacts {
    detect_core(positions, sampling_rate, params)
}

fn detect_core<T: PeakPos>(peaks: &[T], sampling_rate: f64, params: &KubiosParams) -> Artifacts {
    let n = peaks.len();
    if n < 3 || !sampling_rate.is_finite() || sampling_rate <= 0.0 {
        return Artifacts::default();
    }
    let mut rr = vec![0.0; n];
    for i in 1..n {
        rr[i] = (peaks[i].as_f64() - peaks[i - 1].as_f64()) / sampling_rate;
    }
    rr[0] = numpy_sum(&rr[1..]) / (n - 1) as f64;

    let mut drrs = vec![0.0; n];
    for i in 1..n {
        drrs[i] = rr[i] - rr[i - 1];
    }
    drrs[0] = numpy_sum(&drrs[1..]) / (n - 1) as f64;

    let th1 = compute_threshold(&drrs, params.alpha, params.window_width);
    let drrs: Vec<f64> = drrs
        .iter()
        .zip(&th1)
        .map(|(&d, &t)| if t != 0.0 { d / t } else { f64::NAN })
        .collect();
    let (s12, s22) = compute_subspaces(&drrs);

    let medrr = rolling_median_centered(&rr, params.medfilt_order);
    let mrrs: Vec<f64> = rr
        .iter()
        .zip(&medrr)
        .map(|(&r, &m)| {
            let d = r - m;
            if d < 0.0 { d * 2.0 } else { d }
        })
        .collect();
    let th2 = compute_threshold(&mrrs, params.alpha, params.window_width);
    let mrrs: Vec<f64> = mrrs
        .iter()
        .zip(&th2)
        .map(|(&m, &t)| if t != 0.0 { m / t } else { f64::NAN })
        .collect();

    classify_artifacts(
        &drrs, &mrrs, &s12, &s22, &rr, &medrr, &th2, params.c1, params.c2,
    )
}

/// Shift every index in `update_idcs` by `update` for each source index below it.
///
/// Compared against the original indices; NeuroKit2 compares against already-updated values,
/// which differs only for adjacent source indices (e.g. extra = [5, 6], u = 7 → 5 here, 6 in NK).
fn update_indices(source_idcs: &[usize], update_idcs: &[usize], update: isize) -> Vec<usize> {
    if update_idcs.is_empty() {
        return vec![];
    }
    let mut result: Vec<usize> = update_idcs
        .iter()
        .map(|&u| {
            let shift = source_idcs.iter().filter(|&&s| u > s).count() as isize * update;
            (u as isize + shift).max(0) as usize
        })
        .collect();
    result.sort_unstable();
    result.dedup();
    result
}

/// Remove the (sorted, unique) indices `idcs` from `v`.
fn remove_indices<X>(v: &mut Vec<X>, idcs: &[usize]) {
    let mut k = 0;
    let mut i = 0;
    v.retain(|_| {
        let drop = k < idcs.len() && idcs[k] == i;
        if drop {
            k += 1;
        }
        i += 1;
        !drop
    });
}

/// Origin marker of a beat that has no input beat (inserted by a missed-beat correction).
const INSERTED: usize = usize::MAX;

/// Correct detected artifacts (NeuroKit2 `_correct_artifacts`): delete extra beats, insert missed
/// beats, move ectopic and long/short beats to the midpoint of their neighbours (`mid(prev, next)`).
///
/// Each beat carries the index of the input beat it descends from (`INSERTED` for new beats); a
/// moved beat keeps its origin.
fn correct_core<T: PeakPos>(
    artifacts: &Artifacts,
    peaks: &[(T, usize)],
    mid: impl Fn(T, T) -> T,
) -> Vec<(T, usize)> {
    let mut peaks_clean: Vec<(T, usize)> = peaks.to_vec();

    let mut extra = artifacts.extra.clone();
    extra.sort_unstable();
    extra.dedup();
    let mut missed = artifacts.missed.clone();
    let mut ectopic = artifacts.ectopic.clone();
    let mut longshort = artifacts.longshort.clone();

    if !extra.is_empty() {
        extra.retain(|&i| i < peaks_clean.len());
        remove_indices(&mut peaks_clean, &extra);
        missed = update_indices(&extra, &missed, -1);
        ectopic = update_indices(&extra, &ectopic, -1);
        longshort = update_indices(&extra, &longshort, -1);
    }

    if !missed.is_empty() {
        let mut valid: Vec<usize> = missed
            .iter()
            .copied()
            .filter(|&i| i > 1 && i < peaks_clean.len())
            .collect();
        valid.sort_unstable();
        valid.dedup();
        for &idx in valid.iter().rev() {
            let prev = peaks_clean[idx - 1].0;
            let next = peaks_clean[idx].0;
            peaks_clean.insert(idx, (mid(prev, next), INSERTED));
        }
        ectopic = update_indices(&valid, &ectopic, 1);
        longshort = update_indices(&valid, &longshort, 1);
    }

    for idcs in [&ectopic, &longshort] {
        let mut valid: Vec<usize> = idcs
            .iter()
            .copied()
            .filter(|&i| i > 1 && i + 1 < peaks_clean.len())
            .collect();
        valid.sort_unstable();
        valid.dedup();
        if valid.is_empty() {
            continue;
        }
        let interp: Vec<(T, usize)> = valid
            .iter()
            .map(|&idx| {
                let (prev, next) = (peaks_clean[idx - 1].0, peaks_clean[idx + 1].0);
                (mid(prev, next), peaks_clean[idx].1)
            })
            .collect();
        remove_indices(&mut peaks_clean, &valid);
        peaks_clean.extend(interp);
        // stable sort by position (as the previous integer `sort()`)
        peaks_clean.sort_by(|a, b| a.0.cmp_pos(&b.0));
    }

    peaks_clean
}

/// Per-beat status of `output` (positions + origins) against `input` (see [`Corrections`]).
fn compute_corrections<T: PeakPos>(input: &[T], output: &[(T, usize)]) -> Corrections {
    let mut input_to_output: Vec<Option<usize>> = vec![None; input.len()];
    let mut status = vec![BeatStatus::Inserted; output.len()];
    // 1. exact position matches (both sequences sorted): unchanged
    let (mut i, mut j) = (0, 0);
    while i < input.len() && j < output.len() {
        match input[i].cmp_pos(&output[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                input_to_output[i] = Some(j);
                status[j] = BeatStatus::Unchanged;
                i += 1;
                j += 1;
            }
        }
    }
    // 2. remaining output beats descending from a still unpaired input beat: moved
    for (j, &(_, origin)) in output.iter().enumerate() {
        if status[j] == BeatStatus::Unchanged || origin == INSERTED {
            continue;
        }
        if input_to_output[origin].is_none() {
            input_to_output[origin] = Some(j);
            status[j] = BeatStatus::Moved;
        }
    }
    let removed = input_to_output.iter().filter(|o| o.is_none()).count();
    Corrections {
        status,
        input_to_output,
        removed,
    }
}

/// Generic Kubios loop (validated input): detection + correction, repeated while the number of
/// flags decreases (`params.iterative`). Returns (positions, cumulative flags, per-pass flags,
/// per-beat corrections).
fn fix_core<T: PeakPos>(
    peaks: &[T],
    sampling_rate: f64,
    params: &KubiosParams,
    mid: impl Fn(T, T) -> T + Copy,
) -> (Vec<T>, Artifacts, Vec<Artifacts>, Corrections) {
    if peaks.len() < 3 {
        return (
            peaks.to_vec(),
            Artifacts::default(),
            Vec::new(),
            Corrections::identity(peaks.len()),
        );
    }
    let tagged: Vec<(T, usize)> = peaks.iter().copied().zip(0..).collect();
    let first = detect_core(peaks, sampling_rate, params);
    let mut clean = correct_core(&first, &tagged, mid);
    let mut current = first.total_count();
    let mut passes = vec![first];

    if params.iterative {
        let mut values: Vec<T> = Vec::with_capacity(clean.len());
        loop {
            values.clear();
            values.extend(clean.iter().map(|p| p.0));
            let next = detect_core(&values, sampling_rate, params);
            let count = next.total_count();
            if count >= current {
                break;
            }
            current = count;
            clean = correct_core(&next, &clean, mid);
            passes.push(next);
        }
    }

    let mut artifacts = Artifacts::default();
    for p in &passes {
        artifacts.append(p);
    }
    let corrections = compute_corrections(peaks, &clean);
    (
        clean.into_iter().map(|p| p.0).collect(),
        artifacts,
        passes,
        corrections,
    )
}

fn check_strictly_increasing(peaks: &[usize]) -> Result<(), HrvError> {
    match peaks.windows(2).position(|w| w[1] <= w[0]) {
        Some(i) => Err(HrvError::UnsortedPeaks { index: i + 1 }),
        None => Ok(()),
    }
}

/// Kubios artifact detection and correction (NeuroKit2 `signal_fixpeaks(method="Kubios")`).
///
/// `peaks` must be strictly increasing (otherwise [`HrvError::UnsortedPeaks`]). Returns the corrected
/// peaks, the artifact flags of **every** applied pass (cumulative), not only of the last one, and
/// the per-beat [`Corrections`]. Gate on [`KubiosOutput::changed_ratio`] (beats actually changed),
/// not on `artifacts.total_count()` (flags over all passes).
/// Note: one call treats `peaks` as one continuous recording; for recordings with gaps use
/// [`crate::gaps::fix_peaks_segmented`] so that no beats are inserted into a gap. For sub-sample
/// positions use [`fix_positions_with`].
pub fn fix_peaks_with(
    peaks: &[usize],
    sampling_rate: f64,
    params: &KubiosParams,
) -> Result<KubiosOutput, HrvError> {
    check_strictly_increasing(peaks)?;
    if !(sampling_rate.is_finite() && sampling_rate > 0.0) {
        return Err(HrvError::InvalidSamplingRate(sampling_rate));
    }
    let (peaks, artifacts, passes, corrections) =
        fix_core(peaks, sampling_rate, params, |prev: usize, next: usize| {
            prev + (next - prev) / 2
        });
    Ok(KubiosOutput {
        peaks,
        artifacts,
        passes,
        corrections,
    })
}

/// Kubios correction on sub-sample `f64` positions (e.g. [`crate::detect::RefinedPeaks::positions`]
/// or [`crate::stream::StreamPeak::position`]), without rounding them to integers first.
///
/// Same Lipponen–Tarvainen algorithm as [`fix_peaks_with`], with RR taken from the positions
/// (`(p[i] − p[i−1]) / sampling_rate`). Inserted and moved beats are placed at the **exact**
/// midpoint of their neighbours, `prev + (next − prev) / 2` (the integer path floors it); unchanged
/// beats keep their input positions bit for bit. Positions must be finite
/// ([`HrvError::NonFinite`]) and strictly increasing ([`HrvError::UnsortedPeaks`]).
///
/// Positions may be in any unit as long as `sampling_rate` converts them to seconds; times in
/// seconds work with `sampling_rate = 1.0`.
///
/// Relation to the integer path: with [`KubiosMidpoint::Floor`]
/// ([`fix_positions_with_midpoint`]) integer-valued positions give results bit-identical to
/// [`fix_peaks_with`] (every operation is exact on integers below 2^53). With the default exact
/// midpoint the first detection pass is identical, and a corrected beat can differ from the integer
/// path by half a sample.
pub fn fix_positions_with(
    positions: &[f64],
    sampling_rate: f64,
    params: &KubiosParams,
) -> Result<KubiosPositions, HrvError> {
    fix_positions_with_midpoint(positions, sampling_rate, params, KubiosMidpoint::Exact)
}

/// [`fix_positions_with`] with an explicit placement rule for inserted and moved beats.
pub fn fix_positions_with_midpoint(
    positions: &[f64],
    sampling_rate: f64,
    params: &KubiosParams,
    midpoint: KubiosMidpoint,
) -> Result<KubiosPositions, HrvError> {
    if positions.iter().any(|p| !p.is_finite()) {
        return Err(HrvError::NonFinite);
    }
    if let Some(i) = positions.windows(2).position(|w| w[1] <= w[0]) {
        return Err(HrvError::UnsortedPeaks { index: i + 1 });
    }
    if !(sampling_rate.is_finite() && sampling_rate > 0.0) {
        return Err(HrvError::InvalidSamplingRate(sampling_rate));
    }
    let (positions, artifacts, passes, corrections) = match midpoint {
        KubiosMidpoint::Exact => fix_core(positions, sampling_rate, params, |a: f64, b: f64| {
            a + (b - a) / 2.0
        }),
        KubiosMidpoint::Floor => fix_core(positions, sampling_rate, params, |a: f64, b: f64| {
            a + ((b - a) / 2.0).floor()
        }),
    };
    Ok(KubiosPositions {
        positions,
        artifacts,
        passes,
        corrections,
    })
}

/// Fix peaks with artifact correction (Kubios method).
///
/// **Changed in 0.2.0:** the returned [`Artifacts`] are cumulative over all passes (previously only
/// the last pass); unsorted or duplicated peaks are sorted and de-duplicated first instead of
/// panicking; an invalid sampling rate returns the input unchanged.
#[allow(clippy::too_many_arguments)]
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
    let params = KubiosParams {
        c1,
        c2,
        alpha,
        window_width,
        medfilt_order,
        iterative,
    };
    let mut sorted = peaks.to_vec();
    if check_strictly_increasing(&sorted).is_err() {
        sorted.sort_unstable();
        sorted.dedup();
    }
    match fix_peaks_with(&sorted, sampling_rate as f64, &params) {
        Ok(out) => (out.peaks, out.artifacts),
        Err(_) => (sorted, Artifacts::default()),
    }
}

/// Fix peaks with default parameters
pub fn fix_peaks_default(peaks: &[usize], sampling_rate: u32) -> (Vec<usize>, Artifacts) {
    let p = KubiosParams::default();
    fix_peaks(
        peaks,
        sampling_rate,
        p.iterative,
        p.c1,
        p.c2,
        p.alpha,
        p.window_width,
        p.medfilt_order,
    )
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
        // NeuroKit2 0.3.0 signal_fixpeaks(method="Kubios"): longshort [2], 1330 -> 1125
        let peaks = vec![100, 510, 1330, 1740, 2150, 2560, 2970, 3380, 3790];
        let (corrected, artifacts) = fix_peaks_default(&peaks, 512);
        assert_eq!(
            corrected,
            vec![100, 510, 1125, 1740, 2150, 2560, 2970, 3380, 3790]
        );
        assert_eq!(artifacts.longshort, vec![2]);
        assert_eq!(artifacts.total_count(), 1);
    }

    #[test]
    fn test_fix_peaks_nearly_regular_missed() {
        // NeuroKit2 0.3.0: first pass missed [3], longshort [1]; output below
        let peaks = vec![
            100, 510, 921, 1741, 2150, 2560, 2971, 3380, 3790, 4200, 4611, 5020, 5431,
        ];
        let first = detect_artifacts(&peaks, 512.0, &KubiosParams::default());
        assert_eq!(first.missed, vec![3]);
        assert_eq!(first.longshort, vec![1]);
        let (corrected, _) = fix_peaks_default(&peaks, 512);
        assert_eq!(
            corrected,
            vec![
                100, 510, 921, 1331, 1741, 2150, 2560, 2971, 3380, 3790, 4200, 4611, 5020, 5431
            ]
        );
    }

    #[test]
    fn rolling_matches_pandas_center_min_periods_1() {
        let x = [5.0, 1.0, 4.0, 2.0, 8.0, 7.0, 3.0, 6.0];
        assert_eq!(
            rolling_quantile_centered(&x, 5, 0.25),
            vec![2.5, 1.75, 2.0, 2.0, 3.0, 3.0, 5.25, 4.5]
        );
        assert_eq!(
            rolling_quantile_centered(&x, 5, 0.75),
            vec![4.5, 4.25, 5.0, 7.0, 7.0, 7.0, 7.25, 6.5]
        );
        assert_eq!(
            rolling_median_centered(&x, 4),
            vec![3.0, 4.0, 3.0, 3.0, 5.5, 5.0, 6.5, 6.0]
        );
        // window larger than the series (short series < 45 beats)
        assert_eq!(rolling_quantile_centered(&x, 91, 0.25), vec![2.75; 8]);
    }

    #[test]
    fn numpy_reflect_padding() {
        // np.pad([0,1,2,3], 2, "reflect") = [2,1,0,1,2,3,2,1]
        let idx: Vec<usize> = (-2..6).map(|i| reflect_index(i, 4)).collect();
        assert_eq!(idx, vec![2, 1, 0, 1, 2, 3, 2, 1]);
    }

    /// Structural invariants of `Corrections` against input and output.
    fn check_corrections<T: PeakPos + PartialEq + std::fmt::Debug>(
        input: &[T],
        output: &[T],
        c: &Corrections,
    ) {
        assert_eq!(c.n_input(), input.len());
        assert_eq!(c.n_output(), output.len());
        assert_eq!(
            output.len(),
            input.len() - c.removed_count() + c.inserted_count()
        );
        assert_eq!(c.changed_count(), c.moved_count() + c.inserted_count());
        let mut last = None;
        for (i, o) in c.input_to_output.iter().enumerate() {
            if let Some(j) = *o {
                assert!(last.is_none_or(|l| j > l), "mapping not increasing");
                last = Some(j);
                match c.status[j] {
                    BeatStatus::Unchanged => assert_eq!(input[i], output[j]),
                    BeatStatus::Moved => assert_ne!(input[i], output[j]),
                    BeatStatus::Inserted => panic!("inserted beat mapped to input {i}"),
                }
            }
        }
        let back = c.output_to_input();
        for (j, s) in c.status.iter().enumerate() {
            assert_eq!(back[j].is_none(), *s == BeatStatus::Inserted);
        }
        assert_eq!(
            c.changed_mask().iter().filter(|&&m| m).count(),
            c.changed_count()
        );
    }

    #[test]
    fn corrections_regular_series_unchanged() {
        let peaks = vec![100, 510, 920, 1330, 1740, 2150, 2560, 2970, 3380, 3790];
        let out = fix_peaks_with(&peaks, 512.0, &KubiosParams::default()).unwrap();
        assert_eq!(out.changed_count(), 0);
        assert_eq!(out.changed_ratio(), 0.0);
        assert_eq!(out.corrections, Corrections::identity(peaks.len()));
        check_corrections(&peaks, &out.peaks, &out.corrections);
    }

    #[test]
    fn corrections_moved_beat() {
        // NeuroKit2: longshort [2], 1330 -> 1125
        let peaks = vec![100, 510, 1330, 1740, 2150, 2560, 2970, 3380, 3790];
        let out = fix_peaks_with(&peaks, 512.0, &KubiosParams::default()).unwrap();
        let c = &out.corrections;
        assert_eq!(c.moved_count(), 1);
        assert_eq!(c.status[2], BeatStatus::Moved);
        assert_eq!(c.input_to_output[2], Some(2));
        assert_eq!((c.inserted_count(), c.removed_count()), (0, 0));
        assert_eq!(out.changed_ratio(), 1.0 / 9.0);
        check_corrections(&peaks, &out.peaks, c);
    }

    #[test]
    fn corrections_ignore_flags_that_change_nothing() {
        // first pass: missed [3] and longshort [1]; the long/short "correction" of beat 1 puts it
        // back where it was, so only the inserted beat is a change.
        let peaks = vec![
            100, 510, 921, 1741, 2150, 2560, 2971, 3380, 3790, 4200, 4611, 5020, 5431,
        ];
        let out = fix_peaks_with(&peaks, 512.0, &KubiosParams::default()).unwrap();
        assert!(out.artifacts.total_count() >= 2);
        let c = &out.corrections;
        assert_eq!(c.inserted_count(), 1);
        assert_eq!(c.status[3], BeatStatus::Inserted);
        assert_eq!(c.status[1], BeatStatus::Unchanged);
        assert_eq!((c.moved_count(), c.removed_count()), (0, 0));
        assert_eq!(out.changed_count(), 1);
        assert_eq!(c.input_to_output[3], Some(4));
        check_corrections(&peaks, &out.peaks, c);
    }

    #[test]
    fn corrections_removed_beat() {
        // regular series with slight variability and one extra beat between beats 20 and 21
        let mut peaks: Vec<usize> = (0..40)
            .map(|i| 200 + i * 435 + (15.0 * (i as f64 * 0.7).sin()).round() as usize + 15)
            .collect();
        let extra = (peaks[20] + peaks[21]) / 2 + 7;
        peaks.insert(21, extra);
        let out = fix_peaks_with(&peaks, 512.0, &KubiosParams::default()).unwrap();
        let c = &out.corrections;
        assert!(c.removed_count() >= 1, "{:?}", out.passes);
        assert_eq!(c.input_to_output[21], None);
        assert!(!out.peaks.contains(&extra));
        check_corrections(&peaks, &out.peaks, c);
    }

    #[test]
    fn corrections_compare_final_with_input() {
        let input = [0usize, 10, 20, 30, 40];
        // beat 1 removed and a beat inserted at exactly its position: unchanged
        // beat 3 moved (keeps its origin), a new beat inserted after 40
        let output = [
            (0usize, 0usize),
            (10, INSERTED),
            (20, 2),
            (33, 3),
            (40, 4),
            (50, INSERTED),
        ];
        let c = compute_corrections(&input, &output);
        use BeatStatus::*;
        assert_eq!(
            c.status,
            vec![Unchanged, Unchanged, Unchanged, Moved, Unchanged, Inserted]
        );
        assert_eq!(
            c.input_to_output,
            vec![Some(0), Some(1), Some(2), Some(3), Some(4)]
        );
        assert_eq!(c.removed, 0);
        assert_eq!(c.changed_count(), 2);
        assert_eq!(c.changed_ratio(), 2.0 / 6.0);
        let values: Vec<usize> = output.iter().map(|p| p.0).collect();
        check_corrections(&input, &values, &c);
        // empty
        assert!(Corrections::identity(0).changed_ratio().is_nan());
    }

    #[test]
    fn positions_inserted_beat_is_exact_midpoint() {
        let mut pos: Vec<f64> = [
            100, 510, 921, 1741, 2150, 2560, 2971, 3380, 3790, 4200, 4611, 5020, 5431,
        ]
        .iter()
        .map(|&p| p as f64)
        .collect();
        pos[2] += 0.25;
        pos[3] += 0.5;
        let out = fix_positions_with(&pos, 512.0, &KubiosParams::default()).unwrap();
        let c = &out.corrections;
        assert_eq!(c.status[3], BeatStatus::Inserted);
        assert_eq!(c.changed_count(), 1);
        // 921.25 + (1741.5 − 921.25) / 2 = 1331.375, not rounded
        assert_eq!(out.positions[3], 1331.375);
        assert_eq!(out.positions[2], 921.25);
        assert_eq!(out.positions[4], 1741.5);
        check_corrections(&pos, &out.positions, c);
        // the floor rule on the same input
        let fl = fix_positions_with_midpoint(
            &pos,
            512.0,
            &KubiosParams::default(),
            KubiosMidpoint::Floor,
        )
        .unwrap();
        assert_eq!(fl.positions[3], 1331.25);
        assert_eq!(fl.corrections, out.corrections);
    }

    #[test]
    fn positions_moved_beat_keeps_neighbours() {
        let peaks = [100usize, 510, 1330, 1740, 2150, 2560, 2970, 3380, 3790];
        let pos: Vec<f64> = peaks
            .iter()
            .enumerate()
            .map(|(i, &p)| p as f64 + 0.1 * i as f64 - 0.3)
            .collect();
        let out = fix_positions_with(&pos, 512.0, &KubiosParams::default()).unwrap();
        let int = fix_peaks_with(&peaks, 512.0, &KubiosParams::default()).unwrap();
        assert_eq!(out.corrections, int.corrections);
        assert_eq!(out.positions[2], pos[1] + (pos[3] - pos[1]) / 2.0);
        for j in [0, 1, 3, 4, 5, 6, 7, 8] {
            assert_eq!(out.positions[j].to_bits(), pos[j].to_bits());
        }
        // times in seconds with sampling_rate = 1: same per-beat result
        let t: Vec<f64> = pos.iter().map(|p| p / 512.0).collect();
        let ts = fix_positions_with(&t, 1.0, &KubiosParams::default()).unwrap();
        assert_eq!(ts.corrections, out.corrections);
        assert!((ts.positions[2] * 512.0 - out.positions[2]).abs() < 1e-6);
        // RR from the corrected positions
        assert_eq!(
            out.rri_ms(512.0),
            crate::report::rri_from_positions(&out.positions, 512.0)
        );
    }

    #[test]
    fn positions_invalid_and_short_inputs() {
        let p = KubiosParams::default();
        assert_eq!(
            fix_positions_with(&[1.0, f64::NAN, 3.0], 512.0, &p),
            Err(HrvError::NonFinite)
        );
        assert_eq!(
            fix_positions_with(&[1.0, 3.0, 2.0], 512.0, &p),
            Err(HrvError::UnsortedPeaks { index: 2 })
        );
        assert!(matches!(
            fix_positions_with(&[1.0, 3.0, 5.0], 0.0, &p),
            Err(HrvError::InvalidSamplingRate(_))
        ));
        let short = fix_positions_with(&[10.5, 400.25], 512.0, &p).unwrap();
        assert_eq!(short.positions, vec![10.5, 400.25]);
        assert_eq!(short.corrections, Corrections::identity(2));
        let empty = fix_positions_with(&[], 512.0, &p).unwrap();
        assert!(empty.changed_ratio().is_nan());
    }

    #[test]
    fn unsorted_peaks_do_not_panic() {
        assert!(fix_peaks_with(&[100, 600, 550, 1000], 512.0, &KubiosParams::default()).is_err());
        let (p, _) = fix_peaks_default(&[100, 600, 550, 1000, 1500, 1500], 512);
        assert!(p.windows(2).all(|w| w[1] > w[0]));
    }
}
