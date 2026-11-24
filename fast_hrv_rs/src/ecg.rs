//! ECG Processing Module
//!
//! Optimized ECG processing for HRV analysis.
//! Implements signal cleaning, quality assessment, and statistics computation.

use crate::hrv::{compute_pnn20, compute_pnn50, compute_rmssd, std_ddof};
use crate::peaks::ecg_peaks;

// =============================================================================
// SIGNAL CLEANING (Butterworth Filter)
// =============================================================================

/// Second-order section (SOS) coefficients for Butterworth filter
#[derive(Clone, Debug)]
pub struct SosSection {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
}

/// Apply a single SOS section using Direct Form II Transposed (in-place output)
#[inline(always)]
fn apply_sos_section_inplace(section: &SosSection, input: &[f64], output: &mut [f64]) {
    let n = input.len();
    let (b0, b1, b2, a1, a2) = (section.b0, section.b1, section.b2, section.a1, section.a2);
    let mut z1 = 0.0;
    let mut z2 = 0.0;

    for i in 0..n {
        let x = input[i];
        let y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        output[i] = y;
    }
}

/// Apply SOS filter (cascade of second-order sections) - optimized with double buffering
fn sosfilt_inplace(sos: &[SosSection], input: &[f64], buf_a: &mut [f64], buf_b: &mut [f64]) {
    if sos.is_empty() {
        buf_a.copy_from_slice(input);
        return;
    }

    // First section: input -> buf_a
    apply_sos_section_inplace(&sos[0], input, buf_a);

    // Remaining sections: alternate between buffers
    let mut use_a = true;
    for section in &sos[1..] {
        if use_a {
            apply_sos_section_inplace(section, buf_a, buf_b);
        } else {
            apply_sos_section_inplace(section, buf_b, buf_a);
        }
        use_a = !use_a;
    }

    // Ensure result is in buf_a
    if !use_a && sos.len() > 1 {
        buf_a.copy_from_slice(buf_b);
    }
}

/// Reverse array in-place
#[inline]
fn reverse_inplace(arr: &mut [f64]) {
    let n = arr.len();
    let half = n / 2;
    for i in 0..half {
        arr.swap(i, n - 1 - i);
    }
}

/// Apply zero-phase filtering (forward-backward) - optimized
fn sosfiltfilt(sos: &[SosSection], signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let mut buf_a = vec![0.0; n];
    let mut buf_b = vec![0.0; n];

    // Forward pass
    sosfilt_inplace(sos, signal, &mut buf_a, &mut buf_b);

    // Reverse in-place
    reverse_inplace(&mut buf_a);

    // Backward pass (reuse buf_b)
    buf_b.iter_mut().for_each(|x| *x = 0.0);
    let mut buf_c = vec![0.0; n];
    sosfilt_inplace(sos, &buf_a, &mut buf_b, &mut buf_c);

    // Reverse back
    reverse_inplace(&mut buf_b);

    buf_b
}

/// Design Butterworth bandpass filter (simplified for ECG: 0.5-40 Hz)
/// Returns SOS coefficients for 4th order bandpass
pub fn design_butterworth_bandpass(
    lowcut: f64,
    highcut: f64,
    sampling_rate: u32,
    order: usize,
) -> Vec<SosSection> {
    let fs = sampling_rate as f64;
    let nyq = fs / 2.0;

    // Normalized frequencies
    let low = lowcut / nyq;
    let high = highcut / nyq;

    // Pre-warp frequencies for bilinear transform
    let low_w = (std::f64::consts::PI * low).tan();
    let high_w = (std::f64::consts::PI * high).tan();
    let bw = high_w - low_w;
    let w0 = (low_w * high_w).sqrt();

    // For a 4th order bandpass, we need 2 SOS sections
    // This is a simplified implementation - coefficients are approximated
    let mut sos = Vec::new();

    for k in 0..order {
        let theta = std::f64::consts::PI * (2.0 * k as f64 + 1.0) / (4.0 * order as f64);
        let alpha = (bw / 2.0) / theta.cos();

        // Approximate SOS coefficients using bilinear transform
        let k_val = 1.0 / (1.0 + alpha + w0 * w0 / (1.0 + alpha));

        sos.push(SosSection {
            b0: alpha * k_val,
            b1: 0.0,
            b2: -alpha * k_val,
            a1: 2.0 * (w0 * w0 / (1.0 + alpha) - 1.0) * k_val,
            a2: (1.0 - alpha + w0 * w0 / (1.0 + alpha)) * k_val,
        });
    }

    sos
}

/// Clean ECG signal using Butterworth bandpass filter
/// Similar to NeuroKit2's ecg_clean with method="neurokit"
pub fn ecg_clean(ecg_signal: &[f64], sampling_rate: u32) -> Vec<f64> {
    if ecg_signal.len() < 10 {
        return ecg_signal.to_vec();
    }

    // NeuroKit default: 0.5 Hz highpass, ~40 Hz lowpass equivalent
    // Using 4th order Butterworth bandpass (0.5 - 40 Hz)
    let sos = design_butterworth_bandpass(0.5, 40.0, sampling_rate, 2);

    sosfiltfilt(&sos, ecg_signal)
}

// =============================================================================
// QUALITY ASSESSMENT
// =============================================================================

/// Compute ECG signal quality using template matching
///
/// Algorithm:
/// 1. Extract heartbeat segments around each R-peak
/// 2. Standardize (z-score) across time points
/// 3. Compute distance from mean template
/// 4. Normalize to 0-1 (1 = highest quality)
/// 5. Interpolate to continuous signal
pub fn compute_quality(
    ecg_cleaned: &[f64],
    rpeaks: &[usize],
    sampling_rate: u32,
) -> (Vec<f64>, Vec<f64>) {
    let n_samples = ecg_cleaned.len();

    if rpeaks.len() < 5 {
        return (vec![0.0; n_samples], vec![]);
    }

    // Calculate window size based on heart rate
    let rr: Vec<f64> = rpeaks
        .windows(2)
        .map(|w| (w[1] - w[0]) as f64 / sampling_rate as f64)
        .collect();

    if rr.is_empty() {
        return (vec![0.0; n_samples], vec![]);
    }

    let mean_rr: f64 = rr.iter().sum::<f64>() / rr.len() as f64;
    let heart_rate = 60.0 / mean_rr;
    let window_size = 60.0 / heart_rate;
    let ratio_pre = 0.35;

    let start_offset = (-ratio_pre * window_size * sampling_rate as f64) as isize;
    let end_offset = ((1.0 - ratio_pre) * window_size * sampling_rate as f64) as isize;
    let segment_len = (end_offset - start_offset) as usize;

    // Extract valid segments
    let mut segments: Vec<Vec<f64>> = Vec::new();
    let mut valid_indices: Vec<usize> = Vec::new();

    for (i, &peak) in rpeaks.iter().enumerate() {
        let start = peak as isize + start_offset;
        let end = peak as isize + end_offset;

        if start >= 0 && end <= n_samples as isize {
            let start_u = start as usize;
            let end_u = end as usize;
            if end_u - start_u == segment_len {
                segments.push(ecg_cleaned[start_u..end_u].to_vec());
                valid_indices.push(i);
            }
        }
    }

    if segments.len() < 3 {
        return (vec![0.0; n_samples], vec![]);
    }

    let n_beats = segments.len();

    // Compute column means and stds
    let mut col_mean = vec![0.0; segment_len];
    let mut col_std = vec![0.0; segment_len];

    for j in 0..segment_len {
        let col: Vec<f64> = segments.iter().map(|s| s[j]).collect();
        let mean: f64 = col.iter().sum::<f64>() / n_beats as f64;
        col_mean[j] = mean;

        let variance: f64 =
            col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n_beats - 1) as f64;
        col_std[j] = if variance > 0.0 { variance.sqrt() } else { 1.0 };
    }

    // Compute z-scores and mean distance per beat
    let mut dist: Vec<f64> = Vec::with_capacity(n_beats);

    for segment in &segments {
        let z_mean: f64 = segment
            .iter()
            .zip(&col_mean)
            .zip(&col_std)
            .map(|((&x, &m), &s)| (x - m) / s)
            .sum::<f64>()
            / segment_len as f64;
        dist.push(z_mean.abs());
    }

    // Rescale to [0, 1]
    let min_d = dist.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_d = dist.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let quality_per_beat: Vec<f64> = if max_d > min_d {
        dist.iter()
            .map(|&d| 1.0 - (d - min_d) / (max_d - min_d))
            .collect()
    } else {
        vec![1.0; n_beats]
    };

    // Interpolate to continuous signal (step function)
    let mut quality = vec![0.0; n_samples];
    let valid_peaks: Vec<usize> = valid_indices.iter().map(|&i| rpeaks[i]).collect();

    // Before first peak
    if !valid_peaks.is_empty() && valid_peaks[0] > 0 {
        for j in 0..valid_peaks[0] {
            quality[j] = quality_per_beat[0];
        }
    }

    // Between peaks and after
    for (i, &q) in quality_per_beat.iter().enumerate() {
        let start_idx = valid_peaks[i];
        let end_idx = if i < quality_per_beat.len() - 1 {
            valid_peaks[i + 1]
        } else {
            n_samples
        };

        for j in start_idx..end_idx {
            quality[j] = q;
        }
    }

    (quality, quality_per_beat)
}

// =============================================================================
// STATISTICS
// =============================================================================

/// ECG and HRV statistics
#[derive(Debug, Clone, Default)]
pub struct EcgStats {
    pub n_peaks: usize,

    // Heart rate
    pub mean_hr_bpm: f64,
    pub std_hr_bpm: f64,
    pub min_hr_bpm: f64,
    pub max_hr_bpm: f64,

    // RR intervals
    pub mean_rr_ms: f64,
    pub std_rr_ms: f64,
    pub min_rr_ms: f64,
    pub max_rr_ms: f64,

    // Quality
    pub mean_quality: f64,
    pub min_quality: f64,
    pub max_quality: f64,
    pub n_low_quality_beats: usize,

    // HRV time-domain
    pub sdnn_ms: f64,
    pub rmssd_ms: f64,
    pub pnn50: f64,
    pub pnn20: f64,
    pub sdsd_ms: f64,
}

/// Compute ECG and HRV statistics
pub fn compute_stats(rpeaks: &[usize], quality_per_beat: &[f64], sampling_rate: u32) -> EcgStats {
    let mut stats = EcgStats::default();

    if rpeaks.len() < 2 {
        stats.n_peaks = rpeaks.len();
        return stats;
    }

    stats.n_peaks = rpeaks.len();

    // RR intervals in milliseconds
    let rr_ms: Vec<f64> = rpeaks
        .windows(2)
        .map(|w| (w[1] - w[0]) as f64 / sampling_rate as f64 * 1000.0)
        .collect();

    // Heart rate in BPM
    let hr_bpm: Vec<f64> = rr_ms.iter().map(|&rr| 60000.0 / rr).collect();

    // HR stats
    stats.mean_hr_bpm = hr_bpm.iter().sum::<f64>() / hr_bpm.len() as f64;
    stats.std_hr_bpm = std_ddof(&hr_bpm, 1);
    stats.min_hr_bpm = hr_bpm.iter().cloned().fold(f64::INFINITY, f64::min);
    stats.max_hr_bpm = hr_bpm.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // RR stats
    stats.mean_rr_ms = rr_ms.iter().sum::<f64>() / rr_ms.len() as f64;
    stats.std_rr_ms = std_ddof(&rr_ms, 1);
    stats.min_rr_ms = rr_ms.iter().cloned().fold(f64::INFINITY, f64::min);
    stats.max_rr_ms = rr_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Quality stats
    if !quality_per_beat.is_empty() {
        stats.mean_quality = quality_per_beat.iter().sum::<f64>() / quality_per_beat.len() as f64;
        stats.min_quality = quality_per_beat
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        stats.max_quality = quality_per_beat
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        stats.n_low_quality_beats = quality_per_beat.iter().filter(|&&q| q < 0.5).count();
    }

    // HRV time-domain
    if rr_ms.len() >= 3 {
        stats.sdnn_ms = std_ddof(&rr_ms, 1);
        stats.rmssd_ms = compute_rmssd(&rr_ms);
        stats.pnn50 = compute_pnn50(&rr_ms);
        stats.pnn20 = compute_pnn20(&rr_ms);

        // SDSD
        let diff_rr: Vec<f64> = rr_ms.windows(2).map(|w| w[1] - w[0]).collect();
        stats.sdsd_ms = std_ddof(&diff_rr, 1);
    }

    stats
}

// =============================================================================
// MAIN PIPELINE
// =============================================================================

/// ECG processing result
#[derive(Debug)]
pub struct EcgResult {
    pub ecg_cleaned: Vec<f64>,
    pub quality: Vec<f64>,
    pub rpeaks: Vec<usize>,
    pub stats: EcgStats,
}

/// Process ECG signal for HRV analysis
///
/// Pipeline:
/// 1. Clean signal (Butterworth bandpass filter)
/// 2. Detect R-peaks + Correct artifacts
/// 3. Compute quality scores
/// 4. Compute statistics
pub fn process_ecg(ecg_signal: &[f64], sampling_rate: u32, correct_artifacts: bool) -> EcgResult {
    // 1. Clean signal
    let ecg_cleaned = ecg_clean(ecg_signal, sampling_rate);

    // 2. Detect R-peaks
    let (rpeaks, _artifacts) = ecg_peaks(&ecg_cleaned, sampling_rate, correct_artifacts);

    // 3. Compute quality
    let (quality, quality_per_beat) = compute_quality(&ecg_cleaned, &rpeaks, sampling_rate);

    // 4. Compute statistics
    let stats = compute_stats(&rpeaks, &quality_per_beat, sampling_rate);

    EcgResult {
        ecg_cleaned,
        quality,
        rpeaks,
        stats,
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ecg_clean_preserves_length() {
        let signal: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.1).sin()).collect();
        let cleaned = ecg_clean(&signal, 512);
        assert_eq!(cleaned.len(), signal.len());
    }

    #[test]
    fn test_compute_quality_insufficient_peaks() {
        let signal = vec![0.0; 1000];
        let rpeaks = vec![100, 300]; // Only 2 peaks
        let (quality, quality_per_beat) = compute_quality(&signal, &rpeaks, 512);
        assert_eq!(quality.len(), 1000);
        assert!(quality_per_beat.is_empty());
    }

    #[test]
    fn test_compute_stats_basic() {
        // Regular peaks at ~75 BPM (800ms between beats)
        let rpeaks = vec![0, 410, 820, 1230, 1640, 2050];
        let quality = vec![0.9, 0.85, 0.95, 0.88, 0.92];
        let stats = compute_stats(&rpeaks, &quality, 512);

        assert_eq!(stats.n_peaks, 6);
        assert!(stats.mean_hr_bpm > 70.0 && stats.mean_hr_bpm < 80.0);
        assert!(stats.mean_quality > 0.8);
    }

    #[test]
    fn test_process_ecg_synthetic() {
        // Create synthetic ECG-like signal
        let mut signal = vec![0.0; 5000];
        for &pos in &[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500] {
            if pos < signal.len() {
                signal[pos] = 1.0;
                if pos > 0 {
                    signal[pos - 1] = 0.3;
                }
                if pos + 1 < signal.len() {
                    signal[pos + 1] = 0.3;
                }
            }
        }

        let result = process_ecg(&signal, 512, false);
        assert_eq!(result.ecg_cleaned.len(), signal.len());
        assert_eq!(result.quality.len(), signal.len());
    }

    #[test]
    fn test_sos_filter_identity() {
        // Identity filter (b0=1, others=0)
        let sos = vec![SosSection {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        }];
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut buf_a = vec![0.0; signal.len()];
        let mut buf_b = vec![0.0; signal.len()];
        sosfilt_inplace(&sos, &signal, &mut buf_a, &mut buf_b);

        for (f, s) in buf_a.iter().zip(&signal) {
            assert!((f - s).abs() < 1e-10);
        }
    }
}
