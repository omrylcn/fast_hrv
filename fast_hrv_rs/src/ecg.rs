//! ECG Processing Module
//!
//! Signal cleaning, quality assessment and statistics for HRV analysis.
//!
//! ## Cleaning
//!
//! Two cleaning paths, both built on a real Butterworth design ([`crate::filter::butter_sos`]):
//!
//! - **Offline, NeuroKit2-compatible** ([`ecg_clean`], [`ecg_clean_neurokit`]): equivalent to
//!   `nk.ecg_clean(signal, sampling_rate, method="neurokit")`: 0.5 Hz high-pass Butterworth of
//!   order 5 applied zero-phase (`scipy.signal.sosfiltfilt`, odd padding + `sosfilt_zi`), followed by
//!   NeuroKit's powerline step, a moving average of `int(fs / powerline)` samples applied with
//!   `scipy.signal.filtfilt(method="pad")`. Zero-phase filtering is not causal.
//! - **Causal / streaming** ([`EcgStreamFilter`], [`ecg_clean_causal`]): high-pass + low-pass
//!   Butterworth (default 0.5–40 Hz, orders 2 and 4) and an optional IIR notch, with the filter state
//!   carried across packets. Packet-by-packet processing is bit-identical to processing the
//!   concatenated signal in one call. Causal filtering adds a (small, smooth) group delay.
//!
//! [`ecg_clean_zero_phase`] applies the same band as the streaming filter offline, zero-phase, with
//! SciPy-style padding.

use crate::error::HrvError;
use crate::filter::{
    BandType, InitialState, SosFilter, butter_sos, filtfilt_ba, iirnotch, sosfiltfilt,
};
use crate::hrv::{compute_pnn20, compute_pnn50, compute_rmssd, std_ddof};
use crate::peaks::ecg_peaks;

pub use crate::filter::SosSection;

// =============================================================================
// SIGNAL CLEANING
// =============================================================================

/// Butterworth band-pass filter as second-order sections.
///
/// Equivalent to `scipy.signal.butter(order, [lowcut, highcut], "bandpass", output="sos", fs=fs)`:
/// the band-pass has `2 * order` poles and `order` sections.
///
/// **Changed in 0.2.0:** earlier versions returned approximate, non-Butterworth coefficients
/// ("order" was the number of biquads). Invalid parameters (cut-offs outside `(0, fs/2)`,
/// `lowcut >= highcut`, `order == 0`, `sampling_rate == 0`) now return an empty vector (identity);
/// use [`crate::filter::butter_sos`] to get an error instead.
pub fn design_butterworth_bandpass(
    lowcut: f64,
    highcut: f64,
    sampling_rate: u32,
    order: usize,
) -> Vec<SosSection> {
    butter_sos(
        order,
        &[lowcut, highcut],
        BandType::Bandpass,
        sampling_rate as f64,
    )
    .unwrap_or_default()
}

/// NeuroKit2's missing-value handling for `ecg_clean`: forward fill, then backward fill.
fn fill_missing(x: &[f64]) -> Vec<f64> {
    let mut y = x.to_vec();
    let mut last = f64::NAN;
    for v in y.iter_mut() {
        if v.is_finite() {
            last = *v;
        } else {
            *v = last;
        }
    }
    let mut next = f64::NAN;
    for v in y.iter_mut().rev() {
        if v.is_finite() {
            next = *v;
        } else {
            *v = next;
        }
    }
    y
}

/// NeuroKit2-compatible cleaning (`nk.ecg_clean(method="neurokit", powerline=powerline_hz)`).
///
/// 1. Non-finite samples are forward/backward filled (NeuroKit `signal_fillmissing(method="both")`).
/// 2. High-pass Butterworth, order 5, 0.5 Hz, zero-phase (`sosfiltfilt` with SciPy's default padding).
/// 3. Powerline: moving average of `int(fs / powerline_hz)` samples (2 samples if `fs < 100`),
///    zero-phase (`filtfilt(ones(n), [n], method="pad")`). At 512 Hz / 50 Hz this is a 10-sample
///    average: zero at 51.2 Hz, about −32 dB at 50 Hz, −3 dB at ≈22.8 Hz (filtfilt response).
///
/// Errors: invalid sampling rate (needs `fs > 1` for the 0.5 Hz high-pass), invalid powerline
/// frequency, or no finite sample at all.
pub fn ecg_clean_neurokit(
    ecg_signal: &[f64],
    sampling_rate: f64,
    powerline_hz: f64,
) -> Result<Vec<f64>, HrvError> {
    if !(sampling_rate.is_finite() && sampling_rate > 0.0) {
        return Err(HrvError::InvalidSamplingRate(sampling_rate));
    }
    if !(powerline_hz.is_finite() && powerline_hz > 0.0) {
        return Err(HrvError::InvalidFrequency {
            freq: powerline_hz,
            nyquist: sampling_rate / 2.0,
        });
    }
    if ecg_signal.is_empty() {
        return Ok(Vec::new());
    }
    let x = fill_missing(ecg_signal);
    if !x[0].is_finite() {
        return Err(HrvError::NonFinite);
    }
    let sos = butter_sos(5, &[0.5], BandType::Highpass, sampling_rate)?;
    let hp = sosfiltfilt(&sos, &x);
    let n_ma = if sampling_rate >= 100.0 {
        ((sampling_rate / powerline_hz) as usize).max(1)
    } else {
        2
    };
    let b = vec![1.0; n_ma];
    Ok(filtfilt_ba(&b, &[n_ma as f64], &hp))
}

/// Clean an ECG signal, NeuroKit2-compatible (`nk.ecg_clean(method="neurokit")`, 50 Hz powerline).
///
/// See [`ecg_clean_neurokit`]. Zero-phase and therefore offline only; for live processing use
/// [`EcgStreamFilter`].
///
/// **Changed in 0.2.0:** earlier versions used an approximate (non-Butterworth) band-pass without
/// padding and without a powerline step. On invalid input (`sampling_rate` too low, no finite
/// sample) this returns a vector of `NaN` with the input length instead of panicking.
pub fn ecg_clean(ecg_signal: &[f64], sampling_rate: u32) -> Vec<f64> {
    ecg_clean_neurokit(ecg_signal, sampling_rate as f64, 50.0)
        .unwrap_or_else(|_| vec![f64::NAN; ecg_signal.len()])
}

/// Optional IIR notch of the streaming filter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NotchConfig {
    /// Centre frequency (Hz), e.g. 50 or 60.
    pub freq_hz: f64,
    /// Quality factor (centre frequency / −3 dB bandwidth).
    pub q: f64,
}

/// Configuration of the causal ECG filter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EcgFilterConfig {
    /// High-pass cut-off (Hz); `None` disables the high-pass.
    pub highpass_hz: Option<f64>,
    pub highpass_order: usize,
    /// Low-pass cut-off (Hz); `None` disables the low-pass.
    pub lowpass_hz: Option<f64>,
    pub lowpass_order: usize,
    /// Optional notch (powerline).
    pub notch: Option<NotchConfig>,
    /// State on the first sample (default: steady state for the first sample, no DC transient).
    pub initial_state: InitialState,
}

impl Default for EcgFilterConfig {
    /// 0.5 Hz high-pass (order 2) + 40 Hz low-pass (order 4), no notch.
    fn default() -> Self {
        Self {
            highpass_hz: Some(0.5),
            highpass_order: 2,
            lowpass_hz: Some(40.0),
            lowpass_order: 4,
            notch: None,
            initial_state: InitialState::SteadyState,
        }
    }
}

impl EcgFilterConfig {
    /// Add a notch at `freq_hz` with quality factor `q`.
    pub fn with_notch(mut self, freq_hz: f64, q: f64) -> Self {
        self.notch = Some(NotchConfig { freq_hz, q });
        self
    }

    /// The cascade (high-pass sections, low-pass sections, notch) for sampling rate `fs`.
    pub fn design(&self, fs: f64) -> Result<Vec<SosSection>, HrvError> {
        let mut sos = Vec::new();
        if let Some(f) = self.highpass_hz {
            sos.extend(butter_sos(
                self.highpass_order,
                &[f],
                BandType::Highpass,
                fs,
            )?);
        }
        if let Some(f) = self.lowpass_hz {
            sos.extend(butter_sos(self.lowpass_order, &[f], BandType::Lowpass, fs)?);
        }
        if let Some(n) = self.notch {
            sos.push(iirnotch(n.freq_hz, n.q, fs)?);
        }
        Ok(sos)
    }
}

/// Causal, stateful ECG filter for live packet-by-packet processing (one instance per stream).
///
/// ```
/// use fast_hrv::ecg::{EcgFilterConfig, EcgStreamFilter};
/// let cfg = EcgFilterConfig::default().with_notch(50.0, 30.0);
/// let raw: Vec<f64> = (0..2048).map(|i| 412.0 + (i as f64 * 0.1).sin()).collect();
/// let mut live = EcgStreamFilter::new(512.0, &cfg).unwrap();
/// let mut out = Vec::new();
/// for packet in raw.chunks(32) {
///     out.extend(live.process(packet));
/// }
/// // identical to processing everything at once
/// assert_eq!(out, fast_hrv::ecg::ecg_clean_causal(&raw, 512.0, &cfg).unwrap());
/// ```
#[derive(Clone, Debug)]
pub struct EcgStreamFilter {
    inner: SosFilter,
    sampling_rate: f64,
}

impl EcgStreamFilter {
    pub fn new(sampling_rate: f64, config: &EcgFilterConfig) -> Result<Self, HrvError> {
        let sos = config.design(sampling_rate)?;
        Ok(Self {
            inner: SosFilter::new(sos, config.initial_state),
            sampling_rate,
        })
    }

    pub fn sampling_rate(&self) -> f64 {
        self.sampling_rate
    }

    /// Filter one packet (any length, e.g. 32 samples) and return the filtered samples.
    pub fn process(&mut self, packet: &[f64]) -> Vec<f64> {
        self.inner.process(packet)
    }

    /// Filter one packet in place.
    pub fn process_inplace(&mut self, packet: &mut [f64]) {
        self.inner.process_inplace(packet)
    }

    /// Filter one packet into `out` (same length as `packet`).
    pub fn process_into(&mut self, packet: &[f64], out: &mut [f64]) {
        self.inner.process_into(packet, out)
    }

    /// Forget the filter state (e.g. after a gap in the stream).
    pub fn reset(&mut self) {
        self.inner.reset()
    }

    /// The underlying SOS cascade.
    pub fn filter(&self) -> &SosFilter {
        &self.inner
    }
}

/// Causal filtering of a whole signal; identical to feeding it through [`EcgStreamFilter`] in packets.
pub fn ecg_clean_causal(
    ecg_signal: &[f64],
    sampling_rate: f64,
    config: &EcgFilterConfig,
) -> Result<Vec<f64>, HrvError> {
    let mut f = EcgStreamFilter::new(sampling_rate, config)?;
    Ok(f.process(ecg_signal))
}

/// Offline zero-phase version of the configured band (`sosfiltfilt` with SciPy's default padding).
/// The magnitude response is the square of the causal one; no delay.
pub fn ecg_clean_zero_phase(
    ecg_signal: &[f64],
    sampling_rate: f64,
    config: &EcgFilterConfig,
) -> Result<Vec<f64>, HrvError> {
    let sos = config.design(sampling_rate)?;
    Ok(sosfiltfilt(&sos, ecg_signal))
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
        let (out, _) = crate::filter::sosfilt(&sos, &signal, None);
        for (f, s) in out.iter().zip(&signal) {
            assert!((f - s).abs() < 1e-10);
        }
    }

    #[test]
    fn ecg_clean_removes_dc_without_edge_transient() {
        // Raw ADC-like input: DC 412 + slow sine. NeuroKit-compatible cleaning must not produce the
        // large start-up transient of zero-state forward-backward filtering.
        let x: Vec<f64> = (0..20 * 512)
            .map(|i| 412.0 + 0.2 * (i as f64 * 0.05).sin())
            .collect();
        let y = ecg_clean(&x, 512);
        assert!(
            y.iter().all(|v| v.abs() < 1.0),
            "max {}",
            y.iter().fold(0.0f64, |m, v| m.max(v.abs()))
        );
    }

    #[test]
    fn ecg_clean_invalid_sampling_rate_is_nan() {
        let y = ecg_clean(&[1.0; 100], 0);
        assert_eq!(y.len(), 100);
        assert!(y.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn streaming_filter_equals_one_shot() {
        let cfg = EcgFilterConfig::default().with_notch(50.0, 30.0);
        let x: Vec<f64> = (0..10_000)
            .map(|i| 412.0 + 30.0 * (i as f64 * 0.013).sin() + 5.0 * (i as f64 * 0.61).sin())
            .collect();
        let whole = ecg_clean_causal(&x, 512.0, &cfg).unwrap();
        let mut f = EcgStreamFilter::new(512.0, &cfg).unwrap();
        let mut live = Vec::new();
        for p in x.chunks(32) {
            live.extend(f.process(p));
        }
        assert_eq!(whole, live);
    }

    #[test]
    fn design_bandpass_is_real_butterworth() {
        // −3 dB at both edges of a Butterworth band-pass (|H|² = 0.5)
        let sos = design_butterworth_bandpass(0.5, 40.0, 512, 2);
        assert_eq!(sos.len(), 2);
        let gain2 = |f: f64| {
            let w = 2.0 * std::f64::consts::PI * f / 512.0;
            sos.iter()
                .map(|s| {
                    let (c1, s1, c2, s2) = (w.cos(), w.sin(), (2.0 * w).cos(), (2.0 * w).sin());
                    let nr = s.b0 + s.b1 * c1 + s.b2 * c2;
                    let ni = -(s.b1 * s1 + s.b2 * s2);
                    let dr = 1.0 + s.a1 * c1 + s.a2 * c2;
                    let di = -(s.a1 * s1 + s.a2 * s2);
                    (nr * nr + ni * ni) / (dr * dr + di * di)
                })
                .product::<f64>()
        };
        assert!((gain2(0.5) - 0.5).abs() < 1e-9);
        assert!((gain2(40.0) - 0.5).abs() < 1e-9);
    }
}
