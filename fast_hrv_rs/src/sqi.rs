//! # Absolute ECG signal-quality indices (SQI)
//!
//! Window-level, *absolute* quality indices for single-lead ECG: unlike the relative
//! [`crate::ecg::compute_quality`] (which rescales beats against each other), every value here has
//! a fixed meaning, so it can be compared across windows, recordings and users.
//!
//! The module only **computes** indices and returns structured values. Product-specific gate
//! thresholds and the final accept/reject decision belong to the caller. Where a classification
//! helper exists ([`zhao2018_fusion`], [`rr_plausibility`]), its thresholds are parameters whose
//! defaults are the published values.
//!
//! | Index | Function | Source |
//! |---|---|---|
//! | flat line | [`flatline_fraction`] | raw ADC stream check (heuristic) |
//! | saturation / clipping | [`saturation_fraction`] | raw ADC stream check (heuristic) |
//! | kSQI | [`ksqi`] (= [`kurtosis_pearson`]) | Li, Mark & Clifford 2008; Zhao & Zhang 2018; NeuroKit2 |
//! | pSQI | [`psqi`] / [`psqi_from_psd`] | Zhao & Zhang 2018; NeuroKit2 `ecg_quality(method="zhao2018")` |
//! | basSQI | [`bassqi`] / [`bassqi_from_psd`] | Zhao & Zhang 2018; NeuroKit2 |
//! | bSQI | [`bsqi`] | Li, Mark & Clifford 2008; Clifford et al. 2012 |
//! | template correlation | [`template_correlation`] | Orphanidou et al. 2015; BioSPPy beat window |
//! | RR plausibility | [`rr_plausibility`] | Orphanidou et al. 2015 |
//! | 3-class fusion | [`zhao2018_fusion`] | Zhao & Zhang 2018 (NeuroKit2 "simple" heuristic) |
//!
//! pSQI and basSQI share one Welch PSD ([`spectral_sqi`], [`PsdEngine`]). Two PSD flavours exist
//! ([`PsdConfig`]): plain `scipy.signal.welch` defaults with a configurable `nperseg`
//! (default 2048 samples = 4 s at 512 Hz), and an exact replica of what NeuroKit2's zhao2018 code
//! actually runs.
//!
//! All functions are panic-free on bad input (empty slices, NaN, too few beats, nonsensical
//! parameters) and return `Result<_, SqiError>`. No SIMD intrinsics, no extra dependencies; the FFT
//! (mixed radix 2/3/5/..., Bluestein for large prime factors) is implemented here.
//!
//! ## References
//! - Li Q, Mark RG, Clifford GD. *Robust heart rate estimation from multiple asynchronous noisy
//!   sources using signal quality indices and a Kalman filter.* Physiol Meas 29(1):15–32, 2008.
//! - Clifford GD, Behar J, Li Q, Rezek I. *Signal quality indices and data fusion for determining
//!   clinical acceptability of electrocardiograms.* Physiol Meas 33(9):1419–1433, 2012.
//! - Orphanidou C, et al. *Signal-quality indices for the electrocardiogram and photoplethysmogram:
//!   derivation and applications to wireless monitoring.* IEEE J Biomed Health Inform
//!   19(3):832–838, 2015.
//! - Zhao Z, Zhang Y. *SQI quality evaluation mechanism of single-lead ECG signal based on simple
//!   heuristic fusion and fuzzy comprehensive evaluation.* Front Physiol 9:727, 2018.
//! - Makowski D, et al. *NeuroKit2: A Python toolbox for neurophysiological signal processing.*
//!   Behav Res Methods 53:1689–1696, 2021 (`neurokit2/ecg/ecg_quality.py`, v0.3.0).
//! - Carreiras C, et al. *BioSPPy: Biosignal Processing in Python* (`biosppy.signals.ecg.extract_heartbeats`,
//!   before = 0.2 s, after = 0.4 s).
//! - Welch PD. *The use of fast Fourier transform for the estimation of power spectra.* IEEE Trans
//!   Audio Electroacoust 15(2):70–73, 1967; SciPy `scipy.signal.welch`.

use std::f64::consts::PI;
use std::fmt;

// =============================================================================
// Errors
// =============================================================================

/// Why an index could not be computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub enum SqiError {
    /// The input slice is empty.
    EmptyInput,
    /// The input contains NaN or ±inf.
    NonFinite,
    /// A parameter is out of its valid range (the name is given).
    InvalidParameter(&'static str),
    /// The signal is too short for the requested computation.
    TooShort,
    /// Not enough (complete) beats.
    TooFewBeats,
    /// Zero variance where a normalisation needs a non-zero one (constant signal / beat / template).
    ZeroVariance,
    /// A spectral band contains no power (or fewer than two frequency bins).
    ZeroBandPower,
    /// An external template's length does not match the configured beat window.
    TemplateLength,
}

impl fmt::Display for SqiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SqiError::EmptyInput => write!(f, "empty input"),
            SqiError::NonFinite => write!(f, "input contains NaN or infinite values"),
            SqiError::InvalidParameter(p) => write!(f, "invalid parameter: {p}"),
            SqiError::TooShort => write!(f, "signal too short"),
            SqiError::TooFewBeats => write!(f, "too few beats"),
            SqiError::ZeroVariance => write!(f, "zero variance"),
            SqiError::ZeroBandPower => write!(f, "zero power in spectral band"),
            SqiError::TemplateLength => write!(f, "template length does not match beat window"),
        }
    }
}

impl std::error::Error for SqiError {}

/// Result alias for this module.
pub type SqiResult<T> = Result<T, SqiError>;

fn check_fs(fs: f64) -> SqiResult<()> {
    if fs.is_finite() && fs > 0.0 {
        Ok(())
    } else {
        Err(SqiError::InvalidParameter("fs"))
    }
}

fn check_finite(x: &[f64]) -> SqiResult<()> {
    if x.is_empty() {
        return Err(SqiError::EmptyInput);
    }
    if x.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(SqiError::NonFinite)
    }
}

// =============================================================================
// Raw ADC stream checks
// =============================================================================

/// Default flat-line tolerance in ADC counts: `0` = only exactly repeated samples count as flat
/// (stuck ADC, lead-off with a held value). Raise it for front ends with a dithering LSB.
pub const DEFAULT_FLATLINE_EPS: f64 = 0.0;
/// Default minimum flat-run duration (s). Heuristic engineering default, not a published value:
/// a 0.2 s plateau (~ one QRS-to-T stretch) is implausible for a live ECG with front-end noise.
pub const DEFAULT_FLATLINE_MIN_DURATION_S: f64 = 0.2;

/// Fraction (0–1) of samples that belong to a flat run on the **raw integer stream**.
///
/// A flat run is a maximal sequence of consecutive samples whose successive absolute differences
/// are `<= eps`, lasting at least `max(ceil(min_duration_s * fs), 2)` samples. All samples of such
/// runs are counted. NaN samples (float streams) break a run. Works with any sample type that
/// converts losslessly to `f64` (`i8`–`i32`, `u8`–`u32`, `f32`, `f64`).
///
/// Run it on the raw ADC counts, before filtering: band-pass filters turn a held value into a
/// decaying transient that no longer looks flat.
///
/// Errors: [`SqiError::EmptyInput`]; [`SqiError::InvalidParameter`] for non-positive `fs`,
/// negative/non-finite `eps` or `min_duration_s`.
pub fn flatline_fraction<T: Copy + Into<f64>>(
    signal: &[T],
    fs: f64,
    eps: f64,
    min_duration_s: f64,
) -> SqiResult<f64> {
    check_fs(fs)?;
    if !(eps.is_finite() && eps >= 0.0) {
        return Err(SqiError::InvalidParameter("eps"));
    }
    if !(min_duration_s.is_finite() && min_duration_s >= 0.0) {
        return Err(SqiError::InvalidParameter("min_duration_s"));
    }
    let n = signal.len();
    if n == 0 {
        return Err(SqiError::EmptyInput);
    }
    let min_len = ((min_duration_s * fs).ceil() as usize).max(2);
    let mut flat = 0usize;
    let mut run = 1usize;
    for i in 1..=n {
        let continues = i < n && {
            let d: f64 = signal[i].into() - signal[i - 1].into();
            d.abs() <= eps
        };
        if continues {
            run += 1;
        } else {
            if run >= min_len {
                flat += run;
            }
            run = 1;
        }
    }
    Ok(flat as f64 / n as f64)
}

/// Fraction (0–1) of samples at or near the ADC rails on the **raw integer stream**:
/// `x <= adc_min + margin || x >= adc_max - margin`.
///
/// `adc_min`/`adc_max` are the converter's code range (e.g. `0`/`4095` for 12-bit unsigned,
/// `-32768`/`32767` for 16-bit signed). `margin` (counts, `>= 0`) catches front ends that clip a
/// few codes short of full scale. NaN samples are not counted as saturated.
///
/// Errors: [`SqiError::EmptyInput`]; [`SqiError::InvalidParameter`] if `adc_min >= adc_max`,
/// `margin` is negative or anything is non-finite.
pub fn saturation_fraction<T: Copy + Into<f64>>(
    signal: &[T],
    adc_min: f64,
    adc_max: f64,
    margin: f64,
) -> SqiResult<f64> {
    if !(adc_min.is_finite() && adc_max.is_finite() && adc_min < adc_max) {
        return Err(SqiError::InvalidParameter("adc_min/adc_max"));
    }
    if !(margin.is_finite() && margin >= 0.0) {
        return Err(SqiError::InvalidParameter("margin"));
    }
    if signal.is_empty() {
        return Err(SqiError::EmptyInput);
    }
    let (lo, hi) = (adc_min + margin, adc_max - margin);
    let sat = signal
        .iter()
        .filter(|&&v| {
            let v: f64 = v.into();
            v <= lo || v >= hi
        })
        .count();
    Ok(sat as f64 / signal.len() as f64)
}

// =============================================================================
// kSQI — kurtosis
// =============================================================================

/// Pearson kurtosis `m4 / m2²` (biased central moments; normal distribution → 3).
///
/// Identical to `scipy.stats.kurtosis(x, fisher=False, bias=True)`, which is what NeuroKit2's
/// `_ecg_quality_kSQI` (zhao2018) uses. A clean ECG is strongly super-Gaussian (kurtosis well
/// above 5); Gaussian noise gives ~3, sinusoids/baseline wander ~1.5.
///
/// Errors: [`SqiError::EmptyInput`], [`SqiError::NonFinite`], [`SqiError::ZeroVariance`]
/// (constant signal; SciPy returns NaN there).
pub fn kurtosis_pearson(x: &[f64]) -> SqiResult<f64> {
    check_finite(x)?;
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let (mut m2, mut m4) = (0.0, 0.0);
    for &v in x {
        let d = v - mean;
        let d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
    }
    m2 /= n;
    m4 /= n;
    if m2 <= 0.0 || !m2.is_finite() {
        return Err(SqiError::ZeroVariance);
    }
    let k = m4 / (m2 * m2);
    if k.is_finite() {
        Ok(k)
    } else {
        Err(SqiError::ZeroVariance)
    }
}

/// Fisher (excess) kurtosis = Pearson − 3 (normal distribution → 0); `scipy.stats.kurtosis`
/// default (`fisher=True`).
///
/// **Not** the kSQI scale: the zhao2018 threshold `kSQI > 5` refers to the Pearson value
/// ([`ksqi`]). Use this one only when comparing against literature that reports excess kurtosis.
pub fn kurtosis_fisher(x: &[f64]) -> SqiResult<f64> {
    kurtosis_pearson(x).map(|k| k - 3.0)
}

/// kSQI: Pearson kurtosis of the (cleaned) ECG window, as in Li, Mark & Clifford (2008) and
/// Zhao & Zhang (2018), matching NeuroKit2's zhao2018 (`fisher=False`). See [`kurtosis_pearson`].
pub fn ksqi(x: &[f64]) -> SqiResult<f64> {
    kurtosis_pearson(x)
}

// =============================================================================
// Welch PSD (scipy-compatible) + band power (NeuroKit2-compatible)
// =============================================================================

/// pSQI numerator band (Hz): QRS energy, Zhao & Zhang (2018).
pub const PSQI_NUM_BAND: (f64, f64) = (5.0, 15.0);
/// pSQI denominator band (Hz).
pub const PSQI_DEN_BAND: (f64, f64) = (5.0, 40.0);
/// basSQI numerator band (Hz): baseline.
pub const BASSQI_NUM_BAND: (f64, f64) = (0.0, 1.0);
/// basSQI denominator band (Hz).
pub const BASSQI_DEN_BAND: (f64, f64) = (0.0, 40.0);
/// Default Welch segment length in samples (4 s at 512 Hz → 0.25 Hz bins, 4 segments in 10 s).
pub const DEFAULT_NPERSEG: usize = 2048;
/// NeuroKit2's zhao2018 `window` argument. It is in **seconds**, not samples (see [`PsdConfig`]).
pub const NEUROKIT_WINDOW_S: f64 = 1024.0;

/// How the PSD shared by pSQI and basSQI is estimated.
///
/// Band power is always integrated the NeuroKit2 way (`signal_power`): bins with
/// `lo <= f < hi`, trapezoidal rule (`numpy.trapezoid`), zero power → error.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub enum PsdConfig {
    /// `scipy.signal.welch(x, fs, nperseg=nperseg)` with SciPy defaults: periodic Hann window,
    /// `noverlap = nperseg / 2`, `nfft = nperseg`, `detrend='constant'` (per segment),
    /// `scaling='density'`, one-sided, mean averaging. If `nperseg > len`, SciPy (and this code)
    /// uses `nperseg = len`. Default: 2048 samples.
    Welch {
        /// Segment length in samples.
        nperseg: usize,
    },
    /// Exact replica of NeuroKit2 0.3.0 `_ecg_quality_pSQI/_basSQI(window=1024)` →
    /// `signal_power` → `signal_psd(method="welch")`:
    /// 1. the whole window is de-meaned once (no per-segment detrend);
    /// 2. `nperseg = int(window_s * fs)`, reduced to `int(N / 2)` if larger than `N / 2`;
    ///    with the NeuroKit default `window_s = 1024` this is **always** `N / 2`
    ///    (2560 samples for 10 s at 512 Hz) — NeuroKit's "1024" is seconds, not samples;
    /// 3. `nfft = 2 * nperseg` (zero padding), Hann, 50 % overlap, density scaling;
    /// 4. frequencies below `min_frequency = 2 fs / (N / 2)` (0.4 Hz for 10 s) are dropped
    ///    before band integration, so the "0–1 Hz" basSQI band is effectively 0.4–1 Hz.
    NeuroKit {
        /// NeuroKit2 `window` argument in seconds (default [`NEUROKIT_WINDOW_S`]).
        window_s: f64,
    },
}

impl Default for PsdConfig {
    fn default() -> Self {
        PsdConfig::Welch {
            nperseg: DEFAULT_NPERSEG,
        }
    }
}

impl PsdConfig {
    /// SciPy-default Welch with the given segment length (e.g. 2048, or 1024 for 2 s segments).
    pub fn welch(nperseg: usize) -> Self {
        PsdConfig::Welch { nperseg }
    }

    /// Bit-for-bit NeuroKit2 zhao2018 behaviour (`window=1024` s → `nperseg = N/2`).
    pub fn neurokit() -> Self {
        PsdConfig::NeuroKit {
            window_s: NEUROKIT_WINDOW_S,
        }
    }
}

/// Resolved Welch parameters for one signal length.
#[derive(Debug, Clone, Copy, PartialEq)]
struct WelchParams {
    nperseg: usize,
    noverlap: usize,
    nfft: usize,
    detrend_segments: bool,
    demean_global: bool,
    min_frequency: f64,
}

impl WelchParams {
    fn resolve(cfg: &PsdConfig, n: usize, fs: f64) -> SqiResult<Self> {
        match *cfg {
            PsdConfig::Welch { nperseg } => {
                if nperseg < 2 {
                    return Err(SqiError::InvalidParameter("nperseg"));
                }
                let nperseg = nperseg.min(n);
                if nperseg < 2 {
                    return Err(SqiError::TooShort);
                }
                Ok(WelchParams {
                    nperseg,
                    noverlap: nperseg / 2,
                    nfft: nperseg,
                    detrend_segments: true,
                    demean_global: false,
                    min_frequency: 0.0,
                })
            }
            PsdConfig::NeuroKit { window_s } => {
                if !(window_s.is_finite() && window_s > 0.0) {
                    return Err(SqiError::InvalidParameter("window_s"));
                }
                // `int(window * sampling_rate)`; `as` saturates for huge values.
                let mut nperseg = (window_s * fs) as usize;
                if nperseg as f64 > n as f64 / 2.0 {
                    nperseg = n / 2;
                }
                if nperseg < 2 {
                    return Err(SqiError::TooShort);
                }
                Ok(WelchParams {
                    nperseg,
                    noverlap: nperseg / 2,
                    nfft: 2 * nperseg,
                    detrend_segments: false,
                    demean_global: true,
                    min_frequency: (2.0 * fs) / (n as f64 / 2.0),
                })
            }
        }
    }
}

/// One-sided power spectral density.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Psd {
    /// Bin frequencies (Hz), `k * fs / nfft` exactly as `numpy.fft.rfftfreq`.
    pub freqs: Vec<f64>,
    /// Power density (signal unit² / Hz).
    pub power: Vec<f64>,
    /// Bins below this frequency are ignored by [`Psd::band_power`] (0 for SciPy mode,
    /// `2 fs / (N/2)` for NeuroKit mode).
    pub min_frequency: f64,
    /// Segment length actually used.
    pub nperseg: usize,
    /// FFT length actually used.
    pub nfft: usize,
    /// Number of averaged segments.
    pub n_segments: usize,
}

impl Psd {
    /// Band power the NeuroKit2 way (`_signal_power_instant_compute`): bins with
    /// `max(lo, min_frequency) <= f < hi`, trapezoidal integration over those bins only.
    /// Returns [`SqiError::ZeroBandPower`] where NeuroKit2 returns NaN (power exactly 0, which
    /// includes bands with fewer than two bins).
    pub fn band_power(&self, lo: f64, hi: f64) -> SqiResult<f64> {
        if !(lo.is_finite() && hi.is_finite() && lo < hi) {
            return Err(SqiError::InvalidParameter("band"));
        }
        let mut p = 0.0;
        let mut prev: Option<(f64, f64)> = None;
        for (&f, &v) in self.freqs.iter().zip(&self.power) {
            if f >= self.min_frequency && f >= lo && f < hi {
                if let Some((f0, v0)) = prev {
                    p += (f - f0) * (v + v0) / 2.0;
                }
                prev = Some((f, v));
            }
        }
        if p == 0.0 || !p.is_finite() {
            Err(SqiError::ZeroBandPower)
        } else {
            Ok(p)
        }
    }
}

/// Reusable Welch PSD estimator: caches the Hann window and FFT plan between calls with the same
/// segment/FFT length (the usual case for fixed-length windows), so steady-state cost is just
/// the FFTs. Not thread-safe by design; use one per worker.
#[derive(Debug, Default)]
pub struct PsdEngine {
    plan: Option<WelchPlan>,
}

#[derive(Debug)]
struct WelchPlan {
    nperseg: usize,
    nfft: usize,
    window: Vec<f64>,
    win_pow: f64,
    fft: FftPlan,
    buf_in: Vec<C>,
    buf_out: Vec<C>,
}

impl WelchPlan {
    fn new(nperseg: usize, nfft: usize) -> Self {
        // Periodic Hann (`scipy.signal.get_window('hann', nperseg)`, fftbins=True).
        let window: Vec<f64> = (0..nperseg)
            .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / nperseg as f64).cos())
            .collect();
        let win_pow = window.iter().map(|w| w * w).sum();
        WelchPlan {
            nperseg,
            nfft,
            window,
            win_pow,
            fft: FftPlan::new(nfft),
            buf_in: vec![C::ZERO; nfft],
            buf_out: vec![C::ZERO; nfft],
        }
    }
}

impl PsdEngine {
    /// New engine with an empty plan cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Welch PSD of `signal` (see [`PsdConfig`]).
    ///
    /// Errors: [`SqiError::EmptyInput`], [`SqiError::NonFinite`], [`SqiError::TooShort`],
    /// [`SqiError::InvalidParameter`].
    pub fn psd(&mut self, signal: &[f64], fs: f64, cfg: &PsdConfig) -> SqiResult<Psd> {
        check_fs(fs)?;
        check_finite(signal)?;
        let n = signal.len();
        let p = WelchParams::resolve(cfg, n, fs)?;
        let reuse = matches!(&self.plan, Some(pl) if pl.nperseg == p.nperseg && pl.nfft == p.nfft);
        if !reuse {
            self.plan = Some(WelchPlan::new(p.nperseg, p.nfft));
        }
        let plan = self.plan.as_mut().expect("plan set above");

        let global_mean = if p.demean_global {
            signal.iter().sum::<f64>() / n as f64
        } else {
            0.0
        };
        let step = p.nperseg - p.noverlap;
        let n_freq = p.nfft / 2 + 1;
        let mut power = vec![0.0; n_freq];
        let mut n_segments = 0usize;
        let mut start = 0usize;
        while start + p.nperseg <= n {
            let seg = &signal[start..start + p.nperseg];
            let offset = if p.detrend_segments {
                seg.iter().sum::<f64>() / p.nperseg as f64
            } else {
                global_mean
            };
            for (dst, (&x, &w)) in plan.buf_in.iter_mut().zip(seg.iter().zip(&plan.window)) {
                *dst = C::new((x - offset) * w, 0.0);
            }
            for dst in plan.buf_in[p.nperseg..].iter_mut() {
                *dst = C::ZERO;
            }
            plan.fft.process(&plan.buf_in, &mut plan.buf_out);
            for (acc, c) in power.iter_mut().zip(&plan.buf_out[..n_freq]) {
                *acc += c.norm_sqr();
            }
            n_segments += 1;
            start += step;
        }
        if n_segments == 0 {
            return Err(SqiError::TooShort);
        }
        let scale = 1.0 / (fs * plan.win_pow * n_segments as f64);
        let last_doubled = if p.nfft % 2 == 0 { n_freq - 1 } else { n_freq };
        for (k, v) in power.iter_mut().enumerate() {
            *v *= scale;
            if k != 0 && k < last_doubled {
                *v *= 2.0;
            }
        }
        // numpy.fft.rfftfreq: k * (1 / (n * d)), d = 1 / fs — same operation order for identical
        // bin edges (matters for the half-open band limits).
        let val = 1.0 / (p.nfft as f64 * (1.0 / fs));
        let freqs = (0..n_freq).map(|k| k as f64 * val).collect();
        Ok(Psd {
            freqs,
            power,
            min_frequency: p.min_frequency,
            nperseg: p.nperseg,
            nfft: p.nfft,
            n_segments,
        })
    }
}

/// One-shot Welch PSD (builds a fresh plan; prefer [`PsdEngine`] in loops).
pub fn psd(signal: &[f64], fs: f64, cfg: &PsdConfig) -> SqiResult<Psd> {
    PsdEngine::new().psd(signal, fs, cfg)
}

/// pSQI = P(5–15 Hz) / P(5–40 Hz) from an existing PSD (Zhao & Zhang 2018; NeuroKit2
/// `_ecg_quality_pSQI`). Relative QRS energy: ~0.5–0.8 for clean ECG at normal heart rate.
pub fn psqi_from_psd(psd: &Psd) -> SqiResult<f64> {
    let num = psd.band_power(PSQI_NUM_BAND.0, PSQI_NUM_BAND.1)?;
    let den = psd.band_power(PSQI_DEN_BAND.0, PSQI_DEN_BAND.1)?;
    Ok(num / den)
}

/// basSQI = 1 − P(0–1 Hz) / P(0–40 Hz) from an existing PSD (Zhao & Zhang 2018; NeuroKit2
/// `_ecg_quality_basSQI`). Close to 1 without baseline wander.
pub fn bassqi_from_psd(psd: &Psd) -> SqiResult<f64> {
    let num = psd.band_power(BASSQI_NUM_BAND.0, BASSQI_NUM_BAND.1)?;
    let den = psd.band_power(BASSQI_DEN_BAND.0, BASSQI_DEN_BAND.1)?;
    Ok(1.0 - num / den)
}

/// pSQI of a signal (computes its own PSD; use [`spectral_sqi`] to get basSQI from the same PSD).
pub fn psqi(signal: &[f64], fs: f64, cfg: &PsdConfig) -> SqiResult<f64> {
    psqi_from_psd(&psd(signal, fs, cfg)?)
}

/// basSQI of a signal (computes its own PSD; use [`spectral_sqi`] to share it with pSQI).
pub fn bassqi(signal: &[f64], fs: f64, cfg: &PsdConfig) -> SqiResult<f64> {
    bassqi_from_psd(&psd(signal, fs, cfg)?)
}

/// pSQI and basSQI from one shared PSD.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct SpectralSqi {
    /// See [`psqi_from_psd`].
    pub psqi: SqiResult<f64>,
    /// See [`bassqi_from_psd`].
    pub bassqi: SqiResult<f64>,
}

/// pSQI + basSQI with a single Welch PSD. The outer error is a PSD failure (bad input);
/// the inner results carry band-specific failures.
pub fn spectral_sqi(signal: &[f64], fs: f64, cfg: &PsdConfig) -> SqiResult<SpectralSqi> {
    let p = psd(signal, fs, cfg)?;
    Ok(SpectralSqi {
        psqi: psqi_from_psd(&p),
        bassqi: bassqi_from_psd(&p),
    })
}

// =============================================================================
// bSQI — agreement between two beat detectors
// =============================================================================

/// Default bSQI matching tolerance (s), Li, Mark & Clifford (2008) / Clifford et al. (2012).
/// Note: NeuroKit2's `ecg_quality(method="ici")` (Ho et al. 2025) uses a stricter 0.05 s.
pub const DEFAULT_BSQI_TOLERANCE_S: f64 = 0.15;

/// bSQI with the counts behind it.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Bsqi {
    /// `matched / (n_a + n_b − matched)`, 0–1.
    pub bsqi: f64,
    /// One-to-one matched beat pairs.
    pub matched: usize,
    /// Beats from detector A (after sort + dedup).
    pub n_a: usize,
    /// Beats from detector B (after sort + dedup).
    pub n_b: usize,
}

/// bSQI with counts. Beats are matched one-to-one by a greedy two-pointer sweep over the
/// sorted peak lists: a pair matches if `|a − b| <= tolerance_s * fs` samples.
///
/// Errors: [`SqiError::TooFewBeats`] if both lists are empty (undefined); invalid `fs` or
/// negative/non-finite tolerance. One empty list gives `bsqi = 0`.
pub fn bsqi_detail(
    peaks_a: &[usize],
    peaks_b: &[usize],
    fs: f64,
    tolerance_s: f64,
) -> SqiResult<Bsqi> {
    check_fs(fs)?;
    if !(tolerance_s.is_finite() && tolerance_s >= 0.0) {
        return Err(SqiError::InvalidParameter("tolerance_s"));
    }
    let a = sorted_unique(peaks_a);
    let b = sorted_unique(peaks_b);
    if a.is_empty() && b.is_empty() {
        return Err(SqiError::TooFewBeats);
    }
    let tol = tolerance_s * fs;
    let (mut i, mut j, mut matched) = (0usize, 0usize, 0usize);
    while i < a.len() && j < b.len() {
        let d = a[i] as f64 - b[j] as f64;
        if d.abs() <= tol {
            matched += 1;
            i += 1;
            j += 1;
        } else if d < 0.0 {
            i += 1;
        } else {
            j += 1;
        }
    }
    let total = a.len() + b.len() - matched;
    Ok(Bsqi {
        bsqi: matched as f64 / total as f64,
        matched,
        n_a: a.len(),
        n_b: b.len(),
    })
}

/// bSQI: share of beats on which two detectors agree,
/// `matched / (N_a + N_b − matched)` (Li, Mark & Clifford 2008; Clifford et al. 2012).
/// Typical tolerance [`DEFAULT_BSQI_TOLERANCE_S`] (0.15 s). See [`bsqi_detail`].
pub fn bsqi(peaks_a: &[usize], peaks_b: &[usize], fs: f64, tolerance_s: f64) -> SqiResult<f64> {
    bsqi_detail(peaks_a, peaks_b, fs, tolerance_s).map(|r| r.bsqi)
}

fn sorted_unique(p: &[usize]) -> Vec<usize> {
    let mut v = p.to_vec();
    v.sort_unstable();
    v.dedup();
    v
}

// =============================================================================
// Template correlation
// =============================================================================

/// Beat window and minimum beat count for template matching.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct TemplateConfig {
    /// Seconds before the R peak (default 0.2, BioSPPy `extract_heartbeats`).
    pub before_s: f64,
    /// Seconds after the R peak (default 0.4, BioSPPy).
    pub after_s: f64,
    /// Minimum complete beats to build a template from the window itself (default 3).
    pub min_beats: usize,
}

impl Default for TemplateConfig {
    fn default() -> Self {
        TemplateConfig {
            before_s: 0.2,
            after_s: 0.4,
            min_beats: 3,
        }
    }
}

impl TemplateConfig {
    /// `(before, after)` in samples, truncated like BioSPPy's `int(before * fs)`
    /// (102 and 204 at 512 Hz → template length 306).
    pub fn offsets(&self, fs: f64) -> SqiResult<(usize, usize)> {
        check_fs(fs)?;
        if !(self.before_s.is_finite() && self.before_s >= 0.0) {
            return Err(SqiError::InvalidParameter("before_s"));
        }
        if !(self.after_s.is_finite() && self.after_s >= 0.0) {
            return Err(SqiError::InvalidParameter("after_s"));
        }
        let before = (self.before_s * fs) as usize;
        let after = (self.after_s * fs) as usize;
        if before + after < 2 {
            return Err(SqiError::InvalidParameter("before_s + after_s"));
        }
        Ok((before, after))
    }

    /// Template length in samples (`before + after`).
    pub fn len(&self, fs: f64) -> SqiResult<usize> {
        self.offsets(fs).map(|(b, a)| b + a)
    }
}

/// Per-beat correlation against a template.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct TemplateCorrelation {
    /// R peaks whose full window fitted in the signal (same order as `per_beat`).
    pub beat_peaks: Vec<usize>,
    /// Pearson correlation of each beat with the template; NaN for a constant beat.
    pub per_beat: Vec<f64>,
    /// Median of the finite per-beat correlations.
    pub median: f64,
    /// Mean of the finite per-beat correlations.
    pub mean: f64,
    /// Minimum finite per-beat correlation.
    pub min: f64,
    /// The template used (window average, or the externally supplied one). Feed it into a
    /// per-user running/adaptive template if desired.
    pub template: Vec<f64>,
}

fn beat_windows<'a>(
    signal: &'a [f64],
    peaks: &[usize],
    before: usize,
    after: usize,
) -> (Vec<usize>, Vec<&'a [f64]>) {
    let mut used = Vec::new();
    let mut beats = Vec::new();
    for &r in peaks {
        // BioSPPy: skip beats whose window leaves the signal.
        let Some(a) = r.checked_sub(before) else {
            continue;
        };
        let Some(b) = r.checked_add(after) else {
            continue;
        };
        if b > signal.len() {
            continue;
        }
        used.push(r);
        beats.push(&signal[a..b]);
    }
    (used, beats)
}

/// Average beat template: mean of all complete `[r − before, r + after)` windows
/// (BioSPPy `extract_heartbeats` window, −0.2 s … +0.4 s by default).
///
/// Errors: invalid `fs`/config, [`SqiError::EmptyInput`], [`SqiError::NonFinite`],
/// [`SqiError::TooFewBeats`] (fewer than `cfg.min_beats` complete beats; at least 1).
pub fn beat_template(
    signal: &[f64],
    peaks: &[usize],
    fs: f64,
    cfg: &TemplateConfig,
) -> SqiResult<Vec<f64>> {
    let (before, after) = cfg.offsets(fs)?;
    check_finite(signal)?;
    let (_, beats) = beat_windows(signal, peaks, before, after);
    mean_template(&beats, before + after, cfg.min_beats.max(1))
}

fn mean_template(beats: &[&[f64]], len: usize, min_beats: usize) -> SqiResult<Vec<f64>> {
    if beats.len() < min_beats || beats.is_empty() {
        return Err(SqiError::TooFewBeats);
    }
    let mut t = vec![0.0; len];
    for b in beats {
        for (acc, &v) in t.iter_mut().zip(b.iter()) {
            *acc += v;
        }
    }
    let n = beats.len() as f64;
    t.iter_mut().for_each(|v| *v /= n);
    Ok(t)
}

/// Template-matching SQI using the window's own average beat (Orphanidou et al. 2015 style):
/// builds the template with [`beat_template`] and correlates every complete beat with it.
///
/// Errors: as [`beat_template`]; [`SqiError::ZeroVariance`] if the template or every beat is
/// constant.
pub fn template_correlation(
    signal: &[f64],
    peaks: &[usize],
    fs: f64,
    cfg: &TemplateConfig,
) -> SqiResult<TemplateCorrelation> {
    let (before, after) = cfg.offsets(fs)?;
    check_finite(signal)?;
    let (used, beats) = beat_windows(signal, peaks, before, after);
    let template = mean_template(&beats, before + after, cfg.min_beats.max(1))?;
    correlate(used, &beats, template)
}

/// Template-matching SQI against an **externally supplied** template (e.g. a per-user adaptive
/// template). `template.len()` must equal `cfg.len(fs)`; a single complete beat is enough
/// (`cfg.min_beats` is not applied).
///
/// Errors: [`SqiError::TemplateLength`], [`SqiError::NonFinite`] (signal or template),
/// [`SqiError::TooFewBeats`] (no complete beat), [`SqiError::ZeroVariance`].
pub fn template_correlation_with(
    signal: &[f64],
    peaks: &[usize],
    fs: f64,
    cfg: &TemplateConfig,
    template: &[f64],
) -> SqiResult<TemplateCorrelation> {
    let (before, after) = cfg.offsets(fs)?;
    if template.len() != before + after {
        return Err(SqiError::TemplateLength);
    }
    check_finite(template)?;
    check_finite(signal)?;
    let (used, beats) = beat_windows(signal, peaks, before, after);
    if beats.is_empty() {
        return Err(SqiError::TooFewBeats);
    }
    correlate(used, &beats, template.to_vec())
}

fn correlate(
    beat_peaks: Vec<usize>,
    beats: &[&[f64]],
    template: Vec<f64>,
) -> SqiResult<TemplateCorrelation> {
    let n = template.len() as f64;
    let tm = template.iter().sum::<f64>() / n;
    let tc: Vec<f64> = template.iter().map(|v| v - tm).collect();
    let tss: f64 = tc.iter().map(|v| v * v).sum();
    if tss <= 0.0 {
        return Err(SqiError::ZeroVariance);
    }
    let per_beat: Vec<f64> = beats
        .iter()
        .map(|b| {
            let bm = b.iter().sum::<f64>() / n;
            let (mut sxy, mut sxx) = (0.0, 0.0);
            for (&x, &t) in b.iter().zip(&tc) {
                let d = x - bm;
                sxy += d * t;
                sxx += d * d;
            }
            if sxx <= 0.0 {
                f64::NAN
            } else {
                (sxy / (sxx * tss).sqrt()).clamp(-1.0, 1.0)
            }
        })
        .collect();
    let mut finite: Vec<f64> = per_beat.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return Err(SqiError::ZeroVariance);
    }
    finite.sort_by(f64::total_cmp);
    let k = finite.len();
    let median = if k % 2 == 1 {
        finite[k / 2]
    } else {
        0.5 * (finite[k / 2 - 1] + finite[k / 2])
    };
    let mean = finite.iter().sum::<f64>() / k as f64;
    Ok(TemplateCorrelation {
        beat_peaks,
        per_beat,
        median,
        mean,
        min: finite[0],
        template,
    })
}

// =============================================================================
// RR plausibility (Orphanidou et al. 2015)
// =============================================================================

/// Physiological limits for [`rr_plausibility`]; defaults are Orphanidou et al. (2015):
/// HR 40–180 bpm, longest RR gap ≤ 3 s, RRmax/RRmin < 2.2.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct RrLimits {
    /// Lowest acceptable mean heart rate (bpm). Default 40.
    pub min_hr_bpm: f64,
    /// Highest acceptable mean heart rate (bpm). Default 180.
    pub max_hr_bpm: f64,
    /// Longest acceptable gap between beats (s). Default 3.0.
    pub max_gap_s: f64,
    /// RRmax / RRmin must stay below this. Default 2.2.
    pub max_rr_ratio: f64,
}

impl Default for RrLimits {
    fn default() -> Self {
        RrLimits {
            min_hr_bpm: 40.0,
            max_hr_bpm: 180.0,
            max_gap_s: 3.0,
            max_rr_ratio: 2.2,
        }
    }
}

/// RR-interval plausibility values and the Orphanidou-style vetoes.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct RrPlausibility {
    /// Number of (unique) beats.
    pub n_beats: usize,
    /// 60 / mean RR (bpm).
    pub mean_hr_bpm: f64,
    /// Shortest RR (s).
    pub min_rr_s: f64,
    /// Longest RR (s).
    pub max_rr_s: f64,
    /// RRmax / RRmin.
    pub rr_ratio: f64,
    /// Longest beat-free stretch (s): `max_rr_s`, or, if `signal_len` was given, also the gaps
    /// from the window start to the first beat and from the last beat to the window end.
    pub longest_gap_s: f64,
    /// `min_hr_bpm <= mean_hr_bpm <= max_hr_bpm`.
    pub hr_ok: bool,
    /// `longest_gap_s <= max_gap_s`.
    pub gap_ok: bool,
    /// `rr_ratio < max_rr_ratio`.
    pub ratio_ok: bool,
    /// All three checks pass.
    pub plausible: bool,
}

/// RR plausibility of the R peaks of one window (Orphanidou et al. 2015 feasibility rules,
/// returned as values; the decision stays with the caller).
///
/// `signal_len` (samples), if given, extends the gap check to the window edges, which catches a
/// beat-free tail that inter-beat RR intervals cannot see.
///
/// Errors: invalid `fs`, [`SqiError::TooFewBeats`] (fewer than 2 unique peaks).
pub fn rr_plausibility(
    peaks: &[usize],
    fs: f64,
    signal_len: Option<usize>,
    limits: &RrLimits,
) -> SqiResult<RrPlausibility> {
    check_fs(fs)?;
    let p = sorted_unique(peaks);
    if p.len() < 2 {
        return Err(SqiError::TooFewBeats);
    }
    let rr: Vec<f64> = p.windows(2).map(|w| (w[1] - w[0]) as f64 / fs).collect();
    let min_rr = rr.iter().copied().fold(f64::INFINITY, f64::min);
    let max_rr = rr.iter().copied().fold(0.0, f64::max);
    let mean_rr = rr.iter().sum::<f64>() / rr.len() as f64;
    let mean_hr = 60.0 / mean_rr;
    let ratio = max_rr / min_rr;
    let mut gap = max_rr;
    if let Some(len) = signal_len {
        let first = p[0] as f64 / fs;
        let last = len.saturating_sub(p[p.len() - 1]) as f64 / fs;
        gap = gap.max(first).max(last);
    }
    let hr_ok = mean_hr >= limits.min_hr_bpm && mean_hr <= limits.max_hr_bpm;
    let gap_ok = gap <= limits.max_gap_s;
    let ratio_ok = ratio < limits.max_rr_ratio;
    Ok(RrPlausibility {
        n_beats: p.len(),
        mean_hr_bpm: mean_hr,
        min_rr_s: min_rr,
        max_rr_s: max_rr,
        rr_ratio: ratio,
        longest_gap_s: gap,
        hr_ok,
        gap_ok,
        ratio_ok,
        plausible: hr_ok && gap_ok && ratio_ok,
    })
}

// =============================================================================
// Zhao 2018 simple heuristic fusion
// =============================================================================

/// pSQI acceptance band for one heart-rate regime.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct PsqiBand {
    /// Optimal if `optimal_lo < pSQI < optimal_hi`.
    pub optimal_lo: f64,
    /// See `optimal_lo`.
    pub optimal_hi: f64,
    /// Suspicious if `suspicious_lo < pSQI < optimal_lo`; otherwise unqualified.
    pub suspicious_lo: f64,
}

/// Thresholds of the zhao2018 simple fusion. Defaults: NeuroKit2 0.3.0 `_ecg_quality_zhao2018`
/// (from Zhao & Zhang 2018): HR split 130 bpm; pSQI optimal (0.5, 0.8) / suspicious (0.4, 0.5)
/// below the split, (0.4, 0.7) / (0.3, 0.4) above; kSQI > 5; basSQI ≥ 0.95 optimal, < 0.90
/// unqualified.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct ZhaoThresholds {
    /// Heart rate (bpm) at/above which `psqi_high_hr` applies. Default 130.
    pub hr_split_bpm: f64,
    /// pSQI band for HR < split.
    pub psqi_low_hr: PsqiBand,
    /// pSQI band for HR ≥ split.
    pub psqi_high_hr: PsqiBand,
    /// kSQI optimal if `kSQI > ksqi_min` (Pearson scale), else unqualified. Default 5.
    pub ksqi_min: f64,
    /// basSQI optimal if `>= bassqi_optimal`. Default 0.95.
    pub bassqi_optimal: f64,
    /// basSQI unqualified if `< bassqi_suspicious`, suspicious in between. Default 0.90.
    pub bassqi_suspicious: f64,
}

impl Default for ZhaoThresholds {
    fn default() -> Self {
        ZhaoThresholds {
            hr_split_bpm: 130.0,
            psqi_low_hr: PsqiBand {
                optimal_lo: 0.5,
                optimal_hi: 0.8,
                suspicious_lo: 0.4,
            },
            psqi_high_hr: PsqiBand {
                optimal_lo: 0.4,
                optimal_hi: 0.7,
                suspicious_lo: 0.3,
            },
            ksqi_min: 5.0,
            bassqi_optimal: 0.95,
            bassqi_suspicious: 0.90,
        }
    }
}

/// Grade of a single index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub enum IndexGrade {
    /// "optimal" (class 2 in NeuroKit2).
    Optimal,
    /// "suspicious" (class 1).
    Suspicious,
    /// "unqualified" (class 0). Also used for NaN/missing values.
    Unqualified,
}

/// Which index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub enum SqiIndex {
    /// Kurtosis.
    Ksqi,
    /// Relative QRS power.
    Psqi,
    /// Baseline relative power.
    Bassqi,
}

/// Why the fusion did not rate a window as fully optimal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub enum ZhaoReason {
    /// The index was NaN/infinite (counted as unqualified).
    NotComputable(SqiIndex),
    /// The index fell in its unqualified range.
    Unqualified(SqiIndex),
    /// The index fell in its suspicious range.
    Suspicious(SqiIndex),
}

/// zhao2018 3-class outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub enum ZhaoClass {
    /// ≥ 2 optimal and no unqualified index.
    Excellent,
    /// Anything between.
    BarelyAcceptable,
    /// ≥ 2 unqualified, or 1 unqualified + 2 suspicious.
    Unacceptable,
}

impl ZhaoClass {
    /// NeuroKit2's string for this class.
    pub fn as_neurokit_str(&self) -> &'static str {
        match self {
            ZhaoClass::Excellent => "Excellent",
            ZhaoClass::BarelyAcceptable => "Barely acceptable",
            ZhaoClass::Unacceptable => "Unacceptable",
        }
    }
}

/// Fusion outcome with per-index grades and reasons.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct ZhaoResult {
    /// Final class.
    pub class: ZhaoClass,
    /// kSQI grade.
    pub ksqi: IndexGrade,
    /// pSQI grade (band chosen by heart rate).
    pub psqi: IndexGrade,
    /// basSQI grade.
    pub bassqi: IndexGrade,
    /// Every index that was not optimal, and why.
    pub reasons: Vec<ZhaoReason>,
}

/// Heart rate the way NeuroKit2's zhao2018 picks the pSQI band: `60 · fs / min(RR samples)`,
/// i.e. the **maximum instantaneous** HR, not the mean. `None` for < 2 unique peaks (NeuroKit2
/// then uses 1 bpm, i.e. the low-HR band; [`zhao2018_fusion`] does the same for a NaN HR).
pub fn zhao_heart_rate_bpm(peaks: &[usize], fs: f64) -> Option<f64> {
    if check_fs(fs).is_err() {
        return None;
    }
    let p = sorted_unique(peaks);
    let min_rr = p.windows(2).map(|w| w[1] - w[0]).min()?;
    Some(60.0 * fs / min_rr as f64)
}

/// Zhao & Zhang (2018) simple heuristic fusion of kSQI, pSQI and basSQI (qSQI omitted, as in
/// NeuroKit2), with thresholds as parameters ([`ZhaoThresholds::default`] = NeuroKit2's).
///
/// Comparisons replicate NeuroKit2 exactly, including its strict inequalities (pSQI exactly at a
/// band edge is unqualified). Non-finite index values grade as unqualified (NeuroKit2 would grade
/// a NaN basSQI as suspicious; this is deliberately stricter). A non-finite `hr_bpm` selects the
/// low-HR pSQI band.
///
/// Inputs must be on the scales of [`ksqi`] (Pearson), [`psqi_from_psd`] and [`bassqi_from_psd`].
/// For NeuroKit2-identical class labels compute pSQI/basSQI with [`PsdConfig::neurokit`] and HR
/// with [`zhao_heart_rate_bpm`].
pub fn zhao2018_fusion(
    ksqi: f64,
    psqi: f64,
    bassqi: f64,
    hr_bpm: f64,
    th: &ZhaoThresholds,
) -> ZhaoResult {
    let mut reasons = Vec::new();
    let mut grade = |idx: SqiIndex, v: f64, g: &dyn Fn(f64) -> IndexGrade| -> IndexGrade {
        if !v.is_finite() {
            reasons.push(ZhaoReason::NotComputable(idx));
            return IndexGrade::Unqualified;
        }
        let r = g(v);
        match r {
            IndexGrade::Optimal => {}
            IndexGrade::Suspicious => reasons.push(ZhaoReason::Suspicious(idx)),
            IndexGrade::Unqualified => reasons.push(ZhaoReason::Unqualified(idx)),
        }
        r
    };
    let band = if hr_bpm.is_finite() && hr_bpm >= th.hr_split_bpm {
        th.psqi_high_hr
    } else {
        th.psqi_low_hr
    };
    let g_p = grade(SqiIndex::Psqi, psqi, &|p| {
        if p > band.optimal_lo && p < band.optimal_hi {
            IndexGrade::Optimal
        } else if p > band.suspicious_lo && p < band.optimal_lo {
            IndexGrade::Suspicious
        } else {
            IndexGrade::Unqualified
        }
    });
    let g_k = grade(SqiIndex::Ksqi, ksqi, &|k| {
        if k > th.ksqi_min {
            IndexGrade::Optimal
        } else {
            IndexGrade::Unqualified
        }
    });
    let g_b = grade(SqiIndex::Bassqi, bassqi, &|b| {
        if b >= th.bassqi_optimal {
            IndexGrade::Optimal
        } else if b < th.bassqi_suspicious {
            IndexGrade::Unqualified
        } else {
            IndexGrade::Suspicious
        }
    });
    let grades = [g_p, g_k, g_b];
    let count = |g: IndexGrade| grades.iter().filter(|&&x| x == g).count();
    let (n_opt, n_sus, n_unq) = (
        count(IndexGrade::Optimal),
        count(IndexGrade::Suspicious),
        count(IndexGrade::Unqualified),
    );
    let class = if n_unq >= 2 || (n_unq == 1 && n_sus == 2) {
        ZhaoClass::Unacceptable
    } else if n_opt >= 2 && n_unq == 0 {
        ZhaoClass::Excellent
    } else {
        ZhaoClass::BarelyAcceptable
    };
    ZhaoResult {
        class,
        ksqi: g_k,
        psqi: g_p,
        bassqi: g_b,
        reasons,
    }
}

// =============================================================================
// Window report
// =============================================================================

/// Configuration of [`SqiEngine`] / [`sqi_report`]. Defaults are the documented literature /
/// library defaults; product gates are not part of it.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct SqiConfig {
    /// PSD for pSQI/basSQI (default SciPy Welch, nperseg 2048).
    pub psd: PsdConfig,
    /// [`flatline_fraction`] `eps` (default [`DEFAULT_FLATLINE_EPS`]).
    pub flatline_eps: f64,
    /// [`flatline_fraction`] `min_duration_s` (default [`DEFAULT_FLATLINE_MIN_DURATION_S`]).
    pub flatline_min_duration_s: f64,
    /// ADC code range `(min, max)` for [`saturation_fraction`]; `None` skips the check.
    pub adc_range: Option<(f64, f64)>,
    /// [`saturation_fraction`] margin in counts (default 0).
    pub saturation_margin: f64,
    /// bSQI tolerance (default [`DEFAULT_BSQI_TOLERANCE_S`]).
    pub bsqi_tolerance_s: f64,
    /// Template window.
    pub template: TemplateConfig,
    /// RR plausibility limits.
    pub rr_limits: RrLimits,
    /// zhao2018 fusion thresholds.
    pub zhao: ZhaoThresholds,
}

impl Default for SqiConfig {
    fn default() -> Self {
        SqiConfig {
            psd: PsdConfig::default(),
            flatline_eps: DEFAULT_FLATLINE_EPS,
            flatline_min_duration_s: DEFAULT_FLATLINE_MIN_DURATION_S,
            adc_range: None,
            saturation_margin: 0.0,
            bsqi_tolerance_s: DEFAULT_BSQI_TOLERANCE_S,
            template: TemplateConfig::default(),
            rr_limits: RrLimits::default(),
            zhao: ZhaoThresholds::default(),
        }
    }
}

/// Inputs for one window.
#[derive(Debug, Clone, Copy)]
pub struct SqiInput<'a> {
    /// Signal the morphology/spectral indices are computed on (usually the cleaned ECG).
    pub signal: &'a [f64],
    /// Sampling rate (Hz).
    pub fs: f64,
    /// R peaks of the primary detector (sample indices into `signal`).
    pub peaks: &'a [usize],
    /// R peaks of a second detector, for bSQI.
    pub peaks_secondary: Option<&'a [usize]>,
    /// Raw ADC counts of the same window, for flat-line / saturation checks.
    pub raw: Option<&'a [i32]>,
    /// External (e.g. per-user adaptive) template; otherwise the window's own average beat.
    pub template: Option<&'a [f64]>,
}

impl<'a> SqiInput<'a> {
    /// Minimal input: signal, sampling rate and primary peaks.
    pub fn new(signal: &'a [f64], fs: f64, peaks: &'a [usize]) -> Self {
        SqiInput {
            signal,
            fs,
            peaks,
            peaks_secondary: None,
            raw: None,
            template: None,
        }
    }
}

/// Summary of [`TemplateCorrelation`] kept in the report (per-beat values via
/// [`template_correlation`]).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct TemplateSummary {
    /// Median per-beat correlation.
    pub median: f64,
    /// Mean per-beat correlation.
    pub mean: f64,
    /// Minimum per-beat correlation.
    pub min: f64,
    /// Beats correlated.
    pub n_beats: usize,
    /// Whether an external template was used.
    pub external_template: bool,
}

/// An index that could not be computed, and why.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct SqiIssue {
    /// Report field name.
    pub field: &'static str,
    /// Reason.
    pub error: SqiError,
}

/// All indices for one window. `None` = not computed (missing input) or not computable
/// (see `issues`).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct SqiReport {
    /// Window length (samples).
    pub n_samples: usize,
    /// Sampling rate (Hz).
    pub fs: f64,
    /// Flat-line fraction of `raw`.
    pub flatline_fraction: Option<f64>,
    /// Saturation fraction of `raw` (needs `adc_range`).
    pub saturation_fraction: Option<f64>,
    /// Pearson kurtosis.
    pub ksqi: Option<f64>,
    /// pSQI.
    pub psqi: Option<f64>,
    /// basSQI.
    pub bassqi: Option<f64>,
    /// bSQI (needs `peaks_secondary`).
    pub bsqi: Option<f64>,
    /// Template correlation summary.
    pub template: Option<TemplateSummary>,
    /// RR plausibility.
    pub rr: Option<RrPlausibility>,
    /// HR used by the fusion ([`zhao_heart_rate_bpm`]).
    pub zhao_hr_bpm: Option<f64>,
    /// zhao2018 fusion (always present; missing indices grade as not computable).
    pub zhao: ZhaoResult,
    /// Why some field is `None` although its input was given.
    pub issues: Vec<SqiIssue>,
}

/// Reusable per-worker SQI calculator (caches the PSD plan).
#[derive(Debug, Default)]
pub struct SqiEngine {
    /// Configuration used by [`SqiEngine::report`].
    pub config: SqiConfig,
    psd: PsdEngine,
}

impl SqiEngine {
    /// New engine.
    pub fn new(config: SqiConfig) -> Self {
        SqiEngine {
            config,
            psd: PsdEngine::new(),
        }
    }

    /// Compute every index for one window. Never panics; failures land in `issues`.
    pub fn report(&mut self, input: &SqiInput<'_>) -> SqiReport {
        let cfg = self.config;
        let mut issues = Vec::new();
        let mut keep = |field: &'static str, r: SqiResult<f64>| -> Option<f64> {
            r.map_err(|error| issues.push(SqiIssue { field, error }))
                .ok()
        };

        let flatline_fraction = input.raw.and_then(|raw| {
            keep(
                "flatline_fraction",
                flatline_fraction(raw, input.fs, cfg.flatline_eps, cfg.flatline_min_duration_s),
            )
        });
        let saturation_fraction = match (input.raw, cfg.adc_range) {
            (Some(raw), Some((lo, hi))) => keep(
                "saturation_fraction",
                saturation_fraction(raw, lo, hi, cfg.saturation_margin),
            ),
            _ => None,
        };
        let ksqi_v = keep("ksqi", ksqi(input.signal));
        let (psqi_v, bassqi_v) = match self.psd.psd(input.signal, input.fs, &cfg.psd) {
            Ok(p) => (
                keep("psqi", psqi_from_psd(&p)),
                keep("bassqi", bassqi_from_psd(&p)),
            ),
            Err(e) => {
                keep("psqi", Err(e));
                keep("bassqi", Err(e));
                (None, None)
            }
        };
        let bsqi_v = input
            .peaks_secondary
            .and_then(|b| keep("bsqi", bsqi(input.peaks, b, input.fs, cfg.bsqi_tolerance_s)));

        let tc = match input.template {
            Some(t) => {
                template_correlation_with(input.signal, input.peaks, input.fs, &cfg.template, t)
            }
            None => template_correlation(input.signal, input.peaks, input.fs, &cfg.template),
        };
        let template = match tc {
            Ok(tc) => Some(TemplateSummary {
                median: tc.median,
                mean: tc.mean,
                min: tc.min,
                n_beats: tc.per_beat.iter().filter(|v| v.is_finite()).count(),
                external_template: input.template.is_some(),
            }),
            Err(error) => {
                issues.push(SqiIssue {
                    field: "template",
                    error,
                });
                None
            }
        };
        let rr = match rr_plausibility(
            input.peaks,
            input.fs,
            Some(input.signal.len()),
            &cfg.rr_limits,
        ) {
            Ok(r) => Some(r),
            Err(error) => {
                issues.push(SqiIssue { field: "rr", error });
                None
            }
        };
        let zhao_hr_bpm = zhao_heart_rate_bpm(input.peaks, input.fs);
        let zhao = zhao2018_fusion(
            ksqi_v.unwrap_or(f64::NAN),
            psqi_v.unwrap_or(f64::NAN),
            bassqi_v.unwrap_or(f64::NAN),
            zhao_hr_bpm.unwrap_or(f64::NAN),
            &cfg.zhao,
        );
        SqiReport {
            n_samples: input.signal.len(),
            fs: input.fs,
            flatline_fraction,
            saturation_fraction,
            ksqi: ksqi_v,
            psqi: psqi_v,
            bassqi: bassqi_v,
            bsqi: bsqi_v,
            template,
            rr,
            zhao_hr_bpm,
            zhao,
            issues,
        }
    }
}

/// One-shot [`SqiEngine::report`] (prefer a long-lived engine in loops).
pub fn sqi_report(input: &SqiInput<'_>, config: &SqiConfig) -> SqiReport {
    SqiEngine::new(*config).report(input)
}

// =============================================================================
// FFT (mixed radix + Bluestein), private
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct C {
    re: f64,
    im: f64,
}

impl C {
    const ZERO: C = C { re: 0.0, im: 0.0 };
    #[inline]
    fn new(re: f64, im: f64) -> Self {
        C { re, im }
    }
    #[inline]
    fn mul(self, o: C) -> C {
        C::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
    #[inline]
    fn add(self, o: C) -> C {
        C::new(self.re + o.re, self.im + o.im)
    }
    #[inline]
    fn sub(self, o: C) -> C {
        C::new(self.re - o.re, self.im - o.im)
    }
    #[inline]
    fn conj(self) -> C {
        C::new(self.re, -self.im)
    }
    #[inline]
    fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
    #[inline]
    fn expi(theta: f64) -> C {
        C::new(theta.cos(), theta.sin())
    }
}

/// Largest radix handled by the generic O(p²) butterfly; bigger prime factors go to Bluestein.
const MAX_DIRECT_RADIX: usize = 61;

#[derive(Debug)]
enum FftPlan {
    Mixed {
        n: usize,
        /// (radix p, remaining length m) per stage, kissfft style.
        stages: Vec<(usize, usize)>,
        twiddles: Vec<C>,
    },
    Bluestein {
        n: usize,
        m: usize,
        chirp: Vec<C>,
        kernel_fft: Vec<C>,
        inner: Box<FftPlan>,
    },
}

impl FftPlan {
    // `%` instead of `is_multiple_of`: the crate MSRV (1.85) predates its stabilisation (1.87).
    #[allow(clippy::manual_is_multiple_of)]
    fn new(n: usize) -> Self {
        let n = n.max(1);
        let mut stages = Vec::new();
        let mut rest = n;
        let mut largest = 1;
        for p in [4usize, 2, 3, 5] {
            while rest % p == 0 {
                rest /= p;
                stages.push((p, rest));
            }
        }
        let mut p = 7;
        while rest > 1 {
            if p * p > rest {
                p = rest;
            }
            while rest % p == 0 {
                rest /= p;
                stages.push((p, rest));
                largest = largest.max(p);
            }
            p += 2;
        }
        if largest <= MAX_DIRECT_RADIX {
            let twiddles = (0..n)
                .map(|k| C::expi(-2.0 * PI * k as f64 / n as f64))
                .collect();
            if stages.is_empty() {
                stages.push((1, 1));
            }
            return FftPlan::Mixed {
                n,
                stages,
                twiddles,
            };
        }
        // Bluestein: X_k = w_k · Σ (x_j w_j) conj(w_{k−j}),  w_k = exp(−iπ k²/n).
        let m = (2 * n - 1).next_power_of_two();
        let two_n = 2 * n as u128;
        let chirp: Vec<C> = (0..n)
            .map(|k| {
                let k2 = ((k as u128 * k as u128) % two_n) as f64;
                C::expi(-PI * k2 / n as f64)
            })
            .collect();
        let inner = Box::new(FftPlan::new(m));
        let mut kernel = vec![C::ZERO; m];
        kernel[0] = chirp[0].conj();
        for k in 1..n {
            kernel[k] = chirp[k].conj();
            kernel[m - k] = chirp[k].conj();
        }
        let mut kernel_fft = vec![C::ZERO; m];
        inner.process(&kernel, &mut kernel_fft);
        FftPlan::Bluestein {
            n,
            m,
            chirp,
            kernel_fft,
            inner,
        }
    }

    /// Forward DFT: `out[k] = Σ_j input[j] · exp(−2πi jk / n)`. Both slices have length n.
    fn process(&self, input: &[C], out: &mut [C]) {
        match self {
            FftPlan::Mixed {
                n,
                stages,
                twiddles,
            } => {
                if stages[0].0 == 1 {
                    out[0] = input[0];
                    return;
                }
                let mut scratch = vec![C::ZERO; stages.iter().map(|s| s.0).max().unwrap_or(1)];
                fft_work(&mut out[..*n], input, 0, 1, stages, twiddles, &mut scratch);
            }
            FftPlan::Bluestein {
                n,
                m,
                chirp,
                kernel_fft,
                inner,
            } => {
                let (n, m) = (*n, *m);
                let mut a = vec![C::ZERO; m];
                for (dst, (x, w)) in a.iter_mut().zip(input.iter().zip(chirp)) {
                    *dst = x.mul(*w);
                }
                let mut af = vec![C::ZERO; m];
                inner.process(&a, &mut af);
                // inverse FFT via conj(FFT(conj(.))) / m
                for (v, k) in af.iter_mut().zip(kernel_fft) {
                    *v = v.mul(*k).conj();
                }
                inner.process(&af, &mut a);
                let inv_m = 1.0 / m as f64;
                for k in 0..n {
                    let c = a[k].conj();
                    out[k] = C::new(c.re * inv_m, c.im * inv_m).mul(chirp[k]);
                }
            }
        }
    }
}

/// kissfft-style recursive decimation in time.
fn fft_work(
    out: &mut [C],
    input: &[C],
    start: usize,
    fstride: usize,
    stages: &[(usize, usize)],
    tw: &[C],
    scratch: &mut [C],
) {
    let (p, m) = stages[0];
    if m == 1 {
        for (q, o) in out.iter_mut().take(p).enumerate() {
            *o = input[start + q * fstride];
        }
    } else {
        for q in 0..p {
            fft_work(
                &mut out[q * m..(q + 1) * m],
                input,
                start + q * fstride,
                fstride * p,
                &stages[1..],
                tw,
                scratch,
            );
        }
    }
    let n = tw.len();
    match p {
        2 => {
            let (lo, hi) = out.split_at_mut(m);
            for (u, (a, b)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
                let t = b.mul(tw[u * fstride]);
                *b = a.sub(t);
                *a = a.add(t);
            }
        }
        4 => {
            for u in 0..m {
                let x0 = out[u];
                let x1 = out[u + m].mul(tw[u * fstride]);
                let x2 = out[u + 2 * m].mul(tw[2 * u * fstride]);
                let x3 = out[u + 3 * m].mul(tw[3 * u * fstride]);
                let s02 = x0.add(x2);
                let d02 = x0.sub(x2);
                let s13 = x1.add(x3);
                let d13 = x1.sub(x3);
                // forward: multiply d13 by -i
                let d13_mi = C::new(d13.im, -d13.re);
                out[u] = s02.add(s13);
                out[u + m] = d02.add(d13_mi);
                out[u + 2 * m] = s02.sub(s13);
                out[u + 3 * m] = d02.sub(d13_mi);
            }
        }
        _ => {
            for u in 0..m {
                for q1 in 0..p {
                    scratch[q1] = out[u + q1 * m];
                }
                for q1 in 0..p {
                    let k = u + q1 * m;
                    let mut acc = scratch[0];
                    let mut twidx = 0usize;
                    for s in scratch.iter().take(p).skip(1) {
                        twidx += fstride * k;
                        twidx %= n;
                        acc = acc.add(s.mul(tw[twidx]));
                    }
                    out[k] = acc;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dft(x: &[C]) -> Vec<C> {
        let n = x.len();
        (0..n)
            .map(|k| {
                x.iter().enumerate().fold(C::ZERO, |acc, (j, v)| {
                    acc.add(v.mul(C::expi(-2.0 * PI * ((j * k) % n) as f64 / n as f64)))
                })
            })
            .collect()
    }

    #[test]
    fn fft_matches_naive_dft_for_many_sizes() {
        for &n in &[
            1usize, 2, 3, 4, 5, 6, 7, 8, 12, 16, 30, 49, 64, 67, 97, 100, 127, 210, 256, 1000, 1031,
        ] {
            let x: Vec<C> = (0..n)
                .map(|i| {
                    C::new(
                        (i as f64 * 0.37).sin() + 0.1 * i as f64,
                        (i as f64 * 1.3).cos(),
                    )
                })
                .collect();
            let want = naive_dft(&x);
            let mut got = vec![C::ZERO; n];
            FftPlan::new(n).process(&x, &mut got);
            let scale = want.iter().map(|c| c.norm_sqr().sqrt()).fold(1.0, f64::max);
            for (a, b) in got.iter().zip(&want) {
                assert!((a.re - b.re).abs() < 1e-9 * scale, "n={n}");
                assert!((a.im - b.im).abs() < 1e-9 * scale, "n={n}");
            }
        }
    }
}
