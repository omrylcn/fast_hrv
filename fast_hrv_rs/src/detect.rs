//! R-peak detection beyond the NeuroKit2 default method.
//!
//! - [`elgendi_peaks`]: the Elgendi two-moving-average QRS detector (8–20 Hz Butterworth band-pass,
//!   squaring, ~97 ms and ~611 ms moving averages, offset β·mean, block threshold), with presets that
//!   reproduce NeuroKit2's `elgendi2010` and `elgendi2013` exactly.
//! - [`refine_peaks`]: moves candidate peaks (from any detector) onto the R extremum of the clean
//!   signal within ±50 ms and adds 3-point parabolic interpolation (sub-sample positions). Use
//!   [`crate::rri_from_positions`] / [`crate::rri_from_times`] for RR intervals from them.
//!
//! All durations are given in seconds and converted with the sampling rate, so the detector is not
//! tied to one sampling rate. Functions are stateless: the same input always gives the same peaks.

use crate::error::HrvError;
use crate::filter::{BandType, SosSection, butter_sos, sosfilt, sosfilt_zi, sosfiltfilt};
use crate::peaks::numpy_sum;

fn check_fs(fs: f64) -> Result<(), HrvError> {
    if fs.is_finite() && fs > 0.0 {
        Ok(())
    } else {
        Err(HrvError::InvalidSamplingRate(fs))
    }
}

fn check_finite(x: &[f64]) -> Result<(), HrvError> {
    if x.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(HrvError::NonFinite)
    }
}

/// `np.mean` on float64 (pairwise summation), for parity with NeuroKit2.
fn numpy_mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        f64::NAN
    } else {
        numpy_sum(x) / x.len() as f64
    }
}

// =============================================================================
// Elgendi two-moving-average detector
// =============================================================================

/// How the detector-path band-pass is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DetectorFilter {
    /// Causal `sosfilt` with the initial state `sosfilt_zi · mean(signal)`. This is NeuroKit2's
    /// `signal_filter(method="butterworth_zi")`, used by `ecg_clean(method="elgendi2010")`. The
    /// filter delay makes the raw detections late (see [`refine_peaks`]).
    Causal,
    /// Zero-phase `sosfiltfilt` (SciPy padding). Not NeuroKit2-compatible; removes the filter delay.
    ZeroPhase,
}

/// Energy signal fed to the two moving averages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ElgendiEnergy {
    /// `x²` (Elgendi 2013).
    Squared,
    /// `|x|` (NeuroKit2 `elgendi2010`, ported from py-ecg-detectors).
    Abs,
}

/// Block bookkeeping. The two NeuroKit2 implementations differ in small details; both are kept so
/// each can be reproduced exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ElgendiBlocks {
    /// Elgendi (2013) / NeuroKit2 `elgendi2013`: a block is a run of samples with
    /// `MA_qrs > MA_beat + β·mean(energy)`; it is kept if its width (samples) is at least the minimum
    /// block width; the peak is the sample of largest `|x|` inside the block. Window lengths are
    /// rounded (`round(t·fs)`, half to even like Python).
    Elgendi2013,
    /// NeuroKit2 `elgendi2010` (py-ecg-detectors port): block start = last sample *below* the
    /// threshold, block end = last sample above minus one, kept if `end − start > min_block`;
    /// the peak is the largest *signed* sample of the block. Window lengths are truncated
    /// (`int(t·fs)`).
    NeuroKit2010,
}

/// Parameters of the Elgendi detector. All durations in seconds.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ElgendiConfig {
    /// Detector-path band-pass (Hz). Default 8–20 Hz (Elgendi et al. 2010).
    pub band_hz: (f64, f64),
    /// Butterworth order in SciPy semantics (a band-pass of order `n` has `2n` poles). Default 2
    /// (NeuroKit2 `_ecg_clean_elgendi`).
    pub filter_order: usize,
    /// Causal (NeuroKit2) or zero-phase band-pass. Default [`DetectorFilter::Causal`].
    pub filter: DetectorFilter,
    /// Energy signal. Default [`ElgendiEnergy::Squared`].
    pub energy: ElgendiEnergy,
    /// QRS moving-average window. Default 0.097 s (Elgendi 2013).
    pub qrs_window_s: f64,
    /// Beat moving-average window. Default 0.611 s (Elgendi 2013).
    pub beat_window_s: f64,
    /// Offset β: the threshold is `MA_beat + β · mean(energy)` over the analysed input.
    /// Default 0.08 (Elgendi 2013).
    pub beta: f64,
    /// Minimum block width. `None` = the QRS window (Elgendi 2013).
    pub min_block_s: Option<f64>,
    /// Refractory period: a detection closer than this to the previous *accepted* detection is
    /// dropped (the earlier one is kept). Default 0.3 s (NeuroKit2's `mindelay`, 200 bpm).
    /// `0` disables it.
    pub refractory_s: f64,
    /// Block convention. Default [`ElgendiBlocks::Elgendi2013`].
    pub blocks: ElgendiBlocks,
}

impl Default for ElgendiConfig {
    /// Elgendi (2013) parameters (97 ms / 611 ms, β = 0.08, minimum block = QRS window) on the
    /// NeuroKit2 8–20 Hz causal band-pass, plus a 0.3 s refractory period.
    fn default() -> Self {
        Self {
            band_hz: (8.0, 20.0),
            filter_order: 2,
            filter: DetectorFilter::Causal,
            energy: ElgendiEnergy::Squared,
            qrs_window_s: 0.097,
            beat_window_s: 0.611,
            beta: 0.08,
            min_block_s: None,
            refractory_s: 0.3,
            blocks: ElgendiBlocks::Elgendi2013,
        }
    }
}

impl ElgendiConfig {
    /// NeuroKit2 0.3.0 `ecg_findpeaks(method="elgendi2013")` exactly (no refractory period).
    pub fn neurokit2013() -> Self {
        Self {
            refractory_s: 0.0,
            ..Self::default()
        }
    }

    /// NeuroKit2 `ecg_findpeaks(method="elgendi2010")` (= `"elgendi"`) exactly: `|x|`, windows
    /// `int(0.12·fs)` and `int(0.6·fs)`, no offset, block width `> int(0.08·fs)`, refractory
    /// `int(0.3·fs)`. Its detections are ~75–85 ms after the R peak.
    pub fn neurokit2010() -> Self {
        Self {
            energy: ElgendiEnergy::Abs,
            qrs_window_s: 0.12,
            beat_window_s: 0.6,
            beta: 0.0,
            min_block_s: Some(0.08),
            refractory_s: 0.3,
            blocks: ElgendiBlocks::NeuroKit2010,
            ..Self::default()
        }
    }

    /// Samples for a duration under this configuration's rounding convention.
    fn samples(&self, t: f64, fs: f64) -> usize {
        let v = t * fs;
        match self.blocks {
            ElgendiBlocks::Elgendi2013 => v.round_ties_even().max(0.0) as usize,
            ElgendiBlocks::NeuroKit2010 => v.trunc().max(0.0) as usize,
        }
    }

    fn validate(&self, fs: f64) -> Result<(), HrvError> {
        check_fs(fs)?;
        let pos = |v: f64| v.is_finite() && v > 0.0;
        let nonneg = |v: f64| v.is_finite() && v >= 0.0;
        if !pos(self.qrs_window_s) {
            return Err(HrvError::InvalidParameter("qrs_window_s"));
        }
        if !pos(self.beat_window_s) {
            return Err(HrvError::InvalidParameter("beat_window_s"));
        }
        if !self.beta.is_finite() {
            return Err(HrvError::InvalidParameter("beta"));
        }
        if !nonneg(self.refractory_s) {
            return Err(HrvError::InvalidParameter("refractory_s"));
        }
        if self.min_block_s.is_some_and(|m| !nonneg(m)) {
            return Err(HrvError::InvalidParameter("min_block_s"));
        }
        Ok(())
    }

    /// Band-pass sections of the detector path.
    pub fn design(&self, fs: f64) -> Result<Vec<SosSection>, HrvError> {
        butter_sos(
            self.filter_order,
            &[self.band_hz.0, self.band_hz.1],
            BandType::Bandpass,
            fs,
        )
    }
}

/// Detector-path band-pass of `signal` (default 8–20 Hz, causal, as NeuroKit2's
/// `ecg_clean(method="elgendi2010")`).
pub fn elgendi_filter(signal: &[f64], fs: f64, cfg: &ElgendiConfig) -> Result<Vec<f64>, HrvError> {
    check_finite(signal)?;
    let sos = cfg.design(fs)?;
    if signal.is_empty() {
        return Ok(Vec::new());
    }
    Ok(match cfg.filter {
        DetectorFilter::Causal => {
            let m = numpy_mean(signal);
            let zi: Vec<[f64; 2]> = sosfilt_zi(&sos)
                .into_iter()
                .map(|z| [z[0] * m, z[1] * m])
                .collect();
            sosfilt(&sos, signal, Some(&zi)).0
        }
        DetectorFilter::ZeroPhase => sosfiltfilt(&sos, signal),
    })
}

/// Index into `x` with SciPy `ndimage` "reflect" boundary (`d c b a | a b c d | d c b a`).
fn reflect(x: &[f64], k: isize) -> f64 {
    let n = x.len() as isize;
    let m = 2 * n;
    let k = k.rem_euclid(m);
    if k < n {
        x[k as usize]
    } else {
        x[(m - 1 - k) as usize]
    }
}

/// Trailing moving average of `w` samples, bit-identical to NeuroKit2's `_ecg_findpeaks_MWA`
/// (`scipy.ndimage.uniform_filter1d(x, w, origin=(w-1)//2)` with the first `w − 1` outputs
/// replaced by the running mean of the samples so far).
pub fn moving_average_causal(x: &[f64], w: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let w = w.max(1);
    let wf = w as f64;
    let mut out = vec![0.0; n];
    // SciPy keeps the running *sum* and divides each output by the size.
    let mut s = 0.0;
    for k in 0..w {
        s += reflect(x, k as isize - (w as isize - 1));
    }
    out[0] = s / wf;
    for i in 1..n {
        s += x[i] - reflect(x, i as isize - w as isize);
        out[i] = s / wf;
    }
    let head = (w - 1).min(n);
    let mut c = 0.0;
    for i in 0..head {
        c += x[i];
        out[i] = c / (i + 1) as f64;
    }
    out
}

/// First index of the maximum of `f(x[i])` over `lo..=hi` (`np.argmax` semantics).
fn argmax_by(x: &[f64], lo: usize, hi: usize, f: impl Fn(f64) -> f64) -> usize {
    let mut best = lo;
    let mut bv = f(x[lo]);
    for (j, &v) in x.iter().enumerate().take(hi + 1).skip(lo + 1) {
        let fv = f(v);
        if fv > bv {
            bv = fv;
            best = j;
        }
    }
    best
}

/// Elgendi detection on an already band-passed signal (the output of [`elgendi_filter`]).
/// Returns R-peak sample indices (strictly increasing).
pub fn elgendi_detect(
    filtered: &[f64],
    fs: f64,
    cfg: &ElgendiConfig,
) -> Result<Vec<usize>, HrvError> {
    cfg.validate(fs)?;
    check_finite(filtered)?;
    let n = filtered.len();
    if n < 2 {
        return Ok(Vec::new());
    }
    let energy: Vec<f64> = match cfg.energy {
        ElgendiEnergy::Squared => filtered.iter().map(|v| v * v).collect(),
        ElgendiEnergy::Abs => filtered.iter().map(|v| v.abs()).collect(),
    };
    let w_qrs = cfg.samples(cfg.qrs_window_s, fs).max(1);
    let w_beat = cfg.samples(cfg.beat_window_s, fs).max(1);
    let ma_qrs = moving_average_causal(&energy, w_qrs);
    let ma_beat = moving_average_causal(&energy, w_beat);
    let offset = if cfg.beta != 0.0 {
        cfg.beta * numpy_mean(&energy)
    } else {
        0.0
    };
    let above: Vec<bool> = (0..n).map(|i| ma_qrs[i] > ma_beat[i] + offset).collect();
    let refractory = cfg.samples(cfg.refractory_s, fs);
    let mut peaks: Vec<usize> = Vec::new();
    let accept = |d: usize, peaks: &mut Vec<usize>| match peaks.last() {
        Some(&last) if d <= last + refractory => {}
        _ => peaks.push(d),
    };

    match cfg.blocks {
        ElgendiBlocks::Elgendi2013 => {
            let min_block = match cfg.min_block_s {
                Some(m) => cfg.samples(m, fs),
                None => w_qrs,
            };
            let mut i = 0;
            while i < n {
                if !above[i] {
                    i += 1;
                    continue;
                }
                let start = i;
                while i < n && above[i] {
                    i += 1;
                }
                let end = i - 1;
                if end - start + 1 >= min_block {
                    let d = argmax_by(filtered, start, end, f64::abs);
                    accept(d, &mut peaks);
                }
            }
        }
        ElgendiBlocks::NeuroKit2010 => {
            let min_block = cfg.samples(cfg.min_block_s.unwrap_or(cfg.qrs_window_s), fs) as isize;
            let mut start: Option<usize> = None;
            for i in 0..n - 1 {
                let (prev, cur) = (above[i], above[i + 1]);
                if !prev && cur {
                    start = Some(i);
                } else if prev && !cur {
                    let Some(s) = start else { continue };
                    let end = i as isize - 1;
                    if end - s as isize > min_block {
                        let d = argmax_by(filtered, s, end as usize, |v| v);
                        accept(d, &mut peaks);
                    }
                }
            }
        }
    }
    Ok(peaks)
}

/// Elgendi R-peak detection: [`elgendi_filter`] then [`elgendi_detect`].
///
/// `signal` is a raw or cleaned ECG (any band containing 8–20 Hz). The returned indices are the
/// detector's own fiducials; with the default causal filter they lie ~35–40 ms after the R peak of
/// the clean signal. Use [`refine_peaks`] to move them onto the R maximum.
pub fn elgendi_peaks(signal: &[f64], fs: f64, cfg: &ElgendiConfig) -> Result<Vec<usize>, HrvError> {
    cfg.validate(fs)?;
    let filtered = elgendi_filter(signal, fs, cfg)?;
    elgendi_detect(&filtered, fs, cfg)
}

// =============================================================================
// R-peak refinement (local extremum + parabolic interpolation)
// =============================================================================

/// Which extremum [`refine_peaks`] looks for on the clean signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Polarity {
    /// Local maximum of `x` (upright R wave).
    Positive,
    /// Local minimum of `x` (inverted lead / electrode placement).
    Negative,
    /// Local maximum of `|x|`, decided per beat. Robust to a polarity change, but may jump to a
    /// deep S (or Q) wave when it is larger than R.
    Absolute,
    /// One sign for the whole call (i.e. per processed window): each candidate votes with the sign
    /// of the largest `|x|` in its search window; the majority wins (ties → positive). Then as
    /// [`Polarity::Positive`] or [`Polarity::Negative`].
    Auto,
}

/// Parameters of [`refine_peaks`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RefineConfig {
    /// Search half-window around each candidate (s): the extremum is searched in
    /// `[c − w, c + w]`. Default 0.05 s (±50 ms; literature recommendation, see `CHANGES-DETECT.md`).
    /// The default Elgendi detector is ~35–40 ms late, so ±50 ms covers it; NeuroKit2's
    /// `elgendi2010` preset is ~75–85 ms late and needs ≥ 0.09 s.
    pub half_window_s: f64,
    /// Default [`Polarity::Auto`].
    pub polarity: Polarity,
    /// 3-point parabolic interpolation around the extremum (sub-sample position). Default `true`.
    pub interpolate: bool,
}

impl Default for RefineConfig {
    fn default() -> Self {
        Self {
            half_window_s: 0.05,
            polarity: Polarity::Auto,
            interpolate: true,
        }
    }
}

/// Refined R peaks.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct RefinedPeaks {
    /// Sample index of the extremum of each peak (strictly increasing).
    pub indices: Vec<usize>,
    /// Sub-sample peak position in samples (`index + δ`, `|δ| ≤ 0.5`); equal to `indices` when
    /// interpolation is off. Same length as `indices`.
    pub positions: Vec<f64>,
    /// Sign used: `+1` (maximum of `x`), `−1` (minimum of `x`) or `0` ([`Polarity::Absolute`]).
    pub sign: f64,
}

impl RefinedPeaks {
    /// Peak times in seconds (`position / fs`).
    pub fn times_s(&self, fs: f64) -> Vec<f64> {
        self.positions.iter().map(|p| p / fs).collect()
    }
}

/// Vertex offset of the parabola through `(−1, y_m1)`, `(0, y0)`, `(1, y_p1)`:
/// `δ = 0.5 · (y_m1 − y_p1) / (y_m1 − 2·y0 + y_p1)`, clamped to `[−0.5, 0.5]`. Returns 0 if the three
/// points do not form a strict maximum (flat top or non-finite values).
pub fn parabolic_offset(y_m1: f64, y0: f64, y_p1: f64) -> f64 {
    let den = y_m1 - 2.0 * y0 + y_p1;
    if !den.is_finite() || den >= 0.0 {
        return 0.0;
    }
    let d = 0.5 * (y_m1 - y_p1) / den;
    if d.is_finite() {
        d.clamp(-0.5, 0.5)
    } else {
        0.0
    }
}

fn search_bounds(c: usize, w: usize, n: usize) -> (usize, usize) {
    (c.saturating_sub(w), (c + w).min(n - 1))
}

/// Majority sign of the largest `|x|` around each candidate (`+1` on ties or without candidates).
pub fn auto_polarity(signal: &[f64], fs: f64, candidates: &[usize], half_window_s: f64) -> f64 {
    let n = signal.len();
    if n == 0 || !(fs.is_finite() && fs > 0.0) {
        return 1.0;
    }
    let w = (half_window_s.max(0.0) * fs).round() as usize;
    let mut votes: i64 = 0;
    for &c in candidates.iter().filter(|&&c| c < n) {
        let (lo, hi) = search_bounds(c, w, n);
        let i = argmax_by(signal, lo, hi, f64::abs);
        if signal[i] > 0.0 {
            votes += 1;
        } else if signal[i] < 0.0 {
            votes -= 1;
        }
    }
    if votes >= 0 { 1.0 } else { -1.0 }
}

/// Move candidate R peaks onto the R extremum of the clean signal, with sub-sample precision.
///
/// For each candidate `c` the extremum (see [`Polarity`]) of `signal` is searched in
/// `[c − w, c + w]` (`w = round(half_window_s · fs)`, clipped to the signal), then a parabola is fitted
/// through the extremum and its two neighbours (`t = (n + δ) / fs`). `signal` should be the clean,
/// wide-band ECG (e.g. 0.5–40 Hz, [`crate::ecg_clean_zero_phase`]), not the detector band.
///
/// Candidates may come from any detector, unsorted; candidates outside the signal are ignored.
/// Candidates that land on the same sample are merged. Errors: invalid `fs`, negative/non-finite
/// window, non-finite signal.
pub fn refine_peaks(
    signal: &[f64],
    fs: f64,
    candidates: &[usize],
    cfg: &RefineConfig,
) -> Result<RefinedPeaks, HrvError> {
    check_fs(fs)?;
    if !(cfg.half_window_s.is_finite() && cfg.half_window_s >= 0.0) {
        return Err(HrvError::InvalidParameter("half_window_s"));
    }
    check_finite(signal)?;
    let n = signal.len();
    let mut cands: Vec<usize> = candidates.iter().copied().filter(|&c| c < n).collect();
    cands.sort_unstable();
    cands.dedup();
    let sign = match cfg.polarity {
        Polarity::Positive => 1.0,
        Polarity::Negative => -1.0,
        Polarity::Absolute => 0.0,
        Polarity::Auto => auto_polarity(signal, fs, &cands, cfg.half_window_s),
    };
    let value = |v: f64| if sign == 0.0 { v.abs() } else { sign * v };
    let w = (cfg.half_window_s * fs).round() as usize;
    let mut out = RefinedPeaks {
        indices: Vec::with_capacity(cands.len()),
        positions: Vec::with_capacity(cands.len()),
        sign,
    };
    let mut pairs: Vec<(usize, f64)> = Vec::with_capacity(cands.len());
    for &c in &cands {
        let (lo, hi) = search_bounds(c, w, n);
        let i = argmax_by(signal, lo, hi, value);
        let delta = if cfg.interpolate && i > 0 && i + 1 < n {
            parabolic_offset(value(signal[i - 1]), value(signal[i]), value(signal[i + 1]))
        } else {
            0.0
        };
        pairs.push((i, i as f64 + delta));
    }
    // Refined windows of increasing candidates can cross; keep the output sorted and unique.
    pairs.sort_by_key(|p| p.0);
    pairs.dedup_by_key(|p| p.0);
    for (i, p) in pairs {
        out.indices.push(i);
        out.positions.push(p);
    }
    Ok(out)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moving_average_matches_definition() {
        let x: Vec<f64> = (0..50).map(|i| ((i * 37) % 11) as f64 * 0.7).collect();
        for w in [1usize, 2, 5, 8, 49, 50, 60] {
            let m = moving_average_causal(&x, w);
            for i in 0..x.len() {
                let lo = (i + 1).saturating_sub(w);
                let want: f64 = x[lo..=i].iter().sum::<f64>() / (i + 1 - lo) as f64;
                assert!((m[i] - want).abs() < 1e-12, "w={w} i={i}");
            }
        }
        assert!(moving_average_causal(&[], 5).is_empty());
    }

    /// Bursts of a 15 Hz sine (already "band-passed"), `len_s` long, starting at `starts` (s).
    fn bursts(fs: f64, n: usize, starts: &[f64], len_s: f64) -> Vec<f64> {
        let mut x = vec![0.0; n];
        for &t0 in starts {
            let (a, b) = ((t0 * fs) as usize, ((t0 + len_s) * fs) as usize);
            for i in a..b.min(n) {
                let t = (i - a) as f64 / fs;
                x[i] = (2.0 * std::f64::consts::PI * 15.0 * t).sin()
                    * (std::f64::consts::PI * t / len_s).sin();
            }
        }
        x
    }

    #[test]
    fn refractory_keeps_the_earlier_detection() {
        let fs = 512.0;
        // beats every second, plus a spurious burst 0.2 s after the beat at 3 s
        let mut starts: Vec<f64> = (1..8).map(|k| k as f64).collect();
        starts.push(3.2);
        starts.sort_by(f64::total_cmp);
        let x = bursts(fs, 9 * 512, &starts, 0.08);
        let none = ElgendiConfig {
            refractory_s: 0.0,
            ..ElgendiConfig::default()
        };
        let all = elgendi_detect(&x, fs, &none).unwrap();
        assert_eq!(all.len(), 8, "{all:?}");
        let kept = elgendi_detect(&x, fs, &ElgendiConfig::default()).unwrap();
        assert_eq!(kept.len(), 7, "{kept:?}");
        // the beat at 3 s is kept, the burst at 3.2 s dropped
        assert!(kept.iter().any(|&p| (1536..1536 + 41).contains(&p)));
        assert!(!kept.iter().any(|&p| (1638..1638 + 41).contains(&p)));
    }

    #[test]
    fn parabola_vertex() {
        // y = -(t - 0.3)^2 sampled at -1, 0, 1
        let f = |t: f64| -(t - 0.3) * (t - 0.3);
        assert!((parabolic_offset(f(-1.0), f(0.0), f(1.0)) - 0.3).abs() < 1e-12);
        assert_eq!(parabolic_offset(1.0, 1.0, 1.0), 0.0);
        assert_eq!(parabolic_offset(0.0, f64::NAN, 0.0), 0.0);
        assert_eq!(parabolic_offset(2.0, 1.0, 0.0), 0.0); // not a maximum
    }

    #[test]
    fn refine_edges_and_duplicates() {
        let mut x = vec![0.0; 200];
        x[50] = 1.0;
        x[49] = 0.5;
        x[51] = 0.5;
        x[0] = -3.0;
        let cfg = RefineConfig {
            polarity: Polarity::Positive,
            ..RefineConfig::default()
        };
        // two candidates landing on the same peak, one out of range, unsorted
        let r = refine_peaks(&x, 512.0, &[60, 45, 1000], &cfg).unwrap();
        assert_eq!(r.indices, vec![50]);
        assert!((r.positions[0] - 50.0).abs() < 1e-12);
        // absolute picks the larger |x| at the signal edge, no interpolation there
        let abs = RefineConfig {
            polarity: Polarity::Absolute,
            ..cfg
        };
        let r = refine_peaks(&x, 512.0, &[5], &abs).unwrap();
        assert_eq!((r.indices[0], r.positions[0]), (0, 0.0));
        assert_eq!(auto_polarity(&x, 512.0, &[5], 0.05), -1.0);
        assert!(refine_peaks(&x, 0.0, &[5], &cfg).is_err());
        assert!(
            refine_peaks(&[], 512.0, &[5], &cfg)
                .unwrap()
                .indices
                .is_empty()
        );
    }

    #[test]
    fn invalid_inputs_do_not_panic() {
        let cfg = ElgendiConfig::default();
        assert!(elgendi_peaks(&[1.0, 2.0], 0.0, &cfg).is_err());
        assert!(elgendi_peaks(&[1.0, f64::NAN], 512.0, &cfg).is_err());
        assert_eq!(
            elgendi_peaks(&[], 512.0, &cfg).unwrap(),
            Vec::<usize>::new()
        );
        assert_eq!(
            elgendi_peaks(&[1.0], 512.0, &cfg).unwrap(),
            Vec::<usize>::new()
        );
        assert_eq!(
            elgendi_peaks(&[0.0; 1000], 512.0, &cfg).unwrap(),
            Vec::<usize>::new()
        );
        let bad = ElgendiConfig {
            qrs_window_s: 0.0,
            ..cfg
        };
        assert!(elgendi_peaks(&[0.0; 100], 512.0, &bad).is_err());
        let band = ElgendiConfig {
            band_hz: (8.0, 300.0),
            ..cfg
        };
        assert!(elgendi_peaks(&[0.0; 100], 512.0, &band).is_err());
    }
}
