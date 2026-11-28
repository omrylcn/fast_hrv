//! Breathing rate from a single-lead ECG (ECG-derived respiration, EDR).
//!
//! Breathing modulates the ECG in two ways that a chest or wearable lead picks up:
//! * **amplitude modulation (AM):** the R-wave amplitude changes with the chest volume and the
//!   electrical axis;
//! * **frequency modulation (FM):** respiratory sinus arrhythmia, the RR intervals shorten on
//!   inspiration.
//!
//! Each modulation gives one value per beat (unevenly spaced). The rate is the spectral peak of that
//! series in the breathing band (default 0.1–0.5 Hz = 6–30 breaths/min), found with the Lomb–Scargle
//! periodogram ([`crate::frequency::lomb_scargle`]), so missing or excluded beats need no interpolation.
//! The two estimates are fused as in Karlen et al. 2013 ("smart fusion", IEEE TBME 60:1946): when
//! they agree within `max_disagreement_bpm` the mean is returned, otherwise the window has no valid
//! rate. Charlton et al. 2016 (Physiol Meas 37:610) found AM/FM fusion among the most accurate ECG
//! methods; both need clean, ectopy-free beats.
//!
//! Use: windows of 32–64 s (≥ 4 breaths at the lowest rate). A breathing rate below 9/min (0.15 Hz)
//! moves the respiratory component into the LF band: HF then underestimates vagal activity; use
//! [`hf_band_for`] to centre the HF band on the breathing rate, or flag the window.

use crate::frequency::lomb_scargle;

/// R-wave amplitude per beat: value of the (band-passed) ECG at the R peak minus the mean of the
/// isoelectric segment 100–60 ms before it (PR segment). Peaks too close to the start are `NaN`.
pub fn r_amplitudes(ecg: &[f64], peaks: &[usize], fs: f64) -> Vec<f64> {
    let a = (0.100 * fs).round() as usize;
    let b = (0.060 * fs).round() as usize;
    peaks
        .iter()
        .map(|&p| {
            if p < a || p >= ecg.len() || a <= b {
                return f64::NAN;
            }
            let base = &ecg[p - a..p - b];
            let m = base.iter().sum::<f64>() / base.len() as f64;
            ecg[p] - m
        })
        .collect()
}

/// Settings of [`respiratory_rate`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RespirationConfig {
    /// Breathing band searched (Hz). Default 0.1–0.5 (6–30 breaths/min).
    pub band_hz: (f64, f64),
    /// Smallest window (s) accepted. Default 30.
    pub min_window_s: f64,
    /// A modulation is usable when its spectral peak (±1/T) holds at least this share of the band
    /// power. Default 0.25 (≈ 3× the white-noise share in a 60 s window).
    pub min_peak_share: f64,
    /// AM and FM estimates must agree within this (breaths/min). Default 4 (Karlen 2013).
    pub max_disagreement_bpm: f64,
}

impl Default for RespirationConfig {
    fn default() -> Self {
        Self {
            band_hz: (0.1, 0.5),
            min_window_s: 30.0,
            min_peak_share: 0.25,
            max_disagreement_bpm: 4.0,
        }
    }
}

/// One modulation's estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ModulationEstimate {
    /// Breaths per minute at the spectral peak (`NaN` if the series is unusable).
    pub rate_bpm: f64,
    /// Share of the band power within ±1/T of the peak (the main lobe of a sinusoid), 0–1. White
    /// noise gives about (2/T) / band width (≈ 0.08 for 60 s and 0.1–0.5 Hz).
    pub peak_share: f64,
}

/// Breathing rate of one window.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RespirationEstimate {
    /// Fused rate (breaths/min); `NaN` when not valid.
    pub rate_bpm: f64,
    pub valid: bool,
    /// From the R-amplitude series.
    pub am: ModulationEstimate,
    /// From the RR series (respiratory sinus arrhythmia).
    pub fm: ModulationEstimate,
}

/// Peak of the Lomb–Scargle spectrum of `(t, y)` in `band`; grid step 1/(8·T), clipped to the band.
pub fn modulation_rate(t: &[f64], y: &[f64], band: (f64, f64)) -> ModulationEstimate {
    let bad = ModulationEstimate {
        rate_bpm: f64::NAN,
        peak_share: 0.0,
    };
    let (tt, yy): (Vec<f64>, Vec<f64>) = t
        .iter()
        .zip(y)
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(a, b)| (*a, *b))
        .unzip();
    if tt.len() < 8 {
        return bad;
    }
    let span = tt[tt.len() - 1] - tt[0];
    if span <= 0.0 {
        return bad;
    }
    let df = 1.0 / (8.0 * span);
    let k0 = (band.0 / df).ceil().max(1.0) as usize;
    let k1 = (band.1 / df).floor() as usize;
    if k1 <= k0 + 2 {
        return bad;
    }
    let freqs: Vec<f64> = (k0..=k1).map(|k| k as f64 * df).collect();
    let p = lomb_scargle(&tt, &yy, &freqs);
    if p.iter().any(|v| !v.is_finite()) {
        return bad;
    }
    let total: f64 = p.iter().sum();
    let (imax, _) = p
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap();
    if total <= 0.0 {
        return bad;
    }
    // parabolic refinement of the peak frequency
    let f = if imax > 0 && imax + 1 < p.len() {
        let (a, b, c) = (p[imax - 1], p[imax], p[imax + 1]);
        let den = a - 2.0 * b + c;
        let d = if den != 0.0 { 0.5 * (a - c) / den } else { 0.0 };
        freqs[imax] + d.clamp(-0.5, 0.5) * df
    } else {
        freqs[imax]
    };
    // main lobe of a sinusoid in a window of length T: ±1/T = ±8 grid steps of 1/(8T)
    let lo = imax.saturating_sub(8);
    let hi = (imax + 8).min(p.len() - 1);
    let share = p[lo..=hi].iter().sum::<f64>() / total;
    ModulationEstimate {
        rate_bpm: f * 60.0,
        peak_share: share,
    }
}

/// Breathing rate of one window from the beat times (s), the RR intervals ending at those beats (ms)
/// and the R amplitudes ([`r_amplitudes`]). Pass only clean, ectopy-free beats (`NaN` entries are
/// dropped). Smart fusion of the AM and FM estimates (see the module docs).
pub fn respiratory_rate(
    beat_times_s: &[f64],
    rr_ms: &[f64],
    r_amp: &[f64],
    cfg: &RespirationConfig,
) -> RespirationEstimate {
    let fm = modulation_rate(beat_times_s, rr_ms, cfg.band_hz);
    let am = modulation_rate(beat_times_s, r_amp, cfg.band_hz);
    let span = {
        let f: Vec<f64> = beat_times_s
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .collect();
        if f.len() > 1 {
            f[f.len() - 1] - f[0]
        } else {
            0.0
        }
    };
    let ok = |m: &ModulationEstimate| m.rate_bpm.is_finite() && m.peak_share >= cfg.min_peak_share;
    let (rate, valid) = if span < cfg.min_window_s {
        (f64::NAN, false)
    } else if ok(&am) && ok(&fm) && (am.rate_bpm - fm.rate_bpm).abs() <= cfg.max_disagreement_bpm {
        (0.5 * (am.rate_bpm + fm.rate_bpm), true)
    } else {
        (f64::NAN, false)
    };
    RespirationEstimate {
        rate_bpm: rate,
        valid,
        am,
        fm,
    }
}

/// HF band centred on the breathing rate (Hz): `rate ± half_width`, lower edge at least `floor_hz`.
/// E.g. `hf_band_for(12.0, 0.1, 0.04)` = (0.10, 0.30). Use it instead of the fixed 0.15–0.4 Hz band
/// when the breathing rate is known; below 9 breaths/min the fixed band misses the vagal component.
pub fn hf_band_for(rate_bpm: f64, half_width_hz: f64, floor_hz: f64) -> (f64, f64) {
    let f = rate_bpm / 60.0;
    ((f - half_width_hz).max(floor_hz), f + half_width_hz)
}

/// HF power measured around the breathing rate, and where the breathing rate lies relative to the
/// fixed bands.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct RespiratoryHf {
    pub rate_bpm: f64,
    /// Band used (Hz): breathing frequency ± 0.1 Hz, lower edge at least 0.04 Hz.
    pub band: (f64, f64),
    /// Power in that band (ms²) and its natural log.
    pub power: f64,
    pub ln_power: f64,
    /// Breathing below 0.15 Hz (9 /min): the respiratory peak is in the LF band, so fixed-band HF
    /// misses vagal power and LF and LF/HF are inflated.
    pub in_lf_band: bool,
    /// Breathing above 0.4 Hz (24 /min): part of the respiratory power is above the HF band.
    pub above_hf_band: bool,
}

/// [`RespiratoryHf`] for an RR series (ms) and a breathing rate (breaths/min, e.g. from
/// [`respiratory_rate`]). The ±0.1 Hz width is a convention (no validated width was found); the
/// result is meant to be read next to the fixed-band HF, not instead of it.
pub fn respiratory_hf(
    rri_ms: &[f64],
    rate_bpm: f64,
    opts: &crate::frequency::FrequencyOptions,
) -> RespiratoryHf {
    let band = hf_band_for(rate_bpm, 0.1, 0.04);
    let power = if rate_bpm.is_finite() && rate_bpm > 0.0 {
        crate::frequency::band_power_rri(rri_ms, band, opts)
    } else {
        f64::NAN
    };
    let f = rate_bpm / 60.0;
    RespiratoryHf {
        rate_bpm,
        band,
        power,
        ln_power: if power > 0.0 { power.ln() } else { f64::NAN },
        in_lf_band: f < 0.15,
        above_hf_band: f > 0.4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Beats with RSA (FM) and amplitude modulation (AM) at `resp_hz`.
    fn beats(
        resp_hz: f64,
        seconds: f64,
        am_depth: f64,
        fm_depth_s: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (mut t, mut rr, mut amp) = (Vec::new(), Vec::new(), Vec::new());
        let mut s = 0.0;
        let mut k = 7u64;
        while s < seconds {
            k = k
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise = (k >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
            let ph = 2.0 * std::f64::consts::PI * resp_hz * s;
            let r = 0.85 + fm_depth_s * ph.sin() + 0.004 * noise;
            s += r;
            t.push(s);
            rr.push(r * 1000.0);
            amp.push(1000.0 * (1.0 + am_depth * (ph + 0.5).cos()) + 5.0 * noise);
        }
        (t, rr, amp)
    }

    #[test]
    fn finds_the_breathing_rate() {
        for rate in [8.0, 12.0, 15.0, 20.0, 26.0] {
            let (t, rr, amp) = beats(rate / 60.0, 60.0, 0.08, 0.03);
            let e = respiratory_rate(&t, &rr, &amp, &RespirationConfig::default());
            assert!(e.valid, "{rate}: {e:?}");
            assert!((e.rate_bpm - rate).abs() < 1.0, "{rate}: {e:?}");
        }
    }

    #[test]
    fn disagreement_or_short_window_is_invalid() {
        let (t, rr, _) = beats(12.0 / 60.0, 60.0, 0.0, 0.03);
        let (_, _, amp) = beats(22.0 / 60.0, 60.0, 0.08, 0.0); // AM at another rate (same beat count not needed)
        let n = t.len().min(amp.len());
        let e = respiratory_rate(&t[..n], &rr[..n], &amp[..n], &RespirationConfig::default());
        assert!(!e.valid, "{e:?}");
        let (t, rr, amp) = beats(12.0 / 60.0, 20.0, 0.08, 0.03);
        assert!(!respiratory_rate(&t, &rr, &amp, &RespirationConfig::default()).valid);
    }

    #[test]
    fn hf_band_follows_breathing() {
        let (lo, hi) = hf_band_for(12.0, 0.1, 0.04);
        assert!((lo - 0.1).abs() < 1e-12 && (hi - 0.3).abs() < 1e-12);
        assert_eq!(hf_band_for(6.0, 0.1, 0.04).0, 0.04);
    }

    #[test]
    fn r_amplitude_uses_the_pr_baseline() {
        let fs = 512.0;
        let mut x = vec![10.0; 2048];
        x[1000] = 510.0;
        let a = r_amplitudes(&x, &[1000, 10], fs);
        assert_eq!(a[0], 500.0);
        assert!(a[1].is_nan());
    }
}
