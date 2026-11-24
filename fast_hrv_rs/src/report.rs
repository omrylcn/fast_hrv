//! Tek çağrıda bütün HRV metrikleri: zaman alanı, frekans alanı, doğrusal olmayan, asimetri.
//!
//! ```
//! let rri = vec![812.0, 790.5, 845.0, 801.2, 830.0, 815.5, 798.0, 822.3, 808.8, 840.1, 795.0, 818.0];
//! let r = fast_hrv::hrv(&rri);
//! assert!(r.time.rmssd > 0.0);
//! ```

use crate::{
    error::HrvError,
    frequency::{FrequencyDomain, frequency_domain},
    hrv::{compute_apen, compute_asymmetry, compute_dfa, compute_poincare, compute_sampen},
    time_domain::{TimeDomain, time_domain},
};

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NonLinear {
    /// Poincaré: kısa ve uzun dönem değişkenlik (ms), oranları
    pub sd1: f64,
    pub sd2: f64,
    pub sd1_sd2: f64,
    /// Kardiyak sempatik ve vagal indeksler
    pub csi: f64,
    pub cvi: f64,
    /// Örnek ve yaklaşık entropi (m = 2, r = 0.2 × SDNN)
    pub sampen: f64,
    pub apen: f64,
    /// DFA kısa (4–16 atım) ve uzun (16–64 atım) ölçek eğimleri
    pub dfa_alpha1: f64,
    pub dfa_alpha2: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Asymmetry {
    /// Guzik ve Porta indeksleri (%)
    pub gi: f64,
    pub pi: f64,
    /// Yavaşlama / hızlanma katkıları
    pub sd1d: f64,
    pub sd1a: f64,
    pub c1d: f64,
    pub c1a: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HrvReport {
    /// Kullanılan aralık sayısı
    pub n: usize,
    pub time: TimeDomain,
    pub frequency: FrequencyDomain,
    pub nonlinear: NonLinear,
    pub asymmetry: Asymmetry,
    /// Heart rate fragmentation (Costa 2017; NeuroKit2 definitions).
    pub fragmentation: crate::fragmentation::Fragmentation,
}

/// NN aralıklarından (ms) bütün metrikler. Kısa kayıtlarda hesaplanamayanlar NaN döner:
/// zaman alanı ≥ 3 aralık, frekans alanı bant başına HF ≥ 60 sn, LF ≥ 120 sn, VLF ≥ 300 sn (RR toplamı), entropi/DFA ≥ 16 aralık.
///
/// Girdide NaN/sonsuz bir aralık varsa bütün metrikler NaN döner (panik yok).
pub fn hrv(rri: &[f64]) -> HrvReport {
    let nan = f64::NAN;
    if rri.iter().any(|x| !x.is_finite()) {
        return HrvReport::nan(rri.len());
    }
    let (poincare, asym) = if rri.len() >= 3 {
        (Some(compute_poincare(rri)), Some(compute_asymmetry(rri)))
    } else {
        (None, None)
    };
    let (sampen, apen, dfa) = if rri.len() >= 16 {
        (
            compute_sampen(rri, 2, None),
            compute_apen(rri, 2, None),
            Some(compute_dfa(rri, (4, 16), (16, Some(64)))),
        )
    } else {
        (nan, nan, None)
    };
    HrvReport {
        n: rri.len(),
        time: time_domain(rri),
        frequency: frequency_domain(rri),
        nonlinear: NonLinear {
            sd1: poincare.as_ref().map_or(nan, |p| p.sd1),
            sd2: poincare.as_ref().map_or(nan, |p| p.sd2),
            sd1_sd2: poincare.as_ref().map_or(nan, |p| p.sd1 / p.sd2),
            csi: poincare.as_ref().map_or(nan, |p| p.csi),
            cvi: poincare.as_ref().map_or(nan, |p| p.cvi),
            sampen,
            apen,
            dfa_alpha1: dfa.as_ref().map_or(nan, |d| d.alpha1),
            dfa_alpha2: dfa.as_ref().map_or(nan, |d| d.alpha2),
        },
        asymmetry: Asymmetry {
            gi: asym.as_ref().map_or(nan, |a| a.gi),
            pi: asym.as_ref().map_or(nan, |a| a.pi),
            sd1d: asym.as_ref().map_or(nan, |a| a.sd1d),
            sd1a: asym.as_ref().map_or(nan, |a| a.sd1a),
            c1d: asym.as_ref().map_or(nan, |a| a.c1d),
            c1a: asym.as_ref().map_or(nan, |a| a.c1a),
        },
        fragmentation: crate::fragmentation::fragmentation(rri),
    }
}

/// R tepe örnek indekslerinden (ve örnekleme frekansından) NN aralıkları (ms).
///
/// Panik yok: sıralı olmayan ya da yinelenen bir tepe çifti (aralık ≤ 0) veya `sampling_rate == 0`
/// o aralık için NaN üretir. Doğrulama ve hata için [`try_rri_from_peaks`]; boşluklu kayıtlar için
/// [`crate::gaps::rri_with_gaps`].
pub fn rri_from_peaks(peaks: &[usize], sampling_rate: u32) -> Vec<f64> {
    peaks
        .windows(2)
        .map(|w| {
            if w[1] > w[0] && sampling_rate > 0 {
                (w[1] - w[0]) as f64 * 1000.0 / sampling_rate as f64
            } else {
                f64::NAN
            }
        })
        .collect()
}

/// [`rri_from_peaks`] with validation: peaks must be strictly increasing and `sampling_rate > 0`.
pub fn try_rri_from_peaks(peaks: &[usize], sampling_rate: u32) -> Result<Vec<f64>, HrvError> {
    if sampling_rate == 0 {
        return Err(HrvError::InvalidSamplingRate(0.0));
    }
    if let Some(i) = peaks.windows(2).position(|w| w[1] <= w[0]) {
        return Err(HrvError::UnsortedPeaks { index: i + 1 });
    }
    Ok(rri_from_peaks(peaks, sampling_rate))
}

/// NN intervals (ms) from sub-sample R-peak positions (in samples, e.g.
/// [`crate::detect::RefinedPeaks::positions`]) and the sampling rate.
///
/// Same conventions as [`rri_from_peaks`]: a non-increasing pair, a non-finite position or an invalid
/// `sampling_rate` gives `NaN` for that interval (no panic).
pub fn rri_from_positions(positions: &[f64], sampling_rate: f64) -> Vec<f64> {
    let ok_fs = sampling_rate.is_finite() && sampling_rate > 0.0;
    positions
        .windows(2)
        .map(|w| {
            if ok_fs && w[0].is_finite() && w[1].is_finite() && w[1] > w[0] {
                (w[1] - w[0]) * 1000.0 / sampling_rate
            } else {
                f64::NAN
            }
        })
        .collect()
}

/// NN intervals (ms) from R-peak times in seconds (e.g. [`crate::detect::RefinedPeaks::times_s`]).
/// A non-increasing or non-finite pair gives `NaN` for that interval.
pub fn rri_from_times(times_s: &[f64]) -> Vec<f64> {
    rri_from_positions(times_s, 1.0)
}

/// [`rri_from_times`] with validation: times must be finite and strictly increasing.
pub fn try_rri_from_times(times_s: &[f64]) -> Result<Vec<f64>, HrvError> {
    if times_s.iter().any(|t| !t.is_finite()) {
        return Err(HrvError::NonFinite);
    }
    if let Some(i) = times_s.windows(2).position(|w| w[1] <= w[0]) {
        return Err(HrvError::UnsortedPeaks { index: i + 1 });
    }
    Ok(rri_from_times(times_s))
}

impl HrvReport {
    /// Report with every metric NaN.
    pub fn nan(n: usize) -> Self {
        let nan = f64::NAN;
        let mut r = hrv(&[]);
        r.n = n;
        r.nonlinear = NonLinear {
            sd1: nan,
            sd2: nan,
            sd1_sd2: nan,
            csi: nan,
            cvi: nan,
            sampen: nan,
            apen: nan,
            dfa_alpha1: nan,
            dfa_alpha2: nan,
        };
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_counts_and_short_records() {
        let r = hrv(&[800.0, 810.0]);
        assert_eq!(r.n, 2);
        assert!(r.time.sdnn.is_nan());
        assert!(r.frequency.lf.is_nan());
        assert!(r.nonlinear.sampen.is_nan());
    }

    #[test]
    fn five_minute_record_fills_everything() {
        // 5 dk, solunum (0.25 Hz) + baroreseptör (0.1 Hz) modülasyonu
        let mut rri = Vec::new();
        let mut t = 0.0;
        while t < 300.0 {
            let w = 2.0 * std::f64::consts::PI;
            rri.push(800.0 + 25.0 * (w * 0.25 * t).sin() + 15.0 * (w * 0.1 * t).sin());
            t += rri.last().unwrap() / 1000.0;
        }
        let r = hrv(&rri);
        for (name, v) in [
            ("sdnn", r.time.sdnn),
            ("rmssd", r.time.rmssd),
            ("hti", r.time.hti),
            ("lf", r.frequency.lf),
            ("hf", r.frequency.hf),
            ("lf_hf", r.frequency.lf_hf),
            ("sd1", r.nonlinear.sd1),
            ("sampen", r.nonlinear.sampen),
            ("dfa_alpha1", r.nonlinear.dfa_alpha1),
            ("gi", r.asymmetry.gi),
        ] {
            assert!(v.is_finite(), "{name} = {v}");
        }
        assert!(
            r.frequency.hf > r.frequency.lf,
            "solunum genliği daha büyük"
        );
    }

    #[test]
    fn invalid_inputs_do_not_panic() {
        let r = rri_from_peaks(&[100, 50, 50, 600], 512);
        assert!(r[0].is_nan() && r[1].is_nan() && r[2].is_finite());
        assert!(try_rri_from_peaks(&[100, 50], 512).is_err());
        assert!(rri_from_peaks(&[0, 512], 0)[0].is_nan());
        let mut v: Vec<f64> = (0..400).map(|i| 800.0 + (i as f64).sin() * 30.0).collect();
        v[5] = f64::NAN;
        let rep = hrv(&v);
        assert!(
            rep.time.rmssd.is_nan() && rep.frequency.lf.is_nan() && rep.nonlinear.sampen.is_nan()
        );
    }

    #[test]
    fn rri_from_peak_indices() {
        assert_eq!(
            rri_from_peaks(&[0, 512, 1024, 1536], 512),
            vec![1000.0, 1000.0, 1000.0]
        );
    }
}
