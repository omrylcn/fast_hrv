//! Tek çağrıda bütün HRV metrikleri: zaman alanı, frekans alanı, doğrusal olmayan, asimetri.
//!
//! ```
//! let rri = vec![812.0, 790.5, 845.0, 801.2, 830.0, 815.5, 798.0, 822.3, 808.8, 840.1, 795.0, 818.0];
//! let r = fast_hrv::hrv(&rri);
//! assert!(r.time.rmssd > 0.0);
//! ```

use crate::{
    frequency::{frequency_domain, FrequencyDomain},
    hrv::{compute_apen, compute_asymmetry, compute_dfa, compute_poincare, compute_sampen},
    time_domain::{time_domain, TimeDomain},
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
}

/// NN aralıklarından (ms) bütün metrikler. Kısa kayıtlarda hesaplanamayanlar NaN döner:
/// zaman alanı ≥ 3 aralık, frekans alanı ≥ 30 sn, entropi/DFA ≥ 16 aralık.
pub fn hrv(rri: &[f64]) -> HrvReport {
    let nan = f64::NAN;
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
    }
}

/// R tepe örnek indekslerinden (ve örnekleme frekansından) NN aralıkları (ms).
pub fn rri_from_peaks(peaks: &[usize], sampling_rate: u32) -> Vec<f64> {
    peaks.windows(2).map(|w| (w[1] - w[0]) as f64 * 1000.0 / sampling_rate as f64).collect()
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
            ("sdnn", r.time.sdnn), ("rmssd", r.time.rmssd), ("hti", r.time.hti), ("lf", r.frequency.lf),
            ("hf", r.frequency.hf), ("lf_hf", r.frequency.lf_hf), ("sd1", r.nonlinear.sd1),
            ("sampen", r.nonlinear.sampen), ("dfa_alpha1", r.nonlinear.dfa_alpha1), ("gi", r.asymmetry.gi),
        ] {
            assert!(v.is_finite(), "{name} = {v}");
        }
        assert!(r.frequency.hf > r.frequency.lf, "solunum genliği daha büyük");
    }

    #[test]
    fn rri_from_peak_indices() {
        assert_eq!(rri_from_peaks(&[0, 512, 1024, 1536], 512), vec![1000.0, 1000.0, 1000.0]);
    }
}

