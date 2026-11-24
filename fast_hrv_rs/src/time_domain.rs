//! Zaman alanı HRV metrikleri (Task Force 1996 tanımları; NeuroKit2 adlarıyla uyumlu).
//!
//! Girdi: NN (RR) aralıkları, milisaniye. Hepsi tek geçişte ve kopyasız (sıralama için tek kopya).

use crate::hrv::{compute_pnn20, compute_pnn50, compute_rmssd, compute_sdsd, mean, std_ddof};

/// Bütün zaman alanı metrikleri.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TimeDomain {
    /// Ortalama NN (ms)
    pub mean_nn: f64,
    /// NN standart sapması (ms)
    pub sdnn: f64,
    /// Ardışık farkların karekök ortalaması (ms)
    pub rmssd: f64,
    /// Ardışık farkların standart sapması (ms)
    pub sdsd: f64,
    /// |fark| > 50 ms olan ardışık çiftlerin yüzdesi
    pub pnn50: f64,
    /// |fark| > 20 ms olan ardışık çiftlerin yüzdesi
    pub pnn20: f64,
    /// Medyan NN (ms)
    pub median_nn: f64,
    /// Ölçeklenmiş medyan mutlak sapma (ms; 1.4826 × MAD, normal dağılımda SDNN'e denk)
    pub mad_nn: f64,
    /// Çeyrekler arası açıklık (ms)
    pub iqr_nn: f64,
    /// En küçük ve en büyük NN (ms)
    pub min_nn: f64,
    pub max_nn: f64,
    /// Değişim katsayıları: SDNN / MeanNN, RMSSD / MeanNN
    pub cvnn: f64,
    pub cvsd: f64,
    /// Ortalama ve standart sapma kalp hızı (atım/dk)
    pub mean_hr: f64,
    pub sd_hr: f64,
    /// HRV üçgen indeksi (toplam NN / histogram tepesi, kutu genişliği 1/128 s)
    pub hti: f64,
    /// Üçgen enterpolasyonla NN histogram taban genişliği (ms)
    pub tinn: f64,
}

/// Lineer enterpolasyonlu yüzdelik (NumPy `percentile` varsayılanı ile aynı). `sorted` artan sıralı olmalı.
pub fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let pos = (p / 100.0) * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

const HIST_BIN_MS: f64 = 1000.0 / 128.0; // 7.8125 ms (Task Force standardı)

fn histogram(rri: &[f64], min: f64, max: f64) -> Vec<usize> {
    let n_bins = (((max - min) / HIST_BIN_MS).floor() as usize) + 1;
    let mut h = vec![0usize; n_bins.max(1)];
    for &x in rri {
        let i = (((x - min) / HIST_BIN_MS).floor() as usize).min(h.len() - 1);
        h[i] += 1;
    }
    h
}

/// TINN: histogram tepesinin iki yanına üçgen oturtup en küçük karesel hatayı veren taban (M − N).
fn tinn(hist: &[usize], min: f64) -> f64 {
    if hist.len() < 3 {
        return f64::NAN;
    }
    let (peak, &peak_h) = hist.iter().enumerate().max_by_key(|&(_, &v)| v).unwrap();
    let peak_h = peak_h as f64;
    let mut best = (f64::INFINITY, 0usize, 0usize);
    for n in 0..peak {
        for m in (peak + 1)..hist.len() {
            let mut err = 0.0;
            for (i, &v) in hist.iter().enumerate() {
                let q = if i <= n || i >= m {
                    0.0
                } else if i <= peak {
                    peak_h * (i - n) as f64 / (peak - n) as f64
                } else {
                    peak_h * (m - i) as f64 / (m - peak) as f64
                };
                err += (v as f64 - q).powi(2);
            }
            if err < best.0 {
                best = (err, n, m);
            }
        }
    }
    let _ = min;
    (best.2 - best.1) as f64 * HIST_BIN_MS
}

/// Bütün zaman alanı metriklerini hesaplar. En az 3 aralık gerekir; azsa alanlar NaN.
pub fn time_domain(rri: &[f64]) -> TimeDomain {
    let nan = f64::NAN;
    if rri.len() < 3 {
        return TimeDomain {
            mean_nn: nan,
            sdnn: nan,
            rmssd: nan,
            sdsd: nan,
            pnn50: nan,
            pnn20: nan,
            median_nn: nan,
            mad_nn: nan,
            iqr_nn: nan,
            min_nn: nan,
            max_nn: nan,
            cvnn: nan,
            cvsd: nan,
            mean_hr: nan,
            sd_hr: nan,
            hti: nan,
            tinn: nan,
        };
    }
    let mut sorted = rri.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_nn = mean(rri);
    let sdnn = std_ddof(rri, 1);
    let rmssd = compute_rmssd(rri);
    let median_nn = percentile_sorted(&sorted, 50.0);
    let mut abs_dev: Vec<f64> = rri.iter().map(|x| (x - median_nn).abs()).collect();
    abs_dev.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let hr: Vec<f64> = rri.iter().map(|x| 60_000.0 / x).collect();
    let (min_nn, max_nn) = (sorted[0], sorted[sorted.len() - 1]);
    let hist = histogram(rri, min_nn, max_nn);
    let peak = *hist.iter().max().unwrap() as f64;

    TimeDomain {
        mean_nn,
        sdnn,
        rmssd,
        sdsd: compute_sdsd(rri),
        pnn50: compute_pnn50(rri),
        pnn20: compute_pnn20(rri),
        median_nn,
        mad_nn: 1.4826 * percentile_sorted(&abs_dev, 50.0),
        iqr_nn: percentile_sorted(&sorted, 75.0) - percentile_sorted(&sorted, 25.0),
        min_nn,
        max_nn,
        cvnn: sdnn / mean_nn,
        cvsd: rmssd / mean_nn,
        mean_hr: mean(&hr),
        sd_hr: std_ddof(&hr, 1),
        hti: rri.len() as f64 / peak,
        tinn: tinn(&hist, min_nn),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const RRI: [f64; 10] = [
        800.0, 850.0, 790.0, 820.0, 880.0, 750.0, 810.0, 860.0, 800.0, 830.0,
    ];

    #[test]
    fn basic_statistics() {
        let t = time_domain(&RRI);
        assert_relative_eq!(t.mean_nn, 819.0, epsilon = 1e-10);
        assert_relative_eq!(t.median_nn, 815.0, epsilon = 1e-10);
        assert_relative_eq!(t.min_nn, 750.0);
        assert_relative_eq!(t.max_nn, 880.0);
        // NumPy: np.percentile(RRI, 75) - np.percentile(RRI, 25) = 845 - 800
        assert_relative_eq!(t.iqr_nn, 45.0, epsilon = 1e-10);
        assert_relative_eq!(t.cvnn, t.sdnn / t.mean_nn, epsilon = 1e-12);
        assert_relative_eq!(t.cvsd, t.rmssd / t.mean_nn, epsilon = 1e-12);
        // MAD: |x - 815| = [65,25,15,15,5,5,15,35,45,65] → medyan 20 → ×1.4826
        assert_relative_eq!(t.mad_nn, 1.4826 * 20.0, epsilon = 1e-10);
    }

    #[test]
    fn heart_rate_from_rri() {
        let t = time_domain(&[1000.0, 1000.0, 1000.0, 500.0]);
        assert_relative_eq!(t.mean_hr, (60.0 * 3.0 + 120.0) / 4.0, epsilon = 1e-10);
    }

    #[test]
    fn percentile_like_numpy() {
        let s = [1.0, 2.0, 3.0, 4.0];
        assert_relative_eq!(percentile_sorted(&s, 25.0), 1.75);
        assert_relative_eq!(percentile_sorted(&s, 50.0), 2.5);
    }

    #[test]
    fn too_short_is_nan() {
        assert!(time_domain(&[800.0, 810.0]).sdnn.is_nan());
    }

    #[test]
    fn triangular_index_and_tinn_positive() {
        let t = time_domain(&RRI);
        assert!(t.hti > 0.0 && t.hti <= RRI.len() as f64);
        assert!(t.tinn >= 0.0);
    }
}
