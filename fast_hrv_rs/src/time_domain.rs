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

/// TINN için en fazla histogram kutusu (2048 × 7,8125 ms = 16 s RR aralığı). Daha geniş bir dağılım
/// (ör. bir boşluğun tek RR olarak girmesi) TINN = NaN verir; maliyet O(B²) ile sınırlı kalır.
pub const MAX_TINN_BINS: usize = 2048;

/// NN histogramı, kutular 0'dan başlayan sabit ızgarada (NeuroKit2 `np.arange(0, max + bin, bin)` ile
/// aynı kutular). Sadece dolu aralık `[k_min, k_last]` tutulur: `(k_min, sayımlar)`.
/// Kutu sayısı `max_bins`'i aşarsa `None` (bellek ayrılmaz).
fn histogram(sorted: &[f64], max_bins: usize) -> Option<(usize, Vec<usize>)> {
    let max = sorted[sorted.len() - 1];
    // numpy arange uzunluğu: ceil((max + bin) / bin) kenar → kenar - 1 kutu; son kutu sağdan kapalı
    let n_edges = ((max + HIST_BIN_MS) / HIST_BIN_MS).ceil() as usize;
    let k_last = n_edges.saturating_sub(2);
    let bin = |x: f64| ((x / HIST_BIN_MS).floor() as usize).min(k_last);
    let k_min = bin(sorted[0]);
    let width = k_last - k_min + 1;
    if width > max_bins {
        return None;
    }
    let mut h = vec![0usize; width];
    for &x in sorted {
        h[bin(x) - k_min] += 1;
    }
    Some((k_min, h))
}

/// En yüksek kutunun sayımı (HTI paydası), histogram ayırmadan (sıralı dizide koşu uzunluğu).
fn histogram_peak(sorted: &[f64]) -> usize {
    let max = sorted[sorted.len() - 1];
    let n_edges = ((max + HIST_BIN_MS) / HIST_BIN_MS).ceil() as usize;
    let k_last = n_edges.saturating_sub(2);
    let bin = |x: f64| ((x / HIST_BIN_MS).floor() as usize).min(k_last);
    let (mut best, mut run, mut cur) = (0usize, 0usize, usize::MAX);
    for &x in sorted {
        let k = bin(x);
        if k == cur {
            run += 1;
        } else {
            cur = k;
            run = 1;
        }
        best = best.max(run);
    }
    best
}

/// TINN: histogram tepesinin iki yanına üçgen oturtup en küçük karesel hatayı veren taban (M − N).
///
/// Hata `E(N, M) = A(N) + B(M)` biçiminde ayrışır (sol kenar yalnız N'ye, sağ kenar yalnız M'ye bağlı),
/// bu yüzden iki kenar ayrı ayrı aranır: O(B²) (eski tam arama O(B³) idi; sonuç aynı). Tepe, eşit
/// yükseklikte ilk kutu (NeuroKit `argmax` gibi).
fn tinn(hist: &[usize]) -> f64 {
    let len = hist.len();
    if len < 3 {
        return f64::NAN;
    }
    let peak_h = *hist.iter().max().unwrap();
    let peak = hist.iter().position(|&v| v == peak_h).unwrap();
    if peak == 0 || peak + 1 >= len {
        return f64::NAN;
    }
    let h = peak_h as f64;
    let sq = |v: usize| (v as f64) * (v as f64);
    // Sol: A(n) = Σ_{i<=n} v² + Σ_{n<i<=peak} (v_i − h (i−n)/(peak−n))²
    let mut best_n = (f64::INFINITY, 0usize);
    let mut below = 0.0;
    for n in 0..peak {
        below += sq(hist[n]);
        let mut err = below;
        for (i, &v) in hist.iter().enumerate().take(peak + 1).skip(n + 1) {
            let q = h * (i - n) as f64 / (peak - n) as f64;
            err += (v as f64 - q).powi(2);
        }
        if err < best_n.0 {
            best_n = (err, n);
        }
    }
    // Sağ: B(m) = Σ_{peak<i<m} (v_i − h (m−i)/(m−peak))² + Σ_{i>=m} v²
    let mut best_m = (f64::INFINITY, 0usize);
    let mut above: f64 = hist[peak + 1..].iter().map(|&v| sq(v)).sum();
    for m in (peak + 1)..len {
        let mut err = above;
        for (i, &v) in hist.iter().enumerate().take(m).skip(peak + 1) {
            let q = h * (m - i) as f64 / (m - peak) as f64;
            err += (v as f64 - q).powi(2);
        }
        if err < best_m.0 {
            best_m = (err, m);
        }
        above -= sq(hist[m]);
    }
    (best_m.1 - best_n.1) as f64 * HIST_BIN_MS
}

/// Bütün zaman alanı metriklerini hesaplar. En az 3 aralık gerekir; azsa ya da girdide NaN/sonsuz
/// değer varsa bütün alanlar NaN (panik yok).
///
/// HTI ve TINN histogramı 0'dan başlayan 7,8125 ms'lik ızgarayı kullanır (NeuroKit2 ile aynı kutular).
/// TINN, dağılım [`MAX_TINN_BINS`] kutudan genişse NaN döner.
pub fn time_domain(rri: &[f64]) -> TimeDomain {
    let nan = f64::NAN;
    if rri.len() < 3 || rri.iter().any(|x| !x.is_finite()) {
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
    sorted.sort_by(f64::total_cmp);
    let mean_nn = mean(rri);
    let sdnn = std_ddof(rri, 1);
    let rmssd = compute_rmssd(rri);
    let median_nn = percentile_sorted(&sorted, 50.0);
    let mut abs_dev: Vec<f64> = rri.iter().map(|x| (x - median_nn).abs()).collect();
    abs_dev.sort_by(f64::total_cmp);
    let hr: Vec<f64> = rri.iter().map(|x| 60_000.0 / x).collect();
    let (min_nn, max_nn) = (sorted[0], sorted[sorted.len() - 1]);
    let peak = histogram_peak(&sorted) as f64;
    let tinn_ms = histogram(&sorted, MAX_TINN_BINS).map_or(nan, |(_, h)| tinn(&h));

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
        tinn: tinn_ms,
    }
}

// =============================================================================
// MASKED (EXCLUSION) TIME DOMAIN
// =============================================================================

/// Fewest successive differences for the pair-based masked metrics (RMSSD, SDSD, pNNx); fewer give
/// `NaN`. Two, as [`time_domain`] (≥ 3 intervals = ≥ 2 differences) and as SDSD needs.
pub const MIN_MASKED_PAIRS: usize = 2;

/// Result of [`time_domain_masked`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MaskedTimeDomain {
    /// Metrics: distribution metrics on the valid NN, pair metrics on the valid adjacent pairs.
    pub time: TimeDomain,
    /// Valid NN intervals used.
    pub n_nn: usize,
    /// Successive differences used (pairs of adjacent valid NN).
    pub n_pairs: usize,
}

/// `true` where NN interval `k` is used: `valid[k]` and finite.
fn used(nn_ms: &[f64], valid: &[bool], k: usize) -> bool {
    valid.get(k).copied().unwrap_or(false) && nn_ms[k].is_finite()
}

/// Successive differences `nn[k+1] − nn[k]` (ms) over the pairs of **adjacent** NN intervals that
/// are both valid (and finite). No difference spans an excluded interval, a `NaN` entry or a gap
/// marker.
pub fn masked_differences(nn_ms: &[f64], valid: &[bool]) -> Vec<f64> {
    (0..nn_ms.len().saturating_sub(1))
        .filter(|&k| used(nn_ms, valid, k) && used(nn_ms, valid, k + 1))
        .map(|k| nn_ms[k + 1] - nn_ms[k])
        .collect()
}

/// RMSSD (ms) from the valid adjacent pairs only ([`masked_differences`]); `None` with fewer than
/// [`MIN_MASKED_PAIRS`] pairs.
pub fn rmssd_masked(nn_ms: &[f64], valid: &[bool]) -> Option<f64> {
    let d = masked_differences(nn_ms, valid);
    (d.len() >= MIN_MASKED_PAIRS).then(|| rmssd_of_differences(&d))
}

fn rmssd_of_differences(d: &[f64]) -> f64 {
    // same operations and order as `compute_rmssd`
    let sum_sq: f64 = d.iter().map(|x| x.powi(2)).sum();
    (sum_sq / d.len() as f64).sqrt()
}

/// Time-domain metrics with an NN validity mask (exclusion mode, [`crate::exclusion`]).
///
/// `nn_ms[k]` is used if `valid[k]` is `true` and the value is finite (a missing `valid` entry counts
/// as invalid; a `NaN` entry can mark a gap between segments, see [`crate::exclusion::concat_nn`]).
///
/// - **Pair metrics** RMSSD, SDSD, pNN50, pNN20 (and CVSD = RMSSD / MeanNN) use only the successive
///   differences of **adjacent** intervals that are both used: an excluded interval or a gap breaks
///   the series and no difference spans it. Fewer than [`MIN_MASKED_PAIRS`] differences → `NaN`.
/// - **Distribution metrics** (MeanNN, SDNN, median, MAD, IQR, min, max, CVNN, heart rate, HTI,
///   TINN) are [`time_domain`] of the used intervals (≥ 3, else `NaN`).
///
/// With every interval valid and finite the result equals [`time_domain`] bit for bit.
pub fn time_domain_masked(nn_ms: &[f64], valid: &[bool]) -> MaskedTimeDomain {
    let kept: Vec<f64> = (0..nn_ms.len())
        .filter(|&k| used(nn_ms, valid, k))
        .map(|k| nn_ms[k])
        .collect();
    let d = masked_differences(nn_ms, valid);
    let mut time = time_domain(&kept);
    if d.len() >= MIN_MASKED_PAIRS {
        let n = d.len() as f64;
        time.rmssd = rmssd_of_differences(&d);
        time.sdsd = std_ddof(&d, 1);
        time.pnn50 = (d.iter().filter(|x| x.abs() > 50.0).count() as f64 / n) * 100.0;
        time.pnn20 = (d.iter().filter(|x| x.abs() > 20.0).count() as f64 / n) * 100.0;
        time.cvsd = time.rmssd / time.mean_nn;
    } else {
        time.rmssd = f64::NAN;
        time.sdsd = f64::NAN;
        time.pnn50 = f64::NAN;
        time.pnn20 = f64::NAN;
        time.cvsd = f64::NAN;
    }
    MaskedTimeDomain {
        time,
        n_nn: kept.len(),
        n_pairs: d.len(),
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

    /// Eski O(B³) tam arama; ayrıştırılmış O(B²) sürümle aynı sonucu vermeli.
    fn tinn_brute(hist: &[usize]) -> f64 {
        let peak_h = *hist.iter().max().unwrap();
        let peak = hist.iter().position(|&v| v == peak_h).unwrap();
        let h = peak_h as f64;
        let mut best = (f64::INFINITY, 0usize, 0usize);
        for n in 0..peak {
            for m in (peak + 1)..hist.len() {
                let mut err = 0.0;
                for (i, &v) in hist.iter().enumerate() {
                    let q = if i <= n || i >= m {
                        0.0
                    } else if i <= peak {
                        h * (i - n) as f64 / (peak - n) as f64
                    } else {
                        h * (m - i) as f64 / (m - peak) as f64
                    };
                    err += (v as f64 - q).powi(2);
                }
                if err < best.0 - 1e-9 {
                    best = (err, n, m);
                }
            }
        }
        (best.2 - best.1) as f64 * HIST_BIN_MS
    }

    #[test]
    fn tinn_separable_equals_brute_force() {
        let mut state = 12345u64;
        for _ in 0..200 {
            let len = 3 + (state % 30) as usize;
            let hist: Vec<usize> = (0..len)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((state >> 33) % 20) as usize
                })
                .collect();
            let peak = hist
                .iter()
                .position(|&v| v == *hist.iter().max().unwrap())
                .unwrap();
            if peak == 0 || peak + 1 >= len {
                continue;
            }
            assert_eq!(tinn(&hist), tinn_brute(&hist), "{hist:?}");
        }
    }

    #[test]
    fn nan_input_gives_nan_not_panic() {
        let mut v = RRI.to_vec();
        v[3] = f64::NAN;
        let t = time_domain(&v);
        assert!(t.sdnn.is_nan() && t.tinn.is_nan() && t.hti.is_nan());
    }

    fn same(a: f64, b: f64) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }

    #[test]
    fn masked_all_valid_equals_time_domain() {
        let v: Vec<f64> = (0..300)
            .map(|i| 800.0 + (i as f64 * 0.37).sin() * 40.0 + (i % 7) as f64 * 3.0)
            .collect();
        for rri in [&RRI[..], &v[..]] {
            let t = time_domain(rri);
            let m = time_domain_masked(rri, &vec![true; rri.len()]);
            assert_eq!((m.n_nn, m.n_pairs), (rri.len(), rri.len() - 1));
            let (a, b) = (m.time, t);
            for (x, y) in [
                (a.mean_nn, b.mean_nn),
                (a.sdnn, b.sdnn),
                (a.rmssd, b.rmssd),
                (a.sdsd, b.sdsd),
                (a.pnn50, b.pnn50),
                (a.pnn20, b.pnn20),
                (a.median_nn, b.median_nn),
                (a.mad_nn, b.mad_nn),
                (a.iqr_nn, b.iqr_nn),
                (a.cvnn, b.cvnn),
                (a.cvsd, b.cvsd),
                (a.mean_hr, b.mean_hr),
                (a.sd_hr, b.sd_hr),
                (a.hti, b.hti),
                (a.tinn, b.tinn),
            ] {
                assert!(same(x, y), "{x} vs {y}");
            }
            assert_eq!(rmssd_masked(rri, &vec![true; rri.len()]), Some(b.rmssd));
        }
    }

    #[test]
    fn masked_pairs_do_not_span_excluded_intervals() {
        let nn = [800.0, 900.0, 600.0, 1000.0, 810.0, 820.0, 790.0];
        let valid = [true, true, false, false, true, true, true];
        assert_eq!(masked_differences(&nn, &valid), vec![100.0, 10.0, -30.0]);
        let m = time_domain_masked(&nn, &valid);
        assert_eq!((m.n_nn, m.n_pairs), (5, 3));
        let want = ((100.0f64.powi(2) + 10.0f64.powi(2) + 30.0f64.powi(2)) / 3.0).sqrt();
        assert_relative_eq!(m.time.rmssd, want, epsilon = 1e-12);
        assert_relative_eq!(m.time.pnn50, 100.0 / 3.0, epsilon = 1e-12);
        assert_relative_eq!(
            m.time.mean_nn,
            (800.0 + 900.0 + 810.0 + 820.0 + 790.0) / 5.0
        );
        // a NaN entry (gap marker) breaks the series like an invalid one
        let nn2 = [800.0, 900.0, f64::NAN, 1000.0, 810.0, 820.0, 790.0];
        let m2 = time_domain_masked(&nn2, &[true; 7]);
        assert_eq!(m2.n_pairs, 4);
        // shorter mask: missing entries are invalid
        assert_eq!(masked_differences(&nn, &[true, true]), vec![100.0]);
    }

    #[test]
    fn masked_too_few_pairs_is_nan() {
        let nn = [800.0, 810.0, 790.0, 805.0, 800.0];
        // valid NN 0, 2, 4: three intervals but no adjacent pair
        let valid = [true, false, true, false, true];
        let m = time_domain_masked(&nn, &valid);
        assert_eq!((m.n_nn, m.n_pairs), (3, 0));
        assert!(m.time.rmssd.is_nan() && m.time.sdsd.is_nan() && m.time.pnn50.is_nan());
        assert!(m.time.cvsd.is_nan());
        assert!(m.time.mean_nn.is_finite() && m.time.sdnn.is_finite());
        assert_eq!(rmssd_masked(&nn, &valid), None);
        // one pair only
        assert_eq!(rmssd_masked(&nn, &[true, true, false, false, false]), None);
        assert!(time_domain_masked(&[], &[]).time.rmssd.is_nan());
    }

    #[test]
    fn huge_rr_is_bounded() {
        // tek bir 69 sn'lik RR (boşluk) TINN'i NaN yapar, süreyi patlatmaz
        let mut v: Vec<f64> = (0..400).map(|i| 800.0 + (i as f64).sin() * 30.0).collect();
        v[100] = 69_000.0;
        let t0 = std::time::Instant::now();
        let t = time_domain(&v);
        assert!(t.tinn.is_nan());
        assert!(t.hti.is_finite());
        assert!(t0.elapsed().as_millis() < 500);
    }
}
