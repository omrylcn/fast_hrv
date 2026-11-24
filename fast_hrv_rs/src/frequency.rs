//! Frekans alanı HRV metrikleri (Welch yöntemi).
//!
//! Adımlar (Task Force 1996 / NeuroKit2 ile aynı mantık):
//! 1. RR aralıklarından zaman ekseni (atımların zamanı) çıkarılır.
//! 2. RR serisi 4 Hz'e doğal kübik spline ile eşit aralıklı örneklenir.
//! 3. Ortalama çıkarılır; Welch: Hann penceresi, %50 örtüşme, segment 256 örnek (64 sn; kısa kayıtta tamamı).
//! 4. Güç yoğunluğu (ms²/Hz) bantlar üzerinde yamuk kuralıyla, bant kenarlarında kırpılarak toplanır
//!    (kenardaki yamuk iki banda birden yazılmaz).
//! 5. Bant başına en az süre: HF ≥ 60 sn, LF ≥ 120 sn, VLF ≥ 300 sn; altında NaN ([`FrequencyOptions`]).
//!    Sabit (değişkenliği sayısal gürültü düzeyinde) RR serisinde güçler 0, LF/HF ve oranlar NaN.
//!
//! Bantlar: VLF 0.0033–0.04, LF 0.04–0.15, HF 0.15–0.4 Hz. Normalize: LFnu = LF / (LF + HF) × 100.
//! Dış bağımlılık yok (FFT kendi içinde, radix-2).

use std::f64::consts::PI;

pub const RESAMPLE_HZ: f64 = 4.0;
pub const VLF: (f64, f64) = (0.0033, 0.04);
pub const LF: (f64, f64) = (0.04, 0.15);
pub const HF: (f64, f64) = (0.15, 0.4);

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FrequencyDomain {
    /// Bant güçleri (ms²)
    pub vlf: f64,
    pub lf: f64,
    pub hf: f64,
    /// VLF + LF + HF (ms²)
    pub total_power: f64,
    pub lf_hf: f64,
    /// Normalize güçler (%): LF / (LF + HF), HF / (LF + HF)
    pub lfnu: f64,
    pub hfnu: f64,
    /// LF ve HF bantlarındaki tepe frekansları (Hz)
    pub lf_peak: f64,
    pub hf_peak: f64,
}

impl FrequencyDomain {
    fn nan() -> Self {
        let n = f64::NAN;
        Self {
            vlf: n,
            lf: n,
            hf: n,
            total_power: n,
            lf_hf: n,
            lfnu: n,
            hfnu: n,
            lf_peak: n,
            hf_peak: n,
        }
    }
}

/// Doğal kübik spline ile (x, y) noktalarından `xs` üzerinde değer üretir. `x` artan olmalı.
pub fn cubic_spline(x: &[f64], y: &[f64], xs: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n < 3 {
        return xs
            .iter()
            .map(|_| y.first().copied().unwrap_or(f64::NAN))
            .collect();
    }
    // İkinci türevler (doğal sınır: uçlarda 0), üç köşegenli sistem (Thomas algoritması)
    let mut m = vec![0.0; n];
    let mut c = vec![0.0; n];
    let mut d = vec![0.0; n];
    for i in 1..n - 1 {
        let h0 = x[i] - x[i - 1];
        let h1 = x[i + 1] - x[i];
        let a = h0;
        let b = 2.0 * (h0 + h1);
        let cc = h1;
        let rhs = 6.0 * ((y[i + 1] - y[i]) / h1 - (y[i] - y[i - 1]) / h0);
        let denom = b - a * c[i - 1];
        c[i] = cc / denom;
        d[i] = (rhs - a * d[i - 1]) / denom;
    }
    for i in (1..n - 1).rev() {
        m[i] = d[i] - c[i] * m[i + 1];
    }
    let mut out = Vec::with_capacity(xs.len());
    let mut k = 0;
    for &t in xs {
        while k < n - 2 && t > x[k + 1] {
            k += 1;
        }
        let h = x[k + 1] - x[k];
        let a = (x[k + 1] - t) / h;
        let b = (t - x[k]) / h;
        out.push(
            a * y[k]
                + b * y[k + 1]
                + ((a.powi(3) - a) * m[k] + (b.powi(3) - b) * m[k + 1]) * h * h / 6.0,
        );
    }
    out
}

/// Yerinde, özyinelemesiz radix-2 FFT. `re.len()` 2'nin kuvveti olmalı.
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * PI / len as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        for start in (0..n).step_by(len) {
            let (mut cr, mut ci) = (1.0, 0.0);
            for k in 0..len / 2 {
                let (a, b) = (start + k, start + k + len / 2);
                let tr = re[b] * cr - im[b] * ci;
                let ti = re[b] * ci + im[b] * cr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                let ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
        }
        len <<= 1;
    }
}

/// Welch güç spektral yoğunluğu (tek taraflı, yoğunluk ölçekli; SciPy `welch(..., window='hann', scaling='density')` ile aynı).
/// Döner: (frekanslar Hz, PSD birim²/Hz)
pub fn welch(signal: &[f64], fs: f64, nperseg: usize) -> (Vec<f64>, Vec<f64>) {
    let nperseg = nperseg.min(signal.len()).max(2);
    let nfft = nperseg.next_power_of_two();
    let step = (nperseg / 2).max(1);
    // Periyodik Hann penceresi (SciPy get_window('hann') varsayılanı)
    let win: Vec<f64> = (0..nperseg)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / nperseg as f64).cos())
        .collect();
    let win_pow: f64 = win.iter().map(|w| w * w).sum();
    let n_freq = nfft / 2 + 1;
    let mut psd = vec![0.0; n_freq];
    let mut segments = 0;
    let mut start = 0;
    while start + nperseg <= signal.len() {
        let seg = &signal[start..start + nperseg];
        let seg_mean = seg.iter().sum::<f64>() / nperseg as f64; // SciPy detrend='constant'
        let mut re = vec![0.0; nfft];
        let mut im = vec![0.0; nfft];
        for i in 0..nperseg {
            re[i] = (seg[i] - seg_mean) * win[i];
        }
        fft(&mut re, &mut im);
        for k in 0..n_freq {
            psd[k] += re[k] * re[k] + im[k] * im[k];
        }
        segments += 1;
        start += step;
    }
    let scale = 1.0 / (fs * win_pow * segments.max(1) as f64);
    for (k, p) in psd.iter_mut().enumerate() {
        *p *= scale;
        if k != 0 && !(nfft % 2 == 0 && k == n_freq - 1) {
            *p *= 2.0; // tek taraflı
        }
    }
    let freqs = (0..n_freq).map(|k| k as f64 * fs / nfft as f64).collect();
    (freqs, psd)
}

/// Exact integral of the piecewise-linear PSD over `[lo, hi]` (trapezoids clipped at the band edges,
/// PSD linearly interpolated there). Adjacent bands share their edge, so no trapezoid is counted twice
/// and VLF + LF + HF equals the integral over 0.0033–0.4 Hz.
fn band_power(freqs: &[f64], psd: &[f64], (lo, hi): (f64, f64)) -> f64 {
    let mut p = 0.0;
    for k in 1..freqs.len() {
        let (f0, f1) = (freqs[k - 1], freqs[k]);
        let a = f0.max(lo);
        let b = f1.min(hi);
        if b <= a {
            continue;
        }
        let at = |f: f64| psd[k - 1] + (psd[k] - psd[k - 1]) * (f - f0) / (f1 - f0);
        p += 0.5 * (at(a) + at(b)) * (b - a);
    }
    p
}

fn band_peak(freqs: &[f64], psd: &[f64], (lo, hi): (f64, f64)) -> f64 {
    freqs
        .iter()
        .zip(psd)
        .filter(|(f, _)| **f >= lo && **f < hi)
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(f, _)| *f)
        .unwrap_or(f64::NAN)
}

/// Minimum record durations per band (seconds of RR data, i.e. the sum of the intervals).
/// Below the minimum the band's power, its ratios and its peak frequency are NaN.
/// Defaults follow the Task Force (1996) recommendation: about 10 cycles of the band's lowest
/// frequency (HF ≥ 1 min, LF ≥ 2 min, VLF ≥ 5 min).
///
/// These defaults are **technical lower limits** (below them the band cannot be estimated), not
/// recommended analysis lengths: for interpretable LF and LF/HF use ≥ 180 s at the very least and
/// ≥ 300 s when reporting or comparing them (see the README, "Minimum data").
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrequencyOptions {
    pub min_duration_hf_s: f64,
    pub min_duration_lf_s: f64,
    pub min_duration_vlf_s: f64,
}

impl Default for FrequencyOptions {
    fn default() -> Self {
        Self {
            min_duration_hf_s: 60.0,
            min_duration_lf_s: 120.0,
            min_duration_vlf_s: 300.0,
        }
    }
}

/// Relative numerical floor: band powers below `POWER_FLOOR_REL × mean(RR)²` are rounding noise
/// (e.g. a constant RR series) and are reported as 0, with undefined ratios (NaN).
pub const POWER_FLOOR_REL: f64 = 1e-12;

/// RR aralıklarından (ms) frekans alanı metrikleri, varsayılan en az süre kurallarıyla
/// (HF ≥ 60 s, LF ≥ 120 s, VLF ≥ 300 s; bkz. [`FrequencyOptions`]). En az 10 aralık gerekir.
pub fn frequency_domain(rri: &[f64]) -> FrequencyDomain {
    frequency_domain_with(rri, &FrequencyOptions::default())
}

/// [`frequency_domain`] with explicit per-band minimum durations.
///
/// - Bands whose minimum duration is not reached are NaN (power, peak, and every ratio using them;
///   `total_power` needs all three bands).
/// - A (near-)constant RR series gives 0 power and NaN for LF/HF, LFnu, HFnu and the peaks.
/// - NaN, infinite or non-positive intervals give all NaN.
pub fn frequency_domain_with(rri: &[f64], opts: &FrequencyOptions) -> FrequencyDomain {
    if rri.len() < 10 || rri.iter().any(|r| !(r.is_finite() && *r > 0.0)) {
        return FrequencyDomain::nan();
    }
    let record_s = rri.iter().sum::<f64>() / 1000.0;
    // Atım zamanları (sn): ilk aralığın sonu t0
    let mut t = Vec::with_capacity(rri.len());
    let mut acc = 0.0;
    for &r in rri {
        acc += r / 1000.0;
        t.push(acc);
    }
    let duration = t[t.len() - 1] - t[0];
    let min_any = opts
        .min_duration_hf_s
        .min(opts.min_duration_lf_s)
        .min(opts.min_duration_vlf_s);
    if record_s < min_any || duration * RESAMPLE_HZ < 2.0 {
        return FrequencyDomain::nan();
    }
    let n = (duration * RESAMPLE_HZ).floor() as usize + 1;
    let grid: Vec<f64> = (0..n).map(|i| t[0] + i as f64 / RESAMPLE_HZ).collect();
    let resampled = cubic_spline(&t, rri, &grid);
    let (freqs, psd) = welch(&resampled, RESAMPLE_HZ, 256);

    let mean_rr = rri.iter().sum::<f64>() / rri.len() as f64;
    let floor = POWER_FLOOR_REL * mean_rr * mean_rr;
    let band = |b: (f64, f64), min_s: f64| -> f64 {
        if record_s < min_s {
            return f64::NAN;
        }
        let p = band_power(&freqs, &psd, b);
        if p <= floor { 0.0 } else { p }
    };
    let vlf = band(VLF, opts.min_duration_vlf_s);
    let lf = band(LF, opts.min_duration_lf_s);
    let hf = band(HF, opts.min_duration_hf_s);
    let positive = |p: f64| p.is_finite() && p > 0.0;
    let lfhf_sum = lf + hf;
    let peak = |p: f64, b: (f64, f64)| {
        if positive(p) {
            band_peak(&freqs, &psd, b)
        } else {
            f64::NAN
        }
    };
    FrequencyDomain {
        vlf,
        lf,
        hf,
        total_power: vlf + lf + hf,
        lf_hf: if positive(hf) && lf.is_finite() {
            lf / hf
        } else {
            f64::NAN
        },
        lfnu: if positive(lfhf_sum) {
            lf / lfhf_sum * 100.0
        } else {
            f64::NAN
        },
        hfnu: if positive(lfhf_sum) {
            hf / lfhf_sum * 100.0
        } else {
            f64::NAN
        },
        lf_peak: peak(lf, LF),
        hf_peak: peak(hf, HF),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn fft_matches_dft() {
        let x = [1.0, 2.0, 0.0, -1.0, 3.0, 0.5, -2.0, 1.0];
        let (mut re, mut im) = (x.to_vec(), vec![0.0; 8]);
        fft(&mut re, &mut im);
        for k in 0..8 {
            let (mut r, mut i) = (0.0, 0.0);
            for (n, v) in x.iter().enumerate() {
                let a = -2.0 * PI * (k * n) as f64 / 8.0;
                r += v * a.cos();
                i += v * a.sin();
            }
            assert_relative_eq!(re[k], r, epsilon = 1e-9);
            assert_relative_eq!(im[k], i, epsilon = 1e-9);
        }
    }

    #[test]
    fn spline_reproduces_linear_and_knots() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let y = [0.0, 2.0, 4.0, 6.0];
        let v = cubic_spline(&x, &y, &[0.5, 1.0, 2.5]);
        assert_relative_eq!(v[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(v[1], 2.0, epsilon = 1e-12);
        assert_relative_eq!(v[2], 5.0, epsilon = 1e-12);
    }

    #[test]
    fn welch_parseval_white_noise_density() {
        // Sabit olmayan bir sinüs: tepe frekansı doğru bulunmalı
        let fs = 4.0;
        let f0 = 0.25;
        let x: Vec<f64> = (0..1024)
            .map(|i| (2.0 * PI * f0 * i as f64 / fs).sin())
            .collect();
        let (f, p) = welch(&x, fs, 256);
        let peak = f
            .iter()
            .zip(&p)
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_relative_eq!(*peak, f0, epsilon = fs / 256.0);
        // Sinüsün gücü (genlik²/2 = 0.5) ≈ PSD'nin integrali
        let total: f64 = (1..f.len())
            .map(|k| 0.5 * (p[k - 1] + p[k]) * (f[k] - f[k - 1]))
            .sum();
        assert_relative_eq!(total, 0.5, epsilon = 0.05);
    }

    #[test]
    fn respiratory_modulation_lands_in_hf() {
        // 0.25 Hz (solunum) ile modüle RR → güç HF bandında, LF/HF küçük
        let mut rri = Vec::new();
        let mut t = 0.0;
        while t < 300.0 {
            let r = 800.0 + 40.0 * (2.0 * PI * 0.25 * t).sin();
            rri.push(r);
            t += r / 1000.0;
        }
        let fd = frequency_domain(&rri);
        assert!(fd.hf > fd.lf * 5.0, "HF={} LF={}", fd.hf, fd.lf);
        assert!((fd.hf_peak - 0.25).abs() < 0.02, "hf_peak={}", fd.hf_peak);
        assert!(fd.lfnu < 20.0);
    }

    #[test]
    fn baroreflex_modulation_lands_in_lf() {
        let mut rri = Vec::new();
        let mut t = 0.0;
        while t < 300.0 {
            let r = 800.0 + 40.0 * (2.0 * PI * 0.1 * t).sin();
            rri.push(r);
            t += r / 1000.0;
        }
        let fd = frequency_domain(&rri);
        assert!(fd.lf > fd.hf * 5.0);
        assert!((fd.lf_peak - 0.1).abs() < 0.02);
        assert!(fd.lf_hf > 5.0);
    }

    #[test]
    fn band_edges_are_not_counted_twice() {
        // flat PSD of 1: band power must equal the band width exactly
        let freqs: Vec<f64> = (0..129).map(|k| k as f64 * 4.0 / 256.0).collect();
        let psd = vec![1.0; freqs.len()];
        for b in [VLF, LF, HF] {
            assert_relative_eq!(band_power(&freqs, &psd, b), b.1 - b.0, epsilon = 1e-12);
        }
        let total = band_power(&freqs, &psd, (VLF.0, HF.1));
        let sum = band_power(&freqs, &psd, VLF)
            + band_power(&freqs, &psd, LF)
            + band_power(&freqs, &psd, HF);
        assert_relative_eq!(total, sum, epsilon = 1e-12);
    }

    #[test]
    fn constant_rr_has_no_lf_hf() {
        let fd = frequency_domain(&[800.0; 500]);
        assert!(
            fd.lf_hf.is_nan() && fd.lfnu.is_nan() && fd.hfnu.is_nan(),
            "{fd:?}"
        );
        assert_eq!(fd.lf, 0.0);
        assert_eq!(fd.hf, 0.0);
        assert!(fd.lf_peak.is_nan());
    }

    #[test]
    fn per_band_minimum_durations() {
        let series = |secs: f64| {
            let mut rri = Vec::new();
            let mut t = 0.0;
            while t < secs {
                let r =
                    800.0 + 30.0 * (2.0 * PI * 0.25 * t).sin() + 20.0 * (2.0 * PI * 0.1 * t).sin();
                rri.push(r);
                t += r / 1000.0;
            }
            rri
        };
        let f90 = frequency_domain(&series(90.0));
        assert!(f90.hf.is_finite() && f90.lf.is_nan() && f90.vlf.is_nan() && f90.lf_hf.is_nan());
        assert!(f90.hf_peak.is_finite() && f90.lf_peak.is_nan() && f90.total_power.is_nan());
        let f150 = frequency_domain(&series(150.0));
        assert!(
            f150.hf.is_finite()
                && f150.lf.is_finite()
                && f150.vlf.is_nan()
                && f150.lf_hf.is_finite()
        );
        let f300 = frequency_domain(&series(300.0));
        assert!(f300.vlf.is_finite() && f300.total_power.is_finite());
        assert!(frequency_domain(&series(50.0)).hf.is_nan());
        // custom thresholds
        let o = FrequencyOptions {
            min_duration_hf_s: 30.0,
            min_duration_lf_s: 30.0,
            min_duration_vlf_s: 30.0,
        };
        assert!(frequency_domain_with(&series(50.0), &o).vlf.is_finite());
    }

    #[test]
    fn short_record_is_nan() {
        assert!(frequency_domain(&[800.0; 20]).lf.is_nan()); // 16 sn < 60 sn (HF en az süresi)
    }
}
