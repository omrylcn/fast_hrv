//! NeuroKit2-compatible frequency domain (`nk.hrv_frequency` defaults), for parity with NeuroKit2.
//!
//! fast-hrv's default ([`crate::frequency_domain`]) follows the Task Force / Kubios practice: RR resampled
//! at 4 Hz with a cubic spline, Welch with 64 s Hann segments, band power integrated exactly up to the
//! band edges. NeuroKit2 makes different (equally valid) choices, which move LF/HF by a few percent,
//! mostly through the window length. This module reproduces NeuroKit2's choices step by step:
//!
//! 1. Beat times = cumulative sum of the RR intervals (s).
//! 2. Grid `np.arange(t0, tN + 1/100, 1/100)`: 100 Hz.
//! 3. Quadratic interpolating spline (`scipy.interpolate.interp1d(kind="quadratic")` =
//!    `make_interp_spline(k=2)` with knots at the midpoints between samples), constant outside.
//! 4. Mean removed from the whole series.
//! 5. Welch, Hann window of half the series (`nperseg = N/2`), 50 % overlap, FFT length `2·nperseg`,
//!    no per-segment detrending, density scaling, mean over segments.
//! 6. Band power = trapezoid over the PSD points with `lo ≤ f < hi` (no edge interpolation); 0 → NaN.
//!
//! Differences that remain on purpose: band powers are in ms² (NeuroKit2 divides the PSD by its
//! maximum, so its absolute band powers are unitless; ratios are unaffected), and `lfnu`/`hfnu` keep
//! fast-hrv's definition `LF / (LF + HF)` (NeuroKit2's `LFn` is `LF / TP`). No minimum durations are
//! applied, as in NeuroKit2: below ~180 s LF and LF/HF are not interpretable.

use crate::frequency::{FrequencyDomain, HF, LF, VLF, bands_to_domain_pub, fft_pow2};
use std::f64::consts::PI;

/// Resampling rate of the RR series (Hz), NeuroKit2's `interpolation_rate` default.
pub const NK_INTERPOLATION_HZ: f64 = 100.0;

/// Frequency-domain metrics computed as `nk.hrv_frequency(peaks)` does with its defaults (see the
/// module documentation). `rri` in ms; at least 10 intervals.
pub fn frequency_domain_neurokit(rri: &[f64]) -> FrequencyDomain {
    if rri.len() < 10 || rri.iter().any(|r| !(r.is_finite() && *r > 0.0)) {
        return FrequencyDomain::nan_pub();
    }
    let mut t = Vec::with_capacity(rri.len());
    let mut acc = 0.0;
    for &r in rri {
        acc += r / 1000.0;
        t.push(acc);
    }
    let fs = NK_INTERPOLATION_HZ;
    let step = 1.0 / fs;
    let (start, stop) = (t[0], t[t.len() - 1] + step);
    // numpy.arange: length ceil((stop - start) / step), values start + i·step
    let n = ((stop - start) / step).ceil() as usize;
    if n < 8 {
        return FrequencyDomain::nan_pub();
    }
    let grid: Vec<f64> = (0..n).map(|i| start + i as f64 * step).collect();
    let mut x = quadratic_interp_scipy(&t, rri, &grid);
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    for v in x.iter_mut() {
        *v -= mean;
    }
    // nk.signal_psd: min_frequency = (2·fs) / (N/2), nperseg = int((2 / min_frequency)·fs), capped at N/2
    let nn = x.len() as f64;
    let min_frequency = (2.0 * fs) / (nn / 2.0);
    let mut nperseg = ((2.0 / min_frequency) * fs) as usize;
    if nperseg as f64 > nn / 2.0 {
        nperseg = (nn / 2.0) as usize;
    }
    let (freqs, psd) = welch_neurokit(&x, fs, nperseg);
    let band = |(lo, hi): (f64, f64)| -> f64 {
        let (mut p, mut prev): (f64, Option<(f64, f64)>) = (0.0, None);
        for (&f, &v) in freqs.iter().zip(&psd) {
            if f >= lo && f < hi {
                if let Some((f0, v0)) = prev {
                    p += 0.5 * (v0 + v) * (f - f0);
                }
                prev = Some((f, v));
            }
        }
        if p == 0.0 { f64::NAN } else { p }
    };
    bands_to_domain_pub(&freqs, &psd, band(VLF), band(LF), band(HF))
}

/// `scipy.interpolate.interp1d(x, y, kind="quadratic")` evaluated at `xs`, with the constant fill
/// NeuroKit2 uses outside `[x0, xN]` (first / last value). `x` strictly increasing.
pub fn quadratic_interp_scipy(x: &[f64], y: &[f64], xs: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n < 3 {
        return xs.iter().map(|_| y[0]).collect();
    }
    // knots: x0 ×3, midpoints (x[i] + x[i+1]) / 2 for i = 1..n-3, x_{n-1} ×3  (len n + 3)
    let mut kn = Vec::with_capacity(n + 3);
    kn.extend([x[0]; 3]);
    for i in 1..n - 2 {
        kn.push((x[i] + x[i + 1]) / 2.0);
    }
    kn.extend([x[n - 1]; 3]);
    // collocation: row i has the 3 basis functions active at x[i]; solve the banded system
    // (each row's nonzeros are at columns span-2..=span) with partial pivoting on a dense band.
    let span_of = |v: f64, hint: usize| -> usize {
        // largest s in [2, n-1] with kn[s] <= v < kn[s+1] (last interval closed)
        let mut s = hint.max(2);
        while s + 1 < n && kn[s + 1] <= v {
            s += 1;
        }
        s.min(n - 1)
    };
    let basis = |s: usize, v: f64| -> [f64; 3] {
        // de Boor–Cox for degree 2 at knot span s (kn[s] <= v < kn[s+1])
        let mut b = [1.0, 0.0, 0.0];
        let (mut left, mut right) = ([0.0; 3], [0.0; 3]);
        for j in 1..=2 {
            left[j] = v - kn[s + 1 - j];
            right[j] = kn[s + j] - v;
            let mut saved = 0.0;
            for r in 0..j {
                let den = right[r + 1] + left[j - r];
                let tmp = if den != 0.0 { b[r] / den } else { 0.0 };
                b[r] = saved + right[r + 1] * tmp;
                saved = left[j - r] * tmp;
            }
            b[j] = saved;
        }
        b
    };
    // banded matrix stored as rows of width 5 around the diagonal (cols i-2..=i+2)
    let mut a = vec![[0.0f64; 5]; n];
    let mut rhs = y.to_vec();
    let mut s = 2;
    for i in 0..n {
        s = span_of(x[i], s);
        let b = basis(s, x[i]);
        for (k, bk) in b.iter().enumerate() {
            let col = s - 2 + k;
            let off = col as isize - i as isize + 2;
            debug_assert!((0..5).contains(&off));
            a[i][off as usize] = *bk;
        }
    }
    let c = solve_banded5(&mut a, &mut rhs);
    let mut out = Vec::with_capacity(xs.len());
    let mut s = 2;
    for &v in xs {
        if v < x[0] {
            out.push(y[0]);
            continue;
        }
        if v > x[n - 1] {
            out.push(y[n - 1]);
            continue;
        }
        if v < kn[s] {
            s = 2;
        }
        s = span_of(v, s);
        let b = basis(s, v);
        out.push(b[0] * c[s - 2] + b[1] * c[s - 1] + b[2] * c[s]);
    }
    out
}

/// Gaussian elimination with partial pivoting for a matrix with nonzeros only in columns
/// i-2..=i+2 of row i (stored as `a[i][col - i + 2]`). Pivoting between neighbouring rows keeps the
/// fill inside the band (rows are swapped only with the next two).
fn solve_banded5(a: &mut [[f64; 5]], b: &mut [f64]) -> Vec<f64> {
    let n = b.len();
    // dense working copy of the band with room for fill (cols i-2..=i+4)
    let w = 7;
    let mut m = vec![vec![0.0f64; w]; n];
    for i in 0..n {
        for k in 0..5 {
            m[i][k] = a[i][k];
        }
    }
    let at = |m: &Vec<Vec<f64>>, r: usize, c: usize| -> f64 {
        let off = c as isize - r as isize + 2;
        if (0..w as isize).contains(&off) {
            m[r][off as usize]
        } else {
            0.0
        }
    };
    for col in 0..n {
        let last = (col + 2).min(n - 1);
        let mut piv = col;
        for r in col + 1..=last {
            if at(&m, r, col).abs() > at(&m, piv, col).abs() {
                piv = r;
            }
        }
        if piv != col {
            // swap rows in absolute column space
            let cols: Vec<usize> = (col.saturating_sub(2)..(col + 5).min(n)).collect();
            let rowc: Vec<f64> = cols.iter().map(|&c| at(&m, col, c)).collect();
            let rowp: Vec<f64> = cols.iter().map(|&c| at(&m, piv, c)).collect();
            for (j, &c) in cols.iter().enumerate() {
                set(&mut m, col, c, rowp[j]);
                set(&mut m, piv, c, rowc[j]);
            }
            b.swap(col, piv);
        }
        let d = at(&m, col, col);
        if d == 0.0 {
            continue;
        }
        for r in col + 1..=last {
            let f = at(&m, r, col) / d;
            if f == 0.0 {
                continue;
            }
            for c in col..(col + 5).min(n) {
                let v = at(&m, r, c) - f * at(&m, col, c);
                set(&mut m, r, c, v);
            }
            b[r] -= f * b[col];
        }
    }
    let mut xv = vec![0.0; n];
    for r in (0..n).rev() {
        let mut s = b[r];
        for c in r + 1..(r + 5).min(n) {
            s -= at(&m, r, c) * xv[c];
        }
        let d = at(&m, r, r);
        xv[r] = if d != 0.0 { s / d } else { 0.0 };
    }
    let _ = a;
    xv
}

fn set(m: &mut [Vec<f64>], r: usize, c: usize, v: f64) {
    let off = c as isize - r as isize + 2;
    if (0..m[r].len() as isize).contains(&off) {
        m[r][off as usize] = v;
    } else {
        debug_assert!(v.abs() < 1e-300, "fill outside band");
    }
}

/// `scipy.signal.welch(x, fs, window="hann", nperseg, nfft=2·nperseg, detrend=False,
/// scaling="density", average="mean")` (noverlap = nperseg / 2, no boundary padding).
fn welch_neurokit(x: &[f64], fs: f64, nperseg: usize) -> (Vec<f64>, Vec<f64>) {
    let nperseg = nperseg.max(2).min(x.len());
    let nfft = 2 * nperseg;
    let noverlap = nperseg / 2;
    let step = nperseg - noverlap;
    let win: Vec<f64> = (0..nperseg)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / nperseg as f64).cos())
        .collect();
    let win_pow: f64 = win.iter().map(|w| w * w).sum();
    let n_freq = nfft / 2 + 1;
    let mut psd = vec![0.0; n_freq];
    let mut segments = 0usize;
    let mut start = 0;
    let dft = Bluestein::new(nfft);
    while start + nperseg <= x.len() {
        let mut re = vec![0.0; nfft];
        let mut im = vec![0.0; nfft];
        for i in 0..nperseg {
            re[i] = x[start + i] * win[i];
        }
        dft.run(&mut re, &mut im);
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
            *p *= 2.0;
        }
    }
    let freqs = (0..n_freq).map(|k| k as f64 * fs / nfft as f64).collect();
    (freqs, psd)
}

/// DFT of any length through Bluestein's chirp-z algorithm on power-of-two FFTs.
struct Bluestein {
    n: usize,
    m: usize,
    w_re: Vec<f64>,
    w_im: Vec<f64>,
    bf_re: Vec<f64>,
    bf_im: Vec<f64>,
}

impl Bluestein {
    fn new(n: usize) -> Self {
        let m = (2 * n - 1).next_power_of_two();
        // chirp w_j = exp(-iπ j² / n); j² mod 2n keeps the angle exact for large j
        let (mut w_re, mut w_im) = (vec![0.0; n], vec![0.0; n]);
        for j in 0..n {
            let q = ((j as u128 * j as u128) % (2 * n as u128)) as f64;
            let ang = -PI * q / n as f64;
            w_re[j] = ang.cos();
            w_im[j] = ang.sin();
        }
        // b_j = conj(w_j) for |j| < n, wrapped
        let (mut bf_re, mut bf_im) = (vec![0.0; m], vec![0.0; m]);
        for j in 0..n {
            bf_re[j] = w_re[j];
            bf_im[j] = -w_im[j];
            if j > 0 {
                bf_re[m - j] = w_re[j];
                bf_im[m - j] = -w_im[j];
            }
        }
        fft_pow2(&mut bf_re, &mut bf_im);
        Self {
            n,
            m,
            w_re,
            w_im,
            bf_re,
            bf_im,
        }
    }

    fn run(&self, re: &mut [f64], im: &mut [f64]) {
        let (n, m) = (self.n, self.m);
        let (mut ar, mut ai) = (vec![0.0; m], vec![0.0; m]);
        for j in 0..n {
            ar[j] = re[j] * self.w_re[j] - im[j] * self.w_im[j];
            ai[j] = re[j] * self.w_im[j] + im[j] * self.w_re[j];
        }
        fft_pow2(&mut ar, &mut ai);
        for k in 0..m {
            let (xr, xi) = (ar[k], ai[k]);
            ar[k] = xr * self.bf_re[k] - xi * self.bf_im[k];
            ai[k] = xr * self.bf_im[k] + xi * self.bf_re[k];
        }
        // inverse FFT via conjugation
        for v in ai.iter_mut() {
            *v = -*v;
        }
        fft_pow2(&mut ar, &mut ai);
        let inv = 1.0 / m as f64;
        for k in 0..n {
            let (cr, ci) = (ar[k] * inv, -ai[k] * inv);
            re[k] = cr * self.w_re[k] - ci * self.w_im[k];
            im[k] = cr * self.w_im[k] + ci * self.w_re[k];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bluestein_matches_direct_dft() {
        for n in [5usize, 12, 30, 97] {
            let x: Vec<f64> = (0..n).map(|i| ((i * 37 % 11) as f64) - 4.0).collect();
            let (mut re, mut im) = (x.clone(), vec![0.0; n]);
            Bluestein::new(n).run(&mut re, &mut im);
            for k in 0..n {
                let (mut dr, mut di) = (0.0, 0.0);
                for (j, &v) in x.iter().enumerate() {
                    let a = -2.0 * PI * (j * k) as f64 / n as f64;
                    dr += v * a.cos();
                    di += v * a.sin();
                }
                assert!(
                    (re[k] - dr).abs() < 1e-9 && (im[k] - di).abs() < 1e-9,
                    "n {n} k {k}"
                );
            }
        }
    }

    #[test]
    fn quadratic_spline_interpolates_and_is_exact_for_parabolas() {
        let x: Vec<f64> = (0..12)
            .map(|i| i as f64 * 0.8 + (i % 3) as f64 * 0.1)
            .collect();
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v * v - 3.0 * v + 1.0).collect();
        let at_nodes = quadratic_interp_scipy(&x, &y, &x);
        for (a, b) in at_nodes.iter().zip(&y) {
            assert!((a - b).abs() < 1e-9);
        }
        let xs: Vec<f64> = (0..50)
            .map(|i| x[0] + i as f64 * (x[11] - x[0]) / 49.0)
            .collect();
        for (v, q) in xs.iter().zip(quadratic_interp_scipy(&x, &y, &xs)) {
            assert!((q - (2.0 * v * v - 3.0 * v + 1.0)).abs() < 1e-8, "{v}");
        }
    }
}
