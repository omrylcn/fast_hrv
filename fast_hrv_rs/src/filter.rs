//! IIR filter design and filtering (no external dependencies).
//!
//! - [`butter_sos`]: digital Butterworth design, the same way SciPy's
//!   `scipy.signal.butter(N, Wn, btype, output="sos", fs=fs)` does it: analog prototype poles →
//!   frequency transform (lp/hp/bp/bs) with pre-warped edges → bilinear transform → second-order
//!   sections with SciPy's `zpk2sos(pairing="nearest")` pairing and ordering. Coefficients match
//!   SciPy to ~1e-12 (see `tests/filter_design.rs`).
//! - [`iirnotch`]: second-order notch (`scipy.signal.iirnotch`).
//! - [`sosfilt`], [`sosfilt_zi`], [`sosfiltfilt`]: causal and zero-phase filtering with SciPy's
//!   defaults (`sosfiltfilt`: odd extension, `padlen = 3 * ntaps`, initial state from `sosfilt_zi`).
//! - [`filtfilt_ba`]: zero-phase filtering of a transfer function (b, a), SciPy `filtfilt(method="pad")`.
//! - [`SosFilter`]: causal cascade that keeps its state between calls (packet-by-packet streaming).
//!   Feeding a signal in packets of any size gives bit-identical output to one call on the whole signal.
//!
//! Everything is plain scalar `f64` code; nothing assumes AVX or any other CPU extension.

use crate::error::HrvError;
use std::f64::consts::PI;

// =============================================================================
// Second-order sections
// =============================================================================

/// One second-order section (biquad), normalised so that `a0 = 1`:
/// `H(z) = (b0 + b1 z⁻¹ + b2 z⁻²) / (1 + a1 z⁻¹ + a2 z⁻²)`.
/// Same layout as a SciPy SOS row `[b0, b1, b2, 1, a1, a2]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SosSection {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
}

impl SosSection {
    /// From a SciPy-style row `[b0, b1, b2, a0, a1, a2]` (normalised by `a0`).
    pub fn from_row(row: [f64; 6]) -> Self {
        let a0 = row[3];
        Self {
            b0: row[0] / a0,
            b1: row[1] / a0,
            b2: row[2] / a0,
            a1: row[4] / a0,
            a2: row[5] / a0,
        }
    }

    /// As a SciPy-style row `[b0, b1, b2, 1, a1, a2]`.
    pub fn to_row(&self) -> [f64; 6] {
        [self.b0, self.b1, self.b2, 1.0, self.a1, self.a2]
    }

    /// Filter `x` in place with Direct Form II transposed, starting from (and updating) `state`.
    /// Same arithmetic, in the same order, as SciPy's `sosfilt`.
    #[inline]
    pub fn process_inplace(&self, x: &mut [f64], state: &mut [f64; 2]) {
        let (b0, b1, b2, a1, a2) = (self.b0, self.b1, self.b2, self.a1, self.a2);
        let [mut z1, mut z2] = *state;
        for v in x.iter_mut() {
            let xi = *v;
            let y = b0 * xi + z1;
            z1 = b1 * xi - a1 * y + z2;
            z2 = b2 * xi - a2 * y;
            *v = y;
        }
        *state = [z1, z2];
    }

    /// Steady-state initial condition for a unit step (SciPy `lfilter_zi` for a 3-tap section).
    pub fn lfilter_zi(&self) -> [f64; 2] {
        // Solve (I - Aᵀ) zi = B with A the companion matrix of [1, a1, a2]:
        // [1 + a1, -1] [zi0]   [b1 - a1 b0]
        // [a2,      1] [zi1] = [b2 - a2 b0]
        let m00 = 1.0 + self.a1;
        let m10 = self.a2;
        let r0 = self.b1 - self.a1 * self.b0;
        let r1 = self.b2 - self.a2 * self.b0;
        let det = m00 + m10; // m00 * 1 - (-1) * m10
        if det == 0.0 || !det.is_finite() {
            // pole at z = 1 (integrator): no finite steady state
            return [0.0, 0.0];
        }
        let zi0 = (r0 + r1) / det;
        let zi1 = r1 - m10 * zi0;
        [zi0, zi1]
    }

    /// DC gain `H(1)`.
    pub fn dc_gain(&self) -> f64 {
        (self.b0 + self.b1 + self.b2) / (1.0 + self.a1 + self.a2)
    }
}

// =============================================================================
// Minimal complex arithmetic (NumPy semantics)
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq)]
struct C {
    re: f64,
    im: f64,
}

impl C {
    const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    const fn real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }
    fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
    fn is_real(self) -> bool {
        self.im == 0.0
    }
    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    fn add(self, o: C) -> C {
        C::new(self.re + o.re, self.im + o.im)
    }
    fn sub(self, o: C) -> C {
        C::new(self.re - o.re, self.im - o.im)
    }
    fn mul(self, o: C) -> C {
        C::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
    fn scale(self, s: f64) -> C {
        C::new(self.re * s, self.im * s)
    }
    /// Smith's algorithm, as used by NumPy's complex division.
    fn div(self, o: C) -> C {
        let (ar, ai, br, bi) = (self.re, self.im, o.re, o.im);
        if br.abs() >= bi.abs() {
            if br == 0.0 && bi == 0.0 {
                return C::new(ar / br.abs(), ai / br.abs());
            }
            let rat = bi / br;
            let scl = 1.0 / (br + bi * rat);
            C::new((ar + ai * rat) * scl, (ai - ar * rat) * scl)
        } else {
            let rat = br / bi;
            let scl = 1.0 / (bi + br * rat);
            C::new((ar * rat + ai) * scl, (ai * rat - ar) * scl)
        }
    }
    /// Principal square root (C99 `csqrt` algorithm).
    fn sqrt(self) -> C {
        let (a, b) = (self.re, self.im);
        if a == 0.0 && b == 0.0 {
            return C::new(0.0, b);
        }
        let t = ((a.abs() + a.hypot(b)) / 2.0).sqrt();
        if a >= 0.0 {
            C::new(t, b / (2.0 * t))
        } else {
            C::new(b.abs() / (2.0 * t), t.copysign(b))
        }
    }
}

fn cprod(v: &[C]) -> C {
    v.iter().fold(C::real(1.0), |acc, &x| acc.mul(x))
}

// =============================================================================
// Butterworth design (port of scipy.signal.butter → iirfilter → zpk2sos)
// =============================================================================

/// Filter band type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BandType {
    Lowpass,
    Highpass,
    Bandpass,
    Bandstop,
}

type Zpk = (Vec<C>, Vec<C>, f64);

fn buttap(n: usize) -> Zpk {
    let nf = n as f64;
    let p = (0..n)
        .map(|i| {
            let m = -(nf) + 1.0 + 2.0 * i as f64;
            let theta = PI * m / (2.0 * nf);
            C::new(-theta.cos(), -theta.sin())
        })
        .collect();
    (Vec::new(), p, 1.0)
}

fn lp2lp_zpk((z, p, k): Zpk, wo: f64) -> Zpk {
    let degree = p.len() as f64 - z.len() as f64;
    let z = z.iter().map(|&v| v.scale(wo)).collect();
    let p = p.iter().map(|&v| v.scale(wo)).collect();
    (z, p, k * wo.powf(degree))
}

fn lp2hp_zpk((z, p, k): Zpk, wo: f64) -> Zpk {
    let degree = p.len() - z.len();
    let negz: Vec<C> = z.iter().map(|v| v.scale(-1.0)).collect();
    let negp: Vec<C> = p.iter().map(|v| v.scale(-1.0)).collect();
    let k_hp = k * cprod(&negz).div(cprod(&negp)).re;
    let mut z_hp: Vec<C> = z.iter().map(|&v| C::real(wo).div(v)).collect();
    let p_hp: Vec<C> = p.iter().map(|&v| C::real(wo).div(v)).collect();
    z_hp.extend(std::iter::repeat_n(C::real(0.0), degree));
    (z_hp, p_hp, k_hp)
}

fn lp2bp_zpk((z, p, k): Zpk, wo: f64, bw: f64) -> Zpk {
    let degree = p.len() - z.len();
    let wo2 = C::real(wo * wo);
    let tr = |v: &[C]| -> Vec<C> {
        let lp: Vec<C> = v.iter().map(|&x| x.scale(bw).scale(0.5)).collect();
        let roots: Vec<C> = lp.iter().map(|&x| x.mul(x).sub(wo2).sqrt()).collect();
        let mut out: Vec<C> = lp.iter().zip(&roots).map(|(&a, &r)| a.add(r)).collect();
        out.extend(lp.iter().zip(&roots).map(|(&a, &r)| a.sub(r)));
        out
    };
    let mut z_bp = tr(&z);
    let p_bp = tr(&p);
    z_bp.extend(std::iter::repeat_n(C::real(0.0), degree));
    (z_bp, p_bp, k * bw.powf(degree as f64))
}

fn lp2bs_zpk((z, p, k): Zpk, wo: f64, bw: f64) -> Zpk {
    let degree = p.len() - z.len();
    let wo2 = C::real(wo * wo);
    let half = C::real(bw / 2.0);
    let tr = |v: &[C]| -> Vec<C> {
        let hp: Vec<C> = v.iter().map(|&x| half.div(x)).collect();
        let roots: Vec<C> = hp.iter().map(|&x| x.mul(x).sub(wo2).sqrt()).collect();
        let mut out: Vec<C> = hp.iter().zip(&roots).map(|(&a, &r)| a.add(r)).collect();
        out.extend(hp.iter().zip(&roots).map(|(&a, &r)| a.sub(r)));
        out
    };
    let mut z_bs = tr(&z);
    let p_bs = tr(&p);
    z_bs.extend(std::iter::repeat_n(C::new(0.0, wo), degree));
    z_bs.extend(std::iter::repeat_n(C::new(0.0, -wo), degree));
    let negz: Vec<C> = z.iter().map(|v| v.scale(-1.0)).collect();
    let negp: Vec<C> = p.iter().map(|v| v.scale(-1.0)).collect();
    let k_bs = k * cprod(&negz).div(cprod(&negp)).re;
    (z_bs, p_bs, k_bs)
}

fn bilinear_zpk((z, p, k): Zpk, fs: f64) -> Zpk {
    let degree = p.len() - z.len();
    let fs2 = C::real(2.0 * fs);
    let mut z_z: Vec<C> = z.iter().map(|&v| fs2.add(v).div(fs2.sub(v))).collect();
    let p_z: Vec<C> = p.iter().map(|&v| fs2.add(v).div(fs2.sub(v))).collect();
    z_z.extend(std::iter::repeat_n(C::real(-1.0), degree));
    let num: Vec<C> = z.iter().map(|&v| fs2.sub(v)).collect();
    let den: Vec<C> = p.iter().map(|&v| fs2.sub(v)).collect();
    let k_z = k * cprod(&num).div(cprod(&den)).re;
    (z_z, p_z, k_z)
}

/// SciPy `_cplxreal`: split into complex values with positive imaginary part (one per conjugate
/// pair) and real values, both sorted the way SciPy sorts them.
fn cplxreal(z: &[C]) -> (Vec<C>, Vec<f64>) {
    if z.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let tol = 100.0 * f64::EPSILON;
    let mut z = z.to_vec();
    // lexsort((abs(imag), real)): primary key real, secondary |imag|; stable
    z.sort_by(|a, b| {
        a.re.total_cmp(&b.re)
            .then(a.im.abs().total_cmp(&b.im.abs()))
    });
    let is_real = |v: &C| v.im.abs() <= tol * v.abs();
    let zr: Vec<f64> = z.iter().filter(|v| is_real(v)).map(|v| v.re).collect();
    if zr.len() == z.len() {
        return (Vec::new(), zr);
    }
    let zc_all: Vec<C> = z.iter().filter(|v| !is_real(v)).copied().collect();
    let mut zp: Vec<C> = zc_all.iter().filter(|v| v.im > 0.0).copied().collect();
    let mut zn: Vec<C> = zc_all.iter().filter(|v| v.im < 0.0).copied().collect();
    if zp.len() != zn.len() {
        // unmatched conjugate: cannot happen for the filters designed here
        return (zp, zr);
    }
    // runs of (nearly) equal real parts are re-sorted by |imag|
    let mut i = 0;
    while i < zp.len() {
        let mut j = i;
        while j + 1 < zp.len() && (zp[j + 1].re - zp[j].re) <= tol * zp[j].abs() {
            j += 1;
        }
        if j > i {
            zp[i..=j].sort_by(|a, b| a.im.abs().total_cmp(&b.im.abs()));
            zn[i..=j].sort_by(|a, b| a.im.abs().total_cmp(&b.im.abs()));
        }
        i = j + 1;
    }
    let zc = zp
        .iter()
        .zip(&zn)
        .map(|(&a, &b)| a.add(b.conj()).scale(0.5))
        .collect();
    (zc, zr)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Which {
    Real,
    Complex,
    Any,
}

fn nearest_idx(fro: &[C], to: C, which: Which) -> usize {
    let mut order: Vec<usize> = (0..fro.len()).collect();
    order.sort_by(|&a, &b| fro[a].sub(to).abs().total_cmp(&fro[b].sub(to).abs()));
    match which {
        Which::Any => order[0],
        Which::Real => *order
            .iter()
            .find(|&&i| fro[i].is_real())
            .unwrap_or(&order[0]),
        Which::Complex => *order
            .iter()
            .find(|&&i| !fro[i].is_real())
            .unwrap_or(&order[0]),
    }
}

/// `np.poly` for up to two roots, returning real coefficients (roots are real or a conjugate pair).
fn poly2(roots: &[C]) -> Vec<f64> {
    let mut a = vec![C::real(1.0)];
    for &r in roots {
        let mut next = vec![C::real(0.0); a.len() + 1];
        for (i, &ai) in a.iter().enumerate() {
            next[i] = next[i].add(ai);
            next[i + 1] = next[i + 1].add(ai.mul(r.scale(-1.0)));
        }
        a = next;
    }
    a.iter().map(|c| c.re).collect()
}

fn single_zpksos(z: &[C], p: &[C]) -> [f64; 6] {
    let mut sos = [0.0; 6];
    let b = poly2(z);
    let a = poly2(p);
    sos[3 - b.len()..3].copy_from_slice(&b);
    sos[6 - a.len()..6].copy_from_slice(&a);
    sos
}

fn idx_worst(p: &[C]) -> usize {
    let mut best = 0;
    let mut best_v = f64::INFINITY;
    for (i, v) in p.iter().enumerate() {
        let d = (1.0 - v.abs()).abs();
        if d < best_v {
            best_v = d;
            best = i;
        }
    }
    best
}

/// Port of SciPy `zpk2sos(z, p, k, pairing="nearest")` for digital filters.
fn zpk2sos((z, p, k): Zpk) -> Vec<[f64; 6]> {
    let (mut z, mut p) = (z, p);
    if z.is_empty() && p.is_empty() {
        return vec![[k, 0.0, 0.0, 1.0, 0.0, 0.0]];
    }
    let (lz, lp) = (z.len(), p.len());
    p.extend(std::iter::repeat_n(C::real(0.0), lz.saturating_sub(lp)));
    z.extend(std::iter::repeat_n(C::real(0.0), lp.saturating_sub(lz)));
    let n_sections = p.len().max(z.len()).div_ceil(2);
    if p.len() % 2 == 1 {
        p.push(C::real(0.0));
        z.push(C::real(0.0));
    }
    let to_vec = |(c, r): (Vec<C>, Vec<f64>)| -> Vec<C> {
        let mut v = c;
        v.extend(r.into_iter().map(C::real));
        v
    };
    let mut z = to_vec(cplxreal(&z));
    let mut p = to_vec(cplxreal(&p));

    let mut sos = vec![[0.0; 6]; n_sections];
    for si in (0..n_sections).rev() {
        let p1 = p.remove(idx_worst(&p));
        let n_real_p = p.iter().filter(|v| v.is_real()).count();
        if p1.is_real() && n_real_p == 0 {
            let z1 = z.remove(nearest_idx(&z, p1, Which::Real));
            sos[si] = single_zpksos(&[z1, C::real(0.0)], &[p1, C::real(0.0)]);
        } else if p.len() + 1 == z.len()
            && !p1.is_real()
            && n_real_p == 1
            && z.iter().filter(|v| v.is_real()).count() == 1
        {
            let z1 = z.remove(nearest_idx(&z, p1, Which::Complex));
            sos[si] = single_zpksos(&[z1, z1.conj()], &[p1, p1.conj()]);
        } else {
            let p2 = if p1.is_real() {
                let real_idx: Vec<usize> = (0..p.len()).filter(|&i| p[i].is_real()).collect();
                let reals: Vec<C> = real_idx.iter().map(|&i| p[i]).collect();
                p.remove(real_idx[idx_worst(&reals)])
            } else {
                p1.conj()
            };
            if !z.is_empty() {
                let z1 = z.remove(nearest_idx(&z, p1, Which::Any));
                if !z1.is_real() {
                    sos[si] = single_zpksos(&[z1, z1.conj()], &[p1, p2]);
                } else if !z.is_empty() {
                    let z2 = z.remove(nearest_idx(&z, p1, Which::Real));
                    sos[si] = single_zpksos(&[z1, z2], &[p1, p2]);
                } else {
                    sos[si] = single_zpksos(&[z1], &[p1, p2]);
                }
            } else {
                sos[si] = single_zpksos(&[], &[p1, p2]);
            }
        }
    }
    for c in sos[0][..3].iter_mut() {
        *c *= k;
    }
    sos
}

fn check_fs(fs: f64) -> Result<(), HrvError> {
    if fs.is_finite() && fs > 0.0 {
        Ok(())
    } else {
        Err(HrvError::InvalidSamplingRate(fs))
    }
}

/// Digital Butterworth filter as second-order sections.
///
/// Equivalent to `scipy.signal.butter(order, cutoff, btype, output="sos", fs=fs)`. `cutoff` holds one
/// frequency (Hz) for low/high-pass and two (low, high) for band-pass/band-stop. For band filters the
/// resulting filter has `2 * order` poles and `order` sections (SciPy semantics).
pub fn butter_sos(
    order: usize,
    cutoff: &[f64],
    btype: BandType,
    fs: f64,
) -> Result<Vec<SosSection>, HrvError> {
    check_fs(fs)?;
    if order == 0 {
        return Err(HrvError::InvalidOrder(order));
    }
    let nyq = fs / 2.0;
    let needed = match btype {
        BandType::Lowpass | BandType::Highpass => 1,
        BandType::Bandpass | BandType::Bandstop => 2,
    };
    if cutoff.len() != needed {
        return Err(HrvError::InvalidFrequency {
            freq: f64::NAN,
            nyquist: nyq,
        });
    }
    for &f in cutoff {
        if !(f > 0.0 && f < nyq) {
            return Err(HrvError::InvalidFrequency {
                freq: f,
                nyquist: nyq,
            });
        }
    }
    if needed == 2 && cutoff[0] >= cutoff[1] {
        return Err(HrvError::InvalidFrequency {
            freq: cutoff[0],
            nyquist: nyq,
        });
    }
    // SciPy: Wn normalised to Nyquist, then pre-warped with fs = 2
    let warp = |f: f64| {
        let wn = 2.0 * f / fs;
        2.0 * 2.0 * (PI * wn / 2.0).tan()
    };
    let proto = buttap(order);
    let zpk = match btype {
        BandType::Lowpass => lp2lp_zpk(proto, warp(cutoff[0])),
        BandType::Highpass => lp2hp_zpk(proto, warp(cutoff[0])),
        BandType::Bandpass | BandType::Bandstop => {
            let (w0, w1) = (warp(cutoff[0]), warp(cutoff[1]));
            let bw = w1 - w0;
            let wo = (w0 * w1).sqrt();
            if btype == BandType::Bandpass {
                lp2bp_zpk(proto, wo, bw)
            } else {
                lp2bs_zpk(proto, wo, bw)
            }
        }
    };
    let zpk = bilinear_zpk(zpk, 2.0);
    Ok(zpk2sos(zpk).into_iter().map(SosSection::from_row).collect())
}

/// Second-order IIR notch at `f0` Hz with quality factor `q` (`scipy.signal.iirnotch(f0, q, fs)`).
pub fn iirnotch(f0: f64, q: f64, fs: f64) -> Result<SosSection, HrvError> {
    check_fs(fs)?;
    let nyq = fs / 2.0;
    if !(f0 > 0.0 && f0 < nyq) {
        return Err(HrvError::InvalidFrequency {
            freq: f0,
            nyquist: nyq,
        });
    }
    if !(q.is_finite() && q > 0.0) {
        return Err(HrvError::InvalidQ(q));
    }
    let w0 = 2.0 * f0 / fs;
    let bw = w0 / q * PI;
    let w0 = w0 * PI;
    let beta = (bw / 2.0).tan();
    let gain = 1.0 / (1.0 + beta);
    Ok(SosSection {
        b0: gain,
        b1: gain * (-2.0 * w0.cos()),
        b2: gain,
        a1: -2.0 * gain * w0.cos(),
        a2: 2.0 * gain - 1.0,
    })
}

// =============================================================================
// Filtering
// =============================================================================

/// Initial state of a cascade for a step of unit height (SciPy `sosfilt_zi`).
pub fn sosfilt_zi(sos: &[SosSection]) -> Vec<[f64; 2]> {
    let mut scale = 1.0;
    sos.iter()
        .map(|s| {
            let zi = s.lfilter_zi();
            let out = [scale * zi[0], scale * zi[1]];
            scale *= s.dc_gain();
            out
        })
        .collect()
}

/// Run the cascade sample by sample (all sections per sample, like SciPy's `sosfilt`). Each section
/// performs exactly the same operations as [`SosSection::process_inplace`], so the result does not
/// depend on how the input is split into packets; the sample-major order lets the CPU overlap the
/// sections' recursions.
fn cascade_inplace(sos: &[SosSection], x: &mut [f64], state: &mut [[f64; 2]]) {
    match sos.len() {
        0 => {}
        1 => sos[0].process_inplace(x, &mut state[0]),
        // fixed sizes keep coefficients and state in registers (same arithmetic, same order)
        2 => cascade_fixed::<2>(sos, x, state),
        3 => cascade_fixed::<3>(sos, x, state),
        4 => cascade_fixed::<4>(sos, x, state),
        _ => {
            for v in x.iter_mut() {
                let mut xi = *v;
                for (s, st) in sos.iter().zip(state.iter_mut()) {
                    let y = s.b0 * xi + st[0];
                    st[0] = s.b1 * xi - s.a1 * y + st[1];
                    st[1] = s.b2 * xi - s.a2 * y;
                    xi = y;
                }
                *v = xi;
            }
        }
    }
}

fn cascade_fixed<const N: usize>(sos: &[SosSection], x: &mut [f64], state: &mut [[f64; 2]]) {
    cascade_fixed_iter::<N, _>(sos, x.iter_mut(), state);
}

fn cascade_fixed_iter<'a, const N: usize, I: Iterator<Item = &'a mut f64>>(
    sos: &[SosSection],
    x: I,
    state: &mut [[f64; 2]],
) {
    let c: [SosSection; N] = std::array::from_fn(|i| sos[i]);
    let mut st: [[f64; 2]; N] = std::array::from_fn(|i| state[i]);
    for v in x {
        let mut xi = *v;
        for k in 0..N {
            let y = c[k].b0 * xi + st[k][0];
            st[k][0] = c[k].b1 * xi - c[k].a1 * y + st[k][1];
            st[k][1] = c[k].b2 * xi - c[k].a2 * y;
            xi = y;
        }
        *v = xi;
    }
    state[..N].copy_from_slice(&st);
}

/// The cascade run from the last sample to the first (the backward pass of a forward-backward filter,
/// without reversing the buffer).
fn cascade_inplace_rev(sos: &[SosSection], x: &mut [f64], state: &mut [[f64; 2]]) {
    match sos.len() {
        0 => {}
        2 => cascade_fixed_iter::<2, _>(sos, x.iter_mut().rev(), state),
        3 => cascade_fixed_iter::<3, _>(sos, x.iter_mut().rev(), state),
        4 => cascade_fixed_iter::<4, _>(sos, x.iter_mut().rev(), state),
        _ => {
            x.reverse();
            cascade_inplace(sos, x, state);
            x.reverse();
        }
    }
}

/// Causal filtering through the cascade. `zi` (one `[z1, z2]` per section) is the initial state
/// (zero if `None`). Returns the output and the final state.
pub fn sosfilt(
    sos: &[SosSection],
    x: &[f64],
    zi: Option<&[[f64; 2]]>,
) -> (Vec<f64>, Vec<[f64; 2]>) {
    let mut y = x.to_vec();
    let mut state: Vec<[f64; 2]> = match zi {
        Some(z) => z.to_vec(),
        None => vec![[0.0; 2]; sos.len()],
    };
    cascade_inplace(sos, &mut y, &mut state);
    (y, state)
}

/// Odd extension used by SciPy's filtfilt padding (`padtype="odd"`).
fn odd_ext(x: &[f64], edge: usize) -> Vec<f64> {
    let n = x.len();
    let mut ext = Vec::with_capacity(n + 2 * edge);
    let (first, last) = (x[0], x[n - 1]);
    for i in (1..=edge).rev() {
        ext.push(2.0 * first - x[i]);
    }
    ext.extend_from_slice(x);
    for i in 1..=edge {
        ext.push(2.0 * last - x[n - 1 - i]);
    }
    ext
}

/// Zero-phase forward-backward filtering, equivalent to `scipy.signal.sosfiltfilt(sos, x)` with its
/// defaults: odd extension of `padlen = 3 * ntaps` samples on each side
/// (`ntaps = 2 * n_sections + 1`, minus the number of first-order sections), and initial states from
/// [`sosfilt_zi`] scaled by the first sample of each pass.
///
/// SciPy raises an error when `x.len() <= padlen`; here the padding is shortened to `x.len() - 1`
/// instead. Inputs with fewer than 2 samples are returned unchanged.
pub fn sosfiltfilt(sos: &[SosSection], x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n < 2 || sos.is_empty() {
        return x.to_vec();
    }
    let first_order = sos
        .iter()
        .filter(|s| s.b2 == 0.0)
        .count()
        .min(sos.iter().filter(|s| s.a2 == 0.0).count());
    let ntaps = 2 * sos.len() + 1 - first_order;
    let edge = (3 * ntaps).min(n - 1);
    let zi = sosfilt_zi(sos);
    let scaled = |v: f64| -> Vec<[f64; 2]> { zi.iter().map(|z| [z[0] * v, z[1] * v]).collect() };
    // in place: one buffer for both passes (same arithmetic as two `sosfilt` calls)
    let mut y = odd_ext(x, edge);
    let mut st = scaled(y[0]);
    cascade_inplace(sos, &mut y, &mut st);
    let mut st = scaled(y[y.len() - 1]);
    cascade_inplace_rev(sos, &mut y, &mut st);
    y.truncate(edge + n);
    y.drain(..edge);
    y
}

/// SciPy `lfilter_zi(b, a)` for a general transfer function.
fn lfilter_zi_ba(b: &[f64], a: &[f64]) -> Vec<f64> {
    let a0 = a[0];
    let n = b.len().max(a.len());
    let mut bn = vec![0.0; n];
    let mut an = vec![0.0; n];
    for (i, &v) in b.iter().enumerate() {
        bn[i] = v / a0;
    }
    for (i, &v) in a.iter().enumerate() {
        an[i] = v / a0;
    }
    if n < 2 {
        return Vec::new();
    }
    let m = n - 1;
    // (I - companion(a)ᵀ) zi = b[1:] - a[1:] * b[0]
    // companion(a)ᵀ: first column -a[1:], superdiagonal ones
    let mut mat = vec![vec![0.0; m]; m];
    for (i, row) in mat.iter_mut().enumerate() {
        row[i] += 1.0;
        row[0] += an[i + 1];
        if i + 1 < m {
            row[i + 1] -= 1.0;
        }
    }
    let mut rhs: Vec<f64> = (0..m).map(|i| bn[i + 1] - an[i + 1] * bn[0]).collect();
    // Gaussian elimination with partial pivoting
    for col in 0..m {
        let piv = (col..m)
            .max_by(|&i, &j| mat[i][col].abs().total_cmp(&mat[j][col].abs()))
            .unwrap_or(col);
        mat.swap(col, piv);
        rhs.swap(col, piv);
        let d = mat[col][col];
        if d == 0.0 {
            return vec![0.0; m];
        }
        for r in col + 1..m {
            let f = mat[r][col] / d;
            if f != 0.0 {
                for c in col..m {
                    mat[r][c] -= f * mat[col][c];
                }
                rhs[r] -= f * rhs[col];
            }
        }
    }
    let mut zi = vec![0.0; m];
    for r in (0..m).rev() {
        let mut s = rhs[r];
        for c in r + 1..m {
            s -= mat[r][c] * zi[c];
        }
        zi[r] = s / mat[r][r];
    }
    zi
}

/// `scipy.signal.lfilter(b, a, x, zi=zi)` (Direct Form II transposed, coefficients normalised by a[0]).
fn lfilter_ba(b: &[f64], a: &[f64], x: &[f64], zi: &[f64]) -> Vec<f64> {
    let a0 = a[0];
    let n = b.len().max(a.len());
    let mut bn = vec![0.0; n];
    let mut an = vec![0.0; n];
    for (i, &v) in b.iter().enumerate() {
        bn[i] = v / a0;
    }
    for (i, &v) in a.iter().enumerate() {
        an[i] = v / a0;
    }
    let mut z: Vec<f64> = zi.to_vec();
    z.resize(n.saturating_sub(1), 0.0);
    if a.len() == 1 {
        // FIR: y = conv(b, x)[..len] + zi (what SciPy's lfilter does for len(a) == 1);
        // same summation order (k = 0, 1, ...) as before, without per-sample bounds checks
        let nb = bn.len();
        let mut y = Vec::with_capacity(x.len());
        for i in 0..x.len().min(nb.saturating_sub(1)) {
            let mut acc = 0.0;
            for (k, &bk) in bn.iter().enumerate().take(i + 1) {
                acc += bk * x[i - k];
            }
            y.push(acc + z.get(i).copied().unwrap_or(0.0));
        }
        if x.len() >= nb {
            // four outputs at a time: independent sums run in parallel, each in the same order
            let m = x.len() - nb + 1; // outputs from index nb - 1 on
            let mut j = 0;
            while j + 4 <= m {
                let (mut a0, mut a1, mut a2, mut a3) = (0.0, 0.0, 0.0, 0.0);
                for (k, &bk) in bn.iter().enumerate() {
                    let base = j + nb - 1 - k;
                    let w = &x[base..base + 4];
                    a0 += bk * w[0];
                    a1 += bk * w[1];
                    a2 += bk * w[2];
                    a3 += bk * w[3];
                }
                y.extend_from_slice(&[a0 + 0.0, a1 + 0.0, a2 + 0.0, a3 + 0.0]);
                j += 4;
            }
            for w in x[j..].windows(nb) {
                let mut acc = 0.0;
                for (&bk, &xv) in bn.iter().zip(w.iter().rev()) {
                    acc += bk * xv;
                }
                y.push(acc + 0.0);
            }
        }
        return y;
    }
    let mut y = Vec::with_capacity(x.len());
    for &xi in x {
        let yi = bn[0] * xi + z.first().copied().unwrap_or(0.0);
        for k in 0..n.saturating_sub(1) {
            let next = if k + 1 < n - 1 { z[k + 1] } else { 0.0 };
            z[k] = bn[k + 1] * xi - an[k + 1] * yi + next;
        }
        y.push(yi);
    }
    y
}

/// Zero-phase filtering of a transfer function (b, a), equivalent to
/// `scipy.signal.filtfilt(b, a, x, method="pad")` with its defaults (odd extension,
/// `padlen = 3 * max(len(a), len(b))`). The padding is shortened when the input is too short.
pub fn filtfilt_ba(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n < 2 || b.is_empty() || a.is_empty() || a[0] == 0.0 {
        return x.to_vec();
    }
    if a.len() == 1 && b.len() > 1 && b.iter().all(|&v| v == b[0]) {
        return filtfilt_moving_average(b.len(), b[0] / a[0], x);
    }
    let edge = (3 * b.len().max(a.len())).min(n - 1);
    let ext = odd_ext(x, edge);
    let zi = lfilter_zi_ba(b, a);
    let scaled = |v: f64| -> Vec<f64> { zi.iter().map(|z| z * v).collect() };
    let mut y = lfilter_ba(b, a, &ext, &scaled(ext[0]));
    y.reverse();
    let y0 = y[0];
    let mut y = lfilter_ba(b, a, &y, &scaled(y0));
    y.reverse();
    y[edge..edge + n].to_vec()
}

/// `filtfilt_ba` for a moving average (all taps `c`, `a = [1]`), as NeuroKit2's powerline filter: the
/// same odd padding and initial state as SciPy, with a running window sum (exact sum recomputed every
/// 1024 samples, so the difference to the direct sum stays at rounding level, ~1e-15 relative).
fn filtfilt_moving_average(taps: usize, c: f64, x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let edge = (3 * taps).min(n - 1);
    let mut y = odd_ext(x, edge);
    // SciPy lfilter_zi for b = c·ones, a = [1]: zi[k] = c · (taps − 1 − k)
    let pass = |v: &mut Vec<f64>| {
        let x0 = v[0];
        let src = v.clone();
        let mut sum = 0.0;
        for i in 0..src.len() {
            sum += src[i];
            if i >= taps {
                sum -= src[i - taps];
            }
            if i % 1024 == 0 && i >= taps {
                sum = src[i + 1 - taps..=i].iter().sum();
            }
            // first taps − 1 outputs: partial window + initial state (steady state for x0)
            v[i] = if i + 1 < taps {
                c * sum + c * (taps - 1 - i) as f64 * x0
            } else {
                c * sum
            };
        }
    };
    pass(&mut y);
    y.reverse();
    pass(&mut y);
    y.reverse();
    y.truncate(edge + n);
    y.drain(..edge);
    y
}

// =============================================================================
// Stateful causal filter (streaming)
// =============================================================================

/// How a [`SosFilter`] initialises its state on the first sample it sees.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum InitialState {
    /// All-zero state (a DC offset in the input produces a start-up transient).
    Zero,
    /// Steady state for a constant input equal to the first sample (`sosfilt_zi * x[0]`),
    /// which removes the start-up transient caused by a DC offset.
    #[default]
    SteadyState,
}

/// Causal SOS cascade that carries its state across calls.
///
/// ```
/// use fast_hrv::filter::{butter_sos, BandType, SosFilter, InitialState};
/// let sos = butter_sos(2, &[0.5, 40.0], BandType::Bandpass, 512.0).unwrap();
/// let x: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.05).sin() + 400.0).collect();
/// let mut whole = SosFilter::new(sos.clone(), InitialState::SteadyState);
/// let y_whole = whole.process(&x);
/// let mut live = SosFilter::new(sos, InitialState::SteadyState);
/// let y_live: Vec<f64> = x.chunks(32).flat_map(|p| live.process(p)).collect();
/// assert_eq!(y_whole, y_live); // bit-identical
/// ```
#[derive(Clone, Debug)]
pub struct SosFilter {
    sos: Vec<SosSection>,
    zi_unit: Vec<[f64; 2]>,
    state: Vec<[f64; 2]>,
    init: InitialState,
    started: bool,
}

impl SosFilter {
    pub fn new(sos: Vec<SosSection>, init: InitialState) -> Self {
        let zi_unit = sosfilt_zi(&sos);
        let state = vec![[0.0; 2]; sos.len()];
        Self {
            sos,
            zi_unit,
            state,
            init,
            started: false,
        }
    }

    /// The sections of the cascade.
    pub fn sections(&self) -> &[SosSection] {
        &self.sos
    }

    /// Current state, one `[z1, z2]` per section (e.g. to persist a session).
    pub fn state(&self) -> &[[f64; 2]] {
        &self.state
    }

    /// Forget the state; the next sample is treated as the first one (use after a gap).
    pub fn reset(&mut self) {
        self.state.iter_mut().for_each(|s| *s = [0.0; 2]);
        self.started = false;
    }

    fn start(&mut self, x0: f64) {
        if self.init == InitialState::SteadyState && x0.is_finite() {
            for (s, z) in self.state.iter_mut().zip(&self.zi_unit) {
                *s = [z[0] * x0, z[1] * x0];
            }
        }
        self.started = true;
    }

    /// Filter one packet in place.
    pub fn process_inplace(&mut self, x: &mut [f64]) {
        if x.is_empty() {
            return;
        }
        if !self.started {
            self.start(x[0]);
        }
        cascade_inplace(&self.sos, x, &mut self.state);
    }

    /// Filter one packet and return the output.
    pub fn process(&mut self, x: &[f64]) -> Vec<f64> {
        let mut y = x.to_vec();
        self.process_inplace(&mut y);
        y
    }

    /// Filter one packet into `out` (`out.len()` must equal `x.len()`).
    pub fn process_into(&mut self, x: &[f64], out: &mut [f64]) {
        out.copy_from_slice(x);
        self.process_inplace(out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowpass_order2_matches_closed_form() {
        // scipy.signal.butter(2, 10, fs=100, output="sos")
        let sos = butter_sos(2, &[10.0], BandType::Lowpass, 100.0).unwrap();
        let expected = [
            0.06745527388907191,
            0.13491054777814382,
            0.06745527388907191,
            -1.1429805025399011,
            0.41280159809618866,
        ];
        let got = [sos[0].b0, sos[0].b1, sos[0].b2, sos[0].a1, sos[0].a2];
        for (g, e) in got.iter().zip(&expected) {
            assert!((g - e).abs() < 1e-12, "{g} vs {e}");
        }
    }

    #[test]
    fn invalid_parameters_are_errors() {
        assert!(butter_sos(0, &[1.0], BandType::Lowpass, 100.0).is_err());
        assert!(butter_sos(2, &[60.0], BandType::Lowpass, 100.0).is_err());
        assert!(butter_sos(2, &[10.0, 5.0], BandType::Bandpass, 100.0).is_err());
        assert!(butter_sos(2, &[10.0], BandType::Lowpass, 0.0).is_err());
        assert!(iirnotch(50.0, 0.0, 512.0).is_err());
    }

    #[test]
    fn highpass_has_zero_dc_gain_and_unit_nyquist_gain() {
        let sos = butter_sos(5, &[0.5], BandType::Highpass, 512.0).unwrap();
        assert_eq!(sos.len(), 3);
        let dc: f64 = sos.iter().map(|s| s.dc_gain()).product();
        assert!(dc.abs() < 1e-12);
        let ny: f64 = sos
            .iter()
            .map(|s| (s.b0 - s.b1 + s.b2) / (1.0 - s.a1 + s.a2))
            .product();
        assert!((ny - 1.0).abs() < 1e-9);
    }

    #[test]
    fn sosfiltfilt_keeps_constant_signal() {
        let sos = butter_sos(4, &[40.0], BandType::Lowpass, 512.0).unwrap();
        let y = sosfiltfilt(&sos, &[3.0; 200]);
        for v in y {
            assert!((v - 3.0).abs() < 1e-9);
        }
    }

    #[test]
    fn streaming_is_bit_identical_for_any_packet_size() {
        let mut sos = butter_sos(2, &[0.5], BandType::Highpass, 512.0).unwrap();
        sos.extend(butter_sos(4, &[40.0], BandType::Lowpass, 512.0).unwrap());
        sos.push(iirnotch(50.0, 30.0, 512.0).unwrap());
        let x: Vec<f64> = (0..5000)
            .map(|i| 412.0 + (i as f64 * 0.037).sin() * 50.0 + ((i * 7919) % 13) as f64)
            .collect();
        let mut f = SosFilter::new(sos.clone(), InitialState::SteadyState);
        let whole = f.process(&x);
        for packet in [1usize, 7, 32, 100, 4999] {
            let mut g = SosFilter::new(sos.clone(), InitialState::SteadyState);
            let live: Vec<f64> = x.chunks(packet).flat_map(|p| g.process(p)).collect();
            assert_eq!(whole, live, "packet size {packet}");
        }
    }
}
