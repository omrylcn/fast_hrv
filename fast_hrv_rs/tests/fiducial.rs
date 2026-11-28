//! `Fiducial::{Centroid, Adaptive}`: the windowed batch functions and `StreamDetector` must give the
//! same peaks bit for bit (the adaptive memory resets at gaps in both); the adaptive point must
//! leave narrow complexes on the extremum and stabilise a wide complex with two near-equal troughs
//! (the bundle-branch-block shape of MIT-BIH 118). Synthetic signals only; the MIT-BIH numbers are
//! in `tools/mitdb_timing.py`.

use fast_hrv::detect::{
    DualConfig, Fiducial, RefineConfig, WindowConfig, dual_detect_windowed,
    elgendi_refined_windowed,
};
use fast_hrv::ecg::{EcgFilterConfig, ecg_clean_causal};
use fast_hrv::stream::{StreamConfig, StreamDetector, StreamPeaks};

const FS: f64 = 512.0;

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let (u1, u2) = (self.uniform().max(1e-300), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Raw-like ECG (ADC counts) and the true beat times (samples). `wide`: small R and two deep S
/// troughs 36 ms apart whose relative depth changes from beat to beat; otherwise a narrow QRS.
fn synth(seconds: f64, seed: u64, wide: bool) -> (Vec<f64>, Vec<f64>) {
    let n = (seconds * FS) as usize;
    let mut rng = Rng(seed | 1);
    let mut x = vec![0.0; n];
    let mut beats = Vec::new();
    let mut t = 0.5;
    while t < seconds - 0.6 {
        beats.push(t * FS);
        let waves: Vec<(f64, f64, f64)> = if wide {
            let d = 0.06 * rng.gauss();
            vec![
                (0.12, -0.20, 0.025),
                (0.35, 0.0, 0.008),
                (-0.90 * (1.0 + d), 0.030, 0.009),
                (-0.90 * (1.0 - d), 0.066, 0.009),
                (0.30, 0.30, 0.045),
            ]
        } else {
            vec![
                (0.12, -0.20, 0.025),
                (-0.10, -0.035, 0.008),
                (1.00, 0.0, 0.009),
                (-0.25, 0.035, 0.008),
                (0.30, 0.28, 0.045),
            ]
        };
        let lo = ((t - 0.5) * FS).max(0.0) as usize;
        let hi = (((t + 0.6) * FS) as usize).min(n);
        for (i, xi) in x.iter_mut().enumerate().take(hi).skip(lo) {
            let ti = i as f64 / FS - t;
            for &(a, off, s) in &waves {
                let z = (ti - off) / s;
                *xi += a * (-0.5 * z * z).exp();
            }
        }
        t += 0.8 + 0.04 * (2.0 * std::f64::consts::PI * 0.25 * t).sin() + 0.015 * rng.gauss();
    }
    for (i, xi) in x.iter_mut().enumerate() {
        let ti = i as f64 / FS;
        *xi = 420.0
            + 1000.0
                * (*xi + 0.1 * (2.0 * std::f64::consts::PI * 0.3 * ti).sin() + 0.01 * rng.gauss());
    }
    (x, beats)
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn dual_with(fiducial: Fiducial) -> DualConfig {
    DualConfig {
        refine: RefineConfig {
            fiducial,
            ..RefineConfig::default()
        },
        ..DualConfig::default()
    }
}

fn stream(
    clean: &[f64],
    start: u64,
    dual: DualConfig,
    packet: usize,
    gaps: &[usize],
) -> StreamPeaks {
    let cfg = StreamConfig {
        dual,
        ..StreamConfig::default()
    };
    let mut det = StreamDetector::new(FS, start, cfg).unwrap();
    let mut out = StreamPeaks::default();
    let mut at = 0;
    while at < clean.len() {
        if gaps.contains(&at) {
            det.reset_at(start + at as u64, &mut out).unwrap();
        }
        let next_gap = gaps
            .iter()
            .copied()
            .filter(|&g| g > at)
            .min()
            .unwrap_or(clean.len());
        let s = packet.min(next_gap - at);
        det.push(&clean[at..at + s], &mut out).unwrap();
        at += s;
    }
    det.flush(&mut out).unwrap();
    out
}

/// RMSSD of the detected intervals relative to the true ones (each beat matched to the nearest
/// peak within 150 ms; the causal filter delays every peak by a constant ~11 ms).
fn rmssd_error(peaks: &[f64], beats: &[f64]) -> f64 {
    let matched: Vec<(f64, f64)> = beats
        .iter()
        .filter_map(|&b| {
            peaks
                .iter()
                .copied()
                .min_by(|x, y| (x - b).abs().total_cmp(&(y - b).abs()))
                .filter(|p| (p - b).abs() < 0.15 * FS)
                .map(|p| (p, b))
        })
        .collect();
    let rmssd = |v: Vec<f64>| {
        let rr: Vec<f64> = v.windows(2).map(|w| w[1] - w[0]).collect();
        let d: Vec<f64> = rr.windows(2).map(|w| (w[1] - w[0]).powi(2)).collect();
        (d.iter().sum::<f64>() / d.len() as f64).sqrt()
    };
    let got = rmssd(matched.iter().map(|m| m.0).collect());
    let want = rmssd(matched.iter().map(|m| m.1).collect());
    (got - want).abs() / want
}

#[test]
fn narrow_complexes_keep_the_extremum() {
    let (raw, _) = synth(90.0, 3, false);
    let clean = ecg_clean_causal(&raw, FS, &EcgFilterConfig::default()).unwrap();
    let wc = WindowConfig::default();
    let ext = dual_detect_windowed(&clean, FS, 0, &wc, &dual_with(Fiducial::Extremum)).unwrap();
    let ada = dual_detect_windowed(&clean, FS, 0, &wc, &dual_with(Fiducial::ADAPTIVE)).unwrap();
    assert_eq!(bits(&ada.primary), bits(&ext.primary));
    assert_eq!(bits(&ada.secondary), bits(&ext.secondary));
}

#[test]
fn wide_complexes_switch_to_the_centroid_and_rmssd_improves() {
    let (raw, beats) = synth(180.0, 5, true);
    let clean = ecg_clean_causal(&raw, FS, &EcgFilterConfig::default()).unwrap();
    let wc = WindowConfig::default();
    let ext = dual_detect_windowed(&clean, FS, 0, &wc, &dual_with(Fiducial::Extremum)).unwrap();
    let ada = dual_detect_windowed(&clean, FS, 0, &wc, &dual_with(Fiducial::ADAPTIVE)).unwrap();
    assert_ne!(bits(&ada.primary), bits(&ext.primary));
    let (e_ext, e_ada) = (
        rmssd_error(&ext.primary, &beats),
        rmssd_error(&ada.primary, &beats),
    );
    assert!(
        e_ada < 0.5 * e_ext,
        "extremum {e_ext:.3}, adaptive {e_ada:.3}"
    );
    assert!(ada.primary.len().abs_diff(beats.len()) <= 2);
}

#[test]
fn stream_equals_batch_for_centroid_and_adaptive() {
    let wc = WindowConfig::default();
    for wide in [false, true] {
        let (raw, _) = synth(75.0, 9, wide);
        let clean = ecg_clean_causal(&raw, FS, &EcgFilterConfig::default()).unwrap();
        for fid in [
            Fiducial::Centroid {
                half_window_s: 0.08,
            },
            Fiducial::ADAPTIVE,
        ] {
            let dual = dual_with(fid);
            for start in [0u64, 777, 5120] {
                let batch = dual_detect_windowed(&clean, FS, start, &wc, &dual).unwrap();
                let primary =
                    elgendi_refined_windowed(&clean, FS, start, &wc, &dual.elgendi, &dual.refine)
                        .unwrap();
                assert_eq!(bits(&batch.primary), bits(&primary));
                for packet in [1usize, 32, 4099] {
                    let live = stream(&clean, start, dual, packet, &[]);
                    let pos: Vec<f64> = live.primary.iter().map(|p| p.position).collect();
                    assert_eq!(
                        bits(&pos),
                        bits(&batch.primary),
                        "{fid:?} wide {wide} start {start} packet {packet}"
                    );
                    for p in &live.primary {
                        assert!((p.position - p.index as f64).abs() <= 0.5);
                    }
                }
            }
        }
    }
}

#[test]
fn adaptive_memory_resets_at_gaps_like_batch_per_segment() {
    // wide segment, then narrow after a discontinuity: the memory must not carry the wide decision
    let (wide, _) = synth(50.0, 13, true);
    let (narrow, _) = synth(50.0, 17, false);
    let wc = WindowConfig::default();
    let dual = dual_with(Fiducial::ADAPTIVE);
    let cfg = EcgFilterConfig::default();
    let a = ecg_clean_causal(&wide, FS, &cfg).unwrap();
    let b = ecg_clean_causal(&narrow, FS, &cfg).unwrap();
    let mut want = dual_detect_windowed(&a, FS, 0, &wc, &dual).unwrap().primary;
    want.extend(
        dual_detect_windowed(&b, FS, a.len() as u64, &wc, &dual)
            .unwrap()
            .primary,
    );
    let narrow_ext =
        dual_detect_windowed(&b, FS, a.len() as u64, &wc, &dual_with(Fiducial::Extremum)).unwrap();
    let joined: Vec<f64> = a.iter().chain(&b).copied().collect();
    for packet in [32usize, 1000] {
        let live = stream(&joined, 0, dual, packet, &[a.len()]);
        let pos: Vec<f64> = live.primary.iter().map(|p| p.position).collect();
        assert_eq!(bits(&pos), bits(&want), "packet {packet}");
        let tail: Vec<f64> = pos
            .iter()
            .copied()
            .filter(|&p| p >= a.len() as f64)
            .collect();
        assert_eq!(bits(&tail), bits(&narrow_ext.primary));
    }
}
