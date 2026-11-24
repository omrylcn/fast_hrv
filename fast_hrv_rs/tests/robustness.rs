//! Inputs that used to panic (or overflow silently in release) must now return NaN / errors.

use fast_hrv::gaps::{Gap, rri_with_gaps};
use fast_hrv::*;

#[test]
fn empty_inputs() {
    let p = compute_poincare(&[]);
    assert!(p.sd1.is_nan());
    let a = compute_asymmetry(&[]);
    assert!(a.gi.is_nan());
    assert!(compute_dfa(&[], (4, 16), (16, None)).alpha1.is_nan());
    assert!(time_domain(&[]).sdnn.is_nan());
    assert!(frequency_domain(&[]).lf.is_nan());
    assert_eq!(hrv(&[]).n, 0);
    assert!(rri_from_peaks(&[], 512).is_empty());
    assert!(fix_peaks_default(&[], 512).0.is_empty());
    assert!(ecg_clean(&[], 512).is_empty());
    assert!(compute_poincare(&[800.0]).sd1.is_nan());
}

#[test]
fn nan_inputs() {
    let mut v: Vec<f64> = (0..400).map(|i| 800.0 + (i as f64).sin() * 30.0).collect();
    v[5] = f64::NAN;
    assert!(time_domain(&v).sdnn.is_nan());
    let f = frequency_domain(&v);
    assert!(f.lf.is_nan() && f.hf.is_nan());
    assert!(hrv(&v).nonlinear.sampen.is_nan());
}

#[test]
fn unsorted_and_duplicate_peaks() {
    let r = rri_from_peaks(&[100, 50], 512);
    assert!(r[0].is_nan());
    let (p, _) = fix_peaks_default(&[100, 600, 550, 1000, 1500], 512);
    assert!(p.windows(2).all(|w| w[1] > w[0]));
    let (_, q) = compute_quality(&vec![0.0; 5000], &[100, 600, 550, 1100, 1600, 2100], 512);
    assert!(q.is_empty());
    let s = compute_stats(&[100, 50, 600], &[], 512);
    assert_eq!(s.n_peaks, 3);
    assert!(rri_with_gaps(&[100, 50], &[], 512.0, None).is_err());
    assert!(rri_with_gaps(&[100, 500], &[Gap { start: 10, end: 5 }], 512.0, None).is_err());
}

#[test]
fn invalid_sampling_rate() {
    let y = ecg_clean(&[1.0; 100], 0);
    assert!(y.iter().all(|v| v.is_nan()));
    assert!(
        fast_hrv::filter::butter_sos(2, &[0.5], fast_hrv::filter::BandType::Highpass, 0.0).is_err()
    );
    let (p, a) = fix_peaks_default(&[100, 500, 900, 1300], 0);
    assert_eq!(p, vec![100, 500, 900, 1300]);
    assert_eq!(a.total_count(), 0);
}

#[test]
fn quality_with_peaks_beyond_signal() {
    let s: Vec<f64> = (0..5000).map(|i| ((i as f64) * 0.01).sin()).collect();
    let (q, _) = compute_quality(&s, &[100, 600, 1100, 1600, 2100, 9000], 512);
    assert_eq!(q.len(), s.len());
}

#[test]
fn long_gap_rr_is_cheap() {
    // A 69 s RR (a gap fed as one interval) used to make TINN O(B^3): ~0.8 s per call
    let mut rri: Vec<f64> = (0..360)
        .map(|i| 830.0 + 20.0 * (i as f64 * 0.3).sin())
        .collect();
    rri[100] = 69_000.0;
    let t0 = std::time::Instant::now();
    let r = hrv(&rri);
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    eprintln!("hrv() with one 69 s RR: {ms:.2} ms");
    assert!(r.time.tinn.is_nan());
    assert!(ms < 200.0, "{ms} ms");
}
