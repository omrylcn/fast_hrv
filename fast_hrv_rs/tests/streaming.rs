//! Live (packet-by-packet) processing must give exactly the same result as offline reprocessing of
//! the same recording.

use fast_hrv::ecg::{EcgFilterConfig, EcgStreamFilter, ecg_clean_causal};
use fast_hrv::find_peaks_default;

/// Synthetic raw ECG at `fs`: DC offset, Gaussian QRS bumps with slightly varying RR, baseline wander
/// and 50 Hz mains (no real recordings).
fn synthetic_ecg(fs: f64, seconds: f64) -> Vec<f64> {
    let n = (fs * seconds) as usize;
    let mut x = vec![412.0; n];
    let mut t = 0.4;
    let mut k = 0.0f64;
    while t < seconds - 0.5 {
        let c = (t * fs) as isize;
        for d in -20isize..=20 {
            let i = c + d;
            if i >= 0 && (i as usize) < n {
                x[i as usize] += 180.0 * (-(d as f64).powi(2) / 20.0).exp();
            }
        }
        k += 1.0;
        t += 0.82 + 0.04 * (k * 0.6).sin();
    }
    for (i, v) in x.iter_mut().enumerate() {
        let ts = i as f64 / fs;
        *v += 40.0 * (2.0 * std::f64::consts::PI * 0.2 * ts).sin()
            + 15.0 * (2.0 * std::f64::consts::PI * 50.0 * ts).sin();
    }
    x
}

#[test]
fn packets_equal_batch_bit_for_bit() {
    let fs = 512.0;
    let raw = synthetic_ecg(fs, 120.0);
    for cfg in [
        EcgFilterConfig::default(),
        EcgFilterConfig::default().with_notch(50.0, 30.0),
    ] {
        let batch = ecg_clean_causal(&raw, fs, &cfg).unwrap();
        for packet in [32usize, 1, 33, 512, 4096] {
            let mut live = EcgStreamFilter::new(fs, &cfg).unwrap();
            let mut out = Vec::with_capacity(raw.len());
            let mut buf = vec![0.0; packet];
            for p in raw.chunks(packet) {
                let o = &mut buf[..p.len()];
                live.process_into(p, o);
                out.extend_from_slice(o);
            }
            assert_eq!(out, batch, "packet size {packet}");
        }
        // identical filtered signal => identical R-peaks
        let mut live = EcgStreamFilter::new(fs, &cfg).unwrap();
        let streamed: Vec<f64> = raw.chunks(32).flat_map(|p| live.process(p)).collect();
        assert_eq!(
            find_peaks_default(&streamed, 512),
            find_peaks_default(&batch, 512)
        );
    }
}

#[test]
fn reset_after_gap_equals_fresh_filter() {
    let fs = 512.0;
    let raw = synthetic_ecg(fs, 30.0);
    let (a, b) = raw.split_at(raw.len() / 2);
    let cfg = EcgFilterConfig::default();
    let mut live = EcgStreamFilter::new(fs, &cfg).unwrap();
    let _ = live.process(a);
    live.reset();
    let after = live.process(b);
    assert_eq!(after, ecg_clean_causal(b, fs, &cfg).unwrap());
}

#[test]
fn steady_state_start_has_no_dc_transient() {
    let fs = 512.0;
    let raw = vec![412.0; 2048];
    let y = ecg_clean_causal(&raw, fs, &EcgFilterConfig::default()).unwrap();
    assert!(
        y.iter().all(|v| v.abs() < 1e-6),
        "max {}",
        y.iter().fold(0.0f64, |m, v| m.max(v.abs()))
    );
}
