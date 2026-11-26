//! ECG-derived respiration end to end: raw ECG → cleaning → R peaks → R amplitudes + RR →
//! breathing rate (AM/FM smart fusion).
//! 1. Synthetic ECG with known breathing (amplitude modulation + respiratory sinus arrhythmia).
//! 2. NeuroKit2 ECGSYN (respiratory component at 0.25 Hz = 15/min), tests/data/respiration_ecgsyn.json
//!    from tools/respiration_reference.py.

use fast_hrv::{RespirationConfig, ecg_clean, ecg_peaks, r_amplitudes, respiratory_rate};
use serde_json::Value;

const FS: u32 = 512;

fn synthetic(seconds: f64, resp_bpm: f64, seed: u64) -> Vec<f64> {
    let fs = FS as f64;
    let n = (seconds * fs) as usize;
    let mut x = vec![0.0; n];
    let mut st = seed;
    let mut rnd = move || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (st >> 33) as f64 / (1u64 << 31) as f64 - 0.5
    };
    let fr = resp_bpm / 60.0;
    let mut t = 0.5;
    while t < seconds {
        let ph = 2.0 * std::f64::consts::PI * fr * t;
        let amp = 1.0 + 0.10 * ph.cos(); // chest volume modulates the R amplitude
        let c = (t * fs) as isize;
        for d in -40isize..220 {
            let i = c + d;
            if i < 0 || i as usize >= n {
                continue;
            }
            let tt = (i as f64 - t * fs) / fs;
            x[i as usize] += amp
                * (1.0 * (-(tt / 0.012).powi(2)).exp()
                    - 0.15 * (-((tt + 0.03) / 0.01).powi(2)).exp())
                + 0.25 * (-((tt - 0.28) / 0.05).powi(2)).exp();
        }
        t += 0.85 + 0.035 * ph.sin() + 0.004 * rnd(); // RSA
    }
    x.iter()
        .enumerate()
        .map(|(i, v)| {
            v + 0.05 * (2.0 * std::f64::consts::PI * fr * i as f64 / fs).sin() + 0.01 * rnd()
        })
        .collect()
}

/// Breathing rate per 60 s window (30 s step) of a whole recording.
fn rates(raw: &[f64]) -> Vec<(f64, bool)> {
    let fs = FS as f64;
    let clean = ecg_clean(raw, FS);
    let (peaks, _) = ecg_peaks(&clean, FS, false);
    let amp = r_amplitudes(&clean, &peaks, fs);
    let times: Vec<f64> = peaks.iter().map(|&p| p as f64 / fs).collect();
    let mut out = Vec::new();
    let dur = raw.len() as f64 / fs;
    let mut w0 = 0.0;
    while w0 + 60.0 <= dur {
        let idx: Vec<usize> = (1..peaks.len())
            .filter(|&k| times[k] >= w0 && times[k] < w0 + 60.0)
            .collect();
        let t: Vec<f64> = idx.iter().map(|&k| times[k]).collect();
        let rr: Vec<f64> = idx
            .iter()
            .map(|&k| (times[k] - times[k - 1]) * 1000.0)
            .collect();
        let a: Vec<f64> = idx.iter().map(|&k| amp[k]).collect();
        let e = respiratory_rate(&t, &rr, &a, &RespirationConfig::default());
        out.push((e.rate_bpm, e.valid));
        w0 += 30.0;
    }
    out
}

#[test]
fn synthetic_ecg_breathing_rate() {
    for (resp, seed) in [(9.0, 1), (12.0, 2), (16.0, 3), (22.0, 4)] {
        let r = rates(&synthetic(180.0, resp, seed));
        eprintln!("synthetic {resp}/min {r:?}");
        let valid: Vec<f64> = r.iter().filter(|x| x.1).map(|x| x.0).collect();
        assert!(
            valid.len() * 4 >= r.len() * 3,
            "{resp}/min: only {} of {} windows valid: {r:?}",
            valid.len(),
            r.len()
        );
        for v in &valid {
            assert!((v - resp).abs() <= 1.5, "{resp}/min: estimate {v}");
        }
    }
}

#[test]
fn neurokit_ecgsyn_breathing_rate() {
    let s = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/respiration_ecgsyn.json"
    ))
    .expect("reference");
    let d: Value = serde_json::from_str(&s).unwrap();
    for c in d["cases"].as_array().unwrap() {
        let b = base64(c["ecg_i16"].as_str().unwrap());
        let raw: Vec<f64> = b
            .chunks_exact(2)
            .map(|p| i16::from_le_bytes([p[0], p[1]]) as f64 / 1000.0)
            .collect();
        let want = c["resp_bpm"].as_f64().unwrap();
        let r = rates(&raw);
        // FM (RSA) carries the 0.25 Hz component in ECGSYN; report every window
        eprintln!("{} {r:?}", c["name"]);
        let valid: Vec<f64> = r.iter().filter(|x| x.1).map(|x| x.0).collect();
        assert!(!valid.is_empty(), "{}: no valid window {r:?}", c["name"]);
        for v in &valid {
            assert!(
                (v - want).abs() <= 2.0,
                "{}: estimate {v} vs {want} ({r:?})",
                c["name"]
            );
        }
    }
}

fn base64(s: &str) -> Vec<u8> {
    let val = |c: u8| -> u32 {
        match c {
            b'A'..=b'Z' => (c - b'A') as u32,
            b'a'..=b'z' => (c - b'a' + 26) as u32,
            b'0'..=b'9' => (c - b'0' + 52) as u32,
            b'+' => 62,
            _ => 63,
        }
    };
    let bytes: Vec<u8> = s.bytes().filter(|c| *c != b'=').collect();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    for ch in bytes.chunks(4) {
        let mut acc = 0u32;
        for (i, &c) in ch.iter().enumerate() {
            acc |= val(c) << (18 - 6 * i);
        }
        let n = ch.len() * 6 / 8;
        for i in 0..n {
            out.push((acc >> (16 - 8 * i)) as u8);
        }
    }
    out
}
