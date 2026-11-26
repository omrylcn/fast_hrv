//! Lomb–Scargle: astropy parity (tests/data/lomb_reference.json, from tools/lomb_reference.py),
//! power scale (a sine of amplitude A gives A²/2), and agreement with Welch on gap-free series.

use fast_hrv::{FrequencyOptions, frequency_domain_lomb, frequency_domain_with, lomb_scargle};
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

#[test]
fn matches_astropy() {
    let s = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/lomb_reference.json"
    ))
    .expect("reference file");
    let d: Value = serde_json::from_str(&s).unwrap();
    let cases = d["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 4);
    for c in cases {
        let (t, y, f, p) = (
            floats(&c["t"]),
            floats(&c["y"]),
            floats(&c["freqs"]),
            floats(&c["p_astropy"]),
        );
        let scale = 2.0 * (t[t.len() - 1] - t[0]) / t.len() as f64;
        let ours = lomb_scargle(&t, &y, &f);
        let pmax = p.iter().cloned().fold(0.0, f64::max);
        let mut worst = 0.0f64;
        for (a, b) in ours.iter().zip(&p) {
            worst = worst.max((a / scale - b).abs() / pmax);
        }
        assert!(
            worst < 1e-9,
            "{}: max error {worst:e} of the peak",
            c["name"]
        );
    }
}

#[test]
fn sine_power_is_half_amplitude_squared() {
    // 0.25 Hz sine, amplitude 30 ms, on jittered beat times; HF band power ≈ 450 ms²
    let mut t = Vec::new();
    let mut y = Vec::new();
    let mut s = 0.0;
    let mut k = 1u64;
    while s < 300.0 {
        k = k
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s += 0.8 + 0.1 * ((k >> 33) as f64 / (1u64 << 31) as f64 - 0.5);
        t.push(s);
        y.push(850.0 + 30.0 * (2.0 * std::f64::consts::PI * 0.25 * s).sin());
    }
    let fd = frequency_domain_lomb(&t, &y, &FrequencyOptions::default());
    assert!((fd.hf / 450.0 - 1.0).abs() < 0.05, "HF {}", fd.hf);
    assert!(fd.lf < 0.05 * fd.hf, "LF {}", fd.lf);
    assert!((fd.hf_peak - 0.25).abs() < 0.01);
    assert!((fd.lnhf - fd.hf.ln()).abs() < 1e-12);
}

#[test]
fn agrees_with_welch_without_gaps_and_tolerates_dropped_intervals() {
    let mut t = Vec::new();
    let mut nn = Vec::new();
    let mut s = 0.0;
    while s < 300.0 {
        let rr = 0.85
            + 0.03 * (2.0 * std::f64::consts::PI * 0.25 * s).sin()
            + 0.02 * (2.0 * std::f64::consts::PI * 0.1 * s).sin();
        s += rr;
        t.push(s);
        nn.push(rr * 1000.0);
    }
    let opts = FrequencyOptions::default();
    let w = frequency_domain_with(&nn, &opts);
    let l = frequency_domain_lomb(&t, &nn, &opts);
    for (a, b, name) in [
        (w.hf, l.hf, "hf"),
        (w.lf, l.lf, "lf"),
        (w.lf_hf, l.lf_hf, "lf_hf"),
    ] {
        assert!((a / b - 1.0).abs() < 0.15, "{name}: welch {a} lomb {b}");
    }
    // drop 10 % of the intervals (exclusion mode): Lomb stays close, nothing is interpolated
    let (td, nd): (Vec<f64>, Vec<f64>) = t
        .iter()
        .zip(&nn)
        .enumerate()
        .filter(|(i, _)| i % 10 != 3)
        .map(|(_, (a, b))| (*a, *b))
        .unzip();
    let l2 = frequency_domain_lomb(&td, &nd, &opts);
    for (a, b, name) in [(l.hf, l2.hf, "hf"), (l.lf, l2.lf, "lf")] {
        assert!((a / b - 1.0).abs() < 0.10, "{name}: full {a} dropped {b}");
    }
    // NaN entries are dropped, too short spans give NaN
    let mut nn_nan = nn.clone();
    nn_nan[5] = f64::NAN;
    assert!(frequency_domain_lomb(&t, &nn_nan, &opts).hf.is_finite());
    assert!(
        frequency_domain_lomb(&t[..40], &nn[..40], &opts)
            .lf
            .is_nan()
    );
}
