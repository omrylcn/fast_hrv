//! `respiratory_hf`: HF measured around the breathing rate catches slow-breathing RSA that the fixed
//! 0.15–0.4 Hz band misses, and agrees with it at normal breathing.
use fast_hrv::{FrequencyOptions, frequency_domain, respiratory_hf};

fn rr(resp_hz: f64, seconds: f64) -> Vec<f64> {
    let (mut t, mut out) = (0.0, Vec::new());
    while t < seconds {
        let v = 900.0 + 40.0 * (2.0 * std::f64::consts::PI * resp_hz * t).sin();
        out.push(v);
        t += v / 1000.0;
    }
    out
}

#[test]
fn slow_breathing_is_caught_and_flagged() {
    let opts = FrequencyOptions::default();
    // 7.2 breaths/min = 0.12 Hz: below the HF band
    let x = rr(0.12, 300.0);
    let fixed = frequency_domain(&x);
    let resp = respiratory_hf(&x, 7.2, &opts);
    assert!(resp.in_lf_band && !resp.above_hf_band);
    assert!(
        resp.power > 3.0 * fixed.hf,
        "respiratory {} vs fixed HF {}",
        resp.power,
        fixed.hf
    );
    // 15 /min = 0.25 Hz: both bands hold the respiratory power
    let x = rr(0.25, 300.0);
    let fixed = frequency_domain(&x);
    let resp = respiratory_hf(&x, 15.0, &opts);
    assert!(!resp.in_lf_band && !resp.above_hf_band);
    assert!(
        (resp.power / fixed.hf - 1.0).abs() < 0.15,
        "{} vs {}",
        resp.power,
        fixed.hf
    );
    // unknown breathing rate
    assert!(respiratory_hf(&x, f64::NAN, &opts).power.is_nan());
}
