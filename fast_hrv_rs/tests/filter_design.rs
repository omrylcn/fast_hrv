//! Filter design and filtering against SciPy reference values.
//!
//! Reference: `tests/data/filter_reference.json`, generated at dev time by
//! `tools/filter_reference.py` (SciPy version recorded in the file). The crate itself never calls Python.

use fast_hrv::filter::{
    BandType, SosSection, butter_sos, filtfilt_ba, iirnotch, sosfilt, sosfilt_zi, sosfiltfilt,
};
use serde_json::Value;

fn load() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/filter_reference.json"
    );
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{path}: {e}; run tools/filter_reference.py to generate it"));
    serde_json::from_str(&text).unwrap()
}

fn vecf(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn btype(s: &str) -> BandType {
    match s {
        "lowpass" => BandType::Lowpass,
        "highpass" => BandType::Highpass,
        "bandpass" => BandType::Bandpass,
        "bandstop" => BandType::Bandstop,
        _ => panic!("{s}"),
    }
}

/// Max absolute error between two coefficient sets, relative to max(1, |ref|).
fn err(got: &[f64], exp: &[f64]) -> f64 {
    assert_eq!(got.len(), exp.len());
    got.iter()
        .zip(exp)
        .map(|(g, e)| (g - e).abs() / e.abs().max(1.0))
        .fold(0.0, f64::max)
}

#[test]
fn butter_sos_matches_scipy() {
    let r = load();
    let mut worst: f64 = 0.0;
    for case in r["butter"].as_array().unwrap() {
        let order = case["order"].as_u64().unwrap() as usize;
        let cutoff = vecf(&case["cutoff"]);
        let fs = case["fs"].as_f64().unwrap();
        let bt = case["btype"].as_str().unwrap();
        let sos = butter_sos(order, &cutoff, btype(bt), fs).unwrap();
        let exp: Vec<Vec<f64>> = case["sos"].as_array().unwrap().iter().map(vecf).collect();
        assert_eq!(
            sos.len(),
            exp.len(),
            "{bt} order {order} {cutoff:?} fs {fs}: section count"
        );
        for (s, e) in sos.iter().zip(&exp) {
            let e = err(&s.to_row(), e);
            worst = worst.max(e);
            assert!(
                e < 1e-9,
                "{bt} order {order} {cutoff:?} fs {fs}: coefficient error {e:e}"
            );
        }
        let zi = sosfilt_zi(&sos);
        let exp_zi: Vec<Vec<f64>> = case["zi"].as_array().unwrap().iter().map(vecf).collect();
        for (z, e) in zi.iter().zip(&exp_zi) {
            let ez = err(z, e);
            assert!(
                ez < 1e-9,
                "{bt} order {order} {cutoff:?} fs {fs}: zi error {ez:e}"
            );
        }
    }
    eprintln!("butter_sos: max coefficient error vs scipy = {worst:e}");
}

#[test]
fn iirnotch_matches_scipy() {
    let r = load();
    for case in r["iirnotch"].as_array().unwrap() {
        let (f0, q, fs) = (
            case["f0"].as_f64().unwrap(),
            case["q"].as_f64().unwrap(),
            case["fs"].as_f64().unwrap(),
        );
        let s = iirnotch(f0, q, fs).unwrap();
        let b = vecf(&case["b"]);
        let a = vecf(&case["a"]);
        let e = err(&s.to_row(), &[b[0], b[1], b[2], a[0], a[1], a[2]]);
        assert!(e < 1e-12, "notch {f0} Hz Q {q} fs {fs}: {e:e}");
    }
}

fn rel_signal_err(got: &[f64], exp: &[f64]) -> f64 {
    let scale = exp.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-12);
    got.iter()
        .zip(exp)
        .map(|(g, e)| (g - e).abs())
        .fold(0.0, f64::max)
        / scale
}

#[test]
fn zero_phase_filtering_matches_scipy() {
    let r = load();
    let f = &r["filtering"];
    let fs = f["fs"].as_f64().unwrap();
    let x = vecf(&f["x"]);
    let hp5 = butter_sos(5, &[0.5], BandType::Highpass, fs).unwrap();
    let bp2 = butter_sos(2, &[0.5, 40.0], BandType::Bandpass, fs).unwrap();

    let e = rel_signal_err(&sosfiltfilt(&hp5, &x), &vecf(&f["sosfiltfilt_hp5"]));
    eprintln!("sosfiltfilt hp5: max rel error {e:e}");
    assert!(e < 1e-9, "sosfiltfilt hp5 {e:e}");

    let e = rel_signal_err(&sosfiltfilt(&bp2, &x), &vecf(&f["sosfiltfilt_bp2"]));
    eprintln!("sosfiltfilt bp2: max rel error {e:e}");
    assert!(e < 1e-9, "sosfiltfilt bp2 {e:e}");

    let ma = vec![1.0; (fs / 50.0) as usize];
    let e = rel_signal_err(
        &filtfilt_ba(&ma, &[ma.len() as f64], &x),
        &vecf(&f["filtfilt_ma10"]),
    );
    eprintln!("filtfilt moving average: max rel error {e:e}");
    assert!(e < 1e-12, "filtfilt ma {e:e}");

    let (y, _) = sosfilt(&bp2, &x, None);
    let e = rel_signal_err(&y, &vecf(&f["sosfilt_bp2"]));
    assert!(e < 1e-9, "sosfilt bp2 {e:e}");
}

#[test]
fn sos_row_roundtrip() {
    let s = SosSection::from_row([2.0, 4.0, 6.0, 2.0, 1.0, 0.5]);
    assert_eq!(s.to_row(), [1.0, 2.0, 3.0, 1.0, 0.5, 0.25]);
}
