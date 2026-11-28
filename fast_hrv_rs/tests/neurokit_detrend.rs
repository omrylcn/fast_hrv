//! `smoothness_priors` against NeuroKit2 `signal_detrend(method="tarvainen2002")` (fixture: an RR
//! series from a simulated ECG with a slow trend added, λ = 10 and 500).
use fast_hrv::smoothness_priors;

#[test]
fn tarvainen2002_matches_neurokit() {
    let txt = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/neurokit_detrend.json"
    ))
    .unwrap();
    let v: serde_json::Value = serde_json::from_str(&txt).unwrap();
    let f = |a: &serde_json::Value| -> Vec<f64> {
        a.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect()
    };
    let x = f(&v["x"]);
    for c in v["cases"].as_array().unwrap() {
        let lambda = c["lambda"].as_f64().unwrap();
        let want = f(&c["detrended"]);
        let got = smoothness_priors(&x, lambda).unwrap();
        let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let err = got
            .iter()
            .zip(&want)
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()))
            / scale;
        eprintln!("λ = {lambda}: max relative difference {err:e}");
        assert!(err < 1e-9, "λ = {lambda}: {err:e}");
    }
}
