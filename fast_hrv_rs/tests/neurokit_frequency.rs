//! `frequency_domain_neurokit` against `nk.hrv_frequency` defaults (fixture from NeuroKit2 on simulated
//! ECG, see `tests/fixtures/neurokit_frequency.json`). NeuroKit2 divides its PSD by the maximum, so only
//! ratios are compared.
use fast_hrv::{frequency_domain, frequency_domain_neurokit};

#[test]
fn neurokit_frequency_parity() {
    let txt = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/neurokit_frequency.json"
    ))
    .unwrap();
    let v: serde_json::Value = serde_json::from_str(&txt).unwrap();
    for c in v["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let rri: Vec<f64> = c["rri"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let nk = frequency_domain_neurokit(&rri);
        let want = c["lf_hf"].as_f64().unwrap();
        let rel = (nk.lf_hf - want).abs() / want;
        let default = frequency_domain(&rri).lf_hf;
        eprintln!(
            "{name}: LF/HF neurokit-mode {:.6} NeuroKit2 {want:.6} (rel {rel:.1e}); default {default:.4}",
            nk.lf_hf
        );
        assert!(rel < 1e-6, "{name}: LF/HF {} vs {want} ({rel:e})", nk.lf_hf);
        // VLF: compared when it carries power; on these simulated ECGs it is near zero (< 1e-4 of LF),
        // where the PSD values are rounding noise around the removed mean, so both must just be tiny
        if let Some(w) = c["vlf_over_lf"].as_f64() {
            let got = nk.vlf / nk.lf;
            if w > 1e-3 {
                assert!(((got - w) / w).abs() < 1e-5, "{name}: VLF/LF {got} vs {w}");
            } else {
                assert!(got < 1e-3, "{name}: VLF/LF {got} vs {w}");
            }
        }
    }
}
