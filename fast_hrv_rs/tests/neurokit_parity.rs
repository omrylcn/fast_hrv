//! NeuroKit2 ile karşılaştırma. Referanslar `tools/neurokit_reference.py` ile üretilir (tests/data).
//! Her metrik için bağıl tolerans: tanımı birebir aynı olanlar sıkı, yöntem farkı olanlar (frekans alanı:
//! enterpolasyon ve Welch ayarları; TINN: histogram yöntemi) gevşek. Sonuç tablosu `--nocapture` ile basılır.

use std::collections::BTreeMap;

use fast_hrv::{hrv, HrvReport};
use serde_json::Value;

fn field(r: &HrvReport, path: &str) -> Option<f64> {
    let t = &r.time;
    let f = &r.frequency;
    let n = &r.nonlinear;
    let a = &r.asymmetry;
    Some(match path {
        "time.mean_nn" => t.mean_nn, "time.sdnn" => t.sdnn, "time.rmssd" => t.rmssd, "time.sdsd" => t.sdsd,
        "time.pnn50" => t.pnn50, "time.pnn20" => t.pnn20, "time.median_nn" => t.median_nn, "time.mad_nn" => t.mad_nn,
        "time.iqr_nn" => t.iqr_nn, "time.min_nn" => t.min_nn, "time.max_nn" => t.max_nn, "time.cvnn" => t.cvnn,
        "time.cvsd" => t.cvsd, "time.hti" => t.hti, "time.tinn" => t.tinn,
        "frequency.vlf" => f.vlf, "frequency.lf" => f.lf, "frequency.hf" => f.hf, "frequency.lf_hf" => f.lf_hf,
        // NeuroKit LFn/HFn = LF/total, HF/total (kesir)
        "frequency.lfnu_frac" => f.lf / f.total_power, "frequency.hfnu_frac" => f.hf / f.total_power,
        "nonlinear.sd1" => n.sd1, "nonlinear.sd2" => n.sd2, "nonlinear.sd1_sd2" => n.sd1_sd2, "nonlinear.csi" => n.csi,
        "nonlinear.cvi" => n.cvi, "nonlinear.sampen" => n.sampen, "nonlinear.apen" => n.apen,
        "nonlinear.dfa_alpha1" => n.dfa_alpha1, "nonlinear.dfa_alpha2" => n.dfa_alpha2,
        "asymmetry.gi" => a.gi, "asymmetry.pi" => a.pi, "asymmetry.sd1d" => a.sd1d, "asymmetry.sd1a" => a.sd1a,
        "asymmetry.c1d" => a.c1d, "asymmetry.c1a" => a.c1a,
        _ => return None,
    })
}

/// Bağıl tolerans (|fark| / max(|ref|, 1e-9))
fn tolerance(path: &str) -> f64 {
    match path {
        p if p.starts_with("frequency.") => 0.25,
        "time.tinn" | "time.hti" => 0.25,
        "nonlinear.dfa_alpha1" | "nonlinear.dfa_alpha2" => 0.15,
        "nonlinear.sampen" | "nonlinear.apen" => 0.05,
        _ => 0.01,
    }
}

#[test]
fn matches_neurokit_within_tolerance() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/neurokit_reference.json");
    let Ok(text) = std::fs::read_to_string(path) else {
        eprintln!("referans dosyası yok ({path}); önce tools/neurokit_reference.py çalıştırılmalı. Atlandı.");
        return;
    };
    let cases: Vec<Value> = serde_json::from_str(&text).unwrap();
    let mut worst: BTreeMap<String, f64> = BTreeMap::new();
    let mut failures = Vec::new();
    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let rri: Vec<f64> = case["rri_ms"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let r = hrv(&rri);
        for (metric, reference) in case["neurokit"].as_object().unwrap() {
            let Some(expected) = reference.as_f64() else { continue };
            let Some(got) = field(&r, metric) else { continue };
            let rel = (got - expected).abs() / expected.abs().max(1e-9);
            let w = worst.entry(metric.clone()).or_insert(0.0);
            *w = w.max(rel);
            if !(rel <= tolerance(metric)) {
                failures.push(format!("{name}: {metric} fast-hrv={got:.6} neurokit={expected:.6} fark=%{:.2}", rel * 100.0));
            }
        }
    }
    eprintln!("\nmetrik | en kötü bağıl fark | tolerans");
    for (m, w) in &worst {
        eprintln!("{m} | %{:.3} | %{:.0}", w * 100.0, tolerance(m) * 100.0);
    }
    assert!(failures.is_empty(), "{} metrik toleransı aştı:\n{}", failures.len(), failures.join("\n"));
}

