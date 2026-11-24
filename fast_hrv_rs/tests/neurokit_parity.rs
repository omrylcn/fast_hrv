//! Parity with NeuroKit2.
//!
//! References are generated at dev time by `tools/neurokit_reference.py` (NeuroKit2 / SciPy / pandas
//! versions are recorded in each JSON file) and committed under `tests/data/`. Inputs are synthetic
//! only. The tests FAIL if a reference file is missing; set `FAST_HRV_SKIP_PARITY=1` to skip them
//! explicitly.
//!
//! Comparison classes (see `class()`):
//! - **strict** (relative difference ≤ 0.1 %; in practice ~1e-12): metrics whose definition is the
//!   same in both libraries. Both sides get the same integer R-peak indices.
//! - **pNN50 / pNN20**: NeuroKit divides by the number of RR intervals (`len(diff) + 1`), fast-hrv
//!   by the number of successive differences (Task Force). The test rescales fast-hrv's value by
//!   `(n - 1) / n` and then compares strictly.
//! - **loose**, with the reason next to the tolerance: method differences (frequency domain:
//!   interpolation and Welch settings; HTI/TINN: histogram origin).
//! - **report only**: DFA α1/α2, whose definition differs (scales 4–16 / 16–64 beats vs NeuroKit's
//!   4–11 / 12–N/10, non-overlapping windows, mean of RMS instead of RMS of variance). Printed, not
//!   asserted.

use std::collections::BTreeMap;

use fast_hrv::peaks::{KubiosParams, detect_artifacts, find_peaks_default, fix_peaks_with};
use fast_hrv::{HrvReport, ecg_clean, hrv, rri_from_peaks};
use serde_json::Value;

const SKIP_ENV: &str = "FAST_HRV_SKIP_PARITY";

fn load(name: &str) -> Option<Value> {
    let path = format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"));
    match std::fs::read_to_string(&path) {
        Ok(text) => Some(serde_json::from_str(&text).unwrap_or_else(|e| panic!("{path}: {e}"))),
        Err(e) => {
            if std::env::var(SKIP_ENV).is_ok_and(|v| v == "1") {
                eprintln!("{path} missing ({e}); skipped because {SKIP_ENV}=1");
                None
            } else {
                panic!(
                    "NeuroKit2 reference {path} missing ({e}). Generate it with `python tools/neurokit_reference.py` \
                     or set {SKIP_ENV}=1 to skip the parity tests explicitly."
                )
            }
        }
    }
}

fn versions(v: &Value) -> String {
    v["versions"].to_string()
}

fn field(r: &HrvReport, n_rr: usize, path: &str) -> Option<f64> {
    let t = &r.time;
    let f = &r.frequency;
    let n = &r.nonlinear;
    let a = &r.asymmetry;
    let nk_pnn = |p: f64| p * (n_rr as f64 - 1.0) / n_rr as f64;
    Some(match path {
        "time.mean_nn" => t.mean_nn,
        "time.sdnn" => t.sdnn,
        "time.rmssd" => t.rmssd,
        "time.sdsd" => t.sdsd,
        "time.pnn50_nk" => nk_pnn(t.pnn50),
        "time.pnn20_nk" => nk_pnn(t.pnn20),
        "time.median_nn" => t.median_nn,
        "time.mad_nn" => t.mad_nn,
        "time.iqr_nn" => t.iqr_nn,
        "time.min_nn" => t.min_nn,
        "time.max_nn" => t.max_nn,
        "time.cvnn" => t.cvnn,
        "time.cvsd" => t.cvsd,
        "time.hti" => t.hti,
        "time.tinn" => t.tinn,
        "frequency.vlf" => f.vlf,
        "frequency.lf" => f.lf,
        "frequency.hf" => f.hf,
        "frequency.lf_hf" => f.lf_hf,
        "frequency.lfnu_frac" => f.lfnu / 100.0,
        "nonlinear.sd1" => n.sd1,
        "nonlinear.sd2" => n.sd2,
        "nonlinear.sd1_sd2" => n.sd1_sd2,
        "nonlinear.csi" => n.csi,
        "nonlinear.cvi" => n.cvi,
        "nonlinear.sampen" => n.sampen,
        "nonlinear.apen" => n.apen,
        "nonlinear.dfa_alpha1" => n.dfa_alpha1,
        "nonlinear.dfa_alpha2" => n.dfa_alpha2,
        "asymmetry.gi" => a.gi,
        "asymmetry.pi" => a.pi,
        "asymmetry.sd1d" => a.sd1d,
        "asymmetry.sd1a" => a.sd1a,
        "asymmetry.c1d" => a.c1d,
        "asymmetry.c1a" => a.c1a,
        _ => return None,
    })
}

enum Class {
    /// relative tolerance and the reason it is not strict
    Tol(f64, &'static str),
    ReportOnly(&'static str),
}

const STRICT: f64 = 1e-3; // 0.1 %

/// Comparison class of a metric for a record of `duration_s` seconds.
fn class(path: &str, duration_s: f64) -> Class {
    match path {
        "frequency.vlf" => Class::ReportOnly(
            "NeuroKit drops frequencies below 4/T (part of VLF) and uses Welch nperseg=N/2; VLF is not comparable",
        ),
        p if p.starts_with("frequency.") && duration_s < 290.0 => Class::ReportOnly(
            "records < 5 min: 64 s Welch segments (fast-hrv) vs N/2 segments (NK) give few, different averages",
        ),
        "frequency.lf" | "frequency.hf" | "frequency.lf_hf" | "frequency.lfnu_frac" => Class::Tol(
            0.30,
            "method: 4 Hz cubic spline + Welch 256 samples (fast-hrv) vs 100 Hz quadratic + Welch N/2, nfft 2N (NK)",
        ),
        "time.hti" | "time.tinn" => Class::Tol(
            1.5,
            "histogram starts at min(RR) in fast-hrv, at 0 in NeuroKit (bin 7.8125 ms)",
        ),
        "nonlinear.dfa_alpha1" | "nonlinear.dfa_alpha2" => {
            Class::ReportOnly("different DFA definition (scales, overlap, F(n)); see module docs")
        }
        _ => Class::Tol(STRICT, "same definition"),
    }
}

#[test]
fn hrv_metrics_match_neurokit() {
    let Some(r) = load("neurokit_reference.json") else {
        return;
    };
    eprintln!("NeuroKit2 reference versions: {}", versions(&r));
    let mut worst: BTreeMap<(String, &str), f64> = BTreeMap::new();
    let mut failures = Vec::new();
    let mut compared = 0usize;
    for case in r["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let fs = case["sampling_rate"].as_u64().unwrap() as u32;
        let peaks: Vec<usize> = case["peaks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let rri = rri_from_peaks(&peaks, fs);
        let duration_s = rri.iter().sum::<f64>() / 1000.0;
        let rep = hrv(&rri);
        for (metric, reference) in case["neurokit"].as_object().unwrap() {
            let Some(expected) = reference.as_f64() else {
                continue;
            };
            let got =
                field(&rep, rri.len(), metric).unwrap_or_else(|| panic!("unknown metric {metric}"));
            if got.is_nan() {
                // fast-hrv returns NaN below its minimum durations (e.g. LF < 120 s); NK computes anyway
                assert!(
                    metric.starts_with("frequency."),
                    "{name}: {metric} is NaN in fast-hrv (NK {expected})"
                );
                continue;
            }
            let rel = (got - expected).abs() / expected.abs().max(1e-12);
            let cls = class(metric, duration_s);
            let label = match cls {
                Class::Tol(t, why) if t == STRICT => why,
                Class::Tol(_, why) | Class::ReportOnly(why) => why,
            };
            let w = worst.entry((metric.clone(), label)).or_insert(0.0);
            *w = w.max(rel);
            compared += 1;
            if std::env::var("PARITY_VERBOSE").is_ok()
                && !matches!(cls, Class::Tol(t, _) if t == STRICT)
            {
                eprintln!(
                    "  {name}: {metric} fast-hrv={got:.4} nk={expected:.4} diff={:.1}%",
                    rel * 100.0
                );
            }
            if let Class::Tol(tol, _) = cls
                && rel > tol
            {
                failures.push(format!(
                    "{name}: {metric} fast-hrv={got:.9} neurokit={expected:.9} diff={:.3}%",
                    rel * 100.0
                ));
            }
        }
    }
    eprintln!("\nmetric | worst relative difference | class / reason");
    for ((m, why), w) in &worst {
        eprintln!("{m} | {w:.3e} ({:.3}%) | {why}", w * 100.0);
    }
    assert!(compared > 100, "too few comparisons ({compared})");
    assert!(
        failures.is_empty(),
        "{} metric(s) out of tolerance:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ---------------------------------------------------------------------------------------------
// ECG level: cleaning and R-peak detection
// ---------------------------------------------------------------------------------------------

fn b64_decode(s: &str) -> Vec<u8> {
    let val = |c: u8| -> u32 {
        match c {
            b'A'..=b'Z' => (c - b'A') as u32,
            b'a'..=b'z' => (c - b'a' + 26) as u32,
            b'0'..=b'9' => (c - b'0' + 52) as u32,
            b'+' => 62,
            b'/' => 63,
            _ => panic!("bad base64"),
        }
    };
    let bytes: Vec<u8> = s.bytes().filter(|&c| c != b'=').collect();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    for chunk in bytes.chunks(4) {
        let mut acc = 0u32;
        for (i, &c) in chunk.iter().enumerate() {
            acc |= val(c) << (18 - 6 * i);
        }
        let n = chunk.len() * 6 / 8;
        for i in 0..n {
            out.push((acc >> (16 - 8 * i)) as u8);
        }
    }
    out
}

fn decode_i16(s: &str) -> Vec<f64> {
    b64_decode(s)
        .chunks(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f64)
        .collect()
}

fn decode_f64(s: &str) -> Vec<f64> {
    b64_decode(s)
        .chunks(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn ecg_clean_matches_neurokit() {
    let Some(r) = load("neurokit_ecg.json") else {
        return;
    };
    eprintln!("NeuroKit2 reference versions: {}", versions(&r));
    for case in r["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let fs = case["sampling_rate"].as_u64().unwrap() as u32;
        let raw = decode_i16(case["raw_i16_b64"].as_str().unwrap());
        assert_eq!(raw.len(), case["n"].as_u64().unwrap() as usize);
        let step = case["clean_step"].as_u64().unwrap() as usize;
        let expected = decode_f64(case["clean_f64_b64"].as_str().unwrap());
        let got = ecg_clean(&raw, fs);
        let scale = expected.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let max_err = got
            .iter()
            .step_by(step)
            .zip(&expected)
            .map(|(g, e)| (g - e).abs())
            .fold(0.0, f64::max);
        eprintln!(
            "ecg_clean {name}: max |diff| = {max_err:.3e} (signal max {scale:.1}, relative {:.3e})",
            max_err / scale
        );
        assert!(
            max_err / scale < 1e-9,
            "{name}: ecg_clean differs from NeuroKit2 by {max_err}"
        );
    }
}

#[test]
fn r_peaks_match_neurokit() {
    let Some(r) = load("neurokit_ecg.json") else {
        return;
    };
    for case in r["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let fs = case["sampling_rate"].as_u64().unwrap() as u32;
        let raw = decode_i16(case["raw_i16_b64"].as_str().unwrap());
        let nk: Vec<usize> = case["peaks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let got = find_peaks_default(&ecg_clean(&raw, fs), fs);
        let exact = nk.iter().filter(|p| got.contains(p)).count();
        // Known difference: NeuroKit starts with peaks=[0] and so drops R-peaks in the first 0.3 s;
        // fast-hrv keeps them.
        let skip = (0.3 * fs as f64) as usize;
        let extra: Vec<usize> = got.iter().copied().filter(|p| !nk.contains(p)).collect();
        eprintln!(
            "R-peaks {name}: NK {} fast-hrv {} identical {exact} extra {extra:?}",
            nk.len(),
            got.len()
        );
        assert_eq!(
            exact,
            nk.len(),
            "{name}: every NeuroKit2 R-peak must be found at the same sample"
        );
        assert!(
            extra.iter().all(|&p| p < skip),
            "{name}: unexpected extra peaks {extra:?}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Kubios artifact detection and correction
// ---------------------------------------------------------------------------------------------

fn usizes(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}

#[test]
fn kubios_matches_neurokit() {
    let Some(r) = load("neurokit_kubios.json") else {
        return;
    };
    eprintln!("NeuroKit2 reference versions: {}", versions(&r));
    let params = KubiosParams::default();
    let (mut det_ok, mut fix_ok, mut total) = (0, 0, 0);
    let mut failures = Vec::new();
    for case in r["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let fs = case["sampling_rate"].as_f64().unwrap();
        let peaks = usizes(&case["peaks"]);
        let first = detect_artifacts(&peaks, fs, &params);
        let fp = &case["first_pass"];
        let det_same = first.ectopic == usizes(&fp["ectopic"])
            && first.missed == usizes(&fp["missed"])
            && first.extra == usizes(&fp["extra"])
            && first.longshort == usizes(&fp["longshort"]);
        let fixed = fix_peaks_with(&peaks, fs, &params).unwrap();
        let fix_same = fixed.peaks == usizes(&case["fixed"]);
        total += 1;
        det_ok += det_same as usize;
        fix_ok += fix_same as usize;
        if !(det_same && fix_same) {
            failures.push(format!(
                "{name}: NK first pass {fp} / fast-hrv {first:?}; fixed identical: {fix_same}"
            ));
        }
    }
    eprintln!(
        "Kubios: first-pass detection identical {det_ok}/{total}, corrected peaks identical {fix_ok}/{total}"
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
