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
//! - **loose**, with the reason next to the tolerance: method differences in the frequency domain
//!   (interpolation and Welch settings) for records of 5 min or more.
//! - **report only** (printed, not asserted): VLF and frequency metrics of records < 5 min; TINN
//!   (different search); DFA α1/α2, whose definition differs (scales 4–16 / 16–64 beats vs
//!   NeuroKit's 4–11 / 12–N/10, non-overlapping windows, mean of RMS instead of RMS of variance).

use std::collections::BTreeMap;

use fast_hrv::peaks::{
    KubiosMidpoint, KubiosParams, detect_artifacts, detect_artifacts_positions, find_peaks_default,
    fix_peaks_with, fix_positions_with, fix_positions_with_midpoint,
};
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
        "time.sdrmssd" => t.sdrmssd,
        "time.prc20_nn" => t.prc20_nn,
        "time.prc80_nn" => t.prc80_nn,
        "frequency.vlf" => f.vlf,
        "frequency.lf" => f.lf,
        "frequency.hf" => f.hf,
        "frequency.lf_hf" => f.lf_hf,
        "frequency.lnhf" => f.lnhf,
        "frequency.lfnu_frac" => f.lfnu / 100.0,
        "nonlinear.sd1" => n.sd1,
        "nonlinear.sd2" => n.sd2,
        "nonlinear.sd1_sd2" => n.sd1_sd2,
        "nonlinear.csi" => n.csi,
        "nonlinear.cvi" => n.cvi,
        "nonlinear.csi_modified" => n.csi_modified,
        "nonlinear.s" => n.s,
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
        "fragmentation.pip" => r.fragmentation.pip,
        "fragmentation.ials" => r.fragmentation.ials,
        "fragmentation.pss" => r.fragmentation.pss,
        "fragmentation.pas" => r.fragmentation.pas,
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
        "frequency.lnhf" => Class::Tol(
            0.05,
            "ln of HF: the HF method difference (spline + Welch vs NK) shrinks on the log scale",
        ),
        "frequency.lf" | "frequency.hf" | "frequency.lf_hf" | "frequency.lfnu_frac" => Class::Tol(
            0.30,
            "method: 4 Hz cubic spline + Welch 256 samples (fast-hrv) vs 100 Hz quadratic + Welch N/2, nfft 2N (NK)",
        ),
        "time.tinn" => Class::ReportOnly(
            "TINN search differs: NeuroKit's M loop is not reset for each N and its error covers only [N, M]; \
             fast-hrv does the full least-squares search (same 0-based 7.8125 ms histogram, so HTI is strict)",
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
            if matches!(cls, Class::Tol(tol, _) if rel > tol) {
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

/// Corrected-beat counts on the NeuroKit2 Kubios fixtures: flags over all passes
/// (`Artifacts::total_count`) vs beats actually changed (`Corrections`).
#[test]
fn kubios_changed_beats_vs_flags() {
    let Some(r) = load("neurokit_kubios.json") else {
        return;
    };
    let params = KubiosParams::default();
    let (mut flags, mut changed, mut removed, mut beats) = (0, 0, 0, 0);
    eprintln!("case: beats, flags (total_count), moved, inserted, removed, changed_ratio");
    for case in r["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let fs = case["sampling_rate"].as_f64().unwrap();
        let peaks = usizes(&case["peaks"]);
        let out = fix_peaks_with(&peaks, fs, &params).unwrap();
        let c = &out.corrections;
        assert_eq!(out.peaks, usizes(&case["fixed"]), "{name}");
        assert_eq!(c.n_output(), out.peaks.len());
        assert_eq!(
            out.peaks.len(),
            peaks.len() - c.removed_count() + c.inserted_count(),
            "{name}"
        );
        // every change needs a flag; flags that change nothing are not counted
        assert!(
            c.changed_count() + c.removed_count() <= out.artifacts.total_count(),
            "{name}"
        );
        for (i, o) in c.input_to_output.iter().enumerate() {
            if let Some(j) = *o {
                let same = peaks[i] == out.peaks[j];
                assert_eq!(
                    same,
                    c.status[j] == fast_hrv::BeatStatus::Unchanged,
                    "{name}"
                );
            }
        }
        eprintln!(
            "{name}: {}, {}, {}, {}, {}, {:.4}",
            out.peaks.len(),
            out.artifacts.total_count(),
            c.moved_count(),
            c.inserted_count(),
            c.removed_count(),
            out.changed_ratio()
        );
        flags += out.artifacts.total_count();
        changed += c.changed_count();
        removed += c.removed_count();
        beats += out.peaks.len();
    }
    eprintln!("total: {beats} output beats, {flags} flags, {changed} changed + {removed} removed");
}

// ---------------------------------------------------------------------------------------------
// Kubios on f64 positions (`fix_positions_with`)
// ---------------------------------------------------------------------------------------------

/// Deterministic sub-sample offsets in [-0.45, 0.45] (LCG; no RNG dependency).
fn offsets(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.9
        })
        .collect()
}

fn changed_set(c: &fast_hrv::Corrections) -> Vec<usize> {
    (0..c.n_output())
        .filter(|&j| c.status[j] != fast_hrv::BeatStatus::Unchanged)
        .collect()
}

/// Integer-valued positions: `KubiosMidpoint::Floor` is bit-identical to `fix_peaks_with` (and so
/// to NeuroKit2); with the default exact midpoint the per-beat result is the same on these fixtures
/// and corrected beats stay within one sample (0.5 per midpoint; adjacent moved beats can add up).
#[test]
fn kubios_positions_integer_valued_match_integer_path() {
    let Some(r) = load("neurokit_kubios.json") else {
        return;
    };
    let params = KubiosParams::default();
    let mut max_dev: f64 = 0.0;
    for case in r["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let fs = case["sampling_rate"].as_f64().unwrap();
        let peaks = usizes(&case["peaks"]);
        let nk_fixed: Vec<f64> = usizes(&case["fixed"]).iter().map(|&p| p as f64).collect();
        let pos: Vec<f64> = peaks.iter().map(|&p| p as f64).collect();
        let int = fix_peaks_with(&peaks, fs, &params).unwrap();

        assert_eq!(
            detect_artifacts_positions(&pos, fs, &params),
            detect_artifacts(&peaks, fs, &params),
            "{name}"
        );
        let floor = fix_positions_with_midpoint(&pos, fs, &params, KubiosMidpoint::Floor).unwrap();
        assert_eq!(floor.positions, nk_fixed, "{name}: Floor vs NeuroKit2");
        let int_pos: Vec<f64> = int.peaks.iter().map(|&p| p as f64).collect();
        assert!(
            floor
                .positions
                .iter()
                .zip(&int_pos)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{name}"
        );
        assert_eq!(floor.artifacts, int.artifacts, "{name}");
        assert_eq!(floor.passes, int.passes, "{name}");
        assert_eq!(floor.corrections, int.corrections, "{name}");

        let exact = fix_positions_with(&pos, fs, &params).unwrap();
        assert_eq!(exact.corrections, int.corrections, "{name}: exact midpoint");
        assert_eq!(exact.passes, int.passes, "{name}: exact midpoint");
        for (j, (&a, &b)) in exact.positions.iter().zip(&int_pos).enumerate() {
            if exact.corrections.status[j] == fast_hrv::BeatStatus::Unchanged {
                assert_eq!(a.to_bits(), b.to_bits(), "{name}");
            } else {
                assert!((a - b).abs() < 1.0, "{name}: beat {j}: {a} vs {b}");
                max_dev = max_dev.max((a - b).abs());
            }
        }
    }
    eprintln!("exact vs floor midpoint: largest difference {max_dev} samples");
}

/// A constant dyadic sub-sample offset keeps every RR bit-identical, so the result must be the
/// integer-valued result shifted by the offset.
#[test]
fn kubios_positions_constant_subsample_offset() {
    let Some(r) = load("neurokit_kubios.json") else {
        return;
    };
    let params = KubiosParams::default();
    for case in r["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let fs = case["sampling_rate"].as_f64().unwrap();
        let peaks = usizes(&case["peaks"]);
        let pos: Vec<f64> = peaks.iter().map(|&p| p as f64).collect();
        let base = fix_positions_with(&pos, fs, &params).unwrap();
        for delta in [0.25, -0.375, 0.4375] {
            let sub: Vec<f64> = pos.iter().map(|p| p + delta).collect();
            let out = fix_positions_with(&sub, fs, &params).unwrap();
            assert_eq!(out.corrections, base.corrections, "{name} {delta}");
            assert_eq!(out.passes, base.passes, "{name} {delta}");
            let shifted: Vec<f64> = base.positions.iter().map(|p| p + delta).collect();
            assert_eq!(out.positions, shifted, "{name} {delta}");
        }
    }
}

/// Per-beat sub-sample offsets (|δ| ≤ 0.45 samples, as from parabolic refinement) against the
/// integer path on the rounded positions (= the NeuroKit2 fixture input).
///
/// Unchanged beats must keep their sub-sample input positions bit for bit. The changed beats agree
/// with the integer path except where a Kubios decision sits on a boundary that rounding moves:
/// integer RR series contain exact ties (e.g. `|dRR[i+1]| == |dRR[i+2]|` in the long/short candidate
/// rule), which any sub-sample offset breaks (in `none_70` an offset of 1e-9 samples already adds a
/// long/short candidate). Those cases are reported and bounded, not hidden.
#[test]
fn kubios_positions_subsample_vs_rounded() {
    let Some(r) = load("neurokit_kubios.json") else {
        return;
    };
    let params = KubiosParams::default();
    let cases = r["cases"].as_array().unwrap();
    let mut same = 0;
    let mut differing = Vec::new();
    let (mut sum_int, mut sum_sub) = (0usize, 0usize);
    for (ci, case) in cases.iter().enumerate() {
        let name = case["name"].as_str().unwrap();
        let fs = case["sampling_rate"].as_f64().unwrap();
        let peaks = usizes(&case["peaks"]);
        let sub: Vec<f64> = peaks
            .iter()
            .zip(offsets(peaks.len(), ci as u64))
            .map(|(&p, o)| p as f64 + o)
            .collect();
        let rounded: Vec<usize> = sub.iter().map(|p| p.round() as usize).collect();
        assert_eq!(rounded, peaks);
        let int = fix_peaks_with(&rounded, fs, &params).unwrap();
        let out = fix_positions_with(&sub, fs, &params).unwrap();
        let c = &out.corrections;

        // unchanged beats keep their original sub-sample positions
        for (i, o) in c.input_to_output.iter().enumerate() {
            if let Some(j) = *o {
                if c.status[j] == fast_hrv::BeatStatus::Unchanged {
                    assert_eq!(out.positions[j].to_bits(), sub[i].to_bits(), "{name}");
                } else {
                    assert_ne!(out.positions[j], sub[i], "{name}");
                }
            }
        }
        let (a, b) = (changed_set(&int.corrections), changed_set(c));
        sum_int += a.len() + int.corrections.removed_count();
        sum_sub += b.len() + c.removed_count();
        if int.corrections == *c {
            same += 1;
            // same beats changed: corrected positions within a sample of the integer result
            for &j in &b {
                assert!(
                    (out.positions[j] - int.peaks[j] as f64).abs() < 1.0,
                    "{name}"
                );
            }
        } else {
            let sym = a.iter().filter(|j| !b.contains(j)).count()
                + b.iter().filter(|j| !a.contains(j)).count();
            assert!(sym <= 2, "{name}: {a:?} vs {b:?}");
            assert_eq!(c.removed_count(), int.corrections.removed_count(), "{name}");
            differing.push(format!("{name}: integer {a:?}, sub-sample {b:?}"));
        }
    }
    eprintln!(
        "sub-sample vs rounded: identical per-beat result {same}/{}; changed+removed {sum_sub} vs {sum_int}",
        cases.len()
    );
    for d in &differing {
        eprintln!("  {d}");
    }
    assert!(same >= cases.len() - 6, "{same}/{}", cases.len());
}
