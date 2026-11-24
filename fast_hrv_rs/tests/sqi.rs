//! Tests for `fast_hrv::sqi`.
//!
//! Parity fixtures (`tests/fixtures/sqi/*.json`) are produced by `tools/sqi_reference.py` from
//! synthetic 10 s, 512 Hz ECG (neurokit2.ecg_simulate + artefacts) with NeuroKit2 / SciPy.
//! Print the comparison table with `cargo test --test sqi -- --nocapture`.

use std::fs;
use std::path::PathBuf;

use fast_hrv::sqi::*;
use serde_json::Value;

const FS: f64 = 512.0;

// Tolerances. Same algorithm, different summation order only → near machine precision.
const TOL_KURT: f64 = 1e-9; // relative
const TOL_SPEC: f64 = 1e-9; // absolute on ratios in [0, 1]
const TOL_CORR: f64 = 1e-9;

struct Case {
    name: String,
    v: Value,
}

impl Case {
    fn raw(&self) -> Vec<i32> {
        self.v["raw"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap() as i32)
            .collect()
    }
    fn signal(&self) -> Vec<f64> {
        match self.v["signal"].as_array() {
            Some(a) => a.iter().map(|x| x.as_f64().unwrap()).collect(),
            None => self.raw().iter().map(|&x| x as f64).collect(),
        }
    }
    fn peaks(&self, key: &str) -> Vec<usize> {
        self.v[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect()
    }
    fn f(&self, path: &[&str]) -> Option<f64> {
        let mut v = &self.v;
        for p in path {
            v = &v[*p];
        }
        v.as_f64()
    }
}

fn load_cases() -> Vec<Case> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sqi");
    let mut cases: Vec<Case> = fs::read_dir(&dir)
        .expect("fixtures dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|p| Case {
            name: p.file_stem().unwrap().to_string_lossy().into_owned(),
            v: serde_json::from_str(&fs::read_to_string(&p).unwrap()).unwrap(),
        })
        .collect();
    cases.sort_by(|a, b| a.name.cmp(&b.name));
    assert!(cases.len() >= 7, "expected fixtures, found {}", cases.len());
    cases
}

fn close(got: f64, want: f64, tol: f64) -> bool {
    (got - want).abs() <= tol
}

// =============================================================================
// Parity with NeuroKit2 / SciPy
// =============================================================================

#[test]
fn fixtures_record_versions_and_cover_required_scenarios() {
    let cases = load_cases();
    let names: Vec<&str> = cases.iter().map(|c| c.name.as_str()).collect();
    for want in [
        "clean",
        "noisy",
        "baseline_wander",
        "flatline",
        "saturated",
        "mains_50hz",
    ] {
        assert!(names.contains(&want), "missing fixture {want}");
    }
    for c in &cases {
        assert!(
            c.v["neurokit_version"].as_str().is_some(),
            "{}: no neurokit_version",
            c.name
        );
        assert_eq!(c.v["fs"].as_f64(), Some(FS));
    }
}

#[test]
fn kurtosis_matches_scipy() {
    println!(
        "\n{:<16} {:>14} {:>14} {:>10}",
        "case", "kSQI rust", "scipy", "rel err"
    );
    for c in load_cases() {
        let x = c.signal();
        let k = ksqi(&x).unwrap();
        let want = c.f(&["ksqi_pearson"]).unwrap();
        let rel = ((k - want) / want).abs();
        println!("{:<16} {:>14.10} {:>14.10} {:>10.2e}", c.name, k, want, rel);
        assert!(rel < TOL_KURT, "{}: pearson {k} vs {want}", c.name);
        // NeuroKit2's kSQI is the same Pearson kurtosis
        let nk = c.f(&["neurokit", "ksqi"]).unwrap();
        assert!(((k - nk) / nk).abs() < TOL_KURT, "{}: nk ksqi", c.name);
        let fisher = kurtosis_fisher(&x).unwrap();
        let want_f = c.f(&["ksqi_fisher"]).unwrap();
        assert!(
            (fisher - want_f).abs() < TOL_KURT * want.abs(),
            "{}: fisher",
            c.name
        );
    }
}

#[test]
fn psqi_bassqi_match_neurokit_zhao2018() {
    let cfg = PsdConfig::neurokit();
    println!(
        "\n{:<16} {:>12} {:>12} {:>9} {:>12} {:>12} {:>9}",
        "case", "pSQI rust", "NK", "err", "basSQI rust", "NK", "err"
    );
    for c in load_cases() {
        let x = c.signal();
        let s = spectral_sqi(&x, FS, &cfg).unwrap();
        let (p, b) = (s.psqi.unwrap(), s.bassqi.unwrap());
        let wp = c.f(&["neurokit", "psqi"]).unwrap();
        let wb = c.f(&["neurokit", "bassqi"]).unwrap();
        println!(
            "{:<16} {:>12.9} {:>12.9} {:>9.1e} {:>12.9} {:>12.9} {:>9.1e}",
            c.name,
            p,
            wp,
            (p - wp).abs(),
            b,
            wb,
            (b - wb).abs()
        );
        assert!(close(p, wp, TOL_SPEC), "{}: pSQI {p} vs NK {wp}", c.name);
        assert!(close(b, wb, TOL_SPEC), "{}: basSQI {b} vs NK {wb}", c.name);
        // single-index helpers agree with the shared-PSD path
        assert_eq!(psqi(&x, FS, &cfg).unwrap(), p);
        assert_eq!(bassqi(&x, FS, &cfg).unwrap(), b);
    }
}

#[test]
fn psqi_bassqi_match_scipy_welch_defaults() {
    for (key, nperseg) in [("welch_2048", 2048usize), ("welch_1024", 1024)] {
        let cfg = PsdConfig::welch(nperseg);
        println!(
            "\n{key}: {:<16} {:>12} {:>12} {:>12} {:>12}",
            "case", "pSQI", "scipy", "basSQI", "scipy"
        );
        for c in load_cases() {
            let x = c.signal();
            let s = spectral_sqi(&x, FS, &cfg).unwrap();
            let (p, b) = (s.psqi.unwrap(), s.bassqi.unwrap());
            let wp = c.f(&[key, "psqi"]).unwrap();
            let wb = c.f(&[key, "bassqi"]).unwrap();
            println!(
                "{key}: {:<16} {:>12.9} {:>12.9} {:>12.9} {:>12.9}",
                c.name, p, wp, b, wb
            );
            assert!(close(p, wp, TOL_SPEC), "{} {key}: pSQI {p} vs {wp}", c.name);
            assert!(
                close(b, wb, TOL_SPEC),
                "{} {key}: basSQI {b} vs {wb}",
                c.name
            );
        }
    }
    assert_eq!(PsdConfig::default(), PsdConfig::welch(2048));
}

#[test]
fn psd_shape_and_scaling() {
    // White-ish deterministic noise: density integrates to the variance (Parseval, Hann-corrected).
    let x: Vec<f64> = (0..5120)
        .map(|i| ((i as f64 * 12.9898).sin() * 43758.5453).fract() - 0.5)
        .collect();
    let p = psd(&x, FS, &PsdConfig::default()).unwrap();
    assert_eq!(p.nperseg, 2048);
    assert_eq!(p.nfft, 2048);
    assert_eq!(p.n_segments, 4);
    assert_eq!(p.freqs.len(), 1025);
    assert_eq!(p.freqs[1], 0.25);
    let var = {
        let m = x.iter().sum::<f64>() / x.len() as f64;
        x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64
    };
    let total: f64 = p.power.iter().sum::<f64>() * 0.25;
    assert!((total / var - 1.0).abs() < 0.05, "total {total} var {var}");

    let nk = psd(&x, FS, &PsdConfig::neurokit()).unwrap();
    assert_eq!(nk.nperseg, 2560); // window=1024 s → capped at N/2
    assert_eq!(nk.nfft, 5120);
    assert!((nk.min_frequency - 0.4).abs() < 1e-12);
}

#[test]
fn zhao_fusion_matches_neurokit_labels() {
    println!(
        "\n{:<16} {:>18} {:>18}  reasons",
        "case", "rust", "NeuroKit2"
    );
    for c in load_cases() {
        let x = c.signal();
        let peaks = c.peaks("peaks_a");
        let s = spectral_sqi(&x, FS, &PsdConfig::neurokit()).unwrap();
        let hr = zhao_heart_rate_bpm(&peaks, FS).unwrap_or(f64::NAN);
        let r = zhao2018_fusion(
            ksqi(&x).unwrap(),
            s.psqi.unwrap(),
            s.bassqi.unwrap(),
            hr,
            &ZhaoThresholds::default(),
        );
        let want = c.v["neurokit"]["zhao_simple"].as_str().unwrap();
        println!(
            "{:<16} {:>18} {:>18}  {:?}",
            c.name,
            r.class.as_neurokit_str(),
            want,
            r.reasons
        );
        assert_eq!(r.class.as_neurokit_str(), want, "{}", c.name);
    }
}

// =============================================================================
// Raw stream checks
// =============================================================================

#[test]
fn raw_checks_match_reference() {
    for c in load_cases() {
        let raw = c.raw();
        let eps = c.f(&["flat", "eps"]).unwrap();
        let min_s = c.f(&["flat", "min_duration_s"]).unwrap();
        let fl = flatline_fraction(&raw, FS, eps, min_s).unwrap();
        assert!(
            close(fl, c.f(&["flatline_fraction"]).unwrap(), 1e-15),
            "{}: flat {fl}",
            c.name
        );
        let (lo, hi, m) = (
            c.f(&["adc", "min"]).unwrap(),
            c.f(&["adc", "max"]).unwrap(),
            c.f(&["adc", "margin"]).unwrap(),
        );
        let sat = saturation_fraction(&raw, lo, hi, m).unwrap();
        assert!(
            close(sat, c.f(&["saturation_fraction"]).unwrap(), 1e-15),
            "{}: sat {sat}",
            c.name
        );
        match c.name.as_str() {
            "flatline" => assert!(fl > 0.35, "4 s lead-off in 10 s, got {fl}"),
            "saturated" => assert!(sat > 0.01),
            _ => {
                if !c.name.contains("saturated") {
                    assert_eq!(sat, 0.0, "{}", c.name)
                }
            }
        }
        if c.name == "clean" {
            assert_eq!(fl, 0.0);
        }
    }
}

#[test]
fn flatline_unit() {
    let mut x = vec![0i16; 0];
    for i in 0..100 {
        x.push((i % 7) as i16);
    }
    x.extend(std::iter::repeat_n(5i16, 50)); // 50 identical samples
    for i in 0..100 {
        x.push((i % 5) as i16);
    }
    let fs = 100.0;
    // 0.5 s at 100 Hz = 50 samples → counted
    assert!(close(
        flatline_fraction(&x, fs, 0.0, 0.5).unwrap(),
        50.0 / 250.0,
        1e-12
    ));
    // 0.6 s → not long enough
    assert_eq!(flatline_fraction(&x, fs, 0.0, 0.6).unwrap(), 0.0);
    // whole signal constant
    assert_eq!(flatline_fraction(&[3u16; 10], fs, 0.0, 0.0).unwrap(), 1.0);
    // float stream with NaN breaks runs, never panics
    let f = [1.0f64, 1.0, f64::NAN, 1.0, 1.0];
    assert_eq!(flatline_fraction(&f, 1.0, 0.0, 2.0).unwrap(), 0.8);
    assert_eq!(
        flatline_fraction::<i32>(&[], fs, 0.0, 0.2),
        Err(SqiError::EmptyInput)
    );
    assert!(matches!(
        flatline_fraction(&x, 0.0, 0.0, 0.2),
        Err(SqiError::InvalidParameter(_))
    ));
    assert!(matches!(
        flatline_fraction(&x, fs, -1.0, 0.2),
        Err(SqiError::InvalidParameter(_))
    ));
    assert!(matches!(
        flatline_fraction(&x, fs, 0.0, f64::NAN),
        Err(SqiError::InvalidParameter(_))
    ));
}

#[test]
fn saturation_unit() {
    let x = [0i32, 1, 2, 2000, 4093, 4094, 4095, 2048];
    assert_eq!(
        saturation_fraction(&x, 0.0, 4095.0, 0.0).unwrap(),
        2.0 / 8.0
    );
    assert_eq!(
        saturation_fraction(&x, 0.0, 4095.0, 2.0).unwrap(),
        6.0 / 8.0
    );
    assert_eq!(
        saturation_fraction::<i32>(&[], 0.0, 1.0, 0.0),
        Err(SqiError::EmptyInput)
    );
    assert!(matches!(
        saturation_fraction(&x, 10.0, 10.0, 0.0),
        Err(SqiError::InvalidParameter(_))
    ));
    assert!(matches!(
        saturation_fraction(&x, 0.0, 10.0, -1.0),
        Err(SqiError::InvalidParameter(_))
    ));
    assert_eq!(
        saturation_fraction(&[f64::NAN, 1.0], 0.0, 1.0, 0.0).unwrap(),
        0.5
    );
}

// =============================================================================
// bSQI
// =============================================================================

#[test]
fn bsqi_matches_reference_fixtures() {
    for c in load_cases() {
        let (a, b) = (c.peaks("peaks_a"), c.peaks("peaks_b"));
        let tol = c.f(&["bsqi_tol_s"]).unwrap();
        assert_eq!(tol, DEFAULT_BSQI_TOLERANCE_S);
        let got = bsqi(&a, &b, FS, tol).unwrap();
        assert!(
            close(got, c.f(&["bsqi"]).unwrap(), 1e-15),
            "{}: bsqi {got}",
            c.name
        );
        let got5 = bsqi(&a, &b, FS, 0.05).unwrap();
        assert!(
            close(got5, c.f(&["bsqi_tol_005"]).unwrap(), 1e-15),
            "{}",
            c.name
        );
    }
}

#[test]
fn bsqi_unit() {
    let a = [100usize, 600, 1100, 1600];
    // identical → 1
    assert_eq!(bsqi(&a, &a, FS, 0.15).unwrap(), 1.0);
    // order and duplicates do not matter
    assert_eq!(
        bsqi(&[1600, 100, 1100, 600, 600], &a, FS, 0.15).unwrap(),
        1.0
    );
    // within tolerance (0.15 s = 76.8 samples)
    let b = [170usize, 530, 1176, 1524];
    let d = bsqi_detail(&a, &b, FS, 0.15).unwrap();
    assert_eq!((d.matched, d.n_a, d.n_b), (4, 4, 4));
    // just outside tolerance for 2 beats: 77 samples > 76.8
    let b = [177usize, 523, 1100, 1600];
    let d = bsqi_detail(&a, &b, FS, 0.15).unwrap();
    assert_eq!(d.matched, 2);
    assert!(close(d.bsqi, 2.0 / 6.0, 1e-15));
    // one extra and one missed beat: matched 3, Na=4, Nb=4 → 3/5
    let b = [100usize, 600, 1100, 1350];
    assert!(close(bsqi(&a, &b, FS, 0.15).unwrap(), 3.0 / 5.0, 1e-15));
    // NeuroKit `ici` style tighter tolerance lowers agreement
    let b = [130usize, 630, 1130, 1630];
    assert_eq!(bsqi(&a, &b, FS, 0.15).unwrap(), 1.0);
    assert_eq!(bsqi(&a, &b, FS, 0.05).unwrap(), 0.0);
    // one-to-one: two B beats near one A beat → only one match
    let d = bsqi_detail(&[1000], &[990, 1010], FS, 0.15).unwrap();
    assert_eq!(d.matched, 1);
    assert!(close(d.bsqi, 0.5, 1e-15));
    // empty
    assert_eq!(bsqi(&a, &[], FS, 0.15).unwrap(), 0.0);
    assert_eq!(bsqi(&[], &[], FS, 0.15), Err(SqiError::TooFewBeats));
    assert!(matches!(
        bsqi(&a, &a, f64::NAN, 0.15),
        Err(SqiError::InvalidParameter(_))
    ));
    assert!(matches!(
        bsqi(&a, &a, FS, -0.1),
        Err(SqiError::InvalidParameter(_))
    ));
}

// =============================================================================
// Template correlation
// =============================================================================

#[test]
fn template_correlation_matches_numpy_reference() {
    for c in load_cases() {
        let Some(want) = c.v["template"].as_object() else {
            continue;
        };
        let x = c.signal();
        let peaks = c.peaks("peaks_a");
        let tc = template_correlation(&x, &peaks, FS, &TemplateConfig::default()).unwrap();
        let want_peaks: Vec<usize> = want["beat_peaks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(tc.beat_peaks, want_peaks, "{}", c.name);
        let want_cc: Vec<f64> = want["per_beat"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        assert_eq!(tc.per_beat.len(), want_cc.len());
        for (g, w) in tc.per_beat.iter().zip(&want_cc) {
            assert!(close(*g, *w, TOL_CORR), "{}: {g} vs {w}", c.name);
        }
        assert!(
            close(tc.median, want["median"].as_f64().unwrap(), TOL_CORR),
            "{}",
            c.name
        );
        assert!(
            close(tc.mean, want["mean"].as_f64().unwrap(), TOL_CORR),
            "{}",
            c.name
        );
        for (g, w) in tc
            .template
            .iter()
            .zip(want["template_head"].as_array().unwrap())
        {
            let w = w.as_f64().unwrap();
            assert!(close(*g, w, 1e-9 * w.abs().max(1.0)), "{}", c.name);
        }
        assert_eq!(tc.template.len(), 306); // int(0.2*512) + int(0.4*512)
        // feeding the window's own template back as "external" gives identical numbers
        let ext =
            template_correlation_with(&x, &peaks, FS, &TemplateConfig::default(), &tc.template)
                .unwrap();
        assert_eq!(ext.per_beat, tc.per_beat);
    }
}

fn synth_beats(n_beats: usize, rr: usize, fs: f64, noise: f64) -> (Vec<f64>, Vec<usize>) {
    let len = n_beats * rr + rr;
    let mut x = vec![0.0; len];
    let mut peaks = Vec::new();
    let mut seed = 1u64;
    let mut rnd = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    for b in 0..n_beats {
        let r = rr / 2 + b * rr;
        peaks.push(r);
        for (i, v) in x.iter_mut().enumerate() {
            let t = (i as f64 - r as f64) / fs;
            *v += (-(t / 0.012).powi(2)).exp() + 0.25 * (-((t - 0.25) / 0.05).powi(2)).exp();
        }
    }
    for v in x.iter_mut() {
        *v += noise * rnd();
    }
    (x, peaks)
}

#[test]
fn template_correlation_unit() {
    let cfg = TemplateConfig::default();
    let (clean, peaks) = synth_beats(10, 400, FS, 0.0);
    let tc = template_correlation(&clean, &peaks, FS, &cfg).unwrap();
    assert_eq!(tc.per_beat.len(), 10);
    assert!(tc.median > 0.999999 && tc.min > 0.999999);

    let (noisy, _) = synth_beats(10, 400, FS, 1.5);
    let tn = template_correlation(&noisy, &peaks, FS, &cfg).unwrap();
    assert!(
        tn.median < tc.median - 0.05,
        "noise should lower correlation: {}",
        tn.median
    );

    // external template: the clean template scores the noisy window
    let ext = template_correlation_with(&noisy, &peaks, FS, &cfg, &tc.template).unwrap();
    assert!(ext.median < 0.99 && ext.median > 0.0);
    assert_eq!(ext.template, tc.template);

    // beats whose window leaves the signal are skipped (peak at 10: before-window < 0)
    let mut p2 = vec![10usize];
    p2.extend(&peaks);
    p2.push(clean.len() - 5);
    let t2 = template_correlation(&clean, &p2, FS, &cfg).unwrap();
    assert_eq!(t2.beat_peaks, peaks);

    // a single constant beat window → NaN for that beat, excluded from stats
    let mut flat = clean.clone();
    let (a, b) = (peaks[3] - 102, peaks[3] + 204);
    flat[a..b].iter_mut().for_each(|v| *v = 0.0);
    let tf = template_correlation(&flat, &peaks, FS, &cfg).unwrap();
    assert!(tf.per_beat[3].is_nan());
    assert!(tf.median.is_finite() && tf.mean.is_finite());
}

#[test]
fn template_edge_cases() {
    let cfg = TemplateConfig::default();
    let (x, peaks) = synth_beats(10, 400, FS, 0.0);
    assert_eq!(
        template_correlation(&[], &peaks, FS, &cfg),
        Err(SqiError::EmptyInput)
    );
    assert_eq!(
        template_correlation(&x, &[], FS, &cfg),
        Err(SqiError::TooFewBeats)
    );
    assert_eq!(
        template_correlation(&x, &peaks[..2], FS, &cfg),
        Err(SqiError::TooFewBeats)
    );
    assert!(template_correlation(&x, &peaks[..3], FS, &cfg).is_ok());
    let mut nan = x.clone();
    nan[7] = f64::NAN;
    assert_eq!(
        template_correlation(&nan, &peaks, FS, &cfg),
        Err(SqiError::NonFinite)
    );
    let constant = vec![1.0; x.len()];
    assert_eq!(
        template_correlation(&constant, &peaks, FS, &cfg),
        Err(SqiError::ZeroVariance)
    );
    // external template checks
    assert_eq!(
        template_correlation_with(&x, &peaks, FS, &cfg, &[0.0; 10]),
        Err(SqiError::TemplateLength)
    );
    let mut t = beat_template(&x, &peaks, FS, &cfg).unwrap();
    assert!(template_correlation_with(&x, &peaks[..1], FS, &cfg, &t).is_ok());
    assert_eq!(
        template_correlation_with(&x, &[], FS, &cfg, &t),
        Err(SqiError::TooFewBeats)
    );
    t[0] = f64::INFINITY;
    assert_eq!(
        template_correlation_with(&x, &peaks, FS, &cfg, &t),
        Err(SqiError::NonFinite)
    );
    // absurd peaks never panic
    assert_eq!(
        template_correlation(&x, &[usize::MAX, usize::MAX - 1, 0], FS, &cfg),
        Err(SqiError::TooFewBeats)
    );
    let bad = TemplateConfig {
        before_s: -1.0,
        ..cfg
    };
    assert!(matches!(
        template_correlation(&x, &peaks, FS, &bad),
        Err(SqiError::InvalidParameter(_))
    ));
}

// =============================================================================
// RR plausibility
// =============================================================================

#[test]
fn rr_plausibility_unit() {
    let lim = RrLimits::default();
    // 75 bpm regular
    let peaks: Vec<usize> = (0..12).map(|i| 200 + i * 410).collect();
    let r = rr_plausibility(&peaks, FS, Some(5120), &lim).unwrap();
    assert!(close(r.mean_hr_bpm, 60.0 * FS / 410.0, 1e-9));
    assert!(close(r.rr_ratio, 1.0, 1e-12));
    assert!(r.plausible && r.hr_ok && r.gap_ok && r.ratio_ok);
    assert_eq!(r.n_beats, 12);

    // missed beat → ratio 2, still < 2.2; two missed in a row → 3 > 2.2
    let mut p: Vec<usize> = peaks.clone();
    p.remove(5);
    let r = rr_plausibility(&p, FS, None, &lim).unwrap();
    assert!(close(r.rr_ratio, 2.0, 1e-12) && r.ratio_ok);
    p.remove(5);
    let r = rr_plausibility(&p, FS, None, &lim).unwrap();
    assert!(!r.ratio_ok && !r.plausible);

    // long gap 3.5 s
    let r = rr_plausibility(&[100, 500, 900, 900 + 1792, 3100], FS, None, &lim).unwrap();
    assert!(close(r.max_rr_s, 3.5, 1e-12) && !r.gap_ok);

    // edge gap: beats only in the first 3 s of a 10 s window
    let r = rr_plausibility(&[100, 500, 900, 1300], FS, Some(5120), &lim).unwrap();
    assert!(close(r.longest_gap_s, (5120 - 1300) as f64 / FS, 1e-12));
    assert!(!r.gap_ok && close(r.max_rr_s, 400.0 / FS, 1e-12));

    // HR out of range (200 bpm) and configurable limits
    let fast: Vec<usize> = (0..30).map(|i| i * 154).collect();
    let r = rr_plausibility(&fast, FS, None, &lim).unwrap();
    assert!(!r.hr_ok);
    let r = rr_plausibility(
        &fast,
        FS,
        None,
        &RrLimits {
            max_hr_bpm: 220.0,
            ..lim
        },
    )
    .unwrap();
    assert!(r.hr_ok);

    // edge cases
    assert_eq!(
        rr_plausibility(&[], FS, None, &lim),
        Err(SqiError::TooFewBeats)
    );
    assert_eq!(
        rr_plausibility(&[5], FS, None, &lim),
        Err(SqiError::TooFewBeats)
    );
    assert_eq!(
        rr_plausibility(&[5, 5, 5], FS, None, &lim),
        Err(SqiError::TooFewBeats)
    );
    assert!(matches!(
        rr_plausibility(&peaks, -1.0, None, &lim),
        Err(SqiError::InvalidParameter(_))
    ));
}

// =============================================================================
// Fusion helper
// =============================================================================

#[test]
fn zhao_fusion_unit() {
    let th = ZhaoThresholds::default();
    let r = zhao2018_fusion(10.0, 0.6, 0.97, 70.0, &th);
    assert_eq!(r.class, ZhaoClass::Excellent);
    assert!(r.reasons.is_empty());

    // pSQI 0.85 at rest: unqualified → 1 unqualified, 2 optimal → barely acceptable
    let r = zhao2018_fusion(10.0, 0.85, 0.97, 70.0, &th);
    assert_eq!(r.class, ZhaoClass::BarelyAcceptable);
    assert_eq!(r.reasons, vec![ZhaoReason::Unqualified(SqiIndex::Psqi)]);
    // pSQI 0.45: optimal in the HR ≥ 130 band (0.4–0.7), suspicious at rest (0.4–0.5)
    let r = zhao2018_fusion(10.0, 0.45, 0.97, 140.0, &th);
    assert_eq!(r.psqi, IndexGrade::Optimal);
    let r = zhao2018_fusion(10.0, 0.45, 0.97, 70.0, &th);
    assert_eq!(r.psqi, IndexGrade::Suspicious);

    // kSQI low + basSQI low → unacceptable, both reported
    let r = zhao2018_fusion(3.0, 0.6, 0.5, 70.0, &th);
    assert_eq!(r.class, ZhaoClass::Unacceptable);
    assert!(r.reasons.contains(&ZhaoReason::Unqualified(SqiIndex::Ksqi)));
    assert!(
        r.reasons
            .contains(&ZhaoReason::Unqualified(SqiIndex::Bassqi))
    );

    // 1 unqualified + 2 suspicious → unacceptable
    let r = zhao2018_fusion(3.0, 0.45, 0.92, 70.0, &th);
    assert_eq!(r.class, ZhaoClass::Unacceptable);

    // NeuroKit strict inequality: pSQI exactly 0.5 is unqualified
    assert_eq!(
        zhao2018_fusion(10.0, 0.5, 0.97, 70.0, &th).psqi,
        IndexGrade::Unqualified
    );
    // basSQI boundaries
    assert_eq!(
        zhao2018_fusion(10.0, 0.6, 0.95, 70.0, &th).bassqi,
        IndexGrade::Optimal
    );
    assert_eq!(
        zhao2018_fusion(10.0, 0.6, 0.90, 70.0, &th).bassqi,
        IndexGrade::Suspicious
    );
    assert_eq!(
        zhao2018_fusion(10.0, 0.6, 0.8999, 70.0, &th).bassqi,
        IndexGrade::Unqualified
    );

    // NaN → not computable (unqualified); NaN HR → low-HR band
    let r = zhao2018_fusion(f64::NAN, 0.6, 0.97, f64::NAN, &th);
    assert_eq!(r.ksqi, IndexGrade::Unqualified);
    assert_eq!(r.reasons, vec![ZhaoReason::NotComputable(SqiIndex::Ksqi)]);
    assert_eq!(r.class, ZhaoClass::BarelyAcceptable);

    // custom thresholds
    let strict = ZhaoThresholds {
        ksqi_min: 12.0,
        ..th
    };
    assert_eq!(
        zhao2018_fusion(10.0, 0.6, 0.97, 70.0, &strict).ksqi,
        IndexGrade::Unqualified
    );

    // HR helper: max instantaneous HR (min RR), as NeuroKit2
    assert!(close(
        zhao_heart_rate_bpm(&[0, 512, 768, 1280], FS).unwrap(),
        120.0,
        1e-12
    ));
    assert_eq!(zhao_heart_rate_bpm(&[5], FS), None);
}

// =============================================================================
// Edge cases for spectral / kurtosis
// =============================================================================

#[test]
fn spectral_and_kurtosis_edge_cases() {
    assert_eq!(ksqi(&[]), Err(SqiError::EmptyInput));
    assert_eq!(ksqi(&[1.0, f64::NAN, 2.0]), Err(SqiError::NonFinite));
    assert_eq!(ksqi(&[4.0; 100]), Err(SqiError::ZeroVariance));
    assert_eq!(ksqi(&[1.0]), Err(SqiError::ZeroVariance));
    let k = kurtosis_pearson(&[0.0, 0.0, 0.0, 1.0]).unwrap();
    assert!(close(
        kurtosis_fisher(&[0.0, 0.0, 0.0, 1.0]).unwrap(),
        k - 3.0,
        1e-15
    ));

    let cfg = PsdConfig::default();
    assert_eq!(psd(&[], FS, &cfg), Err(SqiError::EmptyInput));
    assert_eq!(
        psd(&[1.0, f64::INFINITY, 0.0], FS, &cfg),
        Err(SqiError::NonFinite)
    );
    assert_eq!(psd(&[1.0], FS, &cfg), Err(SqiError::TooShort));
    assert!(matches!(
        psd(&[1.0; 100], 0.0, &cfg),
        Err(SqiError::InvalidParameter(_))
    ));
    assert!(matches!(
        psd(&[1.0; 100], FS, &PsdConfig::welch(1)),
        Err(SqiError::InvalidParameter(_))
    ));
    assert!(matches!(
        psd(&[1.0; 100], FS, &PsdConfig::NeuroKit { window_s: -1.0 }),
        Err(SqiError::InvalidParameter(_))
    ));
    // constant signal → all band powers zero
    let s = spectral_sqi(&[2.0; 5120], FS, &cfg).unwrap();
    assert_eq!(s.psqi, Err(SqiError::ZeroBandPower));
    assert_eq!(s.bassqi, Err(SqiError::ZeroBandPower));
    // too short for the band resolution (64 samples at 512 Hz → 8 Hz bins: 0–1 Hz has one bin)
    let short: Vec<f64> = (0..64).map(|i| (i as f64 * 0.7).sin()).collect();
    assert_eq!(bassqi(&short, FS, &cfg), Err(SqiError::ZeroBandPower));
    // short signals shrink nperseg like scipy (no panic), odd lengths fine
    for n in [3usize, 17, 1001, 2047, 2049, 5119] {
        let x: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.3).sin() + (i as f64 * 0.05).cos())
            .collect();
        let _ = spectral_sqi(&x, FS, &cfg);
        let _ = spectral_sqi(&x, FS, &PsdConfig::neurokit());
    }
}

#[test]
fn kurtosis_known_value() {
    // x = {0,0,0,1}: mean 0.25, m2 = 3/16, m4 = (3·0.25⁴ + 0.75⁴)/4 = 0.08203125
    let k = kurtosis_pearson(&[0.0, 0.0, 0.0, 1.0]).unwrap();
    assert!(close(k, 0.08203125 / (3.0f64 / 16.0).powi(2), 1e-12));
}

// =============================================================================
// Report
// =============================================================================

#[test]
fn report_aggregates_everything() {
    let cases = load_cases();
    let c = cases.iter().find(|c| c.name == "clean").unwrap();
    let x = c.signal();
    let raw = c.raw();
    let (pa, pb) = (c.peaks("peaks_a"), c.peaks("peaks_b"));
    let cfg = SqiConfig {
        adc_range: Some((0.0, 4095.0)),
        psd: PsdConfig::neurokit(),
        ..SqiConfig::default()
    };
    let mut engine = SqiEngine::new(cfg);
    let input = SqiInput {
        peaks_secondary: Some(&pb),
        raw: Some(&raw),
        ..SqiInput::new(&x, FS, &pa)
    };
    let r = engine.report(&input);
    assert!(r.issues.is_empty(), "{:?}", r.issues);
    assert_eq!(r.n_samples, 5120);
    assert_eq!(r.flatline_fraction, Some(0.0));
    assert_eq!(r.saturation_fraction, Some(0.0));
    assert!(close(
        r.ksqi.unwrap(),
        c.f(&["ksqi_pearson"]).unwrap(),
        1e-9
    ));
    assert!(close(
        r.psqi.unwrap(),
        c.f(&["neurokit", "psqi"]).unwrap(),
        TOL_SPEC
    ));
    assert!(close(
        r.bassqi.unwrap(),
        c.f(&["neurokit", "bassqi"]).unwrap(),
        TOL_SPEC
    ));
    assert!(close(r.bsqi.unwrap(), c.f(&["bsqi"]).unwrap(), 1e-15));
    assert!(r.template.unwrap().median > 0.9);
    assert!(r.rr.unwrap().plausible);
    assert_eq!(
        r.zhao.class.as_neurokit_str(),
        c.v["neurokit"]["zhao_simple"].as_str().unwrap()
    );
    // second call reuses the cached plan and is identical
    assert_eq!(engine.report(&input), r);
    assert_eq!(sqi_report(&input, &cfg), r);
}

#[test]
fn report_never_panics_on_garbage() {
    let cfg = SqiConfig::default();
    let empty: [f64; 0] = [];
    let r = sqi_report(&SqiInput::new(&empty, FS, &[]), &cfg);
    assert!(r.ksqi.is_none() && r.psqi.is_none() && r.template.is_none() && r.rr.is_none());
    assert_eq!(r.zhao.class, ZhaoClass::Unacceptable);
    assert!(
        r.issues
            .iter()
            .any(|i| i.field == "ksqi" && i.error == SqiError::EmptyInput)
    );

    let nan = vec![f64::NAN; 5120];
    let peaks = [100usize, 600, 1100];
    let raw = [0i32; 0];
    let r = sqi_report(
        &SqiInput {
            raw: Some(&raw),
            peaks_secondary: Some(&[]),
            ..SqiInput::new(&nan, f64::NAN, &peaks)
        },
        &cfg,
    );
    assert!(r.ksqi.is_none() && r.psqi.is_none() && r.flatline_fraction.is_none());
    assert!(!r.issues.is_empty());
}

#[test]
fn throughput_budget() {
    // 50 windows/s on one core is the target; assert a very loose bound (debug builds are slow).
    let cases = load_cases();
    let c = cases.iter().find(|c| c.name == "noisy").unwrap();
    let x = c.signal();
    let raw = c.raw();
    let (pa, pb) = (c.peaks("peaks_a"), c.peaks("peaks_b"));
    let mut engine = SqiEngine::new(SqiConfig {
        adc_range: Some((0.0, 4095.0)),
        ..SqiConfig::default()
    });
    let input = SqiInput {
        peaks_secondary: Some(&pb),
        raw: Some(&raw),
        ..SqiInput::new(&x, FS, &pa)
    };
    let n = 50;
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        std::hint::black_box(engine.report(std::hint::black_box(&input)));
    }
    let per = t0.elapsed().as_secs_f64() / n as f64;
    println!(
        "\nSqiEngine::report: {:.1} µs/window (10 s @ 512 Hz)",
        per * 1e6
    );
    assert!(per < 0.02, "{per} s per window");
}

#[cfg(feature = "serde")]
#[test]
fn report_serializes() {
    let (x, peaks) = synth_beats(10, 400, FS, 0.05);
    let r = sqi_report(&SqiInput::new(&x, FS, &peaks), &SqiConfig::default());
    let json = serde_json::to_value(&r).unwrap();
    assert!(json["ksqi"].as_f64().is_some());
    assert!(json["zhao"]["class"].as_str().is_some());
    assert!(json["rr"]["plausible"].as_bool().is_some());
}
