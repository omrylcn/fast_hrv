//! `window_report`: per-metric verdicts from window length, corrections and breathing.
use fast_hrv::{METRICS, Verdict, WindowContext, hrv, window_report, window_report_rri};

/// RR series (ms) of `seconds` with respiratory modulation (15/min) and a little noise.
fn rr(seconds: f64) -> Vec<f64> {
    let (mut t, mut out, mut st) = (0.0, Vec::new(), 11u64);
    while t < seconds {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let noise = ((st >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 20.0;
        let v = 850.0 + 40.0 * (2.0 * std::f64::consts::PI * 0.25 * t).sin() + noise;
        out.push(v);
        t += v / 1000.0;
    }
    out
}

#[test]
fn every_catalogue_metric_has_a_value_accessor() {
    let r = hrv(&rr(300.0));
    for m in METRICS {
        assert!(r.value(m.name).is_some(), "{} has no accessor", m.name);
    }
    assert!(r.value("nope").is_none());
}

#[test]
fn length_rules() {
    let x = rr(60.0);
    let w = window_report_rri(&x, &[], None);
    assert_eq!(w.get("rmssd").unwrap().verdict, Verdict::Valid);
    let hf = w.get("hf").unwrap();
    assert_eq!(hf.verdict, Verdict::Warning, "{hf:?}");
    assert!(hf.reasons.contains(&"below_strict_length"));
    for name in ["lf", "lf_hf", "dfa_alpha1", "sampen"] {
        let m = w.get(name).unwrap();
        assert_eq!(m.verdict, Verdict::Suppressed, "{m:?}");
        assert!(m.reasons.contains(&"too_short"), "{m:?}");
    }
    assert!(w.shown("rmssd").is_some() && w.shown("lf").is_none());
    // 5 min: the same metrics become valid
    let w = window_report_rri(&rr(300.0), &[], Some(15.0));
    for name in ["rmssd", "sdnn", "hf", "lf", "lf_hf", "dfa_alpha1", "sampen"] {
        assert_eq!(w.get(name).unwrap().verdict, Verdict::Valid, "{name}");
    }
    // DFA α2 needs 20 min
    assert_eq!(w.get("dfa_alpha2").unwrap().verdict, Verdict::Suppressed);
}

#[test]
fn correction_rules() {
    let x = rr(300.0);
    let n = x.len();
    // 4 % corrected, spread out: DFA α1 (limit 3 %) suppressed, RMSSD (5 %) not
    let flags: Vec<bool> = (0..n).map(|i| i % 25 == 0).collect();
    let w = window_report_rri(&x, &flags, None);
    assert!((w.artifact_ratio - 0.04).abs() < 0.01);
    assert_eq!(w.get("dfa_alpha1").unwrap().verdict, Verdict::Suppressed);
    assert_eq!(w.get("rmssd").unwrap().verdict, Verdict::Valid);
    // 6 %: suppressed everywhere
    let flags: Vec<bool> = (0..n).map(|i| i % 16 == 0).collect();
    let w = window_report_rri(&x, &flags, None);
    assert!(
        w.get("rmssd")
            .unwrap()
            .reasons
            .contains(&"too_many_corrections")
    );
    // two in a row: beat-to-beat metrics get a warning, SDNN not
    let mut flags = vec![false; n];
    flags[100] = true;
    flags[101] = true;
    let w = window_report_rri(&x, &flags, None);
    let r = w.get("rmssd").unwrap();
    assert_eq!(r.verdict, Verdict::Warning);
    assert!(r.reasons.contains(&"consecutive_corrections"));
    assert_eq!(w.get("sdnn").unwrap().verdict, Verdict::Valid);
}

#[test]
fn breathing_rules() {
    let x = rr(300.0);
    let ctx = |bpm: f64| WindowContext::from_flags(&x, &[], Some(bpm));
    let r = hrv(&x);
    let slow = window_report(&r, &ctx(7.0));
    assert_eq!(slow.get("lf_hf").unwrap().verdict, Verdict::Suppressed);
    assert_eq!(slow.get("hf").unwrap().verdict, Verdict::Warning);
    assert!(
        slow.get("hf")
            .unwrap()
            .reasons
            .contains(&"breathing_in_lf_band")
    );
    assert_eq!(slow.get("rmssd").unwrap().verdict, Verdict::Valid);
    let fast = window_report(&r, &ctx(30.0));
    assert!(
        fast.get("hf")
            .unwrap()
            .reasons
            .contains(&"breathing_above_hf_band")
    );
    let normal = window_report(&r, &ctx(15.0));
    assert_eq!(normal.get("lf_hf").unwrap().verdict, Verdict::Valid);
}
