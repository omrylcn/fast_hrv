//! Presets: Task Force = plain `hrv`, NeuroKit = NeuroKit-compatible frequency domain, Kubios-like
//! detrends slow trends but keeps the mean level; provenance records the choice.
use fast_hrv::{Preset, frequency_domain_neurokit, hrv, hrv_with_preset};

fn rr(seconds: f64, trend: bool) -> Vec<f64> {
    let (mut t, mut out) = (0.0, Vec::new());
    while t < seconds {
        let slow = if trend {
            60.0 * (t / seconds - 0.5)
        } else {
            0.0
        };
        let v = 850.0
            + 35.0 * (2.0 * std::f64::consts::PI * 0.25 * t).sin()
            + 15.0 * (2.0 * std::f64::consts::PI * 0.1 * t).sin()
            + slow;
        out.push(v);
        t += v / 1000.0;
    }
    out
}

#[test]
fn task_force_is_plain_hrv() {
    let x = rr(300.0, false);
    let (r, p) = hrv_with_preset(&x, Preset::TaskForce);
    let plain = hrv(&x);
    assert_eq!(r.time.rmssd, plain.time.rmssd);
    assert_eq!(r.frequency.lf_hf, plain.frequency.lf_hf);
    assert_eq!(p.preset, "task_force");
    assert_eq!(p.detrend_lambda, None);
    assert_eq!(p.version, env!("CARGO_PKG_VERSION"));
    assert_eq!(p.n_intervals, x.len());
    assert!((p.duration_s - x.iter().sum::<f64>() / 1000.0).abs() < 1e-9);
}

#[test]
fn neurokit_uses_the_neurokit_frequency_domain() {
    let x = rr(300.0, false);
    let (r, p) = hrv_with_preset(&x, Preset::NeuroKit);
    assert_eq!(r.frequency.lf_hf, frequency_domain_neurokit(&x).lf_hf);
    assert_eq!(p.preset, "neurokit");
}

#[test]
fn kubios_like_detrends_but_keeps_the_level() {
    let x = rr(300.0, true);
    let (k, p) = hrv_with_preset(&x, Preset::KubiosLike);
    let plain = hrv(&x);
    assert_eq!(p.detrend_lambda, Some(500.0));
    // the slow trend inflates SDNN; detrending removes it
    assert!(
        k.time.sdnn < plain.time.sdnn,
        "{} vs {}",
        k.time.sdnn,
        plain.time.sdnn
    );
    // the level is the original one
    assert_eq!(k.time.mean_nn, plain.time.mean_nn);
    assert_eq!(k.time.mean_hr, plain.time.mean_hr);
    // beat-to-beat variation is hardly touched
    assert!((k.time.rmssd / plain.time.rmssd - 1.0).abs() < 0.05);
    assert!(k.frequency.lf_hf.is_finite() && k.frequency.lf_hf != plain.frequency.lf_hf);
}
