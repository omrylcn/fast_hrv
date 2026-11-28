//! The crate README's window-report example: a 60 s window with two corrected beats in a row and
//! slow breathing (7 /min), and what may be shown from it.
//! `cargo run --release --example window_report`
use fast_hrv::{Preset, Verdict, hrv_with_preset, window_report_rri};

fn main() {
    // 60 s of RR intervals (ms) with a 7 /min breathing rhythm; synthetic, deterministic
    let (mut rri, mut t) = (Vec::new(), 0.0);
    while t < 60.0 {
        let v = 850.0 + 45.0 * (2.0 * std::f64::consts::PI * (7.0 / 60.0) * t).sin();
        rri.push(v);
        t += v / 1000.0;
    }
    // beats 30 and 31 were corrected by the artifact step
    let corrected: Vec<bool> = (0..rri.len()).map(|i| i == 30 || i == 31).collect();

    let w = window_report_rri(&rri, &corrected, Some(7.0));
    println!(
        "window {:.0} s, artifact ratio {:.1} %",
        w.context.duration_s,
        100.0 * w.artifact_ratio
    );
    for name in ["mean_hr", "rmssd", "sdnn", "hf", "lf_hf", "sampen"] {
        let m = w.get(name).unwrap();
        let verdict = match m.verdict {
            Verdict::Valid => "valid",
            Verdict::Warning => "warning",
            Verdict::Suppressed => "suppressed",
        };
        println!(
            "{:<8} {:>8.2}  {:<10} {}",
            m.name,
            m.value,
            verdict,
            m.reasons.join(", ")
        );
    }

    // the same intervals through a named preset, with a record of how they were computed
    let (_report, prov) = hrv_with_preset(&rri, Preset::KubiosLike);
    println!("{prov:?}");
}
