//! Times the parts of `ecg_clean` on a 60 min synthetic signal (dev-time profiling).
use fast_hrv::filter::{BandType, butter_sos, filtfilt_ba, sosfiltfilt};
use std::time::Instant;

fn med(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2] * 1000.0
}

fn main() {
    let fs = 512.0;
    let n = (60.0 * 60.0 * fs) as usize;
    let x: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            (t * 7.0).sin() * 300.0 + (t * 0.2).sin() * 800.0 + ((i * 7919) % 97) as f64
        })
        .collect();
    let sos = butter_sos(5, &[0.5], BandType::Highpass, fs).unwrap();
    let (mut a, mut b, mut c) = (vec![], vec![], vec![]);
    let mut out = vec![];
    for _ in 0..7 {
        let t = Instant::now();
        let hp = sosfiltfilt(&sos, &x);
        a.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        let y = filtfilt_ba(&[1.0; 10], &[10.0], &hp);
        b.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        out = fast_hrv::ecg_clean(&x, 512);
        c.push(t.elapsed().as_secs_f64());
        assert_eq!(y.len(), n);
    }
    println!(
        "n={n} sosfiltfilt {:.1} ms  filtfilt_ba(MA10) {:.1} ms  ecg_clean {:.1} ms  (check {:.3})",
        med(a),
        med(b),
        med(c),
        out[n / 2]
    );
}
