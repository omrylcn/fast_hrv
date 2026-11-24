#![allow(clippy::needless_range_loop)]
use std::time::Instant;
fn main() {
    let fs = 512u32;
    let n = 300 * fs as usize;
    // synthetic ECG-ish: narrow gaussian R every ~0.83 s + noise + DC
    let mut x = vec![412.0f64; n];
    let mut t = 0.3f64;
    let mut k = 0u64;
    while t < 299.5 {
        let c = (t * fs as f64) as isize;
        for d in -15isize..=15 {
            let i = c + d;
            if i >= 0 && (i as usize) < n {
                x[i as usize] += 200.0 * (-(d as f64).powi(2) / 18.0).exp();
            }
        }
        k += 1;
        t += 0.83 + 0.03 * ((k as f64) * 0.7).sin();
    }
    for i in 0..n {
        x[i] += 5.0 * ((i as f64) * 0.37).sin()
            + 20.0 * (2.0 * std::f64::consts::PI * 50.0 * i as f64 / fs as f64).sin();
    }
    let reps = 20;
    let t0 = Instant::now();
    let mut np = 0;
    for _ in 0..reps {
        let r = fast_hrv::process_ecg(&x, fs, true);
        np = r.rpeaks.len();
    }
    let pe = t0.elapsed().as_secs_f64() / reps as f64;
    let r = fast_hrv::process_ecg(&x, fs, true);
    let rri = fast_hrv::rri_from_peaks(&r.rpeaks, fs);
    let t1 = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(fast_hrv::hrv(&rri));
    }
    let he = t1.elapsed().as_secs_f64() / reps as f64;
    let t2 = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(fast_hrv::ecg_clean(&x, fs));
    }
    let ce = t2.elapsed().as_secs_f64() / reps as f64;
    // pathological: one 69 s gap RR inside
    let mut bad = rri.clone();
    bad[100] = 69000.0;
    let t3 = Instant::now();
    let _ = fast_hrv::hrv(&bad);
    let be = t3.elapsed().as_secs_f64();
    println!(
        "peaks={np} process_ecg(5min)={:.2} ms  ecg_clean={:.2} ms  hrv({} RR)={:.2} ms  hrv(with 69s gap RR)={:.1} ms",
        pe * 1e3,
        ce * 1e3,
        rri.len(),
        he * 1e3,
        be * 1e3
    );
}
