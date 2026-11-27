//! The root README's Rust quick start, compiled so the README cannot drift from the API.
//! `cargo run --release --example readme_quickstart`
use fast_hrv::*;

fn main() {
    // 60 s of a synthetic 75 bpm ECG at 512 Hz (stands in for a recording)
    let fs = 512u32;
    let ecg_signal: Vec<f64> = (0..60 * fs as usize)
        .map(|i| {
            let t = i as f64 / fs as f64;
            let phase = (t - 0.5).rem_euclid(0.8);
            1000.0 * (-((phase.min(0.8 - phase)) / 0.012).powi(2)).exp()
        })
        .collect();

    let result = process_ecg(&ecg_signal, fs, true);
    println!("R-peaks: {}", result.rpeaks.len());
    println!("RMSSD: {:.2} ms", result.stats.rmssd_ms);

    // HRV metrics from the peaks
    let rr = rri_from_peaks(&result.rpeaks, fs);
    let report = hrv(&rr);
    println!(
        "SDNN {:.1} ms  RMSSD {:.1} ms",
        report.time.sdnn, report.time.rmssd
    );
}
