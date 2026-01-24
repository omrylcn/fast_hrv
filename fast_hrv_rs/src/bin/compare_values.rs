//! Compare Rust values with Python

use fast_hrv_rs::*;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    // Read signal from file
    let file = File::open("/tmp/test_ecg_signal.txt").expect("Signal file not found");
    let reader = BufReader::new(file);

    let signal: Vec<f64> = reader
        .lines()
        .filter_map(|line| line.ok())
        .filter_map(|line| line.trim().parse::<f64>().ok())
        .collect();

    let sr = 512u32;

    println!("3. Rust Implementation");
    println!("{}", "-".repeat(40));
    println!("  Signal loaded: {} samples", signal.len());

    // Process ECG
    let result = process_ecg(&signal, sr, true);

    let cleaned_min = result.ecg_cleaned.iter().cloned().fold(f64::INFINITY, f64::min);
    let cleaned_max = result.ecg_cleaned.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("  Cleaned signal range: [{:.6}, {:.6}]", cleaned_min, cleaned_max);
    println!("  R-peaks detected: {}", result.rpeaks.len());
    println!("  Mean HR: {:.2} bpm", result.stats.mean_hr_bpm);
    println!("  SDNN: {:.2} ms", result.stats.sdnn_ms);
    println!("  RMSSD: {:.2} ms", result.stats.rmssd_ms);
    println!("  pNN50: {:.2} %", result.stats.pnn50);

    if result.rpeaks.len() >= 5 {
        println!("  First 5 peaks: {:?}", &result.rpeaks[..5]);
    }
    println!();

    // Read Python results for comparison
    let py_json = std::fs::read_to_string("/tmp/python_results.json")
        .expect("Python results not found");
    let py: serde_json::Value = serde_json::from_str(&py_json).unwrap_or_default();

    println!("{}", "=".repeat(70));
    println!("COMPARISON (vs NeuroKit2)");
    println!("{}", "=".repeat(70));

    let nk_hr = py["nk_mean_hr"].as_f64().unwrap_or(0.0);
    let nk_sdnn = py["nk_sdnn"].as_f64().unwrap_or(0.0);
    let nk_rmssd = py["nk_rmssd"].as_f64().unwrap_or(0.0);
    let nk_pnn50 = py["nk_pnn50"].as_f64().unwrap_or(0.0);
    let nk_peaks = py["nk_n_peaks_fixed"].as_u64().unwrap_or(0) as usize;

    let fh_hr = py["fh_mean_hr"].as_f64().unwrap_or(0.0);
    let fh_sdnn = py["fh_sdnn"].as_f64().unwrap_or(0.0);
    let fh_rmssd = py["fh_rmssd"].as_f64().unwrap_or(0.0);
    let fh_pnn50 = py["fh_pnn50"].as_f64().unwrap_or(0.0);
    let fh_peaks = py["fh_n_peaks"].as_u64().unwrap_or(0) as usize;

    println!();
    println!("{:15} {:>12} {:>12} {:>12}", "Metric", "NeuroKit2", "fast_hrv", "Rust");
    println!("{}", "-".repeat(55));
    println!("{:15} {:>12} {:>12} {:>12}", "N Peaks", nk_peaks, fh_peaks, result.rpeaks.len());
    println!("{:15} {:>12.2} {:>12.2} {:>12.2}", "Mean HR (bpm)", nk_hr, fh_hr, result.stats.mean_hr_bpm);
    println!("{:15} {:>12.2} {:>12.2} {:>12.2}", "SDNN (ms)", nk_sdnn, fh_sdnn, result.stats.sdnn_ms);
    println!("{:15} {:>12.2} {:>12.2} {:>12.2}", "RMSSD (ms)", nk_rmssd, fh_rmssd, result.stats.rmssd_ms);
    println!("{:15} {:>12.2} {:>12.2} {:>12.2}", "pNN50 (%)", nk_pnn50, fh_pnn50, result.stats.pnn50);
    println!();

    // Difference from NeuroKit
    println!("Difference from NeuroKit2:");
    println!("{:15} {:>12} {:>12}", "", "fast_hrv", "Rust");
    println!("{}", "-".repeat(40));

    let hr_diff_fh = ((fh_hr - nk_hr) / nk_hr * 100.0).abs();
    let hr_diff_rs = ((result.stats.mean_hr_bpm - nk_hr) / nk_hr * 100.0).abs();
    println!("{:15} {:>11.2}% {:>11.2}%", "Mean HR", hr_diff_fh, hr_diff_rs);

    let sdnn_diff_fh = ((fh_sdnn - nk_sdnn) / nk_sdnn * 100.0).abs();
    let sdnn_diff_rs = ((result.stats.sdnn_ms - nk_sdnn) / nk_sdnn * 100.0).abs();
    println!("{:15} {:>11.2}% {:>11.2}%", "SDNN", sdnn_diff_fh, sdnn_diff_rs);

    let rmssd_diff_fh = ((fh_rmssd - nk_rmssd) / nk_rmssd * 100.0).abs();
    let rmssd_diff_rs = ((result.stats.rmssd_ms - nk_rmssd) / nk_rmssd * 100.0).abs();
    println!("{:15} {:>11.2}% {:>11.2}%", "RMSSD", rmssd_diff_fh, rmssd_diff_rs);

    println!();
}
