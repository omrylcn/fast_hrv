use fast_hrv_rs::*;
use std::time::Instant;

fn generate_ecg(n: usize, sr: u32, seed: u64) -> (Vec<f64>, Vec<usize>) {
    let mut signal = vec![0.0; n];
    let mut peaks = Vec::new();
    let samples_per_beat = (sr as f64 * 0.8) as usize;
    let mut state = seed;
    let mut pos = samples_per_beat;
    while pos < n - 50 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let jitter = ((state >> 16) & 0x3f) as isize - 32;
        let actual_pos = (pos as isize + jitter).max(0) as usize;
        if actual_pos < n && actual_pos >= 5 && actual_pos + 10 < n {
            peaks.push(actual_pos);
            signal[actual_pos] = 1.0;
            signal[actual_pos - 3] = 0.2;
            signal[actual_pos + 3] = -0.3;
        }
        pos += samples_per_beat;
    }
    for i in 0..n {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        signal[i] += ((state >> 16) & 0xff) as f64 / 2560.0 - 0.05;
    }
    (signal, peaks)
}

fn bench<F: FnMut()>(iter: usize, mut f: F) -> f64 {
    for _ in 0..3 { f(); }
    let start = Instant::now();
    for _ in 0..iter { f(); }
    start.elapsed().as_micros() as f64 / iter as f64
}

fn main() {
    let sr = 512u32;
    
    for &duration in &[60, 360] {
        let n = duration * sr as usize;
        let (ecg, peaks) = generate_ecg(n, sr, 42);
        let iter = if duration <= 60 { 20 } else { 10 };
        
        println!("╔══════════════════════════════════════════════════════════╗");
        println!("║  {} seconds ECG ({} samples)                         ║", duration, n);
        println!("╠══════════════════════════════════════════════════════════╣");
        println!("║ Function               │ Rust (µs)  │ For Python compare ║");
        println!("╟────────────────────────┼────────────┼────────────────────╢");
        
        let t = bench(iter, || { std::hint::black_box(ecg_clean(&ecg, sr)); });
        println!("║ ecg_clean              │ {:>10.0} │                    ║", t);
        
        let cleaned = ecg_clean(&ecg, sr);
        
        let t = bench(iter, || { std::hint::black_box(find_peaks_default(&cleaned, sr)); });
        println!("║ find_peaks             │ {:>10.0} │                    ║", t);
        
        let t = bench(iter, || { std::hint::black_box(fix_peaks_default(&peaks, sr)); });
        println!("║ fix_peaks              │ {:>10.0} │                    ║", t);
        
        let t = bench(iter, || { std::hint::black_box(ecg_peaks(&cleaned, sr, true)); });
        println!("║ ecg_peaks (find+fix)   │ {:>10.0} │                    ║", t);
        
        let t = bench(iter, || { std::hint::black_box(compute_quality(&cleaned, &peaks, sr)); });
        println!("║ compute_quality        │ {:>10.0} │                    ║", t);
        
        let t = bench(iter, || { std::hint::black_box(compute_stats(&peaks, &[], sr)); });
        println!("║ compute_stats          │ {:>10.0} │                    ║", t);
        
        let t = bench(iter, || { std::hint::black_box(process_ecg(&ecg, sr, true)); });
        println!("║ process_ecg (TOTAL)    │ {:>10.0} │                    ║", t);
        
        println!("╚══════════════════════════════════════════════════════════╝");
        println!();
    }
}
