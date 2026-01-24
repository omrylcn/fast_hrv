use fast_hrv_rs::*;
use std::time::Instant;

fn generate_ecg(n: usize, sr: u32) -> Vec<f64> {
    let mut signal = vec![0.0; n];
    let mut state = 42u64;
    let mut pos = (sr as f64 * 0.8) as usize;
    while pos < n - 50 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let jitter = ((state >> 16) & 0x3f) as isize - 32;
        let actual_pos = (pos as isize + jitter).max(0) as usize;
        if actual_pos < n && actual_pos >= 5 && actual_pos + 10 < n {
            signal[actual_pos] = 1.0;
            signal[actual_pos - 3] = 0.2;
            signal[actual_pos + 3] = -0.3;
        }
        pos += (sr as f64 * 0.8) as usize;
    }
    for i in 0..n {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        signal[i] += ((state >> 16) & 0xff) as f64 / 2560.0 - 0.05;
    }
    signal
}

fn main() {
    let sr = 512u32;
    
    println!("Duration │ Samples   │ process_ecg │ µs/sample │ vs 60s");
    println!("─────────┼───────────┼─────────────┼───────────┼────────");
    
    let mut base_rate = 0.0;
    
    for &duration in &[60, 120, 180, 360, 720] {
        let n = duration * sr as usize;
        let ecg = generate_ecg(n, sr);
        let iter = if duration <= 120 { 10 } else if duration <= 360 { 5 } else { 3 };
        
        // Warmup
        for _ in 0..2 { std::hint::black_box(process_ecg(&ecg, sr, true)); }
        
        let start = Instant::now();
        for _ in 0..iter { std::hint::black_box(process_ecg(&ecg, sr, true)); }
        let us = start.elapsed().as_micros() as f64 / iter as f64;
        
        let rate = us / n as f64;
        if duration == 60 { base_rate = rate; }
        let ratio = rate / base_rate;
        
        println!("{:>6}s  │ {:>9} │ {:>9.0} µs │ {:>9.4} │ {:.2}x", 
                 duration, n, us, rate, ratio);
    }
}
