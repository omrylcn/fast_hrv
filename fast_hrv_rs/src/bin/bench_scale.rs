//! Scaling benchmark for fast_hrv_rs

use fast_hrv_rs::*;
use std::time::Instant;

fn generate_ecg_signal(n: usize, sampling_rate: u32, seed: u64) -> (Vec<f64>, Vec<usize>) {
    let mut signal = vec![0.0; n];
    let mut peaks = Vec::new();
    let samples_per_beat = (sampling_rate as f64 * 0.8) as usize;
    let mut state = seed;
    let mut pos = samples_per_beat;
    while pos < n - 50 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let jitter = ((state >> 16) & 0x3f) as isize - 32;
        let actual_pos = (pos as isize + jitter).max(0) as usize;
        if actual_pos < n {
            peaks.push(actual_pos);
            if actual_pos >= 5 && actual_pos + 10 < n {
                signal[actual_pos - 5] = -0.1;
                signal[actual_pos - 3] = 0.2;
                signal[actual_pos] = 1.0;
                signal[actual_pos + 3] = -0.3;
                signal[actual_pos + 8] = 0.1;
            }
        }
        pos += samples_per_beat;
    }
    for i in 0..n {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let noise = ((state >> 16) & 0xff) as f64 / 2560.0 - 0.05;
        signal[i] += noise;
    }
    (signal, peaks)
}

fn main() {
    println!("Duration | Samples   | find_peaks | fix_peaks  | process_ecg | µs/sample");
    println!("---------|-----------|------------|------------|-------------|----------");
    
    for &duration in &[5, 30, 60, 120, 180, 360] {
        let sr = 512u32;
        let n = duration * sr as usize;
        let (ecg, peaks) = generate_ecg_signal(n, sr, 42);
        let iterations = if duration <= 60 { 20 } else { 5 };
        
        // find_peaks
        let start = Instant::now();
        for _ in 0..iterations { std::hint::black_box(find_peaks_default(&ecg, sr)); }
        let find_us = start.elapsed().as_micros() as f64 / iterations as f64;
        
        // fix_peaks
        let start = Instant::now();
        for _ in 0..iterations { std::hint::black_box(fix_peaks_default(&peaks, sr)); }
        let fix_us = start.elapsed().as_micros() as f64 / iterations as f64;
        
        // process_ecg
        let start = Instant::now();
        for _ in 0..iterations { std::hint::black_box(process_ecg(&ecg, sr, true)); }
        let proc_us = start.elapsed().as_micros() as f64 / iterations as f64;
        
        let us_per_sample = proc_us / n as f64;
        
        println!("{:>6}s  | {:>9} | {:>8.0} µs | {:>8.0} µs | {:>9.0} µs | {:>8.4}", 
                 duration, n, find_us, fix_us, proc_us, us_per_sample);
    }
}
