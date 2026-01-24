//! Benchmark for fast_hrv_rs

use fast_hrv_rs::*;
use std::time::Instant;

fn generate_rri_data(n: usize, seed: u64) -> Vec<f64> {
    // Simple LCG for reproducible random numbers
    let mut state = seed;
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let rand = ((state >> 16) & 0x7fff) as f64 / 32768.0;
        data.push(800.0 + (rand - 0.5) * 100.0);
    }
    data
}

fn generate_ecg_signal(n: usize, sampling_rate: u32, seed: u64) -> (Vec<f64>, Vec<usize>) {
    let mut signal = vec![0.0; n];
    let mut peaks = Vec::new();

    // Generate R-peaks at ~75 BPM with slight variability
    let samples_per_beat = (sampling_rate as f64 * 0.8) as usize; // ~75 BPM
    let mut state = seed;

    let mut pos = samples_per_beat;
    while pos < n - 50 {
        // Add some variability
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let jitter = ((state >> 16) & 0x3f) as isize - 32; // ±32 samples
        let actual_pos = (pos as isize + jitter).max(0) as usize;

        if actual_pos < n {
            peaks.push(actual_pos);

            // Create QRS complex
            if actual_pos >= 5 && actual_pos + 10 < n {
                signal[actual_pos - 5] = -0.1;
                signal[actual_pos - 3] = 0.2;
                signal[actual_pos] = 1.0;  // R peak
                signal[actual_pos + 3] = -0.3;
                signal[actual_pos + 8] = 0.1;
            }
        }
        pos += samples_per_beat;
    }

    // Add some noise
    for i in 0..n {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let noise = ((state >> 16) & 0xff) as f64 / 2560.0 - 0.05;
        signal[i] += noise;
    }

    (signal, peaks)
}

fn benchmark<F>(name: &str, iterations: usize, mut f: F) -> f64
where
    F: FnMut(),
{
    // Warmup
    for _ in 0..10 {
        f();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let elapsed = start.elapsed();
    let avg_us = elapsed.as_micros() as f64 / iterations as f64;
    println!("{:30} {:>10.2} µs", name, avg_us);
    avg_us
}

fn main() {
    println!("============================================================");
    println!("              fast_hrv_rs Benchmark");
    println!("============================================================");
    println!();

    // ========== HRV METRICS ==========
    println!("==================== HRV METRICS ====================");
    println!();

    for &n in &[100, 300, 1000] {
        let rri = generate_rri_data(n, 42);
        let iterations = if n <= 100 { 10000 } else if n <= 300 { 1000 } else { 100 };

        println!("--- {} RR intervals ({} iterations) ---", n, iterations);

        benchmark("mean", iterations, || {
            std::hint::black_box(mean(&rri));
        });

        benchmark("std_ddof", iterations, || {
            std::hint::black_box(std_ddof(&rri, 1));
        });

        benchmark("compute_rmssd", iterations, || {
            std::hint::black_box(compute_rmssd(&rri));
        });

        benchmark("compute_pnn50", iterations, || {
            std::hint::black_box(compute_pnn50(&rri));
        });

        benchmark("compute_poincare", iterations, || {
            std::hint::black_box(compute_poincare(&rri));
        });

        benchmark("compute_asymmetry", iterations, || {
            std::hint::black_box(compute_asymmetry(&rri));
        });

        benchmark("compute_sampen", iterations, || {
            std::hint::black_box(compute_sampen(&rri, 2, None));
        });

        benchmark("compute_dfa", iterations, || {
            std::hint::black_box(compute_dfa(&rri, (4, 11), (12, None)));
        });

        println!();
    }

    // ========== PEAK DETECTION ==========
    println!("==================== PEAK DETECTION ====================");
    println!();

    for &duration_sec in &[5, 30, 60] {
        let sampling_rate = 512u32;
        let n = duration_sec * sampling_rate as usize;
        let (ecg_signal, peaks) = generate_ecg_signal(n, sampling_rate, 42);
        let iterations = if duration_sec <= 5 { 1000 } else if duration_sec <= 30 { 100 } else { 50 };

        println!("--- {} sec ECG ({} samples, {} iterations) ---", duration_sec, n, iterations);

        benchmark("find_peaks", iterations, || {
            std::hint::black_box(find_peaks_default(&ecg_signal, sampling_rate));
        });

        benchmark("fix_peaks", iterations, || {
            std::hint::black_box(fix_peaks_default(&peaks, sampling_rate));
        });

        benchmark("ecg_peaks (detect+fix)", iterations, || {
            std::hint::black_box(ecg_peaks(&ecg_signal, sampling_rate, true));
        });

        println!();
    }

    // ========== ECG PROCESSING ==========
    println!("==================== ECG PROCESSING ====================");
    println!();

    for &duration_sec in &[5, 30, 60] {
        let sampling_rate = 512u32;
        let n = duration_sec * sampling_rate as usize;
        let (ecg_signal, peaks) = generate_ecg_signal(n, sampling_rate, 42);
        let iterations = if duration_sec <= 5 { 500 } else if duration_sec <= 30 { 50 } else { 20 };

        println!("--- {} sec ECG ({} samples, {} iterations) ---", duration_sec, n, iterations);

        benchmark("ecg_clean", iterations, || {
            std::hint::black_box(ecg_clean(&ecg_signal, sampling_rate));
        });

        let cleaned = ecg_clean(&ecg_signal, sampling_rate);
        benchmark("compute_quality", iterations, || {
            std::hint::black_box(compute_quality(&cleaned, &peaks, sampling_rate));
        });

        benchmark("compute_stats", iterations, || {
            std::hint::black_box(compute_stats(&peaks, &[], sampling_rate));
        });

        benchmark("process_ecg (full pipeline)", iterations, || {
            std::hint::black_box(process_ecg(&ecg_signal, sampling_rate, true));
        });

        println!();
    }

    // ========== SIGNAL PROCESSING ==========
    println!("==================== SIGNAL PROCESSING ====================");
    println!();

    for &n in &[1000, 5000, 10000] {
        let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).sin()).collect();
        let iterations = if n <= 1000 { 1000 } else if n <= 5000 { 200 } else { 100 };

        println!("--- {} samples ({} iterations) ---", n, iterations);

        benchmark("gradient", iterations, || {
            std::hint::black_box(peaks::gradient(&signal));
        });

        benchmark("uniform_filter1d (size=11)", iterations, || {
            std::hint::black_box(peaks::uniform_filter1d(&signal, 11));
        });

        benchmark("median_filter (size=11)", iterations, || {
            std::hint::black_box(peaks::median_filter(&signal, 11));
        });

        benchmark("percentile_filter (size=11)", iterations, || {
            std::hint::black_box(peaks::percentile_filter(&signal, 50.0, 11));
        });

        println!();
    }

    println!("============================================================");
    println!("                    Benchmark Complete");
    println!("============================================================");
}
