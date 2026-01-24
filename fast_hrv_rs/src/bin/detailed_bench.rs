use fast_hrv_rs::*;
use std::time::Instant;

fn main() {
    println!("=== DETAILED FUNCTION BENCHMARK (360s ECG) ===\n");
    
    // Generate 360s ECG
    let sr = 512u32;
    let n = 360 * sr as usize;
    let mut signal = vec![0.0; n];
    let mut peaks = Vec::new();
    let mut state = 42u64;
    let mut pos = 410usize;
    
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
        pos += 410;
    }
    
    let iterations = 5;
    
    // Signal Processing
    println!("--- SIGNAL PROCESSING ({} samples) ---", n);
    
    let start = Instant::now();
    for _ in 0..iterations { std::hint::black_box(peaks::gradient(&signal)); }
    println!("gradient:              {:>8.0} µs", start.elapsed().as_micros() as f64 / iterations as f64);
    
    let start = Instant::now();
    for _ in 0..iterations { std::hint::black_box(peaks::uniform_filter1d(&signal, 51)); }
    println!("uniform_filter1d(51):  {:>8.0} µs", start.elapsed().as_micros() as f64 / iterations as f64);
    
    let start = Instant::now();
    for _ in 0..iterations { std::hint::black_box(peaks::uniform_filter1d(&signal, 384)); }
    println!("uniform_filter1d(384): {:>8.0} µs", start.elapsed().as_micros() as f64 / iterations as f64);
    
    let start = Instant::now();
    for _ in 0..iterations { std::hint::black_box(peaks::median_filter(&signal, 11)); }
    println!("median_filter(11):     {:>8.0} µs  ⚠️ O(n×k)", start.elapsed().as_micros() as f64 / iterations as f64);
    
    let start = Instant::now();
    for _ in 0..iterations { std::hint::black_box(peaks::median_filter(&signal, 91)); }
    println!("median_filter(91):     {:>8.0} µs  ⚠️ O(n×k)", start.elapsed().as_micros() as f64 / iterations as f64);
    
    let start = Instant::now();
    for _ in 0..iterations { std::hint::black_box(peaks::percentile_filter(&signal, 25.0, 91)); }
    println!("percentile_filter(91): {:>8.0} µs  ⚠️ O(n×k)", start.elapsed().as_micros() as f64 / iterations as f64);
    
    // Peak Detection
    println!("\n--- PEAK DETECTION ---");
    
    let start = Instant::now();
    for _ in 0..iterations { std::hint::black_box(find_peaks_default(&signal, sr)); }
    println!("find_peaks:            {:>8.0} µs  ✅ O(n)", start.elapsed().as_micros() as f64 / iterations as f64);
    
    let start = Instant::now();
    for _ in 0..iterations { std::hint::black_box(fix_peaks_default(&peaks, sr)); }
    println!("fix_peaks:             {:>8.0} µs  ⚠️ uses percentile×4", start.elapsed().as_micros() as f64 / iterations as f64);
    
    // ECG Processing  
    println!("\n--- ECG PROCESSING ---");
    
    let start = Instant::now();
    for _ in 0..iterations { std::hint::black_box(ecg_clean(&signal, sr)); }
    println!("ecg_clean:             {:>8.0} µs  ✅ O(n)", start.elapsed().as_micros() as f64 / iterations as f64);
    
    let start = Instant::now();
    for _ in 0..iterations { std::hint::black_box(compute_quality(&signal, &peaks, sr)); }
    println!("compute_quality:       {:>8.0} µs", start.elapsed().as_micros() as f64 / iterations as f64);
    
    // HRV (300 RR intervals)
    let rri: Vec<f64> = peaks.windows(2).take(300).map(|w| (w[1] - w[0]) as f64 / sr as f64 * 1000.0).collect();
    println!("\n--- HRV METRICS ({} RR intervals) ---", rri.len());
    
    let start = Instant::now();
    for _ in 0..1000 { std::hint::black_box(compute_rmssd(&rri)); }
    println!("rmssd:                 {:>8.2} µs  ✅ O(n)", start.elapsed().as_micros() as f64 / 1000.0);
    
    let start = Instant::now();
    for _ in 0..1000 { std::hint::black_box(compute_poincare(&rri)); }
    println!("poincare:              {:>8.2} µs  ✅ O(n)", start.elapsed().as_micros() as f64 / 1000.0);
    
    let start = Instant::now();
    for _ in 0..100 { std::hint::black_box(compute_sampen(&rri, 2, None)); }
    println!("sampen:                {:>8.0} µs  ⚠️ O(n²)", start.elapsed().as_micros() as f64 / 100.0);
    
    let start = Instant::now();
    for _ in 0..100 { std::hint::black_box(compute_dfa(&rri, (4, 11), (12, None))); }
    println!("dfa:                   {:>8.0} µs  ✅ O(n×scales)", start.elapsed().as_micros() as f64 / 100.0);
    
    println!("\n=== BOTTLENECK ANALYSIS ===");
    println!("⚠️ median_filter, percentile_filter: O(n × window_size)");
    println!("⚠️ fix_peaks calls percentile_filter 4 times with size=91");
    println!("⚠️ sampen: O(n²) - unavoidable for this algorithm");
    println!("✅ Diğer tüm fonksiyonlar O(n) veya daha iyi");
}
