use fast_hrv_rs::*;
use std::time::Instant;

fn generate_ecg(n: usize) -> Vec<f64> {
    let mut signal = vec![0.0; n];
    let mut state = 42u64;
    let mut pos = 410usize;
    while pos < n - 50 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let jitter = ((state >> 16) & 0x3f) as isize - 32;
        let actual_pos = (pos as isize + jitter).max(0) as usize;
        if actual_pos < n && actual_pos >= 5 && actual_pos + 10 < n {
            signal[actual_pos] = 1.0;
            signal[actual_pos - 3] = 0.2;
            signal[actual_pos + 3] = -0.3;
        }
        pos += 410;
    }
    for i in 0..n {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        signal[i] += ((state >> 16) & 0xff) as f64 / 2560.0 - 0.05;
    }
    signal
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
        let ecg = generate_ecg(n);
        let iter = if duration <= 60 { 20 } else { 10 };
        
        println!("╔═══════════════════════════════════════════════════════════╗");
        println!("║  {} saniye ECG ({} samples)                            ║", duration, n);
        println!("╠═══════════════════════════════════════════════════════════╣");
        
        // 1. ecg_clean
        let t_clean = bench(iter, || { std::hint::black_box(ecg_clean(&ecg, sr)); });
        let cleaned = ecg_clean(&ecg, sr);
        
        // 2. find_peaks
        let t_find = bench(iter, || { std::hint::black_box(find_peaks_default(&cleaned, sr)); });
        let peaks = find_peaks_default(&cleaned, sr);
        
        // 3. fix_peaks
        let t_fix = bench(iter, || { std::hint::black_box(fix_peaks_default(&peaks, sr)); });
        let (fixed_peaks, _) = fix_peaks_default(&peaks, sr);
        
        // 4. compute_quality
        let t_quality = bench(iter, || { std::hint::black_box(compute_quality(&cleaned, &fixed_peaks, sr)); });
        let (_, quality_per_beat) = compute_quality(&cleaned, &fixed_peaks, sr);
        
        // 5. compute_stats
        let t_stats = bench(iter, || { std::hint::black_box(compute_stats(&fixed_peaks, &quality_per_beat, sr)); });
        
        let total = t_clean + t_find + t_fix + t_quality + t_stats;
        
        println!("║ Fonksiyon        │ Süre (µs)  │ Yüzde      │ Bar        ║");
        println!("╟──────────────────┼────────────┼────────────┼────────────╢");
        
        let bar = |pct: f64| -> String {
            let blocks = (pct / 5.0).round() as usize;
            "█".repeat(blocks.min(10))
        };
        
        let pct_clean = t_clean / total * 100.0;
        let pct_find = t_find / total * 100.0;
        let pct_fix = t_fix / total * 100.0;
        let pct_quality = t_quality / total * 100.0;
        let pct_stats = t_stats / total * 100.0;
        
        println!("║ ecg_clean        │ {:>8.0}   │ {:>6.1}%    │ {:10} ║", t_clean, pct_clean, bar(pct_clean));
        println!("║ find_peaks       │ {:>8.0}   │ {:>6.1}%    │ {:10} ║", t_find, pct_find, bar(pct_find));
        println!("║ fix_peaks        │ {:>8.0}   │ {:>6.1}%    │ {:10} ║", t_fix, pct_fix, bar(pct_fix));
        println!("║ compute_quality  │ {:>8.0}   │ {:>6.1}%    │ {:10} ║", t_quality, pct_quality, bar(pct_quality));
        println!("║ compute_stats    │ {:>8.0}   │ {:>6.1}%    │ {:10} ║", t_stats, pct_stats, bar(pct_stats));
        println!("╟──────────────────┼────────────┼────────────┼────────────╢");
        println!("║ TOPLAM           │ {:>8.0}   │  100.0%    │            ║", total);
        println!("╚═══════════════════════════════════════════════════════════╝");
        println!();
    }
}
