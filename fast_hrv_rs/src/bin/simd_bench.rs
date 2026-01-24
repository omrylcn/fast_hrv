//! SIMD benchmark

use fast_hrv_rs::*;
use std::time::Instant;

#[cfg(feature = "simd")]
use fast_hrv_rs::simd;

fn generate_data(n: usize) -> Vec<f64> {
    let mut data = Vec::with_capacity(n);
    let mut state = 42u64;
    for _ in 0..n {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        data.push(800.0 + ((state >> 16) & 0xffff) as f64 / 1000.0);
    }
    data
}

fn bench<F: FnMut()>(name: &str, iter: usize, mut f: F) {
    for _ in 0..10 { f(); }
    let start = Instant::now();
    for _ in 0..iter { f(); }
    let us = start.elapsed().as_micros() as f64 / iter as f64;
    println!("{:25} {:>10.2} µs", name, us);
}

fn main() {
    println!("=== SIMD Benchmark ===\n");
    
    for &n in &[1000, 10000, 100000] {
        let data = generate_data(n);
        let iter = if n <= 1000 { 10000 } else if n <= 10000 { 1000 } else { 100 };
        
        println!("--- {} samples ({} iterations) ---", n, iter);
        
        // Standard versions
        bench("mean (standard)", iter, || {
            std::hint::black_box(mean(&data));
        });
        
        bench("std_ddof (standard)", iter, || {
            std::hint::black_box(std_ddof(&data, 1));
        });
        
        bench("compute_rmssd (standard)", iter, || {
            std::hint::black_box(compute_rmssd(&data));
        });
        
        // SIMD versions (if enabled)
        #[cfg(feature = "simd")]
        {
            bench("mean (SIMD)", iter, || {
                std::hint::black_box(simd::mean_simd(&data));
            });
            
            bench("std (SIMD)", iter, || {
                std::hint::black_box(simd::std_simd(&data, 1));
            });
            
            bench("rmssd (SIMD)", iter, || {
                std::hint::black_box(simd::rmssd_simd(&data));
            });
        }
        
        #[cfg(not(feature = "simd"))]
        {
            println!("(SIMD disabled - run with --features simd)");
        }
        
        println!();
    }
}
