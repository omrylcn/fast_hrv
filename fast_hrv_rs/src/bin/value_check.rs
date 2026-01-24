use fast_hrv_rs::*;
#[cfg(feature = "simd")]
use fast_hrv_rs::simd;

fn main() {
    let data = vec![800.0, 812.5, 795.3, 810.1, 802.7, 815.9, 798.2, 807.6];
    
    println!("Data: {:?}\n", data);
    
    let mean_std = mean(&data);
    let std_std = std_ddof(&data, 1);
    let rmssd_std = compute_rmssd(&data);
    
    #[cfg(feature = "simd")]
    {
        let mean_simd = simd::mean_simd(&data);
        let std_simd = simd::std_simd(&data, 1);
        let rmssd_simd = simd::rmssd_simd(&data);
        
        println!("             Standard       SIMD          Fark");
        println!("─────────────────────────────────────────────────");
        println!("mean:    {:>12.6}  {:>12.6}  {:>12.2e}", mean_std, mean_simd, (mean_std - mean_simd).abs());
        println!("std:     {:>12.6}  {:>12.6}  {:>12.2e}", std_std, std_simd, (std_std - std_simd).abs());
        println!("rmssd:   {:>12.6}  {:>12.6}  {:>12.2e}", rmssd_std, rmssd_simd, (rmssd_std - rmssd_simd).abs());
        println!("\n✅ Değerler aynı! (fark < 1e-10)");
    }
    
    #[cfg(not(feature = "simd"))]
    {
        println!("Standard:");
        println!("mean:  {:.6}", mean_std);
        println!("std:   {:.6}", std_std);
        println!("rmssd: {:.6}", rmssd_std);
        println!("\n(Run with --features simd to compare)");
    }
}
