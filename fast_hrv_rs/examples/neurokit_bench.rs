//! fast-hrv side of the NeuroKit2 benchmark (`tools/neurokit_benchmark.py` drives it).
//!
//! For each raw ECG file (little-endian f64, 512 Hz) runs the same steps the script times in
//! NeuroKit2 and prints one JSON line: median time per step (ms), the detected peaks and a few
//! metrics so the script can check that both libraries reach the same result.
//!
//! Usage: cargo run --release --example neurokit_bench -- <runs> <file.f64>...

use std::time::Instant;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let runs: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(21);
    let fs = 512u32;
    for path in &args[2..] {
        let bytes = std::fs::read(path).expect("read ecg file");
        let ecg: Vec<f64> = bytes
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let mut t: Vec<Vec<f64>> = (0..6).map(|_| Vec::with_capacity(runs)).collect();
        let (mut peaks, mut fixed) = (Vec::new(), Vec::new());
        let mut report = None;
        for _ in 0..runs {
            let a = Instant::now();
            let clean = fast_hrv::ecg_clean(&ecg, fs);
            t[0].push(a.elapsed().as_secs_f64());
            let a = Instant::now();
            peaks = fast_hrv::find_peaks_default(&clean, fs);
            t[1].push(a.elapsed().as_secs_f64());
            let a = Instant::now();
            fixed = fast_hrv::fix_peaks_default(&peaks, fs).0;
            t[2].push(a.elapsed().as_secs_f64());
            let rri = fast_hrv::rri_from_peaks(&fixed, fs);
            let a = Instant::now();
            let _ = fast_hrv::time_domain(&rri);
            t[3].push(a.elapsed().as_secs_f64());
            let a = Instant::now();
            let _ = fast_hrv::frequency_domain(&rri);
            t[4].push(a.elapsed().as_secs_f64());
            let a = Instant::now();
            report = Some(fast_hrv::hrv(&rri));
            t[5].push(a.elapsed().as_secs_f64());
        }
        let ms: Vec<f64> = t.into_iter().map(|v| median(v) * 1000.0).collect();
        let r = report.unwrap();
        let peaks_json: Vec<String> = fixed.iter().map(|p| p.to_string()).collect();
        println!(
            "{{\"file\":\"{}\",\"runs\":{},\"clean_ms\":{:.4},\"peaks_ms\":{:.4},\"kubios_ms\":{:.4},\"hrv_time_ms\":{:.4},\"hrv_frequency_ms\":{:.4},\"hrv_all_ms\":{:.4},\"n_peaks_raw\":{},\"rmssd\":{},\"sdnn\":{},\"sampen\":{},\"lf_hf\":{},\"peaks\":[{}]}}",
            path,
            runs,
            ms[0],
            ms[1],
            ms[2],
            ms[3],
            ms[4],
            ms[5],
            peaks.len(),
            r.time.rmssd,
            r.time.sdnn,
            r.nonlinear.sampen,
            if r.frequency.lf_hf.is_finite() {
                r.frequency.lf_hf.to_string()
            } else {
                "null".into()
            },
            peaks_json.join(",")
        );
    }
}
