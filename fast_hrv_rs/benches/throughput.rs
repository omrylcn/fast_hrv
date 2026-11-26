//! Reproducible throughput benchmark (std only, no external crates).
//!
//! `cargo bench --bench throughput` — deterministic synthetic input (5 min ECG at 512 Hz:
//! Gaussian QRS bumps, DC offset, baseline wander, 50 Hz mains). Reports the median of several runs.
//! No real recordings are used.

use std::hint::black_box;
use std::time::Instant;

use fast_hrv::detect::{
    DualConfig, ElgendiConfig, RefineConfig, WindowConfig, dual_detect, elgendi_peaks,
    elgendi_refined_windowed, refine_peaks,
};
use fast_hrv::ecg::{EcgFilterConfig, EcgStreamFilter};

fn synthetic_ecg(fs: f64, seconds: f64) -> Vec<f64> {
    let n = (fs * seconds) as usize;
    let mut x = vec![412.0; n];
    let (mut t, mut k) = (0.3f64, 0.0f64);
    while t < seconds - 0.5 {
        let c = (t * fs) as isize;
        for d in -15isize..=15 {
            let i = c + d;
            if i >= 0 && (i as usize) < n {
                x[i as usize] += 200.0 * (-(d as f64).powi(2) / 18.0).exp();
            }
        }
        k += 1.0;
        t += 0.83 + 0.03 * (k * 0.7).sin();
    }
    for (i, v) in x.iter_mut().enumerate() {
        let ts = i as f64 / fs;
        *v += 5.0 * (i as f64 * 0.37).sin()
            + 30.0 * (2.0 * std::f64::consts::PI * 0.2 * ts).sin()
            + 20.0 * (2.0 * std::f64::consts::PI * 50.0 * ts).sin();
    }
    x
}

/// Median wall time (ms) of `reps` runs of `f`.
fn time_ms(reps: usize, mut f: impl FnMut()) -> f64 {
    f(); // warm-up
    let mut v: Vec<f64> = (0..reps)
        .map(|_| {
            let t0 = Instant::now();
            f();
            t0.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    let fs = 512u32;
    let x = synthetic_ecg(fs as f64, 300.0);
    let reps = 21;

    let clean = time_ms(reps, || {
        black_box(fast_hrv::ecg_clean(black_box(&x), fs));
    });
    let cfg = EcgFilterConfig::default().with_notch(50.0, 30.0);
    let stream = time_ms(reps, || {
        let mut f = EcgStreamFilter::new(fs as f64, &cfg).unwrap();
        let mut buf = [0.0; 32];
        for p in x.chunks(32) {
            f.process_into(p, &mut buf[..p.len()]);
            black_box(&buf);
        }
    });
    let cleaned = fast_hrv::ecg_clean(&x, fs);
    let peaks_t = time_ms(reps, || {
        black_box(fast_hrv::find_peaks_default(black_box(&cleaned), fs));
    });
    let peaks = fast_hrv::find_peaks_default(&cleaned, fs);
    let kubios = time_ms(reps, || {
        black_box(fast_hrv::fix_peaks_default(black_box(&peaks), fs));
    });
    let pipeline = time_ms(reps, || {
        black_box(fast_hrv::process_ecg(black_box(&x), fs, true));
    });
    let rri = fast_hrv::rri_from_peaks(&peaks, fs);
    let hrv_t = time_ms(reps, || {
        black_box(fast_hrv::hrv(black_box(&rri)));
    });
    // detectors on the 0.5–40 Hz zero-phase clean signal
    let fsf = fs as f64;
    let wide = fast_hrv::ecg_clean_zero_phase(&x, fsf, &EcgFilterConfig::default()).unwrap();
    let ecfg = ElgendiConfig::default();
    let elg_t = time_ms(reps, || {
        black_box(elgendi_peaks(black_box(&wide), fsf, &ecfg).unwrap());
    });
    let elg = elgendi_peaks(&wide, fsf, &ecfg).unwrap();
    let rcfg = RefineConfig::default();
    let refine_t = time_ms(reps, || {
        black_box(refine_peaks(black_box(&wide), fsf, &elg, &rcfg).unwrap());
    });
    let wcfg = WindowConfig::default();
    let windowed_t = time_ms(reps, || {
        black_box(elgendi_refined_windowed(black_box(&wide), fsf, 0, &wcfg, &ecfg, &rcfg).unwrap());
    });
    let dcfg = DualConfig::default();
    let dual_t = time_ms(reps, || {
        for w in wide.chunks(10 * fs as usize) {
            black_box(dual_detect(black_box(w), fsf, &dcfg).unwrap());
        }
    });
    let mut gap = rri.clone();
    gap[100] = 69_000.0;
    let hrv_gap = time_ms(reps, || {
        black_box(fast_hrv::hrv(black_box(&gap)));
    });

    println!(
        "fast-hrv throughput, 5 min ECG @ {fs} Hz ({} samples, {} R-peaks), median of {reps} runs",
        x.len(),
        peaks.len()
    );
    println!("  ecg_clean (NeuroKit-compatible, zero-phase)  {clean:8.2} ms");
    println!("  EcgStreamFilter, 32-sample packets (+notch)  {stream:8.2} ms");
    println!("  find_peaks_default                           {peaks_t:8.2} ms");
    println!("  fix_peaks_default (Kubios)                   {kubios:8.2} ms");
    println!("  process_ecg (clean+peaks+Kubios+quality)     {pipeline:8.2} ms");
    println!(
        "  hrv() all metrics, {} RR                    {hrv_t:8.2} ms",
        rri.len()
    );
    println!("  hrv() with one 69 s RR (gap)                 {hrv_gap:8.2} ms");
    println!("  elgendi_peaks (8–20 Hz + detection)          {elg_t:8.2} ms");
    println!("  refine_peaks (±50 ms, parabolic)             {refine_t:8.2} ms");
    println!("  elgendi_refined_windowed (10 s + 2×2 s)      {windowed_t:8.2} ms");
    println!("  dual_detect on 30 × 10 s windows (+bSQI)     {dual_t:8.2} ms");
}
