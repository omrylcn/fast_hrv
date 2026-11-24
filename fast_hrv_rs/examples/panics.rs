fn tryit(name: &str, f: impl FnOnce() + std::panic::UnwindSafe) {
    let r = std::panic::catch_unwind(f);
    println!("{name}: {}", if r.is_ok() { "ok" } else { "PANIC" });
}
fn main() {
    std::panic::set_hook(Box::new(|i| eprintln!("  panic: {}", i)));
    tryit("compute_poincare(&[])", || { fast_hrv::compute_poincare(&[]); });
    tryit("compute_asymmetry(&[])", || { fast_hrv::compute_asymmetry(&[]); });
    tryit("compute_dfa(&[],..)", || { fast_hrv::compute_dfa(&[], (4,16),(16,None)); });
    let mut v: Vec<f64> = (0..400).map(|i| 800.0 + (i as f64).sin()*30.0).collect();
    v[5] = f64::NAN;
    tryit("time_domain(with NaN)", || { fast_hrv::time_domain(&v); });
    let v2 = v.clone();
    tryit("frequency_domain(with NaN)", || { fast_hrv::frequency_domain(&v2); });
    tryit("rri_from_peaks(unsorted)", || { fast_hrv::rri_from_peaks(&[100, 50], 512); });
    tryit("fix_peaks(unsorted)", || { fast_hrv::fix_peaks_default(&[100, 600, 550, 1000, 1500], 512); });
    tryit("ecg_clean(fs=0)", || { let y = fast_hrv::ecg_clean(&vec![1.0; 100], 0); println!("  nan? {}", y[0].is_nan()); });
    let s: Vec<f64> = (0..5000).map(|i| ((i as f64)*0.01).sin()).collect();
    tryit("compute_quality(peaks beyond len)", || { fast_hrv::compute_quality(&s, &[100, 600, 1100, 1600, 2100, 9000], 512); });
    let rr = vec![800.0; 100];
    let r = fast_hrv::hrv(&rr);
    println!("constant RR: sampen={} dfa1={} lf={} lf_hf={} csi={}", r.nonlinear.sampen, r.nonlinear.dfa_alpha1, r.frequency.lf, r.frequency.lf_hf, r.nonlinear.csi);
}
