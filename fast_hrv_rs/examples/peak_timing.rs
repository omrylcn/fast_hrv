//! Peak positions of the offline and the live path on a recorded ECG (driven by
//! `tools/mitdb_timing.py`; dev-time timing study).
//! Usage: peak_timing <fs> <file.f64>   → JSON {offline:[..], live:[..], live_secondary:[..]}
use fast_hrv::{
    EcgFilterConfig, EcgStreamFilter, StreamConfig, StreamDetector, StreamPeaks, ecg_clean,
    find_peaks_default,
};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let fs: f64 = a[1].parse().unwrap();
    let bytes = std::fs::read(&a[2]).unwrap();
    let x: Vec<f64> = bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let offline = find_peaks_default(&ecg_clean(&x, fs as u32), fs as u32);
    let mut filt = EcgStreamFilter::new(fs, &EcgFilterConfig::default()).unwrap();
    // POLARITY=pos|neg|abs|auto overrides the refinement polarity (study only)
    let mut cfg = StreamConfig::default();
    if let Ok(p) = std::env::var("POLARITY") {
        cfg.dual.refine.polarity = match p.as_str() {
            "pos" => fast_hrv::Polarity::Positive,
            "neg" => fast_hrv::Polarity::Negative,
            "abs" => fast_hrv::Polarity::Absolute,
            _ => fast_hrv::Polarity::Auto,
        };
    }
    if let Some(fc) = std::env::var("SMOOTH_HZ").ok().and_then(|v| v.parse().ok()) {
        cfg.dual.refine.smoothing_hz = Some(fc);
    }
    let mut det = StreamDetector::new(fs, 0, cfg).unwrap();
    let mut peaks = StreamPeaks::default();
    let (mut live, mut sec) = (Vec::new(), Vec::new());
    let mut clean_all = Vec::with_capacity(x.len());
    for p in x.chunks(32) {
        let c = filt.process(p);
        clean_all.extend_from_slice(&c);
        det.push(&c, &mut peaks).unwrap();
        live.extend(peaks.primary.iter().map(|q| q.position));
        sec.extend(peaks.secondary.iter().map(|q| q.position));
        peaks.clear();
    }
    det.flush(&mut peaks).unwrap();
    live.extend(peaks.primary.iter().map(|q| q.position));
    sec.extend(peaks.secondary.iter().map(|q| q.position));
    // offline detector on the causally filtered signal: separates filter delay from detector choice
    let offline_on_causal = find_peaks_default(&clean_all, fs as u32);
    if let Ok(path) = std::env::var("DUMP_CLEAN") {
        let b: Vec<u8> = clean_all.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(path, b).unwrap();
    }
    let j = |v: &[f64]| {
        v.iter()
            .map(|p| format!("{p:.3}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    let ji = |v: &[usize]| {
        v.iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(",")
    };
    println!(
        "{{\"offline\":[{}],\"offline_on_causal\":[{}],\"live\":[{}],\"live_secondary\":[{}]}}",
        ji(&offline),
        ji(&offline_on_causal),
        j(&live),
        j(&sec)
    );
}
