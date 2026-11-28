//! The quick starts of the root README and the crate README, compiled so the READMEs cannot drift
//! from the API. `cargo run --release --example readme_quickstart`
use fast_hrv::*;

fn main() -> Result<(), HrvError> {
    // 60 s of a synthetic 75 bpm ECG at 512 Hz (stands in for a recording)
    let fs = 512u32;
    let ecg_signal: Vec<f64> = (0..60 * fs as usize)
        .map(|i| {
            let t = i as f64 / fs as f64;
            let phase = (t - 0.5).rem_euclid(0.8);
            1000.0 * (-((phase.min(0.8 - phase)) / 0.012).powi(2)).exp()
        })
        .collect();

    // root README
    let result = process_ecg(&ecg_signal, fs, true);
    println!("R-peaks: {}", result.rpeaks.len());
    println!("RMSSD: {:.2} ms", result.stats.rmssd_ms);

    // crate README: from R-peak sample indices
    let peaks = &result.rpeaks;
    let rri = fast_hrv::rri_from_peaks(peaks, 512); // R-peak indices → NN intervals (ms)
    let r = fast_hrv::hrv(&rri);
    println!(
        "SDNN {:.1} ms  RMSSD {:.1} ms  LF/HF {:.2}  SampEn {:.2}",
        r.time.sdnn, r.time.rmssd, r.frequency.lf_hf, r.nonlinear.sampen
    );

    // crate README: from raw ECG, with sub-sample peaks and artifact correction
    let raw = &ecg_signal;
    {
        use fast_hrv::detect::{ElgendiConfig, RefineConfig, elgendi_peaks, refine_peaks};
        let clean =
            fast_hrv::ecg_clean_zero_phase(raw, 512.0, &fast_hrv::EcgFilterConfig::default())?;
        let cand = elgendi_peaks(&clean, 512.0, &ElgendiConfig::default())?;
        let peaks = refine_peaks(&clean, 512.0, &cand, &RefineConfig::default())?;
        let fixed = peaks.fix_kubios(512.0, &fast_hrv::KubiosParams::default())?;
        let r = fast_hrv::hrv(&fixed.rri_ms(512.0));
        println!("refined: RMSSD {:.1} ms", r.time.rmssd);
    }

    // crate README: live, 32 samples at a time
    let packets = raw.chunks(32);
    {
        use fast_hrv::stream::{StreamConfig, StreamDetector, StreamPeaks};
        let mut filt =
            fast_hrv::EcgStreamFilter::new(512.0, &fast_hrv::EcgFilterConfig::default())?;
        let mut det = StreamDetector::new(512.0, 0, StreamConfig::default())?;
        let mut out = StreamPeaks::default();
        for packet in packets {
            det.push(&filt.process(packet), &mut out)?; // out.primary: confirmed peaks
        }
        det.flush(&mut out)?;
        println!("live peaks: {}", out.primary.len());
    }
    Ok(())
}
