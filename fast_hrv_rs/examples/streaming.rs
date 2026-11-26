//! Live, packet-by-packet processing: raw ECG packets → `EcgStreamFilter` → `StreamDetector`,
//! with a gap in the middle of the recording, then RR intervals and time-domain HRV per segment.
//!
//! Run with: `cargo run --release --example streaming`
//!
//! The same code path is used for live data and for reprocessing history: feed the stored samples
//! through the same causal filter (reset at every gap) and the same `StreamDetector`, and the peaks
//! are bit-identical to what was emitted live.

use fast_hrv::{
    EcgFilterConfig, EcgStreamFilter, StreamConfig, StreamDetector, StreamPeaks,
    rri_from_positions, time_domain,
};

const FS: f64 = 512.0;
const PACKET: usize = 32; // the mobile app sends 32 samples per row

/// A crude synthetic ECG (one Gaussian R wave per beat, small noise) — enough to drive the example.
fn synthetic_ecg(seconds: f64, bpm: f64, seed: u64) -> Vec<f64> {
    let n = (seconds * FS) as usize;
    let rr = 60.0 / bpm;
    let mut state = seed;
    let mut noise = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.04
    };
    (0..n)
        .map(|i| {
            let t = i as f64 / FS;
            let phase = (t % rr) / rr - 0.3;
            (-(phase / 0.012).powi(2)).exp() + noise()
        })
        .collect()
}

fn main() -> Result<(), fast_hrv::HrvError> {
    // Two recording segments separated by a 20 s gap (e.g. the phone lost the connection).
    let seg_a = synthetic_ecg(120.0, 70.0, 1);
    let seg_b = synthetic_ecg(90.0, 75.0, 2);
    let gap_samples = (20.0 * FS) as u64;

    let mut filter = EcgStreamFilter::new(FS, &EcgFilterConfig::default())?;
    let mut detector = StreamDetector::new(FS, 0, StreamConfig::default())?;
    println!(
        "state per stream: {} bytes, max latency: {:.2} s",
        detector.state_bytes(),
        detector.max_latency_samples() as f64 / FS
    );

    let mut out = StreamPeaks::default();
    let mut clean = vec![0.0; PACKET];
    let mut low_bsqi_windows = 0;

    // Segment A, packet by packet. The callback sees each finished window (e.g. to run the SQIs
    // on `w.samples` with the window's own detections) without keeping a second sample buffer.
    for packet in seg_a.chunks(PACKET) {
        let c = &mut clean[..packet.len()];
        filter.process_into(packet, c);
        detector.push_with(c, &mut out, |w| {
            if let Some(b) = w.bsqi {
                if b.bsqi < 0.8 {
                    low_bsqi_windows += 1;
                }
            }
        })?;
    }

    // Gap: close segment A (emits its remaining peaks) and restart the filter state.
    let next = detector.next_index() + gap_samples;
    let gap = detector.reset_at(next, &mut out)?;
    filter.reset();
    let peaks_a: Vec<f64> = out.primary.iter().map(|p| p.position).collect();
    out.clear();
    println!("gap: samples {}..{}", gap.start, gap.end);

    // Segment B.
    for packet in seg_b.chunks(PACKET) {
        let c = &mut clean[..packet.len()];
        filter.process_into(packet, c);
        detector.push(c, &mut out)?;
    }
    detector.flush(&mut out)?;
    let peaks_b: Vec<f64> = out.primary.iter().map(|p| p.position).collect();

    // RR intervals never span the gap: compute them per segment from sub-sample positions.
    for (name, peaks) in [("A", &peaks_a), ("B", &peaks_b)] {
        let rri = rri_from_positions(peaks, FS);
        let td = time_domain(&rri);
        println!(
            "segment {name}: {} peaks, mean NN {:.1} ms, RMSSD {:.2} ms",
            peaks.len(),
            td.mean_nn,
            td.rmssd
        );
    }
    println!("windows with bSQI < 0.8 in segment A: {low_bsqi_windows}");
    Ok(())
}
