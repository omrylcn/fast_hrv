//! `StreamDetector`: live packet-by-packet detection must equal the windowed batch path
//! (`elgendi_refined_windowed`, `dual_detect_windowed`) bit for bit, for any packetisation, start
//! offset and gap pattern. Synthetic signals only.

use fast_hrv::detect::{
    DualConfig, ElgendiConfig, RefineConfig, WindowConfig, dual_detect, dual_detect_windowed,
    elgendi_refined_windowed,
};
use fast_hrv::ecg::{EcgFilterConfig, EcgStreamFilter, ecg_clean_causal};
use fast_hrv::gaps::{Gap, split_at_gaps};
use fast_hrv::stream::{StreamConfig, StreamDetector, StreamPeak, StreamPeaks};

const FS: f64 = 512.0;

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let (u1, u2) = (self.uniform().max(1e-300), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Raw-like synthetic ECG in ADC counts: P-QRS-T Gaussians with respiratory RR modulation, DC
/// offset, baseline wander, 50 Hz mains and white noise. Returns (signal, true R positions).
fn synth_raw(fs: f64, seconds: f64, seed: u64, noise: f64, sign: f64) -> (Vec<f64>, Vec<f64>) {
    let n = (seconds * fs) as usize;
    let mut rng = Rng(seed | 1);
    let mut x = vec![0.0; n];
    let waves = [
        (0.12, -0.20, 0.025),
        (-0.10, -0.035, 0.008),
        (1.00, 0.0, 0.009),
        (-0.25, 0.035, 0.008),
        (0.30, 0.28, 0.045),
    ];
    let mut r_true = Vec::new();
    let mut t = 0.4 + rng.uniform() * 0.3;
    while t < seconds - 0.6 {
        r_true.push(t * fs);
        let amp = 1.0 + 0.05 * rng.gauss();
        let lo = ((t - 0.5) * fs).max(0.0) as usize;
        let hi = (((t + 0.6) * fs) as usize).min(n);
        for (i, xi) in x.iter_mut().enumerate().take(hi).skip(lo) {
            let ti = i as f64 / fs - t;
            for &(a, off, s) in &waves {
                let z = (ti - off) / s;
                *xi += sign * amp * a * (-0.5 * z * z).exp();
            }
        }
        t += 0.8
            + 0.04 * (2.0 * std::f64::consts::PI * 0.25 * t).sin()
            + 0.03 * (2.0 * std::f64::consts::PI * 0.07 * t).sin()
            + 0.015 * rng.gauss();
    }
    for (i, xi) in x.iter_mut().enumerate() {
        let ti = i as f64 / fs;
        let mv = *xi
            + 0.15 * (2.0 * std::f64::consts::PI * 0.3 * ti).sin()
            + 0.03 * (2.0 * std::f64::consts::PI * 50.0 * ti).sin()
            + noise * rng.gauss();
        *xi = 420.0 + 1000.0 * mv;
    }
    (x, r_true)
}

#[derive(Clone, Copy, Debug)]
enum Packets {
    Fixed(usize),
    Random(u64),
}

fn packet_sizes(n: usize, p: Packets) -> Vec<usize> {
    let mut out = Vec::new();
    let mut left = n;
    let mut rng = match p {
        Packets::Random(seed) => Rng(seed | 1),
        Packets::Fixed(_) => Rng(1),
    };
    while left > 0 {
        let s = match p {
            Packets::Fixed(s) => s,
            Packets::Random(_) => 1 + (rng.next_u64() % 1000) as usize,
        }
        .min(left);
        out.push(s);
        left -= s;
    }
    out
}

const PACKETS: [Packets; 6] = [
    Packets::Fixed(1),
    Packets::Fixed(32),
    Packets::Fixed(33),
    Packets::Fixed(512),
    Packets::Fixed(7168),
    Packets::Random(7),
];

/// raw → EcgStreamFilter → StreamDetector, packet by packet, then flush.
fn live_pipeline(raw: &[f64], start: u64, cfg: StreamConfig, packets: Packets) -> StreamPeaks {
    let mut filt = EcgStreamFilter::new(FS, &EcgFilterConfig::default()).unwrap();
    let mut det = StreamDetector::new(FS, start, cfg).unwrap();
    let mut out = StreamPeaks::default();
    let mut buf = vec![0.0; 8192];
    let mut at = 0;
    for s in packet_sizes(raw.len(), packets) {
        let b = &mut buf[..s];
        filt.process_into(&raw[at..at + s], b);
        det.push(b, &mut out).unwrap();
        at += s;
    }
    det.flush(&mut out).unwrap();
    out
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn positions(p: &[StreamPeak]) -> Vec<f64> {
    p.iter().map(|x| x.position).collect()
}

fn check_indices(p: &[StreamPeak]) {
    for x in p {
        assert!((x.position - x.index as f64).abs() <= 0.5, "{x:?}");
    }
    assert!(p.windows(2).all(|w| w[0].index < w[1].index));
}

#[test]
fn end_to_end_raw_filter_stream_equals_batch_for_any_packetisation() {
    let (raw, r_true) = synth_raw(FS, 95.3, 11, 0.02, 1.0);
    let wc = WindowConfig::default();
    let dual = DualConfig::default();
    let clean = ecg_clean_causal(&raw, FS, &EcgFilterConfig::default()).unwrap();
    for start in [0u64, 1, 777, 5119, 5120, 12_345, 10 * 5120 + 1023] {
        let batch = dual_detect_windowed(&clean, FS, start, &wc, &dual).unwrap();
        let primary_only =
            elgendi_refined_windowed(&clean, FS, start, &wc, &dual.elgendi, &dual.refine).unwrap();
        assert_eq!(bits(&batch.primary), bits(&primary_only));
        assert!(batch.primary.len().abs_diff(r_true.len()) <= 2);
        assert!(batch.secondary.len().abs_diff(r_true.len()) <= 2);
        for packets in PACKETS {
            let live = live_pipeline(&raw, start, StreamConfig::default(), packets);
            assert_eq!(
                bits(&positions(&live.primary)),
                bits(&batch.primary),
                "primary, start {start}, {packets:?}"
            );
            assert_eq!(
                bits(&positions(&live.secondary)),
                bits(&batch.secondary),
                "secondary, start {start}, {packets:?}"
            );
            check_indices(&live.primary);
            check_indices(&live.secondary);
        }
        // primary only (no secondary detector): equal to elgendi_refined_windowed
        let cfg = StreamConfig {
            secondary: false,
            ..StreamConfig::default()
        };
        for packets in [Packets::Fixed(32), Packets::Random(start + 3)] {
            let live = live_pipeline(&raw, start, cfg, packets);
            assert_eq!(bits(&positions(&live.primary)), bits(&primary_only));
            assert!(live.secondary.is_empty());
        }
    }
}

#[test]
fn other_grids_rates_and_configurations() {
    // margin larger than the window, a non-default refinement, 250 Hz and an inverted lead
    for (fs, wc, sign) in [
        (
            250.0,
            WindowConfig {
                window_s: 3.0,
                margin_s: 4.0,
                dedup_s: 0.2,
            },
            -1.0,
        ),
        (
            512.0,
            WindowConfig {
                window_s: 7.3,
                margin_s: 0.0,
                dedup_s: 0.1,
            },
            1.0,
        ),
    ] {
        let (raw, _) = synth_raw(fs, 41.0, 5, 0.03, sign);
        let clean = ecg_clean_causal(&raw, fs, &EcgFilterConfig::default()).unwrap();
        let dual = DualConfig {
            elgendi: ElgendiConfig::neurokit2013(),
            refine: RefineConfig {
                half_window_s: 0.07,
                ..RefineConfig::default()
            },
            ..DualConfig::default()
        };
        let cfg = StreamConfig {
            window: wc,
            dual,
            secondary: true,
        };
        for start in [0u64, 999] {
            let batch = dual_detect_windowed(&clean, fs, start, &wc, &dual).unwrap();
            for packets in [Packets::Fixed(1), Packets::Fixed(32), Packets::Random(9)] {
                let mut det = StreamDetector::new(fs, start, cfg).unwrap();
                let mut out = StreamPeaks::default();
                let mut at = 0;
                for s in packet_sizes(clean.len(), packets) {
                    det.push(&clean[at..at + s], &mut out).unwrap();
                    at += s;
                }
                det.flush(&mut out).unwrap();
                assert_eq!(bits(&positions(&out.primary)), bits(&batch.primary));
                assert_eq!(bits(&positions(&out.secondary)), bits(&batch.secondary));
            }
        }
    }
}

#[test]
fn window_callback_sees_the_batch_chunk_and_dual_detect_result() {
    let (raw, _) = synth_raw(FS, 64.0, 21, 0.02, 1.0);
    let clean = ecg_clean_causal(&raw, FS, &EcgFilterConfig::default()).unwrap();
    let start = 3000u64;
    let mut det = StreamDetector::new(FS, start, StreamConfig::default()).unwrap();
    let mut out = StreamPeaks::default();
    let mut seen = Vec::new();
    let mut check = |w: &fast_hrv::stream::StreamWindow<'_>| {
        let a = (w.chunk_start - start) as usize;
        let chunk = &clean[a..a + w.samples.len()];
        assert_eq!(bits(w.samples), bits(chunk));
        let d = dual_detect(chunk, FS, &DualConfig::default()).unwrap();
        assert_eq!(*w.primary, d.primary);
        assert_eq!(w.primary_raw, &d.primary_raw[..]);
        assert_eq!(w.secondary, Some(&d.secondary));
        assert_eq!(w.bsqi, d.bsqi);
        let r = w.core_range();
        assert!(r.end <= w.samples.len() && r.start <= r.end);
        seen.push((w.window, w.chunk_start, w.samples.len()));
    };
    for p in clean.chunks(32) {
        det.push_with(p, &mut out, &mut check).unwrap();
    }
    det.flush_with(&mut out, &mut check).unwrap();
    // windows 0..=12 (core 0 starts before the segment, the last ones are clipped)
    let (w, m) = (5120u64, 1024u64);
    let end = start + clean.len() as u64;
    let want: Vec<(u64, u64, usize)> = (start / w..=(end - 1) / w)
        .map(|k| {
            let c0 = (k * w).saturating_sub(m).max(start);
            let c1 = ((k + 1) * w + m).min(end);
            (k, c0, (c1 - c0) as usize)
        })
        .collect();
    assert_eq!(seen, want);
}

#[test]
fn gap_reset_equals_batch_per_segment() {
    let (raw, _) = synth_raw(FS, 120.0, 31, 0.02, 1.0);
    // segments [0, 40.0 s), lost [40.0, 43.3 s), [43.3, 70.1 s), discontinuity at 70.1 s (filter
    // reset, no lost samples), [70.1, 120 s)
    let cut = |t: f64| (t * FS) as usize;
    let segments = [
        (0, cut(40.0)),
        (cut(43.3), cut(70.1)),
        (cut(70.1), raw.len()),
    ];
    let cfg = EcgFilterConfig::default();
    let wc = WindowConfig::default();
    let mut want = StreamPeaks::default();
    for &(a, b) in &segments {
        let clean = ecg_clean_causal(&raw[a..b], FS, &cfg).unwrap();
        let d = dual_detect_windowed(&clean, FS, a as u64, &wc, &DualConfig::default()).unwrap();
        // batch has no indices; positions are what must match
        want.primary.extend(d.primary.iter().map(|&p| StreamPeak {
            index: p.round() as u64,
            position: p,
        }));
        want.secondary
            .extend(d.secondary.iter().map(|&p| StreamPeak {
                index: p.round() as u64,
                position: p,
            }));
    }
    for packets in [Packets::Fixed(32), Packets::Fixed(1), Packets::Random(77)] {
        let mut filt = EcgStreamFilter::new(FS, &cfg).unwrap();
        let mut det = StreamDetector::new(FS, 0, StreamConfig::default()).unwrap();
        let mut out = StreamPeaks::default();
        let mut gaps: Vec<Gap> = Vec::new();
        let mut per_segment: Vec<usize> = Vec::new();
        for (si, &(a, b)) in segments.iter().enumerate() {
            if si > 0 {
                let g = det.reset_at(a as u64, &mut out).unwrap();
                filt.reset();
                assert_eq!(g.end, a);
                gaps.push(g);
                per_segment.push(out.primary.len());
            }
            let mut at = a;
            for s in packet_sizes(b - a, packets) {
                let p = filt.process(&raw[at..at + s]);
                det.push(&p, &mut out).unwrap();
                at += s;
            }
        }
        det.flush(&mut out).unwrap();
        assert_eq!(
            bits(&positions(&out.primary)),
            bits(&positions(&want.primary)),
            "{packets:?}"
        );
        assert_eq!(
            bits(&positions(&out.secondary)),
            bits(&positions(&want.secondary))
        );
        assert_eq!(
            gaps,
            vec![
                Gap {
                    start: cut(40.0),
                    end: cut(43.3)
                },
                Gap {
                    start: cut(70.1),
                    end: cut(70.1)
                },
            ]
        );
        // segment 1's peaks were all emitted by the reset, none falls in the gap
        assert!(
            out.primary[..per_segment[0]]
                .iter()
                .all(|p| p.index < gaps[0].start as u64)
        );
        assert!(
            out.primary[per_segment[0]..]
                .iter()
                .all(|p| p.index >= gaps[0].end as u64)
        );
        // the returned gaps split the peak list into the three segments (no RR across a gap)
        let idx: Vec<usize> = out.primary.iter().map(|p| p.index as usize).collect();
        let segs = split_at_gaps(&idx, &gaps, FS, None).unwrap();
        assert_eq!(segs.len(), 3);
        assert_eq!(segs[0].len(), per_segment[0]);
        assert_eq!(segs.iter().map(Vec::len).sum::<usize>(), idx.len());
    }
}

#[test]
fn thirty_minutes_with_artifacts_no_lock_up_bounded_state_and_latency() {
    let minutes = 30usize;
    let (mut raw, r_true) = synth_raw(FS, minutes as f64 * 60.0, 1234, 0.03, 1.0);
    // 20 s of strong broadband noise at 7:00, a 3 s electrode pop (step + spikes) at 15:00 and a
    // 2 s flat line at 22:00
    let mut rng = Rng(99);
    let art = [(420.0, 440.0), (900.0, 903.0), (1320.0, 1322.0)];
    for (i, v) in raw.iter_mut().enumerate() {
        let t = i as f64 / FS;
        if (art[0].0..art[0].1).contains(&t) {
            *v += 8000.0 * rng.gauss();
        }
        if (art[1].0..art[1].1).contains(&t) {
            *v += 40_000.0 + if i % 97 == 0 { 200_000.0 } else { 0.0 };
        }
        if (art[2].0..art[2].1).contains(&t) {
            *v = 420.0;
        }
    }
    let cfg = EcgFilterConfig::default();
    let clean = ecg_clean_causal(&raw, FS, &cfg).unwrap();
    let batch = dual_detect_windowed(
        &clean,
        FS,
        0,
        &WindowConfig::default(),
        &DualConfig::default(),
    )
    .unwrap();

    let mut filt = EcgStreamFilter::new(FS, &cfg).unwrap();
    let mut det = StreamDetector::new(FS, 0, StreamConfig::default()).unwrap();
    let bytes = det.state_bytes();
    let mut out = StreamPeaks::default();
    let mut latency: Vec<u64> = Vec::new();
    let mut windows = 0u64;
    for p in raw.chunks(32) {
        let before = out.primary.len();
        let c = filt.process(p);
        det.push_with(&c, &mut out, |_| windows += 1).unwrap();
        for pk in &out.primary[before..] {
            latency.push(det.next_index() - pk.index);
        }
        assert_eq!(det.state_bytes(), bytes, "state must not grow");
    }
    det.flush(&mut out).unwrap();
    assert_eq!(bits(&positions(&out.primary)), bits(&batch.primary));
    assert_eq!(bits(&positions(&out.secondary)), bits(&batch.secondary));

    // no lock-up: every minute away from the artifacts has the true number of beats (±1)
    let per_minute = |p: &[f64]| -> Vec<usize> {
        let mut v = vec![0usize; minutes];
        for &x in p {
            let m = (x / FS / 60.0) as usize;
            if m < minutes {
                v[m] += 1;
            }
        }
        v
    };
    let truth = per_minute(&r_true);
    let live = per_minute(&positions(&out.primary));
    let sec = per_minute(&positions(&out.secondary));
    let near_artifact = |m: usize| {
        let (a, b) = (m as f64 * 60.0 - 12.0, (m + 1) as f64 * 60.0 + 12.0);
        art.iter().any(|&(s, e)| a < e && s < b)
    };
    for m in (0..minutes).filter(|&m| !near_artifact(m)) {
        assert!(
            live[m].abs_diff(truth[m]) <= 1 && sec[m].abs_diff(truth[m]) <= 1,
            "minute {m}: primary {} secondary {} true {}",
            live[m],
            sec[m],
            truth[m]
        );
    }
    let last_minutes = &live[minutes - 5..];
    assert!(last_minutes.iter().all(|&c| c > 60), "{last_minutes:?}");

    // latency and memory
    let max_lat = *latency.iter().max().unwrap();
    let min_lat = *latency.iter().min().unwrap();
    let mean_lat = latency.iter().sum::<u64>() as f64 / latency.len() as f64;
    assert!(max_lat <= det.max_latency_samples() + 32, "{max_lat}");
    assert!(min_lat + 32 >= 1024 + 2 * 77, "{min_lat}");
    eprintln!(
        "30 min, 32-sample packets: {} primary / {} secondary peaks ({} true beats), {windows} windows",
        out.primary.len(),
        out.secondary.len(),
        r_true.len()
    );
    eprintln!(
        "latency (peak → emitting sample): min {:.2} s, mean {:.2} s, max {:.2} s (bound {:.2} s)",
        min_lat as f64 / FS,
        mean_lat / FS,
        max_lat as f64 / FS,
        det.max_latency_samples() as f64 / FS
    );
    eprintln!(
        "state per stream: {bytes} bytes ({:.1} KiB); 500 streams {:.1} MiB",
        bytes as f64 / 1024.0,
        500.0 * bytes as f64 / 1048576.0
    );
    assert!(bytes <= 62 * 1024, "{bytes}");
}

#[test]
fn rejects_non_finite_samples_without_changing_the_output() {
    let (raw, _) = synth_raw(FS, 30.0, 41, 0.02, 1.0);
    let clean = ecg_clean_causal(&raw, FS, &EcgFilterConfig::default()).unwrap();
    let mut det = StreamDetector::new(FS, 0, StreamConfig::default()).unwrap();
    let mut out = StreamPeaks::default();
    for (i, p) in clean.chunks(32).enumerate() {
        if i == 100 {
            let mut bad = p.to_vec();
            bad[5] = f64::INFINITY;
            assert!(det.push(&bad, &mut out).is_err());
        }
        det.push(p, &mut out).unwrap();
    }
    det.flush(&mut out).unwrap();
    let batch = dual_detect_windowed(
        &clean,
        FS,
        0,
        &WindowConfig::default(),
        &DualConfig::default(),
    )
    .unwrap();
    assert_eq!(bits(&positions(&out.primary)), bits(&batch.primary));
}
