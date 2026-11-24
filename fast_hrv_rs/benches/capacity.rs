//! Capacity benchmark: CPU and memory per user of the planned app-pipeline worker pipeline
//! (std only, no external crates, synthetic data only).
//!
//! ```text
//! cargo bench --bench capacity                      # 500 users, 10 min of signal
//! cargo bench --bench capacity -- --users 200 --seconds 600
//! cargo bench --bench capacity -- --stream          # stage 2a through StreamDetector
//! ```
//!
//! Pipeline per user (512 Hz single-lead ECG in 32-sample packets, one worker thread):
//! 1. every packet: `EcgStreamFilter` (0.5–40 Hz, causal, state carried across packets);
//! 2. every 10 s: `dual_detect` on the window plus 2 s margins (14 s), full `SqiReport` on the 10 s
//!    core (flat line + saturation on raw counts, kSQI, pSQI + basSQI from one PSD, template
//!    correlation, bSQI, RR plausibility, zhao2018). With `--stream`, the clean samples go packet
//!    by packet into a `StreamDetector` (same `dual_detect` per window, de-duplicated peak output);
//!    the SQI then runs in its window callback on the detector's own buffer, and the session keeps
//!    no clean-sample ring;
//! 3. every 60 s: Kubios `fix_peaks` on the last 5 min of peaks, time domain on the last 60 s;
//! 4. every 5 min: full `hrv()` (time + frequency + nonlinear + asymmetry) on the last 5 min.
//!
//! Users are simulated interleaved (packet by packet, round robin over all sessions), so the
//! caches see the state of every session, as in the real worker. Two schedules are run:
//! "aligned" (every session starts at t = 0: all 10 s / 60 s / 5 min steps fall in the same
//! second, the worst case) and "staggered" (session start offsets spread over 5 min).
//! Timings are wall-clock time of a single thread (equal to CPU time when the thread is not
//! preempted; the process CPU/wall ratio is printed on Linux as a check).

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::VecDeque;
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

use fast_hrv::detect::{DualConfig, dual_detect};
use fast_hrv::ecg::{EcgFilterConfig, EcgStreamFilter};
use fast_hrv::filter::SosSection;
use fast_hrv::sqi::{SqiConfig, SqiEngine, SqiInput};
use fast_hrv::stream::{StreamConfig, StreamDetector, StreamPeaks, StreamWindow};

// -----------------------------------------------------------------------------
// Heap accounting (counts live and peak heap bytes; used for the memory figures)
// -----------------------------------------------------------------------------

struct CountingAlloc;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn add_live(n: usize) {
    let now = LIVE.fetch_add(n, Relaxed) + n;
    PEAK.fetch_max(now, Relaxed);
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            add_live(layout.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            add_live(layout.size());
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Relaxed);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            LIVE.fetch_sub(layout.size(), Relaxed);
            add_live(new_size);
        }
        p
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn live() -> usize {
    LIVE.load(Relaxed)
}

// -----------------------------------------------------------------------------
// Constants of the planned pipeline
// -----------------------------------------------------------------------------

const FS: u32 = 512;
const FSF: f64 = FS as f64;
const PACKET: usize = 32;
const PACKETS_PER_S: usize = FS as usize / PACKET;
const WINDOW: u64 = 10 * FS as u64; // 10 s core window
const MARGIN: u64 = 2 * FS as u64; // 2 s each side
/// Ring buffer of clean / raw samples: window + 2 margins + 1 s slack (late processing).
const RING: usize = (WINDOW + 2 * MARGIN) as usize + FS as usize;
const PEAK_HISTORY: u64 = 300 * FS as u64; // 5 min of peaks
const KUBIOS_EVERY_S: u64 = 60;
const HRV_EVERY_S: u64 = 300;
const TD_SPAN: u64 = 60 * FS as u64;
/// ADC: 1000 counts per mV, 14-bit range.
const ADC_PER_MV: f64 = 1000.0;
const ADC_MIN: i32 = -8192;
const ADC_MAX: i32 = 8191;

// -----------------------------------------------------------------------------
// Synthetic ECG source, generated sample by sample (one per user, no real data)
// -----------------------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Approximately standard normal (Irwin–Hall, 4 uniforms).
    fn gauss(&mut self) -> f64 {
        let s = self.uniform() + self.uniform() + self.uniform() + self.uniform();
        (s - 2.0) * 3f64.sqrt()
    }
}

/// Sine oscillator by rotation (cheap per sample).
struct Osc {
    c: f64,
    s: f64,
    cr: f64,
    sr: f64,
}
impl Osc {
    fn new(freq: f64, phase: f64) -> Self {
        let w = 2.0 * std::f64::consts::PI * freq / FSF;
        Osc {
            c: phase.cos(),
            s: phase.sin(),
            cr: w.cos(),
            sr: w.sin(),
        }
    }
    fn next(&mut self) -> f64 {
        let (c, s) = (
            self.c * self.cr - self.s * self.sr,
            self.s * self.cr + self.c * self.sr,
        );
        self.c = c;
        self.s = s;
        s
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Artifact {
    None,
    Motion,
    LeadOff,
    Saturation,
}

struct Synth {
    rng: Rng,
    t: u64,
    template: Vec<f64>,
    r_offset: u64,
    mean_rr_s: f64,
    next_r: u64,
    next_rr_scale: f64,
    beats: Vec<u64>, // template start samples of beats still being drawn
    noise_mv: f64,
    resp: Osc,
    lf: Osc,
    wander: Osc,
    mains: Osc,
    artifact: Artifact,
    artifact_end: u64,
    next_artifact: u64,
    artifact_kind: u32,
    held: i32,
    beats_drawn: u64,
}

impl Synth {
    /// User `u`: heart rate 60–100 bpm, R amplitude 0.8–1.6 mV, every 13th user inverted lead,
    /// white noise 0.01–0.04 mV, baseline wander, 50 Hz mains, ~1 % ectopic and ~0.25 % dropped
    /// beats, and every 1.5–4 min an artifact (3 s motion, 1.5 s lead-off flat line or 1 s
    /// saturation at the ADC rails).
    fn new(u: usize) -> Self {
        let mut rng =
            Rng(0x9E37_79B9_7F4A_7C15 ^ ((u as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03)));
        for _ in 0..4 {
            rng.next_u64();
        }
        let golden = (u as f64 * 0.618_033_988_75).fract();
        let hr = 60.0 + 40.0 * golden;
        let amp_r = 0.8 + 0.8 * rng.uniform();
        let sign = if u % 13 == 7 { -1.0 } else { 1.0 };
        // P, Q, R, S, T as Gaussians (amplitude mV, centre s, width s), template −0.3 … +0.5 s
        let waves = [
            (0.12, -0.20, 0.025),
            (-0.12, -0.035, 0.010),
            (amp_r, 0.0, 0.010),
            (-0.25, 0.035, 0.012),
            (0.30, 0.28, 0.045),
        ];
        let r_offset = (0.3 * FSF) as u64;
        let template: Vec<f64> = (0..(0.8 * FSF) as usize)
            .map(|i| {
                let ts = i as f64 / FSF - 0.3;
                sign * waves
                    .iter()
                    .map(|&(a, c, w)| a * (-((ts - c) / w).powi(2) / 2.0).exp())
                    .sum::<f64>()
            })
            .collect();
        let noise_mv = 0.01 + 0.03 * rng.uniform();
        let next_artifact = ((60.0 + 120.0 * rng.uniform()) * FSF) as u64;
        let phase = std::f64::consts::TAU * rng.uniform();
        Synth {
            t: 0,
            template,
            r_offset,
            mean_rr_s: 60.0 / hr,
            next_r: r_offset + (0.3 * FSF) as u64,
            next_rr_scale: 1.0,
            beats: Vec::with_capacity(4),
            noise_mv,
            resp: Osc::new(0.22 + 0.08 * rng.uniform(), phase),
            lf: Osc::new(0.1, phase * 0.7),
            wander: Osc::new(0.15 + 0.2 * rng.uniform(), phase * 1.3),
            mains: Osc::new(50.0, phase * 0.3),
            artifact: Artifact::None,
            artifact_end: 0,
            next_artifact,
            artifact_kind: u as u32,
            held: 0,
            beats_drawn: 0,
            rng,
        }
    }

    fn sample(&mut self) -> i32 {
        let t = self.t;
        let resp = self.resp.next();
        let lf = self.lf.next();
        let wander = self.wander.next();
        let mains = self.mains.next();
        if t == self.next_r {
            // draw this beat unless it is dropped (~0.25 %)
            if self.rng.uniform() > 0.0025 {
                self.beats.push(t - self.r_offset);
                self.beats_drawn += 1;
            }
            let mut rr =
                self.mean_rr_s * (1.0 + 0.05 * resp + 0.03 * lf + 0.015 * self.rng.gauss());
            rr *= self.next_rr_scale;
            self.next_rr_scale = 1.0;
            if self.rng.uniform() < 0.01 {
                // premature (ectopic) beat followed by a compensatory pause
                rr *= 0.65;
                self.next_rr_scale = 1.35;
            }
            self.next_r = t + (rr.max(0.3) * FSF) as u64;
        }
        let mut mv = 0.0;
        let len = self.template.len() as u64;
        self.beats.retain(|&b| b + len > t);
        for &b in &self.beats {
            if t >= b {
                mv += self.template[(t - b) as usize];
            }
        }
        mv += self.noise_mv * self.rng.gauss() + 0.12 * wander + 0.04 * resp + 0.03 * mains;

        if self.artifact == Artifact::None && t >= self.next_artifact {
            self.artifact_kind += 1;
            let (kind, dur) = match self.artifact_kind % 3 {
                0 => (Artifact::Motion, 3.0),
                1 => (Artifact::LeadOff, 1.5),
                _ => (Artifact::Saturation, 1.0),
            };
            self.artifact = kind;
            self.artifact_end = t + (dur * FSF) as u64;
            self.held = (mv * ADC_PER_MV) as i32;
        }
        let counts = match self.artifact {
            Artifact::None => (mv * ADC_PER_MV).round() as i32,
            Artifact::Motion => ((mv + 0.6 * self.rng.gauss() + 1.5 * wander) * ADC_PER_MV) as i32,
            Artifact::LeadOff => self.held,
            Artifact::Saturation => ((mv + 9.0) * ADC_PER_MV) as i32,
        };
        if self.artifact != Artifact::None && t + 1 >= self.artifact_end {
            self.artifact = Artifact::None;
            self.next_artifact = t + ((90.0 + 150.0 * self.rng.uniform()) * FSF) as u64;
        }
        self.t += 1;
        counts.clamp(ADC_MIN, ADC_MAX)
    }
}

// -----------------------------------------------------------------------------
// Worker-side session state and the four pipeline stages
// -----------------------------------------------------------------------------

const STAGES: [&str; 6] = [
    "1 EcgStreamFilter (per 32-sample packet)",
    "2a dual_detect, 14 s window", // replaced by STAGE_2A_STREAM with --stream
    "2b SqiReport (all indices), 10 s",
    "3a Kubios fix_peaks, last 5 min",
    "3b time domain, last 60 s",
    "4  hrv() full, last 5 min",
];
const STAGE_2A_STREAM: &str = "2a StreamDetector (dual), per window";

fn stage_name(i: usize, stream: bool) -> &'static str {
    if i == 1 && stream {
        STAGE_2A_STREAM
    } else {
        STAGES[i]
    }
}

/// Calls per second of signal of each stage (packets, windows, minutes, 5 minutes).
const RATE: [f64; 6] = [
    PACKETS_PER_S as f64,
    0.1,
    0.1,
    1.0 / KUBIOS_EVERY_S as f64,
    1.0 / KUBIOS_EVERY_S as f64,
    1.0 / HRV_EVERY_S as f64,
];

#[derive(Default, Clone)]
struct Stage {
    calls: u64,
    total_ns: u128,
    per_call_ns: Vec<u64>, // not filled for the filter (timed in blocks)
}

impl Stage {
    fn add(&mut self, ns: u64) {
        self.calls += 1;
        self.total_ns += ns as u128;
        self.per_call_ns.push(ns);
    }
    fn mean_us(&self) -> f64 {
        if self.calls == 0 {
            f64::NAN
        } else {
            self.total_ns as f64 / self.calls as f64 / 1e3
        }
    }
    fn pct_us(&self, p: f64) -> f64 {
        if self.per_call_ns.is_empty() {
            return f64::NAN;
        }
        let mut v = self.per_call_ns.clone();
        v.sort_unstable();
        let i = ((v.len() - 1) as f64 * p).round() as usize;
        v[i] as f64 / 1e3
    }
}

/// Per-worker scratch buffers and engines (shared by all sessions; the worker is sequential).
struct Worker {
    sqi: SqiEngine,
    dual: DualConfig,
    chunk: Vec<f64>,
    raw_core: Vec<i32>,
    stages: [Stage; 6],
    sink: f64,
    /// Primary R peaks kept in core windows (sanity check of the synthetic signal).
    peaks_found: u64,
    /// Peaks emitted by one `StreamDetector::push` (scratch).
    out: StreamPeaks,
}

impl Worker {
    fn new() -> Self {
        let sqi_cfg = SqiConfig {
            adc_range: Some((ADC_MIN as f64, ADC_MAX as f64)),
            ..SqiConfig::default()
        };
        Worker {
            sqi: SqiEngine::new(sqi_cfg),
            dual: DualConfig::default(),
            chunk: Vec::with_capacity(RING),
            raw_core: Vec::with_capacity(WINDOW as usize),
            stages: Default::default(),
            sink: 0.0,
            peaks_found: 0,
            out: StreamPeaks::default(),
        }
    }
}

/// Everything the worker keeps per user between packets.
struct Session {
    filt: EcgStreamFilter,
    n: u64,          // samples received
    clean: Vec<f64>, // ring of filtered samples (RING); empty with --stream
    raw: Vec<i32>,   // ring of raw ADC counts (RING)
    /// `--stream`: the detector (holds the clean samples of the current chunk) and the last
    /// filtered packet waiting to be pushed into it.
    det: Option<StreamDetector>,
    pkt: [f64; PACKET],
    pkt_len: usize,
    next_window: u64,
    peaks: VecDeque<usize>, // absolute primary R-peak indices, last 5 min
    corrected: Vec<usize>,  // Kubios output of the last 60 s step
}

impl Session {
    fn new(stream: bool) -> Self {
        Session {
            filt: EcgStreamFilter::new(FSF, &EcgFilterConfig::default()).expect("filter design"),
            n: 0,
            clean: if stream { Vec::new() } else { vec![0.0; RING] },
            raw: vec![0; RING],
            det: stream.then(|| {
                StreamDetector::new(FSF, 0, StreamConfig::default()).expect("stream detector")
            }),
            pkt: [0.0; PACKET],
            pkt_len: 0,
            next_window: 0,
            peaks: VecDeque::with_capacity(512),
            corrected: Vec::new(),
        }
    }

    /// Heap + inline bytes of the session (capacities, not lengths).
    fn bytes(&self) -> usize {
        let sections = self.filt.filter().sections().len();
        std::mem::size_of::<Session>()
            + sections * (std::mem::size_of::<SosSection>() + 2 * std::mem::size_of::<[f64; 2]>())
            + self.clean.capacity() * 8
            + self.raw.capacity() * 4
            + self.peaks.capacity() * std::mem::size_of::<usize>()
            + self.corrected.capacity() * std::mem::size_of::<usize>()
            + self.det.as_ref().map_or(0, |d| {
                d.state_bytes() - std::mem::size_of::<StreamDetector>()
            })
    }

    /// Stage 1: one packet of raw counts.
    fn push_packet(&mut self, packet: &[i32]) {
        let mut buf = [0.0f64; PACKET];
        let buf = &mut buf[..packet.len()];
        for (b, &r) in buf.iter_mut().zip(packet) {
            *b = r as f64;
        }
        self.filt.process_inplace(buf);
        if self.det.is_some() {
            for &r in packet {
                self.raw[(self.n % RING as u64) as usize] = r;
                self.n += 1;
            }
            self.pkt[..buf.len()].copy_from_slice(buf);
            self.pkt_len = buf.len();
            return;
        }
        for (&r, &c) in packet.iter().zip(buf.iter()) {
            let i = (self.n % RING as u64) as usize;
            self.raw[i] = r;
            self.clean[i] = c;
            self.n += 1;
        }
    }

    /// Stage 2a/2b with `--stream`: push the last filtered packet into the detector. The detectors
    /// (and the SQI, in the window callback) run when the packet completes a window. Returns the
    /// SQI time spent inside the call (ns), which the caller removes from the 2a block time.
    fn push_stream(&mut self, w: &mut Worker) -> u64 {
        let Some(det) = self.det.as_mut() else {
            return 0;
        };
        let completes = self.n >= det.next_window_ready_at();
        let t0 = completes.then(Instant::now);
        let raw = &self.raw;
        let Worker {
            sqi,
            raw_core,
            stages,
            sink,
            out,
            ..
        } = w;
        out.clear();
        let mut sqi_ns = 0u64;
        let mut on_window = |win: &StreamWindow<'_>| {
            let t1 = Instant::now();
            let r = win.core_range();
            let prim: Vec<usize> = win
                .core_indices(win.primary)
                .into_iter()
                .map(|i| i - r.start)
                .collect();
            let sec: Vec<usize> = win
                .secondary
                .map(|s| win.core_indices(s))
                .unwrap_or_default()
                .into_iter()
                .map(|i| i - r.start)
                .collect();
            raw_core.clear();
            for a in win.core.0..win.core.1 {
                raw_core.push(raw[(a % RING as u64) as usize]);
            }
            let input = SqiInput {
                peaks_secondary: Some(&sec),
                raw: Some(raw_core),
                ..SqiInput::new(&win.samples[r], FSF, &prim)
            };
            let rep = sqi.report(&input);
            *sink +=
                rep.ksqi.unwrap_or(0.0) + rep.issues.len() as f64 + rep.zhao.reasons.len() as f64;
            let ns = t1.elapsed().as_nanos() as u64;
            stages[2].add(ns);
            sqi_ns += ns;
        };
        det.push_with(&self.pkt[..self.pkt_len], out, &mut on_window)
            .expect("finite samples");
        if let Some(t0) = t0 {
            let ns = t0.elapsed().as_nanos() as u64;
            w.stages[1].calls += 1;
            w.stages[1].per_call_ns.push(ns.saturating_sub(sqi_ns));
        }
        w.peaks_found += w.out.primary.len() as u64;
        for p in &w.out.primary {
            self.peaks.push_back(p.index as usize);
        }
        let done = det.next_window_ready_at() - MARGIN - WINDOW; // end of the last processed core
        let horizon = done.saturating_sub(PEAK_HISTORY) as usize;
        while self.peaks.front().is_some_and(|&p| p < horizon) {
            self.peaks.pop_front();
        }
        sqi_ns
    }

    /// Stages 2–4 that are due after the samples received so far.
    fn tick(&mut self, w: &mut Worker) {
        if let Some(det) = &self.det {
            // windows ran inside `push_stream`; keep the core grid for the 60 s step
            self.next_window = (det.next_window_ready_at() - MARGIN) / WINDOW - 1;
        }
        while self.det.is_none() && self.n >= (self.next_window + 1) * WINDOW + MARGIN {
            self.window(w, self.next_window);
            self.next_window += 1;
        }
        let sec = self.n / FS as u64;
        if self.n % FS as u64 == 0 && sec > 0 {
            if sec % KUBIOS_EVERY_S == 0 {
                self.minute(w);
            }
            if sec % HRV_EVERY_S == 0 {
                self.five_minutes(w);
            }
        }
    }

    fn window(&mut self, w: &mut Worker, k: u64) {
        let t0 = Instant::now();
        let core0 = k * WINDOW;
        let c0 = core0.saturating_sub(MARGIN);
        let c1 = core0 + WINDOW + MARGIN;
        w.chunk.clear();
        w.raw_core.clear();
        for a in c0..c1 {
            w.chunk.push(self.clean[(a % RING as u64) as usize]);
        }
        for a in core0..core0 + WINDOW {
            w.raw_core.push(self.raw[(a % RING as u64) as usize]);
        }
        let cs = (core0 - c0) as usize;
        let ce = cs + WINDOW as usize;
        let Ok(d) = dual_detect(&w.chunk, FSF, &w.dual) else {
            return;
        };
        let prim: Vec<usize> = d
            .primary
            .indices
            .iter()
            .filter(|&&i| i >= cs && i < ce)
            .map(|&i| i - cs)
            .collect();
        let sec: Vec<usize> = d
            .secondary
            .indices
            .iter()
            .filter(|&&i| i >= cs && i < ce)
            .map(|&i| i - cs)
            .collect();
        w.peaks_found += prim.len() as u64;
        for &p in &prim {
            self.peaks.push_back(core0 as usize + p);
        }
        let horizon = (core0 + WINDOW).saturating_sub(PEAK_HISTORY) as usize;
        while self.peaks.front().is_some_and(|&p| p < horizon) {
            self.peaks.pop_front();
        }
        let t1 = Instant::now();
        let input = SqiInput {
            peaks_secondary: Some(&sec),
            raw: Some(&w.raw_core),
            ..SqiInput::new(&w.chunk[cs..ce], FSF, &prim)
        };
        let rep = w.sqi.report(&input);
        w.sink += rep.ksqi.unwrap_or(0.0) + rep.issues.len() as f64 + rep.zhao.reasons.len() as f64;
        let t2 = Instant::now();
        w.stages[1].add((t1 - t0).as_nanos() as u64);
        w.stages[2].add((t2 - t1).as_nanos() as u64);
    }

    fn minute(&mut self, w: &mut Worker) {
        let t0 = Instant::now();
        let peaks: Vec<usize> = self.peaks.iter().copied().collect();
        self.corrected = fast_hrv::fix_peaks_default(&peaks, FS).0;
        let t1 = Instant::now();
        let end = self.next_window * WINDOW;
        let from = end.saturating_sub(TD_SPAN) as usize;
        let start = self.corrected.partition_point(|&p| p < from);
        let rri = fast_hrv::rri_from_peaks(&self.corrected[start..], FS);
        let td = fast_hrv::time_domain(&rri);
        w.sink += black_box(td.rmssd).clamp(0.0, 1.0);
        let t2 = Instant::now();
        w.stages[3].add((t1 - t0).as_nanos() as u64);
        w.stages[4].add((t2 - t1).as_nanos() as u64);
    }

    fn five_minutes(&mut self, w: &mut Worker) {
        let t0 = Instant::now();
        let rri = fast_hrv::rri_from_peaks(&self.corrected, FS);
        let r = fast_hrv::hrv(&rri);
        w.sink += black_box(r.frequency.lf_hf).clamp(0.0, 1.0);
        w.stages[5].add(t0.elapsed().as_nanos() as u64);
    }
}

// -----------------------------------------------------------------------------
// Fleet simulation
// -----------------------------------------------------------------------------

struct FleetResult {
    stages: [Stage; 6],
    /// CPU (ms) spent in each global second of the measured region, all users together.
    second_ms: Vec<f64>,
    measure_from: u64,
    session_bytes: usize,
    sessions_heap_measured: usize,
    user_seconds: f64,
    peaks_found: u64,
    beats_drawn: u64,
}

/// `stagger_s = 0`: every session starts at t = 0. Otherwise session `u` starts at
/// `floor(u · stagger_s / users)` s. Seconds `measure_from..seconds` are recorded per second.
fn run_fleet(
    users: usize,
    seconds: u64,
    stagger_s: u64,
    measure_from: u64,
    stream: bool,
) -> FleetResult {
    let mut synth: Vec<Synth> = (0..users).map(Synth::new).collect();
    let starts: Vec<u64> = (0..users)
        .map(|u| (u as u64 * stagger_s) / users as u64)
        .collect();
    let mut packets = vec![0i32; users * FS as usize];
    let mut worker = Worker::new();
    let heap0 = live();
    let mut sessions: Vec<Session> = (0..users).map(|_| Session::new(stream)).collect();
    let sessions_heap_initial = live() - heap0;
    let mut second_ms = Vec::new();
    let mut user_seconds = 0.0;

    for sec in 0..seconds {
        let active: Vec<usize> = (0..users).filter(|&u| starts[u] <= sec).collect();
        // generate one second for every active user (not timed)
        for &u in &active {
            for v in &mut packets[u * FS as usize..(u + 1) * FS as usize] {
                *v = synth[u].sample();
            }
        }
        // stage 1: packets arrive interleaved over users (with --stream each packet round is
        // followed by a round of StreamDetector pushes, timed as stage 2a / 2b)
        let before: u128 = worker.stages[1..].iter().map(|s| s.total_ns).sum();
        let mut filt_ns = 0u64;
        for p in 0..PACKETS_PER_S {
            let t0 = Instant::now();
            for &u in &active {
                let base = u * FS as usize + p * PACKET;
                sessions[u].push_packet(&packets[base..base + PACKET]);
            }
            filt_ns += t0.elapsed().as_nanos() as u64;
            if stream {
                let t1 = Instant::now();
                let mut sqi_ns = 0u64;
                for &u in &active {
                    sqi_ns += sessions[u].push_stream(&mut worker);
                }
                let block = t1.elapsed().as_nanos() as u64;
                worker.stages[1].total_ns += block.saturating_sub(sqi_ns) as u128;
            }
        }
        let st = &mut worker.stages[0];
        st.calls += (PACKETS_PER_S * active.len()) as u64;
        st.total_ns += filt_ns as u128;
        // stages 2–4
        for &u in &active {
            sessions[u].tick(&mut worker);
        }
        let after: u128 = worker.stages[1..].iter().map(|s| s.total_ns).sum();
        if sec >= measure_from {
            second_ms.push((filt_ns as u128 + after - before) as f64 / 1e6);
        }
        user_seconds += active.len() as f64;
    }
    black_box(worker.sink);
    let session_bytes = sessions.iter().map(Session::bytes).sum::<usize>() / users;
    FleetResult {
        stages: worker.stages,
        second_ms,
        measure_from,
        session_bytes,
        sessions_heap_measured: sessions_heap_initial / users,
        user_seconds,
        peaks_found: worker.peaks_found,
        beats_drawn: synth.iter().map(|s| s.beats_drawn).sum(),
    }
}

/// µs of CPU per user per second of signal, per stage (mean call cost × call rate).
fn per_user_second_us(stages: &[Stage; 6]) -> [f64; 6] {
    let mut out = [0.0; 6];
    for i in 0..6 {
        out[i] = stages[i].mean_us() * RATE[i];
    }
    out
}

// -----------------------------------------------------------------------------
// Environment report
// -----------------------------------------------------------------------------

fn cpu_model() -> String {
    if let Ok(s) = std::fs::read_to_string("/proc/cpuinfo") {
        if let Some(l) = s.lines().find(|l| l.starts_with("model name")) {
            return l.split(':').nth(1).unwrap_or("?").trim().to_string();
        }
    }
    std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "unknown".into())
}

/// (utime + stime in s, last core) of this process, Linux only.
fn proc_stat() -> Option<(f64, i64)> {
    let s = std::fs::read_to_string("/proc/self/stat").ok()?;
    let rest = &s[s.rfind(')')? + 2..];
    let f: Vec<&str> = rest.split_whitespace().collect();
    // fields after ")" start at field 3 (state); utime = 14, stime = 15, processor = 39
    let ut: f64 = f.get(11)?.parse().ok()?;
    let st: f64 = f.get(12)?.parse().ok()?;
    let cpu: i64 = f.get(36)?.parse().ok()?;
    Some(((ut + st) / 100.0, cpu))
}

fn core_mhz(core: i64) -> Option<f64> {
    let s = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    let mut cur = -1i64;
    for l in s.lines() {
        if l.starts_with("processor") {
            cur = l.split(':').nth(1)?.trim().parse().ok()?;
        } else if l.starts_with("cpu MHz") && cur == core {
            return l.split(':').nth(1)?.trim().parse().ok();
        }
    }
    None
}

fn target_features() -> String {
    let mut v = Vec::new();
    if cfg!(target_feature = "sse2") {
        v.push("sse2");
    }
    if cfg!(target_feature = "sse4.2") {
        v.push("sse4.2");
    }
    if cfg!(target_feature = "avx") {
        v.push("AVX");
    }
    if cfg!(target_feature = "avx2") {
        v.push("AVX2");
    }
    if cfg!(target_feature = "fma") {
        v.push("FMA");
    }
    v.join(" ")
}

#[cfg(target_arch = "x86_64")]
fn cpu_has_avx() -> &'static str {
    if std::is_x86_feature_detected!("avx") {
        "yes"
    } else {
        "no"
    }
}
#[cfg(not(target_arch = "x86_64"))]
fn cpu_has_avx() -> &'static str {
    "n/a"
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------

fn fmt1(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.1}")
    } else {
        "-".into()
    }
}

fn arg(name: &str, default: u64) -> u64 {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let users = arg("--users", 500).max(1) as usize;
    let stream = std::env::args().any(|a| a == "--stream");
    let seconds = arg("--seconds", 600).max(2 * HRV_EVERY_S);
    let wall0 = Instant::now();
    let cpu0 = proc_stat();

    println!("fast-hrv {} capacity benchmark", env!("CARGO_PKG_VERSION"));
    println!("  CPU: {}", cpu_model());
    println!(
        "  build target features: {} | build assumes AVX: {} | CPU supports AVX: {}",
        target_features(),
        if cfg!(target_feature = "avx") {
            "YES"
        } else {
            "no"
        },
        cpu_has_avx()
    );
    println!(
        "  {users} users, {seconds} s of synthetic 512 Hz ECG each, 32-sample packets, one thread"
    );
    println!(
        "  stage 2a: {}",
        if stream {
            "StreamDetector (packet by packet, dual detection, SQI in the window callback)"
        } else {
            "dual_detect on a 14 s chunk copied from the session ring (default; --stream for StreamDetector)"
        }
    );

    // SQI engine size (one per worker): heap after the first report.
    let mut tmp = Synth::new(1);
    let probe_raw: Vec<i32> = (0..WINDOW).map(|_| tmp.sample()).collect();
    let probe: Vec<f64> = probe_raw.iter().map(|&r| r as f64).collect();
    let h0 = live();
    let mut eng = SqiEngine::new(SqiConfig::default());
    let pk: Vec<usize> = (1..12).map(|i| i * 420).collect();
    black_box(eng.report(&SqiInput::new(&probe, FSF, &pk)));
    let sqi_engine_bytes = live() - h0 + std::mem::size_of::<SqiEngine>();
    drop(eng);

    // 1 user (hot caches), aligned
    PEAK.store(live(), Relaxed);
    let peak_base = live();
    let one = run_fleet(1, seconds, 0, 0, stream);
    let worker_peak_extra = PEAK.load(Relaxed) - peak_base;
    // N users interleaved, aligned (all steps in the same second) and staggered
    let aligned = run_fleet(users, seconds, 0, 0, stream);
    let staggered = run_fleet(users, seconds, HRV_EVERY_S, HRV_EVERY_S, stream);

    let us = per_user_second_us(&staggered.stages);
    let total: f64 = us.iter().sum();
    let total_one: f64 = per_user_second_us(&one.stages).iter().sum();
    let total_aligned: f64 = per_user_second_us(&aligned.stages).iter().sum();

    println!();
    println!(
        "Per-stage cost, {users} users interleaved, staggered schedule ({:.0} user-seconds)",
        staggered.user_seconds
    );
    println!(
        "| {:<41} | {:>8} | {:>9} | {:>9} | {:>9} | {:>11} | {:>6} |",
        "stage", "calls", "mean µs", "p99 µs", "max µs", "µs/user/s", "share"
    );
    println!(
        "|{:-<43}|{:-<10}|{:-<11}|{:-<11}|{:-<11}|{:-<13}|{:-<8}|",
        "", "", "", "", "", "", ""
    );
    for i in 0..6 {
        let s = &staggered.stages[i];
        println!(
            "| {:<41} | {:>8} | {:>9.2} | {:>9} | {:>9} | {:>11.2} | {:>5.1}% |",
            stage_name(i, stream),
            s.calls,
            s.mean_us(),
            fmt1(s.pct_us(0.99)),
            fmt1(s.pct_us(1.0)),
            us[i],
            100.0 * us[i] / total
        );
    }
    println!(
        "| {:<41} | {:>8} | {:>9} | {:>9} | {:>9} | {:>11.2} | {:>5.1}% |",
        "TOTAL", "", "", "", "", total, 100.0
    );

    println!(
        "  sanity: {} R peaks detected for {} synthetic beats drawn ({:.1} %; windows still waiting for their margin, artifacts and dropped/ectopic beats explain the rest)",
        staggered.peaks_found,
        staggered.beats_drawn,
        100.0 * staggered.peaks_found as f64 / staggered.beats_drawn as f64
    );

    println!();
    println!("Users per core (µs of CPU per user per second of signal)");
    println!(
        "| {:<44} | {:>9} | {:>10} | {:>10} |",
        "run", "µs/user/s", "100% core", "50% core"
    );
    println!("|{:-<46}|{:-<11}|{:-<12}|{:-<12}|", "", "", "", "");
    for (name, t) in [
        (format!("{users} users interleaved, staggered"), total),
        (format!("{users} users interleaved, aligned"), total_aligned),
        ("1 user (hot caches)".to_string(), total_one),
    ] {
        println!(
            "| {:<44} | {:>9.2} | {:>10.0} | {:>10.0} |",
            name,
            t,
            1e6 / t,
            0.5e6 / t
        );
    }
    println!(
        "  {users} users use {:.2} % of one core on average (staggered).",
        100.0 * users as f64 * total / 1e6
    );

    // spikes
    let worst = |r: &FleetResult| {
        let (i, &m) = r
            .second_ms
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("non-empty");
        (r.measure_from + i as u64, m)
    };
    let mean_ms = |r: &FleetResult| r.second_ms.iter().sum::<f64>() / r.second_ms.len() as f64;
    let (wa_s, wa) = worst(&aligned);
    let (ws_s, ws) = worst(&staggered);
    let over = |r: &FleetResult, lim: f64| r.second_ms.iter().filter(|&&v| v > lim).count();
    let a = &aligned.stages;
    let step5 = a[3].mean_us() + a[4].mean_us() + a[5].mean_us();
    let step5_p99 = a[3].pct_us(0.99) + a[4].pct_us(0.99) + a[5].pct_us(0.99);
    let all_steps = step5 + a[1].mean_us() + a[2].mean_us() + a[0].mean_us() * PACKETS_PER_S as f64;
    println!();
    println!("CPU per wall-clock second, all {users} users together (budget 1000 ms)");
    println!(
        "| {:<52} | {:>10} | {:>11} |",
        "schedule", "mean ms/s", "worst ms/s"
    );
    println!("|{:-<54}|{:-<12}|{:-<13}|", "", "", "");
    println!(
        "| {:<52} | {:>10.1} | {:>8.1} @{:<3} |",
        "aligned (all sessions start together)",
        mean_ms(&aligned),
        wa,
        wa_s
    );
    println!(
        "| {:<52} | {:>10.1} | {:>8.1} @{:<3} |",
        "staggered (start offsets spread over 5 min)",
        mean_ms(&staggered),
        ws,
        ws_s
    );
    println!(
        "  aligned: seconds over 500 ms: {}, over 1000 ms: {} (of {})",
        over(&aligned, 500.0),
        over(&aligned, 1000.0),
        aligned.second_ms.len()
    );
    println!(
        "  5-min step (Kubios + time domain + hrv) per user: mean {:.0} µs, p99 {:.0} µs → {users} users in one second: {:.0} ms (p99 {:.0} ms)",
        step5,
        step5_p99,
        users as f64 * step5 / 1e3,
        users as f64 * step5_p99 / 1e3
    );
    println!(
        "  every step at once (1 s packets + window + 60 s + 5 min) per user: {:.0} µs → {users} users: {:.0} ms",
        all_steps,
        users as f64 * all_steps / 1e3
    );

    println!();
    println!("Memory");
    println!(
        "  per session ({}): {:.1} KiB (heap at creation {:.1} KiB)",
        if stream {
            "filter state + StreamDetector + 15 s raw i32 ring + 5 min peaks + Kubios output"
        } else {
            "filter state + 15 s clean f64 ring + 15 s raw i32 ring + 5 min peaks + Kubios output"
        },
        staggered.session_bytes as f64 / 1024.0,
        staggered.sessions_heap_measured as f64 / 1024.0
    );
    println!(
        "  {users} sessions: {:.1} MiB | per worker: SqiEngine {:.1} KiB, scratch + transient peak (1-user run) {:.1} KiB",
        users as f64 * staggered.session_bytes as f64 / 1048576.0,
        sqi_engine_bytes as f64 / 1024.0,
        worker_peak_extra.saturating_sub(one.session_bytes) as f64 / 1024.0
    );
    println!(
        "  (keeping 5 min of samples instead: f64 {:.0} KiB, i16 {:.0} KiB per user)",
        (300 * FS as usize * 8) as f64 / 1024.0,
        (300 * FS as usize * 2) as f64 / 1024.0
    );

    let wall = wall0.elapsed().as_secs_f64();
    match (cpu0, proc_stat()) {
        (Some((c0, _)), Some((c1, core))) => {
            let mhz = core_mhz(core).map_or(String::from("?"), |m| format!("{m:.0} MHz"));
            println!(
                "\n  run: {wall:.1} s wall, process CPU/wall {:.2}, last core {core} ({mhz})",
                (c1 - c0) / wall
            );
        }
        _ => println!("\n  run: {wall:.1} s wall"),
    }
}
