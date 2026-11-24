//! Live (packet-by-packet) R-peak detection with exactly the output of the windowed batch path.
//!
//! [`StreamDetector`] consumes clean ECG samples (e.g. the output of
//! [`crate::ecg::EcgStreamFilter`]) in packets of any size and runs the windowed pipeline of
//! [`crate::detect`]: in every core window `[k·W, (k+1)·W)` of the absolute sample axis (default
//! 10 s) plus a margin `M` on each side (default 2 s), the primary detector (Elgendi + refinement)
//! and, optionally, the secondary detector of [`crate::detect::dual_detect`] (NeuroKit2 method,
//! refined with the primary's sign). A window is processed as soon as its right margin is complete
//! (sample `(k+1)·W + M` received), and the peaks are de-duplicated across window overlaps with the
//! rule of [`crate::detect::detect_windowed`].
//!
//! # Equality with the batch path
//! For a segment of samples starting at absolute index `s0`, the concatenated output is
//! **bit-identical** to [`crate::detect::elgendi_refined_windowed`] (primary) and
//! [`crate::detect::dual_detect_windowed`] (primary and secondary) on the same samples with
//! `start_sample = s0`, for any packetisation:
//! - every window sees exactly the same chunk (`[k·W − M, (k+1)·W + M)` clipped to the segment), so
//!   the detectors return the same values;
//! - the de-duplication is the batch algorithm (a greedy pass over the peaks sorted by position,
//!   window, detection order), fed incrementally: after window `k` only peaks lying before
//!   `(k+1)·W − dedup − 1` are fed (no later window can produce a peak there), and the last kept
//!   peak is emitted only when no later peak can come within `dedup` of it.
//!
//! # Latency
//! A peak in core `k` is emitted when sample `(k+1)·W + M` arrives, i.e. between `M + 2·dedup`
//! (end of the core) and `W + M` (start of the core) after the R peak: with the defaults 2.3–12 s,
//! ~7 s on average. A peak in the last `2·dedup + 2` samples of a core (≤ 0.3 s; at most one per
//! window boundary) can be a duplicate of a peak of the next window, so it waits for that window:
//! worst case `W + M + 2·dedup` + 2 samples (~12.3 s, see [`StreamDetector::max_latency_samples`]).
//! The per-window callback ([`StreamWindow`], e.g. for signal quality on the core) fires `W + M`
//! after the start of the core.
//!
//! # Memory
//! One contiguous buffer of at most `W + 2·M` samples (`f64`, allocated once: 7168 samples =
//! 56 KiB at 512 Hz with the defaults) plus a few pending peaks; see [`StreamDetector::state_bytes`].
//! The buffer is the chunk the detectors read, so no copy is made per window; after a window only
//! the `2·M` samples shared with the next chunk are kept (moved to the front). Detector scratch
//! (filtered signal, moving averages) is allocated per window and freed, as in the batch path.
//!
//! # Gaps
//! [`StreamDetector::reset_at`] ends the current segment (its last windows are processed with
//! clipped chunks, exactly like the batch path at the end of a recording) and starts a new segment
//! at a later absolute index. It returns the [`crate::gaps::Gap`] between the two segments, for
//! [`crate::gaps::split_at_gaps`] / [`crate::gaps::rri_with_gaps`] on the emitted indices.

use crate::detect::{
    DualConfig, RefinedPeaks, WindowConfig, detect_windowed, dual_detect, elgendi_peaks,
    refine_peaks,
};
use crate::error::HrvError;
use crate::gaps::Gap;
use crate::sqi::Bsqi;

/// Parameters of [`StreamDetector`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StreamConfig {
    /// Window grid, margin and de-duplication (default 10 s / 2 s / 0.15 s).
    pub window: WindowConfig,
    /// Detectors: `dual.elgendi` + `dual.refine` give the primary peaks; with
    /// [`StreamConfig::secondary`] the whole [`dual_detect`] runs per window.
    pub dual: DualConfig,
    /// Also run the secondary detector (NeuroKit2 method) and the per-window bSQI, as
    /// [`dual_detect`]. Default `true`. With `false` only the primary detector runs (output equal to
    /// [`crate::detect::elgendi_refined_windowed`], cheaper).
    pub secondary: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            window: WindowConfig::default(),
            dual: DualConfig::default(),
            secondary: true,
        }
    }
}

/// One confirmed R peak.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct StreamPeak {
    /// Absolute sample index of the extremum.
    pub index: u64,
    /// Absolute sub-sample position (samples), bit-identical to the batch positions.
    pub position: f64,
}

/// Peaks emitted by one or more [`StreamDetector`] calls (appended, in increasing position).
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct StreamPeaks {
    /// Primary detector (Elgendi + refinement).
    pub primary: Vec<StreamPeak>,
    /// Secondary detector (NeuroKit2 method + refinement); empty when
    /// [`StreamConfig::secondary`] is off.
    pub secondary: Vec<StreamPeak>,
}

impl StreamPeaks {
    /// Remove all peaks (keeps the allocations).
    pub fn clear(&mut self) {
        self.primary.clear();
        self.secondary.clear();
    }

    /// `true` when no peak is stored.
    pub fn is_empty(&self) -> bool {
        self.primary.is_empty() && self.secondary.is_empty()
    }
}

/// A processed window, passed to the callbacks of [`StreamDetector::push_with`] and friends.
/// Everything is relative to [`StreamWindow::chunk_start`] unless stated otherwise.
#[derive(Debug, Clone, Copy)]
pub struct StreamWindow<'a> {
    /// Window number `k` (core `[k·W, (k+1)·W)` on the absolute axis).
    pub window: u64,
    /// Absolute index of `samples[0]`.
    pub chunk_start: u64,
    /// Absolute core `[k·W, (k+1)·W)` (not clipped to the segment).
    pub core: (u64, u64),
    /// The chunk the detectors ran on (core plus margins, clipped to the segment).
    pub samples: &'a [f64],
    /// Primary peaks of this chunk before de-duplication (chunk-relative).
    pub primary: &'a RefinedPeaks,
    /// Primary detector output before refinement (chunk-relative).
    pub primary_raw: &'a [usize],
    /// Secondary peaks of this chunk (chunk-relative), when enabled.
    pub secondary: Option<&'a RefinedPeaks>,
    /// bSQI of the chunk ([`crate::detect::DualDetection::bsqi`]), when the secondary is enabled.
    pub bsqi: Option<Bsqi>,
}

impl StreamWindow<'_> {
    /// The core window as a range into [`StreamWindow::samples`] (clipped to the chunk).
    pub fn core_range(&self) -> std::ops::Range<usize> {
        let end = self.chunk_start + self.samples.len() as u64;
        let lo = self.core.0.clamp(self.chunk_start, end);
        let hi = self.core.1.clamp(self.chunk_start, end);
        (lo - self.chunk_start) as usize..(hi - self.chunk_start) as usize
    }

    /// Chunk-relative peak indices of `peaks` that fall in the core.
    pub fn core_indices(&self, peaks: &RefinedPeaks) -> Vec<usize> {
        let r = self.core_range();
        peaks
            .indices
            .iter()
            .copied()
            .filter(|i| r.contains(i))
            .collect()
    }
}

/// A peak found in a window, waiting for de-duplication.
#[derive(Debug, Clone, Copy)]
struct Found {
    pos: f64,
    index: u64,
    window: u64,
    /// Detection order within the window (the batch path's stable-sort tie-break).
    order: u32,
}

impl Found {
    fn peak(&self) -> StreamPeak {
        StreamPeak {
            index: self.index,
            position: self.pos,
        }
    }
}

/// Depth of the absolute position `a` inside the core of window `k` (negative outside), with the
/// arithmetic of [`detect_windowed`].
fn depth(a: f64, k: u64, w: u64) -> f64 {
    let (core0, core1) = (k * w, (k + 1) * w);
    (a - core0 as f64).min(core1 as f64 - a)
}

/// Incremental form of the de-duplication pass of [`detect_windowed`].
#[derive(Debug, Clone)]
struct Dedup {
    /// Found peaks not yet fed to the greedy pass.
    pending: Vec<Found>,
    /// Last kept peak (may still be replaced by a duplicate from the next window).
    last: Option<Found>,
}

impl Dedup {
    fn new(capacity: usize) -> Self {
        Self {
            pending: Vec::with_capacity(capacity),
            last: None,
        }
    }

    /// Most peaks that can be pending at once: one window's peaks (`W + 2·dedup` samples) plus the
    /// leftovers of the previous window (`2·dedup + 1` samples), detections at least `spacing`
    /// samples apart. Without a refractory period, 64 (the buffer then grows only if needed).
    fn capacity(w: u64, dedup: f64, spacing: f64) -> usize {
        if spacing >= 1.0 {
            ((w as f64 + 4.0 * dedup + 2.0) / spacing).ceil() as usize + 3
        } else {
            64
        }
    }

    /// One step of the batch loop (`kept.last_mut()` is `self.last`; earlier kept peaks are final).
    fn feed(last: &mut Option<Found>, f: Found, w: u64, dedup: f64, out: &mut Vec<StreamPeak>) {
        match last {
            Some(l) if l.window != f.window && f.pos - l.pos < dedup => {
                let (fd, ld) = (depth(f.pos, f.window, w), depth(l.pos, l.window, w));
                let better = fd > ld || (fd == ld && f.window < l.window);
                if better {
                    *l = f;
                }
            }
            _ => {
                if let Some(prev) = last.replace(f) {
                    out.push(prev.peak());
                }
            }
        }
    }

    fn sort(&mut self) {
        // The batch path sorts by (position, window) with a stable sort over the detection order,
        // so this is the same order.
        self.pending.sort_unstable_by(|a, b| {
            a.pos
                .total_cmp(&b.pos)
                .then(a.window.cmp(&b.window))
                .then(a.order.cmp(&b.order))
        });
    }

    /// Feed every pending peak before `bound` (no peak from a later window can lie before it) and
    /// emit the last kept peak if no later peak can come within `dedup` of it.
    fn advance(&mut self, bound: f64, w: u64, dedup: f64, out: &mut Vec<StreamPeak>) {
        self.sort();
        let n = self.pending.partition_point(|f| f.pos < bound);
        for f in self.pending.drain(..n) {
            Self::feed(&mut self.last, f, w, dedup, out);
        }
        if let Some(l) = self.last {
            if l.pos + dedup + 1.0 <= bound {
                out.push(l.peak());
                self.last = None;
            }
        }
    }

    /// End of segment: feed everything and emit the last kept peak.
    fn finish(&mut self, w: u64, dedup: f64, out: &mut Vec<StreamPeak>) {
        self.sort();
        for f in self.pending.drain(..) {
            Self::feed(&mut self.last, f, w, dedup, out);
        }
        if let Some(l) = self.last.take() {
            out.push(l.peak());
        }
    }

    fn bytes(&self) -> usize {
        self.pending.capacity() * std::mem::size_of::<Found>()
    }
}

/// Streaming wrapper of the windowed detection pipeline (see the [module docs](self)).
///
/// ```
/// use fast_hrv::stream::{StreamConfig, StreamDetector, StreamPeaks};
/// # fn main() -> Result<(), fast_hrv::HrvError> {
/// let fs = 512.0;
/// // clean ECG, e.g. from EcgStreamFilter; here 30 s of synthetic beats every 0.8 s
/// let clean: Vec<f64> = (0..30 * 512)
///     .map(|i| {
///         let d = (i % 410) as f64 - 205.0;
///         (-d * d / 20.0).exp()
///     })
///     .collect();
/// let mut det = StreamDetector::new(fs, 0, StreamConfig::default())?;
/// let mut out = StreamPeaks::default();
/// for packet in clean.chunks(32) {
///     det.push(packet, &mut out)?; // peaks appear ~2–12 s after the beat
/// }
/// det.flush(&mut out)?; // end of the recording: the last windows, clipped
/// let batch = fast_hrv::detect::dual_detect_windowed(
///     &clean, fs, 0, &Default::default(), &Default::default())?;
/// let live: Vec<f64> = out.primary.iter().map(|p| p.position).collect();
/// assert_eq!(live, batch.primary);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct StreamDetector {
    fs: f64,
    cfg: StreamConfig,
    /// Core window and margin (samples), de-duplication distance (samples).
    w: u64,
    m: u64,
    dedup: f64,
    /// First absolute index of the current segment.
    segment_start: u64,
    /// Absolute index of the next sample.
    next_index: u64,
    /// Next window to process.
    next_window: u64,
    /// Samples `[buf_start, next_index)`; at most `w + 2·m` long.
    buf: Vec<f64>,
    buf_start: u64,
    primary: Dedup,
    secondary: Dedup,
    /// Test hook: replaces the primary detector (de-duplication stress tests).
    #[cfg(test)]
    fake: Option<fn(&[f64]) -> RefinedPeaks>,
}

impl StreamDetector {
    /// A detector whose first sample has the absolute index `start_index` (e.g. samples since the
    /// session start, the same clock as `start_sample` of the batch functions).
    ///
    /// Errors: invalid sampling rate or configuration (the same checks as the batch functions).
    pub fn new(fs: f64, start_index: u64, cfg: StreamConfig) -> Result<Self, HrvError> {
        // Validate through the batch code paths themselves (same errors, no duplicated rules).
        detect_windowed(&[0.0], fs, 0, &cfg.window, |_| Ok(Vec::new()))?;
        dual_detect(&[0.0; 64], fs, &cfg.dual)?;
        let w = (cfg.window.window_s * fs).round().max(1.0) as u64;
        let m = (cfg.window.margin_s * fs).round() as u64;
        let cap = usize::try_from(w + 2 * m).map_err(|_| HrvError::InvalidParameter("window_s"))?;
        let dedup = cfg.window.dedup_s * fs;
        // Detections of one detector are at least its refractory period apart (NeuroKit2 method:
        // 0.3 s minimum delay); the refinement moves each by at most its half window.
        let shift = 2.0 * cfg.dual.refine.half_window_s;
        let spacing = |refractory_s: f64| ((refractory_s - shift) * fs).floor();
        let primary_cap = Dedup::capacity(w, dedup, spacing(cfg.dual.elgendi.refractory_s));
        let secondary_cap = if cfg.secondary {
            Dedup::capacity(w, dedup, spacing(0.3))
        } else {
            0
        };
        Ok(Self {
            fs,
            cfg,
            w,
            m,
            dedup,
            segment_start: start_index,
            next_index: start_index,
            next_window: start_index / w,
            buf: Vec::with_capacity(cap),
            buf_start: start_index,
            primary: Dedup::new(primary_cap),
            secondary: Dedup::new(secondary_cap),
            #[cfg(test)]
            fake: None,
        })
    }

    /// Sampling rate (Hz).
    pub fn sampling_rate(&self) -> f64 {
        self.fs
    }

    /// The configuration.
    pub fn config(&self) -> &StreamConfig {
        &self.cfg
    }

    /// Absolute index of the next sample to be pushed.
    pub fn next_index(&self) -> u64 {
        self.next_index
    }

    /// Absolute index of the first sample of the current segment.
    pub fn segment_start(&self) -> u64 {
        self.segment_start
    }

    /// Absolute index of the sample that completes the next window (its right margin); pushing it
    /// runs the detectors.
    pub fn next_window_ready_at(&self) -> u64 {
        (self.next_window + 1) * self.w + self.m
    }

    /// Core window and margin in samples `(W, M)`.
    pub fn window_samples(&self) -> (u64, u64) {
        (self.w, self.m)
    }

    /// Upper bound of the delay (samples) between a peak's position and the sample whose push emits
    /// it: `W + M + 2·dedup + 2` (see the module docs). Peaks are emitted no earlier than
    /// `M + 2·dedup` after their position, except at the end of a segment.
    pub fn max_latency_samples(&self) -> u64 {
        self.w + self.m + (2.0 * self.dedup).ceil() as u64 + 2
    }

    /// Bytes held by this detector between calls (struct + sample buffer + pending peaks). Constant
    /// in steady state: the buffer is allocated once with its maximum size.
    pub fn state_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.buf.capacity() * std::mem::size_of::<f64>()
            + self.primary.bytes()
            + self.secondary.bytes()
    }

    /// Append `samples` (the next samples of the current segment) and emit the peaks confirmed by
    /// them into `out`. Errors: a non-finite sample (the whole packet is rejected and the state is
    /// unchanged).
    pub fn push(&mut self, samples: &[f64], out: &mut StreamPeaks) -> Result<(), HrvError> {
        self.push_with(samples, out, |_| {})
    }

    /// [`StreamDetector::push`], calling `on_window` for every window processed during this call
    /// (e.g. to compute signal quality on the core with the window's own detections, without a
    /// second sample buffer).
    pub fn push_with<F>(
        &mut self,
        samples: &[f64],
        out: &mut StreamPeaks,
        mut on_window: F,
    ) -> Result<(), HrvError>
    where
        F: FnMut(&StreamWindow<'_>),
    {
        if samples.iter().any(|v| !v.is_finite()) {
            return Err(HrvError::NonFinite);
        }
        let mut rest = samples;
        while !rest.is_empty() {
            let ready = self.next_window_ready_at();
            let take = ((ready - self.next_index) as usize).min(rest.len());
            self.buf.extend_from_slice(&rest[..take]);
            self.next_index += take as u64;
            rest = &rest[take..];
            if self.next_index == ready {
                self.process_window(out, &mut on_window)?;
            }
        }
        Ok(())
    }

    /// End the current segment (e.g. end of the recording): process the remaining windows with
    /// chunks clipped at the last sample, emit every pending peak, and start a new segment at
    /// [`StreamDetector::next_index`]. Equal to the end of the batch path.
    pub fn flush(&mut self, out: &mut StreamPeaks) -> Result<(), HrvError> {
        self.flush_with(out, |_| {})
    }

    /// [`StreamDetector::flush`] with a window callback.
    pub fn flush_with<F>(&mut self, out: &mut StreamPeaks, on_window: F) -> Result<(), HrvError>
    where
        F: FnMut(&StreamWindow<'_>),
    {
        self.reset_at_with(self.next_index, out, on_window)
            .map(|_| ())
    }

    /// A gap: end the current segment as [`StreamDetector::flush`] and continue at the absolute
    /// index `next_index` (≥ [`StreamDetector::next_index`]; equal = discontinuity without lost
    /// samples, e.g. a filter reset). Returns the gap `[old next_index, next_index)` for
    /// [`crate::gaps`]. Each segment's peaks equal the batch path run on that segment alone.
    pub fn reset_at(&mut self, next_index: u64, out: &mut StreamPeaks) -> Result<Gap, HrvError> {
        self.reset_at_with(next_index, out, |_| {})
    }

    /// [`StreamDetector::reset_at`] with a window callback (for the last windows of the segment).
    pub fn reset_at_with<F>(
        &mut self,
        next_index: u64,
        out: &mut StreamPeaks,
        mut on_window: F,
    ) -> Result<Gap, HrvError>
    where
        F: FnMut(&StreamWindow<'_>),
    {
        if next_index < self.next_index {
            return Err(HrvError::InvalidParameter("next_index"));
        }
        let to_usize =
            |v: u64| usize::try_from(v).map_err(|_| HrvError::InvalidParameter("next_index"));
        let gap = Gap {
            start: to_usize(self.next_index)?,
            end: to_usize(next_index)?,
        };
        // the batch path processes windows k with k·W < segment end
        while self.next_window * self.w < self.next_index {
            self.process_window(out, &mut on_window)?;
        }
        self.primary.finish(self.w, self.dedup, &mut out.primary);
        self.secondary
            .finish(self.w, self.dedup, &mut out.secondary);
        self.buf.clear();
        self.segment_start = next_index;
        self.next_index = next_index;
        self.next_window = next_index / self.w;
        self.buf_start = next_index;
        Ok(gap)
    }

    /// Run the detectors on window `next_window` (its chunk is in the buffer, possibly clipped at
    /// the segment end), queue its peaks for de-duplication and drop the samples no later window
    /// needs.
    fn process_window<F>(
        &mut self,
        out: &mut StreamPeaks,
        on_window: &mut F,
    ) -> Result<(), HrvError>
    where
        F: FnMut(&StreamWindow<'_>),
    {
        let k = self.next_window;
        let (core0, core1) = (k * self.w, (k + 1) * self.w);
        let c0 = core0.saturating_sub(self.m).max(self.segment_start);
        let c1 = (core1 + self.m).min(self.next_index);
        let chunk = &self.buf[(c0 - self.buf_start) as usize..(c1 - self.buf_start) as usize];
        let fs = self.fs;
        let dedup = self.dedup;
        let queue = |dd: &mut Dedup, r: &RefinedPeaks| {
            for (j, (&i, &p)) in r.indices.iter().zip(&r.positions).enumerate() {
                // exactly the arithmetic of `detect_windowed`
                let a = c0 as f64 + p;
                if depth(a, k, self.w) >= -dedup {
                    dd.pending.push(Found {
                        pos: a,
                        index: c0 + i as u64,
                        window: k,
                        order: j as u32,
                    });
                }
            }
        };
        #[cfg(test)]
        let fake = self.fake;
        #[cfg(not(test))]
        let fake: Option<fn(&[f64]) -> RefinedPeaks> = None;
        if let Some(fake) = fake {
            queue(&mut self.primary, &fake(chunk));
        } else if self.cfg.secondary {
            let d = dual_detect(chunk, fs, &self.cfg.dual)?;
            queue(&mut self.primary, &d.primary);
            queue(&mut self.secondary, &d.secondary);
            on_window(&StreamWindow {
                window: k,
                chunk_start: c0,
                core: (core0, core1),
                samples: chunk,
                primary: &d.primary,
                primary_raw: &d.primary_raw,
                secondary: Some(&d.secondary),
                bsqi: d.bsqi,
            });
        } else {
            let raw = elgendi_peaks(chunk, fs, &self.cfg.dual.elgendi)?;
            let refined = refine_peaks(chunk, fs, &raw, &self.cfg.dual.refine)?;
            queue(&mut self.primary, &refined);
            on_window(&StreamWindow {
                window: k,
                chunk_start: c0,
                core: (core0, core1),
                samples: chunk,
                primary: &refined,
                primary_raw: &raw,
                secondary: None,
                bsqi: None,
            });
        }
        self.next_window = k + 1;
        // Windows > k only keep peaks at or after `core1 − dedup`; one sample of slack covers the
        // rounding of `c0 + p`.
        let bound = core1 as f64 - dedup - 1.0;
        self.primary.advance(bound, self.w, dedup, &mut out.primary);
        self.secondary
            .advance(bound, self.w, dedup, &mut out.secondary);
        // keep only what the next chunk needs
        let keep_from = core1
            .saturating_sub(self.m)
            .max(self.segment_start)
            .min(self.next_index);
        if keep_from > self.buf_start {
            self.buf.drain(..(keep_from - self.buf_start) as usize);
            self.buf_start = keep_from;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn beats(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let d = (i % 410) as f64 - 205.0;
                (-d * d / 20.0).exp() - 0.3 * (-(d - 25.0).powi(2) / 30.0).exp()
            })
            .collect()
    }

    #[test]
    fn invalid_configuration_and_input() {
        assert!(StreamDetector::new(0.0, 0, StreamConfig::default()).is_err());
        let mut cfg = StreamConfig::default();
        cfg.window.dedup_s = 6.0;
        assert!(StreamDetector::new(512.0, 0, cfg).is_err());
        let mut cfg = StreamConfig::default();
        cfg.dual.elgendi.band_hz = (8.0, 300.0);
        assert!(StreamDetector::new(512.0, 0, cfg).is_err());
        let mut cfg = StreamConfig::default();
        cfg.dual.refine.half_window_s = -1.0;
        assert!(StreamDetector::new(512.0, 0, cfg).is_err());

        let mut d = StreamDetector::new(512.0, 100, StreamConfig::default()).unwrap();
        let mut out = StreamPeaks::default();
        d.push(&[0.0; 10], &mut out).unwrap();
        assert!(d.push(&[0.0, f64::NAN], &mut out).is_err());
        assert_eq!(
            d.next_index(),
            110,
            "rejected packet leaves the state unchanged"
        );
        assert!(d.reset_at(50, &mut out).is_err());
        let g = d.reset_at(200, &mut out).unwrap();
        assert_eq!((g.start, g.end), (110, 200));
        // flushing an empty segment is a no-op
        d.flush(&mut out).unwrap();
        assert!(out.is_empty());
    }

    /// Value of the synthetic "signal" at absolute index `i` (a hash), so a fake detector can see
    /// where it is.
    fn hash(i: u64) -> f64 {
        let mut z = i.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64
    }

    /// Dense, window-dependent detections: overlapping windows disagree, positions collide
    /// exactly, come unsorted and fall on both sides of every core boundary.
    fn fake_detector(chunk: &[f64]) -> RefinedPeaks {
        let n = chunk.len();
        let mut out = RefinedPeaks::default();
        for (i, &v) in chunk.iter().enumerate().rev() {
            let h = v as u64;
            if (h + (i + n) as u64 / 3) % 23 != 0 {
                continue;
            }
            let frac = match h % 4 {
                0 => 0.0,
                1 => 0.5,
                2 => -0.5,
                _ => ((h >> 8) % 1000) as f64 / 1000.0 - 0.5,
            };
            out.indices.push(i);
            out.positions.push(i as f64 + frac);
            if h % 7 == 0 {
                // an exact duplicate position in the same window
                out.indices.push(i);
                out.positions.push(i as f64 + frac);
            }
        }
        out
    }

    #[test]
    fn dedup_stress_equals_batch() {
        let fs = 100.0;
        for (window_s, margin_s, dedup_s) in [(1.0, 0.3, 0.2), (0.5, 1.2, 0.24), (2.0, 0.0, 0.5)] {
            let wc = WindowConfig {
                window_s,
                margin_s,
                dedup_s,
            };
            let cfg = StreamConfig {
                window: wc,
                secondary: false,
                ..StreamConfig::default()
            };
            for start in [0u64, 7, 99, 1234] {
                let x: Vec<f64> = (start..start + 3_000).map(hash).collect();
                let batch = detect_windowed(&x, fs, start, &wc, |c| Ok(fake_detector(c).positions))
                    .unwrap();
                assert!(batch.len() > 50);
                for packet in [1usize, 13, 100, 3000] {
                    let mut d = StreamDetector::new(fs, start, cfg).unwrap();
                    d.fake = Some(fake_detector);
                    let mut out = StreamPeaks::default();
                    for p in x.chunks(packet) {
                        d.push(p, &mut out).unwrap();
                    }
                    d.flush(&mut out).unwrap();
                    let live: Vec<u64> = out.primary.iter().map(|p| p.position.to_bits()).collect();
                    let want: Vec<u64> = batch.iter().map(|p| p.to_bits()).collect();
                    assert_eq!(live, want, "{wc:?} start {start} packet {packet}");
                }
            }
        }
    }

    #[test]
    fn buffer_is_bounded_and_state_constant() {
        let x = beats(512 * 60);
        let mut d = StreamDetector::new(512.0, 0, StreamConfig::default()).unwrap();
        let bytes = d.state_bytes();
        let mut out = StreamPeaks::default();
        for p in x.chunks(37) {
            d.push(p, &mut out).unwrap();
            assert!(d.buf.len() <= (d.w + 2 * d.m) as usize);
            assert_eq!(d.state_bytes(), bytes);
        }
        assert!(!out.primary.is_empty());
        assert!(bytes < 64 * 1024, "{bytes}");
    }
}
