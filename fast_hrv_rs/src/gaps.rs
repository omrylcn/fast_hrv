//! Gap-aware RR intervals.
//!
//! A recording with gaps (lost packets, electrode off, device pauses) must not produce RR intervals
//! that span a gap, and artifact correction must not insert beats into a gap. The helpers here split
//! a peak list into **segments** of uninterrupted data; RR intervals and Kubios correction are then
//! computed per segment.
//!
//! Two input forms are supported:
//! - R-peak **sample indices** on one continuous sample clock plus [`Gap`] markers in samples
//!   ([`split_at_gaps`], [`rri_segments`], [`fix_peaks_segmented`]);
//! - R-peak **timestamps** in seconds plus gaps in seconds ([`split_peak_times`],
//!   [`rri_segments_from_times`]), e.g. when peaks carry device timestamps.
//!
//! Optionally, any RR interval longer than `max_rr_s` is also treated as a gap. No default is
//! imposed: choosing that limit is up to the application.
//!
//! Time- and frequency-domain metrics should be computed per segment; concatenating RR intervals of
//! different segments re-introduces successive differences across a gap.

use crate::error::HrvError;
use crate::peaks::{Artifacts, Corrections, KubiosOutput, KubiosParams, fix_peaks_with};

/// A gap: the half-open interval `[start, end)` in samples (same clock as the peaks) with no valid
/// data. Peaks inside a gap are discarded. A zero-length gap (`start == end`) is a discontinuity
/// marker between samples `start - 1` and `start` (e.g. a timestamp jump).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gap {
    pub start: usize,
    pub end: usize,
}

/// RR intervals of one uninterrupted segment.
#[derive(Debug, Clone, PartialEq)]
pub struct RrSegment {
    /// Time of the first peak of the segment (seconds on the input clock).
    pub start_s: f64,
    /// RR intervals in ms; interval `i` ends at peak `i + 1` of the segment.
    pub rri_ms: Vec<f64>,
}

/// Index segments of `times` (strictly increasing) that cross no gap; segments with < 2 peaks dropped.
fn split_indices(
    times: &[f64],
    gaps: &[(f64, f64)],
    max_rr: Option<f64>,
) -> Result<Vec<Vec<usize>>, HrvError> {
    if times.iter().any(|t| !t.is_finite()) {
        return Err(HrvError::NonFinite);
    }
    if let Some(i) = times.windows(2).position(|w| w[1] <= w[0]) {
        return Err(HrvError::UnsortedPeaks { index: i + 1 });
    }
    if let Some(&(s, e)) = gaps
        .iter()
        .find(|&&(s, e)| !(s.is_finite() && e.is_finite() && e >= s))
    {
        return Err(HrvError::InvalidGap { start: s, end: e });
    }
    let in_gap = |t: f64| gaps.iter().any(|&(s, e)| t >= s && t < e);
    let crosses = |a: f64, b: f64| gaps.iter().any(|&(s, _)| a < s && b >= s);
    let mut segments: Vec<Vec<usize>> = vec![Vec::new()];
    let mut prev: Option<f64> = None;
    for (i, &t) in times.iter().enumerate() {
        if in_gap(t) {
            continue;
        }
        if prev.is_some_and(|p| crosses(p, t) || max_rr.is_some_and(|m| t - p > m)) {
            segments.push(Vec::new());
        }
        segments.last_mut().unwrap().push(i);
        prev = Some(t);
    }
    segments.retain(|s| s.len() >= 2);
    Ok(segments)
}

fn collect_segments<T: Copy>(
    values: &[T],
    times: &[f64],
    gaps: &[(f64, f64)],
    max_rr: Option<f64>,
) -> Result<Vec<Vec<T>>, HrvError> {
    Ok(split_indices(times, gaps, max_rr)?
        .into_iter()
        .map(|seg| seg.into_iter().map(|i| values[i]).collect())
        .collect())
}

/// Split strictly increasing R-peak sample indices into segments that do not cross any gap.
///
/// `max_rr_s` (optional) also cuts wherever consecutive peaks are more than `max_rr_s` seconds
/// apart. Segments with fewer than 2 peaks (no RR interval) are dropped.
pub fn split_at_gaps(
    peaks: &[usize],
    gaps: &[Gap],
    sampling_rate: f64,
    max_rr_s: Option<f64>,
) -> Result<Vec<Vec<usize>>, HrvError> {
    if !(sampling_rate.is_finite() && sampling_rate > 0.0) {
        return Err(HrvError::InvalidSamplingRate(sampling_rate));
    }
    let times: Vec<f64> = peaks.iter().map(|&p| p as f64).collect();
    let g: Vec<(f64, f64)> = gaps
        .iter()
        .map(|g| (g.start as f64, g.end as f64))
        .collect();
    collect_segments(peaks, &times, &g, max_rr_s.map(|m| m * sampling_rate))
}

/// Split strictly increasing R-peak timestamps (seconds) into segments; gaps are `(start, end)`
/// in seconds (half-open; zero length = discontinuity marker).
pub fn split_peak_times(
    times_s: &[f64],
    gaps: &[(f64, f64)],
    max_rr_s: Option<f64>,
) -> Result<Vec<Vec<f64>>, HrvError> {
    collect_segments(times_s, times_s, gaps, max_rr_s)
}

/// RR intervals (ms) per segment of sample-index peaks.
pub fn rri_segments(segments: &[Vec<usize>], sampling_rate: f64) -> Vec<RrSegment> {
    segments
        .iter()
        .filter(|s| s.len() >= 2)
        .map(|s| RrSegment {
            start_s: s[0] as f64 / sampling_rate,
            rri_ms: s
                .windows(2)
                .map(|w| (w[1] - w[0]) as f64 * 1000.0 / sampling_rate)
                .collect(),
        })
        .collect()
}

/// RR intervals (ms) per segment of peak timestamps (seconds).
pub fn rri_segments_from_times(segments: &[Vec<f64>]) -> Vec<RrSegment> {
    segments
        .iter()
        .filter(|s| s.len() >= 2)
        .map(|s| RrSegment {
            start_s: s[0],
            rri_ms: s.windows(2).map(|w| (w[1] - w[0]) * 1000.0).collect(),
        })
        .collect()
}

/// Gap-aware RR intervals in one call: [`split_at_gaps`] followed by [`rri_segments`].
pub fn rri_with_gaps(
    peaks: &[usize],
    gaps: &[Gap],
    sampling_rate: f64,
    max_rr_s: Option<f64>,
) -> Result<Vec<RrSegment>, HrvError> {
    Ok(rri_segments(
        &split_at_gaps(peaks, gaps, sampling_rate, max_rr_s)?,
        sampling_rate,
    ))
}

/// Kubios correction run independently on each segment, so no beat is ever inserted into a gap
/// and the rolling thresholds never mix data from both sides of a gap.
pub fn fix_peaks_segmented(
    segments: &[Vec<usize>],
    sampling_rate: f64,
    params: &KubiosParams,
) -> Result<Vec<KubiosOutput>, HrvError> {
    segments
        .iter()
        .map(|s| fix_peaks_with(s, sampling_rate, params))
        .collect()
}

/// Total artifact **flag** count over segment results ([`Artifacts::total_count`] summed).
///
/// Counts flags over all correction passes, including flags that changed nothing: **not** a
/// corrected-beat count. For gating use [`total_changed`] / [`overall_changed_ratio`].
pub fn total_artifacts(outputs: &[KubiosOutput]) -> usize {
    outputs.iter().map(|o| o.artifacts.total_count()).sum()
}

/// Output beats actually changed (moved + inserted) in each segment
/// ([`Corrections::changed_count`]). Works on [`KubiosOutput`] and any other result exposing
/// [`Corrections`].
pub fn segment_changed_counts<T: AsRef<Corrections>>(outputs: &[T]) -> Vec<usize> {
    outputs.iter().map(|o| o.as_ref().changed_count()).collect()
}

/// Output beats actually changed (moved + inserted), summed over segments.
pub fn total_changed<T: AsRef<Corrections>>(outputs: &[T]) -> usize {
    outputs.iter().map(|o| o.as_ref().changed_count()).sum()
}

/// Input beats removed (extra beats), summed over segments.
pub fn total_removed<T: AsRef<Corrections>>(outputs: &[T]) -> usize {
    outputs.iter().map(|o| o.as_ref().removed_count()).sum()
}

/// Share of output beats actually changed over all segments: total (moved + inserted) / total
/// output beats (same definition as [`Corrections::changed_ratio`]; removed beats are reported by
/// [`total_removed`]). `NaN` when there is no output beat.
pub fn overall_changed_ratio<T: AsRef<Corrections>>(outputs: &[T]) -> f64 {
    let n: usize = outputs.iter().map(|o| o.as_ref().n_output()).sum();
    if n == 0 {
        f64::NAN
    } else {
        total_changed(outputs) as f64 / n as f64
    }
}

/// Artifacts of all segments (indices are per segment).
pub fn segment_artifacts(outputs: &[KubiosOutput]) -> Vec<Artifacts> {
    outputs.iter().map(|o| o.artifacts.clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn regular(start: usize, n: usize, step: usize) -> Vec<usize> {
        (0..n).map(|i| start + i * step).collect()
    }

    #[test]
    fn no_rr_spans_a_gap() {
        let fs = 512.0;
        let mut peaks = regular(100, 50, 420);
        let after: Vec<usize> = regular(*peaks.last().unwrap() + 69 * 512, 50, 420);
        let gap = Gap {
            start: peaks.last().unwrap() + 200,
            end: after[0] - 200,
        };
        peaks.extend(&after);
        let segs = rri_with_gaps(&peaks, &[gap], fs, None).unwrap();
        assert_eq!(segs.len(), 2);
        for s in &segs {
            assert!(
                s.rri_ms
                    .iter()
                    .all(|&r| (r - 420.0 * 1000.0 / fs).abs() < 1e-9)
            );
        }
        assert_eq!(segs[0].rri_ms.len() + segs[1].rri_ms.len(), peaks.len() - 2);
    }

    #[test]
    fn max_rr_splits_without_explicit_gap() {
        let mut peaks = regular(0, 10, 400);
        peaks.extend(regular(peaks[9] + 5000, 10, 400));
        let segs = split_at_gaps(&peaks, &[], 512.0, Some(3.0)).unwrap();
        assert_eq!(segs.len(), 2);
        assert_eq!(split_at_gaps(&peaks, &[], 512.0, None).unwrap().len(), 1);
    }

    #[test]
    fn peaks_inside_gap_are_dropped_and_marker_splits() {
        let peaks = regular(0, 20, 400);
        let segs = split_at_gaps(
            &peaks,
            &[Gap {
                start: 3000,
                end: 4500,
            }],
            512.0,
            None,
        )
        .unwrap();
        assert!(segs.iter().flatten().all(|&p| !(3000..4500).contains(&p)));
        assert_eq!(segs.len(), 2);
        let segs = split_at_gaps(
            &peaks,
            &[Gap {
                start: 2001,
                end: 2001,
            }],
            512.0,
            None,
        )
        .unwrap();
        assert_eq!(segs.len(), 2);
        assert_eq!(*segs[0].last().unwrap(), 2000);
        assert_eq!(segs[1][0], 2400);
    }

    #[test]
    fn kubios_does_not_fill_gaps() {
        let fs = 512.0;
        // slightly variable RR (a perfectly regular series gives zero Kubios thresholds)
        let jitter = |start: usize| -> Vec<usize> {
            let mut t = start;
            (0..60)
                .map(|i| {
                    t += 430 + (i * 37) % 11;
                    t
                })
                .collect()
        };
        let mut peaks = jitter(100);
        let resume = peaks.last().unwrap() + 69 * 512;
        peaks.extend(jitter(resume));
        // without gap information Kubios sees one huge "missed/long" interval and inserts a beat
        let whole = fix_peaks_with(&peaks, fs, &KubiosParams::default()).unwrap();
        assert!(whole.peaks.len() > peaks.len() || whole.artifacts.total_count() > 0);
        // with the gap, nothing is inserted between the segments
        let segs = split_at_gaps(
            &peaks,
            &[Gap {
                start: peaks[59] + 1,
                end: resume,
            }],
            fs,
            None,
        )
        .unwrap();
        let fixed = fix_peaks_segmented(&segs, fs, &KubiosParams::default()).unwrap();
        assert_eq!(total_artifacts(&fixed), 0);
        assert_eq!(segment_changed_counts(&fixed), vec![0, 0]);
        assert_eq!(overall_changed_ratio(&fixed), 0.0);
        let all: Vec<usize> = fixed.iter().flat_map(|o| o.peaks.clone()).collect();
        assert_eq!(all, peaks);
    }

    #[test]
    fn segment_changed_counts_per_segment() {
        let fs = 512.0;
        let jitter = |start: usize| -> Vec<usize> {
            (0..40)
                .map(|i| start + i * 435 + (15.0 * (i as f64 * 0.7).sin()).round() as usize + 15)
                .collect()
        };
        let mut a = jitter(100);
        let b = jitter(a.last().unwrap() + 60 * 512);
        // first segment: one beat displaced (moved back), second: clean
        a[20] -= 150;
        let fixed = fix_peaks_segmented(&[a.clone(), b], fs, &KubiosParams::default()).unwrap();
        let counts = segment_changed_counts(&fixed);
        assert!(counts[0] >= 1 && counts[1] == 0, "{counts:?}");
        assert_eq!(total_changed(&fixed), counts[0]);
        assert_eq!(total_removed(&fixed), 0);
        let n: usize = fixed.iter().map(|o| o.peaks.len()).sum();
        assert_eq!(overall_changed_ratio(&fixed), counts[0] as f64 / n as f64);
        assert!(total_artifacts(&fixed) >= counts[0]);
        assert!(overall_changed_ratio::<KubiosOutput>(&[]).is_nan());
    }

    #[test]
    fn timestamps_api() {
        let t: Vec<f64> = (0..10)
            .map(|i| i as f64 * 0.8)
            .chain((0..10).map(|i| 100.0 + i as f64 * 0.8))
            .collect();
        let segs = split_peak_times(&t, &[(10.0, 99.0)], None).unwrap();
        let rr = rri_segments_from_times(&segs);
        assert_eq!(rr.len(), 2);
        assert!(
            rr.iter()
                .all(|s| s.rri_ms.iter().all(|&r| (r - 800.0).abs() < 1e-9))
        );
        assert!(split_peak_times(&[1.0, 0.5], &[], None).is_err());
        assert!(split_peak_times(&[1.0, 2.0], &[(5.0, 4.0)], None).is_err());
    }
}
