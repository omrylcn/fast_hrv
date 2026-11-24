//! Error type shared by the fallible (`Result`-returning) APIs.

use std::fmt;

/// Errors returned by the validating APIs of this crate.
///
/// The older convenience functions (for example [`crate::ecg_clean`] or [`crate::hrv`]) never panic;
/// on invalid input they return `NaN` values instead. The `try_*` / builder APIs return this error.
#[derive(Debug, Clone, PartialEq)]
pub enum HrvError {
    /// Sampling rate must be finite and > 0.
    InvalidSamplingRate(f64),
    /// Filter order must be ≥ 1.
    InvalidOrder(usize),
    /// Cut-off / centre frequency must satisfy 0 < f < fs/2 (and low < high for band filters).
    InvalidFrequency { freq: f64, nyquist: f64 },
    /// Notch quality factor must be finite and > 0.
    InvalidQ(f64),
    /// Input contains NaN or infinite values where finite values are required.
    NonFinite,
    /// Peaks (or peak times) must be strictly increasing.
    UnsortedPeaks { index: usize },
    /// A gap marker is malformed (end < start).
    InvalidGap { start: f64, end: f64 },
    /// Not enough data for the requested computation.
    TooShort { needed: usize, got: usize },
}

impl fmt::Display for HrvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HrvError::InvalidSamplingRate(fs) => {
                write!(f, "invalid sampling rate {fs} (must be finite and > 0)")
            }
            HrvError::InvalidOrder(n) => write!(f, "invalid filter order {n} (must be >= 1)"),
            HrvError::InvalidFrequency { freq, nyquist } => {
                write!(
                    f,
                    "invalid frequency {freq} Hz (must be 0 < f < {nyquist} Hz)"
                )
            }
            HrvError::InvalidQ(q) => {
                write!(f, "invalid quality factor {q} (must be finite and > 0)")
            }
            HrvError::NonFinite => write!(f, "input contains NaN or infinite values"),
            HrvError::UnsortedPeaks { index } => {
                write!(
                    f,
                    "peaks must be strictly increasing (violated at index {index})"
                )
            }
            HrvError::InvalidGap { start, end } => {
                write!(f, "invalid gap [{start}, {end}) (end < start)")
            }
            HrvError::TooShort { needed, got } => {
                write!(f, "input too short: need {needed}, got {got}")
            }
        }
    }
}

impl std::error::Error for HrvError {}
