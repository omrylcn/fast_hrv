//! fast_hrv_rs - Fast HRV Analysis in Rust
//!
//! Python fast_hrv modülünün Rust implementasyonu

pub mod ecg;
pub mod hrv;
pub mod peaks;

// Re-export HRV functions
pub use hrv::*;

// Re-export peak detection functions
pub use peaks::{ecg_peaks, find_peaks, find_peaks_default, fix_peaks, fix_peaks_default, Artifacts};

// Re-export ECG functions
pub use ecg::{process_ecg, compute_quality, compute_stats, ecg_clean, EcgResult, EcgStats};
