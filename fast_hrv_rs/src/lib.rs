//! # fast-hrv
//!
//! EKG'den kalp hızı değişkenliğine (HRV) hızlı ve bağımlılıksız Rust kütüphanesi.
//!
//! - **Sinyal:** SciPy eşdeğeri Butterworth tasarımı ([`filter`]), NeuroKit2 uyumlu sıfır fazlı EKG
//!   temizleme ve paket paket akış filtresi ([`ecg`]), R tepe tespiti (NeuroKit2 yöntemi), Kubios
//!   artefakt düzeltme, boşluk farkında RR ([`gaps`]), mutlak sinyal kalitesi indeksleri ([`sqi`]).
//! - **HRV, 41 metrik:** zaman alanı (17), frekans alanı (9), doğrusal olmayan (9), asimetri (6).
//!
//! ```
//! // R tepe indekslerinden (512 Hz) bütün metrikler
//! let peaks: Vec<usize> = (0..400).map(|i| i * 410 + (i % 7) * 3).collect();
//! let rri = fast_hrv::rri_from_peaks(&peaks, 512);
//! let report = fast_hrv::hrv(&rri);
//! println!("RMSSD {:.1} ms, LF/HF {:.2}", report.time.rmssd, report.frequency.lf_hf);
//! ```
//!
//! ## Özellikler (features)
//! - `serde` — rapor yapılarını serileştirme
//! - `parallel` — ayrılmış, şu an etkisiz
//! - `simd` — taşınabilir döngü açma yardımcıları (CPU'ya özel talimat yok, varsayılan kapalı)
//! - `python` — PyO3 ile Python bağlaması (`import fast_hrv`)
//!
//! Kaynak: `omrylcn/fast_hrv` (fast_hrv_rs) üzerine kurulmuştur.

// Index loops are the clearest form for the DSP kernels here (several arrays indexed together).
#![allow(clippy::needless_range_loop)]

pub mod ecg;
pub mod error;
pub mod filter;
pub mod frequency;
pub mod gaps;
pub mod hrv;
pub mod peaks;
pub mod report;
pub mod sqi;
pub mod time_domain;

#[cfg(feature = "simd")]
pub mod simd;

#[cfg(feature = "python")]
mod python;

pub use ecg::{
    EcgFilterConfig, EcgResult, EcgStats, EcgStreamFilter, compute_quality, compute_stats,
    ecg_clean, ecg_clean_causal, ecg_clean_neurokit, ecg_clean_zero_phase, process_ecg,
};
pub use error::HrvError;
pub use frequency::{
    FrequencyDomain, FrequencyOptions, frequency_domain, frequency_domain_with, welch,
};
pub use gaps::{Gap, RrSegment, fix_peaks_segmented, rri_with_gaps, split_at_gaps};
pub use hrv::*;
pub use peaks::{
    Artifacts, KubiosOutput, KubiosParams, detect_artifacts, ecg_peaks, find_peaks,
    find_peaks_default, fix_peaks, fix_peaks_default, fix_peaks_with,
};
pub use report::{Asymmetry, HrvReport, NonLinear, hrv, rri_from_peaks, try_rri_from_peaks};
pub use time_domain::{TimeDomain, time_domain};
