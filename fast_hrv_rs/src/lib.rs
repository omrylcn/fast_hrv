//! # fast-hrv
//!
//! EKG'den kalp hızı değişkenliğine (HRV) hızlı ve bağımlılıksız Rust kütüphanesi.
//!
//! - **Sinyal:** EKG temizleme (Butterworth, sıfır fazlı), R tepe tespiti (NeuroKit2 yöntemi),
//!   Kubios artefakt düzeltme, atım kalitesi (şablon eşleştirme).
//! - **HRV, 39 metrik:** zaman alanı (17), frekans alanı (9), doğrusal olmayan (9), asimetri (6).
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
//! - `parallel` — rayon ile çoklu iş parçacığı
//! - `simd` — SIMD yardımcıları
//! - `python` — PyO3 ile Python bağlaması
//!
//! Kaynak: `omrylcn/fast_hrv` (fast_hrv_rs) üzerine kurulmuştur.

pub mod ecg;
pub mod frequency;
pub mod hrv;
pub mod peaks;
pub mod report;
pub mod time_domain;

#[cfg(feature = "simd")]
pub mod simd;

#[cfg(feature = "python")]
mod python;

pub use ecg::{compute_quality, compute_stats, ecg_clean, process_ecg, EcgResult, EcgStats};
pub use frequency::{frequency_domain, welch, FrequencyDomain};
pub use hrv::*;
pub use peaks::{ecg_peaks, find_peaks, find_peaks_default, fix_peaks, fix_peaks_default, Artifacts};
pub use report::{hrv, rri_from_peaks, Asymmetry, HrvReport, NonLinear};
pub use time_domain::{time_domain, TimeDomain};

