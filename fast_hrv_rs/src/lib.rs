//! # fast-hrv
//!
//! EKG'den kalp hızı değişkenliğine (HRV) hızlı ve bağımlılıksız Rust kütüphanesi.
//!
//! - **Sinyal:** SciPy eşdeğeri Butterworth tasarımı ([`filter`]), NeuroKit2 uyumlu sıfır fazlı EKG
//!   temizleme ve paket paket akış filtresi ([`ecg`]), R tepe tespiti (NeuroKit2 yöntemi; Elgendi,
//!   alt-örnek inceltme, iki detektörlü bSQI ve uzun kayıtlar için pencereli işleme: [`detect`]; aynı
//!   pencereli çıktıyı paket paket veren akış detektörü: [`stream`]), Kubios
//!   artefakt düzeltme (atım başına değişen/eklenen/silinen sonucu, alt-örnek konumlarda da:
//!   [`peaks::fix_positions_with`]), düzeltme yerine dışlama kipi (ektopik / şüpheli atım sınıfları,
//!   NN geçerlilik maskesi, artefakt oranı: [`exclusion`]), boşluk farkında RR ([`gaps`]), mutlak
//!   sinyal kalitesi indeksleri ([`sqi`]).
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

pub mod detect;
pub mod detrend;
pub mod ecg;
pub mod error;
pub mod exclusion;
pub mod filter;
pub mod fragmentation;
pub mod frequency;
pub mod frequency_nk;
pub mod gaps;
pub mod hrv;
pub mod metrics;
pub mod motion;
pub mod peaks;
pub mod presets;
pub mod report;
pub mod respiration;
pub mod sqi;
pub mod stream;
pub mod time_domain;
pub mod window_report;

#[cfg(feature = "simd")]
pub mod simd;

#[cfg(feature = "python")]
mod python;

pub use detect::{
    DetectorFilter, DualConfig, DualDetection, ElgendiConfig, Fiducial, FiducialMemory, Polarity,
    RefineConfig, RefinedPeaks, WindowConfig, WindowedDual, ambiguous_morphology, detect_windowed,
    dual_detect, dual_detect_windowed, elgendi_peaks, elgendi_refined_windowed, energy_centroids,
    refine_peaks,
};
pub use detrend::{smoothness_priors, smoothness_priors_cutoff_hz, smoothness_priors_trend};
pub use ecg::{
    EcgFilterConfig, EcgResult, EcgStats, EcgStreamFilter, compute_quality, compute_stats,
    ecg_clean, ecg_clean_causal, ecg_clean_neurokit, ecg_clean_zero_phase, process_ecg,
};
pub use error::HrvError;
pub use exclusion::{
    ArtifactCounts, BeatClass, BeatExclusion, BeatFlags, ExclusionParams, ExclusionSummary,
    RemovalReason, RemovedBeat, classify_beats, classify_peaks, classify_segments, concat_nn,
};
pub use fragmentation::{Fragmentation, fragmentation};
pub use frequency::{
    FrequencyDomain, FrequencyOptions, band_power_rri, frequency_domain, frequency_domain_lomb,
    frequency_domain_with, lomb_scargle, welch,
};
pub use frequency_nk::frequency_domain_neurokit;
pub use gaps::{
    Gap, RrSegment, fix_peaks_segmented, fix_positions_segmented, rri_with_gaps, split_at_gaps,
};
pub use hrv::*;
pub use metrics::{METRICS, MetricInfo, metric_info};
pub use motion::{Activity, MotionConfig, MotionContext, angle_deg, motion_context};
pub use peaks::{
    Artifacts, BeatStatus, Corrections, KubiosMidpoint, KubiosOutput, KubiosParams,
    KubiosPositions, detect_artifacts, ecg_peaks, find_peaks, find_peaks_default, fix_peaks,
    fix_peaks_default, fix_peaks_with, fix_positions_with,
};
pub use presets::{Preset, Provenance, hrv_with_preset};
pub use report::{
    Asymmetry, HrvReport, NonLinear, hrv, rri_from_peaks, rri_from_positions, rri_from_times,
    try_rri_from_peaks, try_rri_from_times,
};
pub use respiration::{
    ModulationEstimate, RespirationConfig, RespirationEstimate, RespiratoryHf, hf_band_for,
    modulation_rate, r_amplitudes, respiratory_hf, respiratory_rate,
};
pub use stream::{
    StreamConfig, StreamDetector, StreamPeak, StreamPeaks, StreamWindow, fix_stream_peaks,
    peak_positions,
};
pub use time_domain::{
    MaskedTimeDomain, TimeDomain, masked_differences, rmssd_masked, time_domain, time_domain_masked,
};
pub use window_report::{
    MetricVerdict, Verdict, WindowContext, WindowReport, window_report, window_report_rri,
};
