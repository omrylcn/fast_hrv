//! Python bindings using PyO3.
//!
//! Build with: `maturin develop --release --features python`, then `import fast_hrv`.
//! The module name equals the `[lib] name` in Cargo.toml (`fast_hrv`), so the shared library
//! exports `PyInit_fast_hrv` as Python expects. (Before 0.2.0 the module was declared as
//! `fast_hrv_rs`, which did not match the library file name and failed to import.)
//!
//! Arrays may be non-contiguous (they are copied); negative peak indices raise `ValueError`.

// `useless_conversion` / `type_complexity` are triggered by PyO3 0.22 macro expansion.
#![allow(clippy::useless_conversion, clippy::type_complexity)]

use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;

use crate::{
    compute_apen, compute_dfa, compute_pnn20, compute_pnn50, compute_poincare, compute_quality,
    compute_rmssd, compute_sampen, ecg_clean, find_peaks_default, fix_peaks_default, mean,
    process_ecg, std_ddof,
};

/// Convert peak indices from Python, rejecting negative values.
fn to_indices(a: &PyReadonlyArray1<'_, i64>) -> PyResult<Vec<usize>> {
    a.as_array()
        .iter()
        .map(|&x| {
            usize::try_from(x).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!("negative peak index {x}"))
            })
        })
        .collect()
}

// =============================================================================
// ECG Processing
// =============================================================================

/// Clean ECG signal as NeuroKit2 `ecg_clean(method="neurokit")`: 0.5 Hz high-pass (order 5,
/// zero-phase) and a 50 Hz powerline moving average.
#[pyfunction]
#[pyo3(name = "ecg_clean")]
fn py_ecg_clean<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
    sampling_rate: u32,
) -> Bound<'py, PyArray1<f64>> {
    let signal = &signal.as_array().to_vec();
    let cleaned = ecg_clean(signal, sampling_rate);
    cleaned.to_pyarray_bound(py)
}

/// Find R-peaks in ECG signal
#[pyfunction]
#[pyo3(name = "find_peaks")]
fn py_find_peaks<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
    sampling_rate: u32,
) -> Bound<'py, PyArray1<i64>> {
    let signal = &signal.as_array().to_vec();
    let peaks = find_peaks_default(signal, sampling_rate);
    let peaks_i64: Vec<i64> = peaks.into_iter().map(|x| x as i64).collect();
    peaks_i64.to_pyarray_bound(py)
}

/// Fix peak artifacts using Kubios method
#[pyfunction]
#[pyo3(name = "fix_peaks")]
fn py_fix_peaks<'py>(
    py: Python<'py>,
    peaks: PyReadonlyArray1<'py, i64>,
    sampling_rate: u32,
) -> PyResult<(Bound<'py, PyArray1<i64>>, PyObject)> {
    let peaks: Vec<usize> = to_indices(&peaks)?;
    let (fixed, artifacts) = fix_peaks_default(&peaks, sampling_rate);

    let fixed_i64: Vec<i64> = fixed.into_iter().map(|x| x as i64).collect();

    // Create artifacts dict
    let artifacts_dict = pyo3::types::PyDict::new_bound(py);
    artifacts_dict.set_item("ectopic", artifacts.ectopic)?;
    artifacts_dict.set_item("missed", artifacts.missed)?;
    artifacts_dict.set_item("extra", artifacts.extra)?;
    artifacts_dict.set_item("longshort", artifacts.longshort)?;

    Ok((fixed_i64.to_pyarray_bound(py), artifacts_dict.into()))
}

/// Compute ECG signal quality
#[pyfunction]
#[pyo3(name = "compute_quality")]
fn py_compute_quality<'py>(
    py: Python<'py>,
    ecg_cleaned: PyReadonlyArray1<'py, f64>,
    rpeaks: PyReadonlyArray1<'py, i64>,
    sampling_rate: u32,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let ecg_cleaned = &ecg_cleaned.as_array().to_vec();
    let rpeaks: Vec<usize> = to_indices(&rpeaks)?;

    let (quality, quality_per_beat) = compute_quality(ecg_cleaned, &rpeaks, sampling_rate);

    Ok((
        quality.to_pyarray_bound(py),
        quality_per_beat.to_pyarray_bound(py),
    ))
}

/// Full ECG processing pipeline
#[pyfunction]
#[pyo3(name = "process_ecg", signature = (signal, sampling_rate, correct_artifacts = true))]
fn py_process_ecg<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
    sampling_rate: u32,
    correct_artifacts: bool,
) -> PyResult<PyObject> {
    let signal = &signal.as_array().to_vec();
    let result = process_ecg(signal, sampling_rate, correct_artifacts);

    let dict = pyo3::types::PyDict::new_bound(py);

    // ECG cleaned signal
    dict.set_item("ecg_cleaned", result.ecg_cleaned.to_pyarray_bound(py))?;

    // Quality
    dict.set_item("quality", result.quality.to_pyarray_bound(py))?;

    // R-peaks
    let rpeaks_i64: Vec<i64> = result.rpeaks.into_iter().map(|x| x as i64).collect();
    dict.set_item("rpeaks", rpeaks_i64.to_pyarray_bound(py))?;

    // Stats
    let stats_dict = pyo3::types::PyDict::new_bound(py);
    stats_dict.set_item("n_peaks", result.stats.n_peaks)?;
    stats_dict.set_item("mean_hr_bpm", result.stats.mean_hr_bpm)?;
    stats_dict.set_item("std_hr_bpm", result.stats.std_hr_bpm)?;
    stats_dict.set_item("min_hr_bpm", result.stats.min_hr_bpm)?;
    stats_dict.set_item("max_hr_bpm", result.stats.max_hr_bpm)?;
    stats_dict.set_item("mean_rr_ms", result.stats.mean_rr_ms)?;
    stats_dict.set_item("std_rr_ms", result.stats.std_rr_ms)?;
    stats_dict.set_item("min_rr_ms", result.stats.min_rr_ms)?;
    stats_dict.set_item("max_rr_ms", result.stats.max_rr_ms)?;
    stats_dict.set_item("mean_quality", result.stats.mean_quality)?;
    stats_dict.set_item("sdnn_ms", result.stats.sdnn_ms)?;
    stats_dict.set_item("rmssd_ms", result.stats.rmssd_ms)?;
    stats_dict.set_item("pnn50", result.stats.pnn50)?;
    stats_dict.set_item("pnn20", result.stats.pnn20)?;
    stats_dict.set_item("sdsd_ms", result.stats.sdsd_ms)?;
    dict.set_item("stats", stats_dict)?;

    Ok(dict.into())
}

// =============================================================================
// HRV Metrics
// =============================================================================

/// Compute RMSSD (Root Mean Square of Successive Differences)
#[pyfunction]
#[pyo3(name = "rmssd")]
fn py_rmssd(rr: PyReadonlyArray1<f64>) -> f64 {
    compute_rmssd(&rr.as_array().to_vec())
}

/// Compute SDNN (Standard Deviation of NN intervals)
#[pyfunction]
#[pyo3(name = "sdnn")]
fn py_sdnn(rr: PyReadonlyArray1<f64>) -> f64 {
    std_ddof(&rr.as_array().to_vec(), 1)
}

/// Compute mean of array
#[pyfunction]
#[pyo3(name = "mean")]
fn py_mean(data: PyReadonlyArray1<f64>) -> f64 {
    mean(&data.as_array().to_vec())
}

/// Compute standard deviation with ddof
#[pyfunction]
#[pyo3(name = "std")]
fn py_std(data: PyReadonlyArray1<f64>, ddof: usize) -> f64 {
    std_ddof(&data.as_array().to_vec(), ddof)
}

/// Compute pNN50
#[pyfunction]
#[pyo3(name = "pnn50")]
fn py_pnn50(rr: PyReadonlyArray1<f64>) -> f64 {
    compute_pnn50(&rr.as_array().to_vec())
}

/// Compute pNN20
#[pyfunction]
#[pyo3(name = "pnn20")]
fn py_pnn20(rr: PyReadonlyArray1<f64>) -> f64 {
    compute_pnn20(&rr.as_array().to_vec())
}

/// Compute Approximate Entropy (ApEn)
#[pyfunction]
#[pyo3(name = "apen")]
fn py_apen(data: PyReadonlyArray1<f64>, m: usize, r: f64) -> f64 {
    compute_apen(&data.as_array().to_vec(), m, Some(r))
}

/// Compute Sample Entropy (SampEn)
#[pyfunction]
#[pyo3(name = "sampen")]
fn py_sampen(data: PyReadonlyArray1<f64>, m: usize, r: f64) -> f64 {
    compute_sampen(&data.as_array().to_vec(), m, Some(r))
}

/// Compute DFA alpha1 (short-term)
#[pyfunction]
#[pyo3(name = "dfa_alpha1")]
fn py_dfa_alpha1(rr: PyReadonlyArray1<f64>) -> f64 {
    let result = compute_dfa(&rr.as_array().to_vec(), (4, 16), (16, None));
    result.alpha1
}

/// Compute Poincaré SD1
#[pyfunction]
#[pyo3(name = "poincare_sd1")]
fn py_poincare_sd1(rr: PyReadonlyArray1<f64>) -> f64 {
    compute_poincare(&rr.as_array().to_vec()).sd1
}

/// Compute Poincaré SD2
#[pyfunction]
#[pyo3(name = "poincare_sd2")]
fn py_poincare_sd2(rr: PyReadonlyArray1<f64>) -> f64 {
    compute_poincare(&rr.as_array().to_vec()).sd2
}

/// Compute Poincaré CSI (Cardiac Sympathetic Index)
#[pyfunction]
#[pyo3(name = "poincare_csi")]
fn py_poincare_csi(rr: PyReadonlyArray1<f64>) -> f64 {
    compute_poincare(&rr.as_array().to_vec()).csi
}

/// Compute all HRV metrics at once
#[pyfunction]
#[pyo3(name = "hrv_metrics")]
fn py_hrv_metrics(py: Python<'_>, rr: PyReadonlyArray1<f64>) -> PyResult<PyObject> {
    let rr = &rr.as_array().to_vec();

    let dict = pyo3::types::PyDict::new_bound(py);

    // Time domain
    dict.set_item("mean", mean(rr))?;
    dict.set_item("sdnn", std_ddof(rr, 1))?;
    dict.set_item("rmssd", compute_rmssd(rr))?;
    dict.set_item("pnn50", compute_pnn50(rr))?;
    dict.set_item("pnn20", compute_pnn20(rr))?;

    // Nonlinear
    let r = 0.2 * std_ddof(rr, 1); // Standard r value
    dict.set_item("apen", compute_apen(rr, 2, Some(r)))?;
    dict.set_item("sampen", compute_sampen(rr, 2, Some(r)))?;

    // DFA
    let dfa = compute_dfa(rr, (4, 16), (16, None));
    dict.set_item("dfa_alpha1", dfa.alpha1)?;
    dict.set_item("dfa_alpha2", dfa.alpha2)?;

    // Poincaré
    let poincare = compute_poincare(rr);
    dict.set_item("sd1", poincare.sd1)?;
    dict.set_item("sd2", poincare.sd2)?;
    dict.set_item("csi", poincare.csi)?;
    dict.set_item("cvi", poincare.cvi)?;

    Ok(dict.into())
}

// =============================================================================
// Module Definition
// =============================================================================

/// fast_hrv_rs - Fast HRV analysis in Rust (`import fast_hrv_rs`)
/// Python module `fast_hrv_rs` (the pure-Python package of this repository is `fast_hrv`; the
/// binding keeps its own name so both can be installed together). Must match `module-name` in
/// `pyproject.toml`, otherwise the wheel installs but `import` fails (`PyInit_*` not found).
#[pymodule]
fn fast_hrv_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // ECG functions
    m.add_function(wrap_pyfunction!(py_ecg_clean, m)?)?;
    m.add_function(wrap_pyfunction!(py_find_peaks, m)?)?;
    m.add_function(wrap_pyfunction!(py_fix_peaks, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_quality, m)?)?;
    m.add_function(wrap_pyfunction!(py_process_ecg, m)?)?;

    // HRV functions
    m.add_function(wrap_pyfunction!(py_rmssd, m)?)?;
    m.add_function(wrap_pyfunction!(py_sdnn, m)?)?;
    m.add_function(wrap_pyfunction!(py_mean, m)?)?;
    m.add_function(wrap_pyfunction!(py_std, m)?)?;
    m.add_function(wrap_pyfunction!(py_pnn50, m)?)?;
    m.add_function(wrap_pyfunction!(py_pnn20, m)?)?;
    m.add_function(wrap_pyfunction!(py_apen, m)?)?;
    m.add_function(wrap_pyfunction!(py_sampen, m)?)?;
    m.add_function(wrap_pyfunction!(py_dfa_alpha1, m)?)?;
    m.add_function(wrap_pyfunction!(py_poincare_sd1, m)?)?;
    m.add_function(wrap_pyfunction!(py_poincare_sd2, m)?)?;
    m.add_function(wrap_pyfunction!(py_poincare_csi, m)?)?;
    m.add_function(wrap_pyfunction!(py_hrv_metrics, m)?)?;

    Ok(())
}
