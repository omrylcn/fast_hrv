"""
fast_hrv - Fast ECG Processing and HRV Analysis

A high-performance ECG processing library optimized for HRV analysis.
Up to 100x faster than NeuroKit2.

Two versions available:
- NumPy version: Pure Python, fewer dependencies
- Numba version: JIT compiled, significantly faster

Usage (NumPy version):
    from fast_hrv.ecg import process_ecg
    result = process_ecg(ecg_signal, sampling_rate=512)

Usage (Numba version - recommended):
    from fast_hrv.ecg_numba import process_ecg
    result = process_ecg(ecg_signal, sampling_rate=512)

HRV metrics only:
    from fast_hrv.hrv import compute_hrv
    hrv = compute_hrv(rr_intervals)
"""

__version__ = "0.1.0"

# =============================================================================
# NumPy Version (default)
# =============================================================================

from .ecg import (
    process_ecg,
    compute_quality,
)

from .peaks import (
    ecg_peaks,
    find_peaks,
    fix_peaks,
)

from .hrv import (
    # Main functions
    compute_hrv,
    compute_hrv_full,
    compute_hrv_minimal,
    # Time domain
    compute_time_domain,
    compute_rmssd,
    # Poincaré
    compute_poincare,
    compute_csi,
    # Asymmetry
    compute_asymmetry,
    # Nonlinear
    compute_sampen,
    compute_apen,
    compute_dfa,
    # Frequency domain
    compute_frequency_domain,
)

from .utils import sanitize_signal

# =============================================================================
# Numba Version (optional, faster)
# =============================================================================

try:
    from . import peaks_numba
    from . import ecg_numba
    NUMBA_AVAILABLE = True
except ImportError:
    peaks_numba = None
    ecg_numba = None
    NUMBA_AVAILABLE = False

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Version
    "__version__",
    "NUMBA_AVAILABLE",

    # ECG processing (NumPy)
    "process_ecg",
    "compute_quality",

    # Peak detection (NumPy)
    "ecg_peaks",
    "find_peaks",
    "fix_peaks",

    # HRV metrics
    "compute_hrv",
    "compute_hrv_full",
    "compute_hrv_minimal",
    "compute_time_domain",
    "compute_rmssd",
    "compute_poincare",
    "compute_csi",
    "compute_asymmetry",
    "compute_sampen",
    "compute_apen",
    "compute_dfa",
    "compute_frequency_domain",

    # Utilities
    "sanitize_signal",

    # Numba modules (optional)
    "peaks_numba",
    "ecg_numba",
]


def get_backend():
    """Return which backend is being used."""
    if NUMBA_AVAILABLE:
        return "numba"
    return "numpy"


def use_numba():
    """
    Switch to Numba backend for faster processing.

    Returns the Numba process_ecg function.
    Raises ImportError if Numba is not available.
    """
    if not NUMBA_AVAILABLE:
        raise ImportError(
            "Numba is not installed. Install with: pip install numba"
        )
    return ecg_numba.process_ecg
