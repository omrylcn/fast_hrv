"""
fast_hrv.utils - Utility functions
"""

import numpy as np
from typing import Union

ArrayLike = Union[np.ndarray, list]


def sanitize_signal(signal: ArrayLike) -> np.ndarray:
    """
    Convert input to contiguous float64 numpy array.

    Handles: list, np.ndarray, pd.Series, pd.DataFrame

    Parameters
    ----------
    signal : array-like
        Input signal

    Returns
    -------
    np.ndarray
        Contiguous float64 array
    """
    # Handle Pandas
    if hasattr(signal, 'values'):
        signal = signal.values

    # Convert to numpy
    arr = np.ascontiguousarray(signal, dtype=np.float64)

    # Flatten if needed
    if arr.ndim > 1:
        arr = arr.flatten()

    if arr.ndim != 1:
        raise ValueError(f"Signal must be 1D, got {arr.ndim}D")

    if len(arr) < 10:
        raise ValueError(f"Signal too short: {len(arr)} samples")

    return arr
