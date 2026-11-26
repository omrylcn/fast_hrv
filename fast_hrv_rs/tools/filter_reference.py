"""SciPy reference values for the filter module (dev-time only; the crate never calls Python).

Output: tests/data/filter_reference.json, read by tests/filter_design.rs.

Usage:  python tools/filter_reference.py
Needs:  numpy, scipy
"""
import json
import os
import sys

import numpy as np
import scipy
import scipy.signal as ss

OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "data", "filter_reference.json")


def main():
    butter = []
    cases = [
        # (order, cutoff(s) Hz, btype, fs)
        (1, [0.5], "highpass", 512), (2, [0.5], "highpass", 512), (5, [0.5], "highpass", 512),
        (5, [0.5], "highpass", 250), (5, [0.5], "highpass", 1000), (3, [1.0], "highpass", 360),
        (4, [40.0], "lowpass", 512), (2, [40.0], "lowpass", 512), (7, [15.0], "lowpass", 128),
        (1, [10.0], "lowpass", 100), (8, [100.0], "lowpass", 1000), (6, [3.0], "lowpass", 500),
        (2, [0.5, 40.0], "bandpass", 512), (4, [5.0, 15.0], "bandpass", 512), (3, [0.67, 45.0], "bandpass", 360),
        (1, [8.0, 20.0], "bandpass", 250), (5, [0.05, 0.4], "bandpass", 4),
        (2, [48.0, 52.0], "bandstop", 512), (3, [58.0, 62.0], "bandstop", 1000), (1, [45.0, 55.0], "bandstop", 250),
    ]
    for order, wn, btype, fs in cases:
        sos = ss.butter(order, wn if len(wn) > 1 else wn[0], btype=btype, output="sos", fs=fs)
        butter.append({"order": order, "cutoff": wn, "btype": btype, "fs": fs, "sos": sos.tolist(),
                       "zi": ss.sosfilt_zi(sos).tolist()})
    notch = []
    for f0, q, fs in [(50.0, 30.0, 512), (60.0, 30.0, 512), (50.0, 10.0, 250), (50.0, 35.0, 1000)]:
        b, a = ss.iirnotch(f0, q, fs=fs)
        notch.append({"f0": f0, "q": q, "fs": fs, "b": b.tolist(), "a": a.tolist()})
    # Zero-phase filtering with SciPy's default padding on a deterministic test signal
    fs = 512
    n = 700
    t = np.arange(n) / fs
    x = 412 + 50 * np.sin(2 * np.pi * 1.3 * t) + 20 * np.sin(2 * np.pi * 50 * t) + 10 * np.cos(2 * np.pi * 7 * t) + 3 * t
    hp5 = ss.butter(5, 0.5, btype="highpass", output="sos", fs=fs)
    bp2 = ss.butter(2, [0.5, 40.0], btype="bandpass", output="sos", fs=fs)
    ma = np.ones(int(fs / 50))
    filt = {
        "fs": fs,
        "x": x.tolist(),
        "sosfiltfilt_hp5": ss.sosfiltfilt(hp5, x).tolist(),
        "sosfiltfilt_bp2": ss.sosfiltfilt(bp2, x).tolist(),
        "filtfilt_ma10": ss.filtfilt(ma, [len(ma)], x, method="pad").tolist(),
        "sosfilt_bp2": ss.sosfilt(bp2, x).tolist(),
    }
    out = {"scipy_version": scipy.__version__, "numpy_version": np.__version__,
           "butter": butter, "iirnotch": notch, "filtering": filt}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {os.path.abspath(OUT)} (scipy {scipy.__version__})", file=sys.stderr)


if __name__ == "__main__":
    main()
