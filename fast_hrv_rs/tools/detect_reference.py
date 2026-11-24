"""NeuroKit2 Elgendi reference values for tests/detect.rs (dev-time only, run once).

The crate never calls Python: the JSON file is committed and the Rust tests read it.
All inputs are synthetic (nk.ecg_simulate at 512 Hz plus programmatic noise); no real recordings.

Output (tests/data/):
  neurokit_elgendi.json   for each case: raw int16 ADC counts (base64, little endian),
                          nk.ecg_clean(method="elgendi2010") output (every 32nd sample, float64 base64),
                          nk.ecg_findpeaks(method="elgendi2010") and (method="elgendi2013") on that
                          cleaned signal (raw detector output, no artifact correction).

Usage:  python tools/detect_reference.py
Needs:  numpy, scipy, neurokit2 (>= 0.3.0 for elgendi2013)
"""
import base64
import json
import os
import sys
import warnings

import numpy as np
import scipy
import neurokit2 as nk

warnings.filterwarnings("ignore")
DATA = os.path.join(os.path.dirname(__file__), "..", "tests", "data")
VERSIONS = {"neurokit2": nk.__version__, "numpy": np.__version__, "scipy": scipy.__version__}


def b64(arr, dtype):
    return base64.b64encode(np.asarray(arr, dtype=dtype).astype(np.dtype(dtype).newbyteorder("<")).tobytes()).decode()


def main():
    fs = 512
    dur = 30
    # name, ecg_simulate kwargs, white noise (counts), mains 50 Hz (counts), baseline wander (counts), gain
    variants = [
        ("clean", dict(noise=0.01, heart_rate=70), 0.0, 0.0, 0.0, 200.0),
        ("noisy", dict(noise=0.1, heart_rate=75), 8.0, 0.0, 0.0, 200.0),
        ("powerline_baseline", dict(noise=0.05, heart_rate=65), 3.0, 25.0, 80.0, 200.0),
        ("tachy_140", dict(noise=0.02, heart_rate=140), 2.0, 0.0, 0.0, 200.0),
        ("inverted_lowamp", dict(noise=0.03, heart_rate=60), 2.0, 0.0, 10.0, -60.0),
    ]
    cases = []
    for i, (name, kw, white, mains, wander, gain) in enumerate(variants):
        ecg = nk.ecg_simulate(duration=dur, sampling_rate=fs, heart_rate_std=3, random_state=300 + i, **kw)
        t = np.arange(len(ecg)) / fs
        rng = np.random.default_rng(400 + i)
        raw = (512 + gain * ecg + wander * np.sin(2 * np.pi * 0.3 * t) + mains * np.sin(2 * np.pi * 50 * t)
               + white * rng.standard_normal(len(t)))
        raw = np.round(raw).astype(np.int16)  # ADC counts
        x = raw.astype(float)
        y = nk.ecg_clean(x, sampling_rate=fs, method="elgendi2010")
        p10 = nk.ecg_findpeaks(y, sampling_rate=fs, method="elgendi2010")["ECG_R_Peaks"]
        p13 = nk.ecg_findpeaks(y, sampling_rate=fs, method="elgendi2013")["ECG_R_Peaks"]
        step = 32
        cases.append({
            "name": name, "sampling_rate": fs, "n": len(x),
            "raw_i16_b64": b64(raw, "<i2"),
            "filtered_step": step, "filtered_f64_b64": b64(y[::step], "<f8"),
            "peaks_elgendi2010": [int(p) for p in p10],
            "peaks_elgendi2013": [int(p) for p in p13],
        })
        print(f"{name}: elgendi2010 {len(p10)} elgendi2013 {len(p13)}", file=sys.stderr)
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "neurokit_elgendi.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"versions": VERSIONS, "cases": cases}, f, separators=(",", ":"))
    print(f"wrote {os.path.abspath(path)} ({os.path.getsize(path) // 1024} KiB)", file=sys.stderr)


if __name__ == "__main__":
    main()
