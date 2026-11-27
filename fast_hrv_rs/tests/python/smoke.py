"""Smoke test of the installed wheel: run from outside the repository after `pip install <wheel>`.

Checks what a new user does first: import, HRV from NN intervals (ms), and the ECG pipeline.
"""
import sys

import numpy as np

import fast_hrv_rs as f


def main():
    rr = np.array([800, 810, 790, 805, 795, 820, 780, 800, 815, 790], dtype=float)
    expected = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    got = f.rmssd(rr)
    assert abs(got - expected) < 1e-9, (got, expected)
    m = f.hrv_metrics(rr)
    assert abs(m["rmssd"] - expected) < 1e-9 and abs(m["mean"] - rr.mean()) < 1e-9, m

    fs = 512
    t = np.arange(0, 60, 1 / fs)
    beats = np.arange(0.5, 60, 0.8)
    ecg = np.zeros_like(t)
    for b in beats:
        ecg += 1000 * np.exp(-(((t - b) / 0.012) ** 2))
    out = f.process_ecg(ecg, fs)
    peaks = np.asarray(out["rpeaks"])
    assert abs(len(peaks) - len(beats)) <= 2, (len(peaks), len(beats))
    assert np.all(np.abs(np.diff(peaks) / fs - 0.8) < 0.01), np.diff(peaks)[:5]
    assert f.__version__.count(".") == 2, f.__version__
    print("fast_hrv_rs", f.__version__, "smoke ok", file=sys.stderr)


if __name__ == "__main__":
    main()
