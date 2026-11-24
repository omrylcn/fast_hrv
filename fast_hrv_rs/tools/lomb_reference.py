"""astropy Lomb-Scargle reference for tests/lomb.rs (dev-time only, run once; the crate never calls Python).

Cases: synthetic NN series (respiratory + baroreflex modulation, noise) on their own uneven beat times,
with and without dropped intervals (exclusion mode). Reference: astropy LombScargle(t, y, fit_mean=False,
center_data=True, normalization="psd").power(freqs); fast-hrv's density is 2 * (T / N) * that value.

Usage: python tools/lomb_reference.py      Needs: numpy, astropy
"""
import json
import os

import astropy
import numpy as np
from astropy.timeseries import LombScargle

OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "data", "lomb_reference.json")


def series(dur, resp_bpm, seed, drop):
    rng = np.random.default_rng(seed)
    t, times, nn = 0.0, [], []
    while t < dur:
        rr = 0.85 + 0.03 * np.sin(2 * np.pi * resp_bpm / 60 * t) + 0.02 * np.sin(2 * np.pi * 0.1 * t) + 0.005 * rng.standard_normal()
        t += rr
        times.append(t)
        nn.append(rr * 1000)
    times, nn = np.array(times), np.array(nn)
    if drop:
        keep = rng.random(len(nn)) > drop
        times, nn = times[keep], nn[keep]
    return times, nn


cases = []
for name, dur, resp, seed, drop in [("resp15_300s", 300, 15, 1, 0.0), ("resp12_300s_drop5", 300, 12, 2, 0.05),
                                     ("resp18_120s", 120, 18, 3, 0.0), ("resp15_600s_drop10", 600, 15, 4, 0.10)]:
    t, y = series(dur, resp, seed, drop)
    T = t[-1] - t[0]
    df = 1 / (5 * T)
    freqs = np.arange(1, int(0.5 / df) + 1) * df
    p = LombScargle(t, y, fit_mean=False, center_data=True, normalization="psd").power(freqs)
    cases.append({"name": name, "t": t.tolist(), "y": y.tolist(), "freqs": freqs.tolist(), "p_astropy": p.tolist()})
json.dump({"versions": {"astropy": astropy.__version__, "numpy": np.__version__}, "cases": cases}, open(OUT, "w"))
print("wrote", OUT, os.path.getsize(OUT) // 1024, "KiB")
