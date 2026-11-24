"""NeuroKit2 reference values for tests/neurokit_parity.rs (dev-time only, run once).

The crate never calls Python: these JSON files are committed and the Rust tests read them.
All inputs are synthetic (formulas below or nk.ecg_simulate); no real recordings.

Outputs (tests/data/):
  neurokit_reference.json  HRV metrics. Each case stores integer R-peak sample indices + fs; NeuroKit2
                           and fast-hrv get exactly the same peaks (no rounding to whole ms).
                           Frequency metrics use hrv_frequency(normalize=False) so powers are in ms².
  neurokit_ecg.json        ecg_clean(method="neurokit") and ecg_peaks on 512 Hz simulated ECG with
                           noise variants. Raw input is int16 ADC counts, outputs are float64, both
                           base64 (little endian) to keep the file small.
  neurokit_kubios.json     signal_fixpeaks(method="Kubios") on series with injected artifacts:
                           first-pass detection (_find_artifacts) and final corrected peaks.

Usage:  python tools/neurokit_reference.py
Needs:  numpy, scipy, pandas, neurokit2
"""
import base64
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import scipy
import neurokit2 as nk
from neurokit2.signal.signal_fixpeaks import _find_artifacts

warnings.filterwarnings("ignore")
DATA = os.path.join(os.path.dirname(__file__), "..", "tests", "data")

VERSIONS = {
    "neurokit2": nk.__version__,
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "pandas": pd.__version__,
}

# NeuroKit2 column -> fast-hrv field
TIME = {
    "HRV_MeanNN": "time.mean_nn", "HRV_SDNN": "time.sdnn", "HRV_RMSSD": "time.rmssd", "HRV_SDSD": "time.sdsd",
    "HRV_pNN50": "time.pnn50_nk", "HRV_pNN20": "time.pnn20_nk", "HRV_MedianNN": "time.median_nn",
    "HRV_MadNN": "time.mad_nn", "HRV_IQRNN": "time.iqr_nn", "HRV_MinNN": "time.min_nn", "HRV_MaxNN": "time.max_nn",
    "HRV_CVNN": "time.cvnn", "HRV_CVSD": "time.cvsd", "HRV_HTI": "time.hti", "HRV_TINN": "time.tinn",
}
FREQ = {"HRV_VLF": "frequency.vlf", "HRV_LF": "frequency.lf", "HRV_HF": "frequency.hf", "HRV_LFHF": "frequency.lf_hf",
        "HRV_LnHF": "frequency.lnhf"}
NONLINEAR = {
    "HRV_SD1": "nonlinear.sd1", "HRV_SD2": "nonlinear.sd2", "HRV_SD1SD2": "nonlinear.sd1_sd2",
    "HRV_CSI": "nonlinear.csi", "HRV_CVI": "nonlinear.cvi", "HRV_SampEn": "nonlinear.sampen",
    "HRV_ApEn": "nonlinear.apen", "HRV_DFA_alpha1": "nonlinear.dfa_alpha1", "HRV_DFA_alpha2": "nonlinear.dfa_alpha2",
    "HRV_GI": "asymmetry.gi", "HRV_PI": "asymmetry.pi", "HRV_SD1d": "asymmetry.sd1d", "HRV_SD1a": "asymmetry.sd1a",
    "HRV_C1d": "asymmetry.c1d", "HRV_C1a": "asymmetry.c1a",
    "HRV_PIP": "fragmentation.pip", "HRV_IALS": "fragmentation.ials", "HRV_PSS": "fragmentation.pss",
    "HRV_PAS": "fragmentation.pas",
}


def b64(arr, dtype):
    return base64.b64encode(np.asarray(arr, dtype=dtype).astype(np.dtype(dtype).newbyteorder("<")).tobytes()).decode()


def take(df, mapping):
    out = {}
    for col, field in mapping.items():
        if col in df.columns:
            v = float(df[col].iloc[0])
            out[field] = v if np.isfinite(v) else None
    return out


def hrv_case(name, peaks, fs):
    peaks = np.asarray(peaks, dtype=np.int64)
    ref = {}
    ref.update(take(nk.hrv_time(peaks, sampling_rate=fs), TIME))
    fd = nk.hrv_frequency(peaks, sampling_rate=fs, normalize=False)
    ref.update(take(fd, FREQ))
    # like-with-like normalised power: LF / (LF + HF), as fast-hrv's lfnu (NeuroKit's LFn divides by
    # ULF+VLF+LF+HF+VHF instead)
    lf, hf = float(fd["HRV_LF"].iloc[0]), float(fd["HRV_HF"].iloc[0])
    ref["frequency.lfnu_frac"] = lf / (lf + hf) if np.isfinite(lf + hf) and lf + hf > 0 else None
    ref.update(take(nk.hrv_nonlinear(peaks, sampling_rate=fs), NONLINEAR))
    return {"name": name, "sampling_rate": fs, "peaks": [int(p) for p in peaks], "neurokit": ref}


def synthetic_peaks(seconds, resp, baro, seed, fs):
    """RR (ms) with respiratory (0.25 Hz) and baroreflex (0.1 Hz) modulation + noise -> peak samples."""
    r = np.random.default_rng(seed)
    rri, t = [], 0.0
    while t < seconds:
        v = 800 + resp * np.sin(2 * np.pi * 0.25 * t) + baro * np.sin(2 * np.pi * 0.1 * t) + r.normal(0, 8)
        rri.append(v)
        t += v / 1000
    return np.round((0.5 + np.concatenate([[0], np.cumsum(rri)]) / 1000) * fs).astype(np.int64)


def simulated_peaks(seconds, hr, seed, fs=512):
    ecg = nk.ecg_simulate(duration=seconds, sampling_rate=fs, heart_rate=hr, heart_rate_std=4, random_state=seed)
    _, info = nk.ecg_peaks(nk.ecg_clean(ecg, sampling_rate=fs), sampling_rate=fs)
    return np.asarray(info["ECG_R_Peaks"], dtype=np.int64)


def make_hrv():
    cases = []
    for seed, (resp, baro) in enumerate([(25, 15), (10, 30), (40, 5)]):
        cases.append(hrv_case(f"synthetic_resp{resp}_baro{baro}_300s_512hz", synthetic_peaks(300, resp, baro, seed, 512), 512))
    cases.append(hrv_case("synthetic_resp25_baro15_300s_1000hz", synthetic_peaks(300, 25, 15, 7, 1000), 1000))
    for hr, dur, seed in [(70, 300, 11), (90, 180, 12), (65, 120, 13), (75, 60, 14)]:
        cases.append(hrv_case(f"nk_simulated_hr{hr}_{dur}s_512hz", simulated_peaks(dur, hr, seed), 512))
    return {"versions": VERSIONS, "cases": cases}


def make_ecg():
    fs = 512
    dur = 20
    cases = []
    variants = [
        ("clean", dict(noise=0.01), 0.0, 0.0, 0.0),
        ("noisy", dict(noise=0.1), 6.0, 0.0, 0.0),
        ("powerline_baseline", dict(noise=0.05), 2.0, 20.0, 60.0),
    ]
    for i, (name, kw, white, mains, wander) in enumerate(variants):
        ecg = nk.ecg_simulate(duration=dur, sampling_rate=fs, heart_rate=72, random_state=100 + i, **kw)
        t = np.arange(len(ecg)) / fs
        rng = np.random.default_rng(200 + i)
        raw = (412 + 200 * ecg + wander * np.sin(2 * np.pi * 0.25 * t) + mains * np.sin(2 * np.pi * 50 * t)
               + white * rng.standard_normal(len(t)))
        raw = np.round(raw).astype(np.int16)  # ADC counts
        x = raw.astype(float)
        cleaned = nk.ecg_clean(x, sampling_rate=fs, method="neurokit")
        _, info = nk.ecg_peaks(cleaned, sampling_rate=fs, method="neurokit")
        step = 8
        cases.append({
            "name": name, "sampling_rate": fs, "n": len(x),
            "raw_i16_b64": b64(raw, "<i2"),
            "clean_step": step, "clean_f64_b64": b64(cleaned[::step], "<f8"),
            "peaks": [int(p) for p in info["ECG_R_Peaks"]],
        })
    return {"versions": VERSIONS, "cases": cases}


def realistic_peaks(n, seed, fs=512):
    r = np.random.default_rng(seed)
    t, rr = 0.0, []
    for _ in range(n):
        v = 0.85 + 0.03 * np.sin(2 * np.pi * 0.25 * t) + 0.02 * np.sin(2 * np.pi * 0.1 * t) + r.normal(0, 0.012)
        rr.append(v)
        t += v
    return np.round(np.cumsum(np.r_[0.5, rr]) * fs).astype(np.int64)


def make_kubios():
    fs = 512
    cases = []
    kinds = ["missed", "extra", "ectopic", "longshort", "none"]
    for n in [30, 40, 70, 140, 350]:
        for j, kind in enumerate(kinds):
            seed = 1000 + 10 * n + j
            pk = realistic_peaks(n, seed, fs)
            k = n // 2 + (seed % 7)
            if kind == "missed":
                q = np.delete(pk, k)
            elif kind == "extra":
                q = np.sort(np.append(pk, (pk[k] + pk[k + 1]) // 2 + 7))
            elif kind == "ectopic":
                q = pk.copy(); q[k] -= int(0.3 * fs)
            elif kind == "longshort":
                q = pk.copy(); q[k:] += int(0.35 * fs)
            else:
                q = pk.copy()
            cases.append((f"{kind}_{n}", q, pk))
    # several artifacts in one series (needs more than one correction pass)
    pk = realistic_peaks(200, 77, fs)
    q = np.delete(pk, [60, 61]).copy()
    q[100] -= int(0.25 * fs)
    q = np.sort(np.append(q, (q[150] + q[151]) // 2))
    cases.append(("mixed_200", q, pk))
    out = []
    for name, q, truth in cases:
        info, fixed = nk.signal_fixpeaks(q, sampling_rate=fs, method="Kubios", iterative=True)
        first, _ = _find_artifacts(q, sampling_rate=fs)
        out.append({
            "name": name, "sampling_rate": fs, "peaks": [int(p) for p in q],
            "first_pass": {k: [int(i) for i in first[k]] for k in ["ectopic", "missed", "extra", "longshort"]},
            "fixed": [int(p) for p in fixed],
        })
    return {"versions": VERSIONS, "cases": out}


def write(name, obj):
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))
    print(f"wrote {os.path.abspath(path)} ({os.path.getsize(path) // 1024} KiB)", file=sys.stderr)


def main():
    write("neurokit_reference.json", make_hrv())
    write("neurokit_ecg.json", make_ecg())
    write("neurokit_kubios.json", make_kubios())
    print(f"neurokit2 {nk.__version__}", file=sys.stderr)


if __name__ == "__main__":
    main()
