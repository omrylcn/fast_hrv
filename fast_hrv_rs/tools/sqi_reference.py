"""Reference values for `src/sqi.rs` (dev-time only; the crate itself does not depend on Python).

Output: tests/fixtures/sqi/<case>.json, one file per synthetic 10 s, 512 Hz single-lead ECG window.

Everything here is synthetic (neurokit2.ecg_simulate + programmatic artefacts). No real recordings.

Reference sources per field:
  ksqi_pearson / ksqi_fisher   scipy.stats.kurtosis(x, fisher=False / True)        (bias=True)
  neurokit.{ksqi,psqi,bassqi}   neurokit2.ecg.ecg_quality._ecg_quality_{kSQI,pSQI,basSQI}
                                (the exact functions behind ecg_quality(method="zhao2018"))
  neurokit.zhao_simple          neurokit2.ecg_quality(..., method="zhao2018", approach="simple")
  welch_<nperseg>.{psqi,bassqi} scipy.signal.welch defaults (hann, 50 % overlap, detrend='constant',
                                scaling='density') + NeuroKit2's band integration
                                (bins lo <= f < hi, numpy.trapezoid, 0 -> NaN)
  template                      BioSPPy-style window (-0.2 s, +0.4 s), numpy mean template + np.corrcoef
  bsqi                          Li, Mark & Clifford (2008) matched / (Na + Nb - matched), greedy matching
  raw.{flatline,saturation}     straightforward numpy implementations of the documented definitions

Usage:  /path/to/venv/bin/python tools/sqi_reference.py
Needs:  numpy, scipy, neurokit2
"""

import json
import os

import numpy as np
import scipy
import scipy.signal
import scipy.stats
import neurokit2 as nk
from neurokit2.ecg.ecg_quality import _ecg_quality_basSQI, _ecg_quality_kSQI, _ecg_quality_pSQI

FS = 512
DURATION = 10
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures", "sqi")

ADC_MIN, ADC_MAX = 0, 4095  # 12-bit unsigned front end
ADC_MID = 2048
GAIN = 600.0  # counts per mV
SAT_MARGIN = 2
FLAT_EPS = 0.0  # counts: exact repeats (lead-off / stuck ADC)
FLAT_MIN_S = 0.2
BSQI_TOL_S = 0.15
TEMPLATE_BEFORE_S, TEMPLATE_AFTER_S = 0.2, 0.4


def to_adc(mv, gain=GAIN):
    return np.clip(np.round(ADC_MID + gain * mv), ADC_MIN, ADC_MAX).astype(np.int64)


# ----------------------------------------------------------------------------- reference maths


def band_power_nk(freq, power, lo, hi):
    """NeuroKit2 signal_power: bins lo <= f < hi, trapezoid, 0 -> NaN."""
    where = (freq >= lo) & (freq < hi)
    p = np.trapezoid(y=power[where], x=freq[where])
    return np.nan if p == 0.0 else float(p)


def welch_sqis(x, nperseg):
    f, p = scipy.signal.welch(x, fs=FS, nperseg=nperseg)  # scipy defaults
    psqi = band_power_nk(f, p, 5, 15) / band_power_nk(f, p, 5, 40)
    bassqi = 1.0 - band_power_nk(f, p, 0, 1) / band_power_nk(f, p, 0, 40)
    return {"psqi": psqi, "bassqi": bassqi}


def match_count(a, b, tol):
    i = j = m = 0
    while i < len(a) and j < len(b):
        d = int(a[i]) - int(b[j])
        if abs(d) <= tol:
            m += 1
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return m


def bsqi_ref(a, b, tol_s):
    a, b = np.sort(a), np.sort(b)
    if len(a) == 0 and len(b) == 0:
        return None
    m = match_count(a, b, tol_s * FS)
    return m / (len(a) + len(b) - m)


def template_ref(x, peaks):
    before = int(TEMPLATE_BEFORE_S * FS)
    after = int(TEMPLATE_AFTER_S * FS)
    beats, used = [], []
    for r in peaks:
        a, b = int(r) - before, int(r) + after
        if a < 0 or b > len(x):
            continue
        beats.append(x[a:b])
        used.append(int(r))
    if len(beats) < 3:
        return None
    beats = np.asarray(beats, dtype=float)
    tmpl = beats.mean(axis=0)
    cc = np.array([np.corrcoef(bt, tmpl)[0, 1] for bt in beats])
    return {
        "beat_peaks": used,
        "per_beat": cc.tolist(),
        "median": float(np.median(cc)),
        "mean": float(np.mean(cc)),
        "template_head": tmpl[:8].tolist(),
    }


def flatline_ref(raw, eps, min_s):
    n = len(raw)
    if n == 0:
        return None
    min_len = max(int(np.ceil(min_s * FS)), 2)
    flat = 0
    run = 1
    for i in range(1, n + 1):
        if i < n and abs(float(raw[i]) - float(raw[i - 1])) <= eps:
            run += 1
        else:
            if run >= min_len:
                flat += run
            run = 1
    return flat / n


def saturation_ref(raw, lo, hi, margin):
    raw = np.asarray(raw, dtype=float)
    return float(np.mean((raw <= lo + margin) | (raw >= hi - margin)))


def finite_or_none(v):
    v = float(v)
    return v if np.isfinite(v) else None


# ----------------------------------------------------------------------------- cases


def make_cases():
    rng = np.random.default_rng(2018)
    t = np.arange(FS * DURATION) / FS
    def base(hr, seed):
        ecg = nk.ecg_simulate(duration=DURATION, sampling_rate=FS, heart_rate=hr, noise=0.0, random_state=seed)
        return ecg + 0.005 * rng.standard_normal(t.size)  # ~3 counts of front-end noise
    cases = {}

    clean = base(70, 1)
    cases["clean"] = to_adc(clean)

    noisy = base(75, 2) + 0.15 * rng.standard_normal(t.size)
    # EMG-like bursts
    emg = rng.standard_normal(t.size) * (0.25 * (np.sin(2 * np.pi * 0.4 * t) > 0.3))
    cases["noisy"] = to_adc(noisy + emg)

    wander = base(65, 3) + 0.9 * np.sin(2 * np.pi * 0.25 * t) + 0.4 * np.sin(2 * np.pi * 0.6 * t + 1.0) + 0.05 * t
    cases["baseline_wander"] = to_adc(wander, gain=500.0)

    flat = to_adc(base(70, 4))
    flat[int(3.0 * FS):int(7.0 * FS)] = flat[int(3.0 * FS)]  # 4 s lead-off
    cases["flatline"] = flat

    cases["saturated"] = to_adc(base(80, 5) + 0.3 * np.sin(2 * np.pi * 0.3 * t), gain=2500.0)

    mains = base(70, 6) + 0.35 * np.sin(2 * np.pi * 50.0 * t)
    cases["mains_50hz"] = to_adc(mains)

    tachy = base(150, 7)
    cases["tachy_150"] = to_adc(tachy)
    return cases


def reference(name, raw, filtered=False):
    raw = np.asarray(raw, dtype=np.int64)
    x = raw.astype(float)  # SQIs are scale invariant; computed on the ADC counts as float
    if filtered:
        # NeuroKit-cleaned signal (as zhao2018 expects), rounded so the JSON value is exact
        x = np.round(nk.ecg_clean(x, sampling_rate=FS, method="neurokit"), 6)

    cleaned = nk.ecg_clean(x, sampling_rate=FS, method="neurokit")
    _, info = nk.ecg_peaks(cleaned, sampling_rate=FS, method="neurokit")
    peaks_a = np.asarray(info["ECG_R_Peaks"], dtype=int)
    try:
        _, info_b = nk.ecg_peaks(cleaned, sampling_rate=FS, method="pantompkins1985")
        peaks_b = np.asarray(info_b["ECG_R_Peaks"], dtype=int)
    except Exception:  # noqa: BLE001
        peaks_b = np.asarray([], dtype=int)

    out = {
        "name": name,
        "fs": FS,
        "neurokit_version": nk.__version__,
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "raw": raw.tolist(),
        "signal": x.tolist() if filtered else None,  # None -> SQIs computed on raw counts
        "adc": {"min": ADC_MIN, "max": ADC_MAX, "margin": SAT_MARGIN},
        "flat": {"eps": FLAT_EPS, "min_duration_s": FLAT_MIN_S},
        "flatline_fraction": flatline_ref(raw, FLAT_EPS, FLAT_MIN_S),
        "saturation_fraction": saturation_ref(raw, ADC_MIN, ADC_MAX, SAT_MARGIN),
        "ksqi_pearson": finite_or_none(scipy.stats.kurtosis(x, fisher=False)),
        "ksqi_fisher": finite_or_none(scipy.stats.kurtosis(x, fisher=True)),
        "neurokit": {
            "ksqi": finite_or_none(_ecg_quality_kSQI(x)),
            "psqi": finite_or_none(_ecg_quality_pSQI(x, sampling_rate=FS)),
            "bassqi": finite_or_none(_ecg_quality_basSQI(x, sampling_rate=FS)),
            "zhao_simple": nk.ecg_quality(x, rpeaks=peaks_a, sampling_rate=FS, method="zhao2018", approach="simple"),
        },
        "welch_2048": {k: finite_or_none(v) for k, v in welch_sqis(x, 2048).items()},
        "welch_1024": {k: finite_or_none(v) for k, v in welch_sqis(x, 1024).items()},
        "peaks_a": peaks_a.tolist(),
        "peaks_b": peaks_b.tolist(),
        "bsqi_tol_s": BSQI_TOL_S,
        "bsqi": bsqi_ref(peaks_a, peaks_b, BSQI_TOL_S),
        "bsqi_tol_005": bsqi_ref(peaks_a, peaks_b, 0.05),
        "template": template_ref(x, peaks_a),
    }
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cases = [(n, r, False) for n, r in make_cases().items()]
    cases.append(("clean_filtered", dict((n, r) for n, r, _ in cases)["clean"], True))
    cases.append(("noisy_filtered", dict((n, r) for n, r, _ in cases)["noisy"], True))
    for name, raw, filtered in cases:
        ref = reference(name, raw, filtered)
        path = os.path.join(OUT_DIR, f"{name}.json")
        with open(path, "w") as fh:
            json.dump(ref, fh, separators=(",", ":"))
        print(f"{name:16s} k={ref['ksqi_pearson']:.3f} nk_p={ref['neurokit']['psqi']} nk_bas={ref['neurokit']['bassqi']}"
              f" w2048={ref['welch_2048']} zhao={ref['neurokit']['zhao_simple']}"
              f" bsqi={ref['bsqi']} flat={ref['flatline_fraction']:.3f} sat={ref['saturation_fraction']:.3f}"
              f" nA={len(ref['peaks_a'])} nB={len(ref['peaks_b'])}")


if __name__ == "__main__":
    main()
