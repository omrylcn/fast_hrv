"""Reproducible speed benchmark: fast-hrv vs NeuroKit2 on the same synthetic ECGs (dev-time only).

For each record length (default 1, 5, 30 and 60 min) it generates one 512 Hz ECG with
nk.ecg_simulate, times the same six steps in both libraries (median of several runs, one thread)
and checks that both reach the same answer (peak agreement within ±2 samples, RMSSD / SDNN / SampEn
relative difference). Prints a Markdown table and writes all numbers to a JSON file.

Steps compared (NeuroKit2 function -> fast-hrv function):
  clean      nk.ecg_clean(method="neurokit")        -> ecg_clean
  peaks      nk.ecg_peaks(method="neurokit")        -> find_peaks_default
  kubios     nk.signal_fixpeaks(method="Kubios")    -> fix_peaks_default
  hrv_time   nk.hrv_time                            -> time_domain
  hrv_freq   nk.hrv_frequency                       -> frequency_domain
  hrv_all    nk.hrv (time + frequency + nonlinear)  -> hrv

Usage:  OMP_NUM_THREADS=1 python tools/neurokit_benchmark.py [--minutes 1 5 30 60] [--out results.json]
Needs:  numpy, neurokit2, a Rust toolchain (builds examples/neurokit_bench.rs in release mode).
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import neurokit2 as nk  # noqa: E402

warnings.filterwarnings("ignore")
FS = 512
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
STEPS = ["clean", "peaks", "kubios", "hrv_time", "hrv_frequency", "hrv_all"]


def timed(fn, runs):
    out, times = None, []
    for _ in range(runs):
        t = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t)
    return sorted(times)[len(times) // 2] * 1000.0, out


def neurokit(ecg, runs):
    res = {}
    res["clean"], clean = timed(lambda: nk.ecg_clean(ecg, sampling_rate=FS), runs)
    res["peaks"], (_, info) = timed(lambda: nk.ecg_peaks(clean, sampling_rate=FS), runs)
    peaks = info["ECG_R_Peaks"]
    res["kubios"], fixed = timed(lambda: nk.signal_fixpeaks(peaks, sampling_rate=FS, method="Kubios"), runs)
    fixed = np.asarray(fixed[1] if isinstance(fixed, tuple) else fixed)
    res["hrv_time"], _ = timed(lambda: nk.hrv_time(fixed, sampling_rate=FS), runs)
    res["hrv_frequency"], _ = timed(lambda: nk.hrv_frequency(fixed, sampling_rate=FS), runs)
    res["hrv_all"], allm = timed(lambda: nk.hrv(fixed, sampling_rate=FS), max(1, runs // 3))
    metrics = {k: float(allm[f"HRV_{k}"].iloc[0]) for k in ("RMSSD", "SDNN", "SampEn", "LFHF")}
    return res, fixed, metrics


def agreement(a, b, tol=2):
    a, b = np.sort(np.asarray(a)), np.sort(np.asarray(b))
    i = j = m = 0
    while i < len(a) and j < len(b):
        if abs(int(a[i]) - int(b[j])) <= tol:
            m, i, j = m + 1, i + 1, j + 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return m / max(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, nargs="+", default=[1, 5, 30, 60])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    subprocess.run(["cargo", "build", "--quiet", "--release", "--example", "neurokit_bench"], cwd=ROOT, check=True)
    exe = os.path.join(ROOT, "target", "release", "examples", "neurokit_bench")
    rows = []
    with tempfile.TemporaryDirectory() as d:
        for mins in a.minutes:
            ecg = nk.ecg_simulate(duration=int(mins * 60), sampling_rate=FS, heart_rate=70, noise=0.05, random_state=7)
            path = os.path.join(d, f"ecg_{mins:g}min.f64")
            np.asarray(ecg, dtype="<f8").tofile(path)
            runs = 21 if mins <= 5 else 7
            nk_t, nk_peaks, nk_m = neurokit(ecg, runs)
            fh = json.loads(subprocess.run([exe, str(runs * 3), path], capture_output=True, text=True, check=True).stdout)
            fh_t = {"clean": fh["clean_ms"], "peaks": fh["peaks_ms"], "kubios": fh["kubios_ms"],
                    "hrv_time": fh["hrv_time_ms"], "hrv_frequency": fh["hrv_frequency_ms"], "hrv_all": fh["hrv_all_ms"]}
            rel = lambda x, y: abs(x - y) / abs(y) if y else float("nan")  # noqa: E731
            rows.append({
                "minutes": mins,
                "beats": int(len(nk_peaks)),
                "neurokit_ms": nk_t,
                "fasthrv_ms": fh_t,
                "speedup": {k: nk_t[k] / fh_t[k] for k in STEPS},
                "pipeline_ms": {"neurokit": sum(nk_t[k] for k in ("clean", "peaks", "kubios", "hrv_all")),
                                "fasthrv": sum(fh_t[k] for k in ("clean", "peaks", "kubios", "hrv_all"))},
                "agreement": {"peaks_within_2_samples": agreement(nk_peaks, fh["peaks"]),
                              "rmssd_rel_diff": rel(fh["rmssd"], nk_m["RMSSD"]),
                              "sdnn_rel_diff": rel(fh["sdnn"], nk_m["SDNN"]),
                              "sampen_rel_diff": rel(fh["sampen"], nk_m["SampEn"]),
                              "lf_hf_fasthrv": fh.get("lf_hf"), "lf_hf_neurokit": nk_m["LFHF"]},
            })
            print(f"done {mins:g} min", file=sys.stderr)
    meta = {"neurokit2": nk.__version__, "numpy": np.__version__, "python": platform.python_version(),
            "cpu": platform.processor() or platform.machine(), "load_average": os.getloadavg(),
            "threads": os.environ.get("OMP_NUM_THREADS")}
    print(f"\nNeuroKit2 {meta['neurokit2']} vs fast-hrv, 512 Hz synthetic ECG, median ms (one thread)\n")
    print("| Record | Beats | Step | NeuroKit2 | fast-hrv | Faster |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        for k in STEPS:
            print(f"| {r['minutes']:g} min | {r['beats']} | {k} | {r['neurokit_ms'][k]:.2f} | {r['fasthrv_ms'][k]:.3f} | {r['speedup'][k]:.0f}× |")
        p = r["pipeline_ms"]
        print(f"| {r['minutes']:g} min | {r['beats']} | **pipeline** | **{p['neurokit']:.1f}** | **{p['fasthrv']:.2f}** | **{p['neurokit'] / p['fasthrv']:.0f}×** |")
    print("\nSame result:")
    for r in rows:
        g = r["agreement"]
        print(f"- {r['minutes']:g} min: peaks {100 * g['peaks_within_2_samples']:.1f} % identical (±2 samples), "
              f"RMSSD Δ {100 * g['rmssd_rel_diff']:.3f} %, SDNN Δ {100 * g['sdnn_rel_diff']:.3f} %, SampEn Δ {100 * g['sampen_rel_diff']:.3f} %, "
              f"LF/HF fast-hrv {g['lf_hf_fasthrv']} vs NeuroKit2 {g['lf_hf_neurokit']:.3f} (fast-hrv reports LF only from 180 s; NeuroKit2 computes it on any length)")
    if a.out:
        json.dump({"meta": meta, "rows": rows}, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
