"""Peak timing of the live (streaming) detector on MIT-BIH: detection, fiducial jitter and RMSSD error.

For each record (lead MLII, 360 Hz) the raw signal is fed to `examples/peak_timing.rs` (causal
filter + StreamDetector, 32-sample packets) and the live peaks are matched to the reference beat
annotations (PhysioNet MIT-BIH Arrhythmia Database, ODC-By 1.0). Reported per record:

- F1 at ±50 ms and ±150 ms (a wide tolerance hides timing error; report both),
- median offset of the live peak from the annotation (the causal filter alone adds ~11 ms),
- jitter: RMS of the beat-to-beat change of that offset over consecutive normally conducted beats
  (N, L, R, e, j); this is what reaches RR intervals,
- RMSSD over the same beat triplets from the annotations and from the live peaks.

Usage:  python tools/mitdb_timing.py [--records 100 118 ...] [--env SMOOTH_HZ=15] [--out results.json]
Needs:  wfdb, numpy, a Rust toolchain. Records are downloaded to a temporary directory.
"""
import argparse
import json
import math
import os
import subprocess
import tempfile

import numpy as np
import wfdb

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BEATS = "NLRBAaJSVrFejnE/fQ?"
CONDUCTED = "NLRej"
DEFAULT = "100 101 103 105 106 108 109 111 116 118 119 124 200 203 207 212 214 231".split()


def score(ref, syms, peaks, fs):
    p = np.sort(np.asarray(peaks, float))

    def match(tol):
        pos = {}
        for r in ref:
            k = np.searchsorted(p, r - tol)
            best = None
            while k < len(p) and p[k] <= r + tol:
                if best is None or abs(p[k] - r) < abs(best - r):
                    best = p[k]
                k += 1
            if best is not None:
                pos[r] = best
        return pos

    out = {}
    for tol_ms in (50, 150):
        tp = len(match(tol_ms / 1000 * fs))
        out[f"f1_{tol_ms}ms"] = 100 * 2 * tp / (2 * tp + (len(p) - tp) + (len(ref) - tp))
    pos = match(0.15 * fs)
    offsets = [(pos[r] - r) / fs * 1000 for r in pos]
    jitter, a, b = [], [], []
    for i in range(2, len(ref)):
        if all(syms[i - k] in CONDUCTED and ref[i - k] in pos for k in range(3)):
            jitter.append(((pos[ref[i]] - ref[i]) - (pos[ref[i - 1]] - ref[i - 1])) / fs * 1000)
            a.append(((ref[i] - ref[i - 1]) - (ref[i - 1] - ref[i - 2])) / fs * 1000)
            b.append(((pos[ref[i]] - pos[ref[i - 1]]) - (pos[ref[i - 1]] - pos[ref[i - 2]])) / fs * 1000)
    rms = lambda v: math.sqrt(sum(x * x for x in v) / len(v)) if v else float("nan")  # noqa: E731
    out.update(offset_median_ms=float(np.median(offsets)), jitter_rms_ms=rms(jitter),
               rmssd_reference_ms=rms(a), rmssd_live_ms=rms(b), triplets=len(a))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", default=DEFAULT)
    ap.add_argument("--env", nargs="*", default=[], help="KEY=VALUE passed to the example (e.g. SMOOTH_HZ=15)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    subprocess.run(["cargo", "build", "--quiet", "--release", "--example", "peak_timing"], cwd=ROOT, check=True)
    exe = os.path.join(ROOT, "target", "release", "examples", "peak_timing")
    env = dict(os.environ, **dict(kv.split("=", 1) for kv in a.env))
    rows = []
    with tempfile.TemporaryDirectory() as d:
        for rec in a.records:
            r = wfdb.rdrecord(rec, pn_dir="mitdb")
            ann = wfdb.rdann(rec, "atr", pn_dir="mitdb")
            keep = [(s, sym) for s, sym in zip(ann.sample, ann.symbol) if sym in BEATS]
            path = os.path.join(d, f"{rec}.f64")
            r.p_signal[:, 0].astype("<f8").tofile(path)
            live = json.loads(subprocess.run([exe, str(r.fs), path], capture_output=True, text=True,
                                             check=True, env=env).stdout)["live"]
            s = score([int(x) for x, _ in keep], "".join(sym for _, sym in keep), live, float(r.fs))
            s["record"] = rec
            rows.append(s)
            err = 100 * (s["rmssd_live_ms"] / s["rmssd_reference_ms"] - 1)
            print(f"{rec}: F1 {s['f1_50ms']:.2f} / {s['f1_150ms']:.2f} %  offset {s['offset_median_ms']:+.1f} ms  "
                  f"jitter {s['jitter_rms_ms']:.2f} ms  RMSSD {s['rmssd_live_ms']:.2f} vs {s['rmssd_reference_ms']:.2f} ms ({err:+.2f} %)",
                  flush=True)
    errs = sorted(abs(100 * (s["rmssd_live_ms"] / s["rmssd_reference_ms"] - 1)) for s in rows)
    print(f"median |RMSSD error| {errs[len(errs) // 2]:.2f} %, median jitter "
          f"{sorted(s['jitter_rms_ms'] for s in rows)[len(rows) // 2]:.2f} ms over {len(rows)} records")
    if a.out:
        json.dump({"env": a.env, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
