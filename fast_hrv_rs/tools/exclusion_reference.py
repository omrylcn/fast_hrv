"""Reference outputs for the Kubios threshold floor and the exclusion mode (fast-hrv 0.6.0).

Generates SYNTHETIC beat series only (sinus RR model with RSA + LF + AR(1) noise, with injected
premature + compensatory ectopic beats, missed beats, extra beats and fiducial jitter) and runs the
Python research port of fast-hrv's Kubios (`ektopi_kubios.py`, Lipponen–Tarvainen 2019 with an
absolute threshold floor) and the exclusion reference (`ex_run` below, copied from the research
code `ektopi_gate.py` with the sampling rate as a parameter). No recorded data is read.

usage:  EKTOPI_DIR=/path/to/dir/with/ektopi_kubios.py python tools/exclusion_reference.py

Writes tests/data/kubios_floor_reference.json and tests/data/exclusion_reference.json.
f64 values are stored as their IEEE-754 bit patterns (u64) so that the comparison is exact; boolean
per-beat flags as lists of the indices that are set; output positions relative to the input
(`delta`).
"""
import json
import os
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, os.environ.get("EKTOPI_DIR", "."))
import ektopi_kubios as ek  # noqa: E402

FS = 512.0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "data")
PC_PRE, PC_COMP, SEC_TOL_MS = 0.85, 0.5, 50.0


def bits(x):
    return [struct.unpack("<Q", struct.pack("<d", float(v)))[0] for v in np.asarray(x, float)]


def ints(x):
    return [int(v) for v in x]


def idx(mask):
    return ints(np.flatnonzero(np.asarray(mask, bool)))


def delta(inp, out):
    """Output positions relative to the input: input indices whose position is not in the output
    (removed or moved away) and output positions that are not input positions (inserted or moved
    to), as [output index, bits]. The output is the position-sorted merge of the two."""
    inp, out = np.asarray(inp, float), np.asarray(out, float)
    gone = idx(~np.isin(inp, out))
    new = np.flatnonzero(~np.isin(out, inp))
    return dict(n=len(out), removed=gone, added=[[int(j), bits([out[j]])[0]] for j in new])


# ------------------------------------------------------------------ synthetic series
def synth(level, seed, n_s):
    m, rsa, lf, sig = {"S5": (600, 4.0, 6.0, 2.0), "S10": (650, 9.0, 12.0, 4.0),
                       "S20": (750, 18.0, 20.0, 8.0), "S40": (850, 38.0, 30.0, 15.0)}[level]
    rng = np.random.default_rng(zlib.crc32(f"synth|{level}|{seed}".encode()))
    fr, fl_, p1, p2 = rng.uniform(0.2, 0.3), rng.uniform(0.08, 0.12), rng.uniform(0, 6.3), rng.uniform(0, 6.3)
    t, rr, e = 0.0, [], 0.0
    while t < n_s:
        e = 0.3 * e + sig * rng.standard_t(5) / np.sqrt(5 / 3)
        x = m + rsa * np.sin(2 * np.pi * fr * t + p1) + lf * np.sin(2 * np.pi * fl_ * t + p2) + e
        rr.append(x)
        t += x / 1000.0
    return 1000.0 + np.r_[0.0, np.cumsum(rr)] * FS / 1000.0


def inject(T, re_, ra_, seed, min_sep=3):
    rng = np.random.default_rng(zlib.crc32(f"inject|{len(T)}|{re_}|{ra_}|{seed}".encode()))
    n = len(T)
    ke, ka = int(round(re_ / 100 * n)), int(round(ra_ / 100 * n))
    chosen = []
    for c in rng.permutation(np.arange(2, n - 2)):
        if len(chosen) == ke + ka:
            break
        if all(abs(c - x) >= min_sep for x in chosen):
            chosen.append(int(c))
    types = ["ectopic"] * ke + list(rng.choice(["missed", "extra", "jitter"], len(chosen) - ke))
    X = T.astype(float).copy()
    S = T.astype(float).copy()
    delete, add = [], []
    for c, ty in zip(chosen, types):
        prev, cur = T[c - 1], T[c]
        if ty == "missed":
            delete.append(c)
        elif ty == "extra":
            add.append(prev + rng.uniform(0.3, 0.7) * (cur - prev))
        elif ty == "ectopic":
            X[c] = cur - rng.uniform(0.20, 0.35) * (cur - prev)
            S[c] = X[c]
        elif ty == "jitter":
            X[c] = cur + rng.choice([-1, 1]) * rng.uniform(20.0, 50.0) * FS / 1000.0
    X = np.unique(np.sort(np.r_[np.delete(X, delete), add]))
    return X, np.sort(S)


# ------------------------------------------------------------------ exclusion reference (ektopi_gate.ex_run)
def ex_run(r, prm, dis_in=None, two_stage=False):
    """ektopi_gate.ex_run with agree=False; `dis_in` = per input beat disagreement flags."""
    r = np.asarray(r, float)
    pos, org, _ = ek.fix(r, FS, prm, kinds=("extra", "missed"))
    n = len(pos)
    ect = np.zeros(n, bool)
    sus = np.zeros(n, bool)
    dis = np.zeros(n, bool)
    move_ms = np.zeros(n)
    kub_ect = np.zeros(n, bool)
    pc = np.zeros(n, bool)
    if dis_in is not None:
        dis = (org >= 0) & np.asarray(dis_in, bool)[np.maximum(org, 0)]
    if n >= 3:
        a, sub = ek.detect(pos, FS, prm, return_sub=True)
        rr = np.r_[np.nan, np.diff(pos)]
        m = sub["medrr"] * FS
        sw = np.full(n, -np.inf)
        sw[1:n - 1] = np.abs(rr[2:] - rr[1:n - 1])
        for j in a["ectopic"]:
            c = [b for b in (j - 1, j) if 1 <= b <= n - 2]
            if c:
                kub_ect[max(c, key=lambda b: sw[b])] = True
        b = np.arange(1, n - 1)
        pc[b] = (rr[b] <= PC_PRE * m[b]) & (rr[b + 1] - m[b] >= PC_COMP * (m[b] - rr[b]))
        ect = pc | kub_ect
        ev = np.flatnonzero(ect)
        absorbed = set(ev.tolist()) | set((ev + 1).tolist())
        for j in list(a["longshort"]) + list(a["missed"]) + list(a["extra"]):
            if j in absorbed or not (0 <= j < n):
                continue
            sus[j] = True
        mid = np.full(n, np.nan)
        mid[1:n - 1] = (pos[:-2] + pos[2:]) / 2.0
        move_ms = np.abs(pos - mid) * 1000.0 / FS
        move_ms[~np.isfinite(move_ms)] = np.inf
        for j in list(a["missed"]) + list(a["extra"]):
            if 0 <= j < n and sus[j]:
                move_ms[j] = np.inf
        if two_stage and len(ev):
            cp = pos.copy()
            for bb in ev:
                if 1 <= bb <= n - 2:
                    cp[bb] = (cp[bb - 1] + cp[bb + 1]) / 2.0
            a2 = ek.detect(cp, FS, prm)
            near = set(ev.tolist()) | set((ev + 1).tolist()) | set((ev - 1).tolist())
            mid2 = np.full(n, np.nan)
            mid2[1:n - 1] = (cp[:-2] + cp[2:]) / 2.0
            mv2 = np.abs(cp - mid2) * 1000.0 / FS
            for kind, lst in a2.items():
                for j in lst:
                    if j in near or not (0 <= j < n) or sus[j]:
                        continue
                    sus[j] = True
                    move_ms[j] = np.inf if kind in ("missed", "extra") or not np.isfinite(mv2[j]) else mv2[j]
    flagged = ect | sus | dis
    keep = np.ones(max(n - 1, 0), bool)
    for bb in np.flatnonzero(flagged):
        if bb - 1 >= 0:
            keep[bb - 1] = False
        if bb < n - 1:
            keep[bb] = False
    i2o = np.full(len(r), -1, np.int64)
    i2o[org[org >= 0]] = np.flatnonzero(org >= 0)
    return dict(pos=pos, org=org, ect=ect, sus=sus, dis=dis, move_ms=move_ms, keep=keep, pc=pc,
                kub_ect=kub_ect, inserted=int((org < 0).sum()), removed=int((i2o < 0).sum()))


# ------------------------------------------------------------------ cases
VARIANTS = {
    "default": dict(),
    "alpha6": dict(alpha=6.0),
    "floor30": dict(floor_ms=30.0),
    "floor40_alpha6": dict(floor_ms=40.0, alpha=6.0),
    "floor50": dict(floor_ms=50.0),
}
EX_VARIANTS = ("default", "alpha6", "floor40_alpha6")

CASES = [
    # name, level, duration (s), ectopy %, artifact %, seed
    ("S5_clean", "S5", 200, 0, 0, 0),
    ("S10_clean", "S10", 200, 0, 0, 1),
    ("S10_jitter2", "S10", 200, 0, 2, 2),
    ("S20_e1_a1", "S20", 240, 1, 1, 3),
    ("S10_e3_a1", "S10", 200, 3, 1, 4),
    ("S20_e3_a0", "S20", 240, 3, 0, 5),
    ("S40_e2_a2", "S40", 300, 2, 2, 6),
    ("S10_e5_a2", "S10", 200, 5, 2, 7),
    ("S20_e8_a2", "S20", 240, 8, 2, 8),
    ("S10_e8_a2", "S10", 200, 8, 2, 9),
    ("S20_e0_a3", "S20", 60, 0, 3, 10),
]


def params(v):
    p = dict(ek.DEFAULT)
    p.update(VARIANTS[v])
    return p


def main():
    floor_cases, ex_cases = [], []
    for name, level, dur, re_, ra_, seed in CASES:
        T = synth(level, seed, dur)
        X, S = inject(T, re_, ra_, seed)
        rec = dict(name=name, sampling_rate=FS, positions=bits(X), variants={})
        for v in VARIANTS:
            p = params(v)
            a = ek.detect(X, FS, p)
            pos, _, passes = ek.fix(X, FS, p)
            rec["variants"][v] = dict(
                alpha=p["alpha"], floor_s=p["floor_ms"] / 1000.0,
                first_pass={k: ints(x) for k, x in a.items()},
                fixed=delta(X, pos), n_passes=len(passes))
        floor_cases.append(rec)
        # disagreement model: secondary finds every real beat at its true place (ectopic at its early place)
        d = np.min(np.abs(X[:, None] - S[None, :]), axis=1) * 1000.0 / FS
        dis_in = d > SEC_TOL_MS
        exr = dict(name=name, sampling_rate=FS, positions=bits(X), disagreement=idx(dis_in), runs={})
        for v in EX_VARIANTS:
            p = params(v)
            for two in (False, True):
                for with_dis in (False, True):
                    E = ex_run(X, p, dis_in if with_dis else None, two_stage=two)
                    exr["runs"][f"{v}|{'two' if two else 'one'}|{'dis' if with_dis else 'nodis'}"] = dict(
                        alpha=p["alpha"], floor_s=p["floor_ms"] / 1000.0, two_stage=two, with_disagreement=with_dis,
                        positions=delta(X, E["pos"]), ectopic=idx(E["ect"]), suspect=idx(E["sus"]),
                        disagreement=idx(E["dis"]), kubios_ectopic=idx(E["kub_ect"]), pattern=idx(E["pc"]),
                        suspect_move_ms=[[int(j), bits([E["move_ms"][j]])[0]] for j in np.flatnonzero(E["sus"])],
                        excluded_nn=idx(~E["keep"]), inserted=E["inserted"], removed=E["removed"])
        ex_cases.append(exr)
    meta = dict(generator="tools/exclusion_reference.py", source="ektopi_kubios.py (Python port of fast-hrv v0.5.0 "
                "Kubios + threshold floor) and ex_run from ektopi_gate.py; synthetic data only",
                numpy=np.__version__, pandas=__import__("pandas").__version__,
                pattern=dict(premature=PC_PRE, compensatory=PC_COMP), disagreement_tolerance_ms=SEC_TOL_MS)
    with open(os.path.join(OUT, "kubios_floor_reference.json"), "w") as f:
        json.dump(dict(meta=meta, cases=floor_cases), f, separators=(",", ":"))
    with open(os.path.join(OUT, "exclusion_reference.json"), "w") as f:
        json.dump(dict(meta=meta, cases=ex_cases), f, separators=(",", ":"))
    tot = sum(len(c["positions"]) for c in floor_cases)
    print(f"{len(floor_cases)} cases, {tot} beats")


if __name__ == "__main__":
    main()
