"""NeuroKit2 referans değerleri üretir (sadece geliştirme sırasında, bir kez).

Çıktı: tests/data/neurokit_reference.json
  [{ "name", "rri_ms": [...], "neurokit": { metrik: değer } }]

Kullanım:  python tools/neurokit_reference.py
Gereksinim: numpy, neurokit2 (kütüphanenin kendisi Python'a bağlı değildir).
"""
import json
import os
import sys

import numpy as np
import neurokit2 as nk

OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "data", "neurokit_reference.json")

# NeuroKit2 sütun adı → fast-hrv rapor alanı
MAP = {
    "HRV_MeanNN": "time.mean_nn", "HRV_SDNN": "time.sdnn", "HRV_RMSSD": "time.rmssd", "HRV_SDSD": "time.sdsd",
    "HRV_pNN50": "time.pnn50", "HRV_pNN20": "time.pnn20", "HRV_MedianNN": "time.median_nn",
    "HRV_MadNN": "time.mad_nn", "HRV_IQRNN": "time.iqr_nn", "HRV_MinNN": "time.min_nn", "HRV_MaxNN": "time.max_nn",
    "HRV_CVNN": "time.cvnn", "HRV_CVSD": "time.cvsd", "HRV_HTI": "time.hti", "HRV_TINN": "time.tinn",
    "HRV_VLF": "frequency.vlf", "HRV_LF": "frequency.lf", "HRV_HF": "frequency.hf", "HRV_LFHF": "frequency.lf_hf",
    "HRV_LFn": "frequency.lfnu_frac", "HRV_HFn": "frequency.hfnu_frac",
    "HRV_SD1": "nonlinear.sd1", "HRV_SD2": "nonlinear.sd2", "HRV_SD1SD2": "nonlinear.sd1_sd2",
    "HRV_CSI": "nonlinear.csi", "HRV_CVI": "nonlinear.cvi", "HRV_SampEn": "nonlinear.sampen",
    "HRV_ApEn": "nonlinear.apen", "HRV_DFA_alpha1": "nonlinear.dfa_alpha1", "HRV_DFA_alpha2": "nonlinear.dfa_alpha2",
    "HRV_GI": "asymmetry.gi", "HRV_PI": "asymmetry.pi", "HRV_SD1d": "asymmetry.sd1d", "HRV_SD1a": "asymmetry.sd1a",
    "HRV_C1d": "asymmetry.c1d", "HRV_C1a": "asymmetry.c1a",
}


def reference(name, rri_ms):
    rri_ms = np.asarray(rri_ms, dtype=float)
    peaks = np.concatenate([[0], np.cumsum(rri_ms)]).round().astype(int)  # 1000 Hz → ms = örnek
    df = nk.hrv(peaks, sampling_rate=1000)
    out = {}
    for col, field in MAP.items():
        if col in df.columns:
            v = float(df[col].iloc[0])
            out[field] = v if np.isfinite(v) else None
    return {"name": name, "rri_ms": [float(x) for x in rri_ms], "neurokit": out,
            "neurokit_version": nk.__version__}


def main():
    cases = []
    rng = np.random.default_rng(42)
    # 1) 5 dk, solunum + baroreseptör modülasyonu + gürültü
    for seed, (resp, baro) in enumerate([(25, 15), (10, 30), (40, 5)]):
        rri, t = [], 0.0
        r = np.random.default_rng(seed)
        while t < 300:
            v = 800 + resp * np.sin(2 * np.pi * 0.25 * t) + baro * np.sin(2 * np.pi * 0.1 * t) + r.normal(0, 8)
            rri.append(v)
            t += v / 1000
        cases.append(reference(f"synthetic_resp{resp}_baro{baro}", rri))
    # 2) NeuroKit'in kendi simüle EKG'si → R tepe → RR (gerçekçi atım dizisi)
    for hr, dur in [(70, 300), (90, 180)]:
        ecg = nk.ecg_simulate(duration=dur, sampling_rate=500, heart_rate=hr, random_state=int(rng.integers(1e6)))
        _, info = nk.ecg_peaks(nk.ecg_clean(ecg, sampling_rate=500), sampling_rate=500)
        rri = np.diff(info["ECG_R_Peaks"]) / 500 * 1000
        cases.append(reference(f"nk_simulated_hr{hr}_{dur}s", rri))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=1)
    print(f"{len(cases)} vaka yazıldı: {os.path.abspath(OUT)} (neurokit2 {nk.__version__})", file=sys.stderr)


if __name__ == "__main__":
    main()

