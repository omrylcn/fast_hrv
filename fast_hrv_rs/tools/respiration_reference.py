"""NeuroKit2 ECGSYN reference for tests/respiration.rs (dev-time only, run once).

nk.ecg_simulate(method="ecgsyn") draws the RR process with a respiratory (HF) component at 0.25 Hz
(15 breaths/min) and adds respiratory baseline/amplitude effects; the test checks that fast-hrv finds
~15 breaths/min in each 60 s window. Stores int16-scaled samples (base64) to keep the file small.

Usage: python tools/respiration_reference.py      Needs: numpy, neurokit2
"""
import base64
import json
import os

import neurokit2 as nk
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "data", "respiration_ecgsyn.json")
cases = []
for hr, seed in [(62, 1), (78, 3)]:
    x = nk.ecg_simulate(duration=120, sampling_rate=512, heart_rate=hr, heart_rate_std=4, method="ecgsyn", random_state=seed)
    q = np.round(x * 1000).astype("<i2")
    cases.append({"name": f"ecgsyn_hr{hr}", "fs": 512, "resp_bpm": 15.0, "ecg_i16": base64.b64encode(q.tobytes()).decode()})
json.dump({"versions": {"neurokit2": nk.__version__, "numpy": np.__version__}, "cases": cases}, open(OUT, "w"))
print("wrote", OUT, os.path.getsize(OUT) // 1024, "KiB")
