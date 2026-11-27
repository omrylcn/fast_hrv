# fast_hrv

Fast HRV analysis library. R-peak detection, artifact correction, and HRV metrics from ECG signals.

## Performance

Rust crate (`fast_hrv_rs`, v0.9.1) against NeuroKit2 0.3.0 on the same 512 Hz synthetic ECG
(`nk.ecg_simulate`, 70 bpm), one thread, median of repeated runs. Steps: cleaning, R peaks, Kubios
correction and all HRV metrics (time + frequency + nonlinear).

| Record | Beats | NeuroKit2 | fast-hrv (Rust) | Faster |
|--------|-------|-----------|-----------------|--------|
| 1 min  | 69    | 32.6 ms   | 0.82 ms         | 40×    |
| 5 min  | 350   | 406 ms    | 6.1 ms          | 66×    |
| 30 min | 2100  | 12.8 s    | 95 ms           | 135×   |
| 60 min | 4201  | 50.9 s    | 356 ms          | 143×   |

Per step on the 60 min record: cleaning 1.25×, R peaks 6.8×, Kubios correction 21×, time domain 65×,
frequency domain 106×, all HRV metrics 177× faster. Same result: R peaks 100 % identical (±2 samples),
RMSSD / SDNN / SampEn identical (difference < 1e-14); LF/HF differs by 3–8 % (spectral estimation
details), and fast-hrv does not report LF below 180 s by design.

Reproduce: `OMP_NUM_THREADS=1 python fast_hrv_rs/tools/neurokit_benchmark.py --minutes 1 5 30 60`
(needs `neurokit2` and a Rust toolchain). The Python packages (`fast_hrv` NumPy / Numba) were not
re-measured in this run.

## Options

| Option | Note |
|--------|------|
| **fast_hrv (NumPy)** | Pure Python, few dependencies |
| **fast_hrv (Numba)** | First call slow (JIT warm-up) |
| **fast_hrv_rs (binding)** | Rust core from Python (PyO3), needs a Rust build |
| **fast_hrv_rs (native)** | Rust crate; streaming, signal quality, full metric set |

---

## Installation

### fast_hrv (Python)

```bash
# Dependencies
uv pip install numpy scipy neurokit2

# For Numba (optional, but recommended)
uv pip install numba
```

### fast_hrv_rs (Rust → Python binding)

```bash
cd fast_hrv_rs

# Install with Maturin
pip install maturin
maturin develop --release

# or with uv
uv pip install maturin
uv run maturin develop --release
```

### fast_hrv_rs (Native Rust)

```bash
cd fast_hrv_rs
cargo build --release

# With SIMD optimizations
cargo build --release --features simd
```

---

## Quick Start

### Option 1: fast_hrv NumPy

```python
from fast_hrv.ecg import process_ecg

result = process_ecg(ecg_signal, sampling_rate=512, correct_artifacts=True)

print(f"R-peaks: {len(result['rpeaks'])}")
print(f"Mean HR: {result['stats']['mean_hr_bpm']:.1f} bpm")
print(f"RMSSD: {result['stats']['rmssd_ms']:.2f} ms")
```

### Option 2: fast_hrv Numba (Faster)

```python
from fast_hrv.ecg_numba import process_ecg

result = process_ecg(ecg_signal, sampling_rate=512, correct_artifacts=True)

print(f"R-peaks: {len(result['rpeaks'])}")
print(f"SDNN: {result['stats']['sdnn_ms']:.2f} ms")
```

### Option 3: fast_hrv_rs Python Binding (Fastest)

```python
import numpy as np
import fast_hrv_rs

result = fast_hrv_rs.process_ecg(ecg_signal, 512, True)

print(f"R-peaks: {len(result['rpeaks'])}")
print(f"RMSSD: {result['stats']['rmssd_ms']:.2f} ms")

# HRV metrics only
rr = np.array([800.0, 810.0, 795.0, 805.0])
metrics = fast_hrv_rs.hrv_metrics(rr)
```

### Option 4: fast_hrv_rs Native Rust

```rust
use fast_hrv_rs::*;

let result = process_ecg(&ecg_signal, 512, true);

println!("R-peaks: {}", result.rpeaks.len());
println!("RMSSD: {:.2} ms", result.stats.rmssd_ms);
```

---

## Features

The Rust crate is the most complete; see [`fast_hrv_rs/README.md`](fast_hrv_rs/README.md) for the full API.

- **Cleaning:** NeuroKit2-compatible zero-phase cleaning (0.5 Hz high-pass + powerline moving average),
  a causal streaming filter (0.5–40 Hz, optional adaptive 50/60 Hz notch); live packet-by-packet output is
  bit-identical to processing the whole record.
- **R peaks:** NeuroKit2 method, Elgendi (reproduces NeuroKit2's presets), sub-sample refinement, two
  detectors with their agreement (bSQI), a streaming detector with the same output.
- **Artifacts:** Kubios correction (Lipponen & Tarvainen 2019, NeuroKit2-equivalent output) or an exclusion
  mode that removes ectopic / suspect beats without moving any.
- **Signal quality:** flat line, saturation, kSQI, pSQI, basSQI, bSQI, template correlation, RR plausibility.
- **HRV metrics:** time domain (incl. lnRMSSD, Baevsky stress index), frequency domain (FFT and
  Lomb–Scargle), nonlinear (SampEn, ApEn, DFA, Poincaré, CSI/CVI), asymmetry, fragmentation; a minimum
  window rule for every metric.
- **Breathing rate from ECG** (R amplitude + RR modulation).

---

## Project Structure

```
bio_works/
├── fast_hrv/                    # Python package
│   ├── __init__.py
│   ├── ecg.py                   # ECG processing (NumPy)
│   ├── ecg_numba.py             # ECG processing (Numba JIT)
│   ├── peaks.py                 # Peak detection (NumPy)
│   ├── peaks_numba.py           # Peak detection (Numba JIT)
│   ├── hrv.py                   # HRV metrics
│   └── utils.py                 # Utility functions
│
├── fast_hrv_rs/                 # Rust package
│   ├── src/
│   │   ├── lib.rs               # Main module
│   │   ├── ecg.rs               # ECG processing
│   │   ├── peaks.rs             # Peak detection
│   │   ├── hrv.rs               # HRV metrics
│   │   ├── simd.rs              # SIMD optimizations
│   │   └── python.rs            # PyO3 bindings
│   ├── Cargo.toml
│   └── pyproject.toml           # Maturin config
│
└── README.md                    # This file
```

---

## API Reference

### fast_hrv (Python)

```python
# ECG processing
from fast_hrv.ecg import process_ecg, ecg_clean, compute_quality
from fast_hrv.ecg_numba import process_ecg  # Numba version

# Peak detection
from fast_hrv.peaks import ecg_peaks, find_peaks, fix_peaks
from fast_hrv.peaks_numba import ecg_peaks  # Numba version

# HRV metrics
from fast_hrv.hrv import (
    get_hrv_features,      # All metrics
    get_time_domain,       # SDNN, RMSSD, pNN50...
    get_nonlinear,         # ApEn, SampEn, DFA...
    get_poincare,          # SD1, SD2, CSI...
)
```

### fast_hrv_rs (Python binding)

```python
import fast_hrv_rs

# ECG processing
fast_hrv_rs.process_ecg(signal, sampling_rate, correct_artifacts)
fast_hrv_rs.ecg_clean(signal, sampling_rate)
fast_hrv_rs.find_peaks(signal, sampling_rate)
fast_hrv_rs.fix_peaks(peaks, sampling_rate)

# HRV metrics
fast_hrv_rs.hrv_metrics(rr)  # All metrics
fast_hrv_rs.rmssd(rr)
fast_hrv_rs.sdnn(rr)
fast_hrv_rs.pnn50(rr)
fast_hrv_rs.sampen(rr, m, r)
fast_hrv_rs.dfa_alpha1(rr)
fast_hrv_rs.poincare_sd1(rr)
```

### fast_hrv_rs (Native Rust)

```rust
use fast_hrv_rs::*;

// ECG processing
process_ecg(&signal, sampling_rate, correct_artifacts);
ecg_clean(&signal, sampling_rate);
find_peaks_default(&signal, sampling_rate);
fix_peaks_default(&peaks, sampling_rate);

// HRV metrics
compute_rmssd(&rr);
std_ddof(&rr, 1);  // SDNN
compute_pnn50(&rr);
compute_sampen(&rr, 2, Some(0.2));
compute_dfa(&rr, (4, 16), (16, None));
compute_poincare(&rr);
```

---

## Accuracy

Against NeuroKit2 0.3.0 on 1, 5, 30 and 60 min synthetic records (benchmark above): R peaks identical
(±2 samples) in all records; RMSSD, SDNN and SampEn identical to < 1e-14 relative. LF/HF: the default (Task Force / Kubios practice,
4 Hz cubic spline, 64 s Welch segments) differs from NeuroKit2 by a few percent because of the window length;
`frequency_domain_neurokit` reproduces `nk.hrv_frequency` (LF/HF ≤ 3e-13 relative).
The Rust test suite checks NeuroKit2 parity on fixtures (`fast_hrv_rs/tests/neurokit_parity.rs`).
Agreement with NeuroKit2 shows the methods are implemented as specified; accuracy against a reference
device (e.g. a chest strap or Holter) is a separate validation and not claimed here.

---

## Tests

```bash
# Python tests
cd fast_hrv
pytest tests/

# Rust tests (195 tests, incl. NeuroKit2 parity and streaming = batch)
cd fast_hrv_rs
cargo test --release

# Speed and agreement against NeuroKit2
OMP_NUM_THREADS=1 python tools/neurokit_benchmark.py --minutes 1 5 30 60
```

---

## References

- **NeuroKit2**: Makowski et al. (2021) - "NeuroKit2: A Python toolbox for neurophysiological signal processing"
- **Artifact Correction**: Lipponen & Tarvainen (2019) - "A robust algorithm for heart rate variability time series artefact correction"
- **HRV Guidelines**: Task Force (1996) - "Heart rate variability: standards of measurement"
- **DFA**: Peng et al. (1995) - "Quantification of scaling exponents and crossover phenomena"
- **Sample Entropy**: Richman & Moorman (2000)

---



## Roadmap

- [x] Frequency-domain HRV (LF, HF, LF/HF; FFT and Lomb–Scargle)
- [x] Streaming / real-time mode (bit-identical to batch)
- [x] LF/HF difference to NeuroKit2 explained (window length, interpolation, band integration); `frequency_domain_neurokit` reproduces NeuroKit2 (≤ 3e-13)
- [ ] Validation against a reference device
- [ ] Published packages (crates.io, PyPI) and pre-built wheels (manylinux, macOS, Windows)
