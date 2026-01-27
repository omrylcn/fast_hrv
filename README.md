# fast_hrv

Fast HRV analysis library. R-peak detection, artifact correction, and HRV metrics from ECG signals.

## Performance

Benchmark: 512 Hz, 75 BPM synthetic ECG (`neurokit2.ecg_simulate()`)

| Duration | NeuroKit2 | fast_hrv (NumPy) | fast_hrv (Numba) | Rust | Speedup |
|----------|-----------|------------------|------------------|------|---------|
| 30s  | 114 ms | 3 ms  | 2 ms  | 0.5 ms | ~228x |
| 60s  | 220 ms | 3 ms  | 3 ms  | 2 ms   | ~110x |
| 180s | 590 ms | 7 ms  | 6 ms  | 3 ms   | ~200x |
| 360s | 1163 ms | 13 ms | 12 ms | 6.5 ms | ~180x |

## Options

| Option | Speed | Note |
|--------|-------|------|
| **fast_hrv (NumPy)** | ~90x | Few dependencies |
| **fast_hrv (Numba)** | ~100x | First call slow (JIT warmup) |
| **fast_hrv_rs (binding)** | ~180x | Requires Rust compilation |
| **fast_hrv_rs (native)** | ~180x | Requires Rust knowledge |

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

### ECG Processing Pipeline

```
Raw ECG → Clean → Find Peaks → Fix Artifacts → Quality → Stats
           │           │             │            │         │
    Butterworth    Gradient     Kubios      Template    HRV
    Bandpass       Based        Method      Matching    Metrics
    0.5-40 Hz
```

### HRV Metrics

| Category | Metrics | Description |
|----------|---------|-------------|
| **Time-Domain** | SDNN, RMSSD, pNN50, pNN20, SDSD | Basic HRV measurements |
| **Nonlinear** | ApEn, SampEn, DFA α1, DFA α2 | Complexity and fractal analysis |
| **Poincaré** | SD1, SD2, SD1/SD2, CSI, CVI | Geometric analysis |
| **Asymmetry** | C1d, C1a, SD1d, SD1a, SDNNd, SDNNa | Asymmetry metrics |

### Artifact Correction

Kubios method (Lipponen & Tarvainen 2019):
- **Ectopic beats**: Early/late beats → interpolation
- **Missed beats**: Missing beats → add new peak
- **Extra beats**: Extra beats → deletion

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

Comparison with NeuroKit2 (360s synthetic ECG):

| Metric | NeuroKit2 | Rust | Diff |
|--------|-----------|------|------|
| N Peaks | 450 | 450 | 0 |
| SDNN | 10.94 ms | 10.95 ms | 0.01% |
| RMSSD | 11.04 ms | 11.04 ms | 0.03% |
| pNN50 | 0.00% | 0.00% | 0 |

---

## Technical Details

### Peak Detection Algorithm

NeuroKit2 method:
```
1. Compute gradient (central differences)
2. Square the gradient
3. Moving average (window = 0.2s)
4. Find local maxima
5. Minimum distance filter (0.3s)
```

### Why Is It So Fast?

**NeuroKit2 Slowness Reasons:**
- Pure Python loops
- DataFrame overhead (pandas)
- General-purpose design

**fast_hrv Optimizations:**
- Numba JIT compilation
- NumPy vectorized operations
- Minimum allocation
- HRV-specific design

**Rust Optimizations:**
- Zero-cost abstractions
- SIMD auto-vectorization
- Stack allocation
- No GIL, no GC

### SIMD Performance

```rust
// Standard: 88 µs
for i in 0..n { sum += data[i]; }

// SIMD (4-wide): 33 µs (2.7x faster)
for i in (0..n).step_by(4) {
    sum0 += data[i];
    sum1 += data[i+1];
    sum2 += data[i+2];
    sum3 += data[i+3];
}
```

### Rust vs Numba

360s ECG benchmark:
- **Numba**: 12 ms
- **Rust Binding**: 6.5 ms (~1.8x)

| Factor | Numba | Rust | Impact |
|--------|-------|------|--------|
| **JIT Overhead** | Type dispatch on each call | Compile-time, zero overhead | ~10-15% |
| **Memory Management** | Python GC, heap allocation | Stack allocation, no GC | ~15-20% |
| **Function Calls** | Python → Numba context switch | Native, can be inlined | ~10-15% |
| **LLVM Optimization** | Dynamic, limited | Full LTO, aggressive inlining | ~20-30% |
| **Type Checking** | Runtime type guards | Compile-time, zero cost | ~5-10% |

**Pipeline Comparison:**

```
Numba Pipeline:
Python → NumPy Array → Numba JIT → LLVM IR → Native Code → Python
         ↑                                              ↓
         └──────────── GIL + GC overhead ──────────────┘

Rust Binding Pipeline:
Python → NumPy Array → Rust (zero-copy) → Native Code → NumPy Array
                      ↑                              ↓
                      └──── No GIL, No GC, Pre-compiled ────┘
```

**Critical Differences:**

1. **Pre-compiled vs JIT**: Rust code is pre-compiled with `maturin develop --release`, Numba does type dispatch on each call

2. **Zero-copy FFI**: PyO3 + numpy crate passes NumPy arrays directly to Rust without copying

3. **LTO (Link Time Optimization)**: Rust can inline across all modules, Numba optimizes on a per-function basis

4. **Stack vs Heap**: Rust keeps small buffers on the stack, Python/Numba allocates everything on the heap

---

## Tests

```bash
# Python tests
cd fast_hrv
pytest tests/

# Rust tests (37 tests)
cd fast_hrv_rs
cargo test

# Benchmark
cargo run --release --bin ecg_breakdown
cargo run --release --features simd --bin simd_bench
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

- [ ] Frequency-domain HRV (LF, HF, LF/HF ratio)
- [ ] GPU acceleration (CUDA/OpenCL)
- [ ] Streaming/real-time mode
- [ ] Pre-built wheels (manylinux, macOS, Windows)
