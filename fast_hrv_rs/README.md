# fast_hrv

**Yüksek performanslı HRV (Heart Rate Variability) analiz kütüphanesi.**

ECG sinyallerinden R-peak tespiti, artifact düzeltme ve HRV metriklerini hesaplar. Üç farklı kullanım seçeneği sunar:

| Seçenek | Dil | Hız | Kullanım Kolaylığı |
|---------|-----|-----|-------------------|
| **fast_hrv** | Python + Numba | Hızlı | En kolay |
| **fast_hrv_rs** (Python binding) | Python → Rust | En hızlı | Kolay |
| **fast_hrv_rs** (Native) | Pure Rust | En hızlı | Rust bilgisi gerekli |

## Performans Karşılaştırması

| Süre | NeuroKit2 | fast_hrv (Python) | Rust | Hız Kazanımı |
|------|-----------|-------------------|------|--------------|
| 30s  | 356 ms    | 3.6 ms            | 0.5 ms | **713x** |
| 60s  | 685 ms    | 3.9 ms            | 1.1 ms | **646x** |
| 180s | 2035 ms   | 7.1 ms            | 3.5 ms | **582x** |
| 360s | 4139 ms   | 10.9 ms           | 7.5 ms | **554x** |

---

# Seçenek 1: fast_hrv (Pure Python + Numba)

Rust bilmeden hızlı HRV analizi. Numba JIT compilation ile optimize edilmiş.

## Kurulum

```bash
pip install numpy numba neurokit2
```

## Kullanım

```python
from fast_hrv.ecg_numba import process_ecg

# ECG sinyali (numpy array)
ecg_signal = load_your_ecg_data()
sampling_rate = 512

# Tek satırda tam analiz
result = process_ecg(ecg_signal, sampling_rate=sampling_rate, correct_artifacts=True)

# Sonuçlar
print(f"R-peak sayısı: {len(result['rpeaks'])}")
print(f"Ortalama HR: {result['stats']['mean_hr_bpm']:.1f} bpm")
print(f"SDNN: {result['stats']['sdnn_ms']:.2f} ms")
print(f"RMSSD: {result['stats']['rmssd_ms']:.2f} ms")
print(f"pNN50: {result['stats']['pnn50']:.2f}%")

# Temizlenmiş sinyal ve kalite skoru
ecg_cleaned = result['ecg_cleaned']
quality = result['quality']  # 0-1 arası kalite skoru
```

### Adım Adım İşleme

```python
import numpy as np
from neurokit2.ecg import ecg_clean
from fast_hrv.peaks_numba import ecg_peaks
from fast_hrv.ecg_numba import compute_quality

# 1. Sinyal temizleme (NeuroKit2 Butterworth filter)
ecg_cleaned = ecg_clean(ecg_signal, sampling_rate=512, method="neurokit")

# 2. Peak tespiti + artifact düzeltme
signals, info = ecg_peaks(ecg_cleaned, sampling_rate=512, correct_artifacts=True)
rpeaks = info["ECG_R_Peaks"]

# 3. Kalite değerlendirmesi
quality, quality_per_beat = compute_quality(ecg_cleaned, rpeaks, sampling_rate=512)

# 4. HRV metrikleri
rr_ms = np.diff(rpeaks) / 512 * 1000
sdnn = np.std(rr_ms, ddof=1)
rmssd = np.sqrt(np.mean(np.diff(rr_ms) ** 2))
```

### Sadece HRV Metrikleri

```python
from fast_hrv.hrv import get_hrv_features

# RR intervalleri (ms cinsinden)
rr_intervals = np.array([800.0, 812.5, 795.3, 810.1, 802.7, 815.9])

# Tüm HRV metrikleri
hrv = get_hrv_features(rr_intervals)

print(f"SDNN: {hrv['sdnn']:.2f} ms")
print(f"RMSSD: {hrv['rmssd']:.2f} ms")
print(f"pNN50: {hrv['pnn50']:.2f}%")
print(f"SampEn: {hrv['sampen']:.4f}")
print(f"DFA α1: {hrv['dfa_alpha1']:.3f}")
print(f"SD1: {hrv['sd1']:.2f} ms")
print(f"SD2: {hrv['sd2']:.2f} ms")
```

---

# Seçenek 2: fast_hrv_rs Python Binding (PyO3 + Maturin)

Rust performansını Python'dan kullan. PyO3 ile native binding.

## Kurulum

```bash
# Maturin kurulumu
pip install maturin

# Rust binding'i derle ve kur
cd fast_hrv_rs
maturin develop --release
```

## Cargo.toml Ayarları

```toml
[lib]
name = "fast_hrv_rs"
crate-type = ["cdylib", "rlib"]

[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
numpy = "0.20"
```

## Python'dan Kullanım

```python
import numpy as np
import fast_hrv_rs  # Rust modülü

# ECG sinyali
ecg_signal = np.random.randn(30720)  # 60s @ 512Hz
sampling_rate = 512

# Rust fonksiyonlarını çağır
result = fast_hrv_rs.process_ecg(ecg_signal, sampling_rate, correct_artifacts=True)

print(f"R-peaks: {len(result.rpeaks)}")
print(f"Mean HR: {result.stats.mean_hr_bpm:.1f} bpm")
print(f"RMSSD: {result.stats.rmssd_ms:.2f} ms")

# Sadece HRV metrikleri
rr = np.array([800.0, 810.0, 795.0, 805.0])
rmssd = fast_hrv_rs.compute_rmssd(rr)
sdnn = fast_hrv_rs.std_ddof(rr, 1)
sampen = fast_hrv_rs.compute_sampen(rr, 2, 0.2)
```

## PyO3 Binding Kodu

`src/python.rs` dosyası oluştur:

```rust
use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};

#[pyfunction]
fn process_ecg_py(
    py: Python<'_>,
    signal: PyReadonlyArray1<f64>,
    sampling_rate: u32,
    correct_artifacts: bool,
) -> PyResult<PyObject> {
    let signal = signal.as_slice()?;
    let result = crate::process_ecg(signal, sampling_rate, correct_artifacts);

    // Python dict'e çevir
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("rpeaks", result.rpeaks)?;
    dict.set_item("mean_hr_bpm", result.stats.mean_hr_bpm)?;
    dict.set_item("sdnn_ms", result.stats.sdnn_ms)?;
    dict.set_item("rmssd_ms", result.stats.rmssd_ms)?;
    dict.set_item("pnn50", result.stats.pnn50)?;
    Ok(dict.into())
}

#[pyfunction]
fn compute_rmssd_py(rr: PyReadonlyArray1<f64>) -> PyResult<f64> {
    Ok(crate::compute_rmssd(rr.as_slice()?))
}

#[pyfunction]
fn compute_sampen_py(data: PyReadonlyArray1<f64>, m: usize, r: f64) -> PyResult<f64> {
    Ok(crate::compute_sampen(data.as_slice()?, m, r))
}

#[pymodule]
fn fast_hrv_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_ecg_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_rmssd_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sampen_py, m)?)?;
    Ok(())
}
```

---

# Seçenek 3: fast_hrv_rs (Native Rust)

Maksimum performans için pure Rust kullanımı.

## Kurulum

```bash
# Standart build
cargo build --release

# SIMD optimizasyonları ile
cargo build --release --features simd

# Paralel işleme ile
cargo build --release --features parallel

# Tüm özellikler
cargo build --release --features all
```

## Kullanım

### Temel Kullanım

```rust
use fast_hrv_rs::*;

fn main() {
    let ecg_signal: Vec<f64> = load_ecg_data();
    let sampling_rate = 512;

    // Tam pipeline
    let result = process_ecg(&ecg_signal, sampling_rate, true);

    println!("R-peak sayısı: {}", result.rpeaks.len());
    println!("Ortalama HR: {:.1} bpm", result.stats.mean_hr_bpm);
    println!("SDNN: {:.2} ms", result.stats.sdnn_ms);
    println!("RMSSD: {:.2} ms", result.stats.rmssd_ms);
}
```

### Adım Adım İşleme

```rust
use fast_hrv_rs::*;

// 1. Sinyal temizleme
let cleaned = ecg_clean(&ecg_signal, sampling_rate);

// 2. Peak tespiti
let peaks = find_peaks_default(&cleaned, sampling_rate);

// 3. Artifact düzeltme
let (fixed_peaks, artifacts) = fix_peaks_default(&peaks, sampling_rate);

// 4. Kalite hesaplama
let (quality, quality_per_beat) = compute_quality(&cleaned, &fixed_peaks, sampling_rate);

// 5. İstatistikler
let stats = compute_stats(&fixed_peaks, &quality_per_beat, sampling_rate);
```

### HRV Metrikleri

```rust
use fast_hrv_rs::*;

let rr = vec![800.0, 812.5, 795.3, 810.1, 802.7, 815.9];

// Time-domain
let sdnn = std_ddof(&rr, 1);
let rmssd = compute_rmssd(&rr);
let pnn50 = compute_pnn50(&rr);

// Nonlinear
let sampen = compute_sampen(&rr, 2, 0.2);
let alpha1 = compute_dfa_alpha1(&rr);

// Poincaré
let sd1 = compute_poincare_sd1(&rr);
let sd2 = compute_poincare_sd2(&rr);
```

---

# Modül Yapısı

```
fast_hrv/                    # Python paketi
├── __init__.py
├── ecg_numba.py            # Numba-optimized ECG processing
├── peaks_numba.py          # Peak detection + artifact correction
├── hrv.py                  # HRV metrics
└── utils.py                # Utility functions

fast_hrv_rs/                 # Rust paketi
├── src/
│   ├── lib.rs              # Ana modül, public API
│   ├── ecg.rs              # ECG işleme (clean, quality, stats)
│   ├── peaks.rs            # Peak tespiti ve artifact düzeltme
│   ├── hrv.rs              # HRV metrikleri
│   ├── simd.rs             # SIMD optimizasyonları (opsiyonel)
│   └── python.rs           # PyO3 bindings (opsiyonel)
└── Cargo.toml
```

---

# Özellikler

## ECG İşleme
- **Sinyal Temizleme**: Butterworth bandpass filter (0.5-40 Hz)
- **R-Peak Tespiti**: NeuroKit2 algoritması (gradient-based)
- **Artifact Düzeltme**: Kubios metodu (Lipponen & Tarvainen 2019)
- **Kalite Değerlendirmesi**: Template matching ile beat kalitesi

## HRV Metrikleri

| Kategori | Metrikler |
|----------|-----------|
| Time-domain | SDNN, RMSSD, pNN50, pNN20, SDSD, Mean HR |
| Nonlinear | ApEn, SampEn, DFA α1 |
| Poincaré | SD1, SD2, SD1/SD2, CSI, CVI |
| Asymmetry | C1d, C1a, SD1d, SD1a, Cd, Ca, SDNNd, SDNNa |

---

# Algoritmalar

## Peak Tespiti (NeuroKit2 Metodu)

```
1. Gradient hesapla (merkezi farklar)
2. Gradient karesi al
3. Moving average uygula (window = 0.2s)
4. Lokal maksimumları bul
5. Minimum mesafe filtresi (0.3s)
```

## Artifact Düzeltme (Kubios Metodu)

Lipponen & Tarvainen (2019) algoritması:

```
1. dRR = RR[i] - RR[i-1]
2. mRR = median_filter(RR, window=11)
3. Thresholds:
   - th1 = 0.13 * mRR (short/long)
   - th2 = 0.17 (ectopic)
4. Artifact detection via pattern analysis
5. Correction:
   - Ectopic → interpolate
   - Extra → remove
   - Missed → insert new peak
```

## SIMD Optimizasyonu

```rust
// 2.5x faster with 4-wide unrolling
for i in (0..n).step_by(4) {
    sum0 += data[i];
    sum1 += data[i+1];
    sum2 += data[i+2];
    sum3 += data[i+3];
}
```

---

# Doğruluk Karşılaştırması

Aynı sentetik ECG ile test (30s @ 512Hz):

| Metrik | NeuroKit2 | fast_hrv | Rust | Fark |
|--------|-----------|----------|------|------|
| N Peaks | 37 | 37 | 37 | 0% |
| Mean HR | 74.68 | 74.92 | 74.92 | 0.32% |
| SDNN | 45.65 | 45.65 | 45.65 | **0%** |
| RMSSD | 78.66 | 78.66 | 78.66 | **0%** |
| pNN50 | 62.86 | 62.86 | 62.86 | **0%** |

HRV metrikleri birebir aynı.

---

# Hangi Seçeneği Kullanmalıyım?

| Durum | Öneri |
|-------|-------|
| Hızlı prototipleme, Jupyter notebook | **fast_hrv (Python)** |
| Production, yüksek throughput | **fast_hrv_rs (Python binding)** |
| Embedded, real-time, başka Rust projesi | **fast_hrv_rs (Native)** |
| NeuroKit2 ile uyumluluk gerekli | **fast_hrv (Python)** |

---

# Benchmark Çalıştırma

```bash
# Python benchmark
python -c "
from fast_hrv.ecg_numba import process_ecg
import numpy as np
import time

ecg = np.random.randn(30720)
start = time.time()
for _ in range(100):
    process_ecg(ecg, 512)
print(f'{(time.time()-start)/100*1000:.2f} ms per call')
"

# Rust benchmark
cargo run --release --bin ecg_breakdown

# SIMD benchmark
cargo run --release --features simd --bin simd_bench
```

---

# Testler

```bash
# Python testleri
pytest fast_hrv/tests/

# Rust testleri (37 test)
cargo test

# Belirli modül
cargo test hrv::tests
cargo test peaks::tests

# SIMD testleri
cargo test --features simd
```

---

# Bağımlılıklar

## Python
```
numpy
numba
neurokit2
scipy
```

## Rust
```toml
[dependencies]
ordered-float = "4.2"
rayon = { version = "1.10", optional = true }
pyo3 = { version = "0.20", optional = true }
numpy = { version = "0.20", optional = true }
```

---

# Referanslar

- **NeuroKit2**: Makowski et al. (2021) - "NeuroKit2: A Python toolbox for neurophysiological signal processing"
- **Artifact Correction**: Lipponen & Tarvainen (2019) - "A robust algorithm for heart rate variability time series artefact correction"
- **HRV Guidelines**: Task Force (1996) - "Heart rate variability: standards of measurement"
- **DFA**: Peng et al. (1995) - "Quantification of scaling exponents and crossover phenomena"
- **Sample Entropy**: Richman & Moorman (2000)

---

# Lisans

MIT

# Katkıda Bulunma

1. Fork edin
2. Feature branch: `git checkout -b feature/amazing-feature`
3. Commit: `git commit -m 'Add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Pull Request açın
