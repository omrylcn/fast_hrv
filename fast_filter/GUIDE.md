# DeepFilter PyTorch - Kapsamlı Teknik Dokümantasyon

## 1. Proje Amacı ve Genel Bakış

### 1.1 Problem Tanımı
ECG (Elektrokardiyogram) sinyalleri, kalp aktivitesini ölçmek için kullanılan biyomedikal sinyallerdir. Bu sinyaller çeşitli gürültü kaynaklarından etkilenir. En yaygın gürültü türlerinden biri **Baseline Wander (BLW)** - yani temel çizgi kaymasıdır.

**Baseline Wander Nedenleri:**
- Solunum hareketi
- Hasta hareketi
- Elektrot empedans değişimleri
- Düşük frekanslı interferanslar (0.5 Hz altı)

**Geleneksel Çözümler:**
- High-pass filtreler (0.5-0.67 Hz cutoff)
- Median filtreler
- Wavelet tabanlı yöntemler

**Bu Projenin Yaklaşımı:**
Derin öğrenme modelleri kullanarak BLW gürültüsünü temizlemek. Avantajı: Adaptif filtreleme, sinyal morfolojisini koruma.

### 1.2 Orijinal Proje
Bu proje, Francisco Perdigon Romero'nun Keras/TensorFlow tabanlı [DeepFilter](https://github.com/fperdigon/DeepFilter) projesinin PyTorch'a dönüştürülmüş halidir.

**Orijinal Çalışma:**
```
Romero, F. P., Pellegrin, L., et al. (2021).
DeepFilter: an ECG baseline wander removal filter using deep learning techniques.
Biomedical Signal Processing and Control.
```

---

## 2. Klasör Yapısı ve Dosya Açıklamaları

```
DeepFilter/
├── pytorch/
│   ├── __init__.py              # Paket tanımı
│   ├── models/
│   │   ├── __init__.py          # Model exportları
│   │   ├── modules.py           # Temel yapı taşları (LFilter, NLFilter, vb.)
│   │   ├── vanilla.py           # Vanilla CNN modelleri
│   │   ├── multibranch.py       # Multi-branch modeller
│   │   ├── fcn_dae.py           # FCN-DAE autoencoder
│   │   └── drnn.py              # DRNN (LSTM tabanlı)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py           # Ana eğitim döngüsü
│   │   ├── losses.py            # Özel loss fonksiyonları
│   │   └── callbacks.py         # Early stopping, LR scheduler
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py           # PyTorch Dataset sınıfı
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py           # Değerlendirme metrikleri
│   │   └── weight_converter.py  # Keras→PyTorch dönüştürücü
│   └── main.py                  # Çalıştırılabilir ana script
```

---

## 3. Modül Mimarileri (Yapı Taşları)

### 3.1 LFilter Module (Linear Filter)
**Dosya:** `models/modules.py`

**Amaç:** 4 paralel dalda farklı kernel boyutlarıyla özellik çıkarımı (linear aktivasyon).

```
                 ┌─── Conv1D(k=3) ───┐
                 │                   │
Input ──────────┼─── Conv1D(k=5) ───┼──── Concat ──── Output
  │              │                   │        │
  │              ├─── Conv1D(k=9) ───┤        │
  │              │                   │        │
  │              └─── Conv1D(k=15)───┘        │
  │                                           │
  └───────────────────────────────────────────┘
         (B, C_in, 512) ──────────> (B, C_out, 512)
```

**Neden farklı kernel boyutları?**
- **k=3:** Kısa süreli, hızlı değişimleri yakalar (QRS kompleksi)
- **k=5:** Orta vadeli morfoloji
- **k=9:** P ve T dalgaları gibi daha geniş yapılar
- **k=15:** Baseline drift gibi yavaş değişimler

**Linear aktivasyon sebebi:** ECG sinyali negatif değerler içerir. ReLU negatif bilgiyi kaybeder.

### 3.2 NLFilter Module (Non-Linear Filter)
**Amaç:** Aynı yapı, ReLU aktivasyonlu.

ReLU'nun avantajı: Non-linear dönüşümler sayesinde daha kompleks pattern'ları öğrenebilir.
Dezavantajı: Negatif değerlerin kaybı (dying ReLU problemi).

### 3.3 LANLFilter Module (Linear AND Non-Linear)
**Amaç:** Her iki yaklaşımın birleşimi - 8 paralel dal.

```
         ┌─── L_Conv(k=3) ────┐
         ├─── L_Conv(k=5) ────┤
         ├─── L_Conv(k=9) ────┤
Input ───┼─── L_Conv(k=15)────┼─── Concat ─── Output
         ├─── NL_Conv(k=3) ───┤   (8 dal)
         ├─── NL_Conv(k=5) ───┤
         ├─── NL_Conv(k=9) ───┤
         └─── NL_Conv(k=15)───┘
```

**Çıkış kanalları:** Her dal `out_channels/8` kanal üretir.

### 3.4 LANLFilter Dilated Module
**Amaç:** Dilated (genişletilmiş) convolution ile daha geniş receptive field.

**Dilation Rate = 3** demek: Kernel elemanları arasında 2 boşluk var.

```
Normal Conv (k=5):   [x x x x x]
Dilated Conv (k=5):  [x . . x . . x . . x . . x]
                      ↑     ↑     ↑     ↑     ↑
                   Sadece bu pozisyonlar kullanılır
```

**Avantajı:** Parametre sayısını artırmadan daha geniş bağlamı görebilir.

**Bu modülde 6 dal var:** 3 linear + 3 non-linear (k=5, 9, 15 - k=3 yok çünkü dilated k=3 anlamsız)

---

## 4. Model Mimarileri

### 4.1 Vanilla Linear (VanillaLinear)
**Dosya:** `models/vanilla.py`

En basit model. 7 ardışık Conv1D katmanı:

```
Input (B,1,512)
    │
    ▼
Conv1D(1→64, k=9) ─── linear
    │
    ▼
Conv1D(64→64, k=9) ─── linear
    │
    ▼
Conv1D(64→32, k=9) ─── linear
    │
    ▼
Conv1D(32→32, k=9) ─── linear
    │
    ▼
Conv1D(32→16, k=9) ─── linear
    │
    ▼
Conv1D(16→16, k=9) ─── linear
    │
    ▼
Conv1D(16→1, k=9) ─── linear
    │
    ▼
Output (B,1,512)
```

**Parametre Sayısı:** ~43,785

### 4.2 Vanilla NonLinear (VanillaNonLinear)
Aynı yapı, tüm katmanlarda ReLU aktivasyon (son katman hariç).

### 4.3 MultiBranch LANL
**Dosya:** `models/multibranch.py`

LANLFilter modüllerini ardışık kullanır + BatchNorm:

```
Input (B,1,512)
    │
    ▼
LANLFilter(1→64) ──► BatchNorm
    │
    ▼
LANLFilter(64→64) ──► BatchNorm
    │
    ▼
LANLFilter(64→32) ──► BatchNorm
    │
    ▼
LANLFilter(32→32) ──► BatchNorm
    │
    ▼
LANLFilter(32→16) ──► BatchNorm
    │
    ▼
LANLFilter(16→16) ──► BatchNorm
    │
    ▼
Conv1D(16→1, k=9) ─── linear
    │
    ▼
Output (B,1,512)
```

**Parametre Sayısı:** ~157,441

### 4.4 MultiBranch LANLD (Dilated)
LANL ve LANLDilated modüllerini dönüşümlü kullanır + Dropout(0.4) + BatchNorm:

```
Input
  │
  ▼
LANLFilter(64) → Dropout(0.4) → BatchNorm
  │
  ▼
LANLDilated(64) → Dropout(0.4) → BatchNorm
  │
  ▼
LANLFilter(32) → Dropout(0.4) → BatchNorm
  │
  ▼
LANLDilated(32) → Dropout(0.4) → BatchNorm
  │
  ▼
LANLFilter(16) → Dropout(0.4) → BatchNorm
  │
  ▼
LANLDilated(16) → Dropout(0.4) → BatchNorm
  │
  ▼
Conv1D(1) → Output
```

**Parametre Sayısı:** ~101,393

### 4.5 FCN-DAE (Fully Convolutional Denoising AutoEncoder)
**Dosya:** `models/fcn_dae.py`

**Kaynak:** Chiang et al. (2019) - "Noise reduction in ECG signals using fully convolutional denoising autoencoders"

**Mimari:** Encoder-Decoder yapısı

```
ENCODER                          DECODER
────────                         ────────
Input: (B,1,512)
    │
Conv1D(40, s=2) ──► 256          ConvT(40, s=2) ◄── 256
    │                                │
Conv1D(20, s=2) ──► 128          ConvT(20, s=2) ◄── 128
    │                                │
Conv1D(20, s=2) ──► 64           ConvT(20, s=2) ◄── 64
    │                                │
Conv1D(20, s=2) ──► 32           ConvT(20, s=2) ◄── 32
    │                                │
Conv1D(40, s=2) ──► 16           ConvT(40, s=2) ◄── 16
    │                                │
Conv1D(1, s=1)  ──► 16  ────────► ConvT(1, s=1)
                   (Bottleneck)
                                 Output: (B,1,512)
```

**Tüm katmanlarda:** ELU aktivasyon + BatchNorm
**Son katman:** Linear aktivasyon

**Parametre Sayısı:** ~267,681

### 4.6 DRNN (Deep Recurrent Neural Network)
**Dosya:** `models/drnn.py`

**Kaynak:** Antczak (2018) - "Deep recurrent neural networks for ECG signal denoising"

```
Input: (B, 1, 512)
    │
    ▼ (permute → B, 512, 1 for LSTM)
    │
LSTM(input=1, hidden=64, return_sequences=True)
    │
    ▼
Linear(64 → 64) + ReLU
    │
    ▼
Linear(64 → 64) + ReLU
    │
    ▼
Linear(64 → 1)
    │
    ▼
Output: (B, 1, 512)
```

**Parametre Sayısı:** ~33,857

**Not:** DRNN farklı tensor format kullanır:
- Diğer modeller: (B, Channels, Sequence)
- LSTM: (B, Sequence, Features)

---

## 5. Tensor Shape Dönüşümleri (Keras → PyTorch)

### 5.1 Temel Fark

| Framework | Format | Örnek Shape |
|-----------|--------|-------------|
| **Keras** | (Batch, Sequence, Channels) | (128, 512, 1) |
| **PyTorch** | (Batch, Channels, Sequence) | (128, 1, 512) |

### 5.2 Dönüşüm Kodları

**NumPy → PyTorch Dataset:**
```python
# Keras formatında gelen data: (N, 512, 1)
X_numpy = load_data()  # shape: (N, 512, 1)

# PyTorch formatına çevir
X_torch = torch.FloatTensor(X_numpy)  # (N, 512, 1)
X_torch = X_torch.permute(0, 2, 1)    # (N, 1, 512)
```

**Model içinde (DRNN için):**
```python
# Input: (B, 1, 512) - PyTorch conv format
x = x.permute(0, 2, 1)  # (B, 512, 1) - LSTM format
x, _ = self.lstm(x)     # (B, 512, 64)
# ... FC layers ...
x = x.permute(0, 2, 1)  # (B, 1, 512) - geri çevir
```

### 5.3 Conv1D Padding

Keras'ta `padding='same'` otomatik çalışır. PyTorch'ta:

```python
# PyTorch'ta same padding hesaplama
padding = kernel_size // 2

# Örnek
nn.Conv1d(64, 64, kernel_size=9, padding=4)  # padding = 9//2 = 4
```

---

## 6. Loss Fonksiyonları

### 6.1 SSD Loss (Sum of Squared Differences)
**Dosya:** `training/losses.py`

```python
def ssd_loss(y_pred, y_true):
    """
    Her örnek için kare farkların toplamı

    Keras:   K.sum(K.square(y_pred - y_true), axis=-2)
    PyTorch: torch.sum((y_pred - y_true) ** 2, dim=-1).mean()

    axis=-2 (Keras) = sinyal boyutu (512)
    dim=-1 (PyTorch) = sinyal boyutu (512, çünkü channels önce)
    """
    return torch.sum((y_pred - y_true) ** 2, dim=-1).mean()
```

**Kullanım:** FCN-DAE modeli

### 6.2 MAD Loss (Maximum Absolute Difference)
```python
def mad_loss(y_pred, y_true):
    """
    Her örnek için maksimum kare fark
    Aşırı hataları cezalandırır
    """
    return torch.max((y_pred - y_true) ** 2, dim=-1)[0].mean()
```

### 6.3 Combined SSD+MAD Loss
```python
def combined_ssd_mad_loss(y_pred, y_true):
    """
    MAD * 50 + SSD

    MAD: Outlier'ları cezalandırır
    SSD: Genel doğruluğu sağlar
    50 faktörü: MAD'ın etkisini artırır
    """
    return mad_loss(y_pred, y_true) * 50 + ssd_loss(y_pred, y_true)
```

**Kullanım:** Vanilla ve MultiBranch modeller

### 6.4 Model-Loss Eşleştirmesi

| Model | Loss Fonksiyonu | Sebep |
|-------|-----------------|-------|
| DRNN | MSELoss | Orijinal paper'da kullanılmış |
| FCN-DAE | ssd_loss | Orijinal paper'da kullanılmış |
| Vanilla L | combined_ssd_mad_loss | Hem genel hem peak doğruluğu |
| Vanilla NL | combined_ssd_mad_loss | " |
| MultiBranch LANL | combined_ssd_mad_loss | " |
| MultiBranch LANLD | combined_ssd_mad_loss | " |

---

## 7. Değerlendirme Metrikleri

### 7.1 SSD (Sum of Squared Differences)
```python
SSD = Σ(y - y_pred)²
```
**Anlamı:** Toplam kare hata. Düşük = iyi.

### 7.2 MAD (Maximum Absolute Difference)
```python
MAD = max(|y - y_pred|)
```
**Anlamı:** En kötü durum hatası. Düşük = iyi.

### 7.3 PRD (Percent Root-mean-square Difference)
```python
PRD = √(Σ(y_pred - y)² / Σ(y_pred - mean(y))²) × 100
```
**Anlamı:** Normalize edilmiş hata yüzdesi. %10 altı iyi kabul edilir.

### 7.4 COS_SIM (Cosine Similarity)
```python
COS_SIM = (y · y_pred) / (||y|| × ||y_pred||)
```
**Anlamı:** Sinyal şekil benzerliği. 1.0 = mükemmel eşleşme.

---

## 8. Eğitim Pipeline

### 8.1 Hyperparameters (Orijinal ile Aynı)

```python
epochs = 100_000      # Early stopping ile kesilir
batch_size = 128
learning_rate = 1e-3
min_learning_rate = 1e-10

# Callbacks
early_stop_patience = 10
lr_reduce_patience = 2
lr_reduce_factor = 0.5
```

### 8.2 Learning Rate Schedule

```
Epoch 0-N:    lr = 1e-3
Val loss 2 epoch düşmezse → lr = 5e-4
Val loss 2 epoch düşmezse → lr = 2.5e-4
...
Val loss 10 epoch düşmezse → Early Stop
```

### 8.3 Data Split

```
Toplam Data
    │
    ├── 70% Train Set ─────► Model eğitimi
    │
    └── 30% Validation Set ► Early stopping & LR scheduling

Test Set (ayrı dosyalardan) ► Final değerlendirme
```

---

## 9. Keras → PyTorch Weight Dönüşümü

### 9.1 Conv1D Weight Format

```
Keras Shape:   (kernel_size, in_channels, out_channels)
PyTorch Shape: (out_channels, in_channels, kernel_size)

Dönüşüm: weights.transpose(2, 1, 0)
```

**Örnek:**
```python
# Keras: Conv1D(filters=64, kernel_size=9, input_shape=(512, 1))
# Weight shape: (9, 1, 64)

# PyTorch: nn.Conv1d(1, 64, kernel_size=9)
# Weight shape: (64, 1, 9)

pytorch_weights = keras_weights.transpose(2, 1, 0)
```

### 9.2 Dense/Linear Weight Format

```
Keras Shape:   (in_features, out_features)
PyTorch Shape: (out_features, in_features)

Dönüşüm: weights.T
```

### 9.3 Bias

Bias değerleri aynı kalır, dönüşüm gerekmez.

### 9.4 BatchNorm

```
Keras: gamma, beta, moving_mean, moving_variance
PyTorch: weight, bias, running_mean, running_var

Dönüşüm: Direkt kopyala (aynı format)
```

### 9.5 LSTM

LSTM weight dönüşümü karmaşıktır:

```
Keras LSTM: kernel, recurrent_kernel, bias
- kernel shape: (input_size, 4*hidden_size)  [i,f,c,o gates]
- recurrent_kernel shape: (hidden_size, 4*hidden_size)

PyTorch LSTM: weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
- weight_ih shape: (4*hidden_size, input_size)
- weight_hh shape: (4*hidden_size, hidden_size)

Gate sırası da farklı:
Keras:   i, f, c, o (input, forget, cell, output)
PyTorch: i, f, c, o (aynı - şanslıyız!)
```

**Dönüşüm:**
```python
weight_ih = keras_kernel.T
weight_hh = keras_recurrent_kernel.T
bias_ih = keras_bias[:hidden_size*4]  # ilk yarı
bias_hh = zeros  # Keras tek bias kullanır
```

---

## 10. Kullanım Örnekleri

### 10.1 Model Oluşturma

```python
from pytorch.models import (
    VanillaLinear, VanillaNonLinear,
    MultiBranchLANL, MultiBranchLANLD,
    FCNDAE, DRNN
)

# Basit model
model = VanillaLinear(signal_size=512)

# En iyi performans
model = MultiBranchLANL(signal_size=512)

# En hızlı (LSTM)
model = DRNN(signal_size=512)
```

### 10.2 Forward Pass

```python
import torch

# Input shape: (batch, channels, sequence)
x = torch.randn(32, 1, 512)

# Forward
with torch.no_grad():
    y = model(x)

print(y.shape)  # (32, 1, 512)
```

### 10.3 Eğitim

```python
from pytorch.training import Trainer
from pytorch.data import ECGDataset
from torch.utils.data import DataLoader

# Dataset
train_dataset = ECGDataset(X_train, y_train)
val_dataset = ECGDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)

# Trainer
trainer = Trainer(model, device='cuda')
trainer.fit(train_loader, val_loader, epochs=100000)
```

### 10.4 Keras Weight'lerini Yükleme

```python
from pytorch.utils import load_keras_to_pytorch

model = MultiBranchLANL()
load_keras_to_pytorch(model, 'Multibranch_LANL_weights.best.hdf5')
```

---

## 11. Performans Karşılaştırması

### 11.1 Model Complexity

| Model | Params | Inference (ms) | Memory (MB) |
|-------|--------|----------------|-------------|
| DRNN | 33K | ~2 | ~5 |
| Vanilla L | 44K | ~1 | ~3 |
| Vanilla NL | 44K | ~1 | ~3 |
| MultiBranch LANLD | 101K | ~3 | ~8 |
| MultiBranch LANL | 157K | ~4 | ~10 |
| FCN-DAE | 268K | ~5 | ~15 |

### 11.2 Filtreleme Kalitesi (Orijinal Paper'dan)

| Model | SSD ↓ | MAD ↓ | PRD ↓ | COS_SIM ↑ |
|-------|-------|-------|-------|-----------|
| IIR Butterworth | 0.176 | 0.147 | 14.92 | 0.9854 |
| FIR | 0.124 | 0.120 | 12.51 | 0.9907 |
| **Vanilla L** | 0.032 | 0.067 | 5.52 | 0.9983 |
| **MultiBranch LANL** | **0.024** | **0.053** | **4.78** | **0.9989** |
| FCN-DAE | 0.033 | 0.068 | 5.60 | 0.9982 |
| DRNN | 0.040 | 0.077 | 6.21 | 0.9976 |

---

## 12. Troubleshooting

### 12.1 Shape Mismatch Hatası
```python
RuntimeError: Expected 3D input, got 2D
```
**Çözüm:** Input'a channel boyutu ekle: `x = x.unsqueeze(1)`

### 12.2 CUDA Out of Memory
```python
torch.cuda.OutOfMemoryError
```
**Çözüm:** Batch size'ı azalt veya `torch.cuda.empty_cache()` çağır.

### 12.3 Weight Loading Hatası
```python
KeyError: 'conv1d_1' not found
```
**Çözüm:** Keras model ve PyTorch model katman sayıları eşleşmeli. `strict=False` kullanabilirsin.

### 12.4 NaN Loss
```python
Loss: nan
```
**Çözüm:** Learning rate'i düşür (1e-4), gradient clipping ekle.

---

## 13. Referanslar

1. **Orijinal DeepFilter Paper:**
   Romero, F. P., et al. (2021). DeepFilter: an ECG baseline wander removal filter using deep learning techniques.

2. **FCN-DAE Paper:**
   Chiang, H. T., et al. (2019). Noise reduction in ECG signals using fully convolutional denoising autoencoders.

3. **DRNN Paper:**
   Antczak, K. (2018). Deep recurrent neural networks for ECG signal denoising.

4. **Dataset:**
   - QT Database: PhysioNet
   - MIT-BIH Noise Stress Test Database: PhysioNet

---

## 14. Lisans ve Teşekkür

Bu PyTorch implementasyonu, orijinal Keras projesinin lisansına tabidir.

Orijinal proje: https://github.com/fperdigon/DeepFilter
