# SQI modülü (`src/sqi.rs`): özet

Mutlak EKG sinyal kalitesi indeksleri. Tek kanal, 512 Hz, 10 sn alt pencere için tasarlandı ama `fs` ve
pencere uzunluğu serbest. Modül yalnız **hesaplar**: kapı eşikleri ve son karar app-pipeline'da kalır.
Sınıflandırma yardımcıları eşikleri parametre olarak alır, varsayılanlar literatür değerleridir.

- Yeni bağımlılık yok. FFT modülün içinde (karışık taban 2/3/4/5, büyük asal çarpanlar için Bluestein).
  SIMD/AVX varsayımı yok. MSRV 1.85 korunuyor (`is_multiple_of` bilinçli olarak kullanılmadı).
- Panik yok: bütün fonksiyonlar `Result<_, SqiError>` döndürür (boş girdi, NaN, az atım, geçersiz parametre).
- `serde` özelliği açıkken bütün sonuç yapıları `Serialize` türetir.
- Durumsuz: aynı pencere her zaman aynı sonucu verir (canlı = geçmiş). `PsdEngine`/`SqiEngine` yalnız FFT
  planını ve Hann penceresini önbelleğe alır, veri tutmaz.

## API

| Fonksiyon | Ne döndürür | Varsayılan / kaynak |
|---|---|---|
| `flatline_fraction(raw, fs, eps, min_duration_s)` | düz hat örneklerinin oranı (ham ADC) | `eps = 0` sayım, `0.2 s` (sezgisel, cihaza göre ayarlanmalı) |
| `saturation_fraction(raw, adc_min, adc_max, margin)` | raylara yakın örneklerin oranı | `margin = 0` |
| `ksqi` = `kurtosis_pearson` | Pearson basıklığı `m4/m2²` | `scipy.stats.kurtosis(fisher=False)`, NeuroKit2 zhao2018 |
| `kurtosis_fisher` | Pearson − 3 (ayrı fonksiyon, kSQI ölçeği değil) | SciPy varsayılanı |
| `psd`, `PsdEngine::psd` | Welch PSD (`Psd { freqs, power, ... }`) | aşağıya bakın |
| `psqi_from_psd`, `bassqi_from_psd`, `spectral_sqi` | P(5–15)/P(5–40); 1 − P(0–1)/P(0–40) | Zhao & Zhang 2018, tek PSD paylaşılır |
| `bsqi(a, b, fs, tol)`, `bsqi_detail` | eşleşen / (Na + Nb − eşleşen) | `tol = 0.15 s` (Li, Mark & Clifford 2008; Clifford 2012) |
| `beat_template`, `template_correlation`, `template_correlation_with` | atım başına Pearson r, medyan/ortalama/min, şablon | pencere −0.2 … +0.4 s (BioSPPy), `min_beats = 3`; harici (kişiye özel) şablon desteklenir |
| `rr_plausibility(peaks, fs, signal_len, limits)` | ort. KH, min/maks RR, RRmax/RRmin, en uzun boşluk + 3 veto | 40–180 bpm, boşluk ≤ 3 s, oran < 2.2 (Orphanidou 2015) |
| `zhao2018_fusion(k, p, bas, hr, th)` | Excellent / Barely acceptable / Unacceptable + endeks notları + nedenler | NeuroKit2 basit kural eşikleri |
| `zhao_heart_rate_bpm(peaks, fs)` | NeuroKit2'nin kullandığı KH: **en kısa RR'den** (maks. anlık KH) | NeuroKit2 0.3.0 |
| `SqiEngine::report` / `sqi_report` | `SqiReport`: hepsi bir arada, `issues` listesinde hesaplanamayanların nedeni | `SqiConfig::default()` |

### PSD seçenekleri (`PsdConfig`)
- `PsdConfig::default()` = `Welch { nperseg: 2048 }`: `scipy.signal.welch` varsayılanları (periyodik Hann,
  %50 örtüşme, segment başına `detrend='constant'`, `scaling='density'`, `nfft = nperseg`). 10 sn'de 4 segment,
  0.25 Hz çözünürlük. `PsdConfig::welch(1024)` de var.
- `PsdConfig::neurokit()` = NeuroKit2 zhao2018'in **gerçekte** yaptığı. Önemli bulgu: NeuroKit2'deki
  `window=1024` **saniye** cinsinden, örnek değil. `nperseg = int(1024·fs)` her zaman `N/2`'ye kırpılıyor
  (10 sn @ 512 Hz → 2560), `nfft = 2·nperseg` (sıfır doldurma), tüm pencere bir kez ortalamadan arındırılıyor
  (segment detrend yok) ve `2·fs/(N/2)` altındaki frekanslar atılıyor (10 sn için 0.4 Hz). Bu yüzden basSQI'nin
  "0–1 Hz" bandı NeuroKit'te fiilen 0.4–1 Hz. Yani "NeuroKit uyumlu 1024 örnek" diye bir ayar yok; uyumlu mod budur.
- Bant gücü her iki modda NeuroKit2 gibi hesaplanır: `lo ≤ f < hi` kutuları, yamuk kuralı, güç 0 ise hata (NK: NaN).

### Zhao 2018 füzyonu (varsayılan eşikler, NeuroKit2 0.3.0)
- KH < 130: pSQI optimal (0.5, 0.8), şüpheli (0.4, 0.5). KH ≥ 130: optimal (0.4, 0.7), şüpheli (0.3, 0.4).
  Eşitsizlikler NK'deki gibi katı (tam 0.5 → uygunsuz).
- kSQI > 5 → optimal, değilse uygunsuz. basSQI ≥ 0.95 optimal, < 0.90 uygunsuz, arası şüpheli.
- ≥2 uygunsuz veya 1 uygunsuz + 2 şüpheli → Unacceptable; ≥2 optimal ve uygunsuz yok → Excellent; kalanı Barely acceptable.
- Fark: NaN endeks bizde "uygunsuz" sayılır (NK'de NaN basSQI "şüpheli" çıkıyordu, bilinçli olarak daha sıkı).

## Doğrulama (NeuroKit2 0.3.0, SciPy 1.18.1, NumPy 2.5.5)

`tools/sqi_reference.py` 9 sentetik 10 sn / 512 Hz pencere üretir: clean, noisy, baseline_wander, flatline
(4 sn kopma), saturated (12 bit kırpma), mains_50hz, tachy_150, clean_filtered, noisy_filtered. Hepsi
`nk.ecg_simulate` ve programla eklenen bozulmalar, gerçek veri yok. Sürümler her JSON'a yazılıyor.

| Karşılaştırma | Referans | En büyük fark (9 pencere) |
|---|---|---|
| kSQI (Pearson) | `scipy.stats.kurtosis(fisher=False)` ve NK `_ecg_quality_kSQI` | bağıl 4e-14 |
| pSQI / basSQI, `PsdConfig::neurokit()` | NK `_ecg_quality_pSQI/_basSQI` | mutlak 1.7e-15 / 7.8e-16 |
| pSQI / basSQI, Welch 2048 ve 1024 | `scipy.signal.welch` + NK bant integrali | < 1e-9 (test toleransı), tabloda aynı |
| zhao2018 sınıfı | `nk.ecg_quality(method="zhao2018", approach="simple")` | 9/9 aynı etiket |
| Şablon korelasyonu | NumPy (`np.corrcoef`, ortalama şablon) | < 1e-9 |
| bSQI, düz hat, satürasyon | Python referans uygulaması | tam eşit |

Testlerde ayrıca FFT'nin naif DFT ile karşılaştırması (21 farklı uzunluk, asal sayılar dahil), bSQI ve şablon
birim testleri, RR vetoları ve kenar durumları (boş, tek atım, NaN, sabit sinyal, saçma tepe indeksleri) var.
`cargo test --test sqi`: 21 test (`--features serde` ile 22) geçiyor. `clippy --all-targets`'ta sqi dosyalarına ait uyarı yok.

**Maliyet:** `SqiEngine::report` (bütün endeksler, Welch 2048, release) yaklaşık **170 µs/pencere**, yani tek
çekirdekte ~5900 pencere/sn. Hedef olan 50 pencere/sn'nin 100 katından fazla.

## Dikkat edilecekler
- Temiz sentetik EKG'de pSQI ≈ 0.90 çıkıyor ve NK eşiği 0.8 olduğu için "Barely acceptable" sınıfı alıyor.
  pSQI QRS morfolojisine ve örnekleme hızına duyarlı. Ürün eşikleri kendi verimizle kalibre edilmeli.
- kSQI düz hattı yakalamaz (flatline penceresinde kSQI 11.4 çıktı). Ham akış kontrolleri bu yüzden ayrı.
- Ham akış kontrolleri filtrelemeden **önce** ADC sayımlarında çalışmalı.
- Welch 2048 ile NK modu basSQI'de belirgin farklı sonuç verir (ör. clean: 0.956 vs 0.992) çünkü NK 0.4 Hz
  altını atıyor. Eşikleri hangi modla belirlediyseniz o modu kullanın.

## Dosyalar
- `src/sqi.rs` (yeni), `tests/sqi.rs`, `tests/fixtures/sqi/*.json` (9 dosya, ~370 KB), `tools/sqi_reference.py`
- `lib.rs`'e eklenecek satır: `pub mod sqi;` (bu dosyaya dokunulmadı)
