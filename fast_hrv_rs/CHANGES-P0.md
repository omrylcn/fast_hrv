# fast-hrv: P0 doğruluk düzeltmeleri

Tarih: 2026-11-24 · Temel: `04a6da7` · Dal: çalışma kopyası (push yok)
Araçlar: Rust **1.100.0** (`rust-toolchain.toml`, kuruldu), MSRV 1.85 ayrıca test edildi (`cargo +1.85.0 test`).
Referanslar: NeuroKit2 0.3.0, SciPy 1.18.1, NumPy 2.5.5, pandas 3.1.1 (inceleme 0.2.12 ile yapılmıştı).

## Commit listesi

| Commit | Konu |
|---|---|
| `4f4ae8c` | chore: Rust 1.100.0 sabitlendi, `serde_json` dev-dependency'ye taşındı, `cargo fmt` |
| `70facd2` | fix 1: gerçek Butterworth, NeuroKit2 uyumlu temizleme, akış filtresi |
| `8fc4f31` | feat(sqi): diğer ajanın SQI modülü kendi commit'inde entegre edildi (`pub mod sqi;`) |
| `889fe00` | fix 2: Kubios (NeuroKit2 `signal_fixpeaks` ile birebir) |
| `36527af` | fix 3: gerçek NeuroKit2 parity testi (HRV, temizleme, R tepe, Kubios) |
| `cc7900e` | fix 4: boşluk farkında RR, sınırlı TINN, paniksiz girdi kontrolü, Python modül adı |
| `e7e7381` | fix 5: frekans alanı bant integrali, sabit RR, bant başına en az süre |
| `3fc704d` | akış = toplu entegrasyon testi, clippy temizliği, kullanılmayan rayon kaldırıldı |
| `b5907dc` | perf: SOS kaskadı örnek sıralı; let-chain kaldırıldı (MSRV 1.85) |
| `83f41cf` | docs: README doğruluğu, CHANGELOG v0.2.0, lisans dosyaları, `benches/` |

Her adımdan sonra: `cargo build` (varsayılan ve `--all-features`), `cargo test`, `cargo clippy --all-targets`.
Son durum: `cargo clippy --all-targets` varsayılan, `--no-default-features` ve `--features serde,parallel,simd`
için **uyarısız**. `--all-features` (python) yalnız PyO3 0.22 makrolarının 30 `unsafe_op_in_unsafe_fn`
uyarısını veriyor (PyO3 ≥ 0.23'e geçiş P1). Testler: 77 birim + 4 filtre + 4 parity + 6 sağlamlık + 3 akış
+ 21 SQI (+1 serde) + doctest'ler; hepsi geçiyor (1.100.0 ve 1.85.0).

AVX: `simd` varsayılan kapalı; `.cargo/config` / `target-cpu` yok; `simd.rs`'de intrinsic yok. Release
`libfast_hrv.so` içinde `objdump -d` ile **0** adet ymm/zmm/`vfmadd`/`vbroadcast` ve VEX (`vaddpd` vb.)
talimatı bulundu. Yeni bağımlılık yok (tersine `rayon` kaldırıldı).

---

## Fix 1: `ecg.rs` filtresi (yeni `src/filter.rs`)

**Ne değişti**
- `filter::butter_sos(order, cutoff, BandType, fs)`: analog prototip (buttap) → ön bükmeli (prewarp)
  lp/hp/bp/bs dönüşümü → bilineer dönüşüm → SciPy `zpk2sos(pairing="nearest")` ile aynı eşleme ve sırada
  ikinci derece bölümler. Karmaşık aritmetik NumPy semantiğiyle (Smith bölmesi, C99 csqrt) yazıldı.
- `iirnotch`, `sosfilt`, `sosfilt_zi`, `sosfiltfilt` (tek uzantı `odd`, `padlen = 3·ntaps`, birinci derece
  bölüm düzeltmesi, `zi·x0`), `filtfilt_ba` (SciPy `filtfilt(method="pad")`).
- `SosFilter`: durumunu çağrılar arasında taşıyan nedensel kaskat; `InitialState::SteadyState` (varsayılan,
  ilk örnekte `sosfilt_zi·x0`, DC geçici bozulması yok) veya `Zero`; `reset()` boşluktan sonra.
- (a) `ecg_clean` / `ecg_clean_neurokit(signal, fs, powerline)`: NeuroKit2 0.3.0 kaynağı okundu:
  `signal_filter(lowcut=0.5, method="butterworth", order=5)` → `butter(5, 0.5, "highpass", output="sos")`
  + `sosfiltfilt`; ardından `_signal_filter_powerline`: `b = ones(int(fs/50))` (fs < 100 ise `ones(2)`),
  `a = [len(b)]`, `filtfilt(b, a, method="pad")`. NaN'lar NK gibi ileri/geri dolduruluyor.
- (b) `EcgStreamFilter` / `EcgFilterConfig` / `ecg_clean_causal`: HP 0,5 Hz order 2 + LP 40 Hz order 4,
  isteğe bağlı çentik (`with_notch(50.0, Q)`, Q ayarlanabilir; varsayılan kapalı). `process(&mut self, packet)`,
  `process_into`, `process_inplace`.
- (c) `ecg_clean_zero_phase`: aynı bant, çevrimdışı, SciPy tarzı padding. Kısa girdide SciPy hata verir;
  burada padding `n − 1`'e kısaltılıyor (belgelendi).

**Doğrulama**
- 20 Butterworth tasarımı (alçak/yüksek/bant geçiren/bant durduran, order 1–8, fs 4–1000 Hz) ve `sosfilt_zi`,
  `scipy.signal.butter(..., output="sos")`'a karşı: **en büyük katsayı hatası 2,2e-16** (tolerans 1e-9).
  `iirnotch` 4 vaka < 1e-12.
- `sosfiltfilt` (HP5, BP2): en büyük bağıl hata 4,6e-13; `filtfilt` (10'luk ortalama) 2,4e-16; `sosfilt` < 1e-9.
- `ecg_clean` ↔ `nk.ecg_clean(method="neurokit")`, 512 Hz simüle EKG (temiz / gürültülü / şebeke+taban):
  **en büyük bağıl fark 2,7e-13** (inceleme: 50 Hz'in 7,7/20 birimi kalıyordu, tepelerin ~%30'u aynıydı).
- Akış: 32 örneklik (ve 1, 7, 33, 100, 512, 4096) paketlerle çıktı, tek çağrıyla **bit düzeyinde aynı**
  (`tests/streaming.rs`, birim testleri, doctest).
- Fixture: `tests/data/filter_reference.json` (73 KiB), üretici `tools/filter_reference.py`.

## Fix 2: Kubios (`peaks.rs`)

**Ne değişti**
- Long/short aday döngüsünden sonraki fazladan `i += 1` silindi; her aday `i`'yi tam bir artırıyor (NK gibi).
- Kayan çeyreklikler (91 atım) ve kayan medyan (11 atım): pandas `rolling(w, center=True, min_periods=1)`
  ile aynı: kenarda küçülen pencere, lineer enterpolasyon, çift sayıda medyan = iki ortanın ortalaması.
  Herkese açık `rolling_quantile_centered`, `rolling_median_centered`. 45 atımdan kısa dizilerde artık bozuk
  yansıtma yok.
- s12/s22: `np.pad(..., 2, "reflect")` indekslemesi ve NaN yayan max/min (`np.max` semantiği);
  `rr[0]`, `dRR[0]` NumPy ikili (pairwise) toplamıyla.
- `fix_peaks` artık **bütün geçişlerin birikimli** artefaktlarını döndürüyor (önce yalnız son geçiş).
- Yeni API: `detect_artifacts` (tek geçiş), `KubiosParams`, `fix_peaks_with` →
  `Result<KubiosOutput { peaks, artifacts, passes }>`; sırasız tepe burada hata, eski `fix_peaks` ise
  sıralayıp tekilleştiriyor (panik yok).
- "Missed beat" testi artık NK'nın gerçek çıktısını (`[100, 510, 1125, 1740, …]`, longshort [2]) kontrol ediyor.

**Doğrulama (NeuroKit2 0.3.0 `signal_fixpeaks(method="Kubios")` ve `_find_artifacts`)**
- Dev zamanı tarama: 30/40/70/140/350 atımlık gerçekçi RR, her uzunlukta missed/extra/ectopic/longshort/
  artefaktsız × 5 tohum = 125 dizi: **ilk geçiş tespiti 125/125, düzeltilmiş tepeler 125/125 birebir aynı**.
  (İnceleme, eski kod: ectopic 0/15, extra 6/15; yalnız `i += 1` silinince 13/15, 10/15.)
- Commit'lenen fixture (`tests/data/neurokit_kubios.json`, 26 dizi, çok geçişli "mixed" dahil): 26/26 ve 26/26.

## Fix 3: Parity testi

**Ne değişti**
- `tests/neurokit_parity.rs` referans dosyası yoksa **FAIL** ediyor; yalnız `FAST_HRV_SKIP_PARITY=1` ile açıkça
  atlanıyor (eksik dosya ile denendi: fail; env ile: geçiyor).
- `tools/neurokit_reference.py` yeniden yazıldı: her iki tarafa **aynı tamsayı R tepe indeksleri** + fs
  (ms'ye yuvarlama yok); frekans `hrv_frequency(normalize=False)` (ms²); LF/(LF+HF) NK'nın LF ve HF'sinden
  hesaplanıp fast-hrv `lfnu` ile karşılaştırılıyor (NK `LFn` = LF/TP, VHF dahil, farklı tanım). Sürümler
  her JSON'da.
- Metrik sınıfları (gerekçeler test dosyasında): **sıkı ≤ %0,1** (aynı tanım); pNN50/pNN20 NK paydasına
  `(n−1)/n` ile ölçeklenip sıkı; **gevşek %30** frekans (≥ 5 dk kayıtlar, yöntem farkı); **yalnız rapor**:
  VLF, < 5 dk kayıtların frekans metrikleri, TINN, DFA.
- Yeni testler: `ecg_clean_matches_neurokit`, `r_peaks_match_neurokit`, `kubios_matches_neurokit`.
  512 Hz `nk.ecg_simulate` + gürültü / 50 Hz + taban kayması; ham veri int16, çıktılar float64, base64
  (`neurokit_ecg.json` 120 KiB). HRV vakaları: 3 sentetik 5 dk (512 Hz), 1 sentetik 5 dk (1000 Hz),
  NK simüle 300/180/120/60 sn (23 KiB). Gerçek kişi verisi yok.

**Sonuç tablosu (son durum, fix 4–5 sonrası)**

| Metrik grubu | En kötü bağıl fark | Sınıf |
|---|---|---|
| MeanNN, SDNN, RMSSD, SDSD, Median, MAD, IQR, Min, Max, CVNN, CVSD | ≤ 6e-16 | sıkı |
| HTI | 0 (fix 4 ile 0 tabanlı histogram) | sıkı |
| SD1, SD2, SD1/SD2, CSI, CVI, GI, PI, SD1d/a, C1d/a | ≤ 3,3e-15 | sıkı |
| SampEn / ApEn | 2,8e-16 / 3,3e-14 | sıkı |
| pNN50 / pNN20 (payda ölçeklenmiş) | ≤ 2,4e-16 | sıkı |
| LF / HF / LF/HF / LFnu (≥ 5 dk) | %11,7 / %13,7 / %24,1 / %14,8 | gevşek %30 |
| Frekans, < 5 dk | LF/HF %86'ya kadar (180 sn vakası) | yalnız rapor |
| VLF | %198 | yalnız rapor |
| TINN | %93 | yalnız rapor |
| DFA α1 / α2 | %41 / %83 | yalnız rapor |
| `ecg_clean` | 2,7e-13 | sıkı (1e-9) |
| R tepe (3 × 20 sn) | 69/69 aynı örnek | sıkı |
| Kubios (26 dizi) | 26/26 tespit, 26/26 düzeltme | birebir |

## Fix 4: Boşluklar ve sağlamlık

- **Yeni `src/gaps.rs`**: `Gap { start, end }` (örnek; sıfır uzunluk = süreksizlik işareti),
  `split_at_gaps`, zaman damgalı `split_peak_times` (sn), `rri_segments`, `rri_segments_from_times`,
  `rri_with_gaps` → `Vec<RrSegment { start_s, rri_ms }>`, `fix_peaks_segmented`. Hiçbir RR boşluğun
  üzerinden geçmiyor, boşluk içindeki tepeler atılıyor, Kubios segment başına çalışıyor (testte 69 sn'lik
  boşluk: segmentsiz çağrı sahte tepe eklerken segmentli çağrı 0 artefakt, tepeler aynen). İsteğe bağlı
  `max_rr_s` kesimi var; varsayılan eşik **yok** (ürün kararı app-pipeline'da).
- **TINN**: histogram 0 tabanlı 7,8125 ms ızgarada (NK kutuları; HTI artık NK ile birebir). Arama ayrışabilir
  hale getirildi (E(N, M) = A(N) + B(M)): O(B³) → O(B²), sonuç kaba kuvvetle aynı (200 rastgele histogram).
  En fazla `MAX_TINN_BINS = 2048` kutu (16 sn); üstünde TINN = NaN, HTI bellek ayırmadan hesaplanıyor.
  **Tek 69 sn RR ile `hrv()`: 819 ms → ~2 ms.** Tepe = eşitlikte ilk kutu (NK `argmax`).
- **Panikler**: `compute_poincare(&[])`, `compute_asymmetry(&[])` → NaN; NaN/sonsuz içeren RR ile
  `time_domain`, `frequency_domain`, `hrv` → bütün metrikler NaN (`total_cmp`); `rri_from_peaks`
  sırasız/yinelenen çift veya fs = 0 → o aralık NaN, yeni `try_rri_from_peaks` → `HrvError`;
  `compute_quality` / `compute_stats` işaretli fark; `fix_peaks` sırasız girdi → sırala/tekilleştir;
  `ecg_clean(fs=0)` → NaN vektörü. `examples/panics.rs` artık hiç PANIC basmıyor; `tests/robustness.rs`.
- **Python**: `#[pymodule] fn fast_hrv_rs` → `fast_hrv` (`[lib] name` ile aynı, `PyInit_fast_hrv`;
  `import fast_hrv`). Bitişik olmayan diziler kopyalanıyor, negatif tepe indeksi `ValueError`.
  Not: `--features python` derleniyor, ama maturin ile gerçek bir import burada denenmedi.
- Yeni hata türü `HrvError` (`src/error.rs`).

## Fix 5: Frekans alanı

- Bant integrali kesin: yamuklar bant kenarında kırpılıyor, kenarda PSD lineer enterpole ediliyor; kenar
  yamuğu iki banda birden yazılmıyor (düz PSD'de eski hata VLF +%28, LF +%13,6, HF +%6,3; şimdi tam bant
  genişliği, VLF+LF+HF = 0,0033–0,4 Hz integrali; birim testi).
- Sabit RR: `1e-12 · mean(RR)²` altındaki bant gücü sayısal gürültü → 0; LF/HF, LFnu, HFnu, tepe frekansları
  NaN (eskiden LF = 1e-29 iken LF/HF = 0,165).
- Bant başına en az süre (RR toplamı): HF ≥ 60 sn, LF ≥ 120 sn, VLF ≥ 300 sn → altında o bandın gücü,
  oranları, tepesi ve `total_power` NaN. `FrequencyOptions` / `frequency_domain_with` ile ayarlanabilir.

## Ek işler (lead'in devir dokümanı §4–8)

- `serde_json` yalnız dev-dependency; `rayon` kaldırıldı (`parallel` etkisiz, uyumluluk için duruyor).
- `cargo fmt`, `cargo clippy --all-targets`; özellik kombinasyonları `--no-default-features` ve
  `--features serde,parallel,simd`: derleme, test ve clippy temiz.
- Akış = toplu testi: `tests/streaming.rs` (120 sn sentetik EKG, 5 paket boyu, çentikli/çentiksiz; aynı
  R tepeleri; `reset()` sonrası taze filtreyle aynı; DC başlangıç bozulması yok).
- README: yanlış iddialar ("real recordings", olmayan `benches/`) kaldırıldı; gerçekten test edilenler,
  sürümler ve ölçülen parity tablosu yazıldı; metrik sayısı 39 → 41; SQI modülü eklendi.
- `benches/throughput.rs` (yalnız std, `cargo bench --bench throughput`), CHANGELOG.md (v0.2.0),
  LICENSE-MIT, LICENSE-APACHE.
- SQI modülü kendi commit'inde (`feat(sqi): ...`), README ve CHANGELOG'da anıldı.

Hız (i7-9700K, 5 dk @ 512 Hz, 21 koşunun medyanı): `ecg_clean` ~6,5 ms (eski yaklaşık filtre 3,2 ms; order-5
HP + powerline adımı nedeniyle), `EcgStreamFilter` 32'lik paketler + çentik ~1,0 ms, `find_peaks_default`
~1,0 ms, Kubios ~0,2 ms, `hrv()` ~2,0 ms.

## Kırılan API / davranış değişiklikleri

1. `ecg_clean` çıktısı değişti (artık NK uyumlu HP5 + 50 Hz hareketli ortalama). Geçersiz fs → NaN vektörü.
2. `design_butterworth_bandpass(low, high, fs, order)`: gerçek Butterworth, SciPy order anlamı (bant geçiren:
   `order` bölüm); geçersiz parametre → boş vektör.
3. `fix_peaks` / `fix_peaks_default`: `Artifacts` birikimli (indeksler, bulunduğu geçişin tepe dizisine göre);
   sırasız girdi sıralanıyor. `Artifacts` artık `PartialEq, Eq` (+ serde ile `Serialize`).
4. `frequency_domain`: bant integrali, en az süreler (60/120/300 sn), sabit RR → NaN oranlar. 60 sn'den kısa
   kayıtta her şey NaN (eskiden 30 sn).
5. `time_domain`: HTI/TINN 0 tabanlı histogram; TINN eşitlikte ilk tepe; > 2048 kutu → NaN; NaN girdi → NaN.
6. `rri_from_peaks`: geçersiz çift → NaN (eskiden taşma/panik).
7. Python modülü `fast_hrv` (eski `fast_hrv_rs`); `compute_quality` artık `PyResult`.
8. `ecg::SosSection` artık `filter::SosSection`'ın yeniden ihracı (alanlar aynı, `Copy + PartialEq` eklendi).
9. `parallel` özelliği etkisiz; `serde_json` artık normal bağımlılık değil.

## Bilinen kalan farklar / açık işler

- **R tepe detektörü** NK'dan küçük farklarla duruyor (incelemedeki P2 prominence seçimi, P3 tamsayı
  `min_len`, P4 ilk 0,3 sn tepeleri). Sentetik fixture'da fark çıkmadı; 60 sn'lik ayrı bir denemede fast-hrv
  ilk 0,3 sn'de NK'nın attığı bir tepeyi fazladan buldu. **Akan tepe detektörü yok**: `min_len` bütün
  girdinin ortalamasından (P6). Canlı = geçmiş garantisi şu an filtre düzeyinde; tepe tespiti aynı filtre
  çıktısının aynı blok sınırlarıyla işlenmesini gerektiriyor.
- Kubios `update_indices`: bitişik extra indekslerinde fast-hrv doğru olanı yapıyor, NK farklı (bilinçli).
- **Frekans alanı yöntem farkı** (4 Hz kübik spline + Welch 256 vs NK 100 Hz quadratic + Welch N/2, nfft 2N,
  4/T altı kesim): ≥ 5 dk'da %25'e kadar, kısa kayıtlarda çok daha fazla. "NK uyumlu" Welch profili yok (P1).
  LFnu tanımı NK `LFn`'den farklı (LF/(LF+HF) vs LF/TP).
- DFA tanımı farklı (ölçekler, örtüşme, F(n)): P1. TINN araması NK'dan farklı (NK'nın M döngüsü her N için
  sıfırlanmıyor). pNN paydası farklı (belgelendi, testte ölçekleniyor).
- Boşluklu kayıtta metrikler segment başına hesaplanmalı; segment RR'lerini birleştirip `hrv()` çağırmak
  boşluk üzerinden ardışık fark üretir (belgelendi, birleşik `hrv_segments` yok).
- `compute_quality` göreli (averageQRS) ve `process_ecg` kaliteyi düzeltilmiş tepelerle hesaplıyor (E7/E8,
  bu P0 kapsamında değil; mutlak SQI'ler `sqi` modülünde).
- PyO3 0.22 + edition 2024 uyarıları; python bağlaması gerçek bir Python import'uyla denenmedi.
- SampEn `inf` / −0 semantiği değişmedi (P2).
