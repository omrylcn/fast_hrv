# fast-hrv: detektör iyileştirmeleri (Elgendi, tepe inceltme, iki detektör, pencereli işleme)

Tarih: 2026-11-24 · Temel: `cd5d8ff` (v0.2.0) · Dal: `master` (push yok) · Sürüm: 0.3.0 (yayımlanmadı)
Araçlar: Rust **1.100.0** (`rust-toolchain.toml`), MSRV 1.85 ayrıca test edildi (`cargo +1.85.0 test`).
Referans: NeuroKit2 0.3.0, SciPy 1.18.1, NumPy 2.5.5.

Neden: gerçek giyilebilir veride (RW1, 512 Hz, `local analysis notes` §4) dört bulgu çıktı.
NeuroKit2 `elgendi2010` tepeleri R'den sabit 72–86 ms geç veriyor (inceltme olmadan @50 ms F1 0,02–0,13).
`hamilton2002` uzun kayıtta kilitleniyor. EngZee en tutarsız ikinci detektör (Elg–EngZ F1 0,78).
R zamanlama jitter'ı RMSSD'yi şişiriyor (σ = 12 ms, gerçek 20 ms'yi ~37 ms yapıyor).

## Commit listesi

| Commit | Konu |
|---|---|
| `f887239` | feat(detect): Elgendi iki kayan ortalamalı QRS detektörü + NeuroKit2 parity |
| `fb094c1` | feat(detect): alt-örnek R tepe inceltme; f64 zamanlardan RR |
| `b987c43` | feat(detect): iki detektör API'si (Elgendi + inceltme ile NeuroKit2 yöntemi) + bSQI |
| `898ca05` | feat(detect): uzun kayıt için mutlak örnek ızgarasına bağlı pencereli işleme |
| `c87ec08` | perf(detect): kayan ortalamada doğrudan indeksleme; benchmark'a detektör süreleri |
| (bu commit) | docs: CHANGES-DETECT.md, README, CHANGELOG 0.3.0, sürüm 0.3.0 |

Her adımdan sonra: `cargo fmt`, `cargo clippy --all-targets` (varsayılan, `--no-default-features`,
`--features serde,parallel,simd`: **uyarısız**), `cargo test`. Son durum: 82 birim + 10 detektör + 4 filtre
+ 4 parity + 6 sağlamlık + 21 SQI + 3 akış + 4 doctest; hepsi geçiyor (1.100.0 ve 1.85.0).
Yeni bağımlılık yok, AVX varsayımı yok (yalnız skaler `f64`), yeni `unsafe` yok.

## API (`src/detect.rs`, kök modülden de yeniden ihraç)

| Öğe | Ne yapar |
|---|---|
| `elgendi_peaks(signal, fs, &ElgendiConfig)` | 8–20 Hz bant geçiren + Elgendi tespiti, tamsayı tepe indeksleri |
| `elgendi_filter`, `elgendi_detect` | iki adım ayrı ayrı (hazır filtrelenmiş sinyalde tespit) |
| `ElgendiConfig::default()` / `neurokit2013()` / `neurokit2010()` | aşağıdaki tablo |
| `moving_average_causal(x, w)` | NeuroKit2 `_ecg_findpeaks_MWA` ile bit düzeyinde aynı kayan ortalama |
| `refine_peaks(clean, fs, &candidates, &RefineConfig)` → `RefinedPeaks { indices, positions, sign }` | ±pencerede yerel uç değer + 3 noktalı parabolik enterpolasyon; `times_s(fs)` |
| `Polarity::{Auto, Positive, Negative, Absolute}` | `Auto`: çağrı (= pencere) başına tek işaret, her aday penceresindeki en büyük `|x|`'in işaretiyle çoğunluk oyu (eşitlikte pozitif). `Absolute`: atım başına `|x|` maksimumu |
| `parabolic_offset(y₋₁, y₀, y₊₁)`, `auto_polarity(...)` | yardımcılar |
| `rri_from_positions(positions, fs)`, `rri_from_times(times_s)`, `try_rri_from_times` | alt-örnek tepelerden RR (ms); tamsayı `rri_from_peaks` aynen duruyor |
| `dual_detect(clean, fs, &DualConfig)` → `DualDetection` | birincil (Elgendi + inceltme) ve ikincil (NeuroKit2 yöntemi, `find_peaks_default`) tepeler, ham halleri ve `bsqi: Option<sqi::Bsqi>`. Yalnız sonuç, kapı yok |
| `detect_windowed(signal, fs, start_sample, &WindowConfig, detect)` | genel pencereli çalıştırıcı (herhangi bir detektör kapanışı) |
| `elgendi_refined_windowed(...)` | pencere başına Elgendi + inceltme |
| `HrvError::InvalidParameter(&'static str)` | yeni hata türü (kırıcı: kapsamlı `match` güncellenmeli) |

### Varsayılanlar ve kaynakları

| Parametre | Varsayılan | Kaynak |
|---|---|---|
| Detektör bandı | 8–20 Hz, Butterworth order 2 (SciPy anlamı, 4 kutup), nedensel, başlangıç durumu `sosfilt_zi · mean(x)` | Elgendi ve ark. 2010; NeuroKit2 `_ecg_clean_elgendi` (`butterworth_zi`) |
| Enerji | `x²` | Elgendi 2013 (PLoS ONE 8(9):e73557) |
| QRS / atım kayan ortalaması | 0,097 s / 0,611 s (`round(t·fs)`, Python gibi yarıda çifte) | Elgendi 2013; NeuroKit2 `elgendi2013` |
| Eşik ofseti | `β · mean(x²)`, β = 0,08 | Elgendi 2013 |
| En küçük blok genişliği | QRS penceresi (512 Hz'te 50 örnek) | Elgendi 2013 |
| Blok içi tepe | en büyük `|x|` | NeuroKit2 `elgendi2013` |
| Refrakter süre | 0,3 s (öncekini tutar) | NeuroKit2 `mindelay` (200 bpm); literatür raporu ≥ 250–300 ms |
| İnceltme penceresi | ±0,05 s | literatür raporu (R-tepe inceltme satırı, jitter gerekçesi Porr & Macfarlane) |
| İnceltme polaritesi | `Auto` | RW1 analizinin `polarity()` yaklaşımı, pencereye indirildi |
| Parabolik enterpolasyon | açık, δ = 0,5·(y₋₁ − y₊₁)/(y₋₁ − 2y₀ + y₊₁), δ ∈ [−0,5, 0,5] | literatür raporu |
| bSQI toleransı | 0,15 s; 0,05 s varyantı belgelendi (`refine_secondary = true` gerekir) | Li, Mark & Clifford 2008 / Clifford 2012; NeuroKit2 `ecg_quality(method="ici")` |
| İkincil detektör | NeuroKit2 varsayılan yöntemi, birincilin işaretiyle inceltilmiş | RW1 analizi §8.3 (NK–Elg 0,86 > Elg–EngZ 0,78) |
| Pencere / pay / tekilleştirme | 10 s / her yanda 2 s / 0,15 s | 10 s: Clifford 2012 SQI penceresi; 2 s: RW1 analizindeki pay; 0,15 s: bu iş için seçildi (fizyolojik RR ve 0,3 s refrakterin altında) |

### NeuroKit2'den sapmalar (bilinçli, belgeli)

1. **Görev tanımındaki parametreler NeuroKit2 `elgendi2010` değil, `elgendi2013`.** NeuroKit2 0.3.0'da iki
   uygulama var. `elgendi2010` (py-ecg-detectors portu): `|x|`, 120 ms / 600 ms pencere (`int(t·fs)`), ofset
   yok, blok genişliği `> int(0,08·fs)`, blok başı = eşik altındaki son örnek, blok sonu = eşik üstündeki son
   örnek − 1, işaretli `argmax`, 0,3 s refrakter. `elgendi2013`: `x²`, 97/611 ms, β = 0,08·ortalama, en küçük
   blok = QRS penceresi, `argmax |x|`, refrakter yok. İkisi de birebir uygulandı (`neurokit2010()`,
   `neurokit2013()`); varsayılan = `elgendi2013` + 0,3 s refrakter.
2. Refrakter yalnız varsayılanda (NeuroKit2 `elgendi2013`'te yok). Sentetik fixture'larda hiç tetiklenmedi.
3. `DetectorFilter::ZeroPhase` seçeneği (NeuroKit2'de yok). Varsayılan nedensel (NeuroKit2 ile aynı).
4. NaN girdi: NeuroKit2 doldurur, burada `HrvError::NonFinite`. Örnek 0'da açık blok (β ≤ 0 ile mümkün)
   NeuroKit2'de hizayı bozardı, burada doğru işleniyor (β > 0'da fark yok).

## Parity (NeuroKit2 0.3.0, inceltmeden önce)

`tools/detect_reference.py` → `tests/data/neurokit_elgendi.json` (227 KiB): 5 sentetik sinyal, her biri
30 s / 512 Hz `nk.ecg_simulate` + programla eklenen bozulma (temiz; gürültülü; şebeke 50 Hz + taban kayması;
140 bpm; ters polarite + düşük genlik). Ham veri int16, gerçek kişi verisi yok.

| Karşılaştırma | Sonuç |
|---|---|
| 8–20 Hz filtre ↔ `nk.ecg_clean(method="elgendi2010")` | en büyük fark / max\|y\| = **6,4e-14** |
| `ElgendiConfig::neurokit2010()` ↔ `ecg_findpeaks(method="elgendi2010")` | **201/201** tepe aynı örnek |
| `ElgendiConfig::neurokit2013()` ↔ `ecg_findpeaks(method="elgendi2013")` | **202/202** aynı |
| `ElgendiConfig::default()` ↔ `elgendi2013` | 202/202 aynı (refrakter tetiklenmedi) |

Kayan ortalama: SciPy `uniform_filter1d` toplamı tutup her çıktıyı pencereye bölüyor (ortalamayı değil); bu
sıra korunarak bit düzeyinde aynı sonuç alındı.

## İnceltme doğruluğu (sentetik, bilinen R)

Test sinyali: P, Q, R, S, T Gauss toplamı, solunum + yavaş RR modülasyonu, taban kayması, beyaz gürültü,
R konumları tamsayı değil ve kesin biliniyor. Temiz yol `ecg_clean_zero_phase` (0,5–40 Hz).

| Durum | Ham Elgendi gecikmesi (medyan) | İnceltilmiş hata medyan / maks |
|---|---|---|
| gürültüsüz | 19,2 örnek (37,5 ms) | 0,005 / 0,009 örnek |
| gürültü 0,02 | 37,6 ms | 0,032 / 0,139 örnek |
| gürültü 0,02, ters derivasyon (`Auto` → −1) | 37,4 ms | 0,036 / 0,177 örnek |
| gürültü 0,05 | 37,4 ms | 0,100 / 0,374 örnek |

- Sabit kaydırma testi: adaylar gerçek R'nin −20…+25 örnek ötesine konuldu; `Positive`, `Auto`, `Absolute` ve
  ters sinyalde `Negative`, `Auto`, `Absolute` için **her tepe 1 örnek içinde**, medyan hata < 0,2 örnek.
- 5 dk sentetik kayıtta RMSSD: gerçek 41,250 ms, alt-örnek 41,248 ms, tamsayı 41,260 ms.
- Ham varsayılan detektörün gecikmesi ~37 ms olduğundan ±50 ms pencere yetiyor. NeuroKit2 `elgendi2010`
  ön ayarı ~80 ms geç; onunla pencere ≥ 0,09 s olmalı (aşağıda gerçek veri).

## Pencereli işleme

- Çekirdek pencereler `[k·W, (k+1)·W)` mutlak örnek ekseninde (W = 10 s), detektör her pencerede
  `[k·W − M, (k+1)·W + M)` parçasında sıfırdan çalışıyor (M = 2 s). Çekirdeğin 0,15 s yakınındaki tepeler
  tutuluyor. Komşu pencerelerin aynı atımı: kendi çekirdeğinin daha içinde olan kalıyor, eşitlikte önceki pencere.
- Sonuç, her pencerenin çevresindeki örneklerin saf fonksiyonu. Test: 3 dk'lık kaydı 5 farklı başlangıç
  örneğinden (1, 777, 5120, 12 345, 30 001) işleyince ortak parçalarda çıktı **bit düzeyinde aynı**.
- 30 dk sentetik kayıt (7:00'de 20 s güçlü gürültü, 15:00'te 3 s elektrot atlaması): artefakta dokunmayan
  her dakikada atım/dk gerçeğe **±1** içinde (70–71). Bu pencerelerde 2050/2050 atım bulundu, en büyük hata
  0,30 örnek, örtüşmeden yinelenen tepe yok. Kayıp yalnız parçası artefakta değen pencerelerde (dakika 6, 7,
  14, 15). Sonraki pencere temiz başlıyor, kilitlenme yok.
- Not: β·ortalama eşiği parça üzerinden hesaplanıyor. Payda (2 s) büyük bir artefakt varsa o pencerenin
  atımları kaçabilir. Böyle pencereleri kalite kapısı zaten reddetmeli. Aynı eşik tüm kayıttan hesaplanırsa
  (tek geçiş) bir artefakt kaydın tamamını etkiliyor. Bunun gerçek veri karşılığı aşağıda.

## Gerçek veri (RW1, yerel, commit'lenmedi)

Veri yerinde okundu ve kopyalanmadı. Segmentler Rust ölçüm programına yalnız bellek içi pipe ile gitti.
Betik ve program repo dışında: `analysis/detect_eval.py`, `analysis/detect_probe/`. Aşağıdakiler yalnız
toplu sayılar. Hat: boşluksuz her segment (≥ 10 s) → `ecg_clean_zero_phase` (0,5–40 Hz) →
`elgendi_refined_windowed` (varsayılanlar). Referans: NeuroKit2 0.3.0 `neurokit` yöntemi, analizin path A
sinyalinde (tüm segment). Eşleşmeler analiz raporundaki gibi 10 s alt pencerelerde toplandı:
F1 = 2·Σeşleşme / (ΣN1 + ΣN2).

**F1 @50 ms, Elgendi-inceltilmiş (fast-hrv, pencereli) ile NeuroKit `neurokit`:**

| Oturum | Elgendi atım | NK atım | fast-hrv @50 | Rapor (NK–Elg iyileştirilmiş @50) | fast-hrv @150 | NK inceltilmiş ile @50 | fast-hrv `neurokit` ile @50 |
|---|---|---|---|---|---|---|---|
| S01 | 2673 | 2623 | **0,947** | 0,945 | 0,975 | 0,940 | 0,927 |
| S02 | 1711 | 1770 | **0,941** | 0,940 | 0,966 | 0,939 | 0,932 |
| S03 | 1004 | 1015 | **0,902** | 0,903 | 0,937 | 0,892 | 0,859 |
| S04 | 728 | 698 | **0,443** | 0,402 | 0,640 | 0,396 | 0,374 |
| S05 | 774 | 473 | **0,431** | 0,425 | 0,571 | 0,418 | 0,400 |

**İnceltmenin ve polarite seçiminin etkisi (@50 ms, NK `neurokit` ile):**

| Oturum | Varsayılan, ham | `neurokit2010` ham | Rapor ham (NK `elgendi2010`) | `neurokit2010` + ±50 ms | `neurokit2010` + ±90 ms | Varsayılan + `Absolute` | Tek geçiş (penceresiz) |
|---|---|---|---|---|---|---|---|
| S01 | 0,941 | 0,026 | 0,022 | 0,136 | 0,940 | 0,910 | 0,949 (2667 atım) |
| S02 | 0,940 | 0,057 | 0,052 | 0,714 | 0,933 | 0,930 | **0,740 (1131 atım)** |
| S03 | 0,896 | 0,138 | 0,131 | 0,340 | 0,882 | 0,871 | 0,879 (951 atım) |
| S04 | 0,367 | 0,165 | 0,155 | 0,339 | 0,387 | 0,454 | **0,252 (224 atım)** |
| S05 | 0,326 | 0,122 | 0,135 | 0,346 | 0,393 | 0,439 | **0,364 (198 atım)** |

- `neurokit2010` ön ayarı rapordaki gecikme sorununu aynen üretiyor (0,03–0,17). ±50 ms bu gecikmeyi
  kurtarmıyor, ±90 ms kurtarıyor. Varsayılan (2013 tipi) detektörün ham gecikmesi gerçek veride de sabit:
  medyan S01 38,4 / S02 36,3 / S03 37,0 ms, IQR ≤ 1,4 ms (S04/S05'te dağınık). Bu yüzden ±50 ms yetiyor.
- `Absolute` polarite iyi oturumlarda `Auto`'dan biraz düşük (S01 0,910'a karşı 0,947), kirli oturumlarda
  biraz yüksek. Atım başına `|x|` maksimumu zaman zaman derin S dalgasına atlıyor. Varsayılan `Auto`
  (pencere başına tek işaret) kaldı, `Absolute` seçenek olarak duruyor.
- **Tek geçiş (penceresiz, tüm segment) Elgendi de uzun kayıtta atım kaybediyor.** Pencereli işlemde
  S02 1711, S04 728, S05 774 atım bulunurken tek geçişte 1131, 224 ve 198 atım çıkıyor (%34–74 kayıp).
  Sebep: β·ortalama(x²) eşiği tüm segmentten hesaplanıyor ve artefaktlar bu ortalamayı şişiriyor.
  Hamilton'daki kilitlenmenin Elgendi'deki karşılığı bu. Pencereli işleme bunu gideriyor (S02 0,740 → 0,941).
  S01 temiz olduğu için orada fark yok (2667'ye karşı 2673).

**`dual_detect` bSQI (10 s pencere başına, paysız, Elgendi + inceltme ile fast-hrv `neurokit`, inceltilmiş):**

| Oturum | Pencere | Medyan @0,15 s | p10 | ≥ 0,9 | < 0,8 | Medyan @0,05 s | Rapor medyanı (Elg–EngZee @0,15) | Rapor < 0,8 |
|---|---|---|---|---|---|---|---|---|
| S01 | 180 | 0,97 | 0,82 | %76,7 | %8,3 | 0,94 | 1,00 | %20,0 |
| S02 | 106 | 1,00 | 0,74 | %70,8 | %13,2 | 0,94 | 0,94 | %27,4 |
| S03 | 85 | 0,92 | 0,52 | %60,0 | %25,9 | 0,86 | 0,92 | %32,9 |
| S04 | 91 | 0,43 | 0,17 | %4,4 | %95,6 | 0,25 | 0,08 | %97,8 |
| S05 | 66 | 0,38 | 0,20 | %10,6 | %87,9 | 0,22 | 0,26 | %95,5 |

NeuroKit yöntemi ikinci detektör olunca iyi oturumlarda < 0,8 oranı yaklaşık yarıya iniyor (S01 %20 → %8).
Kötü oturumlar (S04/S05) yine %88–96 oranında < 0,8. Eşik kararı app-pipeline'da.

**S01'de atım/dk kararlılığı (30 tam dakika, pencereli Elgendi + inceltme):** medyan 89, p5 84, p95 96,
min 67, maks 98, toplam 2673 atım. 30 atım/dk altında dakika **0**. Penceresiz tek geçiş: medyan 88,
min 75, maks 97. NK `neurokit`: medyan 88, min 56. Karşılaştırma için rapordaki Hamilton (NK portu, tek
segment): 1800 s'de 18 atım. S01 temiz olduğundan tek geçiş de burada sağlam. Artefaktlı oturumlarda tek
geçiş atım kaybediyor (yukarıda). Pencereli işleme her detektörü (kapanışla) aynı şekilde sarabiliyor.

## Hız (i7-9700K, 5 dk @ 512 Hz, 21 koşunun medyanı, `cargo bench --bench throughput`)

| Adım | Süre |
|---|---|
| `elgendi_peaks` (filtre + tespit) | ~1,8 ms |
| `refine_peaks` | ~0,15 ms |
| `elgendi_refined_windowed` (10 s + 2×2 s pay, ~%40 fazla örnek) | ~2,8 ms |
| `dual_detect`, 30 × 10 s pencere (iki detektör + bSQI) | ~3,2 ms |

`dual_detect` gerçek zamanın yaklaşık 1/94.000'i kadar sürüyor. 500 kullanıcı için bir çekirdeğin ~%0,5'i,
pencereli birincil detektörle birlikte ~%1.

## Bilinen sınırlar / açık işler

- Akan (örnek örnek) Elgendi yok. Canlı = geçmiş, aynı mutlak pencere ızgarasıyla blok bazında sağlanıyor
  (`detect_windowed`). Pencere ancak payı dolunca (W + M sonra) kesinleşiyor.
- β·ortalama parça bazında hesaplanıyor. Paydaki büyük artefakt o pencerenin atımlarını düşürebiliyor.
- İkincil detektör NeuroKit2 yöntemi; tamsayı `fs` istediği için `fs` yuvarlanıyor. EngZee-mod eklenmedi
  (gerçek veride en tutarsız eş çıktı).
- RW1 tek gönüllü, 5 oturum, referans işaretleme yok. F1, detektörler arası uyum, doğruluk değil.
  Doğrulama için GUDB / MIT-BIH (lisanslı açık veri) ile JF skoru sonraki adım.
