# fast-hrv: Kubios eşik tabanı, dışlama kipi ve maskeli RMSSD

Tarih: 2026-11-24 · Temel: `31117b4` (v0.5.0) · Dal: `master` (push yok) · Sürüm: 0.6.0 (yayımlanmadı)
Araçlar: Rust **1.100.0**, MSRV 1.85 ayrıca test edildi (`cargo +1.85.0 test`).

Neden: `review/12-ektopi-kapisi.md`. Kubios düzeltmesi telafili erken atımı (ve çoğu zaman bir komşusunu)
komşu ortasına taşıyor. NN serisinde 50–100 ms basamak kalıyor ve düşük HRV'de RMSSD katlanıyor. Raporun
önerisi: **interpolasyon yok, dışlama var.** Ektopik atımın iki yanındaki NN aralığı atılıyor, RMSSD yalnız
ardışık ve ikisi de tutulmuş NN çiftlerinden hesaplanıyor. Algılama iki aşamalı. Proje yöneticisinin onayıyla
genel algoritmalar fast-hrv'ye girdi. Ürün eşikleri üründe kalıyor. Kütüphane yalnız parametre veriyor, varsayılanlar
nötr ya da literatürden.

## Commit listesi

| Commit | Konu |
|---|---|
| `b6b29d1` | feat(kubios): `KubiosParams::threshold_floor_s` (Th1 / Th2 için mutlak taban) |
| `cc5cd4e` | feat(exclusion): dışlama kipi: atım sınıfları, iki aşamalı algılama, NN maskesi, artefakt oranı |
| `2d7bab5` | feat(time_domain): maskeli zaman alanı (RMSSD yalnız bitişik geçerli NN çiftlerinden) |
| (bu commit) | docs: bu not, README, CHANGELOG 0.6.0, sürüm 0.6.0 |

Her adımda `cargo fmt` ve `cargo clippy --all-targets -- -D warnings` çalıştırıldı (varsayılan,
`--no-default-features`, `--features serde,parallel,simd`: **uyarısız**), ardından `cargo test` ve
`cargo +1.85.0 test`, hepsi geçti. `cargo doc` `-D warnings` ile temiz, `--features python` derleniyor.
Yeni bağımlılık, AVX varsayımı ya da `unsafe` eklenmedi. Repoya gerçek veri girmedi: yeni fixture'lar
sentetik.

## API

| Öğe | Ne yapar |
|---|---|
| `KubiosParams::threshold_floor_s: f64` | varsayılan `0.0`. `> 0` ise `Th1 = max(Th1, taban)`, `Th2 = max(Th2, taban)` (saniye). `≤ 0` iken taban hiç uygulanmıyor, çıktı v0.5.0 ile bit düzeyinde aynı |
| `KubiosParams::alpha` | zaten vardı (varsayılan 5,2). Belgesi eklendi. Ürün 6,0 kullanacak |
| `exclusion::classify_beats(positions: &[f64], fs, &ExclusionParams, disagreement: Option<&[bool]>)` → `BeatExclusion` | tek boşluksuz kayıt, alt-örnek konumlar. Eklenen atım tam orta noktaya |
| `classify_peaks(peaks: &[usize], ...)` | tamsayı indeksler. Eklenen atım `prev + floor((next − prev)/2)`, bütün çıktı konumları tamsayı değerli |
| `classify_segments(&[Vec<f64>], ...)`, `concat_nn(&[BeatExclusion])` | parça başına sınıflama. Birleştirmede parçalar arasına bir `NaN` / geçersiz NN konuyor, hiçbir fark boşluğu aşmıyor |
| `ExclusionParams { kubios, premature_factor: 0.85, compensatory_factor: 0.5, two_stage: true, suspect_move_s: 0.030, exclude_inserted_removed: true, physiological_rr_s: None }` | parametreler (aşağıda) |
| `BeatExclusion { positions, beats: Vec<BeatFlags>, input_classes, input_to_output, removed, splits, nn_valid, nn_usable, sampling_rate, params }` | sonuç. Tutulan girdi atımları konumunu bit düzeyinde koruyor, **hiçbir atım taşınmıyor** |
| `BeatFlags { class, input_index, inserted, ectopic_kubios, ectopic_pattern, suspect_stage, suspect_move_s, disagreement }` | çıktı atımı başına. `is_ectopic()`, `is_suspect()`, `is_far_suspect(x)` |
| `BeatClass::{Normal, Ectopic, Suspect, Disagreement, Inserted, Missed, Extra, PhysiologicalLimit}` | çıktı atımları: Normal / Ectopic / Suspect / Disagreement / Inserted. Girdi atımları: Normal / Ectopic / Suspect / Disagreement / Missed / Extra / PhysiologicalLimit |
| `kept_count()`, `usable_count()`, `kept_share()` | geçerli NN, kullanılabilir NN (fizyolojik bölmeyi aşmayan), oran |
| `artifact_counts()` / `artifact_counts_with(x)` → `ArtifactCounts { inserted, removed, physiological, suspect_far, disagreement }`, `artifact_ratio()` | artefakt olayları ve olay / çıktı atımı |
| `ectopic_count()`, `ectopy_ratio()` | ektopik atım sayısı ve oranı (kayıt için, kapı değil) |
| `summary_between(t0, t1)` → `ExclusionSummary { beats, ectopic, kept_nn, usable_nn, artifacts }` | aynı sayımlar bir pencere için (`kept_share()`, `artifact_ratio()`, `ectopy_ratio()`) |
| `nn_ms()`, `time_domain()`, `rmssd()` | NN (ms), maskeli zaman alanı, maskeli RMSSD (`Option`) |
| `time_domain::time_domain_masked(nn_ms, valid)` → `MaskedTimeDomain { time, n_nn, n_pairs }` | maskeli zaman alanı (aşağıda) |
| `masked_differences(nn_ms, valid)`, `rmssd_masked(nn_ms, valid) -> Option<f64>`, `MIN_MASKED_PAIRS = 2` | yardımcılar |

`fix_peaks*`, `fix_positions_with*`, `detect_artifacts*` imzaları ve sonuçları aynı. İçeride `fix_core`'a
"yalnız artefakt" kipi (ektopik ve uzun/kısa bayrakları sayılmadan ve düzeltilmeden atılıyor) ve medRR'yi de
döndüren `detect_detail` eklendi. Düzeltme yolu bu kipleri kullanmıyor.

## Tanımlar

Notasyon: çıktı konumları `p[k]`, `RR[k] = p[k] − p[k−1]` (k. atımda biten aralık), `m[k]` = Kubios'un
ortalanmış 11 atımlık RR medyanı (`medRR`).

**Eşik tabanı.** Lipponen–Tarvainen'de `Th = alpha · (Q3 − Q1)/2`: Th1 `|dRR|` için, Th2 `|mRR|` için,
91 atımlık kayan pencere. Taban ikisine de uygulanıyor. Th2 aynı zamanda kaçırılmış/fazla sınıflamasında
tolerans (`|RR/2 − medRR| < Th2`, `|RR[j] + RR[j+1] − medRR| < Th2`), taban orada da toleransı genişletiyor.
NaN eşik NaN kalıyor (`np.maximum` gibi).

**Dışlama kipi** (`classify_beats`):

0. *(isteğe bağlı, `physiological_rr_s = Some((min, max))`, varsayılan yok)* son tutulan atıma `min`'den yakın
   atım düşürülüyor (`PhysiologicalLimit`). `max`'tan uzun aralık seriyi bölüyor: her parça ayrı işleniyor,
   o aralık ne kullanılabilir ne geçerli. Her düşürme ve her bölme bir fizyolojik olay.
1. **Artefakt geçişleri:** Kubios yalnız kaçırılmış (orta noktaya atım ekle) ve fazla (atımı çıkar)
   düzeltmeleriyle çalışıyor, bu bayrakların sayısı azaldıkça tekrarlanıyor (`kubios.iterative`). Ektopik ve
   uzun/kısa bayrakları hiçbir atımı taşımıyor.
2. **Tek algılama geçişi.** Atım `b` **ektopik**:
   (a) Kubios `j`'de `ectopic` bayrağı verdiyse, `{j−1, j}` içinden iki komşu RR'si en çok farklı olan atım
   (`|RR[b+1] − RR[b]|` en büyük, eşitlikte `j−1`), yani kısa ile uzun aralığın arasındaki atım; **veya**
   (b) erken + telafi örüntüsü: `RR[b] ≤ premature_factor · m[b]` **ve**
   `RR[b+1] − m[b] ≥ compensatory_factor · (m[b] − RR[b])` (varsayılan 0,85 ve 0,5; `premature_factor = 0`
   örüntüyü kapatıyor). Ektopik atımın `b` ya da `b+1` indeksindeki uzun/kısa (ya da kalan kaçırılmış/fazla)
   bayrağı o olaya ait. Diğer her bayrak atımını **şüpheli** yapıyor (aşama 1).
3. **İki aşamalı** (varsayılan açık): ektopik atım varsa algılama, ektopik atımların komşu ortasına konduğu
   bir **kopyada** tekrar çalışıyor. Kopya yalnız algılama için: eşikler ektopik dRR'den arınıyor. Ektopik
   atımların ±1 atım dışındaki bayraklar şüpheli oluyor (aşama 2). Çıktıda hiçbir atım taşınmıyor.
4. **NN geçerliliği:** `k` ile `k+1` atımları arasındaki aralık geçersiz, eğer (i) atımlardan biri ektopik,
   şüpheli ya da çağıranın verdiği uyuşmazlık atımıysa (yani bu atımların **hem önceki hem sonraki**
   aralığı), (ii) fizyolojik bölmeyi aşıyorsa, (iii) `exclude_inserted_removed` açıkken (varsayılan)
   atımlardan biri eklenmişse ya da aralığın içinde çıkarılmış bir atım varsa. Boşluk: her boşluksuz parça
   ayrı işleniyor (`classify_segments`, `concat_nn`).

**Şüphelinin taşınma mesafesi** (`suspect_move_s`): Kubios'un onu komşu ortasına taşıması için gereken
mesafe, `|p[j] − (p[j−1] + p[j+1])/2| / fs` (aşama 2'de kopya üzerinde). Kalan kaçırılmış/fazla bayrağında
ve kenar atımda `+∞`.

**Artefakt oranı:** `(eklenen + çıkarılan + fizyolojik olay + uyuşmazlık + taşınma mesafesi suspect_move_s'i
aşan şüpheli) / çıktı atım sayısı`. **Ektopik atımlar sayılmıyor.** Uyuşmazlık isteğe bağlı bir girdi
(girdi atımı başına `bool`, örneğin "ikincil detektör 50 ms içinde tespit yok"). Verilirse hem NN'leri
dışlıyor hem orana giriyor. Kategoriler toplanıyor: bir atım iki kategoride birden sayılabilir (örneğin eklenen
ve aynı zamanda uzak şüpheli), raporun tanımıyla aynı.

**Sınıf öncelikleri.** Çıktı atımı: Inserted > Ectopic > Suspect > Disagreement > Normal. Girdi atımı:
Extra / PhysiologicalLimit (çıkarıldıysa), yoksa Ectopic > Suspect > Missed (çıktıda hemen ardından eklenen
atım geliyor) > Disagreement > Normal.

**Maskeli zaman alanı** (`time_domain_masked(nn_ms, valid)`):
- `nn_ms[k]` kullanılıyor ⇔ `valid[k]` ve değer sonlu (eksik `valid` girdisi geçersiz; `NaN` girdisi
  parçalar arası boşluk işareti olabiliyor).
- **Çift metrikleri** RMSSD, SDSD, pNN50, pNN20 (ve CVSD = RMSSD / MeanNN) yalnız **bitişik** ve ikisi de
  kullanılan aralıkların farklarından. Dışlanan aralık ya da boşluk seriyi bölüyor, hiçbir fark onu aşmıyor.
  2'den az fark → NaN (`rmssd_masked` → `None`).
- **Dağılım metrikleri** (MeanNN, SDNN, medyan, MAD, IQR, min, max, CVNN, kalp hızı, HTI, TINN) kullanılan
  aralıkların `time_domain`'i (≥ 3, yoksa NaN).
- Her aralık geçerliyse sonuç `time_domain` ile **bit düzeyinde aynı** (test).

## Varsayılanlar ve kaynakları

| Parametre | Varsayılan | Kaynak |
|---|---|---|
| `alpha`, `c1`, `c2`, pencereler | 5,2 / 0,13 / 0,17 / 91 / 11 | Lipponen & Tarvainen 2019, NeuroKit2 |
| `threshold_floor_s` | 0 (kapalı) | nötr, yayımlanmış algoritma |
| `premature_factor`, `compensatory_factor` | 0,85, 0,5 | yaygın erken atım ölçütü (≥ %15 erken) ve yarı telafi; rapor 12 |
| `two_stage` | açık | rapor 12 §3.3 |
| `suspect_move_s` | 0,030 s | rapor 12 §1.2 (20 ms fiducial sapmasının 1,5 katı, 50 ms uyuşmazlık toleransının altında). 20/30/50 ms sonucu belirlemiyor |
| `exclude_inserted_removed` | açık | görev tanımı. Kapalı = araştırma referansı |
| `physiological_rr_s` | yok | nötr. Ürün örneği `(0.3, 2.0)` |

Ürün eşikleri (artefakt oranı ≤ %2, en az NN, tutulan pay, `alpha = 6`) kütüphanede davranış olarak
**yok**. README'de açıkça "örnek" diye işaretlenmiş bir bölümde duruyor.

## Doğrulama

### NeuroKit2 paritesi
- Varsayılan parametrelerle Kubios değişmedi: `neurokit_kubios.json` 26/26 aynı (ilk geçiş ve düzeltilmiş
  tepeler). `threshold_floor_s` = 0, −0 ve −0,05 ile de 26/26 aynı (`tests/kubios_floor.rs`).
- Diğer bütün parite testleri (`tests/neurokit_parity.rs`) değişmeden geçiyor.

### v0.5.0 ile aynılık (taban 0)
- Yeni sentetik fixture'ın 11 serisinde (3282 atım) varsayılan parametrelerle ilk geçiş bayrakları,
  düzeltilmiş konumlar ve geçiş sayısı, `31117b4`'ten (v0.5.0) ayrı bir worktree'de derlenen kütüphaneyle
  **birebir aynı** (tek seferlik kontrol, sonuç özetinin bit karşılaştırması).
- Fixture'ı üreten Python portu (`ektopi_kubios.py`) rapor 12'de v0.5.0 ile gerçek veride birebir doğrulanmıştı.
  Bu yüzden fixture'ın varsayılan satırları v0.5.0 davranışını da sabitliyor.

### Python referansıyla eşdeğerlik (sentetik)
`tools/exclusion_reference.py` (yalnız sentetik veri: RSA + LF + AR(1) sinüs modeli, enjekte edilmiş erken +
telafi ektopi, kaçırılmış, fazla atım, titreşim), araştırma portunu (`ektopi_kubios.py`) ve `ektopi_gate.py`'deki
`ex_run`'ı (örnekleme frekansı parametre yapılarak birebir kopyalandı) çalıştırıyor. Değerler IEEE-754 bit
deseniyle saklanıyor, karşılaştırma kesin.

| Test | Sonuç |
|---|---|
| Kubios, alpha 5,2 / 6 × taban 0 / 30 / 40 / 50 ms, 11 seri (`kubios_floor_reference.json`) | **55/55** (seri, varyant) çifti aynı: ilk geçiş bayrakları ve düzeltilmiş konumlar bit düzeyinde, geçiş sayısı |
| Dışlama kipi, 3 parametre takımı (varsayılan, alpha 6, taban 40 + alpha 6) × tek/iki aşama × uyuşmazlık var/yok, 11 seri (`exclusion_reference.json`) | **132/132** koşu aynı: çıktı konumları (bit), ektopik (Kubios eşlemesi ve örüntü ayrı ayrı), şüpheliler ve taşınma mesafeleri (≤ 1e-12 göreli), uyuşmazlık, dışlanan NN, eklenen / çıkarılan sayısı. Toplam 1248 ektopik atım, 32 aşama-2 şüphelisi |

Referansla karşılaştırmada `exclude_inserted_removed = false` kullanılıyor (araştırma kodu eklenen/çıkarılan
atım komşusu NN'leri tutuyordu). Varsayılanla (açık) dışlanan NN kümesinin referansınkini kapsadığı da test
ediliyor.

### Sentetik enjeksiyon (`tests/exclusion.rs`, 300 atım, RMSSD ≈ 10 / 20 / 40 ms)

| Test | Sonuç |
|---|---|
| %3 ektopi + %1 artefakt, 20 seri: enjekte ektopik atımlar | **180/180** `Ectopic`, hiçbiri taşınmadı, iki NN'leri de dışlandı. Kaçırılmış ve fazla atımların hepsi yakalandı, artefaktların 58/60'ına dokunuldu |
| %3 ektopi + %1 artefakt, 60 seri: RMSSD göreli hatası | dışlama **medyan %1,0, p90 %2,7, en kötü %4,5** (her seride ≤ %10 assert). Kubios interpolasyonu medyan **%58,6**, p90 %150 (RMSSD ≈ 10 ms'de medyan %146). İnterpolasyon her seride daha kötü |
| %8 ektopi + %2 artefakt, 60 seri | artefaktlara dokunma: tek aşama 219/360, **iki aşama 336/360**. RMSSD hatası medyan / p90: tek aşama %143 / %437, **iki aşama %1,7 / %6,1**. Örüntü kuralıyla ektopik atımların hepsi (1440/1440) iki kipte de dışlanıyor |
| Aynısı, yalnız Kubios bayrakları (`premature_factor = 0`) | iki NN'si de dışlanan ektopik: tek aşama 827/1440 (%57), **iki aşama 1175/1440 (%82)** |
| Hepsi geçerli NN | maskeli RMSSD = `compute_rmssd` (bit), `time_domain()` = `time_domain` (bütün alanlar) |
| Bigemini (her ikinci atım erken) | 298 NN'den 3'ü tutuluyor, 1 çift: `rmssd()` = `None`, RMSSD NaN |
| İki parça + `concat_nn` | hiçbir fark boşluğu aşmıyor, RMSSD parçaların farklarından bit düzeyinde |

Notlar: (1) Örüntü kuralıyla ektopiler tek aşamada da yakalanıyor. İki aşamanın kazancı, şişen eşiklerin
gizlediği fazla atımlar ve titreşim (rapor 12 §3.3 ile aynı yön). Ektopi yalnız Kubios bayrağıyla bulunuyorsa
iki aşama ektopilerin de çoğunu geri kazanıyor, ama onları `Suspect` olarak işaretliyor, `Ectopic` olarak değil.
(2) %8 ektopide iki aşamada bile tek tük seri kötü kalıyor (60 seride p90 %6,1, en kötüsü yüzlerce yüzde). Bu
yüzden testler bu hücrede medyan ve p90 üzerinden assert ediyor. Pencere kapısı (artefakt oranı, tutulan pay,
artık kontrolü) ürün tarafında gerekli.

### Birim testleri
Temiz seri (hepsi Normal, oran 0), tek erken atım (Ectopic, taşınmadı, oran 0), kaçırılmış + fazla atım
(Missed / Extra / Inserted, ilgili NN'ler dışlandı, `exclude_inserted_removed = false` ile tutuluyor),
tamsayı yol (eklenen konum tamsayı), uyuşmazlık + fizyolojik sınır (düşürme, bölme, `summary_between` =
toplamlar), geçersiz girdiler (`NonFinite`, `UnsortedPeaks`, `InvalidSamplingRate`, `InvalidParameter`),
maskeli zaman alanında çiftlerin dışlanan aralığı aşmaması, NaN boşluk işareti, az çift → NaN.

### Süre
400 atım, release (bu makine, tek seferlik ölçüm): `classify_beats` iki aşama ~0,49 ms, tek aşama ~0,36 ms,
`fix_positions_with` ~0,25 ms, `time_domain_masked` ~0,02 ms.

## Uyumluluk

- **Kırıcı (küçük):** `KubiosParams`'a public alan `threshold_floor_s` eklendi. Bütün alanlarıyla
  `KubiosParams { .. }` kuran (ve `..Default::default()` kullanmayan) kod güncellenmeli. Crate içinde tek
  yer (`fix_peaks`) güncellendi. Python bağlaması (`fix_peaks`'i konumsal argümanlarla çağırıyor) etkilenmedi.
- `KubiosParams` ve `ExclusionParams` `serde` özelliğiyle `Serialize` / `Deserialize` türetiyor.
- Düzeltme API'leri aynı. Dışlama kipi ayrı bir yol, mevcut kod etkilenmiyor.

## Sınırlar

- Ektopik tanımı yalnız RR'den geliyor (Kubios + örüntü). Morfoloji kullanılmıyor, klinik anlamı yok. Titreşim
  Kubios `ectopic` bayrağı alırsa ektopik sayılıyor (orana girmiyor ama NN'leri dışlanıyor). Rapor 12'de
  kayıtlı ektopi oranı enjekte edilenin medyan 1,08–1,11 katıydı.
- Sentetik enjeksiyon düzgün dağılımlı, olaylar ≥ 3 atım arayla. Gerçek artefaktlar kümelenir. Gerçek veride
  sınıfların doğruluğu etiket olmadan ölçülemedi (rapor 12 §6).
- Kubios'un algoritmadan gelen sınırları değişmedi: çok düzenli serilerde kaçırılmış atım uzun/kısa olarak
  sınıflanabiliyor (Th2 küçük). Dışlama kipinde bu atım `Suspect` olur ve NN'leri dışlanır.
- Uyuşmazlık sinyali (ikincil detektör) kütüphanede hesaplanmıyor. Çağıran veriyor.
