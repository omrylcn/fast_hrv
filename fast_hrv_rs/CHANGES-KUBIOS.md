# fast-hrv: Kubios, atım başına düzeltme sonucu ve alt-örnek konumlar

Tarih: 2026-11-24 · Temel: `ff53c77` · Dal: `master` (push yok) · Sürüm: 0.5.0 (yayımlanmadı)
Araçlar: Rust **1.100.0**, MSRV 1.85 ayrıca test edildi (`cargo +1.85.0 test`).

Neden: `review/12-onay-olcumleri.md` §5.3 ve §6. (1) `Artifacts::total_count()` bütün düzeltme
geçişlerindeki işaretleri topluyor, hiçbir şeyi değiştirmeyen işaretler de dahil. "Düzeltilen atım oranı"
olarak kullanılınca 2–5 kat yüksek çıkıyor (S01: %38,8; gerçekte değişen atım %14,4). Ürün kapısı
"gerçekten değişen atım payı" istiyor (60 sn'de ≤ %2, 300 sn'de ≤ %1). (2) `fix_peaks_with` yalnız tamsayı
indeks alıyor. Detektörün alt-örnek konumları yuvarlanmak zorunda kalıyor ve inceltmenin kazandırdığı
hassasiyet kayboluyor.

## Commit listesi

| Commit | Konu |
|---|---|
| `d4f007e` | feat(kubios): atım başına düzeltme sonucu (değişmedi/taşındı/eklendi, silinen sayısı), değişen atım oranı |
| (bu commit) | feat(kubios): `f64` alt-örnek konumlarda Kubios, kısayollar, belgeler (bu not, README, CHANGELOG) |

Her adımda `cargo fmt` ve `cargo clippy --all-targets -- -D warnings` çalıştırıldı (varsayılan,
`--no-default-features`, `--features serde,parallel,simd`: **uyarısız**), ardından `cargo test` ve
`cargo +1.85.0 test`, hepsi geçti. `cargo doc` `-D warnings` ile temiz. Yeni bağımlılık, AVX varsayımı,
`unsafe` ya da eşik eklenmedi.

## API

| Öğe | Ne yapar |
|---|---|
| `KubiosOutput::corrections: Corrections` | yeni alan; `fix_peaks_with` dolduruyor |
| `Corrections { status, input_to_output, removed }` | `status[j]`: her **çıktı** atımı için `BeatStatus`. `input_to_output[i]`: girdi atımının çıktı indeksi (`None` = silindi). `removed`: silinen girdi atımı sayısı |
| `BeatStatus::{Unchanged, Moved, Inserted}` | değişmedi / ektopik ya da long-short düzeltmesiyle taşındı / kaçırılmış atım olarak eklendi |
| `Corrections::changed_count()`, `changed_ratio()`, `moved_count()`, `inserted_count()`, `removed_count()`, `changed_mask()`, `output_to_input()`, `n_output()`, `n_input()` | yardımcılar |
| `KubiosOutput::changed_count()` / `changed_ratio()` | kısayol |
| `gaps::segment_changed_counts`, `total_changed`, `total_removed`, `overall_changed_ratio` | segment başına ve toplam. `AsRef<Corrections>` üzerinden genel, `KubiosOutput` ve `KubiosPositions` ile çalışıyor |
| `fix_positions_with(positions: &[f64], fs, params)` → `KubiosPositions { positions, artifacts, passes, corrections }` | aynı Lipponen–Tarvainen algoritması, RR `f64` konumlardan. Değişmeyen atımlar girdi konumunu bit düzeyinde koruyor. Eklenen ve taşınan atımlar komşuların **tam** orta noktasına gidiyor (yuvarlama yok) |
| `fix_positions_with_midpoint(..., KubiosMidpoint::{Exact, Floor})` | `Floor` = tamsayı yolunun ve NeuroKit2'nin `prev + floor((next − prev)/2)` kuralı |
| `detect_artifacts_positions` | tek tespit geçişi, `f64` konumlarla |
| `KubiosPositions::rri_ms(fs)` | düzeltilmiş konumlardan RR (`rri_from_positions`) |
| `RefinedPeaks::fix_kubios(fs, params)`, `stream::peak_positions`, `stream::fix_stream_peaks`, `gaps::fix_positions_segmented` | detektör çıktısı → düzeltilmiş konum → RR kısayolları |

Konumların birimi serbest. `sampling_rate` konumları saniyeye çevirmeli: saniye cinsinden zamanlar
`sampling_rate = 1.0` ile verilebilir.

## Oran tanımı

- Durum **son çıktı ile girdi karşılaştırılarak** belirleniyor, işaretlerden çıkarılmıyor. Çıktıda girdideki
  bir konumla birebir aynı olan atım `Unchanged` sayılıyor (iki dizi de sıralı, eşit konumlar sırayla
  eşleşiyor). Kalan çıktı atımı, geçişler boyunca izlenen kökeni henüz eşleşmemiş bir girdi atımıysa
  `Moved`, değilse `Inserted`. Çıktısı olmayan girdi atımları **silinmiş** sayılıyor.
- Bu nedenle bir geçişte taşınıp sonraki geçişte geri dönen ya da "düzeltmesi" aynı yere düşen işaretli
  atım `Unchanged` kalıyor. Silinip tam aynı konuma yeniden eklenen atım da öyle.
- **`changed_ratio()` = (taşınan + eklenen) / çıktı atım sayısı.** Silinenler (fazla atım) çıktıda yer
  almadığı için paya girmiyor, `removed_count()` ile ayrıca raporlanıyor. Çıktı boşsa `NaN`.
- `Artifacts::total_count()` olduğu gibi duruyor. Belgesinde artık açıkça yazıyor: bütün geçişlerin
  **işaret** sayısı, düzeltilen atım oranı değil. `changed_ratio()` kullanılmalı.
- Not: review/12'deki ölçüm "max(silinen özgün, eklenen)" tanımını kullanıyordu. Buradaki tanım taşınanları
  da sayıyor ve silinenleri ayrı tutuyor. Kapı eşiği bu tanımla yeniden doğrulanmalı.

## Önce / sonra (NeuroKit2 fixture'ları, 26 dizi, 3375 çıktı atımı)

| | İşaret (`total_count`) | Değişen (taşınan + eklenen) | Silinen |
|---|---:|---:|---:|
| Toplam | 56 | 47 (41 + 6) | 6 |
| `extra_*` (5 dizi) | her biri 2 | her biri 1 taşınan | her biri 1 |
| `longshort_70` | 5 | 4 | 0 |
| `missed_350` | 4 | 3 | 0 |
| `mixed_200` (çok geçişli) | 10 | 8 | 1 |

Fixture'lar tek artefaktlı ve az geçişli olduğundan fark burada küçük (56'ya karşı 53). Gerçek veride fark
2–5 kat (review/12 §5.3). Etkisiz işaretlerin asıl kaynağı geçişler boyu birikim. Düzeltilmiş tepeler
NeuroKit2 ile hâlâ 26/26 aynı. Her dizide `changed + removed ≤ total_count`, çıktı sayısı = girdi − silinen
+ eklenen, `Unchanged` atımların değeri girdiyle aynı (test).

## Alt-örnek konumlar: doğrulama

1. **Tamsayı değerli konumlar, `KubiosMidpoint::Floor`:** 26/26 dizide konumlar, işaretler, geçişler ve
   `corrections` `fix_peaks_with` ile **bit düzeyinde aynı**, düzeltilmiş konumlar NeuroKit2 ile aynı.
   Neden kanıtlanabilir: çekirdek tek bir genel koddan geliyor. 2^53 altındaki tamsayılarda `f64` çıkarma,
   2'ye bölme, `floor` ve toplama kesin, bu yüzden her ara değer `usize` yolundakiyle aynı.
2. **Tamsayı değerli konumlar, varsayılan tam orta nokta:** ilk tespit geçişi aynı. 26/26 dizide atım
   başına sonuç ve geçişler aynı. Değişen atımlar en çok **0,75 örnek** farklı: orta nokta başına 0,5,
   ardışık iki taşınan atımda 0,75'e çıkıyor.
3. **Sabit ikili kesirli kaydırma** (+0,25, −0,375, +0,4375): bütün RR'ler bit düzeyinde aynı kalıyor.
   Sonuç tamsayı sonucunun kaydırılmışı, 26/26.
4. **Atım başına rastgele kaydırma** (|δ| ≤ 0,45 örnek, inceltmenin aralığı), yuvarlanınca fixture
   girdisi geri geliyor:
   - Değişmeyen her atım alt-örnek girdi konumunu **bit düzeyinde** koruyor (26/26).
   - Atım başına sonuç yuvarlanmış tamsayı yoluyla **20/26** dizide aynı. Toplam değişen + silinen iki yolda
     da 53. Farklı 6 dizide fark en çok 2 atım ve silinen sayısı aynı:
     `none_40` [14, 16, 18] → [14, 18] · `ectopic_70` [36, 37] → [36, 37, 38] · `none_70` [2, 63] → [2] ·
     `longshort_140` [72] → [16, 72] · `missed_350` [110, 180, 181] → [181] ·
     `longshort_350` [176, 177, 211] → [127, 176, 177, 178, 211].
   - Neden: tamsayı RR dizilerinde karar sınırlarında **tam eşitlikler** oluşuyor (örneğin long/short aday
     kuralında `|dRR[i+1]| == |dRR[i+2]|`). Herhangi bir alt-örnek kaydırma bu eşitliği bozuyor.
     `none_70`'te 1e-9 örneklik kaydırma bile bir long/short adayı ekliyor. Yani "yuvarlanmış konumlarla
     aynı atımlar" genel olarak sağlanamaz. Test bunu gizlemiyor: farkları basıyor ve sınırlıyor (≤ 2 atım,
     ≥ 20/26 aynı).
5. **Sentetik EKG** (5 dk, bilinen R, inceltilmiş tepeler, ortadan bir atım silinmiş):
   `RefinedPeaks::fix_kubios` yalnız eksik atımı ekledi (1 değişen, oran %0,28). Yuvarlanmış yol aynı atımı
   ekledi, ama yuvarlama gürültüsü yüzünden iki atımı da long/short olarak taşıdı (3 değişen). RMSSD:
   gerçek 41,250 ms, alt-örnek düzeltilmiş **41,229 ms**, yuvarlanmış düzeltilmiş 41,004 ms.

## Uyumluluk

- `fix_peaks`, `fix_peaks_default`, `fix_peaks_with`, `detect_artifacts`, `fix_peaks_segmented` aynı
  imzayla aynı sonucu veriyor (NeuroKit2 fixture'ları 26/26). Kubios süresi değişmedi (`fix_peaks_default`
  ~0,19 ms).
- **Kırıcı (küçük):** `KubiosOutput`'a yeni public alan `corrections` eklendi. Bütün alanlarıyla
  `KubiosOutput { .. }` kuran ya da `..` olmadan parçalayan kod güncellenmeli. `Default` duruyor.
- Görevdeki iki istek çelişiyordu: "tamsayı değerli konumlarda tamsayı yoluyla bit düzeyinde aynı" ve
  "eklenen atımlar yuvarlamasız". Tamsayı yolu orta noktayı aşağı yuvarlıyor (NeuroKit2 de `int`'e
  çeviriyor). Çözüm: `fix_positions_with` yuvarlamasız (`Exact`) çalışıyor. Bit düzeyinde eşitlik ve
  NeuroKit2 paritesi `KubiosMidpoint::Floor` ile sağlanıyor. İkisi de test edildi (yukarıda 1 ve 2).
- `f64` yolu sonlu ve kesin artan konum istiyor (`HrvError::NonFinite` / `UnsortedPeaks`). 3'ten az atımda
  girdi aynen dönüyor.
- Kubios'un algoritmadan gelen sınırları değişmedi (review/12 §6.3): telafili erken atımda iki atım
  taşınıyor, ~%15 artefaktta eşikler şişip neredeyse hiçbir şey raporlanmıyor, kendi çıktısında sabit
  nokta değil. Oran kapısı tek başına yetmez. Artık RR kontrolü ve SQI kapısıyla birlikte kullanılmalı.
- Gerçek veri repoya eklenmedi. Yeni testler mevcut NeuroKit2 fixture'larını ve sentetik sinyalleri
  kullanıyor.
