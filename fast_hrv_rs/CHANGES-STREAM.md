# fast-hrv: akan R tepe detektörü (`StreamDetector`)

Tarih: 2026-11-24 · Dal: `wip/stream-peaks` (`master` `a81ff37` üzerine yeniden temellendi; birleştirilmedi,
push yok) · Sürüm: 0.4.0 (yayımlanmadı) · Araçlar: Rust **1.100.0**, MSRV 1.85 ayrıca test edildi
(`cargo +1.85.0 test`).

## Kapsam (güncellenmiş)

v0.3.0 ile ürünün kullanacağı detektör **pencereli Elgendi + inceltme** (`elgendi_refined_windowed`)
ve ikincil olarak NeuroKit yöntemi (`dual_detect`) oldu. Bu yüzden asıl hedef değişti:

- **(A) yapıldı:** pencereli boru hattının akış sarmalayıcısı `stream::StreamDetector`. Paket paket temiz
  örnek alır, her 10 sn pencere + 2 sn sağ payı tamamlanınca onaylanan tepeleri (mutlak indeks + alt-örnek
  konum) verir; çıktı `elgendi_refined_windowed` ve `dual_detect` ikincilinin pencereli hâliyle **bit
  düzeyinde aynı**.
- **(B) yapılmadı:** düşük gecikmeli tek geçişli NeuroKit tarzı akış detektörü (eski devir notundaki
  tasarım). (A) tamamlanıp testler yeşil olduktan sonra bilinçli olarak ertelendi; tasarım aşağıda
  (§9) güncel hâliyle duruyor.

## Commit listesi

| Commit | Konu |
|---|---|
| `0301094` | wip: devir notu (eski `805f61d`, `master`'a yeniden temellendi) |
| `3cbb0b8` | feat(stream): `StreamDetector`, `dual_detect_windowed`, `tests/stream_detect.rs` |
| `3039aa8` | bench(capacity): 2a adımı isteğe bağlı `StreamDetector` ile (`--stream`) |
| (bu commit) | docs: bu not, README, CHANGELOG 0.4.0, CAPACITY.md §8, sürüm 0.4.0 |

Her adımdan sonra: `cargo fmt`, `cargo clippy --all-targets` (varsayılan, `--no-default-features`,
`--features serde,parallel,simd`: **uyarısız**), `cargo test`. Son durum: 85 birim, 10 detektör, 4 filtre,
4 parity, 6 sağlamlık, 21 SQI, **6 akış detektörü**, 3 akış filtresi, 5 doctest; hepsi geçiyor
(1.100.0 ve 1.85.0). Yeni bağımlılık yok, AVX varsayımı yok (skaler `f64`), `unsafe` yok, eşik yok
(genel algoritma; karar app-pipeline'da), repoda gerçek veri yok (bütün testler sentetik).

## 1. API (`src/stream.rs`, kök modülden de yeniden ihraç)

| Öğe | Ne yapar |
|---|---|
| `StreamConfig { window: WindowConfig, dual: DualConfig, secondary: bool }` | pencere ızgarası (10 sn / 2 sn / 0,15 sn), detektörler; `secondary = true` (varsayılan) pencere başına tam `dual_detect`, `false` yalnız birincil (daha ucuz) |
| `StreamDetector::new(fs, start_index, cfg)` | ilk örneğin mutlak indeksi `start_index` (batch'teki `start_sample` ile aynı saat). Yapılandırma batch fonksiyonlarının kendi doğrulamasından geçer (aynı hatalar) |
| `push(&[f64], &mut StreamPeaks)` | örnekleri ekler, onaylanan tepeleri `out.primary` / `out.secondary`'ye ekler. Sonlu olmayan örnek: bütün paket reddedilir, durum değişmez (`HrvError::NonFinite`) |
| `push_with(samples, out, \|w: &StreamWindow\| …)` | aynı, her işlenen pencere için geri çağrı: `samples` (detektörün okuduğu parça), `core`, `chunk_start`, pencerenin kendi `primary` / `primary_raw` / `secondary` sonuçları, `bsqi`; `core_range()`, `core_indices()`. SQI'yi çekirdek üzerinde ikinci bir örnek tamponu olmadan hesaplamak için |
| `flush(out)` / `flush_with` | segment sonu (kayıt sonu): kalan pencereler kırpılmış parçalarla işlenir, bekleyen bütün tepeler verilir; yeni segment aynı indeksten başlar |
| `reset_at(next_index, out) -> Gap` / `reset_at_with` | boşluk: segmenti `flush` gibi bitirir, `next_index`'ten (≥ `next_index()`) yeni segment başlatır; `gaps::Gap { start: eski next_index, end: next_index }` döner (`split_at_gaps` / `rri_with_gaps` ile). Eşit indeks = örnek kaybı olmadan süreksizlik (ör. filtre sıfırlama), `gaps.rs`'teki sıfır uzunluklu boşlukla aynı anlam |
| `state_bytes()` | çağrılar arasında tutulan bayt (yapı + örnek tamponu + bekleyen tepeler) |
| `max_latency_samples()`, `next_window_ready_at()`, `next_index()`, `segment_start()`, `window_samples()` | gecikme sınırı, sıradaki pencerenin işleneceği örnek, saat |
| `StreamPeak { index: u64, position: f64 }` | mutlak örnek indeksi ve alt-örnek konum (batch konumlarıyla bit düzeyinde aynı) |
| `detect::dual_detect_windowed(clean, fs, start, &WindowConfig, &DualConfig) -> WindowedDual` | yeni batch fonksiyonu: pencere başına bir `dual_detect`, birincil ve ikincil konumlar aynı kuralla ayrı ayrı tekilleştirilir. `StreamDetector`'ın batch referansı; `primary` = `elgendi_refined_windowed` |

Kullanım (ham → filtre → detektör):

```rust
let mut filt = EcgStreamFilter::new(512.0, &EcgFilterConfig::default())?;
let mut det = StreamDetector::new(512.0, oturum_baslangic_indeksi, StreamConfig::default())?;
let mut out = StreamPeaks::default();
for paket in paketler {                       // 32 ham örnek
    det.push(&filt.process(paket), &mut out)?;
}
det.flush(&mut out)?;                          // kayıt sonu; boşlukta: det.reset_at(yeni_indeks, &mut out)? + filt.reset()
```

## 2. Gecikme

Varsayılanlar (512 Hz, W = 10 sn = 5120, M = 2 sn = 1024, dedup = 0,15 sn = 76,8 örnek):

- Pencere `k`, örnek `(k+1)·W + M` gelince işlenir. Çekirdek `k`'deki tepe o anda verilir: gecikme
  çekirdek başındaki tepe için `W + M` = **12 sn**, çekirdek sonuna yakın tepe için `M + 2·dedup` ≈ **2,3 sn**;
  ortalama ~7 sn.
- İstisna: çekirdeğin son `2·dedup + 2` örneğindeki (≤ 0,3 sn) bir tepe, sonraki pencerenin bulacağı bir
  tepenin kopyası olabilir; batch kuralı "çekirdeğin daha içinde olan kalır" olduğu için sonraki pencereyi
  bekler. Pencere sınırı başına en fazla bir tepe (refrakter 0,3 sn). En kötü durum
  `W + M + 2·dedup + 2` örnek ≈ **12,3 sn** (`max_latency_samples()`).
- Ölçüm (30 dk sentetik, 32'lik paket, `tests/stream_detect.rs`): tepe → onu veren örnek arası
  **min 2,30 sn, ort. 7,31 sn, maks. 12,29 sn** (sınır 12,30 sn).
- Pencere geri çağrısı (SQI) çekirdek başından `W + M` = 12 sn sonra çalışır (mevcut bench ile aynı).

## 3. Durum boyutu

| Kalem | 512 Hz, varsayılan |
|---|---:|
| Örnek tamponu: en fazla `W + 2·M` = 7168 `f64`, bir kez ayrılır | 56,0 KiB |
| Bekleyen tepeler, iki detektör (kapasite yapılandırmadan hesaplanır, 32 bayt/tepe) | ~3,6 KiB |
| Yapı | ~0,2 KiB |
| **`state_bytes()`** | **61 352 bayt = 59,9 KiB** |
| 500 akış | ~29,3 MiB |

- Sabit: 30 dk boyunca her paketten sonra `state_bytes()` aynı kaldı (test ediyor), tampon hiç büyümüyor.
- Bekleyen kapasite = bir pencerenin tepeleri + önceki pencereden kalanlar; tepeler en az
  `refrakter − 2·inceltme yarı penceresi` (0,3 − 0,1 = 0,2 sn) aralıklı → detektör başına 54 yer. Refrakter
  0 ise 64 ile başlar, gerekirse büyür (belgelendi).
- Detektör ara dizileri (filtrelenmiş sinyal, kayan ortalamalar, NeuroKit gradyanı) pencere başına ayrılıp
  bırakılır, batch yoluyla aynı; oturum başına değil işçi başına geçici bellektir.
- CAPACITY.md bütçesi (~98 KiB/oturum) korunuyor: detektör tamponu, oturumun 60 KiB'lik temiz `f64`
  halkasının yerini alıyor (SQI geri çağrıda detektörün parçasını okuyor). Bench: oturum 98,6 → 98,1 KiB.

## 4. Eşitlik kanıtı

İddia: başlangıç indeksi `s0` olan bir segmentte, birleştirilmiş çıktı `elgendi_refined_windowed` /
`dual_detect_windowed(clean, fs, s0, …)` ile **bit düzeyinde aynı**, paketleme ne olursa olsun.

Yapısal gerekçe:

1. **Aynı parça:** pencere `k`, `[k·W − M, (k+1)·W + M)` aralığının segmente kırpılmış hâlini görür;
   akışta da tam bu dilim (aynı `f64` değerleri) detektörlere verilir. Pencere, sağ payının son örneği
   geldiği anda işlenir; `push` paketi pencere sınırında böler, yani paket boyutu dilimi etkilemez. Segment
   sonunda (`flush` / `reset_at`) kalan pencereler, batch'in kayıt sonundaki gibi kırpılmış parçayla
   işlenir (`k·W < segment sonu` olan bütün `k`'ler).
2. **Aynı aritmetik:** mutlak konum `c0 as f64 + p`, derinlik `min(a − çekirdek0, çekirdek1 − a)`, süzgeç
   `derinlik ≥ −dedup`: batch'teki ifadelerin aynısı.
3. **Aynı tekilleştirme:** batch, bütün bulunanları (konum, pencere) ile kararlı sıralayıp açgözlü bir
   geçiş yapar (yalnız son tutulan tepe değişebilir). Akış aynı geçişi artımlı yapar:
   - pencere `k`'den sonra yalnız konumu `(k+1)·W − dedup − 1`'den küçük olanlar beslenir; daha sonraki
     pencerelerin tepeleri süzgeç gereği `≥ (k+1)·W − dedup` olduğundan beslenenlerin sırası küresel
     sıralamayla aynıdır (eşitlikte pencere, sonra pencere içi tespit sırası: kararlı sıralamanın aynısı);
   - son tutulan tepe, gelecekteki hiçbir tepe `dedup` içine giremeyecekse (`son + dedup + 1 ≤ sınır`)
     verilir; aksi hâlde bir sonraki pencereyi bekler. 1 örneklik pay, kayan nokta yuvarlamasına karşı
     yalnız gecikmeyi etkiler, sonucu değil.

Testler (`tests/stream_detect.rs`, karşılaştırma `f64::to_bits` ile):

| Test | Ne |
|---|---|
| `end_to_end_raw_filter_stream_equals_batch_for_any_packetisation` | ham (DC, taban kayması, 50 Hz, gürültü) → `EcgStreamFilter` → `StreamDetector`, paketler **1, 32, 33, 512, 7168, rastgele (1–1000)**, başlangıç ofsetleri **0, 1, 777, 5119, 5120, 12 345, 52 223**; birincil ve ikincil `dual_detect_windowed` ile, yalnız-birincil mod `elgendi_refined_windowed` ile aynı; indeks–konum tutarlılığı |
| `other_grids_rates_and_configurations` | 250 Hz + ters derivasyon + pay > pencere (3 sn / 4 sn), 7,3 sn pencere + pay 0; `neurokit2013` ön ayarı, ±70 ms inceltme |
| `window_callback_sees_the_batch_chunk_and_dual_detect_result` | geri çağrıdaki parça = batch parçası; `primary`, `primary_raw`, `secondary`, `bsqi` = o parçada `dual_detect`; pencere listesi = batch pencereleri |
| `gap_reset_equals_batch_per_segment` | 40 sn / 3,3 sn kayıp / 26,8 sn / süreksizlik / 49,9 sn; her segment ayrı batch'le aynı; dönen `Gap`'ler; boşlukta tepe yok; `split_at_gaps` üç segment veriyor |
| `thirty_minutes_with_artifacts_no_lock_up_bounded_state_and_latency` | 30 dk, 20 sn güçlü gürültü + 3 sn elektrot sıçraması + 2 sn düz hat; batch ile aynı; artefakt dışı her dakikada iki detektör de gerçek atım sayısının ±1'i (kilitlenme yok); durum sabit; gecikme sınır içinde |
| `rejects_non_finite_samples_without_changing_the_output` | `inf` içeren paket reddedilir, çıktı değişmez |
| birim: `dedup_stress_equals_batch` | sahte yoğun detektör (pencereye bağlı, sırasız, aynı konumda kopyalar, sınırların iki yanında), 3 ızgara × 4 başlangıç × 4 paket boyutu = `detect_windowed`. Mutasyon kontrolü: birleştirme kapatılınca ya da son tepe erken verilince test **kırılıyor** |
| birim: `buffer_is_bounded_and_state_constant`, `invalid_configuration_and_input`; doctest | tampon ≤ `W + 2M`, durum sabit; geçersiz yapılandırma / geri giden `reset_at` hata |

## 5. Kapasite benchmark'ı (`cargo bench --bench capacity -- --stream`)

Aynı makine (i7-9700K, çekirdek 3), 500 kullanıcı × 600 sn, yayılmış takvim (CAPACITY.md §8):

| | varsayılan | `--stream` |
|---|---:|---:|
| 2a µs / kullanıcı / sn | 16,79 | 15,47 |
| 2b µs / kullanıcı / sn | 16,29 | 17,27 |
| **Toplam** | **50,54** | **50,33** |
| Oturum belleği | 98,6 KiB | 98,1 KiB |

**Maliyet değişmiyor** (gürültü içinde). 2a'nın biraz ucuzlaması 14 sn parçanın halkadan kopyalanmamasından;
paket başına defter tutma (32 örnek kopyası) 2a'ya dahil. Varsayılan mod da aynı kalıyor (artık paket
turları bloklar hâlinde ölçülüyor; aşama 1: 3,91 µs, önce 3,94).

## 6. Sapmalar ve kararlar

- **Kapsam:** eski tasarımdaki tek geçişli NK tarzı detektör yerine önce pencereli boru hattının akış hâli
  (lead'in güncel kararı). Tek geçişli detektör yapılmadı (§9).
- **Halka yerine sabit kapasiteli doğrusal tampon:** detektörler bitişik dilim istiyor; halka tampon her
  pencerede 14 sn'lik kopya gerektirirdi (bench bugün bunu yapıyor). Tampon bir kez `W + 2M` ile ayrılır,
  pencereden sonra yalnız sonraki parçanın ihtiyacı olan `2M` örnek başa kaydırılır (10 sn'de bir 16 KiB
  `memmove`). Bellek sınırı aynı (~14 sn).
- **Gecikme sınırı `W + M`'den biraz fazla:** çekirdek sonundaki ≤ 0,3 sn'lik bölgedeki tepe, batch'le
  bit düzeyinde eşitlik için sonraki pencereyi bekler (en kötü ~12,3 sn). Alternatif (beklememek) batch'ten
  sapma demek; seçilmedi.
- **`reset_at` eski segmenti boşaltır** (son pencereleri kırpılmış parçayla işler, tepeleri verir) ve
  yalnız ileri gider (`next_index ≥ next_index()`; geri giden saat hata). Böylece her segment, o segmentte
  ayrı çalıştırılmış batch'le aynı. Filtreyi (`EcgStreamFilter::reset`) sıfırlamak çağıranın işi.
- **İkincil detektör**, `dual_detect` ile aynı: birincilin işaretiyle inceltilir; `bsqi` (geri çağrıda)
  parçanın tamamı üzerinden (`dual_detect` gibi), çekirdek üzerinden değil. Çekirdek bSQI'si gerekiyorsa
  `core_indices()` ile SQI tarafında hesaplanır (bench böyle yapıyor).
- **Sonlu olmayan örnek** paketi bütünüyle reddeder (batch tüm çağrıyı reddeder). Akışta devam etmek için
  çağıran `reset_at` ile atlayabilir.
- `dual_detect_windowed`, `detect_windowed`'i iki kez çağırır (ikincisi saklanan ikincil sonuçları aynı
  pencere sırasıyla tekrar oynatır): tekilleştirme kuralı tek yerde kalıyor, detektörler pencere başına bir
  kez çalışıyor.
- `detect_windowed` / `elgendi_refined_windowed` **değişmedi** (referans olarak bağımsız kalsın diye akış
  tarafı ayrı yazıldı; eşitlik testle kanıtlanıyor).

## 7. app-pipeline için notlar

- Oturum başına: `EcgStreamFilter` + `StreamDetector` + ham sayım halkası (düz hat / satürasyon için); temiz
  halka gerekmez. SQI `push_with` geri çağrısında, pencerenin kendi tespitleriyle çalışır.
- Canlı = geçmiş: geçmişi yeniden işlerken `dual_detect_windowed` (ya da yeni bir `StreamDetector`) aynı
  mutlak `start_index` ile aynı tepeleri verir.
- Tepe çıktısı `u64` indeks + `f64` konum; RR için `rri_from_positions`, boşluk için dönen `Gap`'ler.

## 8. Ölçülmeyenler / açık

- Sunucuda (Xeon E7-4850) `--stream` ölçümü yapılmadı; aynı hesap yapıldığı için oranın (×~4,8) aynı
  kalması beklenir.
- NeuroKit2 ile ek doğruluk ölçümü gerekmedi: akış çıktısı batch'le birebir aynı, batch'in NK parity'si
  ve doğruluk testleri (`tests/detect.rs`) geçerli.

## 9. (B) Düşük gecikmeli tek geçişli detektör: tasarım (uygulanmadı)

Gerekirse (ör. canlı KH göstergesi için < 1 sn gecikme) ayrı bir tip olarak eklenebilir; pencereli
detektörün yerine değil, yanında. Ürünün HRV yolu pencereli detektörde kalır.

- Tip `peaks::StreamPeakDetector` (batch `find_peaks` / `ecg_peaks` değişmez). API `StreamDetector` ile aynı
  kalıp: `new(fs, start_index, params)`, `push(&[f64], &mut Vec<StreamPeak>)`, `flush`, `reset_at(next) -> Gap`,
  `state_bytes()`.
- NK ile aynı tutulacaklar: `np.gradient`, boxcar `uniform_filter1d(mode="nearest")` (başta ilk değer,
  sonda yalnız `flush` ile son değer), `qrs = smoothgrad > 1.5·avggrad`, beg/end eşleme, QRS içinde en
  belirgin yerel maksimum (SciPy plato kuralı), `peak − last > mindelay` açgözlü kural.
- Bilinçli sapma (nedensellik için): `min_len = 0,4 · (son W sn içinde biten QRS adaylarının uzunluk
  ortalaması, en fazla M aday)`, öneri W = 10 sn, M = 128; `max_qrs` (0,5–1,0 sn) üstü adaylar tepe vermez.
  Bu yüzden batch `find_peaks` ile bit düzeyinde aynı **olmaz**; eşitlik ölçütü "her paketlemede kendi
  `stream_peaks(signal)` toplu çağrısıyla aynı" + NK `ecg_peaks` ile ±1 örnek eşleşme oranı olur.
- Gecikme (512 Hz): ileri bakış 25 + 191 + 1 = 217 örnek (~0,42 sn); tipik ~0,5–0,6 sn, üst sınır
  `max_qrs` + 0,43 sn. Durum ~6–9 KiB/akış.
- İsteğe bağlı: tepeleri `refine_peaks` ile ±50 ms inceltme (sinyal tamponu 50 ms uzar).
- Yapılacaklar: uygulama + birim testleri; paketleme eşitliği; NK `ecg_peaks` ile sentetik fixture'larda
  eşleşme oranı; W, M, `max_qrs` seçimi; gecikme / durum testleri; belgeler.
