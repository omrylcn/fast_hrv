# fast-hrv: akan R tepe detektörü (WIP, DURAKLATILDI)

## Devir notu (2026-11-24)

Durum: lead'in önceliği değiştirmesi üzerine **duraklatıldı**. Bu dalda henüz **kod yok**; yalnız okuma ve
tasarım yapıldı. `master` (`cd5d8ff`, v0.2.0) dokunulmadı.

### Yapılanlar
- Okundu: `CHANGES-P0.md`, `README.md`, `src/peaks.rs` (`find_peaks`), `src/filter.rs` (`EcgStreamFilter`),
  `src/gaps.rs`, `tests/streaming.rs`, `tools/neurokit_reference.py`, `review/09-fast-hrv-devir.md`.
- Sorun doğrulandı: `find_peaks` içinde `min_len = 0.4 · mean(end_qrs − beg_qrs)` **bütün girdiden**
  hesaplanıyor (P6); ayrıca `uniform_filter` 'nearest' kenarları girdinin sonuna bağlı. Bu yüzden tepeler
  girdinin nerede başlayıp bittiğine bağlı.
- NeuroKit2 kaynağı yerelde var: `/home/omrylcn/Work/bio_works/NeuroKit-master/neurokit2` (gürültü
  varyantları için referans üretiminde kullanılabilir; kurulu bir venv kontrol edilmedi).

### Tasarım kararları (önerilen, uygulanmadı)
- Yeni tip `peaks::StreamPeakDetector` (batch `find_peaks` / `ecg_peaks` **değişmeden** kalır).
  API: `new(fs, params)`, `push(&[f64], &mut Vec<u64>)` (onaylanan mutlak indeksler), `flush(&mut Vec<u64>)`
  (segment sonu: 'nearest' kenar dolgusu, kalanları ver, durumu sıfırla), `reset()` (bekleyeni at),
  `reset_at(next_index)` (boşluktan sonra saat), `state_bytes()`. Kolaylık: `stream_peaks(signal, fs)` =
  push(hepsi) + flush (testlerde "toplu" referans).
- Determinizm: örnek örnek işleme, ring buffer'lar mutlak indekse hizalı (ring idx = abs mod boyut), böylece
  her paketlemede aynı aritmetik. Kayan toplamların sürüklenmesine karşı ring sarıldığında (sabit mutlak
  indekslerde) tam yeniden toplama — deterministik ve sınırlı hata.
- NK ile aynı tutulacaklar: `np.gradient`, boxcar `uniform_filter1d(mode="nearest")` (başta ilk değer,
  sonda yalnız `flush` ile son değer), `qrs = smoothgrad > 1.5·avggrad`, beg/end eşleme (ilk beg'den önceki
  end atılır, sonu gelmeyen beg atılır), `data = signal[beg:end]` içinde **en belirgin (prominence) yerel
  maksimum** (SciPy `find_peaks` plato kuralı), `peaks=[0]` başlangıcı (segment başından ilk 0,3 sn atılır),
  `peak − last > mindelay` açgözlü kural (kabul edilen tepe kesin).
- Bilinçli sapma: `min_len` = 0,4 · (bitişi son W sn içinde olan QRS adaylarının uzunluk ortalaması, mevcut
  dahil; en fazla M aday). Öneri W = 10 sn, M = 128 (ayarlanabilir). Nedensel; adayın bitişinde karar.
- İkinci küçük sapma: `max_qrs` (öneri 0,5–1,0 sn; fixture'larda segment uzunlukları ölçülüp seçilecek)
  üzerindeki adaylar tepe vermez (yine de ortalamaya girer) → sinyal tamponu ve gecikme sınırlı.
- Gecikme (512 Hz, varsayılanlar): avggrad için ileri bakış 25 + 191 + 1 = 217 örnek (~0,42 sn); tepe,
  QRS adayının bitişi + 217 örnekte onaylanır → tipik ~0,5–0,6 sn, üst sınır `max_qrs` + 0,43 sn.
- Durum boyutu (512 Hz): sinyal ring (~`max_qrs`+220 örnek) + |grad| ring 51 + smoothgrad ring 384 +
  aday ring M × (u64 bitiş, u32 uzunluk) → ~6–9 KiB/akış; 500 akış ≈ 3–5 MB.
- İndeksler `u64` (32 bit hedeflerde taşma yok).

### Kalanlar
1. `StreamPeakDetector` uygulaması + birim testleri.
2. `tests/stream_peaks.rs`: 1/32/512/rastgele paketleme = `stream_peaks` (bit düzeyinde aynı indeksler);
   `reset_at` sonrası = taze detektör + ofset; uçtan uca ham → `EcgStreamFilter` → detektör, çeşitli
   paketlemeler = aynı boru hattının tek çağrısı.
3. Doğruluk: `tests/data/neurokit_ecg.json` (3 × 20 sn, 512 Hz) + gürültü varyantları; NK `ecg_peaks` ile
   ±1 örnek ve ±50 ms eşleşme oranı, kaçan/fazla sayıları. Gürültü varyantları için NK referansı gerekirse
   `tools/neurokit_reference.py`'ye ekle (yalnız sentetik veri).
4. W, M, `max_qrs` değerlerini doğruluk ölçümüyle seç; gecikme ve durum boyutunu testle doğrula.
5. `cargo fmt`, `cargo clippy --all-targets` (varsayılan, `--no-default-features`,
   `--features serde,parallel,simd`), `cargo test`; MSRV 1.85 (let-chain yok), yeni bağımlılık yok, AVX yok.
6. CHANGELOG (Unreleased/0.2.1), README, bu notun Türkçe son hali (API, gecikme, durum, NK sayıları, sapmalar).
