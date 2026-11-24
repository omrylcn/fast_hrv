# Kapasite ölçümü: app-pipeline işçisi, kullanıcı başına CPU ve bellek

Tarih: 2026-11-24 · Sürüm: fast-hrv 0.3.0 (`master`) · Rust 1.100.0 · Benchmark: `benches/capacity.rs`

Soru: "Tek CPU çekirdeğinde 500 eşzamanlı oturum" gerçekçi mi?

**Kısa cevap: evet, geniş payla, ama 5 dakikalık adım sıraya yayılmalı.** Bu makinede 500 kullanıcı
ortalama tek çekirdeğin **%2,6**'sını kullanıyor (kullanıcı başına sinyal saniyesi başına 51 µs).
Kullanıcıların hepsi 5 dakikalık adıma aynı saniyede girerse o saniye **1,6 s CPU** tutuyor (bütçe
1 s), yani yayma (stagger) gerekli. Sunucu CPU'su farklı ve eski olduğu için sayılar **sunucuda yeniden
ölçülmeli** (komut aşağıda); burada önerilen güvenlik çarpanı 5 ile de 500 kullanıcı çekirdeğin
~%13'ü ediyor.

## 1. Yöntem

### Ölçülen boru hattı (kullanıcı başına)
512 Hz tek derivasyon EKG, 32 örneklik paketler, tek işçi iş parçacığı, oturumlar sırayla işleniyor.

| Adım | Ne zaman | Ne yapılıyor (fast-hrv çağrısı) |
|---|---|---|
| 1 | her paket (16/sn) | ham ADC sayımı → `f64`, `EcgStreamFilter` (varsayılan 0,5–40 Hz: HP order 2 + LP order 4, nedensel, paketler arası durum), 15 sn'lik halka tampona yazma (temiz `f64` + ham `i32`) |
| 2a | her 10 sn | 10 sn çekirdek + her yanda 2 sn pay = 14 sn parça; `dual_detect` (Elgendi + inceltme, NeuroKit2 yöntemi + inceltme, bSQI); çekirdekteki tepeler 5 dk'lık tepe geçmişine eklenir |
| 2b | her 10 sn | 10 sn çekirdekte tam `SqiEngine::report`: düz hat + satürasyon (ham sayımlarda), kSQI, tek PSD'den pSQI + basSQI (Welch 2048), şablon korelasyonu, bSQI, RR olabilirliği, zhao2018 |
| 3a | her 60 sn | Kubios `fix_peaks_default`, son 5 dk'nın tepeleri (~300–500 tepe) |
| 3b | her 60 sn | `rri_from_peaks` + `time_domain`, son 60 sn |
| 4 | her 5 dk | `rri_from_peaks` + tam `hrv()` (zaman + frekans + doğrusal olmayan + asimetri), son 5 dk |

Kubios'un "biriken tepeler" üzerinde son 5 dk ile çalıştırılması kötümser seçim (5 dk'lık `hrv()` aynı
düzeltilmiş tepeleri kullanıyor). Daha kısa geçmişle 3a daha ucuz olur.

### Sentetik veri (gerçek veri yok)
Her kullanıcı için örnek örnek üretilen sentetik EKG (`Synth`, bench içinde):
- KH 60–100 atım/dk (kullanıcıya göre), solunum (0,22–0,30 Hz, ±%5) ve 0,1 Hz (±%3) RR modülasyonu,
  %1,5 RR gürültüsü; P-Q-R-S-T Gauss dalgaları, R genliği 0,8–1,6 mV, her 13. kullanıcıda ters derivasyon.
- Beyaz gürültü 0,01–0,04 mV, taban kayması, 50 Hz şebeke.
- Artefaktlar: atımların ~%1'i erken (ektopik) + telafi duraklaması, ~%0,25'i çizilmiyor (kaçan atım);
  her 1,5–4 dakikada bir 3 sn hareket artefaktı, 1,5 sn elektrot kopması (düz hat) ya da 1 sn ADC
  rayına dayanma (satürasyon).
- ADC: 1000 sayım/mV, 14 bit (−8192 … 8191).

Sağlama: 500 kullanıcı × 10 dk'da üretilen 300 261 atımın **%98,3**'ü (295 174) tespit edildi; kalan,
artefakt pencereleri ve henüz payı gelmemiş son pencereler.

### Simülasyon ve zaman ölçümü
- **Serpiştirilmiş filo:** N oturumun hepsi aynı işçide; paketler kullanıcılar arasında sırayla
  (paket 1: kullanıcı 0…N−1, paket 2: …) işleniyor. Böylece önbellekler gerçek işçide olduğu gibi her
  oturumun durumunu görüyor. Sinyal üretimi zaman ölçümünün dışında.
- İki takvim:
  - **hizalı:** bütün oturumlar t = 0'da başlıyor; 10 sn / 60 sn / 5 dk adımları aynı saniyeye düşüyor
    (en kötü durum). 600 sn.
  - **yayılmış:** oturum `u`, `floor(u · 300 / N)` saniyesinde başlıyor; ölçüm 300–600 sn arası (hepsi
    sürekli rejimde).
- Ayrıca 1 kullanıcı (sıcak önbellek) karşılaştırma için.
- Adım başına süre `Instant` ile tek iş parçacığında duvar saati (çağrı başına; filtre saniyelik
  bloklar halinde). Linux'ta süreç CPU/duvar oranı 1,00 çıktı (önceden alınma yok).
- Kullanıcı başına saniyelik maliyet = adımın ortalama çağrı süresi × çağrı sıklığı (16/sn, 1/10,
  1/60, 1/300).
- Bellek: oturum yapısının kapasiteleri (hesap) ve sayan bir global ayırıcı (ölçüm). Yeni bağımlılık yok.

### Makine
| | |
|---|---|
| CPU | Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz (Coffee Lake, 8 çekirdek, SMT yok, L2 256 KiB/çekirdek, L3 12 MiB) |
| Saat | turbo açık (`intel_pstate`, `no_turbo = 0`, governor `powersave`); ölçüm sırasında çekirdek **~4,6–4,7 GHz** (maks. 4,9 GHz) |
| Çekirdek | `taskset -c 3` ile **çekirdek 3**'e sabitlendi, tek iş parçacığı |
| Sistem | Linux 6.14, Rust 1.100.0, `release` profili; masaüstü makine (yük ortalaması ~1,8, ölçüme gürültü katabilir) |

### AVX
- Varsayılan hedef (genel x86-64) ve ayrıca açıkça `RUSTFLAGS="-C target-cpu=x86-64"` ile çalıştırıldı.
  Bench başlığı derleyicinin varsaydığı özellikleri basıyor: `sse2 | build assumes AVX: no`.
- `objdump -d` ile iki bench ikilisinde de **VEX kodlu (AVX) komut sayısı 0**, `%ymm` kullanımı yok.
- İki derleme aynı sonucu verdi (toplam 51,12 ve 51,09 µs/kullanıcı/sn).
- Yalnız karşılaştırma için bir kez `target-cpu=native` (AVX2 + FMA) denendi: 50,01 µs, yani **AVX'in
  katkısı ~%2**. Sunucuda AVX olmaması bu iş yükünü yavaşlatmaz; fark saat hızı ve mikro mimariden gelir.

## 2. Adım başına maliyet

500 kullanıcı, serpiştirilmiş, yayılmış takvim (225 350 kullanıcı-saniye), varsayılan hedef:

| Adım | Çağrı | Ort. µs/çağrı | p99 µs | Maks µs | µs / kullanıcı / sn | Pay |
|---|---:|---:|---:|---:|---:|---:|
| 1 `EcgStreamFilter` (32 örneklik paket) | 3 605 600 | 0,25 | – | – | 3,94 | %7,7 |
| 2a `dual_detect`, 14 sn pencere | 22 210 | 165 | 206 | 323 | 16,55 | %32,4 |
| 2b `SqiReport` (bütün endeksler), 10 sn | 22 210 | 170 | 208 | 313 | 17,02 | %33,3 |
| 3a Kubios `fix_peaks`, son 5 dk | 3 510 | 336 | 884 | 1 201 | 5,61 | %11,0 |
| 3b zaman alanı, son 60 sn | 3 510 | 6,9 | 20 | 48 | 0,12 | %0,2 |
| 4 tam `hrv()`, son 5 dk | 502 | 2 369 | 3 628 | 3 793 | 7,90 | %15,4 |
| **Toplam** | | | | | **51,1** | %100 |

- Maliyetin üçte ikisi 10 sn'lik adımda (iki detektör + SQI). Paket filtresi önemsiz (%8).
- Kubios'un p99'u ortalamanın 2,6 katı: artefaktlı geçmişte yinelemeli düzeltme birkaç tur dönüyor.
- `elgendi_refined_windowed` (yalnız birincil detektör) 14 sn pencere başına ~93 µs tutar
  (`benches/throughput.rs`: 30 pencere 2,80 ms), 2a'nın ~%56'sı, ama bSQI vermez; planlanan boru hattı
  `dual_detect` kullanıyor, ölçülen de o.

## 3. Bir çekirdekte kaç kullanıcı

| Çalıştırma | µs / kullanıcı / sn | %100 çekirdek | %50 bütçe |
|---|---:|---:|---:|
| 500 kullanıcı, serpiştirilmiş, yayılmış | 51,1 | ~19 500 | **~9 800** |
| 500 kullanıcı, serpiştirilmiş, hizalı | 55,8 | ~17 900 | ~9 000 |
| 1 kullanıcı (sıcak önbellek) | 50,0 | ~20 000 | ~10 000 |

- 500 kullanıcı = ortalama **26 ms CPU / sn** = tek çekirdeğin **%2,6**'sı (bu makinede).
- 500 oturumun önbellek etkisi küçük (sıcak önbelleğe göre +%2, hizalıda +%12).
- **Sunucu için güvenlik çarpanı 5** (aşağıda §6) ile: ~255 µs/kullanıcı/sn → %50 bütçede ~2 000
  kullanıcı; 500 kullanıcı çekirdeğin ~%13'ü. Hedefin (500) 4 katı pay kalıyor.

## 4. Bellek

| Kalem | Boyut | Not |
|---|---:|---|
| Filtre durumu (`EcgStreamFilter`) | ~0,2 KiB | 3 SOS bölümü × (katsayı + başlangıç + durum); çentik açılırsa +1 bölüm |
| Temiz sinyal halka tamponu, 15 sn `f64` | 60 KiB | 14 sn pencere + 1 sn gecikme payı |
| Ham ADC halka tamponu, 15 sn `i32` | 30 KiB | düz hat / satürasyon için (`i16` ile 15 KiB) |
| 5 dk tepe geçmişi + Kubios çıktısı | ~8 KiB | ~500 + ~500 `usize` |
| **Oturum başına toplam** | **98 KiB** | ölçülen oluşturma anı yığını 94,5 KiB |
| **500 oturum** | **~48 MiB** | |
| `SqiEngine` (işçi başına bir tane, oturum başına değil) | 113 KiB | FFT planı + Hann penceresi; veri tutmaz |
| İşçi karalama + geçici tepe (1 kullanıcılık koşu) | ~440 KiB | pencere kopyası, `hrv()` ara dizileri; işçi sıralı olduğu için kullanıcı sayısından bağımsız |

5 dk ham **örnek** tutulmuyor; 5 dk'lık adım yalnız tepelere ihtiyaç duyuyor. Örnekler tutulsaydı
kullanıcı başına `f64` ile 1 200 KiB (500 kullanıcı ~590 MiB), `i16` ile 300 KiB (~150 MiB) olurdu.
Gerekirse temiz tampon `f32`'ye indirilebilir (oturum ~68 KiB), ama gerek yok.

## 5. Ani yük (spike) analizi ve yayma önerisi

İşçinin her duvar saati saniyesinde 500 kullanıcı için harcadığı CPU (bütçe 1000 ms):

| Takvim | Ortalama ms/sn | En kötü saniye |
|---|---:|---:|
| hizalı (hepsi t = 0'da başlıyor) | 27,6 | **1 601 ms** (t = 600 sn; t = 300 sn de > 1000 ms) |
| yayılmış (başlangıçlar 5 dk'ya dağılmış) | 26,2 | **31 ms** |

- 5 dk'lık adım (Kubios + zaman alanı + `hrv()`) kullanıcı başına ortalama **2,95 ms**, p99 **4,9 ms**.
  500 kullanıcı aynı saniyede: **~1,5 s** (p99 ile ~2,5 s). Sunucuda çarpan 5 ile **~7 s**: işçi o kadar
  geride kalır, bu sürede canlı uçların 10 sn SQI'si ve paket filtrelemesi de gecikir.
- Hizalı takvimde 10 sn adımları ayrı saniyelere düşüyor (pencere hazır olunca, t = 12, 22, … sn);
  500 × 0,34 ms ≈ 170 ms/sn. Tek başına sorun değil ama sunucuda ~0,85 s'ye yaklaşır.
- Yayılmış takvimde saniyede ~1,7 kullanıcı 5 dk adımına, ~50 kullanıcı 10 sn adımına giriyor;
  en kötü saniye ortalamanın sadece 1,2 katı.

**Öneri: yayma gerekli.**
1. Ağır adımlar (60 sn, 5 dk) oturumun kendi başlangıcına bağlı olsun, **duvar saati sınırlarına
   (her dakikanın / her 5 dakikanın başı) değil**. Analiz pencereleri raporlama için duvar saatine
   hizalı olmak zorundaysa, pencerenin **hesaplanma zamanı** oturuma özgü sabit bir gecikmeyle kaydırılsın:
   `gecikme = hash(session_id) mod 300 sn` (60 sn adımı için `mod 60`). Sonuç değişmez (fast-hrv
   pencereleri mutlak örnek ızgarasına bağlı ve deterministik: canlı = geçmiş), yalnız ne zaman
   hesaplandığı değişir.
2. Doğal yayılma yeterli değil: işçi yeniden başlatma, sunucu açılışı, ağ kesintisi sonrası toplu
   yeniden bağlanma ya da birikmiş kuyruk aynı anda yüzlerce oturumu hizalayabilir. Bu yüzden 60 sn /
   5 dk işleri **öncelikli bir kuyruktan** alınsın: önce paket filtresi ve 10 sn pencereleri (canlı uçlar),
   sonra ağır işler saniyede bir bütçeyle (örneğin ≤ 100 ms/sn). 500 kullanıcının 5 dk işi ortalama
   ~5 ms/sn tuttuğu için bu bütçe birikmeyi birkaç saniyede eritir.
3. 10 sn pencereleri için de aynı kural: oturum başlangıcına bağlı olduklarından doğal olarak yayılırlar;
   yeniden bağlanma fırtınasında kuyruk aynı şekilde korur.

## 6. Sunucu farkı ve güvenlik çarpanı

Sunucu CPU'su farklı ve eski, AVX yok. Bu ölçüm onun yerine geçmez; **sayı sunucuda yeniden
ölçülmeli.** Tahmin:
- AVX'in kendisi etkisiz (§1: native/AVX2 derleme sadece %2 hızlı). Fark saat hızı ve IPC'den gelir.
- Bu ölçüm turbo altında (~4,6–4,7 GHz). Temel saatte (3,6 GHz) aynı kod ~1,3 kat yavaş olur.
- AVX'siz olası sunucu tipleri ve kaba çarpanları (bu makineye göre):
  AVX'i gizlenmiş sanal makine (modern ana makine, `kvm64`/uyumluluk modu): ~1,2–2×;
  Pentium/Celeron (Skylake–Comet Lake kuşağı, AVX kapalı): ~1,3–1,8×;
  Nehalem/Westmere Xeon (2009–2011, 2,4–3,0 GHz): ~2–2,5×; Atom/Celeron J/N serisi: ~3–4×.
- Ölçülmeyenler: veritabanı okuma/yazma, paket çözme, serileştirme, Windows yığın ayırıcısı (adımlar
  her çağrıda küçük ara diziler ayırıyor), aynı çekirdeği paylaşan başka süreçler.
- **Planlama için güvenlik çarpanı 5:** 500 kullanıcı ≈ 128 ms/sn (%13 çekirdek); yayılmış en kötü
  saniye ≈ 155 ms; hizalı 5 dk adımı ≈ 7 s (yayma olmadan kabul edilemez).

Sonuç: 500 eşzamanlı oturum tek çekirdekte gerçekçi; fast-hrv'nin hesap maliyeti bütçenin küçük bir
kısmı. Asıl risk ani yük; yayma + öncelikli kuyruk ile çözülür. İşçinin geri kalanı (G/Ç, veritabanı)
muhtemelen hesaptan pahalı olacak, ayrıca ölçülmeli.

## 7. Sunucuda yeniden üretme

Tek komut, repo kökünde (`rust-toolchain.toml` 1.100.0'ı seçer):

```
cargo bench --bench capacity
```

Varsayılan: 500 kullanıcı × 600 sn. Bu makinede ~35 sn sürüyor (derleme hariç); sunucuda birkaç dakika
beklenmeli. Seçenekler: `cargo bench --bench capacity -- --users 1000 --seconds 900`.

Çıktı küçük tablolar halinde: adım başına maliyet, çekirdek başına kullanıcı (%100 / %50), saniyelik
CPU (hizalı / yayılmış, en kötü saniye), bellek. Başlıktaki satırı kontrol edin:
`build assumes AVX: no | CPU supports AVX: no` olmalı.

AVX varsayılmadığını açıkça göstermek için (sonuç aynı olmalı; `RUSTFLAGS` değişince her şey yeniden
derlenir):

```
# Windows PowerShell
$env:RUSTFLAGS="-C target-cpu=x86-64"; cargo bench --bench capacity; Remove-Item Env:RUSTFLAGS
# Windows cmd
set "RUSTFLAGS=-C target-cpu=x86-64" && cargo bench --bench capacity
# Linux
RUSTFLAGS="-C target-cpu=x86-64" cargo bench --bench capacity
```

Tek çekirdeğe sabitlemek (önerilir, gürültüyü azaltır):

```
# Windows cmd (maske onaltılık: 4 = çekirdek 2)
start /b /wait /affinity 4 cargo bench --bench capacity > capacity.txt
# Linux
taskset -c 2 cargo bench --bench capacity | tee capacity.txt
```

Windows'ta CPU modeli `PROCESSOR_IDENTIFIER`'dan basılır (aile/model kodu); tam ad ve saat için:
`Get-CimInstance Win32_Processor | Select-Object Name, MaxClockSpeed, CurrentClockSpeed`.
Sonucu (`capacity.txt`) ve CPU adını locaya yazın; bu belgedeki tabloları sunucu sayılarıyla güncelleriz.
