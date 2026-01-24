# HR ve HRV 101: Otonom Sinir Sistemi ve Kalp Ritmi Rehberi

Kalp hızı (HR) ve kalp hızı değişkenliği (HRV), otonom sinir sisteminin (OSS) vücudun iç ve dış dünyaya nasıl uyum sağladığını gösteren en hassas pencerelerdir. Geleneksel inanışın aksine, sağlıklı bir kalp bir metronom gibi düzenli atmaz; aksine, sağlıklı biyolojik sistemler matematiksel kaos ile tarif edilebilen karmaşık değişkenlik modelleri sergiler.

## 1. Temel Fizyoloji: Kalp Neden Değişken Atar?

Kalbin ritmi, sağ atriyumda bulunan ve "doğal pacemaker" (kalp pili) olarak adlandırılan **Sinoatriyal (SA) nod** tarafından başlatılır. SA nod, otonom sinir sisteminin iki ana dalından sürekli ve dinamik etkiler alır:

* **Sempatik Sinir Sistemi (SNS):** "Savaş veya Kaç" mekanizmasıdır. Kalp hızını artırır ve vücudu strese hazırlar. Tepki süresi nispeten yavaştır (yaklaşık 5 saniyeden uzun).


* **Parasempatik Sinir Sistemi (PNS):** "Dinlen ve Sindir" mekanizmasıdır. Vagus siniri üzerinden kalbi yavaşlatır. Tepki süresi çok hızlıdır (0.5 saniye altı), bu sayede kalbi atımdan atıma (beat-to-beat) kontrol edebilir.

**HRV Nedir?** İki kalp atımı arasındaki sürenin (R-R aralığı) milisaniye (ms) cinsinden dalgalanmasıdır. Atımlar arası süreler (örneğin 850 ms, 1050 ms, 900 ms) ne kadar değişken ise, vücut o kadar esnek ve adaptasyon kabiliyeti yüksek kabul edilir.

## 2. HRV Analiz Metrikleri

HRV verileri üç ana kategoride incelenir: Zaman Alanı, Frekans Alanı ve Non-lineer (Doğrusal Olmayan) yöntemler.

### A. Zaman Alanı (Time-Domain)

Bu metrikler, R-R aralıklarındaki ham değişkenliği ölçen en temel istatistiksel değerlerdir.

* **SDNN (ms):** Normal atımlar arası sürelerin standart sapmasıdır. Toplam otonomik kapasiteyi ve genel değişkenliği yansıtır.
* **RMSSD (ms):** Ardışık atımlar arasındaki farkların karelerinin ortalamasının kareköküdür. **Parasempatik (vagal) aktivitenin en güvenilir zaman alanı göstergesidir**.
* **pNN50 (%):** Ardışık atımlar arasında 50 ms'den fazla fark olanların tüm atımlara oranıdır.

### B. Frekans Alanı (Frequency-Domain)

Sinyali frekans bileşenlerine ayırarak (Güç Spektral Yoğunluğu - PSD) otonomik dengenin izini sürer.

* **HF (Yüksek Frekans):** 0.15 – 0.40 Hz arası. Doğrudan parasempatik aktivite ve solunum döngüsü (RSA) ile ilgilidir.


* **LF (Düşük Frekans):** 0.04 – 0.15 Hz arası. Barorefleks aktivitesini, sempatik ve parasempatik sistemlerin karışımını yansıtır.


* **VLF (Çok Düşük Frekans):** 0.0033 – 0.04 Hz arası. Termoregülasyon, Renin-Anjiyotensin sistemi ve uzun dönemli hormonal süreçlerle ilişkilidir.



## 3. İleri Düzey Yöntemler ve Karmaşıklık Analizi

Basit oranların (LF/HF gibi) yetersiz kaldığı durumlarda sistemin "sağlık kalitesini" ölçmek için daha sofistike yöntemler kullanılır.

* **Non-lineer Analizler:**
* **Sample Entropy (SampEn):** Sinyalin ne kadar düzensiz ve öngörülemez olduğunu ölçer. Sağlıklı sistemler daha yüksek entropiye (düzensizliğe) sahiptir.
* **DFA (Detrended Fluctuation Analysis):** Sinyalin fraktal yapısını ölçer.  değerinin 1.0 civarında olması sağlıklı bir öz-benzerlik işaretidir.


* **Kalp Hızı Fragmentasyonu (HRF):** Kalp hızındaki ivme değişimlerinin (hızlanma/yavaşlanma) ne kadar sık yön değiştirdiğini ölçer (PIP, IALS gibi metrikler). Yüksek fragmentasyon, vagal kontrolün bozulduğunun ve sistemin "parçalandığının" bir işaretidir.

## 4. Ölçüm Teknolojileri: ECG vs PPG

* **ECG (Elektrokardiyografi):** Kalbin elektriksel sinyalini (R-tepesi) doğrudan ölçer. Milisaniye hassasiyetinde en yüksek doğruluğu sunan altın standarttır.
* **PPG (Fotopletismografi):** Akıllı saatlerdeki yeşil ışık teknolojisidir. Kan hacmindeki değişiklikleri optik olarak algılar. İstirahat halinde başarılı olsa da, hareket sırasında ve kısa süreli ölçümlerde hata payı yüksektir.

## 5. Pratik Uygulama ve Kritik İpuçları

Doğru bir HRV takibi için şu standartlara uymak kritiktir:

* **Zamanlama:** En ideal ölçüm zamanı sabah uyandıktan hemen sonradır (yemek, kahve veya stresten önce).
* **Pozisyon:** Daima aynı pozisyonda ölçüm yapın. Kalp hızı 50 bpm altındaysa, yatar pozisyon yerine oturmak veya ayakta durmak daha güvenilir sonuçlar verir.
* **Solunum:** Kontrollü solunum (dakikada 6-9 nefes) LF gücünü yapay olarak artırabilir. Doğal trendi görmek için spontane solunum tercih edilmelidir.


* **Dış Etkenler:** Alkol tüketimi, ağır egzersiz, dehidrasyon (susuzluk) ve hastalık HRV'yi dramatik şekilde düşürür. Bu etkilerin toparlanması 24-48 saat sürebilir.

**Özetle:** Yüksek HRV, vücudun strese karşı esnek ve toparlanmış olduğunu; düşük HRV ise sistemin yorgun veya bir tehdit (stres, hastalık, inflamasyon) altında olduğunu gösterir. Ancak HRV kişiseldir; rakamları başkalarıyla değil, kendi geçmiş trendlerinizle kıyaslamanız esastır.
