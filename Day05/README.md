# Ahmet Çimen 07/24/2026 Çalışma Notları


## 1. Ray Koridoru Segmentasyonu ve Veri Seti Hazırlığı

Bugün, trenin hareket doğrultusunda üzerinde ilerleyeceği ray koridorunu hassas piksel düzeyinde segmente etmek amacıyla **RailSem19** veri seti üzerinde gelişmiş bir maske üretim ve ön işleme süreci gerçekleştirdim.

Çalışmada, yalnızca görüntüdeki tüm rastgele raylar değil, doğrudan trenin üzerinde bulunduğu aktif ray çifti ve bu çift arasındaki sürüş koridoru hedeflenmiştir.

- **Sınıf Tanımları:**
  - `rail-track` (ID: 12) — İki ray arasındaki ana sürüş koridoru.
  - `tram-track` (ID: 3) — Tramvay sürüş koridoru.
  - `rail-raised` (ID: 17) & `rail-embedded` (ID: 18) — Yükseltilmiş ve gömülü metal ray hatları.
- **Veri Bölme & Çözünürlük:** Toplam ~8,500 yüksek çözünürlüklü görselin **%70'i eğitim**, **%15'i doğrulama** ve **%15'i test** seti olarak ayrıldı. Görüntüler ve maskeler **640×640** piksel standart çözünürlüğe boyutlandırıldı.

---

## 2. Derin Öğrenme Model Mimarileri ve Kayıp Fonksiyonu

Bu çalışmada segmentasyon için **DeepLabV3+ (ResNet-50)** ve daha hafif bir alternatif olarak **DeepLabV3+ (MobileNetV2)** mimarileri denenmiştir. Denemelerde MobileNet tabanlı model yetersiz temsil kapasitesi ve başarısız segmentasyon sonuçları verdiği için değerlendirmeye tabi tutulmamış, performans ölçümleri ve testlerde ana model olarak **DeepLabV3+ (ResNet-50)** kullanılmıştır.

| Model Mimarisi | Backbone Encoder | Parametre Sayısı | Hedef Kullanım Senaryosu | Durum / Değerlendirme |
| :--- | :--- | :--- | :--- | :--- |
| **DeepLabV3+** | ResNet-50 | ~26M | Yüksek doğruluklu ana segmentasyon modeli | **Başarılı** (Değerlendirmeye ve benchmark'a tabi tutuldu) |
| **DeepLabV3+** | MobileNetV2 | ~3.5M | Hafif mobil ve gömülü cihaz alternatifi | **Başarısız** (Yetersiz segmentasyon nedeniyle değerlendirmeye tabi tutulmadı) |

Eğitim sürecinde piksel bazlı kararlılığı artırmak ve sınıf dengesizliği sorununu aşmak için Dice Loss, Binary Cross Entropy (BCE) Loss ve Boundary Loss kombinasyonundan oluşan **Hibrit Kayıp Fonksiyonu** uyguladım:

- **Dice Loss:** İnce ve uzun ray çizgilerindeki arka plan ve ön plan sınıf dengesizliğine karşı optimizasyonu stabilize eder.
- **BCE Loss:** Piksel seviyesinde ikili sınıflandırma kararlılığı sağlar.
- **Boundary Loss:** Ray sınırlarındaki gradyanları dikleştirerek sınır çizgi keskinliğini artırır.

---

## 3. Bölgesel IoU ve Perspektif Derinlik Metrikleri

Kamera açısı nedeniyle uzak mesafedeki piksel alanlarının azlığı standart IoU metriğinin yanıltıcı olmasına yol açabilmektedir. Bu durumu çözmek adına derinlik perspektifine özel bölgesel metrik hesaplama mantığı kurguladım:

- **$\text{IoU}_{\text{yakın}}$:** Görüntünün alt %35'lik alanı (Trenin anlık frenleme ve acil durum bölgesindeki IoU başarımı).
- **$\text{IoU}_{\text{orta}}$:** Görüntünün orta %30'luk alanı (Orta mesafe hat takibi IoU başarımı).
- **$\text{IoU}_{\text{uzak}}$:** Görüntünün üst %35'lik alanı (Uzak görüş ufkundaki hat tespiti IoU başarımı).
- **Boundary F1 Score:** Ray sınır hatlarının piksel toleransı içerisindeki sınır keskinliği doğruluğu.

---

## 4. Gerçek Zamanlı Saf GPU İnference ve Benchmark Altyapısı

Modelin lokal bilgisayarda canlı video akışlarında yüksek FPS ve düşük gecikmeyle (latency) koşturulabilmesi amacıyla saf GPU üzerinde çalışan çıkarım ve benchmark altyapısını geliştirdim.

- **Kullanıcı Arayüzü:** Dosya seçim arayüzü eklenerek kullanıcının istediği model ağırlıklarını ve işlenecek video dosyasını dinamik olarak seçebilmesi sağlandı.
- **Saf GPU İşleme (Zero Disk I/O):**
  - Kamera karesinin GPU bellek alanına aktarılması, BGR-RGB dönüşümü, normalizasyon (0.0–1.0) ve boyutlandırma adımları tamamen NVIDIA CUDA çekirdekleri üzerinde gerçekleştirildi.
  - PyTorch **AMP FP16 (Half Precision)** ivmelendirmesi ve cuDNN benchmark modülü aktif edilerek GPU bellek bant genişliği ve Tensor Çekirdeği kullanımı maksimuma çıkarıldı.
  - GPU ön ısınma (warmup) döngüsü eklenerek ilk kare gecikmeleri elendi.
- **Görsel Kaplama & Kontur Çizimi:** Tahmin edilen ikili maske orijinal çözünürlüğe GPU üzerinde ölçeklendi; ray koridoru yarı-saydam renk kaplaması ve sarı kontur çizgisi ile belirginleştirildi.
- **Metrik Loglama:** Otomatik olarak kare başı milisaniye (ms) gecikme, FPS istatistikleri (Mean, Std, Min, Max) ve ekran görüntüleri kaydedilip loglandı.

---

## 5. Örnek Benchmark ve Çıkarım Sonuçları

Geliştirilen GPU çıkarım pipeline'ı ile NVIDIA GeForce RTX 3050 GPU ortamında elde edilen canlı test ve benchmark sonuçları aşağıda özetlenmiştir *(DeepLabV3+ MobileNetV2 modeli segmentasyonda başarısız olduğu için değerlendirmeye tabi tutulmamış, benchmark yalnızca **DeepLabV3+ ResNet-50** modeli üzerinde yapılmıştır)*:

| Metrik / Parametre | Değer |
| :--- | :--- |
| **Model Mimarisi** | DeepLabV3+ (ResNet-50) |
| **Çıkarım Hassasiyeti** | Mixed Precision FP16 (CUDA AMP) |
| **Giriş Çözünürlüğü** | 768×768 piksel |
| **GPU Model Çıkarım Gecikmesi** | ~47.5 ms |
| **GPU Model Çıkarım Hızı** | ~21.3 FPS |
| **Sistem Ekran Akış Gecikmesi** | ~70.4 ms |
| **Sistem Ekran Akış Hızı** | ~14.5 FPS |

---

## 6. Hedefler

Önümüzdeki günlerde sistem başarımını ve hızı daha ileriye taşımak için aşağıdaki çalışmaların yürütülmesi planlanmaktadır:

1. **YOLOv8 Nano Segmentasyon Entegrasyonu:** DeepLabV3+ modeline alternatif olarak çok daha hafif ve hızlı olan YOLOv8 Nano (`yolov8n-seg.pt`) segmentasyon modelinin RailSem19 veri seti üzerinde eğitilmesi.
2. **Google Colab T4 Üzerinde 50 Epoch Eğitim:** YOLOv8 Nano modelinin 6 hedef sınıf (`background`, `rail-raised`, `rail-track`, `trackbed`, `rail-embedded`, `tram-track`) ile Colab T4 GPU üzerinde eğitilmesi.
