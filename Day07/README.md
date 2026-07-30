# Günlük Çalışma Raporu — Gün 7 (Salı)

## 1. Google Colab 50 Epoch Çoklu Sınıf Eğitim Altyapısı
Bugün, YOLOv8 Nano segmentasyon modelini daha kapsamlı bir çevre algılama yeteneğine kavuşturmak amacıyla RailSem19 veri setindeki tüm hedef sınıfları kapsayan çoklu sınıf eğitim sürecini başlattım.

- **Veri Seti Dönüştürme Algoritması:** RailSem19 veri setindeki semantik maskeleri ve poligon geometrilerini YOLOv8 segmentasyon formatına dönüştüren modüler veri hazırlık mimarisini tasarladım.
- **Hedef Sınıf Yapılandırması (6 Sınıf):**
  - `0 — background` (Arka plan)
  - `1 — rail-raised` (Yükseltilmiş metal ray)
  - `2 — rail-track` (Ray sürüş koridoru)
  - `3 — trackbed` (Balast ve zemin yatağı)
  - `4 — rail-embedded` (Gömülü ray)
  - `5 — tram-track` (Tramvay hattı)
- **Bulut GPU Eğitimi:** Google Colab T4 GPU ortamında 50 epoch boyunca yüksek çözünürlüklü eğitim yürütecek eğitim notebook altyapısı hazırlandı.

---

## 2. Sabit ve Bağımsız Test Veri Seti İzolasyonu
Farklı mimarilere (DeepLabV3+, YOLOv8 2-Sınıf, YOLOv8 6-Sınıf) sahip modelleri adil ve bilimsel ilkeler doğrultusunda karşılaştırabilmek adına bağımsız bir test kümesi oluşturdum.

- **Veri Sızıntısını (Data Leakage) Önleme:** Tüm eğitim süreçlerinden tamamen izole edilmiş 100 adet yüksek çözünürlüklü test görseli ve bu görsellere ait orijinal semantik etiket maskeleri sabit bir test dizininde kilitlendi.
- **Kilitli Test:** Hiçbir model eğitilirken bu görselleri görmediğinden, elde edilen metriklerin gerçek dünya şartlarındaki genelleme başarısını yansıtması sağlandı.

---

## 3. Sınıf Bazlı IoU (Class-wise IoU) ve Adil Değerlendirme Mantığı
Farklı sınıf sayılarına sahip modellerin tek bir genel ortalama skoru (mIoU) üzerinden değerlendirilmesinin adil olmadığını tespit ettim. 

- **Sınıf Bazlı Ayrıştırma:** Tespiti zor veya ince yapılı sınıfların (`rail-raised`, `rail-embedded`) genel ortalamayı düşürerek ana koridor başarımını gölgelemesini önlemek amacıyla her sınıf için bağımsız IoU hesaplama altyapısını kurdum.
- **Ortak Payda Karşılaştırması:** Tüm modeller arasında ortak olan **`rail-track` (sürüş koridoru)** başarımı doğrudan yan yana kıyaslanabilir hale getirildi.

---

## 4. Otomatik Liderlik Tablosu (Leaderboard) ve Log Altyapısı
Tüm modellerin çıkarım hızlarını ve doğruluk metriklerini tek bir merkezde toplayan modüler test ve değerlendirme aracını geliştirdim.

- **Benzersiz Log Kaydı:** Çalıştırılan her test için model adına özel versiyonlu log dosyaları oluşturularak tüm deneysel geçmiş korundu.
- **Merkezi Liderlik Tablosu:** Model adı, GPU çıkarım FPS'i, GPU gecikmesi (ms), ekran gösterim FPS'i ve sistem gecikmesi değerleri otomatik olarak merkezi bir özet tablosuna işlendi.

---

## 5. Model Karşılaştırma ve Benchmark Sonuçları

Görülmemiş test veri seti üzerinde yapılan detaylı sınıf bazlı IoU ve hız analiz sonuçları aşağıda özetlenmiştir:

### YOLOv8 Çoklu Sınıf Modelinin Sınıf Bazlı IoU Başarımı:

| Sınıf Adı | Açıklama | Sınıf IoU Skoru |
| :--- | :--- | :---: |
| **`rail-track`** | Ray Arası Sürüş Koridoru | **%64.68** |
| **`tram-track`** | Tramvay Koridoru | **%58.01** |
| **`trackbed`** | Balast ve Zemin Yatağı | **%54.21** |
| **`rail-raised`** | Yükseltilmiş Metal Ray | **%22.84** |
| **`rail-embedded`** | Zemine Gömülü Ray | **%17.27** |
| **Genel mIoU** | **Tüm Sınıfların Ortalaması** | **%43.40** |

---

### Genel Model Performans Karşılaştırma Tablosu:

| Model Mimarisi | Sınıf Sayısı | GPU Inference FPS | GPU Inference Delay (ms) | IoU (`rail-track`) |
| :--- | :---: | :---: | :---: | :---: |
| **DeepLabV3+ (ResNet-50)** | 1 Sınıf | 23.98 ± 1.10 | 41.82 ms | **%68.60** |
| **YOLOv8 Nano (2 Sınıflı)** | 2 Sınıf | 35.57 ± 3.42 | 28.73 ms | %58.51 |
| **YOLOv8 Nano (6 Sınıflı)** | 6 Sınıf | 32.97 ± 3.18 | 31.11 ms | %64.68 |

---

## 6. Gelecek Planlar
1. **Model Optimizasyonu:** Çoklu sınıf modelinde `rail-raised` gibi ince sınıfların IoU başarısını artırmak için kenar ağırlıklı kayıp fonksiyonlarının entegrasyonu.
2. **ONNX Export & Kenar Cihaz Hazırlığı:** Eğitilen modellerin kenar cihazlarda (NVIDIA Jetson / TensorRT) koşturulmak üzere ONNX formatına aktarılması.
