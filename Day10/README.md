# Günlük Çalışma Raporu — Gün 10

## 1-) Görüntü İşleme Tabanlı Ray Tespit Sistemleri (Sliding Window vs Hough Transform)

Demiryolu takip sistemlerinde kullanılmak üzere klasik görüntü işleme teknikleri ve matematiksel eğri uydurma yöntemlerini kıyaslayan, modüler ve grafik kullanıcı arayüzüne (GUI) sahip kapsamlı bir test ve analiz altyapısı geliştirilmiştir.

- **Ortak Ön İşleme Boru Hattı (Preprocessing Pipeline):**
  - **Region of Interest Maskesi (ROI):** Görüntünün yalnızca ray hatlarını barındıran alt trapezoid bölgesine odaklanılarak gereksiz gökyüzü ve çevre gürültüleri elenmiştir.
  - **Grayscale Dönüşümü ve CLAHE:** Farklı ışık, parlaklık ve gölge koşullarında ray çizgilerinin belirginliğini korumak amacıyla adaptif kontrast artırımı uygulanmıştır.
  - **Noise ve Motion Blur Filtreleme:** Kamera titreşimleri ve yüksek hızlı hareketten kaynaklanan görüntü bozulmalarını yumuşatmak için filtreleme teknikleri entegre edilmiştir.

- **Geliştirilen 3 Farklı Ray Tespit Mimarisi:**
  - **Kayan Pencere ve Bézier Eğrisi (Sliding Window + Bézier Curve):**
    - Özel konvolüsyon filtreleri (özel matris katsayıları) kullanılarak dikey ray kenarları ve ray başı profili vurgulanmıştır.
    - Görüntünün alt kısmından başlayarak yukarıya doğru adaptif kayan pencereler ile piksel yoğunluk merkezleri takip edilmiştir.
    - Elde edilen merkez noktalarına Bernstein polinomları tabanlı 3. derece Bézier eğri uydurma (Bézier Curve Fitting) uygulanmış, keskin virajlar ve eğri ray hatlarında yüksek stabilite elde edilmiştir.
  - **Hough Dönüşümü ve Kaçış Noktası Analizi (Hough Transform + Vanishing Point):**
    - Canny kenar algılama ve Region of Interest maskesi uygulanmıştır.
    - Olasılıksal Hough Çizgi Dönüşümü ile doğrudan çizgi segmentleri tespit edilmiştir.
    - Açısal filtreleme ile yatay gürültüler elenerek sol ve sağ ray çizgileri ayrıştırılmıştır.
    - Çizgilerin kesişim noktası hesaplanarak kaçış noktası (Vanishing Point) kestirilmiş ve düz ray hatlarında yüksek hız ve doğruluk sağlanmıştır.
  - **Hibrit Yaklaşım (Hybrid Approach):**
    - Hough Dönüşümü ile global ray yönü ve kaçış noktası kestirilmiş, Kayan Pencere yöntemi ile lokal detaylar ve eğrilikler ince ayardan (local refinement) geçirilmiştir.

- **Gelişmiş Grafik Kullanıcı Arayüzü (GUI) ve Canlı Metrik Takibi:**
  - Canlı video akışı üzerinde algılama yöntemi seçimi, ROI koordinatları, CLAHE, bulanıklık, Canny eşik değerleri ve kayan pencere boyutlarının anlık olarak ayarlanması sağlanmıştır.
  - Ön işleme adımlarının canlı takibi için boru hattı görüntüleme paneli eklenmiştir.
  - Gerçek zamanlı FPS, işlem süresi (ms) ve tespit güvenilirlik skoru (reliability metric) canlı HUD ekranına entegre edilmiştir.

- **Region of Interest Bağımlılığı ve Limitasyonlar:**
  - Modelin çizgileri tespit etmede başarılı olabilmesi için Region of Interest'in sadece rayları kapsayacak alan ile sınırlı kalması gerekmektedir. Bu yüzden sürekli Region of Interest değiştirmenin çok kullanışlı olmayacağı, Region of Interest raylardan fazla alanı kapsadığında ise sapmalardan başka bir şey görülmediği tespit edilmiştir.

---

## 2-) YOLOv8 ve Optik Akış Hibrit Segmentasyon Mimarisi (YOLO + Optical Flow Propagation)

YOLOv8 derin öğrenme segmentasyon modellerinin yüksek GPU kaynak tüketimini ve gecikmesini azaltmak, gömülü cihazlarda saniyedeki kare sayısını (FPS) artırmak amacıyla YOLOv8 ile Görüntü İşleme (Optik Akış / Kamera Hareketi) tabanlı hibrit bir segmentasyon ve takip altyapısı geliştirilmiştir.

- **Yaklaşımın Temel Amacı:** Asıl amaç YOLO modelini her frame'de çalıştırmak değil, bir frame çalıştırdıktan sonra bir süre boyunca o maskenin gelecek frame'lerde nerede olması gerektiğini tahmin eden matematiksel algoritmalar kullanarak yüksek FPS ile videoyu işlemektir.
- **Yöntemin Başarısızlığı ve Nedenleri:** Yapılan testlerde bu hibrit yaklaşımın başarısız olduğu görülmüştür. Başarısız olmasındaki temel sebep, tren dümdüz gitmediği için rayların ekrandaki konumunun sürekli değişmesi ve bir sonraki frame'deki konumu tahmin edilmeye çalışılan YOLO ile oluşturulan maskenin sürekli olması gerektiği konumdan kaymasıdır.

- **Döngüsel Hibrit Çıkarım Mimarisi:**
  - Belirlenen N karelik (örneğin 30 kare) döngüsel pencerede, yalnızca ilk X karede tam YOLOv8 segmentasyon çıkarımı çalıştırılarak yüksek doğruluklu referans maskeler üretilir.
  - Kalan (N - X) kare boyunca ağır derin öğrenme modeli yerine hafif optik akış ve hareket kestirimi yöntemleri çalıştırılarak maskeler sonraki karelere taşınır (propagation).
  - N kare tamamlandığında maske sıfırlanarak süreç tekrarlanır.

- **Desteklenen Hareket Kestirimi ve Takip Yöntemleri:**
  - **Farneback Yoğun Optik Akış (Farneback Dense Optical Flow):** Tüm piksel ızgarasının hareket vektörleri hesaplanarak maske akıcı şekilde kaydırılır.
  - **Lucas-Kanade Seyrek Optik Akış (Lucas-Kanade Sparse Flow):** Belirli köşe ve kenar noktaları takip edilerek maske pozisyonu güncellenir.
  - **ECC Kamera Hareketi Hizalaması (Enhanced Correlation Coefficient):** Kamera hareket matrisi hesaplanarak maske perspektif dönüşümüne uğratılır.
  - **Statik Maske Tutma (Static Hold):** Yüksek hızlı veya sabit sahnelerde son maskenin korunması sağlanır.

- **Canlı Kontrol ve Performans Arayüzü (Trackbar Interface):**
  - Grafik penceresi ve dinamik kaydırma çubuğu (Trackbar / Slider) ile 30 kare içerisindeki YOLO çıkarım sayısının (X: 1 ile 30 arası) canlı olarak değiştirilebilmesi sağlanmıştır.
  - Kullanıcı slider'ı kaydırdıkça sistem gecikmesi ve FPS üzerindeki değişim anlık HUD göstergesinde izlenebilmektedir.

- **Özel Segmentasyon Kayıp Fonksiyonları (Loss Functions) Tasarımı:**
  - **Dice-BCE Karma Kayıp Fonksiyonu:** Sınıf dengesizliğine dayanıklı Dice Loss ile piksel bazlı öğrenme kararlılığı sağlayan Binary Cross Entropy (BCE) birleştirilmiştir.
  - **Sınır-Farkındalıklı Kayıp (Boundary-Aware Loss):** Ray kenarlarının ve ince detayların daha keskin tespit edilebilmesi amacıyla aşınma (erosion) ve genişleme (dilation) matrisleri ile sınır bölgeleri çıkarılmış, bu piksellere 2.0 kat daha yüksek kayıp ağırlığı uygulanmıştır.
  - **Birleşik Kayıp Fonksiyonu (Combined Loss):** Tüm kayıp bileşenlerini adaptif ağırlıklarla harmanlayan bütünleşik model eğitim altyapısı kurulmuştur.

- **Sınıf Bazlı IoU ve Liderlik Tablosu Metrik Değerlendiricisi (Evaluator & Benchmark):**
  - Bağımsız kilitli test veri kümesi üzerindeki tüm görseller için `rail-track`, `rail-raised`, `trackbed`, `rail-embedded` ve `tram-track` sınıflarında sınıf bazlı IoU, Precision ve Recall metriklerinin hesaplanması altyapısı kurulmuştur.
  - Saf GPU çıkarım FPS/ms ve sistem ekran FPS/ms metrikleri ölçülerek otomatik olarak merkezi liderlik tablosuna kaydetme ve versiyonlu loglama mekanizması entegre edilmiştir.

---

## 3-) Gelecek Planlar ve Sonraki Adımlar

1. **Model Seçimi (YOLO ile Devam Edilmesi):** Yukarıda test edilen iki alternatif yaklaşım (klasik görüntü işleme ve optik akış tabanlı maske tahmini), rayların konumunun değişkenliği, maske kaymaları ve sınırlılıkları nedeniyle elenmiştir. Gelecek planlarda tek model olarak **YOLO** ile devam edilmesi planlanmaktadır.
2. **Kenar Cihaz (Edge Device) Entegrasyonu:** Doğrudan YOLO modelinin gömülü platformlarda (NVIDIA Jetson / Raspberry Pi) yüksek FPS ve düşük gecikme ile çalıştırılması için optimizasyon ve performans testlerinin gerçekleştirilmesi.
3. **Metrik ve Segmentasyon İyileştirmeleri:** Sınır-farkındalıklı (Boundary-Aware) kayıp fonksiyonları ile saf YOLO modelinin ray tespit hassasiyetinin artırılması ve liderlik tablosu metriklerinin raporlanması.
