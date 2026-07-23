# Ahmet Çimen 07/20/2026 Çalışma Notları

## Özet

Bugün, verilen video üzerinde tren raylarını kesintisiz ve farklı koşullara uyum sağlayabilecek şekilde tespit edebilmek için model geliştirmede kullanılabilecek yöntemleri araştırmaya başladım. Literatürde benzer problemlerin çözümünde kullanılan modelleri ve görüntü işleme yöntemlerini inceleyerek, bunlardan hangilerinin bizim kullanım senaryomuza daha uygun olabileceğini anlamaya çalıştım. Daha sonrasında ise modelin üreteceği segmentasyon maskesinin daha kesintisiz ve doğru sonuç verebilmesi amacıyla, bu maskeyi matematiksel bir eğriye dönüştüren ve oluşabilecek kesintileri ile anomalileri filtreleyebilecek yöntemleri, bu yöntemlerin senaryomuzda uygulanabilirliği konularını araştırdım.

- Tren rayı tespiti için kullanılan temel yaklaşımları araştırdım.
  - Semantic Segmentation tabanlı yöntemler
  - Lane Detection tabanlı yöntemler

- Semantic Segmentation tabanlı modelleri inceledim.
  - SegFormer
  - DeepLabV3+
  - Mask2Former

- Lane Detection için geliştirilen modelleri inceledim.
  - CLRNet
  - SCNN

- Modellerin çalışma prensiplerini, avantajlarını, dezavantajlarını ve tren rayı tespitine uygunluklarını karşılaştırdım.

- Semantic Segmentation ile Lane Detection yaklaşımları arasındaki farkları araştırdım.

- Segmentasyon tabanlı modellerin ürettiği maskelerden ray çizgilerinin çıkarılması için kullanılabilecek görüntü işleme yöntemlerini inceledim.
  - Skeletonization
  - Curve Fitting
  - Kalman Filter

- Gerçek zamanlı ray takibi için önerilen iş akışını inceledim.

- Farklı model ve algoritmaların ileride projeye adapte edilebilmesi için alternatif yöntemleri araştırdım.

---

# Önerilen İş Akışı

```text
Video
   ↓
Seçilen Model ile Segmentasyon
   ↓
Ray Segmentasyon Maskesi
   ↓
Skeletonization
(Kalın maske tek piksellik merkez çizgisine dönüştürülür.)
   ↓
Seçilen Yöntem ile Curve Fitting
(Çizgi matematiksel olarak düzgün bir eğriye dönüştürülür.)
   ↓
Kalman Filter (Opsiyonel)
(Matematiksel eğri üzerinde kesinti oluşumunu engellemek için filtreleme yöntemi uygulanır.)
   ↓
OpenCV ile Video Üzerine Çizim

```

---

# Alternatif Yöntemler

### Segmentasyon İşlemi İçin Opsiyonlar

- SegFormer
- DeepLabV3+
- Mask2Former
- ve benzeri Semantic Segmentation modelleri

### Curve Fitting İçin Opsiyonlar

- Polynomial Fit
- Cubic Spline
- B-Spline
- Bezier Curve
- ve benzeri eğri uydurma yöntemleri

### Kararlılık İçin Değerlendirebilecek Yöntemler

- Kalman Filter
- Optical Flow
- SORT
- DeepSORT
- ByteTrack
- ve benzeri algoritmalar

---

# Gün Sonu Değerlendirmesi

Bugün tren rayı segmentasyonu için kullanılabilecek teknoloji seçeneklerini araştırdım. Farklı modellerin avantaj ve dezavantajlarını inceleyerek, kullanım senaryomuza en uygun yaklaşımı belirleyebilmek adına çalışma prensiplerini öğrenmeye başladım. Araştırma sürecinde yalnızca belirli modellerle sınırlı kalmayıp, değerlendirilebilecek farklı alternatifleri de inceledim.

Ayrıca, segmentasyon sonrasında oluşabilecek ray maskesindeki gürültü ve kesintileri azaltmak amacıyla **Skeletonization** ile maskenin eğriye dönüştürülmesi ve bu eğriye **Kalman Filter** uygulanması gibi yöntemleri değerlendirdim.

Gerçek zamanlı ve akıcı bir sistem oluşturabilmek için **Skeletonization** ve **Kalman Filter** süreçlerinin çalışma mantığını ve bu yöntemlerin kullanım senaryomuza uygunluğunu araştırdım.

# Sonraki Adımlar

Önümüzdeki günlerde, rayları tespit edebilmek için değerlendirmediğim diğer modelleri incelemeyi, araştırdığım segmentasyon yöntemlerinden kullanım senaryomuza en uygun olanını teknik detaylarıyla incelemeyi planlıyorum. Ayrıca, etiketli eğitim verisi gerektiren modeller için elimizde bulunan video verisinden yeterli miktarda etiketli eğitim veri seti oluşturmada ele alabileceğim yöntemleri değerlendireceğim. Bu çalışmaların ardından seçilen modeli uygulamaya alarak ilk implementasyon sürecine başlamayı hedefliyorum.