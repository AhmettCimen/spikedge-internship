# Ahmet Çimen Çalışma Notları (Gün 13)

## Özet

Bugün, tıpkı bir önceki gün yaptığım gibi ağır derin öğrenme modellerini edge cihazlar üzerinde çalıştırmak yerine, yüksek FPS ve düşük gecikme elde edebilmek adına kullanılabilecek diğer Geleneksel Görüntü İşleme yöntemlerini denedim. Bu kapsamda Frangi filtresi ve Dinamik Programlama (Dynamic Programming) temelli yöntemleri test ettim.

- **Frangi + Hough Yöntemi:** Damar ve benzeri tüp/çizgi şeklindeki yapıları belirginleştiren Frangi filtresini kullanarak rayları ön plana çıkarmayı ve ardından Hough dönüşümüyle bu doğruları tespit etmeyi amaçlayan bir yöntemdir.
- **Dynamic Programming (Dinamik Programlama) Yöntemi:** Görüntü üzerindeki potansiyel ray pikselleri arasında en düşük maliyetli yolu bularak, ray hattını kesintisiz ve sürekli bir eğri şeklinde takip etmeye çalışan optimizasyon tabanlı bir algoritmadır.

Testlerim sonucunda, bu iki klasik yöntemin de değişen çevre koşullarına karşı oldukça kırılgan olduğu ve rayları kararlı bir şekilde tespit edemediği için başarısız olduğu görülmüştür.

## Karşılaştırmalı Sonuçlar

Aşağıda denenen yöntemlerin ürettiği sonuçlar ile referans YOLO modelinin ürettiği sonuçların karşılaştırması yer almaktadır.

### Frangi + Hough Yöntemi Sonuçları

| Frangi + Hough Sonucu | YOLO Sonucu |
|---|---|
| ![Frangi 00m02s](src/m3_00m02s.jpg) | ![YOLO 00m02s](src/yolo_00m02s.jpg) |
| ![Frangi 00m36s](src/m3_00m36s.jpg) | ![YOLO 00m36s](src/yolo_00m36s.jpg) |
| ![Frangi 01m56s](src/m3_01m56s.jpg) | ![YOLO 01m56s](src/yolo_01m56s.jpg) |

### Dynamic Programming Yöntemi Sonuçları

| Dynamic Programming Sonucu | YOLO Sonucu |
|---|---|
| ![DP 00m06s](src/m6_00m06s.jpg) | ![YOLO 00m06s](src/yolo_00m06s.jpg) |
| ![DP 01m42s](src/m6_01m42s.jpg) | ![YOLO 01m42s](src/yolo_01m42s.jpg) |
| ![DP 02m08s](src/m6_02m08s.jpg) | ![YOLO 02m08s](src/yolo_02m08s.jpg) |

## Gün Sonu Değerlendirmesi

Frangi + Hough ve Dynamic Programming yöntemleri, hedeflenen düşük donanım kaynağı tüketimini sağlayabilme potansiyeline sahip olsalar da pratik kullanımda oldukça yetersiz kaldılar. Klasik yöntemler, gerçek dünyanın değişken koşullarında (aydınlatma, kamera açısı, gölge gibi etkilerde) çok fazla yanlış pozitif üretmiş ya da rayları tamamen kaçırmıştır. Bu durum, YOLO gibi yapay zeka modellerine kıyasla matematiksel algoritmaların oldukça dayanıksız olduğunu göstermiştir.
