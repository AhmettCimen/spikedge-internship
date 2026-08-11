# Ahmet Çimen Çalışma Notları (Gün 12)

## Özet

Bugün, tren raylarının tespiti için derin öğrenme modellerinden biri olan YOLO'yu bir edge cihaz üzerinde çalıştırmak yerine, hedeflenen yüksek FPS ve düşük gecikme (latency) değerlerine ulaşabilmek amacıyla Geleneksel Görüntü İşleme (Klasik Bilgisayarlı Görme) algoritmalarını test ettim. Bu kapsamda Canny + Hough ve CLAHE + Canny + Hough yöntemlerini inceledim.

- **Canny + Hough:** Görüntüdeki belirgin kenarları Canny algoritmasıyla bulup, Hough dönüşümü kullanarak doğrusal hatları (rayları) tespit etmeyi amaçlayan klasik bir yöntemdir.
- **CLAHE + Canny + Hough:** Temel Canny + Hough yöntemine ek olarak, görüntü kontrastını bölgesel olarak artıran CLAHE yöntemini kullanarak aydınlatma farklılıklarına ve parlamalara karşı daha dirençli bir kenar tespiti yapmayı hedefler.

Yaptığım denemeler sonucunda, bu iki klasik yöntemin de çevresel etkilere ve gürültüye karşı yetersiz kaldığını ve ray tespiti konusunda başarısız olduğunu gözlemledim.

## Karşılaştırmalı Sonuçlar

Aşağıda denenen yöntemlerin ürettiği sonuçlar ile referans YOLO modelinin ürettiği sonuçların karşılaştırması yer almaktadır.

### Canny + Hough Yöntemi Sonuçları

| Canny + Hough Sonucu | YOLO Sonucu |
|---|---|
| ![Canny 00m00s](src/m1_00m00s.png) | ![YOLO 00m00s](src/yolo_00m00s.jpg) |
| ![Canny 00m04s](src/m1_00m04s.png) | ![YOLO 00m04s](src/yolo_00m04s.jpg) |
| ![Canny 00m26s](src/m1_00m26s.png) | ![YOLO 00m26s](src/yolo_00m26s.jpg) |

### CLAHE + Canny + Hough Yöntemi Sonuçları

| CLAHE + Canny + Hough Sonucu | YOLO Sonucu |
|---|---|
| ![CLAHE 00m00s](src/m2_00m00s.png) | ![YOLO 00m00s](src/yolo_00m00s.jpg) |
| ![CLAHE 00m10s](src/m2_00m10s.png) | ![YOLO 00m10s](src/yolo_00m10s.jpg) |
| ![CLAHE 00m20s](src/m2_00m20s.png) | ![YOLO 00m20s](src/yolo_00m20s.jpg) |

## Gün Sonu Değerlendirmesi

Canny + Hough ve CLAHE + Canny + Hough yöntemleri gölgeler, yansımalar ve ray dışındaki düz hatlar gibi gürültülerle karşılaştığında son derece yanıltıcı sonuçlar verdi. Edge device üzerinde YOLO gibi derin öğrenme tabanlı bir model kullanmaktan kaçınarak hız kazanmayı hedeflesek de, bu geleneksel algoritmaların ray tespiti doğrulukları bakımından tamamen başarısız olduğu sonucuna vardım.
