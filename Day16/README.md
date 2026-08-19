# Ahmet Çimen Çalışma Notları (Gün 16)

## Özet

Ray segmentasyonunun ardından, anomali tespiti çalışmalarını test etmek ve değerlendirmek amacıyla sentetik veri üretimine devam ediyoruz. Yöntem ne olursa olsun sistemin başarısını doğru ölçmek için zengin bir anomali veri setine ihtiyaç bulunmaktadır.

Bugün **Nano Banana Pro** kullanılarak düşmüş direkler, canlılar, yayalar, toprak kaymaları, yol çalışmaları ve kırık köprü gibi birçok farklı senaryoyu simüle eden yeni sentetik görseller ürettim.

## Senaryo ve Sınır Durumu İncelemeleri

Üretilen her örnek doğrudan belirgin bir tehlike içermemektedir. Sistemin hassasiyetini ve ayrım yeteneğini doğru test edebilmek adına tehlikeli, tehlikesiz ve sınıra yakın senaryolar da simüle edilmiştir:

| Görsel | Durum Değerlendirmesi |
|---|---|
| ![Sentetik Görsel - Tehlikeli Durum](src/img037.png) | Bu fotoğrafta rayın üzerinde doğrudan engel oluşturan bir saman balyası bulunmaktadır (tehlikeli durum). |
| ![Sentetik Görsel - Güvenli Durum](src/img046.png) | Bu fotoğrafta ise saman balyası ray hattının dışında, tamamen güvenli bir bölgede yer almaktadır. |
| ![Sentetik Görsel - Potansiyel Tehlike](src/img044.png) | Bu fotoğrafta köpek rayın tam üstünde olmasa da raya çok yakın konumlandığı için potansiyel bir tehlike arz etmektedir. |

## Üretilen Sentetik Veri Örnekleri

| Nano Banana Pro Çıktıları |
|---|
| ![Sentetik Görsel](src/img011.png) |
| ![Sentetik Görsel](src/img013.png) |
| ![Sentetik Görsel](src/img014.png) |
| ![Sentetik Görsel](src/img015.png) |
| ![Sentetik Görsel](src/img016.png) |
| ![Sentetik Görsel](src/img019.png) |
| ![Sentetik Görsel](src/img020.png) |
| ![Sentetik Görsel](src/img021.png) |
| ![Sentetik Görsel](src/img022.png) |
| ![Sentetik Görsel](src/img023.png) |
| ![Sentetik Görsel](src/img024.png) |
| ![Sentetik Görsel](src/img025.png) |
| ![Sentetik Görsel](src/img026.png) |
| ![Sentetik Görsel](src/img032.png) |
| ![Sentetik Görsel](src/img034.png) |
| ![Sentetik Görsel](src/img038.png) |
| ![Sentetik Görsel](src/img039.png) |
| ![Sentetik Görsel](src/img043.png) |

## Gün Sonu Değerlendirmesi

Tehlikeli, potansiyel riskli ve güvenli durumları bir arada içeren sentetik veri setimiz zenginleşmeye devam ediyor. Üretilen bu senaryolar, anomali tespit algoritmalarının doğruluğunu ve sınır hassasiyetini sınayabileceğimiz önemli bir test altyapısı sunmaktadır.
