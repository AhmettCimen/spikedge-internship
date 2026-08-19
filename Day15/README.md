# Ahmet Çimen Çalışma Notları (Gün 15)

## Özet

Ray bölgesini YOLO modeli ile segment edebilen modelimiz neredeyse hazır durumda. Şimdi bu ray bölgeleri üzerinde bir anomali olup olmadığını tespit etmek için, gerek eğitim veri seti gerekse anomalileri test edebileceğimiz bir test setine ihtiyacımız var. Ancak gerçek dünyadan derlenmiş, rayların üzerinde olmaması gereken nesnelerin (anomalilerin) veya raylardaki bozunmaların bulunduğu veri setlerine erişimimiz olmadığı için sentetik veri üretmeye başladık.

Sentetik veri ile bir model eğitmenin çeşitli sakıncaları olabileceğinin farkındayım. Anomali tespiti yaparken doğrudan anomalileri öğrenen bir derin öğrenme metodu geliştirilebilir; fakat bu durum hem çok sayıda veri gerektirir hem de modelin eğitim sırasında hiç görmediği farklı türdeki anomaliler için zayıf kalmasına neden olabilir. Bunun üstesinden gelmek için farklı yöntemler de değerlendirilebilir.

Fakat denenecek her yöntem için (gerek test etmek gerekse eğitmek için) elimizde anomali içeren bir veri seti bulunması gerekiyor. Bu doğrultuda sentetik veri üretimi için aşağıdaki alternatifleri değerlendirmeye karar verdim:
- Stable Diffusion + Inpainting
- FLUX + Inpainting / Kontext
- SAM + Görüntü Kompozitleme
- Veya Nano Banana, Seedream gibi hazır modellerin değerlendirilmesi

Bu alternatifler arasından ilk olarak **Nano Banana Pro** ile çalışmaya başladım. Aşağıda Nano Banana Pro kullanılarak üretilmiş bazı sentetik anomali örnekleri yer almaktadır. Bu görsellerde genel olarak raya düşmüş bisiklet, direk, köpek veya yaya gibi durumlar simüle edilmiştir.

## Üretilen Sentetik Veri Örnekleri

| Nano Banana Pro Çıktıları |
|---|
| ![img001](src/img001.png) |
| ![img003](src/img003.png) |
| ![img004](src/img004.png) |
| ![img005](src/img005.png) |
| ![img007](src/img007.png) |
| ![img009](src/img009.png) |

## Gün Sonu Değerlendirmesi

Sentetik veri üretimi, anomali tespiti sistemimizi geliştirebilmek ve test edebilmek için atılmış zorunlu ve kritik bir adımdır. Nano Banana Pro ile denediğimiz bu ilk sonuçlar umut vericidir. Bir sonraki aşamada farklı senaryoları simüle eden sentetik görsellerin sayısını artırarak çalışmalarımıza devam edeceğiz.
