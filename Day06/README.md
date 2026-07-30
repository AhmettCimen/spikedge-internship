# Ahmet Çimen 07/27/2026 Çalışma Notları

## 1. YOLOv8 Nano Segmentasyon Mimarisine Geçiş
Bugün, önceki çalışmalarda kullanılan yüksek parametreli DeepLabV3+ mimarisine alternatif olarak, gömülü sistemler ve gerçek zamanlı uygulamalar için son derece hafif ve yüksek hızlı olan **YOLOv8 Nano Segmentasyon** modelini sisteme entegre ettim.

- **Pretrained Model Hazırlığı:** Pretrained ağırlıklar temel alınarak model mimarisi 2 ana hedef sınıf (`rail-track` ve `rail-raised`) üzerinden yapılandırıldı.
- **Sınıf Tanımları:**
  - `rail-track` — İki ray arasındaki ana sürüş koridoru.
  - `rail-raised` — Yükseltilmiş metal ray hatları.

---

## 2. Saf GPU Çıkarım Betiği ve Dinamik Arayüz Geliştirmesi
Modelin canlı video akışlarında maksimum hızda çalışması amacıyla saf GPU üzerinde koşan çıkarım altyapısında kapsamlı güncellemeler gerçekleştirdim.

- **Hibrit Model Desteği:** Tek bir çıkarım altyapısı üzerinden hem PyTorch SMP modellerinin hem de Ultralytics YOLOv8 segmentasyon modellerinin çalıştırılabilmesi sağlandı.
- **Dinamik Model Seçim Penceresi:** Çalıştırma esnasında varsayılan model zorlaması kaldırılarak kullanıcının işletim sistemi dosya seçme penceresi üzerinden dilediği model ağırlığını dinamik olarak seçebileceği arayüz kuruldu.
- **Temiz Görselleştirme:** Tespit kutuları (bounding box) ve metin etiketleri gizlenerek görüntü üzerinde yalnızca pürüzsüz segmentasyon maskelerinin gösterilmesi sağlandı.
- **Renk Paleti Özelleştirmesi:** Sınıf maskelerinin renk paleti özelleştirildi. Ray koridoru turkuaz, metal raylar mavi ve zemin yatağı toprak kahverengisi tonlarında renklendirildi.

---

## 3. Otomatik Video Kaydı ve Ekran Görüntüsü Altyapısı
Çıkarım sonuçlarının görsel olarak incelenebilmesi ve raporlanabilmesi için otomatik kayıt sistemi kuruldu.

- **Otomatik Numaralandırmalı Video Kaydı:** Çıkarım yapılan videolar otomatik olarak kaydedilir. Aynı modelle tekrar video işlendiğinde önceki kayıtlar ezilmez, sonuna versiyon numarası eklenerek kaydedilir.
- **Periyodik Ekran Görüntüsü:** Canlı video işleme sırasında her 5 saniyede bir işlenmiş yüksek çözünürlüklü kareler saniye ve kare bilgisi ile kaydedilerek arşivlendi.

---

## 4. Gerçek Zamanlı Performans ve Benchmark Sonuçları
Geliştirilen altyapı ile yapılan testlerde `best_yolov8n_seg` modelinin DeepLabV3+ modeline kıyasla belirgin bir hız avantajı sağladığı deneysel olarak doğrulandı.

| Metrik / Parametre | DeepLabV3+ (ResNet-50) | YOLOv8 Nano Seg (`best_yolov8n_seg`) | Değişim / Kazanç |
| :--- | :--- | :--- | :--- |
| **Giriş Çözünürlüğü** | 768×768 piksel | 640×640 piksel | Optimize çözünürlük |
| **GPU Model Çıkarım Hızı** | 23.78 ± 1.23 FPS | **35.57 ± 3.42 FPS** | **~%50 FPS Artışı** |
| **GPU Çıkarım Gecikmesi** | 42.24 ± 4.04 ms | **28.73 ± 14.33 ms** | **~13.5 ms Düşüş** |
| **Sistem Ekran Akış Hızı** | 17.06 ± 0.99 FPS | **22.49 ± 1.55 FPS** | **Daha akıcı görüntü** |
| **Sistem Ekran Gecikmesi** | 58.62 ± 3.40 ms | **45.06 ± 16.25 ms** | **~13.5 ms Düşüş** |

### Görsel Görselleştirme Karşılaştırması (DeepLabV3+ vs YOLOv8 Nano Seg)

<table>
  <thead>
    <tr>
      <th width="14%" align="center">Kare / Zaman</th>
      <th width="43%" align="center">DeepLabV3+ (ResNet-50)</th>
      <th width="43%" align="center">YOLOv8 Nano Seg (best_yolov8n_seg_4)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><b>Frame 300</b><br>(10.0s)</td>
      <td><img src="../../../Outputs/Cmp/Deeplab+Resnet/Screenshots(10)/frame_000300_sec_10.0s.jpg" width="450" style="width:100%; display:block;" alt="DeepLab 10s" /></td>
      <td><img src="../Outputs/screenshots/best_yolov8n_seg_4/frame_000300_sec_10.0s.jpg" width="450" style="width:100%; display:block;" alt="YOLOv8n 10s" /></td>
    </tr>
    <tr>
      <td align="center"><b>Frame 450</b><br>(15.0s)</td>
      <td><img src="../../../Outputs/Cmp/Deeplab+Resnet/Screenshots(10)/frame_000450_sec_15.0s.jpg" width="450" style="width:100%; display:block;" alt="DeepLab 15s" /></td>
      <td><img src="../Outputs/screenshots/best_yolov8n_seg_4/frame_000450_sec_15.0s.jpg" width="450" style="width:100%; display:block;" alt="YOLOv8n 15s" /></td>
    </tr>
    <tr>
      <td align="center"><b>Frame 1200</b><br>(40.0s)</td>
      <td><img src="../../../Outputs/Cmp/Deeplab+Resnet/Screenshots(10)/frame_001200_sec_40.0s.jpg" width="450" style="width:100%; display:block;" alt="DeepLab 40s" /></td>
      <td><img src="../Outputs/screenshots/best_yolov8n_seg_4/frame_001200_sec_40.0s.jpg" width="450" style="width:100%; display:block;" alt="YOLOv8n 40s" /></td>
    </tr>
  </tbody>
</table>


> **Değerlendirme Notu:**  
> DeepLabV3+ ile eğitilen modelin uzak mesafedeki detaylarda daha başarılı olduğu gözlemlenmiştir. Ancak saniyede daha az kare işleyebilmesi ve kareleri YOLO'ya göre daha geç işlemesi (düşük FPS / yüksek gecikme) onu gerçek zamanlı sistemlerde dezavantajlı kılabilir.  
> 
> Daha nesnel bir tercih yapmak için önümüzdeki günlerde modellerin IoU (Intersection over Union) skorları detaylı olarak incelenecektir.

---

## 5. Hedefler
Önümüzdeki günler için planlanan çalışmalar:
1. **Test Aracı ve Liderlik Tablosu:** Tüm modellerin hiç görmediği bağımsız test görselleri üzerinde sınıf bazlı IoU skorlarını ölçen değerlendirme aracının oluşturulması.
2. **Colab Üzerinde 50 Epoch Çoklu Sınıf Eğitim:** RailSem19 veri setindeki diğer classları da kullanarak (trackbed, rail-embedded vb.) 6 hedef sınıf için Colab GPU altyapısında 50 epoch boyunca YOLOv8 Nano eğitiminin yürütülmesi.


