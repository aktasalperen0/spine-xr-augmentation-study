# VinDr-SpineXR Sınıflandırma ve Veri Artırımı Projesi: Stratejik Yol Haritası

## 1. Mevcut Durum Özeti ve Analiz

Mevcut denemeler sonucunda Baseline modelin %47, geleneksel artırımın %50 başarıda kalması ve generatif modellerin (GAN/Diffusion) sadece gürültü üreterek performansı %40'a düşürmesi, sistemin tıbbi veri karakteristiğine uygun kurgulanmadığını göstermektedir.

### Temel Sorunlar

- Ağırlık Uyumsuzluğu: Standart ImageNet (kedi-köpek) ağırlıkları tıbbi XR dokularını anlamakta zorlanmaktadır.
- Metrik Eksikliği: Sadece doğruluğa odaklanmak, dengesiz tıbbi veride modelin "her şeye normal" demesini ödüllendirmektedir.
- Generatif Çöküş: Küçük veri setinde diskriminatörün ezberlemesi (GAN) ve yanlış gürültü takvimi (Diffusion) üretimi engellemektedir.

## 2. Proje Yapılandırması ve Veri Hazırlığı

### Sınıf Tanımlama: 7 + 1 (No Finding)

Veri setindeki 7 hastalık etiketine ek olarak "No finding" etiketi 8. bir sınıf olarak eklenecektir. Bu hamle modelin sağlıklı doku özelliklerini pozitif bir sinyal olarak öğrenmesini sağlayacak ve Macro F1 skorunu (normal vakaları bildiği için) doğrudan yükseltecektir.

### Görüntü Ön İşleme (Preprocessing)

Tüm görüntülere CLAHE (Contrast Limited Adaptive Histogram Equalization) uygulanacaktır. Bu teknik, omurga XR görüntülerindeki düşük kontrastlı lezyonların (osteophytes vb.) model tarafından daha net yakalanmasını sağlar.

## 3. Adım Adım Uygulama Planı

### Adım 1: Baseline (Referans Model)

- Mimari: EfficientNet-B0
- Ağırlıklar: ImageNet yerine RadImageNet (1.35M tıbbi görüntü ile ön eğitilmiş) kullanılacaktır.
- Beklenen katkı: Bu değişim tek başına %5-9 başarı artışı vaat eder.

#### Hiperparametreler

- Optimizer: AdamW (Weight Decay: $1 \times 10^{-5}$)
- Learning Rate: $1 \times 10^{-4}$ (başlangıç)
- Scheduler: ReduceLROnPlateau (Patience: 5, Factor: 0.1)
- Loss: BCEWithLogitsLoss
- Epoch ve Patience: 100 Epoch, 15 Patience (en iyi validasyon ağırlıkları kaydedilecek)

### Adım 2: Traditional Augmentation (Geleneksel Artırım)

- Strateji: Sadece geometrik değil, tıbbi gerçekliğe uygun transformasyonlar
- Yöntemler: $\pm 15$ derece rotasyon, $0.8 - 1.2$ zoom, horizontal flip, Random Brightness/Contrast
- Not: Artırımlar sadece eğitim setine uygulanacak, test seti saf bırakılacaktır.

### Adım 3: GAN Augmentation (StyleGAN2-ADA)

Mevcut "saf gürültü" sorununu çözmek için mimari değişikliğe gidilecektir.

#### Neden StyleGAN2-ADA?

"Adaptive Discriminator Augmentation" (ADA) mekanizması, diskriminatörün az sayıdaki tıbbi görüntüyü ezberlemesini engelleyerek stabil eğitim sağlar.

#### Üretim Stratejisi

Her sınıf için ayrı modeller eğitilerek dengesizliği gidermek üzere sentetik görüntüler üretilecek ve veri seti dengelenecektir (oversampling).

### Adım 4: Diffusion Augmentation (Stabilize MONAI DDPM)

- Stabilizasyon: Öğrenme oranı $5 \times 10^{-5}$ seviyesine çekilecek ve `scaled_linear` gürültü takvimi kullanılacaktır.
- Gradyan Biriktirme (Gradient Accumulation): Küçük batch boyutlarının yarattığı gürültülü gradyanlar, sanal büyük batch oluşturularak (step 4 veya 8) giderilecektir.
- Kalite Kontrol: Üretilen resimler piksel gürültüsü veriyorsa, eğitim süresi artırılacak ve MONAI `DiffusionInferer` sınıfı üzerinden 1000 adım yerine 100-250 adımlık DDIM örneklemesi denenecektir.

## 4. Değerlendirme ve Raporlama Standartları (Hoca Talepleri)

Eğitim bittiğinde hazırlanan test scripti, her augmentation yöntemi için aşağıdaki çıktıları ayrı ayrı üretecektir:

- mAP (Mean Average Precision): Modelin tüm olasılık eşikleri boyunca sergilediği kararlılık
- Sınıf Bazlı Rapor (Classification Report)
  - Macro Average: Tüm sınıfları eşit ağırlıkta görür (küçük sınıflardaki başarıyı anlamak için kritik)
  - Weighted Average: Sınıf sayısına göre ağırlıklandırır (genel başarının illüzyonunu gösterebilir)
- Karmaşıklık Matrisi (Confusion Matrix): Hangi hastalıkların birbirine karıştırıldığının analizi

## 5. Tahmini Başarı Beklentisi (Macro F1 / mAP)

| Yöntem          |  Mevcut Durum | Tahmin Edilen (Yeni) | Temel Fark Yaratan Unsur                 |
| --------------- | ------------: | -------------------: | ---------------------------------------- |
| Baseline        |           %47 |            %65 - %72 | RadImageNet + CLAHE + No Finding         |
| Traditional Aug |           %50 |            %68 - %75 | Medikal odaklı agresif transformasyonlar |
| GAN Aug         | %40 (Gürültü) |            %74 - %80 | StyleGAN2-ADA ile stabil sentetik veri   |
| Diffusion Aug   | %40 (Gürültü) |            %76 - %82 | Düşük LR + Gradyan Biriktirme + DDPM     |

## 6. Claude Code ve Colab Pro İçin Uygulama Notları

- Bellek Yönetimi: Colab Pro'da A100 kullanırken Mixed Precision (FP16) eğitimini aktif edin; bu, eğitimi hızlandırır ve bellekten tasarruf sağlar.
- Library: `torchmetrics` kütüphanesini `MultilabelAveragePrecision` için, `scikit-learn` kütüphanesini detaylı raporlama için kullanın.
- Checkpointing: Her 5 epoch'ta bir model ağırlıklarını ve ürettiği örnek resimleri kaydedip "saf gürültü" üretilip üretilmediğini manuel kontrol edin.
