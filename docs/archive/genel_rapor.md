# Omurga XR Augmentation Çalışması - Hoca Sunum Raporu

Tarih: 9 Nisan 2026

## 1) Yönetici Özeti

Bu çalışmada VinDr-SpineXR veri setinde class imbalance problemini azaltmak için 3 yaklaşım uygulandı:

- Baseline
- Traditional augmentation
- Generative augmentation (GAN ve Diffusion)

Güncel durum:

- Step1, Step2, Step3, Step4 tamamlandı ve dosyalanmış metrikler mevcut.
- Step5 ve Step6 pipeline'ları tamamlandı (a-b-c-d-e-f akışı çalıştırıldı).
- Step5f ve Step6f tek fold testte başarı yaklaşık %40 seviyelerine düştü (kullanıcı/çalıştırma gözlemi).
- Sonuç: Bu veri rejiminde generative augmentation, özellikle minority sınıflar nedeniyle, performansı artırmadı.

## 2) Amaç ve Problem

Amaç: Multi-label omurga lezyon sınıflandırmasında augmentation türlerinin etkisini karşılaştırmak.

Ana zorluklar:

- Veri dengesizliği (özellikle Osteophytes baskın, bazı sınıflar çok az örnekli).
- Aynı görüntüde birden fazla etiket (multi-label yapı).
- Generative yöntemlerin düşük örnek sayılarında kararsız olması.

## 3) Script ve Artefakt Özeti

Ana scriptler:

- step1_data_audit.py
- step2_make_splits.py
- step3_train_baseline.py
- step3_predict_single.py
- step4_train_traditional_aug.py
- step4_predict_single.py
- step5_a_prepare_gan_real_pool.py
- step5_b_train_dcgan_per_class.py
- step5_c_generate_gan_images.py
- step5_d_build_gan_metadata.py
- step5_e_precheck.py
- step5_f_train_gan_aug.py
- step6_a_prepare_diffusion_real_pool.py
- step6_b_train_diffusion_per_class.py
- step6_c_generate_diffusion_images.py
- step6_d_build_diffusion_metadata.py
- step6_e_precheck.py
- step6_f_train_diffusion_aug.py

Bu workspace'te görülen önemli output'lar:

- outputs/step1_data_audit/\*
- outputs/step2_cv_splits/\*
- outputs/step3_baseline/\*
- outputs/step4_traditional_aug/\*
- outputs/gan_models/\*
- outputs/diffusion_models/\*
- outputs/gan_synthetic/by_class/\*
- outputs/diffusion_synthetic/by_class/\*

Not:

- Tek fold Step5f/Step6f çalıştırma sonucu kullanıcı tarafından %40'lara düşüş olarak raporlandı.
- Bu workspace'te step5_gan_aug/step6_diffusion_aug sonuç dosyaları görünmediği için ilgili metrik satırı kullanıcı gözlemi olarak raporlandı.

## 4) Step Bazlı Yapılanlar

## Step1 - Data Audit

Script: step1_data_audit.py

Yapılan iş:

- Abnormal/normal train sayıları çıkarıldı.
- Image-level sınıf dağılımı ve multi-label oranları hesaplandı.
- CSV raporları outputs/step1_data_audit altında kaydedildi.

Bulgular:

- Abnormal: 4129
- Normal: 4260
- Toplam: 8389
- Abnormal multi-label oranı: %25.551
- Tüm train multi-label oranı: %12.576

## Step2 - Multilabel Stratified 3-Fold

Script: step2_make_splits.py

Yapılan iş:

- Normal+abnormal birlikte tek multilabel tablo üretildi.
- MultilabelStratifiedKFold ile 3 fold train/val ayrımı yapıldı.
- Fold prevalans farkları raporlandı.

Bulgular:

- Fold boyutları yaklaşık 5593/2796 ve 5592/2797
- Maks prevalans farkı ~0.00036-0.00038
- Fold dengesi çok iyi.

## Step3 - Baseline

Scriptler:

- step3_train_baseline.py
- step3_predict_single.py

Yapılan iş:

- EfficientNet-B0 + BCEWithLogitsLoss ile multi-label eğitim.
- Threshold sweep ile en iyi macro-F1 eşiği bulundu.

Sonuç (3 fold):

- Macro-F1 mean: 0.4739
- mAP mean: 0.4562
- En stabil referans model bu aşama oldu.

## Step4 - Traditional Augmentation

Scriptler:

- step4_train_traditional_aug.py
- step4_predict_single.py

Yapılan iş:

- Yalnızca train tarafında klasik augmentation (flip/affine/jitter/blur).
- Val tarafı temiz tutuldu.

Sonuç (3 fold):

- Macro-F1 mean: 0.4678
- mAP mean: 0.4451
- Baseline'a göre ortalamada düşüş, foldlar arası oynaklık artışı.

## Step5 - GAN Augmentation

Script zinciri:

- step5_a_prepare_gan_real_pool.py
- step5_b_train_dcgan_per_class.py
- step5_c_generate_gan_images.py
- step5_d_build_gan_metadata.py
- step5_e_precheck.py
- step5_f_train_gan_aug.py

Yapılan iş:

- Single-label abnormal örneklerden class-wise GAN eğitim havuzu oluşturuldu.
- Sınıf başına DCGAN eğitimi ve sentetik üretim yapıldı.
- Metadata hazırlanıp precheck sonrası classifier eğitimine bağlandı.

Önemli gözlem:

- Çok az örnekli sınıflar (17, 16) eğitimden atlandı.
- Tek fold testte Step5f performansı yaklaşık %40 seviyelerine düştü.

## Step6 - Diffusion Augmentation (MONAI)

Script zinciri:

- step6_a_prepare_diffusion_real_pool.py
- step6_b_train_diffusion_per_class.py
- step6_c_generate_diffusion_images.py
- step6_d_build_diffusion_metadata.py
- step6_e_precheck.py
- step6_f_train_diffusion_aug.py

Yapılan iş:

- MONAI tabanlı DDPM pipeline kuruldu.
- Farklı MONAI sürümleri için UNet constructor uyumluluğu eklendi.
- Class-wise eğitim, sentetik üretim, metadata ve classifier entegrasyonu tamamlandı.

Önemli gözlem:

- Üretilen görsellerin bir kısmı noise-benzeri kalite gösterdi.
- Tek fold Step6f testinde başarı yaklaşık %40 seviyelerine düştü.

## 5) Karşılaştırmalı Durum Özeti

Dosyalanmış metriklere göre:

- Baseline (Step3): en iyi genel denge
- Traditional (Step4): ortalamada baseline altında

Koşu gözlemine göre:

- GAN (Step5f, tek fold): belirgin düşüş
- Diffusion (Step6f, tek fold): belirgin düşüş

Genel yorum:

- Bu veri dağılımında generative augmentation beklenen iyileşmeyi sağlamadı.
- Özellikle çok az örnekli sınıflar, üretim kalitesini ve nihai sınıflandırma performansını aşağı çekti.

## 6) Hoca Sunumunda Ana Mesaj

1. Problem doğru tanımlandı: ciddi class imbalance + multi-label yapı.
2. Veri analizi ve split kalitesi güçlü şekilde kuruldu.
3. 3 yöntem de teknik olarak uygulandı ve çalıştırıldı.
4. Sonuçlar gösterdi ki bu veri rejiminde GAN/Diffusion avantaj sağlamadı.
5. Negatif sonuç da bilimsel olarak değerlidir: generative yöntemler veri eşiğine duyarlı.

## 7) Kısa Sonuç

Proje hedefi olan 3 yaklaşım karşılaştırması teknik olarak tamamlandı. Elde edilen bulgular, mevcut veri dengesizliği ve minority sınıf azlığı altında baseline/traditional yaklaşımın daha güvenilir kaldığını, GAN ve diffusion tabanlı artırmanın ise tek fold testte performansı yaklaşık %40 seviyelerine düşürdüğünü göstermektedir.
