# Omurga XR Projesi - Kısa Rapor

Tarih: 9 Nisan 2026

## 1) Proje Amaci

VinDr-SpineXR veri setinde class imbalance problemini azaltmak icin 3 farkli yaklasimi karsilastirdik:

- Baseline
- Traditional augmentation
- Generative augmentation (GAN + Diffusion)

## 2) Adim Adim Ne Yaptik

## Step1 - Veri Analizi

- Train veri dagilimini cikardik.
- Multi-label oranlarini ve sinif bazli image count degerlerini hesapladik.
- Sonuc: Veri ciddi dengesiz, minority siniflar cok az.

## Step2 - 3-Fold Split

- Multilabel stratified 3-fold split olusturduk.
- Foldlarin prevalans farklari cok dusuk cikti (dengeli split).

## Step3 - Baseline Egitim

- EfficientNet-B0 ile multi-label egitim yaptik.
- 3 fold sonucu:
  - Macro-F1 mean: 0.4739
  - mAP mean: 0.4562
- Baseline referans performansimiz oldu.

## Step4 - Traditional Augmentation

- Sadece train tarafina klasik augmentation (flip/affine/jitter/blur) ekledik.
- 3 fold sonucu:
  - Macro-F1 mean: 0.4678
  - mAP mean: 0.4451
- Sonuc baseline'in bir miktar altinda kaldi.

## Step5 - GAN Augmentation

- Single-label abnormal goruntulerden class-wise GAN egittik.
- Az ornekli siniflar (17, 16) egitimden atlandi.
- Sentetik goruntu uretip classifier'a bagladik (Step5f, tek fold test).
- Sonuc: Basari yaklasik %40 seviyelerine dustu.

## Step6 - Diffusion Augmentation (MONAI)

- MONAI tabanli diffusion pipeline kurduk ve calistirdik.
- Model uyumlulugu icin surum-fallback duzeltmeleri yaptik.
- Sentetik goruntulerde noise-benzeri kalite gozlemi oldu.
- Step6f tek fold testte basari yaklasik %40 seviyelerine dustu.

## 3) Genel Degerlendirme

- 3 yontemi de teknik olarak uygulayip test ettik.
- Mevcut veri rejiminde (ozellikle minority sinif azligi) GAN ve Diffusion beklenen katkıyı vermedi.
- Bu nedenle bu projede en guvenilir sonuc baseline/traditional tarafinda kaldi.

## 4) Ana Mesaj

- "Yontemleri eksiksiz denedik, sonuclari olctuk."
- "Negatif sonuc da bilimsel bir bulgudur: Generative augmentation bu veri dagiliminda veri esigine takildi."
- "Devam adimi olarak ya minority veri arttirma (gercek veri) ya da daha farkli class-conditional/transfer stratejileri gerekir."
