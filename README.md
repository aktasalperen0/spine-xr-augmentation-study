# Spine X-Ray Augmentation Study

VinDr-SpineXR üzerinde multi-label lezyon sınıflandırması için augmentation yöntem karşılaştırması: baseline, traditional, StyleGAN2-ADA, class-conditional Latent Diffusion.

## Yapı

```
configs/    YAML config dosyaları (her deney bir config)
src/        Kaynak kod: data, models, train, eval, utils
scripts/    01_audit → 10_make_report sıralı çalıştırılır
notebooks/  Colab runner
docs/archive/ Eski pipeline raporları (referans)
```

## Kurulum (lokal)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Çalıştırma sırası

```bash
python scripts/01_audit.py --config configs/base.yaml
python scripts/02_splits.py --config configs/base.yaml
python scripts/03_train_classifier.py --config configs/classifier/baseline.yaml --fold 1
# ... (02_5 sweep, 03 × 3 fold, 04 stylegan, ...)
python scripts/09_test_eval.py
python scripts/10_make_report.py
```

Detaylı yol haritası: `docs/archive/` içindeki planı veya proje sahibinin plan dosyasını referans alın.

## Dataset

`dataset/` VinDr-SpineXR PNG'ye dönüştürülmüş hali. 7 lezyon sınıfı (multi-label) + normal.
Detaylı etiket dağılımı `outputs/01_audit/audit_report.md` (01 scriptinden sonra).
