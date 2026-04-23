# RadImageNet pretrained weights

Classifier configs default to `backbone: radimagenet_resnet50`. Download the
PyTorch state_dict from the official RadImageNet release and place it at:

```
weights/radimagenet_resnet50.pth       # required for ResNet50 backbone
weights/radimagenet_densenet121.pth    # optional, if you swap the backbone
```

Source: https://github.com/BMEII-AI/RadImageNet

Mirror note: the official release ships TensorFlow checkpoints as well as
PyTorch-converted state_dicts. Use the PyTorch version. If only TF weights are
available, convert once with the authors' provided script and commit the
resulting `.pth` file (or keep it local — `weights/` is ignored by default).

## Swap to ImageNet (ablation)

Each classifier YAML still lists `tf_efficientnetv2_s.in21k_ft_in1k` as the
timm fallback; simply replace the `backbone:` line with that value to run an
ablation without RadImageNet weights.
