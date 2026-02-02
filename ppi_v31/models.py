# ppi_v31/models.py
from __future__ import annotations

import torch.nn as nn
from torchvision import models


def _freeze_all(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False


def _unfreeze(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = True


def build_image_backbone(
    arch: str,
    n_classes: int,
    unfreeze: str = "last2",
    pretrained: bool = True,
) -> nn.Module:
    """
    Builds a torchvision vision backbone and replaces the classification head.

    Args:
        arch: one of {"resnet50","convnext_tiny","efficientnet_b0","vit_b_16"}
        n_classes: number of output classes
        unfreeze: "none" | "last2" | "all"
        pretrained: whether to load torchvision pretrained weights

    Returns:
        torch.nn.Module (classifier head is always trainable)
    """
    arch = arch.lower()
    unfreeze = (unfreeze or "none").lower()
    if unfreeze not in ("none", "last2", "all"):
        raise ValueError(f"Unknown unfreeze mode: {unfreeze}")

    if arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_f = m.fc.in_features
        _freeze_all(m)
        if unfreeze in ("last2", "all"):
            _unfreeze(m.layer4)
        if unfreeze == "all":
            _unfreeze(m)
        m.fc = nn.Linear(in_f, n_classes)  # new head (trainable)
        return m

    if arch == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        in_f = m.classifier[2].in_features
        _freeze_all(m)
        if unfreeze in ("last2", "all"):
            _unfreeze(m.features[-1])
        if unfreeze == "all":
            _unfreeze(m)
        m.classifier[2] = nn.Linear(in_f, n_classes)  # new head (trainable)
        return m

    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_f = m.classifier[1].in_features
        _freeze_all(m)
        if unfreeze in ("last2", "all"):
            _unfreeze(m.features[-1])
        if unfreeze == "all":
            _unfreeze(m)
        m.classifier[1] = nn.Linear(in_f, n_classes)  # new head (trainable)
        return m

    if arch == "vit_b_16":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)
        in_f = m.heads.head.in_features
        _freeze_all(m)
        if unfreeze in ("last2", "all"):
            _unfreeze(m.encoder.layers[-1])
        if unfreeze == "all":
            _unfreeze(m)
        m.heads.head = nn.Linear(in_f, n_classes)  # new head (trainable)
        return m

    raise ValueError(f"Unknown arch: {arch}")
