"""
The ANN: an ImageNet-pretrained ResNet whose output is restricted to the classes of the
crop the officer selected. Used by training, evaluation and the web application.
"""
import torch
import torch.nn as nn
from torchvision import models

BACKBONES = {
    18: (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    34: (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    50: (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
}


class CropMaskedResNet(nn.Module):
    def __init__(self, depth, crop_mask, dropout=0.2, pretrained=True):
        super().__init__()
        build, weights = BACKBONES[depth]
        self.backbone = build(weights=weights if pretrained else None)
        n_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_features, crop_mask.shape[1]))
        self.register_buffer("crop_mask", crop_mask)

    def forward(self, images, crop_idx):
        logits = self.head(self.backbone(images))
        allowed = self.crop_mask[crop_idx]
        return logits.masked_fill(~allowed, -1e4)