"""
Explanations for the ANN (Question 5): Grad-CAM heatmaps of the image regions that raised a class's score.
"""
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017) for CropMaskedResNet."""

    def __init__(self, model, layer=None):
        self.model = model.eval()
        self.layer = layer if layer is not None else model.backbone.layer4
        self.activations = None
        self.layer.register_forward_hook(self._keep)

    def _keep(self, module, inputs, output):
        self.activations = output

    def __call__(self, image, crop_idx, class_idx=None):
        """image: 1 x 3 x H x W (normalised); crop_idx: tensor([crop]).
        Returns the heatmap (H x W, 0-1), the explained class and the logits."""
        with torch.enable_grad():
            logits = self.model(image, crop_idx)[0]
            if class_idx is None:
                class_idx = int(logits.argmax())
            grads = torch.autograd.grad(logits[class_idx], self.activations)[0]
        weights = grads[0].mean(dim=(1, 2))
        cam = F.relu((weights[:, None, None] * self.activations[0]).sum(dim=0))
        cam = F.interpolate(cam[None, None], size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.detach().cpu(), class_idx, logits.detach().cpu()


def model_view(photo, size=224):
    """The part of a PIL photo the model sees (shorter side resized, centre crop), at display size."""
    w, h = photo.size
    new_size = (size, int(size * h / w)) if w <= h else (int(size * w / h), size)   # same rounding as torchvision
    photo = photo.resize(new_size, Image.BILINEAR)
    left, top = int(round((photo.width - size) / 2)), int(round((photo.height - size) / 2))
    return photo.crop((left, top, left + size, top + size))


def overlay(photo, cam, size=448, alpha=0.6, cmap="inferno"):
    """The model's view of the photo with the heatmap on top; cool regions stay as the photo."""
    view = model_view(photo.convert("RGB"), size)
    heat = F.interpolate(cam[None, None], size=view.size[::-1], mode="bilinear", align_corners=False)[0, 0].numpy()
    colours = matplotlib.colormaps[cmap](heat)[..., :3]
    weight = alpha * heat[..., None]
    mixed = (1 - weight) * (np.asarray(view) / 255) + weight * colours
    return view, Image.fromarray((mixed * 255).round().astype(np.uint8))