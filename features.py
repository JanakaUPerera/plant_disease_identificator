"""
Hand-crafted colour and texture features for the classical baseline (Question 3.a).

Every image is first cropped like the ANN's evaluation input (shorter side 224, central 224 x 224
square), so the classical model and the ANN see the same region of every image. Then three feature groups
are computed:

  colour  34 numbers  HSV histograms: hue 18 bins, saturation 8, value 8 (each sums to 1)
  lbp     28 numbers  local binary pattern histograms of the grey image: fine surface texture
                      (8 neighbours at radius 1 -> 10 bins, 16 neighbours at radius 2 -> 18 bins)
  glcm     8 numbers  grey-level co-occurrence statistics (contrast, homogeneity, energy, correlation)
                      at pixel distances 1 and 3, averaged over 4 directions
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from torchvision.transforms import v2 as T

GROUPS = {"colour": slice(0, 34), "lbp": slice(34, 62), "glcm": slice(62, 70)}
N_FEATURES = 70
_crop = T.Compose([T.Resize(224, antialias=True), T.CenterCrop(224)])   # same view as eval_transform


def _hist(values, bins, value_range):
    h, _ = np.histogram(values, bins=bins, range=value_range)
    return h / max(h.sum(), 1)


def image_features(path):
    """The 70 features of one image (float32)."""
    with Image.open(path) as im:
        im = _crop(im.convert("RGB"))
    hsv = np.asarray(im.convert("HSV"))
    grey = np.asarray(im.convert("L"))

    colour = np.concatenate([_hist(hsv[..., 0], 18, (0, 256)),
                             _hist(hsv[..., 1], 8, (0, 256)),
                             _hist(hsv[..., 2], 8, (0, 256))])

    lbp1 = local_binary_pattern(grey, P=8, R=1, method="uniform")      # values 0..9
    lbp2 = local_binary_pattern(grey, P=16, R=2, method="uniform")     # values 0..17
    lbp = np.concatenate([_hist(lbp1, 10, (0, 10)), _hist(lbp2, 18, (0, 18))])

    grey32 = (grey // 8).astype(np.uint8)                              # 32 grey levels keep the matrix small
    glcm = graycomatrix(grey32, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32, symmetric=True, normed=True)
    tex = np.concatenate([graycoprops(glcm, p).mean(axis=1)            # average over the 4 directions
                          for p in ("contrast", "homogeneity", "energy", "correlation")])

    return np.concatenate([colour, lbp, tex]).astype(np.float32)


def extract(paths, workers=8):
    """Features of many images, computed in parallel. Returns an array (images x 70)."""
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return np.stack(list(ex.map(image_features, paths, chunksize=64)))


def save_split(dataset, split, out_dir="features", workers=8):
    """Compute and save the features of one split (a PlantDataset), with its labels and crop numbers."""
    Path(out_dir).mkdir(exist_ok=True)
    X = extract(dataset.paths, workers)
    np.savez_compressed(Path(out_dir) / f"{split}.npz", X=X, y=dataset.labels.numpy(),
                        crop=dataset.crop_idx.numpy())
    return X


def load_split(split, out_dir="features"):
    d = np.load(Path(out_dir) / f"{split}.npz")
    return d["X"], d["y"], d["crop"]