"""
Data loading for the plant-disease models.

Reads the cached images (cache_256, made by cache_images.py) of one split of splits_v2, and returns
for each image: the image tensor, its crop number (for the crop mask) and its class number.

It lives in a .py file, not in the notebook, because on Windows the DataLoader's worker processes
can only use code they can import from a file.
"""
import json
from pathlib import Path
from torchvision.transforms import v2 as T

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # the colour statistics the ResNet weights were trained with
IMAGENET_STD = (0.229, 0.224, 0.225)

def load_classes(splits_dir):
    """Class names, crop names and the crop mask (crops x classes), in the order of classes.json."""
    classes = sorted(json.loads((Path(splits_dir) / "classes.json").read_text()), key=lambda c: c["index"])
    class_ids = [c["id"] for c in classes]
    crops = sorted({c["crop"] for c in classes})
    crop_mask = torch.zeros(len(crops), len(class_ids), dtype=torch.bool)
    for c in classes:
        crop_mask[crops.index(c["crop"]), c["index"]] = True
    return class_ids, crops, crop_mask


def read_manifest(cache_dir):
    """The table of all cached images (one row per image), written by cache_images.py."""
    return pd.read_csv(Path(cache_dir) / "cached_manifest.csv", dtype={"leaf_id": str}, keep_default_na=False)


class PlantDataset(Dataset):
    """The images of one split. dataset[i] gives (image, crop number, class number)."""

    def __init__(self, manifest, split, cache_dir, crops, transform=None):
        df = manifest[manifest["split"] == split].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"no images for split '{split}' in the manifest")
        self.paths = [str(Path(cache_dir) / p) for p in df["cache_path"]]
        self.labels = torch.tensor(df["class_index"].to_numpy(), dtype=torch.long)
        self.crop_idx = torch.tensor([crops.index(c) for c in df["crop"]], dtype=torch.long)
        self.info = df[["class_id", "crop", "dataset", "source_split", "rel_path"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            image = im.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.crop_idx[i], self.labels[i]

def eval_transform():
    """Validation, test and the web app: shorter side to 224, then the central 224 x 224 square."""
    return T.Compose([
        T.ToImage(),
        T.Resize(IMG_SIZE, antialias=True),
        T.CenterCrop(IMG_SIZE),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def train_transform(augment=True, min_scale=0.5, colour=0.3):
    """Training images. augment=False gives exactly the evaluation transform (no random changes)."""
    if not augment:
        return eval_transform()
    return T.Compose([
        T.ToImage(),
        T.RandomResizedCrop(IMG_SIZE, scale=(min_scale, 1.0), ratio=(3 / 4, 4 / 3), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=colour, contrast=colour, saturation=colour, hue=0.02),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
def make_loader(dataset, batch_size, train, num_workers=4, seed=0):
    """Batches of (images, crop numbers, class numbers). Training loaders shuffle (with a fixed seed)
    and drop the last incomplete batch; evaluation loaders keep the original order and every image."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=train,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0, generator=generator)