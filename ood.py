"""
Out-of-distribution (OOD) warning (Question 4): "this photo does not look like anything the model
was trained on".

Three familiarity scores are compared, each computed from the model's raw scores (logits) for the
classes of the selected crop; for all three, HIGHER = more familiar:
  msp        largest calibrated probability (logits / T, softmax)
  max_logit  largest raw logit
  energy     log-sum-exp of the raw logits (uses all of the crop's classes, not only the top one)

The threshold is chosen on validation data only: in-scope val_wild images (PlantDoc, the 20 classes)
against out-of-scope PlantDoc images (the excluded crops), so the test sets are never touched.
"""
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset


class PathDataset(Dataset):
    """Images from a list of file paths, for photos that are not in the split manifests."""

    def __init__(self, paths, transform):
        self.paths, self.transform = list(paths), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            image = im.convert("RGB")
        return self.transform(image), 0, 0          # crop and label are not used here


@torch.no_grad()
def raw_logits(model, loader, device="cuda"):
    """Logits for all 20 classes, BEFORE the crop mask, so any crop can be applied afterwards."""
    model.eval()
    out = []
    for images, *_ in loader:
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            z = model.head(model.backbone(images.to(device, non_blocking=True)))
        out.append(z.float().cpu())
    return torch.cat(out)


def familiarity_scores(raw, crop_idx, crop_mask, temperature):
    """The three scores for each image, using only the classes of its crop."""
    z = raw.masked_fill(~crop_mask[crop_idx], float("-inf"))
    return {"msp": (z / temperature).softmax(dim=1).max(dim=1).values,
            "max_logit": z.max(dim=1).values,
            "energy": torch.logsumexp(z, dim=1)}


def separation(in_scores, out_scores, keep=0.95):
    """AUROC, and the threshold that keeps `keep` of in-scope images (with the share of OOD images it
    fails to flag: the false-positive rate at that true-positive rate)."""
    s_in, s_out = np.asarray(in_scores, dtype=float), np.asarray(out_scores, dtype=float)
    y = np.r_[np.ones(len(s_in)), np.zeros(len(s_out))]
    threshold = float(np.quantile(s_in, 1 - keep))
    return {"auroc": float(roc_auc_score(y, np.r_[s_in, s_out])),
            "threshold": threshold,
            "in_scope_kept": float((s_in >= threshold).mean()),
            "ood_missed": float((s_out >= threshold).mean())}