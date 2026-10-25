"""
Training and evaluation code shared by every ANN experiment.

Scoring: Every model is scored with the same scores() function on the same splits, so all
results in the experiment log are comparable.
"""
import time

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score


def scores(y_true, y_pred, class_ids):
    """The metrics fixed in Question 1. y_true and y_pred are class numbers (0-19)."""
    disease = [i for i, c in enumerate(class_ids) if not c.endswith("_healthy")]
    late_blight = [class_ids.index(c) for c in ("potato_late_blight", "tomato_late_blight")]
    return {
        "macro_f1_16_disease": f1_score(y_true, y_pred, labels=disease, average="macro", zero_division=0),
        "macro_f1_20": f1_score(y_true, y_pred, labels=list(range(len(class_ids))), average="macro",
                                zero_division=0),
        "late_blight_recall": recall_score(y_true, y_pred, labels=late_blight, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


@torch.no_grad()
def predict(model, loader, device="cuda"):
    """Probabilities (images x 20), true classes, and the model's own time per image in seconds.
    Only the model is timed (not image loading), and the first batch is skipped because it includes
    one-off GPU start-up work."""
    model.eval()
    probs, labels = [], []
    model_seconds, timed_images = 0.0, 0
    for b, (images, crop_b, y) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        crop_b = crop_b.to(device, non_blocking=True)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(images, crop_b)
        if device == "cuda":
            torch.cuda.synchronize()
        if b > 0:
            model_seconds += time.perf_counter() - start
            timed_images += len(y)
        probs.append(logits.float().softmax(dim=1).cpu())
        labels.append(y)
    per_image = model_seconds / timed_images if timed_images else float("nan")
    return torch.cat(probs), torch.cat(labels), per_image


def evaluate(model, loader, class_ids, device="cuda"):
    """Scores of the model on one split, plus the model's inference time per image in milliseconds."""
    probs, labels, per_image = predict(model, loader, device)
    result = scores(labels.numpy(), probs.argmax(dim=1).numpy(), class_ids)
    result["ms_per_image"] = 1000 * per_image
    return result


def freeze(model, part):
    """Which weights may change. part = "none" (train everything), "backbone" (only the head;
    the 'initial ANN'), or "stages1-3" (stem and stages 1-3 frozen; stage 4 and the head train)."""
    bb = model.backbone
    frozen = {"none": [], "backbone": [bb],
              "stages1-3": [bb.conv1, bb.bn1, bb.layer1, bb.layer2, bb.layer3]}[part]
    for p in model.parameters():
        p.requires_grad = True
    for module in frozen:
        for p in module.parameters():
            p.requires_grad = False
    return frozen


def make_optimiser(model, name, lr, weight_decay):
    """SGD with momentum, or AdamW, over the weights that are allowed to change."""
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimiser {name}")


def train_one_epoch(model, loader, optimiser, scaler, device="cuda", frozen=(), class_weights=None):
    """One pass over the training images. Returns the mean training loss and training accuracy."""
    model.train()
    for module in frozen:          # frozen parts keep their ImageNet BatchNorm statistics
        module.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for images, crop_b, y in loader:
        images = images.to(device, non_blocking=True)
        crop_b = crop_b.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(images, crop_b)
            loss = F.cross_entropy(logits, y, weight=class_weights)
        optimiser.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        seen += len(y)
    return total_loss / seen, correct / seen