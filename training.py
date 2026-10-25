"""
Training and evaluation code shared by every ANN experiment.

  part 1  scores, predict, evaluate     the metrics fixed in Question 1, applied identically to every model
  part 2  freeze, make_optimiser,       what may learn, how it learns, and one pass over the training images
          train_one_epoch
  part 3  run_experiment                one complete, logged experiment (best epoch chosen on val_wild)
"""
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score

from plantdata import PlantDataset, eval_transform, make_loader, train_transform
from plantmodel import CropMaskedResNet

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
    """Scores of the model on one split, plus inference time per image in milliseconds."""
    probs, labels, per_image = predict(model, loader, device)
    result = scores(labels.numpy(), probs.argmax(dim=1).numpy(), class_ids)
    result["loss"] = float(-torch.log(probs[torch.arange(len(labels)), labels].clamp_min(1e-8)).mean())
    result["ms_per_image"] = 1000 * per_image
    return result


#   part 2: training
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


#   part 3: one experiment
DEFAULTS = dict(depth=18, pretrained=True, freeze="backbone", augment=False, optimiser="sgd", lr=0.01,
                weight_decay=1e-4, batch_size=64, epochs=10, patience=5, class_weights="none",
                dropout=0.2, seed=42, num_workers=8)


def set_seed(seed):
    """Fix every random-number generator used in training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def balanced_weights(labels, n_classes):
    """Class weights n / (classes x n_c): rare classes count more in the loss (mean weight = 1)."""
    counts = torch.bincount(labels, minlength=n_classes).float()
    w = counts.sum() / (n_classes * counts.clamp_min(1))
    return w / w[counts > 0].mean()


def run_experiment(run_id, note, manifest, cache_dir, class_ids, crops, crop_mask,
                   device="cuda", results_dir="results", checkpoint_dir="checkpoints", **settings):
    """Train one configuration, keep the epoch with the best val_wild disease macro-F1, and log
    everything. Returns (summary dict, per-epoch history DataFrame)."""
    unknown = set(settings) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")
    cfg = {**DEFAULTS, **settings}
    run_dir = Path(results_dir) / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} exists: every run needs a new run_id (old results are never overwritten)")
    run_dir.mkdir(parents=True)
    Path(checkpoint_dir).mkdir(exist_ok=True)
    checkpoint = Path(checkpoint_dir) / f"{run_id}.pt"
    (run_dir / "config.json").write_text(json.dumps({"run_id": run_id, "note": note, **cfg}, indent=2))

    set_seed(cfg["seed"])
    train_ds = PlantDataset(manifest, "train", cache_dir, crops, train_transform(cfg["augment"]))
    train_loader = make_loader(train_ds, cfg["batch_size"], train=True, num_workers=cfg["num_workers"],
                               seed=cfg["seed"])
    eval_workers = max(1, cfg["num_workers"] // 2)
    val_loaders = {s: make_loader(PlantDataset(manifest, s, cache_dir, crops, eval_transform()), 128,
                                  train=False, num_workers=eval_workers) for s in ("val_lab", "val_wild")}

    model = CropMaskedResNet(cfg["depth"], crop_mask, cfg["dropout"], cfg["pretrained"]).to(device)
    frozen = freeze(model, cfg["freeze"])
    optimiser = make_optimiser(model, cfg["optimiser"], cfg["lr"], cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))
    weights = (balanced_weights(train_ds.labels, len(class_ids)).to(device)
               if cfg["class_weights"] == "balanced" else None)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{run_id}: {note} | {n_trainable:,} trainable weights")

    history, best, best_epoch, start = [], -1.0, 0, time.perf_counter()
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.perf_counter()
        lr = optimiser.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(model, train_loader, optimiser, scaler, device, frozen, weights)
        scheduler.step()
        lab = evaluate(model, val_loaders["val_lab"], class_ids, device)
        wild = evaluate(model, val_loaders["val_wild"], class_ids, device)
        row = {"epoch": epoch, "lr": lr, "train_loss": train_loss, "train_acc": train_acc,
               **{f"val_lab_{k}": v for k, v in lab.items()}, **{f"val_wild_{k}": v for k, v in wild.items()},
               "epoch_seconds": time.perf_counter() - t0}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        improved = wild["macro_f1_16_disease"] > best
        if improved:
            best, best_epoch = wild["macro_f1_16_disease"], epoch
            torch.save({"state_dict": model.state_dict(), "config": cfg, "class_ids": class_ids,
                        "crops": crops, "epoch": epoch}, checkpoint)
        print(f"  epoch {epoch:2d} | lr {lr:.2e} | train loss {train_loss:.3f} acc {train_acc:.3f} | "
              f"val_lab F1-16 {lab['macro_f1_16_disease']:.3f} | val_wild F1-16 {wild['macro_f1_16_disease']:.3f}"
              f"{' *' if improved else ''} | {row['epoch_seconds']:.0f} s")
        if epoch - best_epoch >= cfg["patience"]:
            print(f"  early stop: no val_wild improvement for {cfg['patience']} epochs")
            break

    b = history[best_epoch - 1]
    summary = {"run_id": run_id, "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "note": note, **cfg,
               "trainable_weights": n_trainable, "epochs_run": len(history), "best_epoch": best_epoch,
               "val_lab_f1_16": b["val_lab_macro_f1_16_disease"],
               "val_wild_f1_16": b["val_wild_macro_f1_16_disease"], "val_wild_f1_20": b["val_wild_macro_f1_20"],
               "val_wild_late_blight_recall": b["val_wild_late_blight_recall"],
               "val_wild_accuracy": b["val_wild_accuracy"],
               "train_minutes": (time.perf_counter() - start) / 60, "ms_per_image": b["val_wild_ms_per_image"]}
    summary_file = Path(results_dir) / "runs_summary.csv"
    old = pd.read_csv(summary_file) if summary_file.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame([summary])], ignore_index=True).to_csv(summary_file, index=False)
    print(f"  best: epoch {best_epoch}, val_wild F1-16 {best:.3f} (checkpoint {checkpoint})")
    return summary, pd.DataFrame(history)