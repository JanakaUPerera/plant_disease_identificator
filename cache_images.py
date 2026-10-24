#!/usr/bin/env python3
"""
Resize every image in the fixed split once, so training and evaluation read small files fast.

For each image listed in splits/all_splits.csv this script:
  - applies the EXIF orientation and converts to RGB,
  - resizes so the SHORTER side is --size pixels (default 256), keeping the aspect ratio
    (models later take 224-pixel crops from these),
  - saves it as a JPEG under <cache>/<split>/<class_id>/<id>.jpg,
  - records simple image statistics used in the EDA (Question 1):
      brightness      mean grey level, 0-255
      saturation      mean HSV saturation, 0-255
      border_std      standard deviation of grey levels in a 16-pixel border strip:
                      low = plain, uniform background (typical of lab photos)
      border_green    share of border pixels that are plant-coloured (green-yellow hue,
                      saturated, not dark): near 0 = plain backdrop, high = foliage or field
                      around the leaf

Output: <cache>/cached_manifest.csv = all_splits.csv + cache_path + the statistics.
Re-running skips images already cached, so it is safe to stop and restart.

The cache contains resized copies of PlantWild images (CC BY-NC-ND): keep it on your own
computer for training only and never include it in the submission.

Usage
  python cache_images.py --splits splits --cache cache_256
  python cache_images.py --splits splits --cache cache_256 --root "D:/MSc/AI Datasets/plant_audit"
"""
import argparse
import hashlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None
BORDER = 16


def cache_name(dataset, rel_path):
    """Short, collision-free, OS-independent file name for one source image."""
    return hashlib.sha1(f"{dataset}|{rel_path}".encode("utf-8")).hexdigest()[:20] + ".jpg"


def image_stats(im):
    """Brightness, saturation, border uniformity and edge density of a resized RGB image."""
    grey = np.asarray(im.convert("L"), dtype=np.float32)
    sat = np.asarray(im.convert("HSV"), dtype=np.float32)[..., 1]
    b = min(BORDER, grey.shape[0] // 4, grey.shape[1] // 4)
    border = np.concatenate([grey[:b].ravel(), grey[-b:].ravel(), grey[:, :b].ravel(), grey[:, -b:].ravel()])
    hsv = np.asarray(im.convert("HSV"), dtype=np.float32)
    strip = np.concatenate([hsv[:b].reshape(-1, 3), hsv[-b:].reshape(-1, 3),
                            hsv[:, :b].reshape(-1, 3), hsv[:, -b:].reshape(-1, 3)])
    hue, s_, v_ = strip[:, 0], strip[:, 1], strip[:, 2]
    # PIL hue is 0-255 for 0-360 degrees: 40-120 is roughly 56-170 degrees (yellow-green to green-cyan)
    plant = (hue >= 40) & (hue <= 120) & (s_ >= 60) & (v_ >= 40)
    return {
        "brightness": float(grey.mean()),
        "saturation": float(sat.mean()),
        "border_std": float(border.std()),
        "border_green": float(plant.mean()),
    }


def process(job):
    src, dst, size, quality = job
    out = {"cache_ok": False, "cache_error": ""}
    try:
        dst = Path(dst)
        if dst.exists():
            with Image.open(dst) as im:
                im = im.convert("RGB")
                out.update(image_stats(im))
                out["cache_w"], out["cache_h"] = im.size
        else:
            with Image.open(src) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                w, h = im.size
                scale = size / min(w, h)
                if scale < 1:  # only shrink; small images are kept at their size
                    im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_suffix(".tmp")
                im.save(tmp, "JPEG", quality=quality)
                os.replace(tmp, dst)  # atomic: an interrupted run never leaves a half-written file
                out.update(image_stats(im))
                out["cache_w"], out["cache_h"] = im.size
        out["cache_ok"] = True
    except Exception as exc:
        out["cache_error"] = f"{type(exc).__name__}: {exc}"[:200]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", type=Path, default=Path("splits"), help="folder written by make_splits.py")
    ap.add_argument("--cache", type=Path, default=Path("cache_256"), help="output folder")
    ap.add_argument("--root", type=Path, default=Path("."),
                    help="folder the manifest paths are relative to (where make_splits.py was run)")
    ap.add_argument("--size", type=int, default=256, help="shorter side in pixels (default 256)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality (default 95)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    manifest = args.splits / "all_splits.csv"
    if not manifest.exists():
        sys.exit(f"{manifest} not found: run make_splits.py first")
    df = pd.read_csv(manifest, dtype={"leaf_id": str}, keep_default_na=False)
    print(f"{len(df):,} images in {manifest}")

    def source(p):
        p = Path(str(p).replace("\\", "/"))
        return p if p.is_absolute() else args.root / p

    df["cache_path"] = [f"{s}/{c}/{cache_name(d, r)}" for s, c, d, r in zip(df.split, df.class_id, df.dataset,
                                                                            df.rel_path)]
    jobs = [(str(source(p)), str(args.cache / cp), args.size, args.quality) for p, cp in zip(df.path, df.cache_path)]
    todo = sum(not (args.cache / cp).exists() for cp in df.cache_path)
    print(f"{len(df) - todo:,} already cached, {todo:,} to resize with {args.workers} workers")

    t0, results = time.time(), []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process, jobs, chunksize=64), 1):
            results.append(r)
            if i % 2000 == 0 or i == len(jobs):
                print(f"  {i:,}/{len(jobs):,} ({time.time() - t0:.0f}s)")
    stats = pd.DataFrame(results)
    df = pd.concat([df.reset_index(drop=True), stats], axis=1)
    bad = df[~df.cache_ok]
    if len(bad):
        print(f"WARNING: {len(bad)} images could not be cached; see cache_errors.csv")
        bad[["split", "class_id", "path", "cache_error"]].to_csv(args.cache / "cache_errors.csv", index=False)
    args.cache.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.cache / "cached_manifest.csv", index=False)
    size_mb = sum(f.stat().st_size for f in args.cache.rglob("*.jpg")) / 1e6
    print(f"Done: {int(df.cache_ok.sum()):,} images cached ({size_mb:,.0f} MB) -> {args.cache / 'cached_manifest.csv'}")


if __name__ == "__main__":
    main()
