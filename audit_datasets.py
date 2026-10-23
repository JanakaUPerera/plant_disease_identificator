#!/usr/bin/env python3
"""
Pre-proposal data audit for the plant-disease coursework.

For PlantVillage (A), PlantDoc (B) and PlantWild (C) it produces the outputs below.
PlantWild v1 and v2 are audited side by side as two datasets, because the in-the-wild test
set takes its disease classes from v2 and its healthy classes from v1 (see make_splits.py).
  1. class_counts.csv        images per canonical class in every split, with
                             flags for classes below --min-count in an evaluation set
  2. unmapped_folders.csv    dataset folders not in class_mapping.json, with the
                             closest canonical class suggested (never auto-accepted)
  3. duplicates.csv          exact and near-duplicate image pairs (perceptual hash)
                             across and within datasets, with a label-conflict flag
  4. duplicate_summary.csv   pairs and affected images per comparison
  5. image_quality.csv       corrupt, tiny and greyscale images per dataset
  6. contact_sheets/*.jpg    sample images per class from each dataset, to check the
                             approximate mappings by eye
  7. duplicate_pairs/*.jpg   side-by-side previews of detected duplicate pairs
  8. audit_summary.md        everything above in one readable report

Usage
  python audit_datasets.py --plantvillage data/plantvillage \
                           --plantdoc data/plantdoc \
                           --plantwild data/plantwild

Any dataset can be left out; the comparisons involving it are skipped. Both PlantWild releases
are audited when both are found under --plantwild (--plantwild-version both, the default).
Hashes are cached in <out>/hash_cache.csv, so re-runs are fast.
"""
import argparse
import difflib
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

try:
    import imagehash
except ImportError:
    sys.exit("imagehash is not installed: pip install -r requirements.txt")

Image.MAX_IMAGE_PIXELS = None  # some web images are very large; they are only read, not trusted
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif", ".jfif"}
SEED = 42

DS_PV, DS_PD, DS_PW = "plantvillage", "plantdoc", "plantwild"
# the audit keeps the two PlantWild releases apart; both use the "plantwild" folder names in the mapping
DS_PW1, DS_PW2 = "plantwild_v1", "plantwild_v2"
DS_LABEL = {DS_PV: "PlantVillage", DS_PD: "PlantDoc", DS_PW: "PlantWild",
            DS_PW1: "PlantWild v1", DS_PW2: "PlantWild v2"}
MAP_KEY = {DS_PV: DS_PV, DS_PD: DS_PD, DS_PW1: DS_PW, DS_PW2: DS_PW}  # dataset -> key in class_mapping.json
COL_PREFIX = {DS_PV: "PV", DS_PD: "PD", DS_PW1: "PW1", DS_PW2: "PW2"}

# PlantWild v1 class folders (from the authors' MVPDR repository prompt file).
# Used to locate the v1 images folder and to report PlantWild classes outside the mapping.
PLANTWILD_V1_CLASSES = [
    "apple black rot", "apple leaf", "apple mosaic virus", "apple rust", "apple scab", "banana leaf",
    "banana panama disease", "basil downy mildew", "basil leaf", "bean halo blight", "bean leaf",
    "bean mosaic virus", "bean rust", "bell pepper leaf", "bell pepper leaf spot", "blueberry leaf",
    "blueberry rust", "broccoli downy mildew", "broccoli leaf", "cabbage alternaria leaf spot",
    "cabbage leaf", "carrot cavity spot", "cauliflower alternaria leaf spot", "cauliflower leaf",
    "celery anthracnose", "celery early blight", "celery leaf", "cherry leaf", "cherry leaf spot",
    "cherry powdery mildew", "citrus canker", "citrus greening disease", "coffee leaf",
    "coffee leaf rust", "corn gray leaf spot", "corn leaf", "corn northern leaf blight", "corn rust",
    "corn smut", "cucumber angular leaf spot", "cucumber bacterial wilt", "cucumber leaf",
    "cucumber powdery mildew", "eggplant cercospora leaf spot", "eggplant leaf", "garlic leaf",
    "garlic leaf blight", "garlic rust", "ginger leaf", "ginger leaf spot", "ginger sheath blight",
    "grape black rot", "grape downy mildew", "grape leaf", "grape leaf spot",
    "grapevine leafroll disease", "lettuce downy mildew", "lettuce leaf", "lettuce mosaic virus",
    "maple leaf", "maple tar spot", "peach leaf", "peach leaf curl", "plum leaf",
    "plum pocket disease", "potato early blight", "potato late blight", "potato leaf",
    "raspberry leaf", "rice blast", "rice leaf", "rice sheath blight", "soybean leaf", "squash leaf",
    "squash powdery mildew", "strawberry anthracnose", "strawberry leaf", "strawberry leaf scorch",
    "tobacco leaf", "tobacco mosaic virus", "tomato bacterial leaf spot", "tomato early blight",
    "tomato late blight", "tomato leaf", "tomato leaf mold", "tomato mosaic virus",
    "tomato septoria leaf spot", "tomato yellow leaf curl virus", "zucchini yellow mosaic virus",
]

# PlantWild v2 class folders: 115 disease-only classes, no healthy classes. Taken from the
# authors' companion PlantSeg repository (plantseg115.py), which uses the same 115 classes.
PLANTWILD_V2_CLASSES = [
    "apple black rot", "apple mosaic virus", "apple rust", "apple scab", "banana anthracnose",
    "banana black leaf streak", "banana bunchy top", "banana cigar end rot", "banana cordana leaf spot",
    "banana panama disease", "basil downy mildew", "bean halo blight", "bean mosaic virus", "bean rust",
    "bell pepper bacterial spot", "bell pepper blossom end rot", "bell pepper frogeye leaf spot",
    "bell pepper powdery mildew", "blueberry anthracnose", "blueberry botrytis blight",
    "blueberry mummy berry", "blueberry rust", "blueberry scorch", "broccoli alternaria leaf spot",
    "broccoli downy mildew", "broccoli ring spot", "cabbage alternaria leaf spot", "cabbage black rot",
    "cabbage downy mildew", "carrot alternaria leaf blight", "carrot cavity spot",
    "carrot cercospora leaf blight", "cauliflower alternaria leaf spot", "cauliflower bacterial soft rot",
    "celery anthracnose", "celery early blight", "cherry leaf spot", "cherry powdery mildew",
    "citrus canker", "citrus greening disease", "coffee berry blotch", "coffee black rot",
    "coffee brown eye spot", "coffee leaf rust", "corn gray leaf spot", "corn northern leaf blight",
    "corn rust", "corn smut", "cucumber angular leaf spot", "cucumber bacterial wilt",
    "cucumber powdery mildew", "eggplant cercospora leaf spot", "eggplant phomopsis fruit rot",
    "eggplant phytophthora blight", "garlic leaf blight", "garlic rust", "ginger leaf spot",
    "ginger sheath blight", "grape black rot", "grape downy mildew", "grape leaf spot",
    "grapevine leafroll disease", "lettuce downy mildew", "lettuce mosaic virus", "maple tar spot",
    "peach anthracnose", "peach brown rot", "peach leaf curl", "peach rust", "peach scab",
    "plum bacterial spot", "plum brown rot", "plum pocket disease", "plum pox virus", "plum rust",
    "potato early blight", "potato late blight", "raspberry fire blight", "raspberry gray mold",
    "raspberry leaf spot", "raspberry yellow rust", "rice blast", "rice sheath blight",
    "soybean bacterial blight", "soybean brown spot", "soybean downy mildew", "soybean frog eye leaf spot",
    "soybean mosaic", "soybean rust", "squash powdery mildew", "strawberry anthracnose",
    "strawberry leaf scorch", "tobacco blue mold", "tobacco brown spot", "tobacco frogeye leaf spot",
    "tobacco mosaic virus", "tomato bacterial leaf spot", "tomato early blight", "tomato late blight",
    "tomato leaf mold", "tomato mosaic virus", "tomato septoria leaf spot", "tomato yellow leaf curl virus",
    "wheat bacterial leaf streak (black chaff)", "wheat head scab", "wheat leaf rust", "wheat loose smut",
    "wheat powdery mildew", "wheat septoria blotch", "wheat stem rust", "wheat stripe rust",
    "zucchini bacterial wilt", "zucchini downy mildew", "zucchini powdery mildew",
    "zucchini yellow mosaic virus",
]


# --------------------------------------------------------------------------- helpers
def norm(name: str) -> str:
    """Case-, underscore- and punctuation-insensitive folder-name key."""
    s = name.lower().replace("_", " ").replace(",", " ")
    return re.sub(r"\s+", " ", s).strip()


def folder_names(cls: dict, ds: str) -> list:
    """Folder name(s) of a class in one dataset; the mapping may give a string or a list of aliases."""
    v = cls.get(ds)
    return [] if not v else ([v] if isinstance(v, str) else list(v))


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXT and not p.name.startswith(".")


def list_images(folder: Path):
    return sorted(p for p in folder.rglob("*") if is_image(p))


def child_dirs(p: Path):
    try:
        return [c for c in p.iterdir() if c.is_dir() and not c.name.startswith(".")]
    except (PermissionError, FileNotFoundError):
        return []


def class_root_candidates(root: Path, known: set, max_depth: int = 6):
    """All directories whose sub-folder names overlap with `known`, best first: [(score, dir), ...]."""
    found = []
    root_depth = len(root.parts)
    for dirpath, dirnames, _ in os.walk(root):
        d = Path(dirpath)
        if len(d.parts) - root_depth > max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = [n for n in dirnames if not n.startswith(".")]
        score = sum(norm(n) in known for n in dirnames)
        if score:
            found.append((score, d))
    # highest overlap first; on ties prefer folders literally called "images" (PlantWild layout)
    return sorted(found, key=lambda t: (-t[0], t[1].name != "images", str(t[1])))


def best_class_root(root: Path, known: set, max_depth: int = 6):
    """Find the directory whose sub-folder names overlap most with `known` class names."""
    found = class_root_candidates(root, known, max_depth)
    return (found[0][1], found[0][0]) if found else (None, 0)


# --------------------------------------------------------------------------- indexing
def index_plantvillage(root: Path, known: set):
    color = root / "raw" / "color"
    class_root = color if color.is_dir() else best_class_root(root, known)[0]
    if class_root is None:
        sys.exit(f"Could not find PlantVillage class folders under {root}")
    rows = []
    for c in child_dirs(class_root):
        for p in list_images(c):
            rows.append((DS_PV, "all", c.name, str(p)))
    return rows, class_root


def index_plantdoc(root: Path, known: set):
    if not (root / "train").is_dir():
        found = [d for d in root.rglob("train") if d.is_dir() and (d.parent / "test").is_dir()]
        if not found:
            sys.exit(f"Could not find PlantDoc train/ and test/ folders under {root}")
        root = found[0].parent
    rows = []
    for split in ("train", "test"):
        for c in child_dirs(root / split):
            for p in list_images(c):
                rows.append((DS_PD, split, c.name, str(p)))
    return rows, root


def read_plantwild_split(class_root: Path):
    """Parse the official v1 trainval.txt (lines: path=label=domain; 1=train, 2=val, else test).
    Only the images folder and its parent are searched, so a v1 split file is never applied to v2."""
    candidates = [f for f in (class_root / "trainval.txt", class_root.parent / "trainval.txt") if f.is_file()]
    if not candidates:
        return {}, None
    split = {}
    for line in candidates[0].read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split("=")
        if len(parts) < 3:
            continue
        p = Path(parts[0].replace("\\", "/"))
        key = (norm(p.parent.name), p.name)
        split[key] = {"1": "train", "2": "val"}.get(parts[-1].strip(), "test")
    return split, candidates[0]


PW_RELEASE_NAMES = {"v1": {norm(n) for n in PLANTWILD_V1_CLASSES},
                    "v2": {norm(n) for n in PLANTWILD_V2_CLASSES}}
PW_HEALTHY = {"tomato leaf", "apple leaf", "grape leaf", "corn leaf"}  # only v1 has healthy classes


def plantwild_release(d: Path) -> str:
    return "v1" if {norm(c.name) for c in child_dirs(d)} & PW_HEALTHY else "v2"


def plantwild_releases(root: Path, known: set) -> list:
    """PlantWild releases ("v1", "v2") that have an images folder under `root`."""
    cands = class_root_candidates(root, PW_RELEASE_NAMES["v1"] | PW_RELEASE_NAMES["v2"] | known)
    return sorted({plantwild_release(d) for _, d in cands})


def index_plantwild(root: Path, known: set, version: str = "auto"):
    release = plantwild_release
    candidates = class_root_candidates(root, PW_RELEASE_NAMES["v1"] | PW_RELEASE_NAMES["v2"] | known)
    if not candidates:
        sys.exit(f"Could not find PlantWild class folders under {root}")
    if version in ("v1", "v2"):
        wanted = [d for _, d in candidates if release(d) == version]
        if not wanted:
            sys.exit(f"No PlantWild {version} images folder found under {root} "
                     f"(found: {', '.join(f'{release(d)} at {d}' for _, d in candidates[:3])})")
        class_root = wanted[0]
    else:
        class_root = candidates[0][1]
        releases = sorted({release(d) for _, d in candidates[:4]})
        if len(releases) > 1:
            print(f"  Both PlantWild releases found; using {release(class_root)} at {class_root}. "
                  f"Choose explicitly with --plantwild-version v1 or v2.")
    detected = release(class_root)
    version_note = ("v1 (89 classes, includes healthy classes)" if detected == "v1" else
                    "v2 (disease-only classes: healthy classes cannot be tested on PlantWild)")
    split_map, split_file = read_plantwild_split(class_root) if detected == "v1" else ({}, None)
    rows = []
    for c in child_dirs(class_root):
        for p in list_images(c):
            s = split_map.get((norm(c.name), p.name), "unknown") if split_map else "unknown"
            rows.append((DS_PW, s, c.name, str(p)))
    return rows, class_root, version_note, split_file, detected


# --------------------------------------------------------------------------- hashing
def hash_one(path: str):
    out = {"path": path, "ok": False, "width": np.nan, "height": np.nan, "mode": "",
           "phash": "", "dhash": "", "error": ""}
    try:
        with Image.open(path) as im:
            out["mode"] = im.mode
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            out["width"], out["height"] = im.size
            out["phash"] = str(imagehash.phash(im))
            out["dhash"] = str(imagehash.dhash(im))
            out["ok"] = True
    except Exception as exc:  # corrupt / unsupported file
        out["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return out


def compute_hashes(paths, cache_file: Path, workers: int):
    cache = {}
    if cache_file.exists():
        old = pd.read_csv(cache_file, dtype={"phash": str, "dhash": str}, keep_default_na=False)
        for r in old.to_dict("records"):
            cache[r["path"]] = r
    todo, results = [], []
    for p in paths:
        st = os.stat(p)
        c = cache.get(p)
        if c and int(c["size"]) == st.st_size and abs(float(c["mtime"]) - st.st_mtime) < 1:
            results.append(c)
        else:
            todo.append(p)
    print(f"  {len(results)} hashes from cache, {len(todo)} to compute with {workers} workers")
    t0 = time.time()
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, r in enumerate(ex.map(hash_one, todo, chunksize=32), 1):
                st = os.stat(r["path"])
                r["size"], r["mtime"] = st.st_size, st.st_mtime
                results.append(r)
                if i % 2000 == 0 or i == len(todo):
                    print(f"    {i}/{len(todo)} hashed ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(results)
    # normalise types (values read back from the cache CSV arrive as strings)
    for col in ("width", "height", "size", "mtime"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ok"] = df["ok"].map(lambda v: v is True or str(v) == "True")
    for col in ("phash", "dhash", "mode", "error"):
        df[col] = df[col].fillna("").astype(str)
    df.to_csv(cache_file, index=False)
    return df


def popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    return np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1).sum(-1)


def hex_to_u64(series: pd.Series) -> np.ndarray:
    return np.array([int(h, 16) if h else 0 for h in series], dtype=np.uint64)


def near_pairs(idx_a, idx_b, ph, dh, p_thr, d_thr, same=False, budget=8_000_000):
    """Pairs (i, j) with phash distance <= p_thr AND dhash distance <= d_thr."""
    idx_a, idx_b = np.asarray(idx_a), np.asarray(idx_b)
    if len(idx_a) == 0 or len(idx_b) == 0:
        return []
    B, Bd = ph[idx_b], dh[idx_b]
    chunk = max(1, budget // len(idx_b))
    out = []
    for s in range(0, len(idx_a), chunk):
        a = idx_a[s:s + chunk]
        dp = popcount(ph[a][:, None] ^ B[None, :])
        mask = dp <= p_thr
        if same:  # each unordered pair once, no self-pairs
            mask &= idx_b[None, :] > a[:, None]
        ii, jj = np.nonzero(mask)
        if len(ii) == 0:
            continue
        dd = popcount(dh[a[ii]] ^ Bd[jj])
        keep = dd <= d_thr
        for i, j, p, d in zip(a[ii][keep], idx_b[jj][keep], dp[ii, jj][keep], dd[keep]):
            out.append((int(i), int(j), int(p), int(d)))
    return out


# --------------------------------------------------------------------------- visuals
def thumb(path, size=128):
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            return ImageOps.fit(im, (size, size))
    except Exception:
        return Image.new("RGB", (size, size), (200, 50, 50))


def contact_sheet(class_id, rows_by_ds, out_file, n=6, size=128):
    label_w, gap = 160, 4
    ds_order = [d for d in (DS_PV, DS_PD, DS_PW2, DS_PW1) if d in rows_by_ds]
    W = label_w + n * (size + gap)
    H = 30 + len(ds_order) * (size + gap)
    sheet = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 8), f"{class_id}  (random sample, seed {SEED})", fill="black")
    rng = random.Random(SEED)
    for r, ds in enumerate(ds_order):
        y = 30 + r * (size + gap)
        paths = rows_by_ds[ds]
        draw.text((6, y + size // 2 - 12), f"{DS_LABEL[ds]}\n{paths[0][1][:24]}\nn={len(paths)}", fill="black")
        for k, (p, _) in enumerate(rng.sample(paths, min(n, len(paths)))):
            sheet.paste(thumb(p, size), (label_w + k * (size + gap), y))
    sheet.save(out_file, quality=88)


def pair_sheet(pairs_df, out_file, max_pairs=24, size=128):
    pairs_df = pairs_df.head(max_pairs)
    if pairs_df.empty:
        return
    cols = 4
    rows = (len(pairs_df) + cols - 1) // cols
    cell_w, cell_h = 2 * size + 12, size + 34
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for k, r in enumerate(pairs_df.itertuples()):
        x, y = (k % cols) * cell_w, (k // cols) * cell_h
        sheet.paste(thumb(r.path_a, size), (x, y))
        sheet.paste(thumb(r.path_b, size), (x + size + 4, y))
        flag = " CONFLICT" if r.label_conflict else ""
        draw.text((x + 2, y + size + 2), f"p={r.phash_dist} d={r.dhash_dist}{flag}", fill="red" if flag else "black")
        draw.text((x + 2, y + size + 16), f"{r.class_a[:18]} | {r.class_b[:18]}", fill="black")
    sheet.save(out_file, quality=88)


# --------------------------------------------------------------------------- main
def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plantvillage", type=Path, help="PlantVillage repo folder (contains raw/color)")
    ap.add_argument("--plantdoc", type=Path, help="PlantDoc repo folder (contains train/ and test/)")
    ap.add_argument("--plantwild", type=Path, help="PlantWild download folder (the images folder is found automatically)")
    ap.add_argument("--plantwild-version", choices=["both", "auto", "v1", "v2"], default="both",
                    help="PlantWild release(s) to audit: both (default; v1 and v2 as two datasets, "
                         "whichever are found), v1, v2, or auto (the one matching more classes)")
    ap.add_argument("--plantwild-v1", type=Path, help="PlantWild v1 folder, if it is not inside --plantwild")
    ap.add_argument("--mapping", type=Path, default=Path(__file__).with_name("class_mapping.json"))
    ap.add_argument("--out", type=Path, default=Path("audit_outputs"))
    ap.add_argument("--tiers", default="core,candidate", help="mapping tiers to audit (default: core,candidate)")
    ap.add_argument("--min-count", type=int, default=10,
                    help="minimum images per class in each evaluation set (default: 10)")
    ap.add_argument("--phash-threshold", type=int, default=6, help="max pHash Hamming distance (0-64, default 6)")
    ap.add_argument("--dhash-threshold", type=int, default=8, help="max dHash Hamming distance (0-64, default 8)")
    ap.add_argument("--pv-internal", action="store_true",
                    help="also search for near-duplicates inside PlantVillage (slow; many hits on plain backgrounds)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--no-sheets", action="store_true", help="skip contact sheets and duplicate previews")
    args = ap.parse_args()

    if not any([args.plantvillage, args.plantdoc, args.plantwild, args.plantwild_v1]):
        ap.error("give at least one of --plantvillage / --plantdoc / --plantwild")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    tiers = {t.strip() for t in args.tiers.split(",")}
    classes = [c for c in mapping["classes"] if c["tier"] in tiers]
    lookup = {ds: {norm(n): c["id"] for c in classes for n in folder_names(c, ds)} for ds in (DS_PV, DS_PD, DS_PW)}
    all_mapped = {ds: {norm(n) for c in mapping["classes"] for n in folder_names(c, ds)}
                  for ds in (DS_PV, DS_PD, DS_PW)}
    meta = {c["id"]: c for c in mapping["classes"]}

    # ---- 1. index
    print("1/6 Indexing image folders")
    rows, info = [], {}
    if args.plantvillage:
        r, croot = index_plantvillage(args.plantvillage, all_mapped[DS_PV])
        rows += r
        info[DS_PV] = f"class folders: `{croot}`"
    if args.plantdoc:
        r, croot = index_plantdoc(args.plantdoc, all_mapped[DS_PD])
        rows += r
        info[DS_PD] = f"train/test folders under: `{croot}`"
    # PlantWild: one (root, version) job per release to audit
    pw_jobs = []
    if args.plantwild_version == "both":
        for v, root in (("v1", args.plantwild_v1 or args.plantwild), ("v2", args.plantwild)):
            if root and v in plantwild_releases(root, all_mapped[DS_PW]):
                pw_jobs.append((root, v))
            elif root:
                print(f"  PlantWild {v} not found under {root}; auditing the other release only")
        if not pw_jobs and (args.plantwild or args.plantwild_v1):
            pw_jobs = [(args.plantwild or args.plantwild_v1, "auto")]  # exits with a clear message
    elif args.plantwild or args.plantwild_v1:
        v1_root = args.plantwild_v1 if args.plantwild_version == "v1" else None
        pw_jobs = [(v1_root or args.plantwild or args.plantwild_v1, args.plantwild_version)]
    for root, v in pw_jobs:
        r, croot, vnote, sfile, detected = index_plantwild(root, all_mapped[DS_PW], v)
        ds = DS_PW1 if detected == "v1" else DS_PW2
        rows += [(ds, *row[1:]) for row in r]
        info[ds] = (f"class folders: `{croot}`; {vnote}; official split file: "
                    f"{'`' + str(sfile) + '`' if sfile else 'not found (split reported as unknown)'}")
    pw_sets = [ds for ds in (DS_PW2, DS_PW1) if ds in info]  # PlantWild datasets audited, v2 first
    df = pd.DataFrame(rows, columns=["dataset", "split", "folder", "path"])
    df["class_id"] = [lookup[MAP_KEY[d]].get(norm(f)) for d, f in zip(df.dataset, df.folder)]
    for ds, txt in info.items():
        print(f"  {DS_LABEL[ds]}: {(df.dataset == ds).sum()} images; {txt}")
    df.to_csv(out / "image_index.csv", index=False)

    # ---- 2. unmapped folders and missing folders
    print("2/6 Checking the class mapping")
    canon_names = [c["id"] for c in mapping["classes"]]
    um = (df[df.class_id.isna()].groupby(["dataset", "folder"]).size().rename("n_images").reset_index())
    sugg, in_other_tier = [], []
    for d, f in zip(um.dataset, um.folder):
        mapped_other = norm(f) in all_mapped[MAP_KEY[d]]
        in_other_tier.append(mapped_other)
        m = difflib.get_close_matches(norm(f).replace(" ", "_"), canon_names, n=1, cutoff=0.0)
        sugg.append(m[0] if m else "")
    um["mapped_in_excluded_tier"] = in_other_tier
    um["closest_canonical_class"] = sugg
    um.to_csv(out / "unmapped_folders.csv", index=False)

    audited = list(info)
    present = {ds: {norm(f) for f in df[df.dataset == ds].folder.unique()} for ds in audited}
    release_names = {DS_PW1: PW_RELEASE_NAMES["v1"], DS_PW2: PW_RELEASE_NAMES["v2"]}
    not_in_release = {ds: set() for ds in pw_sets}  # mapped classes a PlantWild release does not have (healthy in v2)
    missing = []
    for c in classes:
        for ds in audited:
            names_ = folder_names(c, MAP_KEY[ds])
            if not names_ or any(norm(n) in present[ds] for n in names_):
                continue
            if ds in release_names and not any(norm(n) in release_names[ds] for n in names_):
                not_in_release[ds].add(c["id"])
                continue
            missing.append((c["id"], DS_LABEL[ds], " | ".join(names_)))
    for ds, ids in not_in_release.items():
        if ids:
            print(f"  {DS_LABEL[ds]} has no folder for {len(ids)} mapped classes: {', '.join(sorted(ids))}")
    pd.DataFrame(missing, columns=["class_id", "dataset", "expected_folder"]).to_csv(out / "missing_folders.csv", index=False)
    if missing:
        print(f"  WARNING: {len(missing)} mapped folders were not found on disk "
              f"(see missing_folders.csv; check spelling in class_mapping.json). First few:")
        for m in missing[:5]:
            print("   ", m)

    # ---- 3. counts
    print("3/6 Counting images per class")
    mapped = df[df.class_id.notna()].copy()
    mapped["column"] = mapped.dataset.map(COL_PREFIX) + "_" + mapped.split
    counts = mapped.pivot_table(index="class_id", columns="column", values="path", aggfunc="count", fill_value=0)
    counts = counts.reindex([c["id"] for c in classes], fill_value=0)
    for col in ["PV_all", "PD_train", "PD_test"]:
        if col not in counts:
            counts[col] = 0
    split_order = {"train": 0, "val": 1, "test": 2, "unknown": 3}
    pw_cols = []
    for ds in (DS_PW1, DS_PW2):
        pre = COL_PREFIX[ds]
        cols = sorted((c for c in counts.columns if c.startswith(pre + "_")),
                      key=lambda c: split_order.get(c.split("_", 1)[1], 9))
        counts[f"{pre}_total"] = counts[cols].sum(axis=1) if cols else 0
        if ds in pw_sets:
            # a release with no split file has only an "unknown" column, which would repeat the total
            pw_cols += ([] if cols == [f"{pre}_unknown"] else cols) + [f"{pre}_total"]
    # external test as built by make_splits.py: v2 images where the class exists in v2, otherwise v1
    use_v2 = counts.PW2_total > 0
    counts["PW_external"] = counts.PW2_total.where(use_v2, counts.PW1_total)
    counts["PW_external_from"] = np.where(use_v2, "v2", np.where(counts.PW1_total > 0, "v1", ""))
    counts.insert(0, "crop", [meta[i]["crop"] for i in counts.index])
    counts.insert(1, "tier", [meta[i]["tier"] for i in counts.index])
    counts.insert(2, "approximate_mapping", [meta[i].get("approximate", False) for i in counts.index])
    no_wild = {i for i in counts.index if pw_sets and counts.at[i, "PW_external"] == 0
               and all(i in not_in_release[ds] for ds in pw_sets)}  # no folder in any audited release
    releases_txt = " / ".join(DS_LABEL[ds] for ds in pw_sets)
    eval_sets = {"PD_train": "field-val", "PD_test": "field-test", "PW_external": "external-test"}
    below = []
    for i in counts.index:
        low = []
        for col, name in eval_sets.items():
            if col == "PW_external" and not pw_sets:
                continue
            if col == "PW_external" and i in no_wild:
                low.append(f"{name}=not in {releases_txt}")
            elif meta[i].get(DS_PD if col.startswith("PD") else DS_PW) and counts.at[i, col] < args.min_count:
                low.append(f"{name}={counts.at[i, col]}")
        below.append("; ".join(low))
    counts[f"below_min_{args.min_count}"] = below
    ordered = ["crop", "tier", "approximate_mapping", "PV_all", "PD_train", "PD_test", *pw_cols,
               *(["PW_external", "PW_external_from"] if pw_sets else []), f"below_min_{args.min_count}"]
    counts = counts[[c for c in ordered if c in counts.columns]]
    counts.index.name = "class_id"
    counts.to_csv(out / "class_counts.csv")

    # ---- 4. hashing
    print("4/6 Computing perceptual hashes (mapped classes only)")
    h = compute_hashes(mapped.path.tolist(), out / "hash_cache.csv", args.workers)
    mapped = mapped.merge(h, on="path", how="left").reset_index(drop=True)

    # quality
    q = []
    for ds, g in mapped.groupby("dataset"):
        ok = g[g.ok == True]  # noqa: E712
        min_side = ok[["width", "height"]].min(axis=1)
        q.append({"dataset": DS_LABEL[ds], "images": len(g), "unreadable": int((g.ok != True).sum()),  # noqa: E712
                  "min_side_lt_100px": int((min_side < 100).sum()),
                  "min_side_lt_224px": int((min_side < 224).sum()),
                  "greyscale_or_palette": int(g["mode"].isin(["L", "LA", "1", "P"]).sum()),
                  "median_width": float(ok.width.median()) if len(ok) else np.nan,
                  "median_height": float(ok.height.median()) if len(ok) else np.nan})
    quality = pd.DataFrame(q)
    quality.to_csv(out / "image_quality.csv", index=False)
    bad = mapped[(mapped.ok != True) | (mapped[["width", "height"]].min(axis=1) < 100)]  # noqa: E712
    bad[["dataset", "split", "folder", "path", "width", "height", "error"]].to_csv(out / "problem_images.csv", index=False)

    # ---- 5. duplicates
    print("5/6 Searching for exact and near-duplicate images")
    good = mapped[mapped.ok == True].reset_index(drop=True)  # noqa: E712
    ph, dh = hex_to_u64(good.phash), hex_to_u64(good.dhash)
    sel = lambda ds, split=None: good.index[(good.dataset == ds) & ((good.split == split) if split else True)]
    comparisons = [
        ("PlantDoc train vs PlantDoc test (split leakage)", sel(DS_PD, "train"), sel(DS_PD, "test"), False),
        *[(f"{DS_LABEL[ds]} vs PlantDoc (external-test independence)", sel(ds), sel(DS_PD), False)
          for ds in pw_sets],
        ("PlantVillage vs PlantDoc", sel(DS_PV), sel(DS_PD), False),
        *[(f"PlantVillage vs {DS_LABEL[ds]}", sel(DS_PV), sel(ds), False) for ds in pw_sets],
        # v2 reuses v1 images, so same-class pairs are expected; label conflicts matter because the
        # external test set mixes v1 healthy folders with v2 disease folders
        ("PlantWild v1 vs PlantWild v2 (release overlap)", sel(DS_PW1), sel(DS_PW2), False),
        ("Within PlantDoc", sel(DS_PD), sel(DS_PD), True),
        *[(f"Within {DS_LABEL[ds]}", sel(ds), sel(ds), True) for ds in pw_sets],
    ]
    if args.pv_internal:
        comparisons.append(("Within PlantVillage", sel(DS_PV), sel(DS_PV), True))

    dup_rows, summary = [], []
    for name, a, b, same in comparisons:
        if len(a) == 0 or len(b) == 0:
            continue
        t0 = time.time()
        prs = near_pairs(a, b, ph, dh, args.phash_threshold, args.dhash_threshold, same=same)
        for i, j, p, d in prs:
            ra, rb = good.loc[i], good.loc[j]
            dup_rows.append({"comparison": name, "kind": "exact" if p == 0 and d == 0 else "near",
                             "phash_dist": p, "dhash_dist": d,
                             "dataset_a": DS_LABEL[ra.dataset], "split_a": ra.split, "class_a": ra.class_id,
                             "dataset_b": DS_LABEL[rb.dataset], "split_b": rb.split, "class_b": rb.class_id,
                             "label_conflict": ra.class_id != rb.class_id,
                             "path_a": ra.path, "path_b": rb.path})
        ia = {i for i, _, _, _ in prs}
        ib = {j for _, j, _, _ in prs}
        if same:
            n_a, n_b, pct_b = len(ia | ib), "", ""
        else:
            n_a, n_b, pct_b = len(ia), len(ib), round(100 * len(ib) / len(b), 2)
        summary.append({"comparison": name, "images_a": len(a), "images_b": len(b) if not same else "",
                        "pairs": len(prs), "exact_pairs": sum(1 for _, _, p, d in prs if p == 0 and d == 0),
                        "label_conflict_pairs": sum(good.at[i, "class_id"] != good.at[j, "class_id"] for i, j, _, _ in prs),
                        "affected_images_a": n_a, "affected_pct_a": round(100 * n_a / len(a), 2),
                        "affected_images_b": n_b, "affected_pct_b": pct_b})
        print(f"  {name}: {len(prs)} pairs ({time.time() - t0:.1f}s)")
    dups = pd.DataFrame(dup_rows, columns=["comparison", "kind", "phash_dist", "dhash_dist", "dataset_a", "split_a",
                                           "class_a", "dataset_b", "split_b", "class_b", "label_conflict",
                                           "path_a", "path_b"])
    dups.sort_values(["comparison", "phash_dist", "dhash_dist"]).to_csv(out / "duplicates.csv", index=False)
    dup_summary = pd.DataFrame(summary)
    dup_summary.to_csv(out / "duplicate_summary.csv", index=False)

    # ---- 6. visuals
    if not args.no_sheets:
        print("6/6 Writing contact sheets and duplicate previews")
        cs_dir, dp_dir = out / "contact_sheets", out / "duplicate_pairs"
        cs_dir.mkdir(exist_ok=True)
        dp_dir.mkdir(exist_ok=True)
        for cid, g in mapped.groupby("class_id"):
            by_ds = {ds: list(zip(gg.path, gg.folder)) for ds, gg in g.groupby("dataset")}
            contact_sheet(cid, by_ds, cs_dir / f"{cid}.jpg")
        for name, g in dups.groupby("comparison"):
            slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:60]
            pair_sheet(g.sort_values(["label_conflict", "phash_dist"], ascending=[False, True]),
                       dp_dir / f"{slug}.jpg")
    else:
        print("6/6 Skipping visuals (--no-sheets)")

    # ---- summary report
    core = counts[counts.tier == "core"]
    ok_core = core[core[f"below_min_{args.min_count}"] == ""]
    lines = [
        "# Pre-proposal data audit", "",
        f"Mapping file: `{args.mapping}` · tiers audited: {', '.join(sorted(tiers))} · "
        f"min count per evaluation set: {args.min_count} · near-duplicate rule: pHash ≤ {args.phash_threshold} "
        f"and dHash ≤ {args.dhash_threshold} (64-bit hashes)", "",
        "## Datasets found", "",
        *[f"- **{DS_LABEL[ds]}**: {(df.dataset == ds).sum():,} images in total; {txt}" for ds, txt in info.items()],
        "",
        "## Headline numbers", "",
        f"- Core classes: {len(core)}; core classes meeting the minimum count in every evaluation set: {len(ok_core)}",
    ]
    if "PD_test" in core and len(core):
        lines.append(f"- PlantDoc test images per core class: median {core.PD_test.median():.0f}, "
                     f"min {core.PD_test.min()} ({core.PD_test.idxmin()})")
    if len(core) and core.PW_external.sum() > 0:
        lines.append(f"- PlantWild external-test images per core class (v2, or v1 where v2 has no folder): "
                     f"median {core.PW_external.median():.0f}, "
                     f"min {core.PW_external.min()} ({core.PW_external.idxmin()})")
    for ds in pw_sets:
        if len(core) and core[f"{COL_PREFIX[ds]}_total"].sum() > 0:
            col = core[f"{COL_PREFIX[ds]}_total"]
            lines.append(f"- {DS_LABEL[ds]} images per core class: median {col.median():.0f}, "
                         f"min {col.min()} ({col.idxmin()})")
    for r in dup_summary.itertuples():
        lines.append(f"- {r.comparison}: {r.pairs} duplicate pairs ({r.exact_pairs} exact, "
                     f"{r.label_conflict_pairs} with conflicting labels); images affected: "
                     + (f"{r.affected_images_a} ({r.affected_pct_a}%)" if r.images_b == "" else
                        f"{r.affected_images_a} of the first set ({r.affected_pct_a}%), "
                        f"{r.affected_images_b} of the second ({r.affected_pct_b}%)"))
    if DS_PW1 in pw_sets and DS_PW2 in pw_sets and not_in_release[DS_PW2]:
        from_v1 = sorted(i for i in not_in_release[DS_PW2] if counts.at[i, "PW1_total"] > 0)
        if from_v1:
            lines.append(f"- PlantWild v2 has no folder for {len(from_v1)} mapped classes "
                         f"({', '.join(from_v1)}); their external-test images come from PlantWild v1.")
    if no_wild:
        lines.append(f"- {len(no_wild)} mapped classes are not in {releases_txt} "
                     f"({', '.join(sorted(no_wild))}); these can only be evaluated on PlantDoc.")
    if missing:
        lines.append(f"- **{len(missing)} mapped folders were not found on disk**; see `missing_folders.csv`.")
    lines += ["", "## Images per class", "",
              "PD_train is the in-the-wild validation set and PD_test the same-source test set in the proposed design. "
              "PW1_* and PW2_* are the two PlantWild releases (PW1 splits from the official v1 trainval.txt); "
              "PW_external is the external test set (v2, or v1 for classes v2 lacks) and PW_external_from its source.",
              "", md_table(counts.reset_index()), "",
              "## Duplicate search", "",
              "Near-duplicates are candidates: open `duplicate_pairs/*.jpg` and confirm them by eye before removing. "
              "In PlantWild v1 vs v2, same-class pairs are expected (v2 reuses v1 images); pairs with conflicting "
              "labels are the ones to check.",
              "", md_table(dup_summary) if len(dup_summary) else "No comparisons run.", "",
              "## Image quality (mapped classes)", "", md_table(quality), "",
              "## Folders not in the mapping", "",
              f"{len(um)} folders ({int(um.n_images.sum()) if len(um) else 0} images) are not in the audited tiers; "
              "see `unmapped_folders.csv`. PlantWild classes outside the label space are useful later as "
              "out-of-distribution test inputs for the web application.", ""]
    (out / "audit_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone. Open {out / 'audit_summary.md'} first.")


if __name__ == "__main__":
    main()
