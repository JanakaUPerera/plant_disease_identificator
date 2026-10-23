#!/usr/bin/env python3
"""
Build the fixed train / validation / test manifests for the plant-disease coursework.

Design (decided before any modelling):
  train, val_lab, test_lab   PlantVillage, core classes, group-aware split (default 70/15/15)
  val_wild                   all of PlantDoc (official train + test merged): every model-selection,
                             threshold and prompt decision is made here
  test_wild                  PlantWild v2 disease classes + PlantWild v1 healthy folders for the
                             classes v2 lacks; evaluated once, at the end

Cleaning rules, applied in this order and logged image by image in removed_images.csv:
  R1  unreadable images, or shortest side below --min-side pixels
  R5  images rejected in the manual label review (--exclude), together with every copy of them
      in any dataset, so a copy of a rejected image is never kept. Applied after R1 and before
      R2; numbered R5 because it was added after R1-R4 were fixed
  R2  duplicate groups whose copies carry different labels: every copy is removed
      (the true label is unknown)
  R3  copies of the same image in a later set are removed (train < val_wild < test_wild),
      so no image appears in two sets
  R4  repeated copies inside val_wild or test_wild: one copy is kept
  PlantVillage near-identical copies of the same class are not removed; they are kept in the
  same split (group-aware split).

PlantVillage leaf grouping: PlantVillage has several photos of the same physical leaf. Its
repository ships leaf-map.json, which assigns images to leaf IDs. All photos of one leaf (plus
any near-identical copies) are kept in the same split, so a leaf seen in training never appears
in val_lab or test_lab. The file is found automatically in the PlantVillage folder (root or
leaf_grouping/), or given with --leaf-map. Images with no leaf ID form a group of their own.

Duplicates use the same perceptual-hash rule as audit_datasets.py. Pairs you have checked by eye
and judged to be different images can be listed in a CSV (columns path_a,path_b) and passed with
--not-duplicates; they are then ignored.

Manual label review (rule R5): a CSV passed with --exclude, columns dataset, source_split, rel_path
(exactly as in all_splits.csv) and reason, for example: PlantWild, v1, tomato leaf/abc.jpg, symptoms.
The reason must be a code from label_review_rules.md, and the script enforces where each code may be
used: symptoms (healthy classes only), no_symptoms (disease classes only), not_leaf and wrong_crop
(any class). Images are matched by these columns, not by full path, so the same list works on every
computer. Listed images that are not found are reported, never silently ignored.

Usage
  python make_splits.py --plantvillage data/plantvillage --plantdoc data/plantdoc \
                        --plantwild data/plantwild --out splits

Healthy classes for the wild test set:
  --healthy-test plantwild_v1   (default) PlantWild v1 healthy folders
  --healthy-test plantdoc       fallback: official PlantDoc test images of the healthy classes
                                move from val_wild to test_wild
  --healthy-test none           healthy classes are not tested in the wild
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from audit_datasets import (DS_LABEL, DS_PD, DS_PV, DS_PW, compute_hashes, folder_names, hex_to_u64,
                            index_plantdoc, index_plantvillage, index_plantwild, near_pairs, norm)

STAGE = {"train_pool": 0, "val_wild": 1, "test_wild": 2}
SPLIT_ORDER = ["train", "val_lab", "test_lab", "val_wild", "test_wild"]


class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)

    def find(self, i):
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:  # path compression
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def index_all(args, classes):
    """Index the three datasets and attach the canonical class id. Returns (DataFrame, notes)."""
    lookup = {ds: {norm(n): c["id"] for c in classes for n in folder_names(c, ds)} for ds in (DS_PV, DS_PD, DS_PW)}
    notes, frames = [], []

    rows, root = index_plantvillage(args.plantvillage, set(lookup[DS_PV]))
    pv = pd.DataFrame(rows, columns=["dataset", "source_split", "folder", "path"])
    pv["dataset_root"] = str(root)
    frames.append(pv)

    rows, root = index_plantdoc(args.plantdoc, set(lookup[DS_PD]))
    pdoc = pd.DataFrame(rows, columns=["dataset", "source_split", "folder", "path"])
    pdoc["dataset_root"] = str(root)
    frames.append(pdoc)

    rows, root, vnote, _, version = index_plantwild(args.plantwild, set(lookup[DS_PW]), "v2")
    pw = pd.DataFrame(rows, columns=["dataset", "source_split", "folder", "path"])
    pw["dataset_root"] = str(root)
    pw["source_split"] = "v2"
    frames.append(pw)
    notes.append(f"PlantWild disease classes from v2: `{root}`")

    df = pd.concat(frames, ignore_index=True)
    df["class_id"] = [lookup[d].get(norm(f)) for d, f in zip(df.dataset, df.folder)]
    df = df[df.class_id.notna()].copy()

    # classes with no PlantWild v2 folder (the healthy classes) -> PlantWild v1 or fallback
    in_v2 = set(df[df.dataset == DS_PW].class_id)
    lacking = [c["id"] for c in classes if c["id"] not in in_v2]
    if lacking and args.healthy_test == "plantwild_v1":
        try:
            rows, root1, _, _, _ = index_plantwild(args.plantwild_v1 or args.plantwild, set(lookup[DS_PW]), "v1")
        except SystemExit as exc:
            sys.exit(f"{exc}\nPlantWild v1 is needed for the healthy test classes. Point --plantwild-v1 at the "
                     f"v1 download, or use --healthy-test plantdoc / none.")
        v1 = pd.DataFrame(rows, columns=["dataset", "source_split", "folder", "path"])
        v1["dataset_root"] = str(root1)
        v1["source_split"] = "v1"
        v1["class_id"] = [lookup[DS_PW].get(norm(f)) for f in v1.folder]
        v1 = v1[v1.class_id.isin(lacking)]
        df = pd.concat([df, v1], ignore_index=True)
        notes.append(f"PlantWild v1 used only for: {', '.join(sorted(v1.class_id.unique()))} (`{root1}`)")
    elif lacking:
        notes.append(f"Classes with no PlantWild v2 folder: {', '.join(lacking)} "
                     f"(--healthy-test {args.healthy_test})")

    # which set each image belongs to before cleaning
    df["pool"] = df.dataset.map({DS_PV: "train_pool", DS_PD: "val_wild", DS_PW: "test_wild"})
    if args.healthy_test == "plantdoc":
        move = (df.dataset == DS_PD) & (df.source_split == "test") & df.class_id.isin(lacking)
        df.loc[move, "pool"] = "test_wild"
        notes.append(f"Fallback: {int(move.sum())} official PlantDoc test images of "
                     f"{', '.join(lacking)} moved to test_wild")
    # a fixed, OS-independent order, so the same seed gives the same split on Windows, macOS and Linux
    df["rel_path"] = [Path(os.path.relpath(p, r)).as_posix() for p, r in zip(df.path, df.dataset_root)]
    df = df.sort_values(["dataset", "source_split", "rel_path"])
    return df.reset_index(drop=True), notes


def find_leaf_map(args):
    """Locate PlantVillage's leaf-map.json: --leaf-map, else inside the PlantVillage folder."""
    if args.leaf_map:
        return args.leaf_map if args.leaf_map.is_file() else None
    root = args.plantvillage
    for cand in (root / "leaf-map.json", root / "leaf_grouping" / "leaf-map.json"):
        if cand.is_file():
            return cand
    found = sorted(root.rglob("leaf-map.json"))
    return found[0] if found else None


def pv_leaf_id(file_name, folder, leaf_map):
    """Leaf ID of a PlantVillage image, using the same matching rule as the dataset's own loader
    (plant_village.py): the part after '___', without 'copy' suffixes or the extension, lower-cased.
    Returns None when the image has no (unambiguous) leaf ID."""
    ident = file_name.replace("_final_masked", "")
    if "___" in ident:
        ident = ident.split("___")[-1]
    ident = ident.split("copy")[0]
    for ext in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
        ident = ident.replace(ext, "")
    suggestions = leaf_map.get(ident.strip().lower())
    if not suggestions:
        return None
    if len(suggestions) == 1:
        return suggestions[0]
    same_class = [s for s in suggestions if s.split(":::")[0] == folder]
    return same_class[0] if same_class else None


class KeyUnionFind:
    """Union-find over hashable keys (used to merge duplicate groups with leaf IDs)."""
    def __init__(self):
        self.parent = {}

    def find(self, k):
        self.parent.setdefault(k, k)
        root = k
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[k] != root:
            self.parent[k], k = root, self.parent[k]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb, key=str)] = min(ra, rb, key=str)


def find_pairs(df, args):
    """All duplicate pairs as (i, j, phash_dist, dhash_dist, kind)."""
    ph, dh = hex_to_u64(df.phash), hex_to_u64(df.dhash)
    wild = df.index[df.pool != "train_pool"]
    pv = df.index[df.pool == "train_pool"]
    pairs = []
    t0 = time.time()
    for i, j, p, d in near_pairs(wild, wild, ph, dh, args.phash_threshold, args.dhash_threshold, same=True):
        pairs.append((i, j, p, d, "wild-wild"))
    for i, j, p, d in near_pairs(pv, wild, ph, dh, args.phash_threshold, args.dhash_threshold):
        pairs.append((i, j, p, d, "lab-wild"))
    # inside PlantVillage only near-identical copies are grouped (plain backgrounds make looser
    # thresholds match different leaves); compared within each class
    for _, g in df.loc[pv].groupby("class_id"):
        for i, j, p, d in near_pairs(g.index, g.index, ph, dh, args.pv_phash_threshold,
                                     args.pv_dhash_threshold, same=True):
            pairs.append((i, j, p, d, "lab-lab"))
    print(f"  {len(pairs)} duplicate pairs found ({time.time() - t0:.0f}s)")
    return pairs


def load_not_duplicates(path):
    if not path:
        return set()
    nd = pd.read_csv(path)
    return {frozenset((a, b)) for a, b in zip(nd.path_a, nd.path_b)}


def load_exclusions(path):
    """Manual label-review list (rule R5) as {(dataset, source_split, rel_path): reason}.
    `dataset` may be written as in all_splits.csv (PlantDoc, PlantWild, PlantVillage) or as the
    internal name (plantdoc, ...); rel_path may use / or \\."""
    if not path:
        return {}
    ex = pd.read_csv(path, dtype=str, keep_default_na=False)
    need = ["dataset", "source_split", "rel_path", "reason"]
    if missing := [c for c in need if c not in ex.columns]:
        sys.exit(f"{path} is missing the column(s) {missing}; it needs {need}")
    to_ds = {}
    for internal in (DS_PV, DS_PD, DS_PW):
        to_ds[internal] = internal
        to_ds[DS_LABEL[internal].lower()] = internal
    out = {}
    for r in ex.itertuples(index=False):
        ds = to_ds.get(r.dataset.strip().lower())
        if ds is None:
            sys.exit(f"{path}: unknown dataset '{r.dataset}' (use PlantVillage, PlantDoc or PlantWild)")
        code = r.reason.strip().lower()
        if code not in REVIEW_CODES:
            sys.exit(f"{path}: reason '{r.reason}' for {r.rel_path} is not one of {sorted(REVIEW_CODES)} "
                     f"(see label_review_rules.md)")
        out[(ds, r.source_split.strip(), r.rel_path.strip().replace("\\", "/"))] = code
    return out


# reason codes of label_review_rules.md, and which classes each one may be used for
REVIEW_CODES = {"symptoms": "healthy", "no_symptoms": "disease", "not_leaf": "any", "wrong_crop": "any"}


def check_review_codes(excl_rows):
    """Enforce label_review_rules.md: 'symptoms' only on healthy classes, 'no_symptoms' only on
    disease classes. excl_rows: [(rel_path, class_id, code)]. Stops with a list of every violation."""
    bad = []
    for rel, cid, code in excl_rows:
        healthy = cid.endswith("_healthy")
        if (REVIEW_CODES[code] == "healthy" and not healthy) or (REVIEW_CODES[code] == "disease" and healthy):
            bad.append(f"{rel} ({cid}): '{code}' is only for {REVIEW_CODES[code]} classes")
    if bad:
        sys.exit("Exclusion list breaks label_review_rules.md:\n  " + "\n  ".join(bad))


def group_split(g, fractions, rng):
    """Assign whole groups of one class to train/val_lab/test_lab, closest to the target sizes."""
    groups = list(g.groupby("split_group").size().items())
    rng.shuffle(groups)
    groups.sort(key=lambda t: -t[1])  # big groups first, so they do not overshoot at the end
    names = ["train", "val_lab", "test_lab"]
    target = np.array(fractions) * len(g)
    filled = np.zeros(3)
    assign = {}
    for gid, size in groups:
        k = int(np.argmax(target - filled))  # the split furthest below its target
        assign[gid] = names[k]
        filled[k] += size
    return g.split_group.map(assign)


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plantvillage", type=Path, required=True)
    ap.add_argument("--plantdoc", type=Path, required=True)
    ap.add_argument("--plantwild", type=Path, required=True, help="folder containing PlantWild v2 (and v1)")
    ap.add_argument("--plantwild-v1", type=Path, help="PlantWild v1 folder, if it is not inside --plantwild")
    ap.add_argument("--healthy-test", choices=["plantwild_v1", "plantdoc", "none"], default="plantwild_v1")
    ap.add_argument("--mapping", type=Path, default=Path(__file__).with_name("class_mapping.json"))
    ap.add_argument("--tiers", default="core", help="mapping tiers in the label space (default: core)")
    ap.add_argument("--out", type=Path, default=Path("splits"))
    ap.add_argument("--audit-cache", type=Path, default=Path("audit_outputs/hash_cache.csv"),
                    help="hash cache written by audit_datasets.py, reused to save time")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lab-fractions", default="0.70,0.15,0.15", help="PlantVillage train,val_lab,test_lab")
    ap.add_argument("--min-side", type=int, default=100, help="R1: drop images whose shortest side is smaller")
    ap.add_argument("--phash-threshold", type=int, default=6)
    ap.add_argument("--dhash-threshold", type=int, default=8)
    ap.add_argument("--pv-phash-threshold", type=int, default=2,
                    help="stricter rule for grouping PlantVillage copies (default 2)")
    ap.add_argument("--pv-dhash-threshold", type=int, default=2)
    ap.add_argument("--leaf-map", type=Path,
                    help="PlantVillage leaf-map.json (default: found inside the --plantvillage folder)")
    ap.add_argument("--no-leaf-grouping", action="store_true",
                    help="do not group PlantVillage photos of the same leaf (not recommended)")
    ap.add_argument("--not-duplicates", type=Path,
                    help="CSV with path_a,path_b pairs you checked by eye and judged to be different images")
    ap.add_argument("--exclude", type=Path,
                    help="R5: CSV of images rejected in the manual label review "
                         "(columns dataset, source_split, rel_path, reason code)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    fractions = [float(x) for x in args.lab_fractions.split(",")]
    if len(fractions) != 3 or abs(sum(fractions) - 1) > 1e-6:
        ap.error("--lab-fractions needs three numbers that sum to 1")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    tiers = {t.strip() for t in args.tiers.split(",")}
    classes = [c for c in mapping["classes"] if c["tier"] in tiers]
    class_index = {c["id"]: k for k, c in enumerate(classes)}

    print("1/5 Indexing")
    df, notes = index_all(args, classes)
    for n in notes:
        print("  " + n)

    print("2/5 Hashing (reusing the audit cache)")
    cache = out / "hash_cache.csv"
    if not cache.exists() and args.audit_cache.exists():
        shutil.copy(args.audit_cache, cache)
    h = compute_hashes(df.path.tolist(), cache, args.workers)
    df = df.merge(h[["path", "ok", "width", "height", "phash", "dhash", "error"]], on="path", how="left")

    removed = []  # (path, dataset, source_split, rel_path, class_id, pool, rule, detail)

    def drop(idx, rule, detail):
        for i in idx:
            r = df.loc[i]
            text = detail(i) if callable(detail) else detail
            removed.append((r.path, DS_LABEL[r.dataset], r.source_split, r.rel_path, r.class_id, r.pool, rule, text))

    print("3/5 Applying cleaning rules")
    min_side = df[["width", "height"]].min(axis=1)
    r1 = df.index[(df.ok != True) | (min_side < args.min_side)]  # noqa: E712
    drop(r1, "R1", f"unreadable or shortest side < {args.min_side}px")
    df = df.drop(index=r1).reset_index(drop=True)

    pairs = find_pairs(df, args)
    skip = load_not_duplicates(args.not_duplicates)
    pairs = [p for p in pairs if frozenset((df.at[p[0], "path"], df.at[p[1], "path"])) not in skip]
    pd.DataFrame([(df.at[i, "path"], df.at[j, "path"], df.at[i, "class_id"], df.at[j, "class_id"], p, d, k)
                  for i, j, p, d, k in pairs],
                 columns=["path_a", "path_b", "class_a", "class_b", "phash_dist", "dhash_dist", "kind"]
                 ).to_csv(out / "duplicate_pairs_used.csv", index=False)

    uf = UnionFind(len(df))
    for i, j, *_ in pairs:
        uf.union(i, j)
    df["group_id"] = [uf.find(i) for i in range(len(df))]
    keep = np.ones(len(df), dtype=bool)

    # R5: manual label review. The rejected image and every copy of it (same duplicate group) go.
    excl = load_exclusions(args.exclude)
    ex_groups = set()
    if excl:
        reason = {i: excl[k] for i, k in enumerate(zip(df.dataset, df.source_split, df.rel_path)) if k in excl}
        check_review_codes([(df.at[i, "rel_path"], df.at[i, "class_id"], code) for i, code in reason.items()])
        ex_groups = set(df.group_id.iloc[list(reason)])
        r5 = df.index[df.group_id.isin(ex_groups)]
        drop(r5, "R5", lambda i: reason.get(i, "copy of an image rejected in the label review"))
        keep[r5] = False
        not_found = len(excl) - len(reason)
        notes.append(f"Manual label review (--exclude `{args.exclude}`): {len(excl)} images listed, "
                     f"{len(reason)} found and removed, plus {len(r5) - len(reason)} copies of them; {not_found} not found")
        print("  " + notes[-1])
        if not_found:
            found = {k for k in zip(df.dataset, df.source_split, df.rel_path)}
            lost = [k for k in excl if k not in found]
            print("  WARNING: listed but not found (check spelling, or already removed by R1):")
            for ds, src, rel in lost[:10]:
                print(f"     {DS_LABEL[ds]} | {src} | {rel}")

    group_sizes = df.group_id.value_counts()
    for gid in group_sizes[group_sizes > 1].index:
        if gid in ex_groups:
            continue  # already removed by R5
        g = df[df.group_id == gid]
        members = ", ".join(sorted(g.class_id.unique()))
        if g.class_id.nunique() > 1:  # R2
            drop(g.index, "R2", f"copies labelled differently: {members}")
            keep[g.index] = False
            continue
        first = min(STAGE[p] for p in g.pool)
        later = g.index[g.pool.map(STAGE) > first]  # R3
        drop(later, "R3", f"copy of an image already in {[k for k, v in STAGE.items() if v == first][0]}")
        keep[later] = False
        same = g[g.pool.map(STAGE) == first]
        if first > 0 and len(same) > 1:  # R4
            # keep the copy without a "Copy" style name, then the shortest path
            order = sorted(same.index, key=lambda i: ("copy" in Path(df.at[i, "rel_path"]).name.lower(),
                                                      len(df.at[i, "rel_path"]), df.at[i, "rel_path"]))
            extra = order[1:]
            drop(extra, "R4", "repeated copy inside the same set; one copy kept")
            keep[extra] = False
    df = df[keep].copy()

    print("4/5 Splitting PlantVillage by groups (photos of the same leaf stay together)")
    df["leaf_id"] = ""
    pv_rows = df.index[df.pool == "train_pool"]
    leaf_map_path = find_leaf_map(args)  # always looked up, so the same-leaf check can run
    if leaf_map_path:
        leaf_map = json.loads(Path(leaf_map_path).read_text(encoding="utf-8"))
        for i in pv_rows:
            lid = pv_leaf_id(Path(df.at[i, "path"]).name, df.at[i, "folder"], leaf_map)
            df.at[i, "leaf_id"] = lid or ""
        matched = int((df.loc[pv_rows, "leaf_id"] != "").sum())
        leaf_sizes = df.loc[pv_rows][df.loc[pv_rows, "leaf_id"] != ""].leaf_id.value_counts()
        notes.append(f"PlantVillage leaf grouping from `{leaf_map_path}`: {matched:,} of {len(pv_rows):,} "
                     f"images matched to {len(leaf_sizes):,} leaves; {int((leaf_sizes > 1).sum()):,} leaves have "
                     f"more than one photo (up to {int(leaf_sizes.max()) if len(leaf_sizes) else 0}); "
                     f"unmatched images form groups of their own")
        print("  " + notes[-1])
        if args.no_leaf_grouping:
            notes.append("PlantVillage leaf grouping switched OFF (--no-leaf-grouping): see the same-leaf check")
            print("  WARNING: leaf grouping switched off; leaf IDs are used only for the check")
    else:
        msg = "WARNING: leaf-map.json not found, so photos of the same leaf may cross splits; pass --leaf-map"
        notes.append(f"PlantVillage {msg}")
        print("  " + msg)
    # merge duplicate groups and leaf IDs: one split group = everything connected by either
    kuf = KeyUnionFind()
    for i in df.index:
        kuf.find(f"g{df.at[i, 'group_id']}")
        if df.at[i, "leaf_id"] and not args.no_leaf_grouping:
            kuf.union(f"g{df.at[i, 'group_id']}", f"l{df.at[i, 'leaf_id']}")
    df["split_group"] = [kuf.find(f"g{g}") for g in df.group_id]

    rng = random.Random(args.seed)
    df["split"] = df.pool
    for cid in [c["id"] for c in classes]:
        m = (df.pool == "train_pool") & (df.class_id == cid)
        if m.any():
            df.loc[m, "split"] = group_split(df[m], fractions, rng).values
    df["class_index"] = df.class_id.map(class_index)
    df["crop"] = df.class_id.map({c["id"]: c["crop"] for c in classes})
    df["dataset"] = df.dataset.map(DS_LABEL)
    largest_pv_group = int(df[df.dataset == "PlantVillage"].split_group.value_counts().max()) if len(df) else 0

    print("5/5 Verifying that no duplicate crosses two splits and writing manifests")
    df = df.sort_values(["split", "class_index", "path"]).reset_index(drop=True)
    ph, dh = hex_to_u64(df.phash), hex_to_u64(df.dhash)
    checks = []
    for a in range(len(SPLIT_ORDER)):
        for b in range(a + 1, len(SPLIT_ORDER)):
            ia, ib = df.index[df.split == SPLIT_ORDER[a]], df.index[df.split == SPLIT_ORDER[b]]
            both_lab = SPLIT_ORDER[a] in ("train", "val_lab", "test_lab") and SPLIT_ORDER[b] in ("val_lab", "test_lab")
            pt, dt = ((args.pv_phash_threshold, args.pv_dhash_threshold) if both_lab
                      else (args.phash_threshold, args.dhash_threshold))
            n = len(near_pairs(ia, ib, ph, dh, pt, dt))
            checks.append({"split_a": SPLIT_ORDER[a], "split_b": SPLIT_ORDER[b], "duplicate_pairs": n})
    checks = pd.DataFrame(checks)
    # same-leaf check: how many PlantVillage leaves have photos in more than one split
    leaf_rows = df[(df.dataset == "PlantVillage") & (df.leaf_id != "")]
    leaves_crossing = int((leaf_rows.groupby("leaf_id").split.nunique() > 1).sum()) if len(leaf_rows) else 0

    cols = ["split", "class_index", "class_id", "crop", "dataset", "source_split", "folder", "rel_path", "path",
            "group_id", "leaf_id", "width", "height", "phash", "dhash"]
    df[cols].to_csv(out / "all_splits.csv", index=False)
    digests = {}
    for s in SPLIT_ORDER:
        f = out / f"{s}.csv"
        part = df[df.split == s]
        part[cols].to_csv(f, index=False)
        # fingerprint of the content (dataset, relative path, label): identical on every OS
        key = "\n".join(sorted(f"{d}|{r}|{c}" for d, r, c in zip(part.dataset, part.rel_path, part.class_id)))
        digests[s] = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    (out / "classes.json").write_text(json.dumps(
        [{"index": class_index[c["id"]], "id": c["id"], "crop": c["crop"]} for c in classes], indent=2),
        encoding="utf-8")
    rem = pd.DataFrame(removed, columns=["path", "dataset", "source_split", "rel_path", "class_id", "pool",
                                         "rule", "detail"])
    rem.to_csv(out / "removed_images.csv", index=False)

    # ---- summary
    counts = (df.pivot_table(index="class_id", columns="split", values="path", aggfunc="count", fill_value=0)
              .reindex(index=[c["id"] for c in classes], columns=SPLIT_ORDER, fill_value=0))
    counts.loc["TOTAL"] = counts.sum()
    rule_counts = (rem.groupby(["rule", "dataset"]).size().unstack(fill_value=0) if len(rem)
                   else pd.DataFrame())
    small = [f"{cid} ({s}={counts.at[cid, s]})" for cid in counts.index[:-1]
             for s in ("val_wild", "test_wild") if counts.at[cid, s] < 20]
    lines = [
        "# Fixed data splits", "",
        f"Seed {args.seed} · PlantVillage fractions {args.lab_fractions} · label space: {len(classes)} classes "
        f"(tiers: {', '.join(sorted(tiers))}) · healthy wild test: {args.healthy_test}", "",
        *[f"- {n}" for n in notes],
        f"- Duplicate rule: pHash ≤ {args.phash_threshold} and dHash ≤ {args.dhash_threshold}; PlantVillage "
        f"copies grouped at pHash ≤ {args.pv_phash_threshold} and dHash ≤ {args.pv_dhash_threshold}; "
        f"largest PlantVillage group (same leaf or copies): {largest_pv_group} images",
        f"- Pairs excluded after visual check (--not-duplicates): {len(skip)}", "",
        "## Images per class and split", "", md_table(counts.reset_index()), "",
        "## Images removed by each cleaning rule", "",
        md_table(rule_counts.reset_index()) if len(rule_counts) else "None.", "",
        "R1 unreadable or too small · R5 rejected in the manual label review, with all its copies "
        "(applied after R1, before R2) · R2 copies with conflicting labels (all copies removed) · "
        "R3 copy of an image already in an earlier set · R4 repeated copy inside one set.", "",
        "## Leakage check: duplicate pairs between final splits (should all be 0)", "", md_table(checks), "",
        f"**Same-leaf check:** PlantVillage leaves with photos in more than one split: {leaves_crossing} "
        f"(should be 0).", "",
        "## Split fingerprints (SHA-256 of dataset|relative path|label, first 16 hex digits; same on every OS)", "",
        *[f"- `{s}.csv`: {d}" for s, d in digests.items()], "",
    ]
    if small:
        lines += ["## Warnings", "", f"Classes with fewer than 20 images in a wild set: {', '.join(small)}", ""]
    (out / "split_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(counts.to_string())
    print(f"\nRemoved: {len(rem)} images; leakage pairs between splits: {int(checks.duplicate_pairs.sum())}; "
          f"leaves crossing splits: {leaves_crossing}")
    print(f"Done. Open {out / 'split_summary.md'}")


if __name__ == "__main__":
    main()