#!/usr/bin/env python3
"""
Download the three datasets used in the plant-disease coursework.

  PlantVillage  huggingface.co/datasets/mohanty/PlantVillage (data.zip -> raw/color, leaf-map.json)
  PlantDoc      github.com/pratikkayal/PlantDoc-Dataset     (GitHub zip, train/ + test/)
  PlantWild     huggingface.co/datasets/uqtwei2/PlantWild   (plantwild_v2.zip + plantwild.zip for v1)

Result:
  data/plantvillage/raw/color/<class>/...   data/plantvillage/leaf-map.json
  data/plantdoc/train/<class>/...           data/plantdoc/test/<class>/...
  data/plantwild/v2/<class>/...             data/plantwild/v1/images/<class>/...  (skip with --skip-plantwild-v1)

Usage
  python download_data.py                         # all three into ./data
  python download_data.py --dest data --only plantdoc
  python download_data.py --skip-plantwild-v1     # skip PlantWild v1 (2.7 GB, has the healthy classes)

Needs huggingface_hub (for PlantVillage and PlantWild). Datasets already on disk are skipped.
"""
import argparse
import os
import shutil
import stat
import sys
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

PV_REPO = "mohanty/PlantVillage"
PD_ZIP = "https://codeload.github.com/pratikkayal/PlantDoc-Dataset/zip/refs/heads/master"
PW_REPO = "uqtwei2/PlantWild"
WIN_BAD_CHARS = str.maketrans({c: "_" for c in '<>:"|?*\\'})


def rmtree(path: Path):
    """shutil.rmtree that also removes read-only files (on Windows)."""
    def retry(func, p, _):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    if path.exists():
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=retry)
        else:
            shutil.rmtree(path, onerror=retry)


def download_file(url: str, out: Path):
    print(f"  downloading {url}")
    tmp = out.with_suffix(out.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0) or None  # GitHub zips send no size
        with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc="  download") as bar:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(out)


def extract(archive: Path, prefix: str, target: Path, keep=None):
    """Extract members under `prefix` into `target`, dropping the prefix.
    `keep` optionally limits extraction to these top-level folders (after the prefix)."""
    print(f"  extracting {archive.name} -> {target}")
    target = target.resolve()
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.infolist() if m.filename.startswith(prefix) and not m.is_dir()]
        if keep:
            members = [m for m in members if m.filename[len(prefix):].split("/")[0] in keep]
        for m in tqdm(members, unit="file", desc="  extract"):
            # Some PlantDoc names contain '?' (e.g. IMG_1629.JPG?1507122477.jpg), which Windows rejects.
            rel = m.filename[len(prefix):].translate(WIN_BAD_CHARS)
            out = (target / rel).resolve()
            if target not in out.parents:  # zip-slip guard
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)


def hf_download(repo: str, filename: str, local_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub is not installed: pip install huggingface_hub")
    return Path(hf_hub_download(repo, filename, repo_type="dataset", local_dir=str(local_dir)))


def download_plantvillage(dest: Path):
    target = dest / "plantvillage"
    if (target / "raw" / "color").is_dir() and (target / "leaf-map.json").is_file():
        print(f"[PlantVillage] already present at {target}, skipping.")
        return
    tmp = dest / "_hf_plantvillage"
    target.mkdir(parents=True, exist_ok=True)
    print("[PlantVillage] downloading leaf-map.json from Hugging Face...")
    shutil.copyfile(hf_download(PV_REPO, "leaf_grouping/leaf-map.json", tmp), target / "leaf-map.json")
    if not (target / "raw" / "color").is_dir():
        # data.zip holds raw/{color,grayscale,segmented}; only raw/color is extracted.
        print("[PlantVillage] downloading data.zip from Hugging Face (about 2.2 GB)...")
        archive = hf_download(PV_REPO, "data.zip", tmp)
        extract(archive, "raw/color/", target / "raw" / "color")
    rmtree(tmp)
    print(f"[PlantVillage] done -> {target}")


def download_plantdoc(dest: Path):
    target = dest / "plantdoc"
    if (target / "train").is_dir() and (target / "test").is_dir():
        print(f"[PlantDoc] already present at {target}, skipping.")
        return
    print("[PlantDoc] downloading GitHub archive (about 1 GB)...")
    archive = dest / "plantdoc.zip"
    if not archive.exists():
        download_file(PD_ZIP, archive)
    if not zipfile.is_zipfile(archive):  # GitHub sends no size, so a dropped connection shows up here
        archive.unlink()
        sys.exit("[PlantDoc] download was incomplete. Run the script again.")
    extract(archive, "PlantDoc-Dataset-master/", target, keep={"train", "test"})
    archive.unlink()
    print(f"[PlantDoc] done -> {target}")


def download_plantwild(dest: Path, with_v1: bool):
    target = dest / "plantwild"
    releases = [("v2", "plantwild_v2.zip", "plantwild_v2/")]
    if with_v1:
        releases.append(("v1", "plantwild.zip", "plantwild/"))
    for name, filename, prefix in releases:
        out = target / name
        if out.is_dir() and any(out.iterdir()):
            print(f"[PlantWild {name}] already present at {out}, skipping.")
            continue
        print(f"[PlantWild {name}] downloading {filename} from Hugging Face...")
        archive = hf_download(PW_REPO, filename, dest / "_hf")
        extract(archive, prefix, out)
        print(f"[PlantWild {name}] done -> {out}")
    rmtree(dest / "_hf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default="data", help="download folder (default: data)")
    ap.add_argument("--only", choices=["plantvillage", "plantdoc", "plantwild"], action="append",
                    help="download only this dataset (can be repeated)")
    ap.add_argument("--skip-plantwild-v1", action="store_true",
                    help="do not download PlantWild v1 (2.7 GB). make_splits.py needs it for its default "
                         "--healthy-test plantwild_v1; use --healthy-test plantdoc there instead")
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    wanted = args.only or ["plantvillage", "plantdoc", "plantwild"]
    if "plantvillage" in wanted:
        download_plantvillage(dest)
    if "plantdoc" in wanted:
        download_plantdoc(dest)
    if "plantwild" in wanted:
        download_plantwild(dest, not args.skip_plantwild_v1)

    print("\nNext step:")
    print(f"  python audit_datasets.py --plantvillage {dest / 'plantvillage'} "
          f"--plantdoc {dest / 'plantdoc'} --plantwild {dest / 'plantwild'}")


if __name__ == "__main__":
    main()
