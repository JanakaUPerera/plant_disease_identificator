#!/usr/bin/env python3
"""
Manual label review for the in-the-wild sets (rule R5 in make_splits.py).

The rules are in label_review_rules.md. This tool only shows the images and records your decisions;
every decision is yours.

Step A  make    numbered review sheets, 40 images each, one class at a time
  python review_labels.py make --splits splits --out review

  Writes review/sheets/S001.jpg, S002.jpg, ... (healthy classes first), review/sheet_index.csv
  (which image is number k on which sheet) and review/decisions.txt with one line per sheet.

Step B  you     open each sheet and edit its line in review/decisions.txt
  S001: none                          nothing to exclude on this sheet
  S002: 7 symptoms                    image 7 shows clear symptoms
  S003: 4 18 symptoms, 23 not_leaf    several numbers can share one code
  S004:                               empty = not reviewed yet

Step C  build   check the decisions and write label_exclusions.csv for make_splits.py --exclude
  python review_labels.py build --out review

  Stops on any mistake (unknown sheet, number not on the sheet, unknown code, a code used on the
  wrong kind of class). Reports how many sheets are still unreviewed and the exclusion rate per
  class and source (review/review_summary.csv, for Question 1).

The sheets show only the image and its label: no model output, so the review is blind to models.
Sheets contain PlantWild thumbnails (CC BY-NC-ND): keep the review folder out of the submission,
except decisions.txt, sheet_index.csv, label_exclusions.csv and review_summary.csv.
"""
import argparse
import math
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

Image.MAX_IMAGE_PIXELS = None

# the reason codes of label_review_rules.md, and which classes each one may be used on
REVIEW_CODES = {"symptoms": "healthy", "no_symptoms": "disease", "not_leaf": "any", "wrong_crop": "any"}
COLS, ROWS, THUMB, GAP, HEADER, LABEL = 8, 5, 200, 6, 70, 26
PER_SHEET = COLS * ROWS


def font(size):
    try:
        return ImageFont.load_default(size=size)  # Pillow 10.1 or newer
    except TypeError:
        return ImageFont.load_default()


def thumb(path):
    """Whole image scaled into a THUMB x THUMB square (no cropping, so collages and text stay visible)."""
    cell = Image.new("RGB", (THUMB, THUMB), (235, 235, 235))
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im = ImageOps.contain(im, (THUMB, THUMB))
            cell.paste(im, ((THUMB - im.width) // 2, (THUMB - im.height) // 2))
    except Exception:
        ImageDraw.Draw(cell).text((10, 90), "cannot open", fill=(200, 0, 0), font=font(18))
    return cell


def draw_sheet(sheet_id, title, codes_hint, paths, out_file):
    W = COLS * (THUMB + GAP) + GAP
    H = HEADER + ROWS * (LABEL + THUMB + GAP) + GAP
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    d.text((GAP, 8), f"{sheet_id}   {title}", fill="black", font=font(26))
    d.text((GAP, 42), codes_hint, fill=(90, 90, 90), font=font(18))
    for k, p in enumerate(paths):
        x = GAP + (k % COLS) * (THUMB + GAP)
        y = HEADER + (k // COLS) * (LABEL + THUMB + GAP)
        d.rectangle([x, y, x + THUMB, y + LABEL - 2], fill="black")  # number strip above the image,
        d.text((x + 6, y + 1), str(k + 1), fill="yellow", font=font(22))  # so it hides nothing
        sheet.paste(thumb(p), (x, y + LABEL))
    sheet.save(out_file, quality=90)


def cmd_make(args):
    frames = [pd.read_csv(args.splits / f"{s}.csv", dtype=str, keep_default_na=False)
              for s in ("val_wild", "test_wild")]
    df = pd.concat(frames, ignore_index=True)
    df["kind"] = df.class_id.str.endswith("_healthy").map({True: "healthy", False: "disease"})
    # healthy classes first (where the problem is known), then disease classes; a fixed order
    df = df.sort_values(["kind", "split", "class_id", "source_split", "rel_path"],
                        key=lambda s: s.map({"healthy": 0, "disease": 1}) if s.name == "kind" else s)
    sheets_dir = args.out / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    dec_file = args.out / "decisions.txt"
    if dec_file.exists() and not args.force:
        sys.exit(f"{dec_file} already exists. It may hold your decisions, so it is not overwritten. "
                 f"Use --force only if you want to start again.")
    rows, lines, n = [], [], 0
    for (kind, split, cid), g in df.groupby(["kind", "split", "class_id"], sort=False):
        pages = math.ceil(len(g) / PER_SHEET)
        allowed = [c for c, k in REVIEW_CODES.items() if k in (kind, "any")]
        for p in range(pages):
            n += 1
            sid = f"S{n:03d}"
            part = g.iloc[p * PER_SHEET:(p + 1) * PER_SHEET]
            source = "/".join(sorted(set(part.dataset + " " + part.source_split)))
            title = f"{cid}  ({kind})   {split}: {source}   page {p + 1}/{pages}"
            draw_sheet(sid, title, "codes for this class: " + ", ".join(allowed) + "   |   unsure = keep",
                       part.path.tolist(), sheets_dir / f"{sid}.jpg")
            for k, r in enumerate(part.itertuples(index=False), 1):
                rows.append({"sheet": sid, "number": k, "split": r.split, "dataset": r.dataset,
                             "source_split": r.source_split, "rel_path": r.rel_path, "class_id": r.class_id,
                             "kind": kind})
            lines.append(f"{sid}:")
            print(f"  {sid}  {title}  ({len(part)} images)")
    pd.DataFrame(rows).to_csv(args.out / "sheet_index.csv", index=False)
    header = ["# One line per sheet. Write the image numbers to exclude, each group followed by its code.",
              "# Codes: symptoms (healthy classes), no_symptoms (disease classes), not_leaf, wrong_crop (any class).",
              "# 'none' = sheet reviewed, nothing excluded. Empty = not reviewed yet. Unsure = keep.",
              "# Example:  S003: 4 18 symptoms, 23 not_leaf", ""]
    dec_file.write_text("\n".join(header + lines) + "\n", encoding="utf-8")
    print(f"\n{n} sheets, {len(rows)} images -> {sheets_dir}\nNow fill in {dec_file}, then run: "
          f"python review_labels.py build --out {args.out}")


def parse_decisions(text, index):
    """{(sheet, number): code} and the set of reviewed sheets. Stops with every error found."""
    sizes = index.groupby("sheet").number.max().to_dict()
    kind = index.groupby("sheet").kind.first().to_dict()
    decided, reviewed, errors, seen = {}, set(), [], set()
    for ln, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.fullmatch(r"(S\d{3})\s*:\s*(.*)", line, flags=re.I)
        if not m:
            errors.append(f"line {ln}: cannot read '{raw.strip()}' (expected 'S001: ...')")
            continue
        sid, body = m.group(1).upper(), m.group(2).strip()
        if sid not in sizes:
            errors.append(f"line {ln}: there is no sheet {sid}")
            continue
        if sid in seen:
            errors.append(f"line {ln}: sheet {sid} appears twice")
            continue
        seen.add(sid)
        if not body:
            continue  # not reviewed yet
        reviewed.add(sid)
        if body.lower() == "none":
            continue
        for chunk in re.split(r"[;,]", body):
            words = chunk.split()
            if not words:
                continue
            code = words[-1].lower()
            if code not in REVIEW_CODES:
                errors.append(f"line {ln}: '{words[-1]}' is not a code ({', '.join(REVIEW_CODES)})")
                continue
            if REVIEW_CODES[code] not in ("any", kind[sid]):
                errors.append(f"line {ln}: '{code}' is only for {REVIEW_CODES[code]} classes; {sid} is {kind[sid]}")
                continue
            if len(words) == 1:
                errors.append(f"line {ln}: '{code}' has no image numbers before it")
            for w in words[:-1]:
                if not w.isdigit() or not 1 <= int(w) <= sizes[sid]:
                    errors.append(f"line {ln}: '{w}' is not an image number on {sid} (1-{sizes[sid]})")
                elif (sid, int(w)) in decided:
                    errors.append(f"line {ln}: image {w} on {sid} is listed twice")
                else:
                    decided[(sid, int(w))] = code
    if errors:
        sys.exit("decisions.txt has mistakes, nothing was written:\n  " + "\n  ".join(errors))
    return decided, reviewed


def cmd_build(args):
    index = pd.read_csv(args.out / "sheet_index.csv", dtype={"number": int}, keep_default_na=False)
    decided, reviewed = parse_decisions((args.out / "decisions.txt").read_text(encoding="utf-8"), index)
    index["reviewed"] = index.sheet.isin(reviewed)
    index["reason"] = [decided.get((s, k), "") for s, k in zip(index.sheet, index.number)]

    ex = index[index.reason != ""]
    cols = ["dataset", "source_split", "rel_path", "reason", "class_id", "split", "sheet", "number"]
    ex[cols].to_csv(args.out / "label_exclusions.csv", index=False)

    done = index[index.reviewed]
    summary = (done.assign(excluded=done.reason != "")
               .groupby(["split", "dataset", "source_split", "kind", "class_id"])
               .agg(reviewed=("rel_path", "size"), excluded=("excluded", "sum"),
                    **{c: ("reason", lambda s, c=c: int((s == c).sum())) for c in REVIEW_CODES})
               .reset_index())
    summary["excluded_pct"] = (100 * summary.excluded / summary.reviewed).round(1)
    summary.to_csv(args.out / "review_summary.csv", index=False)

    n_sheets = index.sheet.nunique()
    todo = sorted(set(index.sheet) - reviewed)
    print(f"Sheets reviewed: {len(reviewed)} of {n_sheets}; images reviewed: {len(done)} of {len(index)}")
    if len(done):
        by = (done.assign(excluded=done.reason != "").groupby(["split", "dataset", "source_split", "kind"])
              .agg(reviewed=("rel_path", "size"), excluded=("excluded", "sum")))
        by["excluded_pct"] = (100 * by.excluded / by.reviewed).round(1)
        print(by.to_string())
        print("\nBy reason:", ", ".join(f"{c} {int((done.reason == c).sum())}" for c in REVIEW_CODES))
    print(f"\nWrote {args.out / 'label_exclusions.csv'} ({len(ex)} images) and {args.out / 'review_summary.csv'}")
    if todo:
        print(f"NOT FINISHED: {len(todo)} sheets still empty ({', '.join(todo[:8])}{' ...' if len(todo) > 8 else ''}). "
              f"Do not run make_splits.py --exclude until every sheet is reviewed.")
    else:
        print("Every sheet reviewed. Next: python make_splits.py ... --exclude "
              f"{args.out / 'label_exclusions.csv'} --out splits_v2")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="make the numbered review sheets")
    m.add_argument("--splits", type=Path, default=Path("splits"))
    m.add_argument("--out", type=Path, default=Path("review"))
    m.add_argument("--force", action="store_true", help="overwrite an existing decisions.txt")
    b = sub.add_parser("build", help="check decisions.txt and write label_exclusions.csv")
    b.add_argument("--out", type=Path, default=Path("review"))
    args = ap.parse_args()
    {"make": cmd_make, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()