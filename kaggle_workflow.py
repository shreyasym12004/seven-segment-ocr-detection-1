"""End-to-end workflow for the Kaggle digital-meter dataset.

  python kaggle_workflow.py download                      # needs ~/.kaggle/kaggle.json
  python kaggle_workflow.py inspect  data/kaggle
  python kaggle_workflow.py split    data/kaggle --heldout 0.2
  python kaggle_workflow.py export   [--labels labels.csv]   # train images -> dataset/0..9
  python kaggle_workflow.py train    [--backend torch|sklearn] [--epochs 30] [--synthetic 600]
  python kaggle_workflow.py evaluate --labels labels.csv  [--model models/digit_cnn.pt]

labels.csv (optional, any file with these two columns):  filename,reading     e.g.  meter_01.jpg,128.4
With labels, `export` labels crops automatically (only images whose digit count matches the
reading are used); without labels, crops are filed under the segment-reader's guess and you
fix them with label_crops.py.
"""
import argparse
import collections
import csv
import glob
import os
import random
import shutil
import subprocess
import sys

import cv2
import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
TRAIN_DIR, HELD_DIR, DATASET = "data/train_images", "data/heldout", "dataset"


def list_images(root):
    return sorted(p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
                  if p.lower().endswith(IMG_EXT))


def read_labels(path):
    """filename,reading CSV -> {basename: reading-string}. Column names are matched loosely."""
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        cols = {c.lower().strip(): c for c in rd.fieldnames}
        fcol = next(cols[c] for c in cols if c in ("filename", "file", "image", "img", "name", "path"))
        rcol = next(cols[c] for c in cols if c in ("reading", "label", "value", "text", "number", "ground_truth"))
        for r in rd:
            out[os.path.basename(r[fcol].strip())] = r[rcol].strip()
    return out


# ------------------------------------------------------------------ commands
def cmd_download(a):
    subprocess.check_call(["kaggle", "datasets", "download", "-d", "prathamjindal/digital-meter",
                           "--unzip", "-p", a.out])
    print("downloaded to", a.out)


def cmd_inspect(a):
    c = collections.Counter()
    for r, _, fs in os.walk(a.root):
        for x in fs:
            c[(r, os.path.splitext(x)[1].lower())] += 1
    for (r, e), n in sorted(c.items()):
        print(f"{n:6d}  {e or '(none)':8s} {r}")
    imgs = list_images(a.root)
    print(f"\n{len(imgs)} images")
    for f in glob.glob(os.path.join(a.root, "**", "*.csv"), recursive=True) + \
             glob.glob(os.path.join(a.root, "**", "*.json"), recursive=True) + \
             glob.glob(os.path.join(a.root, "**", "*.txt"), recursive=True):
        print("\nannotation file:", f)
        with open(f, errors="ignore") as fh:
            print("".join(fh.readlines()[:4]))
    for p in random.Random(0).sample(imgs, min(3, len(imgs))):
        print(p, cv2.imread(p).shape if cv2.imread(p) is not None else "unreadable")


def cmd_split(a):
    imgs = list_images(a.root)
    random.Random(a.seed).shuffle(imgs)
    k = max(1, int(len(imgs) * a.heldout))
    for d, files in ((HELD_DIR, imgs[:k]), (TRAIN_DIR, imgs[k:])):
        shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
        for p in files:
            shutil.copy(p, os.path.join(d, os.path.basename(p)))
    print(f"{len(imgs) - k} train images -> {TRAIN_DIR}, {k} held-out -> {HELD_DIR}")
    print("NEVER train on the held-out images; they are only used by `evaluate`.")


def cmd_export(a):
    from sevenseg.classifier import make_crop
    from sevenseg.crops import crops_from_image
    from sevenseg.preprocess import load_image
    labels = read_labels(a.labels) if a.labels else {}
    shutil.rmtree(DATASET, ignore_errors=True)
    n_img = n_used = n_crop = 0
    for p in list_images(a.images):
        n_img += 1
        crops, guess = crops_from_image(load_image(p))
        if not crops:
            continue
        name = os.path.basename(p)
        if name in labels:                                   # trusted labels
            digits = [c for c in labels[name] if c.isdigit()]
            if len(digits) != len(crops):
                continue
            guess = digits
        for i, (c, ch) in enumerate(zip(crops, guess)):
            if ch == "?":
                continue
            d = os.path.join(DATASET, ch); os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, f"{os.path.splitext(name)[0]}_{i}.png"), (c * 255).astype(np.uint8))
            n_crop += 1
        n_used += 1
    print(f"{n_img} images scanned, {n_used} usable, {n_crop} crops written to {DATASET}/")
    for ch in "0123456789":
        print(ch, len(glob.glob(f"{DATASET}/{ch}/*.png")), end="   ")
    print("\n" + ("labels came from your CSV." if labels else "labels are GUESSES: run `python label_crops.py` to correct them."))


def cmd_train(a):
    out = a.out or ("models/digit_cnn.pt" if a.backend == "torch" else "models/digit_mlp.pkl")
    subprocess.check_call([sys.executable, "train_cnn.py", "--data", DATASET, "--backend", a.backend,
                           "--epochs", str(a.epochs), "--synthetic", str(a.synthetic), "--out", out])


def cmd_evaluate(a):
    from sevenseg import read_meter
    from sevenseg.classifier import load_classifier
    truth = read_labels(a.labels)
    clf = load_classifier(a.model) if a.model and os.path.exists(a.model) else None
    rows, seg_ok, hyb_ok = [], 0, 0
    files = [p for p in list_images(a.images) if os.path.basename(p) in truth]
    for p in files:
        t = float(truth[os.path.basename(p)])
        s = read_meter(p)["detected_reading"]
        h = read_meter(p, clf=clf)["detected_reading"] if clf else s
        seg_ok += s == t; hyb_ok += h == t
        rows.append((os.path.basename(p), t, s, h, "OK" if h == t else "FAIL"))
    n = max(len(files), 1)
    print(f"held-out images: {len(files)}\nsegment-only exact match: {seg_ok}/{len(files)} = {seg_ok / n:.1%}")
    if clf:
        print(f"hybrid (with model)     : {hyb_ok}/{len(files)} = {hyb_ok / n:.1%}")
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/evaluation.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["filename", "truth", "segment_only", "hybrid", "status"]); w.writerows(rows)
    print("per-image results -> outputs/evaluation.csv   (use the FAIL rows for the README failure section)")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("download"); s.add_argument("--out", default="data/kaggle")
    s = sub.add_parser("inspect"); s.add_argument("root")
    s = sub.add_parser("split"); s.add_argument("root"); s.add_argument("--heldout", type=float, default=0.2); s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("export"); s.add_argument("--images", default=TRAIN_DIR); s.add_argument("--labels")
    s = sub.add_parser("train"); s.add_argument("--backend", choices=["torch", "sklearn"], default="torch")
    s.add_argument("--epochs", type=int, default=30); s.add_argument("--synthetic", type=int, default=600); s.add_argument("--out")
    s = sub.add_parser("evaluate"); s.add_argument("--images", default=HELD_DIR); s.add_argument("--labels", required=True); s.add_argument("--model")
    a = ap.parse_args()
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
