"""Build a digit dataset.
  export  : run the pipeline on real images, save every digit crop into  OUT/<guess>/xxx.png
            -> you only have to MOVE wrongly-guessed crops into the right folder (fast labelling).
  synth   : generate labelled synthetic crops (used to bootstrap / augment training).
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np

from .classifier import make_crop
from .pipeline import _read_display
from .preprocess import enhance_gray, load_image
from .recognize import recognise
from .roi import find_display, locate_by_strokes
from .synth import random_reading, render


def crops_from_image(img: np.ndarray):
    """Best-effort crops + the segment-reader's guess for each."""
    g = enhance_gray(img)
    for locate in (find_display, locate_by_strokes):
        roi = locate(img, g)
        clean, boxes, _, gray, _ = _read_display(roi.image)
        if len(boxes) >= 3:
            text, _ = recognise(clean, boxes)
            guess = [c for c in text if c != "."]
            if len(guess) == len(boxes):
                return [make_crop(gray, b) for b in boxes], guess
    return [], []


def export(images, out_dir):
    n = 0
    for p in images:
        img = load_image(p)
        crops, guess = crops_from_image(img)
        stem = os.path.splitext(os.path.basename(p))[0]
        for i, (c, ch) in enumerate(zip(crops, guess)):
            d = os.path.join(out_dir, ch if ch != "?" else "unknown")
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, f"{stem}_{i}.png"), (c * 255).astype(np.uint8))
            n += 1
    print(f"saved {n} crops to {out_dir}")


def synthetic(n_images, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    conds = [{}, {"italic": .2}, {"glare": True}, {"noise": 12., "blur": 2}, {"tilt": .6}, {"ghost": False}]
    for k in range(n_images):
        txt = random_reading(rng)
        img = render(txt, rng, **conds[k % len(conds)])
        crops, _ = crops_from_image(img)
        digits = [c for c in txt if c != "."]
        if len(crops) == len(digits):
            X += crops; y += [int(c) for c in digits]
    return np.array(X, np.float32), np.array(y)


def load_folder(root):
    X, y = [], []
    for c in range(10):
        for f in glob.glob(os.path.join(root, str(c), "*.png")):
            im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if im is not None:
                X.append(cv2.resize(im, (28, 28)).astype(np.float32) / 255.0); y.append(c)
    return np.array(X, np.float32), np.array(y)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("images_dir"); e.add_argument("out_dir")
    a = ap.parse_args()
    if a.cmd == "export":
        files = sorted(sum((glob.glob(os.path.join(a.images_dir, "**", ext), recursive=True)
                            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")), []))
        export(files, a.out_dir)
