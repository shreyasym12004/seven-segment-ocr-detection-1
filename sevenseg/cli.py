"""python -m sevenseg.cli IMAGE [IMAGE ...] [--debug-dir DIR]"""
from __future__ import annotations

import argparse
import json
import os

import cv2

from .pipeline import read_meter


def save_debug(path: str, dbg: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    disp = dbg["display"].copy()
    for b in dbg["boxes"]:
        cv2.rectangle(disp, (b.x, b.y), (b.x + b.w, b.y + b.h), (0, 200, 0), 2)
        if b.dp_after:
            cv2.circle(disp, (b.x + b.w + 6, b.y + b.h - 4), 4, (0, 0, 255), -1)
    cv2.putText(disp, dbg["text"], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    bin_bgr = cv2.cvtColor(dbg["binary"], cv2.COLOR_GRAY2BGR)
    bin_bgr = cv2.resize(bin_bgr, (disp.shape[1], disp.shape[0]))
    stem = os.path.splitext(os.path.basename(path))[0]
    cv2.imwrite(os.path.join(out_dir, f"{stem}_debug.png"), cv2.vconcat([disp, bin_bgr]))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Seven-segment meter OCR")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--debug-dir", help="write annotated debug images here")
    ap.add_argument("--model", help="trained digit classifier (.pt or .pkl) for the hybrid reader")
    args = ap.parse_args(argv)
    clf = None
    if args.model:
        from .classifier import load_classifier
        clf = load_classifier(args.model)
    results = []
    for p in args.images:
        dbg = {} if args.debug_dir else None
        res = read_meter(p, dbg, clf)
        if dbg:
            save_debug(p, dbg, args.debug_dir)
        results.append(res)
    print(json.dumps(results[0] if len(results) == 1 else results, indent=2))


if __name__ == "__main__":
    main()
