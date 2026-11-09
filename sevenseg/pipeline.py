"""End-to-end: image path -> JSON-able dict."""
from __future__ import annotations

import os
import time
from typing import Optional

import cv2
import numpy as np

from .digits import apply_shear, estimate_shear, find_digits
from .preprocess import binarize_strokes, enhance_gray, load_image
from .recognize import recognise
from .roi import ROI, find_display, locate_by_strokes

DISPLAY_H = 220   # rectified display is normalised to this height


def _read_display(roi_bgr: np.ndarray):
    s = DISPLAY_H / roi_bgr.shape[0]
    roi = cv2.resize(roi_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    gray = enhance_gray(roi)
    binary = binarize_strokes(gray)
    boxes, clean = find_digits(binary)
    if len(boxes) >= 2:                               # de-slant italic fonts, then redo
        top = min(b.y for b in boxes); bot = max(b.y + b.h for b in boxes)
        shear = estimate_shear(clean[top:bot])
        if abs(shear) > 0.04:
            gray = apply_shear(gray, shear)
            binary = binarize_strokes(gray)
            boxes, clean = find_digits(binary)
    if len(boxes) > 1 and boxes[-1].w / boxes[-1].h > 0.65:
        end = boxes[-2].x + boxes[-2].w + int(0.08 * DISPLAY_H)
        if end < roi.shape[1]:
            source_end = max(1, round(end * roi_bgr.shape[1] / roi.shape[1]))
            clean, boxes, disp, gray, _ = _read_display(roi_bgr[:, :source_end])
            return clean, boxes, disp, gray, True
    return clean, boxes, roi, gray, False


def _score(text: str, confs) -> float:
    if not confs:
        return 0.0
    ok = 0.0 if "?" in text else 1.0
    return (0.3 + 0.7 * ok) * float(np.mean(confs)) * min(1.0, len(confs) / 4)


def _recover_digit_row(img: np.ndarray, gray: np.ndarray):
    """Propose tight crops around horizontal clusters of tall, narrow digit strokes."""
    H, W = gray.shape[:2]
    binary = binarize_strokes(gray)
    best = None

    for frac in (0.06, 0.075, 0.09):
        kv = max(5, int(frac * H))
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((kv, 1), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(vertical, connectivity=8)
        strokes = [
            tuple(map(int, stats[i, :4]))
            for i in range(1, count)
            if kv <= stats[i, 3] <= 2.5 * kv and stats[i, 2] <= 0.7 * kv
        ]
        strokes.sort(key=lambda stroke: stroke[0])

        groups = []
        for stroke in strokes:
            if groups and stroke[0] - (groups[-1][-1][0] + groups[-1][-1][2]) <= 1.5 * kv:
                groups[-1].append(stroke)
            else:
                groups.append([stroke])

        for group in groups:
            if len(group) < 5:
                continue
            median_y = float(np.median([stroke[1] for stroke in group]))
            row = [stroke for stroke in group if abs(stroke[1] - median_y) <= 1.3 * kv]
            if len(row) < 5:
                continue

            x0 = min(stroke[0] for stroke in row)
            x1 = max(stroke[0] + stroke[2] for stroke in row)
            median_height = float(np.median([stroke[3] for stroke in row]))
            crop_h = max(2 * kv, round(2 * median_height))
            crop_w = round(3.1 * crop_h)
            center_x = (x0 + x1) / 2 - 0.15 * crop_h
            base_x = round(center_x - crop_w / 2)
            base_y = round(median_y) + 2

            for dx in (-4, 0, 4):
                for dy in (-2, 0, 2):
                    left = max(0, base_x + dx)
                    top = max(0, base_y + dy)
                    right = min(W, left + crop_w)
                    bottom = min(H, top + crop_h)
                    if right - left < 0.8 * crop_w or bottom - top < 0.8 * crop_h:
                        continue

                    crop = img[top:bottom, left:right]
                    clean, boxes, display, crop_gray, trimmed_unit = _read_display(crop)
                    text, confs = recognise(clean, boxes)
                    digit_count = sum(ch.isdigit() for ch in text)
                    if ("?" in text or text.count(".") > 1 or digit_count < 4
                            or ("." in text and not 1 <= len(text.rsplit(".", 1)[1]) <= 3)):
                        continue
                    score = float(np.mean(confs)) * min(1.0, digit_count / 6)
                    if "." in text:
                        score *= 1.05
                    if best is None or score > best[0]:
                        box = np.array(
                            [[left, top], [right, top], [right, bottom], [left, bottom]],
                            dtype=np.float32,
                        )
                        roi = ROI(crop.copy(), box, 0.5)
                        best = (score, text, confs, roi, clean, boxes, display, crop_gray, trimmed_unit)

    return best


def _fuse_with_classifier(text, confs, boxes, gray, clf):
    """Per digit: classifier and segment reader agree -> high confidence; otherwise trust the
    classifier (it saw real data) but with reduced confidence."""
    from .classifier import make_crop
    seg_chars = [c for c in text if c != "."]
    if len(seg_chars) != len(boxes):
        return text, confs
    labels, probs = clf.predict(np.stack([make_crop(gray, b) for b in boxes]))
    out, new_conf = "", []
    for b, seg, lab, p, c in zip(boxes, seg_chars, labels, probs, confs):
        if lab == seg:
            ch, cf = seg, max(float(c), float(p))
        elif float(p) > float(c):          # disagreement: the more confident reader wins
            ch, cf = lab, 0.7 * float(p)
        else:
            ch, cf = seg, 0.7 * float(c)
        out += ch + ("." if b.dp_after else "")
        new_conf.append(cf)
    return out, new_conf


def read_image(img: np.ndarray, debug: Optional[dict] = None, clf=None) -> dict:
    gray_full = enhance_gray(img)
    attempts = []
    for locate in (find_display, locate_by_strokes):    # 1) bezel/contour  2) stroke-based
        roi = locate(img, gray_full)
        clean, boxes, disp, gray, trimmed_unit = _read_display(roi.image)
        text, confs = recognise(clean, boxes)
        attempts.append((_score(text, confs) * (0.6 + 0.4 * roi.score), len(boxes), text, confs, roi, clean, boxes, disp, gray, trimmed_unit))
        if len(boxes) >= 3 and roi.score >= 0.5:       # a plausible display with digits: trust it
            break
    chosen = attempts[-1] if len(attempts[-1][6]) >= 3 and len(attempts) == 1 \
        else max(attempts, key=lambda a: a[0])
    sc, _, text, confs, roi, clean, boxes, disp, gray, trimmed_unit = chosen
    if len([ch for ch in text if ch.isdigit()]) < 4 or _score(text, confs) < 0.4:
        recovered = _recover_digit_row(img, gray_full)
        if recovered is not None and recovered[0] > max(0.6, _score(text, confs)):
            _, text, confs, roi, clean, boxes, disp, gray, trimmed_unit = recovered
    if clf is not None and boxes:
        if trimmed_unit:
            from .classifier import make_crop
            labels, probs = clf.predict(np.stack([make_crop(gray, b) for b in boxes]))
            text, confs = "".join(labels), probs.tolist()
        else:
            text, confs = _fuse_with_classifier(text, confs, boxes, gray, clf)
    elif trimmed_unit:
        text = text.rstrip(".")
    if debug is not None:
        debug.update(roi=roi, display=disp, binary=clean, boxes=boxes, text=text, gray=gray)
    value = None
    if text and "?" not in text:
        try:
            value = float(text) if "." in text else int(text)
        except ValueError:
            value = None
    conf = float(np.mean(confs)) * min(1.0, 0.6 + 0.4 * len(boxes) / 4) if confs else 0.0
    return {"detected_reading": value, "confidence_score": round(conf, 3), "raw_text": text}


def read_meter(path: str, debug: Optional[dict] = None, clf=None) -> dict:
    t0 = time.perf_counter()
    img = load_image(path)
    res = read_image(img, debug, clf)
    return {
        "filename": os.path.basename(path),
        "detected_reading": res["detected_reading"],
        "confidence_score": res["confidence_score"],
        "processing_time_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
