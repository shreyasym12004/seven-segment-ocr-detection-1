"""Classical seven-segment state recognition (segments A..G -> digit)."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .digits import DigitBox

#            a b c d e f g
PATTERNS = {
    "0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg",
    "5": "acdfg", "6": "acdefg", "7": "abc", "8": "abcdefg", "9": "abcdfg",
}
VARIANTS = {"6": ["cdefg"], "7": ["abcf"], "9": ["bcdfg", "abcfg"]}   # alt glyph shapes
SEGS = "abcdefg"

# (x0, x1, y0, y1) sample windows as fractions of the digit box
ZONES = {
    "a": (0.20, 0.80, 0.00, 0.17),
    "g": (0.20, 0.80, 0.41, 0.59),
    "d": (0.20, 0.80, 0.83, 1.00),
    "f": (0.00, 0.32, 0.08, 0.46),
    "b": (0.68, 1.00, 0.08, 0.46),
    "e": (0.00, 0.32, 0.54, 0.92),
    "c": (0.68, 1.00, 0.54, 0.92),
}


def _lookup():
    table = {}
    for ch, segs in PATTERNS.items():
        table[tuple(int(s in segs) for s in SEGS)] = ch
    for ch, alts in VARIANTS.items():
        for segs in alts:
            table[tuple(int(s in segs) for s in SEGS)] = ch
    return table


TABLE = _lookup()


def segment_fills(binary: np.ndarray, box: DigitBox, is_one: bool) -> np.ndarray:
    """Fraction of 'ink' in each of the 7 sample windows."""
    sub = binary[box.y:box.y + box.h, box.x:box.x + box.w] > 0
    if is_one:  # a '1' is drawn on the right-hand segments: pad it to full digit width
        full_w = max(int(box.h * 0.55), box.w)
        pad = np.zeros((box.h, full_w), bool)
        pad[:, full_w - box.w:] = sub
        sub = pad
    H, W = sub.shape
    out = []
    for s in SEGS:
        x0, x1, y0, y1 = ZONES[s]
        win = sub[int(y0 * H):max(int(y1 * H), int(y0 * H) + 1), int(x0 * W):max(int(x1 * W), int(x0 * W) + 1)]
        out.append(float(win.mean()) if win.size else 0.0)
    return np.array(out)


def classify(fills: np.ndarray, on_level: float) -> Tuple[str, float]:
    """Soft nearest-pattern match. Returns (char, confidence 0..1)."""
    soft = np.clip(fills / max(on_level, 1e-6), 0, 1)
    best_ch, best_err = "?", 9.0
    for pat, ch in TABLE.items():
        err = float(np.abs(soft - np.array(pat)).mean())
        if err < best_err:
            best_ch, best_err = ch, err
    return best_ch, float(max(0.0, 1.0 - 2.0 * best_err))


def recognise(binary: np.ndarray, boxes: List[DigitBox]) -> Tuple[str, List[float]]:
    """Return the reading string (digits + optional '.') and per-digit confidences."""
    if not boxes:
        return "", []
    hs = np.array([b.h for b in boxes], float)
    ones = [(b.w / b.h) < 0.34 for b in boxes]
    all_fills = np.array([segment_fills(binary, b, o) for b, o in zip(boxes, ones)])
    on_level = max(float(np.percentile(all_fills, 85)), 0.25)
    text, confs = "", []
    for b, f, is_one in zip(boxes, all_fills, ones):
        ch, c = ("1", float(np.clip(1 - np.abs(np.clip(f / on_level, 0, 1) - np.array([0, 1, 1, 0, 0, 0, 0])).mean() * 2, 0, 1))) \
            if is_one else classify(f, on_level)
        text += ch + ("." if b.dp_after else "")
        confs.append(c)
    return text, confs
