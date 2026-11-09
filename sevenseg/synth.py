"""Synthetic seven-segment meter images (for unit tests and robustness evaluation)."""
from __future__ import annotations

import cv2
import numpy as np

from .recognize import PATTERNS


def render(text: str, rng: np.random.Generator, italic=0.0, glare=False, noise=4.0,
           blur=1, tilt=0.0, ghost=True, size=(240, 560)) -> np.ndarray:
    H, W = size
    bg = np.full((H, W, 3), (150, 165, 150), np.uint8)          # greenish LCD
    cv2.rectangle(bg, (0, 0), (W - 1, H - 1), (60, 60, 60), 6)
    chars = [(c, None) for c in text if c != "."]
    dpafter = {i for i, c in enumerate([c for c in text if c != "."]) if False}
    idx, dp_idx = 0, set()
    for c in text:
        if c == ".":
            dp_idx.add(idx - 1)
        else:
            idx += 1
    n = len(chars)
    dh = int(H * 0.5); dw = int(dh * 0.5); t = max(4, dh // 9); gap = int(dw * 0.45)
    total = n * dw + (n - 1) * gap
    x0 = (W - total) // 2; y0 = int(H * 0.28)
    ink, ghostc = (45, 50, 45), (128, 142, 128)
    layer = bg.copy()
    for i, (c, _) in enumerate(chars):
        x = x0 + i * (dw + gap)
        segs = {"a": (x + t, y0, x + dw - t, y0 + t), "g": (x + t, y0 + dh // 2 - t // 2, x + dw - t, y0 + dh // 2 + t // 2),
                "d": (x + t, y0 + dh - t, x + dw - t, y0 + dh), "f": (x, y0 + t // 2, x + t, y0 + dh // 2),
                "b": (x + dw - t, y0 + t // 2, x + dw, y0 + dh // 2), "e": (x, y0 + dh // 2, x + t, y0 + dh - t // 2),
                "c": (x + dw - t, y0 + dh // 2, x + dw, y0 + dh - t // 2)}
        for s, (xa, ya, xb, yb) in segs.items():
            on = s in PATTERNS[c]
            if on or ghost:
                cv2.rectangle(layer, (xa, ya), (xb, yb), ink if on else ghostc, -1)
        if i in dp_idx:
            cv2.circle(layer, (x + dw + gap // 2, y0 + dh - t // 2), max(3, t // 2 + 1), ink, -1)
    M = np.float32([[1, -italic, italic * (y0 + dh / 2)], [0, 1, 0]])
    img = cv2.warpAffine(layer, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.clip(img.astype(np.float32) * (0.75 + 0.35 * xx / W)[..., None], 0, 255)   # light gradient
    if glare:
        mask = np.zeros((H, W), np.float32)      # soft, semi-transparent specular patch
        cv2.ellipse(mask, (int(W * rng.uniform(.3, .7)), int(H * .35)), (int(W * .22), int(H * .22)), 20, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 18)[..., None] * 0.6
        img = img * (1 - mask) + 255 * mask
    img += rng.normal(0, noise, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    # place on a light meter body + perspective
    canvas = np.full((H + 160, W + 200, 3), (205, 208, 205), np.uint8)
    canvas[80:80 + H, 100:100 + W] = img
    src = np.float32([[100, 80], [100 + W, 80], [100 + W, 80 + H], [100, 80 + H]])
    d = tilt * 40
    dst = src + np.float32([[d, d], [-d, 0], [-d, -d], [d, 0]])
    P = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(canvas, P, (W + 200, H + 160), borderMode=cv2.BORDER_REPLICATE)


def random_reading(rng: np.random.Generator) -> str:
    n = int(rng.integers(4, 7))
    digits = "".join(str(int(x)) for x in rng.integers(0, 10, n))
    if rng.random() < 0.7:
        k = int(rng.integers(1, n))
        digits = digits[:k] + "." + digits[k:]
    return digits
