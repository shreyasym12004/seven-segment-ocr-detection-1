"""Image conditioning: resizing, glare suppression, illumination flattening, binarisation."""
from __future__ import annotations

import cv2
import numpy as np

MAX_WIDTH = 900


def load_image(path: str, max_width: int = MAX_WIDTH) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return resize_keep_aspect(img, max_width)


def resize_keep_aspect(img: np.ndarray, max_width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    s = max_width / w
    return cv2.resize(img, (max_width, int(h * s)), interpolation=cv2.INTER_AREA)


def suppress_glare(img_bgr: np.ndarray) -> np.ndarray:
    """Inpaint specular highlights (very bright + low saturation pixels)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 2] > 245) & (hsv[..., 1] < 40)).astype(np.uint8) * 255
    if mask.sum() == 0:
        return img_bgr
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    if mask.mean() / 255 > 0.15:  # huge saturated area carries no information
        return img_bgr
    return cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)


def enhance_gray(img_bgr: np.ndarray) -> np.ndarray:
    """Grayscale + denoise + CLAHE contrast stretch."""
    gray = cv2.cvtColor(suppress_glare(img_bgr), cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 40, 5)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)


def flatten_illumination(gray: np.ndarray) -> np.ndarray:
    """Divide by a heavily blurred copy -> removes gradients, shadows and glare halos."""
    k = max(15, (min(gray.shape[:2]) // 3) | 1)
    bg = cv2.GaussianBlur(gray, (k, k), 0).astype(np.float32) + 1.0
    return np.clip(gray.astype(np.float32) / bg * 160.0, 0, 255).astype(np.uint8)


def binarize_strokes(gray: np.ndarray) -> np.ndarray:
    """Dark LCD strokes -> 255, background -> 0 (expects an already cropped display)."""
    flat = cv2.GaussianBlur(flatten_illumination(gray), (3, 3), 0)
    block = max(11, (gray.shape[0] // 4) | 1)
    adapt = cv2.adaptiveThreshold(flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, block, 10)
    _, otsu = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # adaptive copes with uneven light; Otsu rejects faint "ghost" segments -> intersect
    out = cv2.bitwise_and(adapt, otsu)
    return cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
