"""Learned digit classifier (0-9) on 28x28 crops. Backends: PyTorch CNN (.pt) or scikit-learn (.pkl)."""
from __future__ import annotations

import pickle
from typing import Tuple

import cv2
import numpy as np

CROP = 28
CLASSES = "0123456789"


def make_crop(gray: np.ndarray, box) -> np.ndarray:
    """Digit box -> 28x28 float32 in [0,1], strokes bright. Aspect ratio is preserved (so a
    narrow '1' stays narrow), illumination is normalised per crop."""
    pad = max(2, int(0.08 * box.h))
    y0, y1 = max(0, box.y - pad), min(gray.shape[0], box.y + box.h + pad)
    x0, x1 = max(0, box.x - pad), min(gray.shape[1], box.x + box.w + pad)
    return normalise_crop(gray[y0:y1, x0:x1])


def normalise_crop(g: np.ndarray) -> np.ndarray:
    g = g.astype(np.float32)
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    g = np.clip((g - lo) / max(hi - lo, 1.0), 0, 1)
    g = 1.0 - g                                   # dark strokes -> bright
    h, w = g.shape
    s = (CROP - 4) / max(h, w)
    g = cv2.resize(g, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    out = np.zeros((CROP, CROP), np.float32)
    y, x = (CROP - g.shape[0]) // 2, (CROP - g.shape[1]) // 2
    out[y:y + g.shape[0], x:x + g.shape[1]] = g
    return out


def augment(c: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    M = np.float32([[1 + rng.normal(0, .06), rng.normal(0, .12), rng.normal(0, 1.5)],
                    [rng.normal(0, .03), 1 + rng.normal(0, .06), rng.normal(0, 1.5)]])
    c = cv2.warpAffine(c, M, (CROP, CROP), borderValue=0)
    c = c * rng.uniform(.6, 1.2) + rng.uniform(-.1, .15)
    if rng.random() < .5:
        c = cv2.GaussianBlur(c, (0, 0), rng.uniform(.3, 1.0))
    return np.clip(c + rng.normal(0, rng.uniform(0, .08), c.shape), 0, 1).astype(np.float32)


# ------------------------------------------------------------------ backends
class SkClassifier:
    def __init__(self, model):
        self.model = model

    def predict(self, crops: np.ndarray) -> Tuple[list, np.ndarray]:
        proba = self.model.predict_proba(crops.reshape(len(crops), -1))
        idx = proba.argmax(1)
        return [CLASSES[self.model.classes_[i]] for i in idx], proba.max(1)


def build_torch_model():
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),   # 14
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),  # 7
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, 10))


class TorchClassifier:
    def __init__(self, path: str):
        import torch
        self.torch = torch
        self.model = build_torch_model()
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
        self.model.eval()

    def predict(self, crops: np.ndarray):
        t = self.torch
        with t.no_grad():
            p = t.softmax(self.model(t.from_numpy(crops[:, None]).float()), 1).numpy()
        return [CLASSES[i] for i in p.argmax(1)], p.max(1)


def load_classifier(path: str):
    if path.endswith(".pt"):
        return TorchClassifier(path)
    with open(path, "rb") as f:
        return SkClassifier(pickle.load(f))
