"""Train the digit classifier.

  python train_cnn.py --data dataset --synthetic 600 --epochs 30 --out models/digit_cnn.pt
  python train_cnn.py --data dataset --backend sklearn --out models/digit_mlp.pkl   # no PyTorch needed

`dataset/` = folders 0..9 of 28x28 PNG crops (made by `python -m sevenseg.crops export ...`
then corrected by hand).  --synthetic N adds crops from N generated meter images.
"""
import argparse
import os
import pickle

import numpy as np

from sevenseg.classifier import augment, build_torch_model
from sevenseg.crops import load_folder, synthetic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--synthetic", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--backend", choices=["torch", "sklearn"], default="torch")
    ap.add_argument("--out", default="models/digit_cnn.pt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    Xs, ys = [], []
    if a.data and os.path.isdir(a.data):
        X, y = load_folder(a.data); print("real crops:", len(X)); Xs.append(X); ys.append(y)
    if a.synthetic:
        X, y = synthetic(a.synthetic, a.seed); print("synthetic crops:", len(X)); Xs.append(X); ys.append(y)
    X, y = np.concatenate(Xs), np.concatenate(ys)
    idx = rng.permutation(len(X)); cut = int(0.85 * len(X))
    tr, va = idx[:cut], idx[cut:]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    def aug_set(ix, k):
        return np.stack([augment(X[i], rng) for i in np.repeat(ix, k)]), np.repeat(y[ix], k)

    if a.backend == "sklearn":
        from sklearn.neural_network import MLPClassifier
        Xa, ya = aug_set(tr, 4)
        m = MLPClassifier((128, 64), max_iter=200, random_state=a.seed).fit(Xa.reshape(len(Xa), -1), ya)
        print("val acc:", m.score(X[va].reshape(len(va), -1), y[va]))
        pickle.dump(m, open(a.out, "wb")); return

    import torch, torch.nn as nn
    model = build_torch_model(); opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 5e-3, total_steps=a.epochs * (len(tr) * 3 // 64 + 1))
    Xv = torch.from_numpy(X[va][:, None]); yv = torch.from_numpy(y[va])
    best = 0.0
    for ep in range(a.epochs):
        model.train(); Xa, ya = aug_set(tr, 3); p = rng.permutation(len(Xa))
        for i in range(0, len(p), 64):
            b = p[i:i + 64]
            loss = nn.functional.cross_entropy(model(torch.from_numpy(Xa[b][:, None])), torch.from_numpy(ya[b]))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xv).argmax(1) == yv).float().mean().item()
        print(f"epoch {ep + 1:2d}  loss {loss.item():.3f}  val acc {acc:.3f}")
        if acc >= best:
            best = acc; torch.save(model.state_dict(), a.out)
    print("best val acc", best, "->", a.out)


if __name__ == "__main__":
    main()
