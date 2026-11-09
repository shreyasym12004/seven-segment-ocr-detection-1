"""python -m sevenseg.evaluate [--n 60] [--seed 0]  -> accuracy on a synthetic stress set."""
from __future__ import annotations

import argparse
import time

import numpy as np

from .pipeline import read_image
from .synth import random_reading, render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    conds = {"clean": {}, "italic": {"italic": 0.2}, "glare": {"glare": True},
             "noisy": {"noise": 14.0, "blur": 2}, "tilted": {"tilt": 0.6}, "no-ghost": {"ghost": False}}
    for name, kw in conds.items():
        ok, tot, ms = 0, 0, []
        for _ in range(a.n // len(conds) + 1):
            txt = random_reading(rng)
            img = render(txt, rng, **kw)
            t0 = time.perf_counter()
            r = read_image(img)
            ms.append((time.perf_counter() - t0) * 1000)
            ok += (r["raw_text"] == txt); tot += 1
        print(f"{name:9s} exact-match {ok}/{tot} = {ok / tot:5.1%}   mean {np.mean(ms):5.0f} ms")


if __name__ == "__main__":
    main()
