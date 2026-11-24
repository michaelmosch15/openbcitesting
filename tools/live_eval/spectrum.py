"""Blink-free median spectrum per channel (is there a theta / alpha peak above 1/f?).
    python tools/live_eval/spectrum.py [session ...]"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FS, newest_sessions  # noqa: E402
from candidates import blink_mask, load_session  # noqa: E402
from dsp import psd  # noqa: E402

for d in ([Path(s) for s in sys.argv[1:]] or newest_sessions()):
    t, xf, labs = load_session(d)
    bm = blink_mask(xf)
    seg = int(2 * FS)
    S, ok = [], []
    for s in range(0, len(xf) - seg + 1, seg // 2):
        x = xf[s:s + seg]
        f, P = psd(x, FS)
        S.append(P)
        ok.append((~bm[s:s + seg].any()) & (x.max(axis=0) - x.min(axis=0) <= 150) & (np.abs(x).max(axis=0) <= 100))
    S, ok = np.asarray(S), np.asarray(ok)
    print(f"\n=== {d.name}: blink-masked fraction of samples {bm.mean():.0%}; clean 2-s segments per channel "
          + " ".join(f"{l}:{int(n)}" for l, n in zip(labs, ok.sum(axis=0))) + f" of {len(ok)}")
    fr = np.arange(2, 14.5, 0.5)
    print("  Hz  " + " ".join(f"{v:5.1f}" for v in fr) + " | 16-20 20-30 30-45")
    for c, lab in enumerate(labs):
        if ok[:, c].sum() < 8:
            continue
        Pm = np.median(S[ok[:, c]][:, :, c], axis=0)
        hi = [np.mean(Pm[(f >= a) & (f < b)]) for a, b in ((16, 20), (20, 30), (30, 45))]
        print(f"{lab:>5} " + " ".join(f"{Pm[np.argmin(np.abs(f - v))]:5.2f}" for v in fr) + " | " + " ".join(f"{v:5.2f}" for v in hi))
        fit = ((f >= 2) & (f < 3.5)) | ((f >= 14) & (f <= 30))
        a, b = np.polyfit(np.log10(f[fit]), np.log10(Pm[fit]), 1)
        ex = np.log10(Pm) - (a * np.log10(np.maximum(f, 1e-9)) + b)
        print(f"   x1/f " + " ".join(f"{10 ** ex[np.argmin(np.abs(f - v))]:5.2f}" for v in fr) + f" | slope {a:.2f}")
    # mean over the usable channels (not AF3), to see a shared peak
    use = [c for c, l in enumerate(labs) if l != "AF3" and ok[:, c].sum() >= 8]
    allok = ok[:, use].all(axis=1)
    if allok.sum() >= 8:
        Pm = np.median(S[allok][:, :, use].mean(axis=2), axis=0)
        print(f"  avg  " + " ".join(f"{Pm[np.argmin(np.abs(f - v))]:5.2f}" for v in fr) + f"   (n={allok.sum()} segments clean on {[labs[c] for c in use]})")
