"""Per-channel physical contact + spectrum plausibility for a session.

    python tools/live_eval/contact.py [sessions/<folder> ...]   (default: today's sessions)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FS, RAIL_UV, analysis_filter, load_raw, newest_sessions, seg_psd  # noqa: E402


def band(f, P, lo, hi):
    m = (f >= lo) & (f < hi)
    return P[..., m, :].sum(axis=-2) * (f[1] - f[0])


def report(d: Path) -> None:
    t, x, labels = load_raw(d)
    dur = len(x) / FS
    print(f"\n=== {d.name}  {dur:.0f} s  ({len(x)} samples, gaps>0.1s: {(np.diff(t) > 0.1).sum()})")
    # raw: rail, DC, drift, hum
    fr, Praw, _ = seg_psd(x - x.mean(axis=0), FS)
    hum = band(fr, Praw, 58, 62) / np.maximum(band(fr, Praw, 1, 100), 1e-12)
    xf = analysis_filter(x)
    f, P, st = seg_psd(xf, FS)
    n = int(2 * FS)
    p2p = np.array([xf[s:s + n].max(axis=0) - xf[s:s + n].min(axis=0) for s in st])
    amax = np.array([np.abs(xf[s:s + n]).max(axis=0) for s in st])
    clean = (p2p <= 150) & (amax <= 100)
    railed = np.array([(np.abs(x[s:s + n]) >= 0.9 * RAIL_UV).mean(axis=0) > 0.5 for s in st])
    print(f"{'ch':>4} {'railed%':>7} {'dc_mV':>7} {'hum_med':>7} {'clean%':>6} {'p2p_med':>8} {'th/4-45':>7} {'th/4-30':>7} "
          f"{'slope':>6} {'th_pk':>6} {'al_pk':>6} {'rms1-45':>8}")
    for i, lab in enumerate(labels):
        c = clean[:, i] & ~railed[:, i]
        if c.sum() >= 5:
            Pm = np.median(P[c][:, :, i], axis=0)
            th = Pm[(f >= 4) & (f < 7)].sum(); tot = Pm[(f >= 4) & (f < 45)].sum(); t30 = Pm[(f >= 4) & (f < 30)].sum()
            # aperiodic fit (log-log) on 2-4 + 13-40 Hz, peaks = max log10 excess in theta / alpha
            m = ((f >= 2) & (f < 3.5)) | ((f >= 14) & (f <= 40))
            a, b = np.polyfit(np.log10(f[m]), np.log10(Pm[m]), 1)
            ex = np.log10(Pm) - (a * np.log10(np.maximum(f, 1e-9)) + b)
            thp = ex[(f >= 4) & (f < 7.5)].max(); alp = ex[(f >= 8) & (f < 13)].max()
            rms = np.sqrt(Pm[(f >= 1) & (f < 45)].sum() * (f[1] - f[0]))
            s = f"{th / tot:7.2f} {th / t30:7.2f} {a:6.2f} {thp:6.2f} {alp:6.2f} {rms:8.1f}"
        else:
            s = "   (too few clean segments)"
        print(f"{lab:>4} {100 * railed[:, i].mean():7.0f} {x[:, i].mean() / 1000:7.1f} {np.median(hum[:, i]):7.2f} "
              f"{100 * c.mean():6.0f} {np.median(p2p[:, i]):8.0f} {s}")
    live = [i for i in range(len(labels)) if railed[:, i].mean() < 0.5]
    print(f"segments bad on ALL non-railed channels: {100 * (~clean[:, live]).all(axis=1).mean():.0f}%")
    c = np.corrcoef(xf[:, live].T)
    print("corr (1-45 Hz) between non-railed channels:", [labels[i] for i in live])
    print(np.round(c, 2))


if __name__ == "__main__":
    ds = [Path(a) for a in sys.argv[1:]] or newest_sessions()
    for d in ds:
        report(d)
