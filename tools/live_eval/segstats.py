"""Per-2-s-segment statistics: is 'theta' (4-7 Hz) power driven by blink / slow-wave leakage?
    python tools/live_eval/segstats.py [session] [t0_s t1_s]"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FS, analysis_filter, load_raw, newest_sessions, seg_psd  # noqa: E402

d = Path(sys.argv[1]) if len(sys.argv) > 1 else newest_sessions()[-1]
t, x, labels = load_raw(d)
if len(sys.argv) > 3:
    a, b = int(float(sys.argv[2]) * FS), int(float(sys.argv[3]) * FS)
    x = x[a:b]
xf = analysis_filter(x)
f, P, st = seg_psd(xf, FS)
df = f[1] - f[0]
n = int(2 * FS)
bp = lambda lo, hi: P[:, (f >= lo) & (f < hi), :].sum(axis=1) * df
p2p = np.array([xf[s:s + n].max(axis=0) - xf[s:s + n].min(axis=0) for s in st])
amax = np.array([np.abs(xf[s:s + n]).max(axis=0) for s in st])
clean = (p2p <= 150) & (amax <= 100)
live = [i for i in range(len(labels)) if np.abs(x[:, i]).mean() < 150000]
lth, ldl, lal, lbe, lhi = (np.log(bp(4, 7)), np.log(bp(1, 3.5)), np.log(bp(8, 13)), np.log(bp(13, 30)), np.log(bp(30, 45)))
r = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if len(a) > 5 else float("nan")
print(d.name, f"{len(st)} segments")
print(f"{'ch':>4} {'clean%':>6} | ALL segs: r(th,delta) r(th,p2p) r(th,beta) | CLEAN segs: r(th,delta) r(th,p2p) r(th,beta) r(al,delta) | med theta uV2: clean / dirty")
for i in live:
    c = clean[:, i]
    lp = np.log(p2p[:, i])
    print(f"{labels[i]:>4} {100 * c.mean():6.0f} | {r(lth[:, i], ldl[:, i]):6.2f} {r(lth[:, i], lp):6.2f} {r(lth[:, i], lbe[:, i]):6.2f} | "
          f"{r(lth[c, i], ldl[c, i]):6.2f} {r(lth[c, i], lp[c]):6.2f} {r(lth[c, i], lbe[c, i]):6.2f} {r(lal[c, i], ldl[c, i]):6.2f} | "
          f"{np.exp(np.median(lth[c, i])) if c.any() else float('nan'):7.1f} / {np.exp(np.median(lth[~c, i])) if (~c).any() else float('nan'):9.1f}")
print("\ncross-channel r of per-segment log power, segments clean on both (theta | alpha | beta | 30-45):")
for a in live:
    for b in live:
        if a < b:
            c = clean[:, a] & clean[:, b]
            if c.sum() > 20:
                print(f"  {labels[a]:>3}-{labels[b]:<3} n={c.sum():3d}  {r(lth[c, a], lth[c, b]):5.2f} | {r(lal[c, a], lal[c, b]):5.2f} | {r(lbe[c, a], lbe[c, b]):5.2f} | {r(lhi[c, a], lhi[c, b]):5.2f}")
# median spectrum of clean segments, 1-Hz resolution print
print("\nmedian PSD of CLEAN segments (uV^2/Hz) by frequency:")
fs_ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 18, 22, 26, 30, 35, 40, 45]
print("  Hz " + " ".join(f"{v:6d}" for v in fs_))
for i in live:
    c = clean[:, i]
    if c.sum() < 5:
        continue
    Pm = np.median(P[c][:, :, i], axis=0)
    print(f"{labels[i]:>4} " + " ".join(f"{Pm[np.argmin(np.abs(f - v))]:6.2f}" for v in fs_))
    Pd = np.median(P[~c][:, :, i], axis=0)
    print(f"  dirty " + " ".join(f"{Pd[np.argmin(np.abs(f - v))]:6.1f}" for v in fs_))
