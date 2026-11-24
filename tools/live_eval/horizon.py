"""Stage 2 of the offline evaluation: blink-masked short segments pooled over a longer horizon.

    python tools/live_eval/horizon.py [sessions/<folder> ...]

Per session the filtered signal (60 Hz notch + 1-45 Hz) is cut into Hann segments (50 % overlap);
a segment is clean for a channel when no sample is inside the blink mask (slow common frontal
component) and the channel passes the p2p / abs rule. At each 2-s tick the per-channel value
is log(theta / total) pooled over the clean segments of the last H seconds; the helmet value is
the median over the channels with enough clean segments. Metrics:
  cov      fraction of ticks with a value
  jit      median |v[k]-v[k-1]| between successive 2-s ticks (log units)
  sd       SD of v over the session (signal + noise)
  noise    estimator noise SD from a split-half: v from odd vs even clean segments, SD(diff)/2
  rel      1 - noise^2 / sd^2  (share of the spread that is not estimator noise; <= 0 = all noise)
  r_blink  r of v with the blink count in the horizon (ideal 0)
  r_slow   r of v with log 1-3.5 Hz power of the same clean segments (leakage; ideal 0)
  r_ch     mean pairwise r of per-channel values (AF4 F4 F7 F8), at non-overlapping windows
Nothing here uses felt ratings or the band.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FS, newest_sessions  # noqa: E402
from candidates import blink_mask, blink_times, load_session, r_  # noqa: E402
from dsp import psd  # noqa: E402

N2 = int(2 * FS)


def seg_stats(xf, seg_s, theta, total, rule="blink", thr=35.0, margin=0.35, p2p=150.0, absv=100.0):
    seg = int(seg_s * FS)
    step = seg // 2
    bm = blink_mask(xf, thr, margin) if rule == "blink" else np.zeros(len(xf), bool)
    ends, th, tot, slow, ok = [], [], [], [], []
    for s in range(0, len(xf) - seg + 1, step):
        x = xf[s:s + seg]
        f, P = psd(x, FS)
        df = f[1] - f[0]
        th.append(P[(f >= theta[0]) & (f < theta[1])].sum(axis=0) * df)
        tot.append(P[(f >= total[0]) & (f < total[1])].sum(axis=0) * df)
        slow.append(P[(f >= 1) & (f < 3.5)].sum(axis=0) * df)
        g = (x.max(axis=0) - x.min(axis=0) <= p2p) & (np.abs(x).max(axis=0) <= absv)
        if bm[s:s + seg].any():
            g[:] = False
        ok.append(g)
        ends.append(s + seg)
    return np.asarray(ends), np.asarray(th), np.asarray(tot), np.asarray(slow), np.asarray(ok)


def pooled(th, tot, m, agg):
    if agg == "ratio":
        return float(np.log(th[m].sum() / tot[m].sum()))
    return float(np.median(np.log(th[m] / tot[m])))


def evaluate(xf, labs, *, seg_s=1.0, H=30.0, theta=(4.0, 7.0), total=(4.0, 30.0), rule="blink", agg="ratio",
             min_clean_s=6.0, min_ch=1, thr=35.0, margin=0.35, p2p=150.0, absv=100.0):
    ends, th, tot, slow, ok = seg_stats(xf, seg_s, theta, total, rule, thr, margin, p2p, absv)
    need = max(2, int(np.ceil(min_clean_s / (seg_s / 2))) - 1)      # overlapping segments covering min_clean_s
    ticks = np.arange(int(max(H, 10) * FS), len(xf) + 1, N2)
    bt = blink_times(xf)
    n_ch = len(labs)
    V = np.full((len(ticks), n_ch), np.nan)
    Vo, Ve, SL = V.copy(), V.copy(), V.copy()
    nbl = np.zeros(len(ticks))
    for k, e in enumerate(ticks):
        w = (ends <= e) & (ends > e - int(H * FS))
        nbl[k] = ((bt >= e - int(H * FS)) & (bt < e)).sum()
        idx = np.flatnonzero(w)
        for c in range(n_ch):
            m = idx[ok[idx, c]]
            if len(m) < need:
                continue
            V[k, c] = pooled(th[:, c], tot[:, c], m, agg)
            SL[k, c] = np.log(slow[m, c].mean())
            # split-half on pairs of adjacent segments (overlapping neighbours stay together)
            grp = (m // 2) % 2
            if (grp == 0).sum() >= 2 and (grp == 1).sum() >= 2:
                Vo[k, c] = pooled(th[:, c], tot[:, c], m[grp == 0], agg)
                Ve[k, c] = pooled(th[:, c], tot[:, c], m[grp == 1], agg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nv = np.isfinite(V).sum(axis=1)
        v = np.where(nv >= min_ch, np.nanmedian(V, axis=1), np.nan)
        vo, ve, sl = np.nanmedian(Vo, axis=1), np.nanmedian(Ve, axis=1), np.nanmedian(SL, axis=1)
    ok_t = np.isfinite(v)
    dv = np.abs(np.diff(v))
    m2 = ok_t[1:] & ok_t[:-1]
    jit = float(np.median(dv[m2])) if m2.sum() > 5 else np.nan
    sd = float(np.std(v[ok_t])) if ok_t.sum() > 5 else np.nan
    d = (vo - ve)[np.isfinite(vo - ve)]
    noise = float(np.sqrt(np.mean(d ** 2)) / 2) if len(d) > 5 else np.nan
    stride = max(1, int(H / 2))                                   # non-overlapping windows
    use = [c for c, l in enumerate(labs) if l != "AF3"]
    pairs = [(a, b) for i, a in enumerate(use) for b in use[i + 1:]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rch = float(np.nanmean([r_(V[::stride, a], V[::stride, b]) for a, b in pairs])) if pairs else np.nan
    return dict(cov=float(ok_t.mean()), jit=jit, sd=sd, noise=noise,
                rel=1 - noise ** 2 / sd ** 2 if sd and np.isfinite(noise) else np.nan,
                r_blink=r_(v, nbl), r_slow=r_(v, sl), r_ch=rch, v=v, V=V, ticks=ticks, n_ch=np.nanmedian(nv))


GRID = {
    "2s seg p2p-only  H10 th4-7/4-45 (= current rule)": dict(seg_s=2.0, H=10, rule="p2p", total=(4.0, 45.0), min_clean_s=5.0),
    "2s seg p2p-only  H30 th4-7/4-30":                  dict(seg_s=2.0, H=30, rule="p2p", min_clean_s=6.0),
    "2s seg blinkmask H10 th4-7/4-30":                  dict(seg_s=2.0, H=10, min_clean_s=4.0),
    "2s seg blinkmask H30 th4-7/4-30":                  dict(seg_s=2.0, H=30),
    "1s seg blinkmask H10 th4-7/4-30":                  dict(seg_s=1.0, H=10, min_clean_s=4.0),
    "1s seg blinkmask H20 th4-7/4-30":                  dict(seg_s=1.0, H=20),
    "1s seg blinkmask H30 th4-7/4-30":                  dict(seg_s=1.0, H=30),
    "1s seg blinkmask H30 th4-8/4-30":                  dict(seg_s=1.0, H=30, theta=(4.0, 8.0)),
    "1s seg blinkmask H30 th5-8/5-30":                  dict(seg_s=1.0, H=30, theta=(5.0, 8.0), total=(5.0, 30.0)),
    "1s seg blinkmask H30 th4-7/4-30 medratio":         dict(seg_s=1.0, H=30, agg="median"),
    "1s seg blinkmask H30 th4-7/4-45":                  dict(seg_s=1.0, H=30, total=(4.0, 45.0)),
    "1s seg blinkmask H30 th4-7/4-30 thr25":            dict(seg_s=1.0, H=30, thr=25.0),
    "1s seg blinkmask H30 th4-7/4-30 thr50":            dict(seg_s=1.0, H=30, thr=50.0),
    "1s seg blinkmask H30 th4-7/4-30 margin0.6":        dict(seg_s=1.0, H=30, margin=0.6),
    "1s seg blinkmask H30 th4-7/4-30 p2p100/60":        dict(seg_s=1.0, H=30, p2p=100.0, absv=60.0),
    "1s seg blinkmask H30 th4-7/4-30 >=2 ch":           dict(seg_s=1.0, H=30, min_ch=2),
    "1s seg blinkmask H60 th4-7/4-30":                  dict(seg_s=1.0, H=60),
    "2s seg blinkmask H60 th4-7/4-30":                  dict(seg_s=2.0, H=60),
}


def main(argv):
    ds = [Path(s) for s in argv] or newest_sessions()
    data = [(d, load_session(d)) for d in ds]
    for d, (t, xf, labs) in data:
        print(f"{d.name}: {len(xf) / FS:.0f} s {labs}")
    print(f"\n{'config (pooled over sessions, n-weighted)':<50} {'cov':>5} {'jit':>6} {'sd':>6} {'noise':>6} {'rel':>6} {'r_blink':>7} {'r_slow':>6} {'r_ch':>6}")
    for name, kw in GRID.items():
        rs = [evaluate(xf, labs, **kw) for _, (t, xf, labs) in data]
        w = np.array([len(r["ticks"]) for r in rs], float)
        avg = lambda k: float(np.nansum([r[k] * wi for r, wi in zip(rs, w) if np.isfinite(r[k])])
                              / max(1e-9, sum(wi for r, wi in zip(rs, w) if np.isfinite(r[k]))))
        print(f"{name:<50} {avg('cov'):5.2f} {avg('jit'):6.3f} {avg('sd'):6.3f} {avg('noise'):6.3f} {avg('rel'):6.2f} "
              f"{avg('r_blink'):7.2f} {avg('r_slow'):6.2f} {avg('r_ch'):6.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
