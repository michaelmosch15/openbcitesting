"""Stage 3: a moving wearer (Michael, 2026-11-19 15:03+). Slow eye / motion drift fails the
1-45 Hz artifact rule and trips the 1-45 Hz blink mask almost all the time. Candidates here
test and estimate on a 3 Hz high-passed copy instead, with different rejection rules.

    python tools/live_eval/motion.py [sessions/<folder> ...]

Pooled over a 60-s horizon, >= 10 clean 2-s segments, median over channels (>= 2). Metrics:
  cov      fraction of 2-s ticks with a value
  jit      median |v[k]-v[k-1]| (log units)
  sd/noise SD of v over the session / split-half estimator noise
  r_blink  r of v with the number of blinks (slow common wave > 40 uV) in the horizon
  r_p2p    r of v with log median 1-45 Hz p2p of the horizon's segments
  r_acc    r of v with the accelerometer's SD over the horizon (head movement)
  clean    mean fraction of segments kept per usable channel
Nothing here uses felt ratings or the band.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FS, analysis_filter, load_raw, newest_sessions  # noqa: E402
from candidates import blink_mask, blink_times, r_, smooth  # noqa: E402
from dsp import FilterChain, psd  # noqa: E402

N2 = int(2 * FS)


def load_accel(d: Path, n: int) -> np.ndarray:
    """|accel| per sample (g), zeros (no accel in that packet) forward-filled."""
    import csv
    out = np.zeros(n)
    with open(d / "eeg_raw.csv", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        h = next(r)
        ix = [h.index(c) for c in ("acc_x", "acc_y", "acc_z")]
        last = 0.0
        for k, row in enumerate(r):
            if k >= n:
                break
            try:
                a = [float(row[i]) for i in ix]
            except (ValueError, IndexError):
                continue
            if any(a):
                last = float(np.sqrt(sum(v * v for v in a)))
            out[k] = last
    return out


def highpass(x: np.ndarray, hz: float) -> np.ndarray:
    ch = FilterChain(FS, x.shape[1], 0, hz, 0)
    n = int(FS)
    return ch.process(np.concatenate([x[::-1][:n], x], axis=0))[n:]


def evaluate(xf, acc, labs, *, hp=0.0, mask="slow", thr=35.0, margin=0.35, p2p=150.0, absv=100.0, p2p_on="hp",
             theta=(4.0, 7.0), total=(4.0, 30.0), H=60.0, need=10, acc_max=None, regress=None):
    use = [i for i, l in enumerate(labs) if not l.startswith("Fp")]
    xh = highpass(xf, hp) if hp else xf
    if regress is not None:                       # remove the Fp (EOG) slow component from every channel
        fp = [i for i, l in enumerate(labs) if l.startswith("Fp")]
        if fp:
            ref = smooth(np.median(xf[:, fp], axis=1), int(0.1 * FS))
            ref = ref - ref.mean()
            xh = xh - np.outer(ref, (xh.T @ ref) / (ref @ ref))
    if mask == "slow":
        bm = blink_mask(xf, thr, margin)
    elif mask == "hp":                            # common component of the high-passed signal
        bm = blink_mask(xh, thr, margin)
    else:
        bm = np.zeros(len(xf), bool)
    seg, step = N2, N2 // 2
    ends, th, tot, ok, pp, am = [], [], [], [], [], []
    for s in range(0, len(xf) - seg + 1, step):
        x = xh[s:s + seg]
        f, P = psd(x, FS)
        df = f[1] - f[0]
        th.append(P[(f >= theta[0]) & (f < theta[1])].sum(axis=0) * df)
        tot.append(P[(f >= total[0]) & (f < total[1])].sum(axis=0) * df)
        y = x if p2p_on == "hp" else xf[s:s + seg]
        g = (np.ptp(y, axis=0) <= p2p) & (np.abs(y).max(axis=0) <= absv)
        if bm[s:s + seg].any():
            g[:] = False
        a = acc[s:s + seg]
        am.append(a.std())
        if acc_max is not None and a.std() > acc_max:
            g[:] = False
        ok.append(g)
        pp.append(np.median(np.ptp(xf[s:s + seg][:, use], axis=0)))
        ends.append(s + seg)
    ends, th, tot, ok, pp, am = map(np.asarray, (ends, th, tot, ok, pp, am))
    ticks = np.arange(int(H * FS), len(xf) + 1, N2)
    bt = blink_times(xf[:, use])
    V = np.full((len(ticks), len(labs)), np.nan)
    Vo, Ve = V.copy(), V.copy()
    nbl, lp, la = np.zeros(len(ticks)), np.zeros(len(ticks)), np.zeros(len(ticks))
    for k, e in enumerate(ticks):
        idx = np.flatnonzero((ends <= e) & (ends > e - int(H * FS)))
        nbl[k] = ((bt >= e - int(H * FS)) & (bt < e)).sum()
        lp[k] = np.log(np.median(pp[idx]))
        la[k] = acc[e - int(H * FS):e].std()
        for c in use:
            m = idx[ok[idx, c]]
            if len(m) < need:
                continue
            V[k, c] = np.log(th[m, c].sum() / tot[m, c].sum())
            g = (m // 2) % 2
            if (g == 0).sum() >= 2 and (g == 1).sum() >= 2:
                Vo[k, c] = np.log(th[m[g == 0], c].sum() / tot[m[g == 0], c].sum())
                Ve[k, c] = np.log(th[m[g == 1], c].sum() / tot[m[g == 1], c].sum())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = np.where(np.isfinite(V).sum(axis=1) >= 2, np.nanmedian(V, axis=1), np.nan)
        d = np.nanmedian(Vo, axis=1) - np.nanmedian(Ve, axis=1)
    okt = np.isfinite(v)
    dv = np.abs(np.diff(v))
    dv = dv[np.isfinite(dv)]
    d = d[np.isfinite(d)]
    sd = float(np.std(v[okt])) if okt.sum() > 5 else np.nan
    return dict(cov=float(okt.mean()), jit=float(np.median(dv)) if len(dv) else np.nan, sd=sd,
                noise=float(np.sqrt(np.mean(d ** 2)) / 2) if len(d) > 5 else np.nan,
                r_blink=r_(v, nbl), r_p2p=r_(v, lp), r_acc=r_(v, la), clean=float(ok[:, use].mean()), n=len(ticks))


GRID = {
    "S0 shipped: 1-45, slow mask 35, p2p 150/100":      dict(hp=0.0, p2p_on="raw"),
    "S1 1-45, no mask, p2p 150/100 (old rule)":         dict(hp=0.0, mask="none", p2p_on="raw"),
    "H1 hp3, no mask, hp p2p 150/100":                  dict(hp=3.0, mask="none"),
    "H2 hp3, no mask, hp p2p 100/60":                   dict(hp=3.0, mask="none", p2p=100.0, absv=60.0),
    "H3 hp3, no mask, hp p2p 70/40":                    dict(hp=3.0, mask="none", p2p=70.0, absv=40.0),
    "H4 hp3, hp-common mask 15, hp p2p 100/60":         dict(hp=3.0, mask="hp", thr=15.0, p2p=100.0, absv=60.0),
    "H5 hp3, slow mask 75, hp p2p 100/60":              dict(hp=3.0, mask="slow", thr=75.0, p2p=100.0, absv=60.0),
    "H6 hp3, slow mask 35, hp p2p 100/60":              dict(hp=3.0, mask="slow", thr=35.0, p2p=100.0, absv=60.0),
    "H7 hp3, no mask, hp 100/60, accel sd<0.05 g":      dict(hp=3.0, mask="none", p2p=100.0, absv=60.0, acc_max=0.05),
    "H8 hp3, Fp regress, hp p2p 100/60":                dict(hp=3.0, mask="none", p2p=100.0, absv=60.0, regress=True),
    "H9 hp4, no mask, hp p2p 100/60, th 5-8/5-30":      dict(hp=4.0, mask="none", p2p=100.0, absv=60.0, theta=(5.0, 8.0), total=(5.0, 30.0)),
}


def main(argv):
    ds = [Path(s) for s in argv] or newest_sessions()
    for d in ds:
        t, x, labels = load_raw(d)
        live = [i for i in range(len(labels)) if (np.abs(x[:, i]) >= 0.9 * 187500).mean() < 0.5]
        xf = analysis_filter(x)[:, live]
        labs = [labels[i] for i in live]
        acc = load_accel(d, len(x))
        print(f"\n=== {d.name}: {len(x) / FS:.0f} s, {labs}, accel SD {acc.std():.3f} g")
        print(f"{'candidate':<46} {'cov':>5} {'clean':>5} {'jit':>6} {'sd':>6} {'noise':>6} {'r_blink':>7} {'r_p2p':>6} {'r_acc':>6}")
        for name, kw in GRID.items():
            r = evaluate(xf, acc, labs, **kw)
            print(f"{name:<46} {r['cov']:5.2f} {r['clean']:5.2f} {r['jit']:6.3f} {r['sd']:6.3f} {r['noise']:6.3f} "
                  f"{r['r_blink']:7.2f} {r['r_p2p']:6.2f} {r['r_acc']:6.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
