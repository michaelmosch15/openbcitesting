"""Offline comparison of candidate estimators for the live helmet flow's raw value
v = log(relative frontal theta), BEFORE baseline / sigmoid / display smoothing.

    python tools/live_eval/candidates.py [--win 10] [sessions/<folder> ...]

Every candidate sees the same thing `_live_flow` sees: the last `win` s of the fixed analysis
filter output (60 Hz notch + 1-45 Hz) at each 2-s tick, on the non-railed frontal channels
(F*/AF*). Metrics (per session, pooled over ticks):
  cov      fraction of ticks with a value
  jit      median |v[k]-v[k-1]| (log units) between successive ticks, steady periods
  noise    jit / (1.48*MAD of v): how much of the spread of the number is tick-to-tick noise
  r_blink  Pearson r of v with the number of blinks inside the window (0 is ideal)
  r_p2p    r of v with log(median-channel p2p in the window)
  r_ch     mean pairwise r between the per-channel values of the usable channels (not AF3)
  r_F4F8   r between the F4 and the F8 value
Nothing here uses felt ratings or the band.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FS, analysis_filter, load_raw, newest_sessions  # noqa: E402
from dsp import psd  # noqa: E402

N2 = int(2 * FS)


# ---------------------------------------------------------------- building blocks
def smooth(x: np.ndarray, k: int) -> np.ndarray:
    """Centred moving average over k samples (edges held)."""
    if k <= 1:
        return x
    return np.convolve(np.pad(x, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, "valid")


def blink_mask(w: np.ndarray, thr: float = 35.0, margin_s: float = 0.35) -> np.ndarray:
    """Samples inside a blink / eye movement: the slow (0.1-s mean) COMMON frontal component
    (median across channels) beyond `thr` uV, widened by `margin_s` on both sides."""
    cm = smooth(np.median(w, axis=1), int(0.1 * FS))
    hit = np.abs(cm) > thr
    k = int(margin_s * FS)
    return np.convolve(hit.astype(float), np.ones(2 * k + 1), "same") > 0


def regress_common(w: np.ndarray, lp_s: float = 0.1) -> np.ndarray:
    """Remove from each channel the least-squares projection of the slow common component."""
    cm = smooth(np.median(w, axis=1), int(lp_s * FS))
    cm = cm - cm.mean()
    den = float(cm @ cm)
    if den <= 0:
        return w
    b = (w - w.mean(axis=0)).T @ cm / den
    return w - np.outer(cm, b)


def estimate(w: np.ndarray, *, seg_s=2.0, rule="p2p", p2p=150.0, absv=100.0, agg="mean", hi=45.0, lo=4.0,
             theta=(4.0, 7.0), min_frac=0.5, regress=False, detrend=False, one_f=False, min_n=0,
             thr=35.0, margin=0.35):
    """-> per-channel v (nan where no value), per-channel clean-segment count, n_total."""
    if regress:
        w = regress_common(w)
    seg = int(seg_s * FS)
    step = seg // 2
    n_ch = w.shape[1]
    bm = blink_mask(w, thr, margin) if rule in ("blink", "blink+p2p") else None
    th, tot, ok, spec = [], [], [], []
    f = None
    for s in range(0, len(w) - seg + 1, step):
        x = w[s:s + seg]
        good = np.ones(n_ch, bool)
        if rule in ("p2p", "blink+p2p"):
            good &= (x.max(axis=0) - x.min(axis=0) <= p2p) & (np.abs(x).max(axis=0) <= absv)
        if bm is not None and bm[s:s + seg].any():
            good[:] = False
        if detrend:
            tt = np.arange(seg) - (seg - 1) / 2
            x = x - x.mean(axis=0)
            x = x - np.outer(tt, (tt @ x) / (tt @ tt))
        f, P = psd(x, FS)
        df = f[1] - f[0]
        th.append(P[(f >= theta[0]) & (f < theta[1])].sum(axis=0) * df)
        tot.append(P[(f >= lo) & (f < hi)].sum(axis=0) * df)
        spec.append(P)
        ok.append(good)
    th, tot, ok = np.asarray(th), np.asarray(tot), np.asarray(ok)
    n_total = len(ok)
    used = ok.sum(axis=0)
    need = max(1, int(np.ceil(min_frac * n_total)), min_n)
    v = np.full(n_ch, np.nan)
    for c in range(n_ch):
        if used[c] < need:
            continue
        m = ok[:, c]
        if one_f:
            S = np.asarray(spec)[m][:, :, c]
            Pm = np.median(S, axis=0) if agg == "median" else S.mean(axis=0)
            fit = ((f >= 2) & (f < 3.5)) | ((f >= 13) & (f <= 40))
            a, b = np.polyfit(np.log(f[fit]), np.log(Pm[fit]), 1)
            tm = (f >= theta[0]) & (f < theta[1])
            v[c] = float(np.mean(np.log(Pm[tm]) - (a * np.log(f[tm]) + b)))
        elif agg == "mean":
            v[c] = np.log(th[m, c].mean() / tot[m, c].mean())
        elif agg == "median":           # median of per-segment band powers, then the ratio
            v[c] = np.log(np.median(th[m, c]) / np.median(tot[m, c]))
        elif agg == "medratio":         # median of per-segment log ratios
            v[c] = np.median(np.log(th[m, c] / tot[m, c]))
    return v, used, n_total


CANDS = {
    "A0 current (2s seg, 150/100, mean, th/4-45)": dict(),
    "A1 no rejection at all":                      dict(rule="none"),
    "B1 th/4-30":                                  dict(hi=30.0),
    "B2 median PSD, th/4-30":                      dict(hi=30.0, agg="median"),
    "B3 median log-ratio, th/4-30":                dict(hi=30.0, agg="medratio"),
    "C1 tighter p2p 100/60":                       dict(hi=30.0, p2p=100.0, absv=60.0, min_frac=0.3),
    "D1 blink mask, 2s seg":                       dict(hi=30.0, rule="blink+p2p", min_frac=0.3),
    "D2 blink mask, 1s seg":                       dict(hi=30.0, rule="blink+p2p", seg_s=1.0, min_frac=0.3),
    "D3 blink mask, 1s seg, detrend":              dict(hi=30.0, rule="blink+p2p", seg_s=1.0, min_frac=0.3, detrend=True),
    "D4 blink mask, 1s, detrend, medratio":        dict(hi=30.0, rule="blink+p2p", seg_s=1.0, min_frac=0.3, detrend=True, agg="medratio"),
    "D5 blink mask, 1s, detrend, th 5-8/5-30":     dict(hi=30.0, lo=5.0, theta=(5.0, 8.0), rule="blink+p2p", seg_s=1.0, min_frac=0.3, detrend=True),
    "E1 regress common slow, 2s 150/100":          dict(hi=30.0, regress=True),
    "E2 regress + no rejection":                   dict(hi=30.0, regress=True, rule="none"),
    "F1 1/f-corrected theta (2s, 150/100)":        dict(one_f=True),
    "F2 1/f-corrected, blink mask 2s":             dict(one_f=True, rule="blink+p2p", min_frac=0.3),
}


def blink_times(xf: np.ndarray) -> np.ndarray:
    """Reference blink events for the evaluation (sample indices): the slow common component > 40 uV."""
    cm = np.abs(smooth(np.median(xf, axis=1), int(0.1 * FS)))
    idx = np.flatnonzero(cm > 40.0)
    if not len(idx):
        return idx
    return idx[np.r_[True, np.diff(idx) > int(0.5 * FS)]]


def r_(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8 or a[m].std() < 1e-12 or b[m].std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def load_session(d: Path):
    t, x, labels = load_raw(d)
    xf_all = analysis_filter(x)
    live = [i for i, lab in enumerate(labels) if np.abs(x[:, i]).mean() < 150000 and not lab.startswith("Fp")]
    return t, xf_all[:, live], [labels[i] for i in live]


def run_session(d: Path, win_s: float, cands=None, verbose=True, data=None):
    cands = CANDS if cands is None else cands
    t, xf, labs = data or load_session(d)
    W = int(win_s * FS)
    ticks = list(range(W, len(xf) + 1, N2))
    bt = blink_times(xf)
    nbl = np.array([((bt >= e - W) & (bt < e)).sum() for e in ticks])
    p2pw = np.array([np.log(np.median(xf[e - W:e].max(axis=0) - xf[e - W:e].min(axis=0))) for e in ticks])
    steady = p2pw < np.log(1000.0)
    if verbose:
        print(f"\n=== {d.name}: {len(xf) / FS:.0f} s, channels {labs}, window {win_s:.0f} s, {len(ticks)} ticks, "
              f"{len(bt)} blinks ({60 * len(bt) / (len(xf) / FS):.0f}/min), steady ticks {steady.mean():.0%}")
        print(f"{'candidate':<44} {'cov':>5} {'jit':>6} {'noise':>6} {'r_blink':>7} {'r_p2p':>6} {'r_ch':>6} {'r_F4F8':>6} {'sd':>5}")
    out = {}
    pairs = [(a, b) for a in range(len(labs)) for b in range(a + 1, len(labs)) if "AF3" not in (labs[a], labs[b])]
    f4, f8 = (labs.index("F4"), labs.index("F8")) if "F4" in labs and "F8" in labs else (None, None)
    for name, kw in cands.items():
        V = np.full((len(ticks), len(labs)), np.nan)
        for k, e in enumerate(ticks):
            V[k], _, _ = estimate(xf[e - W:e], **kw)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                v = np.nanmedian(V, axis=1)
                rch = np.nanmean([r_(V[:, a], V[:, b]) for a, b in pairs]) if pairs else np.nan
        ok = np.isfinite(v)
        dv = np.abs(np.diff(v))
        m = ok[1:] & ok[:-1] & steady[1:] & steady[:-1]
        jit = float(np.median(dv[m])) if m.sum() > 5 else np.nan
        vs = v[ok & steady]
        sd = 1.4826 * float(np.median(np.abs(vs - np.median(vs)))) if len(vs) > 5 else np.nan
        res = dict(cov=ok.mean(), jit=jit, noise=jit / sd if sd else np.nan, r_blink=r_(v, nbl), r_p2p=r_(v, p2pw),
                   r_ch=rch, r_F4F8=r_(V[:, f4], V[:, f8]) if f4 is not None else np.nan, sd=sd, v=v, V=V,
                   t=t[np.array(ticks) - 1], labs=labs, nbl=nbl, steady=steady)
        out[name] = res
        if verbose:
            print(f"{name:<44} {res['cov']:5.2f} {jit:6.3f} {res['noise']:6.2f} {res['r_blink']:7.2f} {res['r_p2p']:6.2f} "
                  f"{rch:6.2f} {res['r_F4F8']:6.2f} {sd:5.2f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--win", type=float, default=10.0)
    a = ap.parse_args()
    for d in ([Path(s) for s in a.sessions] or newest_sessions()):
        run_session(d, a.win)
