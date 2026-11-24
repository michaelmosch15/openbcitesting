"""Before / after of the live helmet flow (`flow_live`), by replaying recorded eeg_raw.csv through
the real Pipeline (same filters, same epoch loop). Nothing is written, nothing is fitted.

    python tools/live_eval/replay_live.py [--set LIVE_HORIZON_S=60 ...] [sessions/<folder> ...]

"before" = the frozen copy of the 14:55 `_live_flow` (old_live_flow.py); "after" = pipeline.py as
it is now. Both run inside the same replay, each with its own baseline. Per session:
  cov       fraction of 2-s epochs with a number
  first_s   seconds from the start to the first number
  jit50/90  median / 90th percentile |change| between successive 2-s numbers (0-1 scale)
  swing60   median (max - min) of the number inside any 60 s
  r_blink   r with the blink count of the last 30 s      (ideal 0)
  r_p2p     r with log p2p of the last 10 s              (ideal 0)
  r_acc     r with the accelerometer SD of the last 30 s (ideal 0)
DESCRIPTIVE ONLY (nothing was tuned to these): mean of the number over the 60 s before each
felt_flow rating, and r with the band's flow (30-s means).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FS, HERE, analysis_filter, load_raw, newest_sessions, read_csv  # noqa: E402
from candidates import blink_times, r_  # noqa: E402
from motion import load_accel  # noqa: E402
from old_live_flow import old_live_flow  # noqa: E402

import pipeline as pl  # noqa: E402
from cyton import SampleBuffer  # noqa: E402
from flow_index import Baseline  # noqa: E402
from collections import deque  # noqa: E402
import json  # noqa: E402


class FakeBoard:
    def __init__(self, n_ch, fs):
        self.n_ch, self.fs = n_ch, float(fs)
        self.buffer = SampleBuffer(n_ch, fs, 30.0)

    def status(self):
        return {"kind": "replay"}


class OldView:
    """The pipeline as the old function saw it, with its own baseline and smoothing state."""

    def __init__(self, p):
        object.__setattr__(self, "_p", p)
        object.__setattr__(self, "live_base", Baseline("rest", warmup=15))
        object.__setattr__(self, "_live_recent", deque(maxlen=5))

    def __getattr__(self, k):
        return getattr(self._p, k)


def replay(d: Path):
    t, x, labels = load_raw(d)
    cfg = json.loads((HERE / "config.json").read_text("utf-8"))
    cfg["labels_8"] = labels[:8]
    board = FakeBoard(x.shape[1], FS)
    p = pl.Pipeline(board, cfg, recorder=None, log=print)
    old = OldView(p)
    if not hasattr(p, "_helmet_flow"):                 # another edit in progress in pipeline.py: keep the epoch loop alive
        p._helmet_flow = lambda *a_, **k_: {}
    rows = []
    new_fn = pl.Pipeline._live_flow

    def both(self, filt_long, good):
        a = new_fn(self, filt_long, good)
        b = old_live_flow(old, filt_long, good)
        rows.append((board.buffer.count, b["flow_live"], a["flow_live"], a["live_channels"], b["live_channels"]))
        return a

    p._live_flow = both.__get__(p)
    chunk = 25
    for i in range(0, len(x) - chunk + 1, chunk):
        board.buffer.push(x[i:i + chunk].astype(np.float32), np.zeros((chunk, 3), np.float32), t[i:i + chunk])
    return t, x, labels, rows


def metrics(f: np.ndarray, idx: np.ndarray, nbl: np.ndarray, p2p: np.ndarray, acc: np.ndarray) -> dict:
    ok = np.isfinite(f)
    d = np.abs(np.diff(f))
    d = d[np.isfinite(d)]
    sw = []
    for k in range(len(f)):
        w = f[(idx > idx[k] - 60 * FS) & (idx <= idx[k])]
        w = w[np.isfinite(w)]
        if len(w) >= 20:
            sw.append(w.max() - w.min())
    return dict(cov=ok.mean(), first_s=idx[ok][0] / FS if ok.any() else np.nan,
                jit50=np.median(d) if len(d) else np.nan, jit90=np.percentile(d, 90) if len(d) else np.nan,
                swing60=np.median(sw) if sw else np.nan, r_blink=r_(f, nbl), r_p2p=r_(f, p2p), r_acc=r_(f, acc),
                lo=np.nanmin(f) if ok.any() else np.nan, hi=np.nanmax(f) if ok.any() else np.nan)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--set", action="append", default=[], help="override a pipeline LIVE_ constant, e.g. LIVE_HORIZON_S=60")
    a = ap.parse_args(argv)
    for kv in a.set:
        k, v = kv.split("=")
        setattr(pl, k, float(v))
        print(f"override {k} = {float(v)}")
    ds = [Path(s) for s in a.sessions] or newest_sessions()
    print(f"{'session':<28} {'version':<7} {'cov':>5} {'first_s':>7} {'jit50':>6} {'jit90':>6} {'swing60':>7} {'r_blink':>7} {'r_p2p':>6} {'r_acc':>6} {'min':>5} {'max':>5}")
    felt_rows, band_rows, pooled = [], [], {"before": [], "after": []}
    for d in ds:
        t, x, labels, rows = replay(d)
        if not rows:
            continue
        idx = np.array([r[0] for r in rows])
        fb = np.array([np.nan if r[1] is None else r[1] for r in rows], float)
        fa = np.array([np.nan if r[2] is None else r[2] for r in rows], float)
        xf = analysis_filter(x)
        live = [i for i, lab in enumerate(labels) if np.abs(x[:, i]).mean() < 150000 and not lab.startswith("Fp")]
        bt = blink_times(xf[:, live])
        nbl = np.array([((bt >= e - 30 * FS) & (bt < e)).sum() for e in idx], float)
        p2p = np.array([np.log(np.median(np.ptp(xf[max(0, e - int(10 * FS)):e, live], axis=0))) for e in idx])
        te = t[np.minimum(idx - 1, len(t) - 1)]
        am = load_accel(d, len(x))
        acc = np.array([am[max(0, e - int(30 * FS)):e].std() for e in idx])
        for name, f in (("before", fb), ("after", fa)):
            m = metrics(f, idx, nbl, p2p, acc)
            pooled[name].append((len(f), m))
            print(f"{d.name[11:]:<28} {name:<7} {m['cov']:5.2f} {m['first_s']:7.0f} {m['jit50']:6.3f} {m['jit90']:6.3f} {m['swing60']:7.2f} "
                  f"{m['r_blink']:7.2f} {m['r_p2p']:6.2f} {m['r_acc']:6.2f} {m['lo']:5.2f} {m['hi']:5.2f}")
        chans = {}
        for r in rows:
            if r[2] is not None:
                chans[r[3]] = chans.get(r[3], 0) + 1
        print(f"{'':<28} after: channels used -> " + ", ".join(f"[{k}] x{v}" for k, v in sorted(chans.items(), key=lambda kv: -kv[1])[:4]))
        # descriptive: felt ratings and the band
        for mk in read_csv(d / "marks.csv"):
            if mk.get("kind") != "felt_flow":
                continue
            try:
                tm, val = float(mk["t"]), float(mk["value"])
            except ValueError:
                continue
            w = (te > tm - 60) & (te <= tm)
            band = [float(b["flow"]) for b in read_csv(d / "band.csv") if b.get("flow") and tm - 60 < float(b["t"]) <= tm]
            felt_rows.append((d.name[11:], val, np.nanmean(fb[w]) if np.isfinite(fb[w]).any() else np.nan,
                              np.nanmean(fa[w]) if np.isfinite(fa[w]).any() else np.nan, np.mean(band) if band else np.nan))
        bd = [(float(b["t"]), float(b["flow"])) for b in read_csv(d / "band.csv") if b.get("flow")]
        if bd:
            bd = np.array(bd)
            for t0 in np.arange(te[0], te[-1] - 30 + 1e-9, 30.0):
                w = (te >= t0) & (te < t0 + 30)
                wb = (bd[:, 0] >= t0) & (bd[:, 0] < t0 + 30)
                if wb.any():
                    band_rows.append((np.nanmean(fb[w]) if np.isfinite(fb[w]).any() else np.nan,
                                      np.nanmean(fa[w]) if np.isfinite(fa[w]).any() else np.nan, bd[wb, 1].mean()))
    for name in ("before", "after"):
        ms = pooled[name]
        if len(ms) > 1:
            avg = lambda k: np.nansum([m[k] * n for n, m in ms if np.isfinite(m[k])]) / max(1, sum(n for n, m in ms if np.isfinite(m[k])))
            print(f"{'ALL (epoch-weighted mean)':<28} {name:<7} {avg('cov'):5.2f} {avg('first_s'):7.0f} {avg('jit50'):6.3f} {avg('jit90'):6.3f} "
                  f"{avg('swing60'):7.2f} {avg('r_blink'):7.2f} {avg('r_p2p'):6.2f} {avg('r_acc'):6.2f}")
    if felt_rows:
        print("\nDESCRIPTIVE (not used for any choice): number averaged over the 60 s before each felt rating")
        print(f"{'session':<28} {'felt':>5} {'before':>7} {'after':>7} {'band':>6}")
        for s, v, b, af, bn in felt_rows:
            print(f"{s:<28} {v:5.2f} {b:7.2f} {af:7.2f} {bn:6.2f}")
        fr = np.array([r[1:] for r in felt_rows], float)
        print(f"n={len(fr)} ratings (range {fr[:, 0].min():.2f}-{fr[:, 0].max():.2f}); Pearson r with felt: before {r_(fr[:, 0], fr[:, 1]):.2f}, "
              f"after {r_(fr[:, 0], fr[:, 2]):.2f}, band {r_(fr[:, 0], fr[:, 3]):.2f}   (r_ needs >= 8 points, else nan)")
    if band_rows:
        br = np.array(band_rows, float)
        print(f"\nDESCRIPTIVE: 30-s means vs the band's flow, n={len(br)} windows: r before {r_(br[:, 2], br[:, 0]):.2f}, after {r_(br[:, 2], br[:, 1]):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
