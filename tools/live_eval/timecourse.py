"""Per-10-s time course of contact: raw 60 Hz amplitude (uV rms), filtered 1-45 Hz p2p, residual
hum after the analysis filter, per channel.   python tools/live_eval/timecourse.py [session]"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FS, analysis_filter, load_raw, newest_sessions  # noqa: E402
from dsp import psd  # noqa: E402

d = Path(sys.argv[1]) if len(sys.argv) > 1 else newest_sessions()[-1]
t, x, labels = load_raw(d)
xf = analysis_filter(x)
W = int(10 * FS)
live = [i for i in range(len(labels)) if np.abs(x[:, i]).mean() < 150000]
print(d.name, "channels", [labels[i] for i in live])
print("t_s | raw 60Hz rms uV | hum share | filtered p2p uV (10 s) | filtered rms 20-45 Hz")
for s in range(0, len(x) - W + 1, W):
    f, P = psd(x[s:s + W] - x[s:s + W].mean(axis=0), FS)
    df = f[1] - f[0]
    hum = np.sqrt(P[(f >= 58) & (f < 62)].sum(axis=0) * df)
    share = P[(f >= 58) & (f < 62)].sum(axis=0) / P[(f >= 1) & (f < 100)].sum(axis=0)
    seg = xf[s:s + W]
    f2, P2 = psd(seg, FS)
    hf = np.sqrt(P2[(f2 >= 20) & (f2 < 45)].sum(axis=0) * df)
    p2p = seg.max(axis=0) - seg.min(axis=0)
    print(f"{s / FS:4.0f} | " + " ".join(f"{hum[i]:6.0f}" for i in live) + " | " + " ".join(f"{share[i]:.2f}" for i in live)
          + " | " + " ".join(f"{p2p[i]:5.0f}" for i in live) + " | " + " ".join(f"{hf[i]:5.1f}" for i in live))
