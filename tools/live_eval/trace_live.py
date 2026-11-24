"""Per-epoch trace of the live helmet flow internals for one session (replay through the real Pipeline).
    python tools/live_eval/trace_live.py sessions/<folder> [every_n]"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FS, HERE, load_raw  # noqa: E402
import pipeline as pl  # noqa: E402
from replay_live import FakeBoard  # noqa: E402

d = Path(sys.argv[1])
every = int(sys.argv[2]) if len(sys.argv) > 2 else 1
t, x, labels = load_raw(d)
cfg = json.loads((HERE / "config.json").read_text("utf-8"))
cfg["labels_8"] = labels[:8]
board = FakeBoard(x.shape[1], FS)
p = pl.Pipeline(board, cfg, recorder=None, log=print)
orig = pl.Pipeline._live_flow
k = [0]


def traced(self, filt_long, good):
    r = orig(self, filt_long, good)
    k[0] += 1
    if k[0] % every == 0:
        segs = getattr(self, "_live_segs", None) or []
        ok = np.asarray([e[3] for e in segs]) if segs else np.zeros((0, self.n_ch), bool)
        th = np.asarray([e[1] for e in segs]) if segs else np.zeros((0, self.n_ch))
        tot = np.asarray([e[2] for e in segs]) if segs else np.zeros((0, self.n_ch))
        vals = []
        for i in range(self.n_ch):
            m = ok[:, i] if len(ok) else np.zeros(0, bool)
            vals.append(f"{np.log(th[m, i].sum() / tot[m, i].sum()):5.2f}" if m.sum() >= pl.LIVE_MIN_SEGMENTS else "  .  ")
        print(f"{board.buffer.count / FS:5.0f}s n={len(segs):2d} clean/ch " + " ".join(f"{int(v):2d}" for v in ok.sum(axis=0)) + " | v/ch " + " ".join(vals)
              + f" | base {self.live_base.mean:5.2f}+-{self.live_base.std:4.2f} z {r['live_z']} flow {r['flow_live']} [{r['live_channels']}] {','.join(self.readiness.get('reasons', []))}")
    return r


p._live_flow = traced.__get__(p)
print("channels:", labels)
for i in range(0, len(x) - 25 + 1, 25):
    board.buffer.push(x[i:i + 25].astype(np.float32), np.zeros((25, 3), np.float32), t[i:i + 25])
