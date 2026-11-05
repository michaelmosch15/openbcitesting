"""Replay a recorded session's eeg_raw.csv through the live pipeline, offline.

    python tools/replay_eeg.py sessions/<folder> [--scope channel|epoch] [--disable 1,5] [--config config.json]

Feeds the raw samples into a Pipeline exactly as the board thread would (same filters, same
10-s Welch window, same artifact rule, same FlowIndex with the session baseline) and prints,
per minute, how many epochs were clean, which channels carried them, and the index values;
then a summary. Use it to see what a rule change does to a session that already happened,
before changing anything live. Nothing is written.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from cyton import SampleBuffer  # noqa: E402
from pipeline import Pipeline  # noqa: E402

sys.path.insert(0, str(HERE / "tools"))
from session_diagnose import load_eeg  # noqa: E402


class FakeBoard:
    def __init__(self, n_ch: int, fs: float):
        self.n_ch, self.fs = n_ch, float(fs)
        self.buffer = SampleBuffer(n_ch, fs, 30.0)
        self.streaming = True

    def status(self):
        return {"kind": "replay", "port": "file", "n_ch": self.n_ch, "fs": self.fs, "streaming": True}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--scope", choices=["channel", "epoch"], default=None)
    ap.add_argument("--disable", default="", help="1-based channels to disable, e.g. 1,5")
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--fs", type=float, default=250.0)
    args = ap.parse_args(argv)
    d = Path(args.session)
    t, x, labels, _ = load_eeg(d / "eeg_raw.csv", None)
    fs = args.fs
    cfg = json.loads(Path(args.config).read_text("utf-8"))
    cfg = dict(cfg)
    cfg["labels_8"] = labels[:8]
    cfg["flow_index"] = dict(cfg.get("flow_index", {}))
    if args.scope:
        cfg["flow_index"]["artifact_scope"] = args.scope
    board = FakeBoard(x.shape[1], fs)
    p = Pipeline(board, cfg, recorder=None, log=lambda *_: None)
    for s in args.disable.split(","):
        if s.strip():
            p.set_channel(int(s) - 1, False)
    print(f"{d.name}: {len(x) / fs / 60:.1f} min, scope={p.artifact_scope}, min_clean_channels={p.min_clean_channels}, "
          f"labels={labels}, disabled={[labels[i] for i in range(p.n_ch) if not p.enabled[i]]}")
    recs = []
    chunk = 25
    t0 = t[0]
    for i in range(0, len(x) - chunk + 1, chunk):
        tt = t0 + np.arange(i, i + chunk) / fs
        board.buffer.push(x[i:i + chunk].astype(np.float32), np.zeros((chunk, 3), np.float32), tt)
        a = p.last_analysis
        if a is not None and (not recs or a["t"] != recs[-1]["t"]):
            recs.append({"t": a["t"], "flow": a["flow_eeg"]["flow_eeg"], "artifact": a["flow_eeg"]["artifact"],
                         "clean": a["flow_eeg"].get("clean_channels", []), "reasons": a["readiness"]["reasons"],
                         "corr": a["readiness"].get("common_mode_corr"), "gated": a["flow_eeg"].get("gated")})
    if not recs:
        print("no epochs")
        return 1
    import datetime as _dt
    print(f"\n{'clock':>5} {'epochs':>6} {'clean':>5} {'gated':>5} {'index mean':>10} {'index sd':>8}  clean channels (most used)   readiness reasons")
    per_min = {}
    for r in recs:
        per_min.setdefault(int((r["t"] - recs[0]["t"]) // 60), []).append(r)
    for m, rs in sorted(per_min.items()):
        clean = [r for r in rs if not r["artifact"] and r["flow"] is not None]
        gated = [r for r in rs if r.get("gated")]
        vals = [r["flow"] for r in clean]
        ch = Counter(c for r in clean for c in r["clean"]).most_common(4)
        reasons = Counter(x for r in rs for x in r["reasons"]).most_common(3)
        clock = _dt.datetime.fromtimestamp(rs[0]["t"]).strftime("%H:%M")
        print(f"{clock:>5} {len(rs):>6} {len(clean):>5} {len(gated):>5} {np.mean(vals) if vals else float('nan'):>10.3f} {np.std(vals) if vals else float('nan'):>8.3f}  "
              f"{' '.join(f'{c}:{n}' for c, n in ch):<28} {' '.join(f'{k}:{n}' for k, n in reasons)}")
    clean = [r for r in recs if not r["artifact"] and r["flow"] is not None]
    print(f"\nTOTAL: {len(recs)} epochs, {len(clean)} clean ({100 * len(clean) / len(recs):.0f}%), baseline {p.flow.baseline_state}, "
          f"index mean {np.mean([r['flow'] for r in clean]) if clean else float('nan'):.3f}")
    corr = [r["corr"] for r in recs if r["corr"] is not None]
    if corr:
        print(f"common-mode |corr| among good channels: mean {np.mean(corr):.2f} (>= 0.9 = reference clip problem)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
