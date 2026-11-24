"""The live helmet flow (`flow_live`, pipeline._live_flow): artifact test and theta share on a
4 Hz high-passed copy (slow eye / head movement neither blocks nor drives it), pooling of clean
segments over a horizon, the floored z, and the rule that it never raises inside the epoch loop. Synthetic signals only; the function is driven
directly with an already "filtered" window, as `_epoch` hands it over."""
from __future__ import annotations

import unittest

import numpy as np

import _path  # noqa: F401

import pipeline as pl
from cyton import SampleBuffer
from pipeline import Pipeline

FS = 250.0
LABELS = ["Fp1", "Fp2", "AF3", "AF4", "F3", "F4", "F7", "F8"]
KEYS = {"flow_live", "live_channels", "live_z"}


class FakeBoard:
    def __init__(self, n_ch=8, fs=FS):
        self.n_ch, self.fs = n_ch, fs
        self.buffer = SampleBuffer(n_ch, fs, 30.0)

    def status(self):
        return {"kind": "fake", "n_ch": self.n_ch, "fs": self.fs}


def make_pipeline(labels=LABELS):
    logs = []
    cfg = {"labels_8": labels, "notch_hz": 60, "bp_lo": 1.0, "bp_hi": 50.0, "flow_index": {}}
    p = Pipeline(FakeBoard(), cfg, recorder=None, log=logs.append)
    p.logs = logs
    return p


def eeg(seconds, theta_uv=6.0, seed=0, n_ch=8):
    """Independent white-ish noise per channel (~8 uV) plus a 6 Hz theta tone of `theta_uv`."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = np.arange(n) / FS
    return rng.normal(0, 8.0, (n, n_ch)) + theta_uv * np.sin(2 * np.pi * 6 * t)[:, None]


def add_blinks(x, every_s=4.0, amp=90.0, width_s=0.15):
    """The same slow positive wave on every channel, like a blink on a forehead montage."""
    t = np.arange(len(x)) / FS
    y = x.copy()
    for c in np.arange(every_s / 2, t[-1], every_s):
        y += (amp * np.exp(-0.5 * ((t - c) / width_s) ** 2))[:, None]
    return y


def drive(p, x, good=None, win_s=10.0, step_s=2.0, start=0):
    """Call _live_flow as the epoch loop does: every 2 s with the newest 10 s."""
    win, step = int(win_s * FS), int(step_s * FS)
    out = []
    for end in range(win, len(x) + 1, step):
        p.board.buffer.count = start + end
        out.append(p._live_flow(x[end - win:end], good or [True] * p.n_ch))
    return out


def values(out):
    return [r["flow_live"] for r in out if r["flow_live"] is not None]


class TestLiveFlow(unittest.TestCase):
    def test_steady_signal_gives_a_calm_number_near_half(self):
        p = make_pipeline()
        out = drive(p, eeg(150, seed=3))
        v = values(out)
        self.assertGreater(len(v), 30)
        self.assertTrue(all(set(r) == KEYS for r in out))
        self.assertTrue(all(0.3 < x < 0.7 for x in v), (min(v), max(v)))       # noise is not stretched over 0-1
        self.assertLess(float(np.median(np.abs(np.diff(v)))), 0.02)            # 2-s to 2-s jitter
        self.assertIn("F4", out[-1]["live_channels"])
        self.assertNotIn("Fp2", out[-1]["live_channels"])                      # Fp sites only when nothing else is frontal

    def test_more_theta_raises_the_number(self):
        p = make_pipeline()
        x = np.concatenate([eeg(120, theta_uv=4.0, seed=4), eeg(120, theta_uv=12.0, seed=5)])
        out = drive(p, x)
        v = values(out)
        self.assertLess(abs(v[10] - 0.5), 0.2)
        self.assertGreater(v[-1], 0.7)                         # z vs the whole session, which now holds both halves
        self.assertGreater(out[-1]["live_z"], 1.0)

    def test_blinks_and_slow_drift_neither_block_nor_move_the_number(self):
        base = np.concatenate([eeg(120, seed=6), eeg(120, seed=7)])
        moving = base.copy()
        h = len(base) // 2
        t = np.arange(h) / FS
        moving[h:] = add_blinks(base[h:]) + 250.0 * np.sin(2 * np.pi * 0.4 * t)[:, None]   # blinks + head sway
        a, b = make_pipeline(), make_pipeline()
        va, vb = values(drive(a, base)), values(drive(b, moving))
        self.assertEqual(len(vb), len(va))                                     # still a number every epoch
        self.assertLess(abs(va[-1] - vb[-1]), 0.1)
        self.assertGreater(np.mean([e[3][5] for e in b._live_segs]), 0.9)      # the segments stay usable

    def test_fast_artifacts_are_dropped_per_channel(self):
        x = eeg(150, seed=17)
        rng = np.random.default_rng(18)
        for c in range(20, 150, 8):                            # a muscle burst on F8 every 8 s
            i = int(c * FS)
            x[i:i + 100, 7] += rng.normal(0, 120.0, 100)
        p = make_pipeline()
        v = values(drive(p, x))
        self.assertTrue(v and all(0.3 < q < 0.7 for q in v), (min(v), max(v)))
        f4 = np.mean([e[3][5] for e in p._live_segs])
        f8 = np.mean([e[3][7] for e in p._live_segs])
        self.assertGreater(f4, 0.95)
        self.assertLess(f8, 0.85)

    def test_each_segment_is_analysed_once_and_the_horizon_is_bounded(self):
        p = make_pipeline()
        drive(p, eeg(100, seed=8))
        starts = [e[0] for e in p._live_segs]
        self.assertEqual(len(starts), len(set(starts)))
        self.assertEqual(sorted(starts), starts)
        self.assertTrue(all(b - a == int(FS) for a, b in zip(starts, starts[1:])))      # 2-s segments, 1-s grid
        self.assertLessEqual(starts[-1] - starts[0], pl.LIVE_HORIZON_S * FS)

    def test_one_clean_electrode_alone_is_not_trusted(self):
        p = make_pipeline()
        x = eeg(120, seed=9)
        x[:, [2, 3, 4, 6, 7]] *= 30.0                          # every frontal site but F4 is far over the artifact limits
        out = drive(p, x)
        self.assertEqual(values(out), [])
        self.assertEqual(out[-1]["live_channels"], "no clean signal")

    def test_no_frontal_contact_and_reference_gate(self):
        p = make_pipeline()
        out = drive(p, eeg(30, seed=10), good=[False] * 8)
        self.assertTrue(all(r == {"flow_live": None, "live_channels": None, "live_z": None} for r in out))
        p = make_pipeline()
        p.readiness = {"ready": False, "reasons": ["reference"]}
        self.assertEqual(values(drive(p, eeg(120, seed=11))), [])

    def test_fp_fallback_when_no_other_frontal_site(self):
        p = make_pipeline(["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"])
        out = drive(p, eeg(150, seed=12))
        self.assertTrue(values(out))
        self.assertEqual(out[-1]["live_channels"], "Fp1 Fp2")

    def test_sample_counter_restart_starts_over(self):
        p = make_pipeline()
        drive(p, eeg(60, seed=13), start=1_000_000)
        self.assertGreater(len(p._live_segs), 10)
        drive(p, eeg(14, seed=14), start=0)                    # a new board: the counter is small again
        self.assertTrue(all(e[0] < 14 * FS for e in p._live_segs))

    def test_never_raises(self):
        p = make_pipeline()
        none = {"flow_live": None, "live_channels": None, "live_z": None}
        p.board.buffer.count = 2500
        for bad in (np.zeros((0, 8)), np.zeros((2500, 3)), np.full((2500, 8), np.nan), None, "x"):
            r = p._live_flow(bad, [True] * 8)
            self.assertEqual(set(r), KEYS)
            self.assertIsNone(r["flow_live"])
        self.assertEqual(p._live_flow(eeg(10), []), none)       # `good` of the wrong length
        self.assertLessEqual(sum("live flow error" in s for s in p.logs), 1)    # one log line, not one per epoch
        x = eeg(150, seed=15)
        x[5000:5200] = np.nan                                  # a burst of NaN must not poison the number for good
        v = values(drive(make_pipeline(), x))
        self.assertTrue(v and all(np.isfinite(v)))

    def test_through_the_real_epoch_loop(self):
        """End to end: raw samples in, epochs out, the record carries the three live keys."""
        board = FakeBoard()
        cfg = {"labels_8": LABELS, "notch_hz": 60, "bp_lo": 1.0, "bp_hi": 50.0,
               "flow_index": {"gate_on_readiness": False}}
        p = Pipeline(board, cfg, recorder=None, log=lambda *_: None)
        x = eeg(24, seed=16)
        for i in range(0, len(x) - 24, 25):
            board.buffer.push(x[i:i + 25].astype(np.float32), np.zeros((25, 3), np.float32), 1.8e9 + np.arange(i, i + 25) / FS)
        rec = p.last_analysis["flow_eeg"]
        self.assertTrue(KEYS <= set(rec))
        self.assertGreater(len(p._live_segs), 5)


if __name__ == "__main__":
    unittest.main()
