"""The per-epoch analysis filter in Pipeline._epoch: a clean signal sitting on a large electrode
DC offset (the Cyton reads -100..-200 mV right after put-on) must pass the artifact test from the
FIRST Welch segment on - the filter warm-up may not turn the offset into a transient."""
from __future__ import annotations

import unittest

import numpy as np

import _path  # noqa: F401

from cyton import SampleBuffer
from pipeline import Pipeline

FS = 250.0
LABELS = ["Fp1", "Fp2", "AF3", "AF4", "F3", "F4", "F7", "F8"]


class FakeBoard:
    def __init__(self, n_ch=8, fs=FS):
        self.n_ch, self.fs = n_ch, fs
        self.buffer = SampleBuffer(n_ch, fs, 30.0)

    def status(self):
        return {"kind": "fake", "n_ch": self.n_ch, "fs": self.fs}


def run(offset_uv: float, seconds: float = 12.0, drift_uv_per_s: float = 0.0):
    board = FakeBoard()
    cfg = {"labels_8": LABELS, "notch_hz": 60, "bp_lo": 1.0, "bp_hi": 50.0,
           "flow_index": {"gate_on_readiness": False}}      # "settling" (offset > 100 mV) would withhold it
    p = Pipeline(board, cfg, recorder=None, log=lambda *_: None)
    recs = []
    orig = p.flow.update

    def spy(*a, **k):
        r = orig(*a, **k)
        recs.append(r)
        return r
    p.flow.update = spy
    rng = np.random.default_rng(3)
    n = int(seconds * FS)
    t = np.arange(n) / FS
    x = rng.normal(0, 5.0, (n, 8)) + 6.0 * np.sin(2 * np.pi * 6 * t)[:, None]
    x += offset_uv + drift_uv_per_s * t[:, None] + np.linspace(0, 3000, 8)[None, :]   # per-channel offsets too
    for i in range(0, n - 25 + 1, 25):
        board.buffer.push(x[i:i + 25].astype(np.float32), np.zeros((25, 3), np.float32), 1.8e9 + t[i:i + 25])
    return p, recs


class TestEpochFilter(unittest.TestCase):
    def test_large_dc_offset_is_clean_in_the_first_segment(self):
        p, recs = run(-110000.0)          # > 100 mV ("settling") but below the 75 % near-rail mark
        self.assertTrue(recs)
        self.assertFalse(any(r["artifact"] for r in recs))     # from the very first epoch on
        first = p.last_analysis["flow_eeg"]
        # 10-s window, 2-s segments, 50 % overlap = 9 segments: every one of them clean
        self.assertFalse(first["artifact"])
        self.assertEqual(first["n_segments_used"], 9)
        self.assertEqual(len(first["clean_channels"]), 8)

    def test_slow_drift_on_the_offset_stays_clean(self):
        p, _ = run(-110000.0, drift_uv_per_s=40.0)
        rec = p.last_analysis["flow_eeg"]
        self.assertFalse(rec["artifact"])
        self.assertGreaterEqual(rec["n_segments_used"], 8)

    def test_no_offset_unchanged(self):
        p, _ = run(0.0)
        self.assertEqual(p.last_analysis["flow_eeg"]["n_segments_used"], 9)


if __name__ == "__main__":
    unittest.main()
