"""The v0.3 index guards learned from the first worn session (2026-11-05): per-channel
artifact scope, the readiness verdict, the gate that withholds the index while the channels
are one shared signal (no reference clip / headset on the desk), and the band's "no reading"
semantics for a cold-start 0.0 (firmware_v2 rows and Supabase rows)."""
from __future__ import annotations

import json
import struct
import unittest

import numpy as np

import _path  # noqa: F401

import band_link
import supabase_link
from cyton import SampleBuffer
from dsp import welch_masked
from pipeline import Pipeline

FS = 250.0


class FakeBoard:
    def __init__(self, n_ch=8, fs=FS):
        self.n_ch, self.fs = n_ch, fs
        self.buffer = SampleBuffer(n_ch, fs, 30.0)

    def status(self):
        return {"kind": "fake", "n_ch": self.n_ch, "fs": self.fs}


def make_pipeline(**fi):
    cfg = {"labels_8": ["Fp1", "Fp2", "AF3", "AF4", "F3", "F4", "F7", "F8"], "notch_hz": 60, "bp_lo": 1.0, "bp_hi": 50.0,
           "flow_index": {"baseline_mode": "session", "warmup_epochs": 2, **fi}}
    board = FakeBoard()
    p = Pipeline(board, cfg, recorder=None, log=lambda *_: None)
    return board, p


def feed(board, x, t0=1_800_000_000.0, chunk=25):
    for i in range(0, len(x) - chunk + 1, chunk):
        tt = t0 + np.arange(i, i + chunk) / FS
        board.buffer.push(x[i:i + chunk].astype(np.float32), np.zeros((chunk, 3), np.float32), tt)


def eeg_like(n, n_ch=8, seed=0, amp=15.0):
    """Independent band-limited noise per channel, +- a few tens of uV, plus a 6 Hz theta tone."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    x = rng.normal(0, amp, (n, n_ch))
    x = np.cumsum(x, axis=0); x -= np.cumsum(x, axis=0) / np.arange(1, n + 1)[:, None]   # crude 1/f-ish
    x = x / (x.std(axis=0) + 1e-9) * amp
    x += 8.0 * np.sin(2 * np.pi * 6 * t)[:, None]
    return x


class TestWelchMasked(unittest.TestCase):
    def test_per_channel_rejection(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 5, (int(10 * FS), 3))
        x[int(3 * FS):int(5 * FS), 1] += 500.0          # channel 1 is bad for 2 s only
        f, P, used, total = welch_masked(x, FS, 2.0, 0.5, lambda s: (s.max(axis=0) - s.min(axis=0)) > 150)
        self.assertEqual(total, 9)
        self.assertEqual(used[0], 9); self.assertEqual(used[2], 9)
        self.assertLess(used[1], 9); self.assertGreater(used[1], 0)
        self.assertFalse(np.isnan(P).any())
        f2, P2, used2, _ = welch_masked(x, FS, 2.0, 0.5, lambda s: np.ones(3, bool))
        self.assertTrue(np.isnan(P2).all()); self.assertEqual(list(used2), [0, 0, 0])


class TestArtifactScopeAndGate(unittest.TestCase):
    def test_clean_independent_eeg_is_scored(self):
        board, p = make_pipeline()
        feed(board, eeg_like(int(16 * FS)))
        a = p.last_analysis
        self.assertIsNotNone(a)
        fe = a["flow_eeg"]
        self.assertFalse(fe["artifact"]); self.assertIsNone(fe["gated"])
        self.assertEqual(len(fe["clean_channels"]), 8)
        self.assertTrue(a["readiness"]["ready"], a["readiness"])
        self.assertLess(a["readiness"]["common_mode_corr"], 0.9)

    def test_one_noisy_channel_keeps_the_others_in_channel_scope(self):
        x = eeg_like(int(16 * FS), seed=2)
        x[:, 6] += np.random.default_rng(3).normal(0, 400, len(x))     # F7 is junk the whole time
        board, p = make_pipeline(artifact_scope="channel")
        feed(board, x)
        fe = p.last_analysis["flow_eeg"]
        self.assertFalse(fe["artifact"])
        self.assertNotIn("F7", fe["clean_channels"]); self.assertEqual(len(fe["clean_channels"]), 7)
        board2, p2 = make_pipeline(artifact_scope="epoch")
        feed(board2, x)
        self.assertTrue(p2.last_analysis["flow_eeg"]["artifact"])     # the v0.2 rule drops every segment

    def test_shared_signal_is_gated_as_reference(self):
        rng = np.random.default_rng(4)
        one = np.cumsum(rng.normal(0, 1, int(16 * FS))); one = (one - one.mean()) / one.std() * 20.0
        x = np.repeat(one[:, None], 8, axis=1) + rng.normal(0, 0.3, (len(one), 8))   # the same signal on all eight
        board, p = make_pipeline()
        feed(board, x)
        a = p.last_analysis
        self.assertIn("reference", a["readiness"]["reasons"])
        self.assertGreaterEqual(a["readiness"]["common_mode_corr"], 0.9)
        self.assertTrue(a["flow_eeg"]["artifact"]); self.assertEqual(a["flow_eeg"]["gated"], "reference")
        self.assertIsNone(a["flow_eeg"]["flow_eeg"])
        self.assertEqual(len(p.eeg_hist), 0)                           # nothing enters the pairs or the baseline
        board2, p2 = make_pipeline(gate_on_readiness=False)
        feed(board2, x)
        self.assertIsNone(p2.last_analysis["flow_eeg"]["gated"])

    def test_too_few_clean_channels_is_gated(self):
        x = eeg_like(int(16 * FS), seed=5)
        x[:, 1:] += np.random.default_rng(6).normal(0, 400, (len(x), 7))   # only Fp1 is clean
        board, p = make_pipeline()
        feed(board, x)
        a = p.last_analysis
        self.assertIn("few", a["readiness"]["reasons"]); self.assertTrue(a["flow_eeg"]["artifact"])
        self.assertEqual(a["readiness"]["n_clean"], 1)

    def test_flow_status_reports_scope_and_readiness(self):
        board, p = make_pipeline()
        st = p.flow_status()
        self.assertEqual(st["artifact_scope"], "channel"); self.assertEqual(st["min_clean_channels"], 2)
        self.assertIn("readiness", st)


class TestBandNoReading(unittest.TestCase):
    def test_v2_cold_start_zero_is_no_reading(self):
        row = struct.pack("<IhhhhhhHHHHBBBBHBB", 1495, 666, 0, 0, 0, 166, 440, 4219, 0, 0, 3572, 0, 1, 0x00, 0, 6250, 16, 0)
        s = band_link.decode_v2_row(row)
        self.assertIsNone(s["flow"]); self.assertEqual(s["flow_raw"], 0.0)
        self.assertIn("HRV", s["no_reading"]); self.assertFalse(s["hrv_ok"])
        row2 = struct.pack("<IhhhhhhHHHHBBBBHBB", 1500, 700, 970, 500, 400, 150, 300, 1000, 0, 4200, 3572, 0, 1, 0x01, 0, 0, 0, 0)
        s2 = band_link.decode_v2_row(row2)
        self.assertEqual(s2["flow"], 0.42); self.assertIsNone(s2["no_reading"])
        row3 = struct.pack("<IhhhhhhHHHHBBBBHBB", 1500, 700, 970, 500, 400, 150, 300, 1000, 0, 0, 3572, 0, 1, 0x01, 0, 0, 0, 0)
        s3 = band_link.decode_v2_row(row3)                               # HRV valid and a real 0.0: a reading
        self.assertEqual(s3["flow"], 0.0); self.assertIsNone(s3["no_reading"])

    def test_supabase_cold_start_zero_is_no_reading(self):
        base = {"sampled_at": "2026-11-05T20:40:14.270493+00:00", "flow_score": 0.0, "flow_state": 1, "anomaly": 0.5,
                "risk": 0.0, "mode": 0, "phase": 1}
        s = supabase_link.row_to_sample({**base, "hrv_valid": False})
        self.assertIsNone(s["flow"]); self.assertIn("HRV", s["no_reading"])
        s2 = supabase_link.row_to_sample({**base, "hrv_valid": True})
        self.assertEqual(s2["flow"], 0.0); self.assertIsNone(s2["no_reading"])
        s3 = supabase_link.row_to_sample({**base, "flow_score": 0.6, "hrv_valid": False})
        self.assertEqual(s3["flow"], 0.6)

    def test_pipeline_skips_unusable_rows(self):
        board, p = make_pipeline()
        p.eeg_hist.append((100.0, 0.5))
        p.on_band({"t": 110.0, "flow": 0.4, "usable": False})
        self.assertEqual(len(p.pairs), 0)
        p.on_band({"t": 120.0, "flow": 0.4, "usable": True})
        self.assertEqual(len(p.pairs), 1)
        self.assertEqual(p.band_view()["usable"], True)


if __name__ == "__main__":
    unittest.main()
