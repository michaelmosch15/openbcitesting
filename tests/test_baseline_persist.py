"""The frozen rest baseline survives a restart: saved at the end of the protocol's rest block,
restored on --resume (same labels, frozen only), and `rest_block_needed` says when there is none."""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

import _path  # noqa: F401
from pipeline import Pipeline


class FakeBuffer:
    def __init__(self, n_ch):
        self.listeners = []
        self.n_ch = n_ch

    def last(self, n):
        return np.zeros((n, self.n_ch)), np.arange(n) / 250.0


class FakeBoard:
    fs = 250.0
    n_ch = 8

    def __init__(self):
        self.buffer = FakeBuffer(self.n_ch)

    def status(self):
        return {"kind": "fake"}


def make_pipeline(labels=None):
    cfg = {"flow_index": {"baseline_mode": "rest"}}
    if labels:
        cfg["labels_8"] = labels
    return Pipeline(FakeBoard(), cfg, recorder=None, log=lambda *a: None)


def freeze(p, seed=0):
    rng = np.random.default_rng(seed)
    p.flow.start_rest()
    p.live_base.start_rest()
    for _ in range(40):
        for b in p.flow.baselines.values():
            b.push(float(rng.normal(-1.5, 0.2)))
        p.live_base.push(float(rng.normal(-1.0, 0.1)))
    p.flow.end_rest()
    p.live_base.end_rest()


class TestBaselinePersist(unittest.TestCase):
    def test_round_trip(self):
        p = make_pipeline()
        self.assertTrue(p.rest_block_needed())
        freeze(p)
        self.assertFalse(p.rest_block_needed())
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "s" / "baseline.json"
            p.save_baseline(f)
            q = make_pipeline()
            self.assertTrue(q.load_baseline(f))
            self.assertFalse(q.rest_block_needed())
            for k in ("theta", "alpha", "beta"):
                self.assertAlmostEqual(q.flow.baselines[k].mean, p.flow.baselines[k].mean)
                self.assertAlmostEqual(q.flow.baselines[k].std, p.flow.baselines[k].std)
                self.assertTrue(q.flow.baselines[k].frozen)
            self.assertEqual(q.flow.baseline_state, "rest-frozen")
            self.assertAlmostEqual(q.live_base.mean, p.live_base.mean)
            self.assertTrue(q.live_base.ready)
            self.assertEqual(q.flow_status()["baseline_restored_from"], str(f))
            # the same z for the same value after the restart
            self.assertAlmostEqual(q.flow.baselines["theta"].push(-1.2), p.flow.baselines["theta"].push(-1.2))

    def test_refuses_other_montage_and_unfrozen(self):
        p = make_pipeline()
        freeze(p)
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "baseline.json"
            p.save_baseline(f)
            other = make_pipeline(["Fz", "F3", "F4", "Cz", "C3", "C4", "P3", "P4"])
            self.assertFalse(other.load_baseline(f))
            self.assertTrue(other.rest_block_needed())
            raw = make_pipeline()
            raw.save_baseline(f)                               # session baseline only, never frozen
            self.assertFalse(make_pipeline().load_baseline(f))
            self.assertFalse(make_pipeline().load_baseline(Path(d) / "missing.json"))

    def test_server_saves_at_rest_end_and_app_restores(self):
        import app as app_mod
        from server import App

        class Rec:
            active, dir = False, None

            def status(self):
                return {}

        class Bands:
            def status(self):
                return {}

        with tempfile.TemporaryDirectory() as d:
            sessions = Path(d)
            p = make_pipeline()
            a = App(p, p.board, Rec(), Bands(), {}, log=lambda *_: None)
            a.sessions_dir = sessions
            a.on_protocol_event({"phase": "block_start", "baseline": True, "block_index": 0, "condition": "rest"})
            rng = np.random.default_rng(1)
            for _ in range(40):
                for b in p.flow.baselines.values():
                    b.push(float(rng.normal(-1.5, 0.2)))
            a.on_protocol_event({"phase": "block_end", "baseline": True, "block_index": 0, "condition": "rest"})
            day_file = sessions / f"{time.strftime('%Y-%m-%d')}_baseline.json"
            self.assertTrue(day_file.exists())                 # no recording running: the day file
            self.assertEqual(json.loads(day_file.read_text("utf-8"))["state"], "rest-frozen")
            # a recording folder's copy, newer, wins on restore
            Rec.active, Rec.dir = True, sessions / f"{time.strftime('%Y-%m-%d')}_10-00-00_lily"
            p.flow.baselines["theta"].mean = -9.0
            a.save_baseline()
            self.assertTrue((Rec.dir / "baseline.json").exists())
            os.utime(day_file, (time.time() - 60, time.time() - 60))
            q = make_pipeline()
            got = app_mod.restore_baseline(q, sessions, "")
            self.assertEqual(got, Rec.dir / "baseline.json")
            self.assertEqual(q.flow.baselines["theta"].mean, -9.0)
            # another wearer's folder is skipped with a history match; the day file is still a candidate
            q2 = make_pipeline()
            self.assertEqual(app_mod.restore_baseline(q2, sessions, "michael"), day_file)
            self.assertIsNone(app_mod.restore_baseline(make_pipeline(["Fz", "F3", "F4", "Cz", "C3", "C4", "P3", "P4"]), sessions, ""))


if __name__ == "__main__":
    unittest.main()
