"""Small 11-19 review fixes: fresh start is the default (--resume restores), epochs.csv widens
its header for a late key, the live-flow error is rate-limited (not logged once and never again)."""
from __future__ import annotations

import csv
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import _path  # noqa: F401


class TestEpochHeader(unittest.TestCase):
    def test_late_key_widens_the_header(self):
        from recorder import Recorder
        with tempfile.TemporaryDirectory() as d:
            rec = Recorder(Path(d), ["Fp1"], {})
            sdir = rec.start()
            rec.epoch({"t": 1, "flow_eeg": 0.5})
            rec.epoch({"t": 2, "flow_eeg": 0.6, "q_F3": "good"})       # relabelled mid-session
            rec.epoch({"t": 3, "flow_eeg": 0.7, "q_F3": "near"})
            rec.stop()
            with open(sdir / "epochs.csv", newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows[0], ["t", "flow_eeg", "q_F3"])
            self.assertEqual(rows[1:], [["1", "0.5", ""], ["2", "0.6", "good"], ["3", "0.7", "near"]])


class TestLiveErrorRateLimit(unittest.TestCase):
    def test_logged_again_after_a_minute(self):
        import numpy as np
        from test_live_flow import make_pipeline
        p = make_pipeline()
        p._live_flow_inner = lambda *a: 1 / 0
        t = [1000.0]
        with mock.patch("pipeline.time.time", lambda: t[0]):
            for _ in range(5):
                p._live_flow(np.zeros((10, 8)), [True] * 8)
            t[0] += 61
            p._live_flow(np.zeros((10, 8)), [True] * 8)
        lines = [s for s in p.logs if "live flow error" in s]
        self.assertEqual(len(lines), 2)
        self.assertIn("4 more since the last report", lines[1])


class TestFreshDefault(unittest.TestCase):
    def _run(self, argv):
        import app
        calls = []

        class Stop(Exception):
            pass

        def board(*a, **k):
            raise Stop

        with mock.patch.object(app, "preload_today", lambda *a, **k: calls.append("preload")), \
                mock.patch.object(app, "restore_baseline", lambda *a, **k: calls.append("baseline")), \
                mock.patch.object(app, "Pipeline") as P, mock.patch.object(app, "Recorder"), \
                mock.patch.object(app, "open_board", lambda *a: mock.MagicMock()), \
                mock.patch("server.App", side_effect=Stop), mock.patch.object(app, "App", side_effect=Stop):
            P.return_value.labels = ["Fp1"]
            try:
                app.main(argv + ["--no-browser", "--astra", "off"])
            except Stop:
                pass
        return calls

    def test_default_is_fresh(self):
        self.assertEqual(self._run([]), [])

    def test_resume_restores_chart_and_baseline(self):
        self.assertEqual(self._run(["--resume"]), ["preload", "baseline"])


if __name__ == "__main__":
    unittest.main()
