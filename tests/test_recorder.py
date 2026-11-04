"""Recorder: session folder, CSV headers / row counts, session.json."""
import csv
import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

import _path  # noqa: F401
from recorder import Recorder

LABELS = ["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"]


def read_csv(p: Path):
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def epoch_rec(t, flow):
    rec = {"t": t, "flow_eeg": flow, "flow_eeg_alt": flow, "artifact": False, "theta_frontal_rel": 0.21,
           "alpha_cp_rel": 0.3, "beta_rel": 0.18, "engagement_index": 0.52, "theta_alpha_ratio": 0.8,
           "z_theta": 0.4, "z_alpha": -0.1, "z_beta": -0.3, "baseline_ready": True,
           "baseline_state": "rest-frozen", "n_good": 8, "frontal_sites": "Fp1,Fp2"}
    for b in ("delta", "theta", "alpha", "beta", "gamma"):
        rec[f"{b}_rel_mean"] = 0.2
    rec["p2p_uv"] = 42.1
    rec["n_segments_used"] = 9
    for lab in LABELS:
        rec[f"theta_abs_{lab}"] = 12.5
        rec[f"alpha_abs_{lab}"] = 20.0
        rec[f"beta_abs_{lab}"] = 8.0
        rec[f"q_{lab}"] = "good"
    return rec


class TestRecorder(unittest.TestCase):
    def test_full_session(self):
        with tempfile.TemporaryDirectory() as d:
            rec = Recorder(Path(d) / "sessions", LABELS, {"board": {"kind": "sim", "fs": 250}, "fs": 250})
            self.assertFalse(rec.active)
            self.assertIsNone(rec.stop())                       # stop before start is a no-op
            rec.mark("felt_flow", 0.5)                          # ignored while inactive
            sdir = rec.start("desk")
            self.assertTrue(rec.active)
            self.assertIs(rec.start("other"), sdir)             # second start returns the same dir
            self.assertRegex(sdir.name, r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_desk$")
            self.assertTrue(sdir.is_dir())

            t0 = 1762178471.0
            samples = np.arange(24, dtype=np.float32).reshape(3, 8)
            accel = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (3, 1))
            rec.eeg(samples, accel, t0 + np.arange(3) / 250.0, 100)
            rec.eeg(samples * 2, accel, t0 + np.arange(3, 6) / 250.0, 103)
            rec.epoch(epoch_rec(t0 + 2, 0.61))
            rec.epoch(epoch_rec(t0 + 4, 0.63))
            rec.band({"t": t0 + 1, "flow": 0.58, "hr": 71.5, "anom": 0.12, "held": False,
                      "settling": False, "fhi": None})
            rec.band({"t": t0 + 3, "flow": None, "hr": None, "anom": 0.2, "held": True,
                      "settling": True, "fhi": 0.4})
            rec.mark("felt_flow", 0.7, "deep")
            rec.mark("note", "coffee")
            rec.mark("flow_start", "")

            out = rec.stop()
            self.assertEqual(out, sdir)
            self.assertFalse(rec.active)
            self.assertEqual(rec.counts, {"eeg": 6, "epochs": 2, "band": 2, "marks": 3, "trials": 0})
            st = rec.status()
            self.assertEqual(st["counts"]["eeg"], 6)
            self.assertIsNone(st["since_s"])
            self.assertEqual(st["dir"], str(sdir))

            for name in ("eeg_raw.csv", "epochs.csv", "band.csv", "marks.csv", "task.csv", "session.json"):
                self.assertTrue((sdir / name).exists(), name)

            h, rows = read_csv(sdir / "eeg_raw.csv")
            self.assertEqual(h, ["t_host", "sample"] + [f"ch{i + 1}_{l}" for i, l in enumerate(LABELS)]
                             + ["acc_x", "acc_y", "acc_z"])
            self.assertEqual(len(rows), 6)
            self.assertEqual([r[1] for r in rows], ["100", "101", "102", "103", "104", "105"])
            self.assertEqual(rows[0][2:10], [f"{v:.2f}" for v in samples[0]])
            self.assertEqual(rows[0][10:], ["0.000", "0.000", "1.000"])

            h, rows = read_csv(sdir / "epochs.csv")
            self.assertEqual(h, list(epoch_rec(0, 0).keys()))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0][h.index("flow_eeg")], "0.61")
            self.assertEqual(rows[1][h.index("baseline_ready")], "True")
            self.assertEqual(rows[1][h.index("baseline_state")], "rest-frozen")
            self.assertEqual(rows[0][h.index("artifact")], "False")
            self.assertEqual(rows[0][h.index("alpha_cp_rel")], "0.3")
            self.assertEqual(rows[0][h.index("n_segments_used")], "9")
            self.assertEqual(rows[0][h.index("frontal_sites")], "Fp1,Fp2")
            self.assertEqual(rows[0][h.index("q_O2")], "good")
            self.assertEqual(rows[0][h.index("theta_abs_Fp1")], "12.5")

            h, rows = read_csv(sdir / "band.csv")
            self.assertEqual(h, ["t", "flow", "hr", "anom", "held", "settling", "fhi", "asleep", "poor_contact"])
            self.assertEqual(rows[0], [f"{t0 + 1:.3f}", "0.58", "71.5", "0.12", "0", "0", "", "0", "0"])
            self.assertEqual(rows[1], [f"{t0 + 3:.3f}", "", "", "0.2", "1", "1", "0.4", "0", "0"])

            h, rows = read_csv(sdir / "marks.csv")
            self.assertEqual(h, ["t", "kind", "value", "note"])
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0][1:], ["felt_flow", "0.7", "deep"])
            self.assertEqual(rows[1][1:], ["note", "coffee", ""])
            self.assertTrue(re.fullmatch(r"\d+\.\d{3}", rows[0][0]))

            meta = json.loads((sdir / "session.json").read_text("utf-8"))
            self.assertEqual(meta["labels"], LABELS)
            self.assertEqual(meta["board"], {"kind": "sim", "fs": 250})
            self.assertEqual(meta["fs"], 250)
            self.assertEqual(meta["started"], sdir.name[:19])

            # producers after stop are ignored and nothing is written
            rec.mark("note", "late")
            self.assertEqual(rec.counts["marks"], 3)

    def test_task_csv(self):
        with tempfile.TemporaryDirectory() as d:
            rec = Recorder(Path(d), LABELS, {})
            rec.trial({"block_index": 1, "condition": "flow", "task": "arithmetic_adaptive"})   # inactive: ignored
            sdir = rec.start("difficulty_ladder")
            self.assertEqual(rec.counts["trials"], 0)                   # counter exists from the start
            t0 = 1762178471.0
            rec.trial({"t": t0 + 1.5, "block_index": 1, "condition": "flow", "task": "arithmetic_adaptive",
                       "level": 4, "problem": "23 + 45", "answer": 68, "response": 68, "correct": True,
                       "rt_ms": 2345})
            rec.trial({"t": t0 + 9.25, "block_index": 1, "condition": "flow", "task": "arithmetic_adaptive",
                       "level": 5, "problem": "67 - 29", "answer": 38, "response": 36, "correct": False,
                       "rt_ms": 4100.5})
            rec.trial({"t": t0 + 20, "block_index": 3, "condition": "overload", "task": "arithmetic_hard",
                       "level": 12, "problem": "47 x 83", "answer": 3901, "response": None, "correct": None,
                       "rt_ms": None})
            rec.epoch(epoch_rec(t0 + 2, 0.5))
            rec.stop()
            self.assertEqual(rec.counts["trials"], 3)
            self.assertEqual(rec.counts["epochs"], 1)
            self.assertEqual(rec.status()["counts"]["trials"], 3)
            h, rows = read_csv(sdir / "task.csv")
            self.assertEqual(h, ["t", "block_index", "condition", "task", "level", "problem", "answer",
                                 "response", "correct", "rt_ms"])
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0], [f"{t0 + 1.5:.3f}", "1", "flow", "arithmetic_adaptive", "4", "23 + 45",
                                       "68", "68", "1", "2345"])
            self.assertEqual(rows[1], [f"{t0 + 9.25:.3f}", "1", "flow", "arithmetic_adaptive", "5", "67 - 29",
                                       "38", "36", "0", "4100.5"])
            self.assertEqual(rows[2][:4], [f"{t0 + 20:.3f}", "3", "overload", "arithmetic_hard"])
            self.assertEqual(rows[2][7:], ["", "0", ""])                 # None response / correct / rt -> empty, 0
            # a trial without t gets a clock stamp
            rec2 = Recorder(Path(d) / "b", LABELS, {})
            sdir2 = rec2.start()
            rec2.trial({"block_index": 0, "task": "arithmetic_easy", "correct": 1})
            rec2.stop()
            h, rows = read_csv(sdir2 / "task.csv")
            self.assertTrue(re.fullmatch(r"\d+\.\d{3}", rows[0][0]))
            self.assertEqual(rows[0][8], "1")
            # trials after stop are ignored
            rec2.trial({"block_index": 0, "correct": 1})
            self.assertEqual(rec2.counts["trials"], 1)

    def test_no_epochs_means_no_epochs_csv(self):
        with tempfile.TemporaryDirectory() as d:
            rec = Recorder(Path(d), LABELS, {})
            sdir = rec.start()
            self.assertRegex(sdir.name, r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
            rec.stop()
            self.assertFalse((sdir / "epochs.csv").exists())
            h, rows = read_csv(sdir / "band.csv")
            self.assertEqual(rows, [])
            h, rows = read_csv(sdir / "task.csv")                  # task.csv always exists, header only
            self.assertEqual(h[:3], ["t", "block_index", "condition"])
            self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
