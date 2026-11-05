"""autocal.py: the band-pass and event detector, the template (file + geometry fallback), the
capture, the features, and above all the solver recovering a known placement from synthetic
signals built on the template; plus edfread on a synthetic EDF."""
from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

import _path  # noqa: F401

import autocal
import edfread

TRUTH = ["AF7", "Fp1", "Fp2", "AF8", "F7", "F3", "F4", "F8"]


class TestSignalHelpers(unittest.TestCase):
    def test_bandpass_keeps_band_and_removes_rest(self):
        fs = 250.0
        t = np.arange(int(10 * fs)) / fs
        x = (np.sin(2 * np.pi * 6 * t) + np.sin(2 * np.pi * 50 * t) + 5.0)[:, None]
        y = autocal.bandpass(x, fs, 1.0, 30.0)[:, 0]
        mid = slice(int(2 * fs), int(8 * fs))
        ref = np.sin(2 * np.pi * 6 * t)[mid]
        self.assertLess(np.abs(y[mid] - ref).max(), 0.3)       # 6 Hz kept, 50 Hz + DC gone (FFT edge ripple allowed)
        self.assertLess(np.abs(y[mid]).max(), 1.3)

    def test_detect_events(self):
        fs = 250.0
        x = np.random.default_rng(0).normal(0, 1, int(20 * fs))
        for k in (3.0, 7.0, 7.2, 15.0):
            i = int(k * fs); x[i: i + 10] += 30
        pk = autocal.detect_events(x, fs, min_gap_s=0.5, k_mad=4.0)
        self.assertEqual(len(pk), 3)                           # 7.0 and 7.2 merge (either may win)
        self.assertLess(abs(pk[0] / fs - 3.0), 0.1)
        self.assertLess(abs(pk[1] / fs - 7.1), 0.2)
        self.assertLess(abs(pk[2] / fs - 15.0), 0.1)
        self.assertEqual(autocal.detect_events(np.zeros(100), fs), [])


class TestTemplate(unittest.TestCase):
    def test_file_template_if_present_else_geometry(self):
        t = autocal.load_template()
        self.assertEqual(len(t["sites"]), 17)
        self.assertEqual(len(t["corr"]), 17)
        self.assertEqual(len(t["blink"]), 17); self.assertEqual(len(t["heog"]), 17)
        b = dict(zip(t["sites"], t["blink"]))
        self.assertGreater(b["Fp1"], b["F3"])                   # blinks fall with distance from the eyes
        h = dict(zip(t["sites"], t["heog"]))
        self.assertGreater(h["F7"], 0); self.assertLess(h["F8"], 0)   # left positive, right negative
        g = autocal.geometry_template()
        self.assertEqual(g["corr"][0][0], 1.0)

    def test_missing_file_falls_back(self):
        t = autocal.load_template(Path("nope.json"))
        self.assertIn("geometry", t["source"])


class TestCaptureAndFeatures(unittest.TestCase):
    def test_capture_phases_and_cap(self):
        cap = autocal.Capture(250.0, 8, max_s=1.0)
        cap.push(np.zeros((100, 8)))                            # no phase: dropped
        cap.set_phase("rest")
        cap.push(np.zeros((200, 8))); cap.push(np.zeros((200, 8)))   # capped at 250 samples
        self.assertEqual(cap.counts["rest"], 250)
        self.assertEqual(cap.data("rest").shape, (250, 8))
        self.assertIsNone(cap.data("blink"))
        with self.assertRaises(ValueError):
            cap.set_phase("dance")
        st = cap.status()
        self.assertEqual(st["phase"], "rest"); self.assertEqual(st["seconds"]["rest"], 1.0)

    def test_features_need_rest(self):
        cap = autocal.Capture(250.0, 8)
        with self.assertRaises(ValueError):
            autocal.features(cap)

    def test_features_shape(self):
        cap = autocal.capture_from_synthetic(TRUTH, seed=3)
        f = autocal.features(cap)
        self.assertEqual(f["corr"].shape, (8, 8))
        self.assertEqual(len(f["blink"]), 8); self.assertEqual(len(f["heog"]), 8); self.assertEqual(len(f["emg"]), 8)
        self.assertGreaterEqual(f["n_blink"], 5)
        self.assertGreater(f["heog_var"], 0.5)
        self.assertEqual(f["notes"], [])


SLOTS = ["Fp1", "Fp2", "AF7", "AF8", "F7", "F3", "F4", "F8"]


def _dist(a: str, b: str) -> float:
    xa, ya = autocal.SITE_XY[a]; xb, yb = autocal.SITE_XY[b]
    return ((xa - xb) ** 2 + (ya - yb) ** 2) ** 0.5


class TestSolver(unittest.TestCase):
    def test_recovers_known_placement_with_the_helmet_slots(self):
        tpl = autocal.load_template()
        for seed, noise in ((1, 0.3), (2, 1.0), (5, 1.0)):
            cap = autocal.capture_from_synthetic(TRUTH, tpl, seed=seed, noise=noise)
            r = autocal.solve(autocal.features(cap), tpl, slots=SLOTS)
            correct = sum(a == b for a, b in zip(r["labels"], TRUTH))
            self.assertGreaterEqual(correct, 7, (seed, noise, r["labels"]))
            self.assertEqual(len(r["confidence"]), 8)
            self.assertIn(r["heog_sign"], (1.0, -1.0))

    def test_all_17_sites_lands_on_the_true_site_or_its_neighbour(self):
        tpl = autocal.load_template()
        for seed in (1, 3):
            cap = autocal.capture_from_synthetic(TRUTH, tpl, seed=seed, noise=0.5)
            r = autocal.solve(autocal.features(cap), tpl)
            for got, true in zip(r["labels"], TRUTH):
                self.assertLessEqual(_dist(got, true), 0.36, (seed, got, true))   # same site or an adjacent 10-10 site
            self.assertGreaterEqual(sum(a == b for a, b in zip(r["labels"], TRUTH)), 5)

    def test_restricted_slots_and_errors(self):
        tpl = autocal.load_template()
        cap = autocal.capture_from_synthetic(TRUTH, tpl, seed=4)
        f = autocal.features(cap)
        r = autocal.solve(f, tpl, slots=SLOTS)
        self.assertEqual(sorted(r["labels"]), sorted(TRUTH))
        self.assertGreaterEqual(sum(a == b for a, b in zip(r["labels"], TRUTH)), 7)
        with self.assertRaises(ValueError):
            autocal.solve(f, tpl, slots=["Fp1", "Fp2"])

    def test_rest_only_still_solves(self):
        tpl = autocal.load_template()
        cap = autocal.Capture(250.0, 8)
        cap.set_phase("rest"); cap.push(autocal.synthesize(TRUTH, tpl, seed=2)["rest"]); cap.set_phase(None)
        f = autocal.features(cap)
        self.assertIn("no blink phase", f["notes"])
        r = autocal.solve(f, tpl)
        self.assertEqual(len(set(r["labels"])), 8)


class TestEdfRead(unittest.TestCase):
    def test_roundtrip_synthetic_edf(self):
        fs, n_rec, n_sig = 100, 3, 3
        labels = ["Fp1.", "T9..", "EDF Annotations"]
        hdr = (b"0".ljust(8) + b"x".ljust(80) + b"y".ljust(80) + b"01.01.20" + b"00.00.00"
               + str(256 * (1 + n_sig)).encode().ljust(8) + b"".ljust(44) + str(n_rec).encode().ljust(8)
               + b"1".ljust(8) + str(n_sig).encode().ljust(4))
        self.assertEqual(len(hdr), 256)
        def fld(vals, w): return b"".join(str(v).encode().ljust(w) for v in vals)
        sh = (fld(labels, 16) + fld([""] * n_sig, 80) + fld(["uV"] * n_sig, 8) + fld([-1000] * n_sig, 8)
              + fld([1000] * n_sig, 8) + fld([-32768] * n_sig, 8) + fld([32767] * n_sig, 8)
              + fld([""] * n_sig, 80) + fld([fs] * n_sig, 8) + fld([""] * n_sig, 32))
        self.assertEqual(len(sh), 256 * n_sig)
        body = b""
        for r in range(n_rec):
            body += struct.pack("<%dh" % fs, *([1000 * (r + 1)] * fs))     # Fp1: 1000, 2000, 3000 digital
            body += struct.pack("<%dh" % fs, *([-16384] * fs))             # T9: -500 uV
            body += bytes(fs * 2)                                         # annotations
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.edf"
            p.write_bytes(hdr + sh + body)
            x, labs, rate = edfread.read_edf(p)
        self.assertEqual(labs, ["Fp1", "T9"]); self.assertEqual(rate, 100.0)
        self.assertEqual(x.shape, (300, 2))
        self.assertAlmostEqual(float(x[0, 0]), 1000 * 2000 / 65535, places=1)
        self.assertAlmostEqual(float(x[0, 1]), -500.0, places=0)

    def test_clean_label(self):
        self.assertEqual(edfread.clean_label("Fcz."), "FCz"); self.assertEqual(edfread.clean_label("AF7"), "AF7")


if __name__ == "__main__":
    unittest.main()
