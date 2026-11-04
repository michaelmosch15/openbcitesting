"""Pipeline block-level statistics (_block_stats) from protocol events, synthetic histories
and ratings; rating marks populate pipeline.ratings."""
import json
import time
import unittest

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


def make_pipeline():
    board = FakeBoard()
    p = Pipeline(board, {"flow_index": {"baseline_mode": "rest"}}, recorder=None, log=lambda *a: None)
    return p, board


# synthetic timeline (seconds relative to NOW - 3000): condition, start, end
SCHEDULE = [
    (0, "rest", 0, 180),
    (1, "flow", 180, 480),
    (2, "pause", 480, 540),
    (3, "boredom", 540, 840),
    (4, "pause", 840, 900),
    (5, "overload", 900, 1200),
    (6, "pause", 1200, 1260),
    (7, "flow", 1260, 1560),
    (8, "pause", 1560, 1620),
    (9, "overload", 1620, 1920),
    (10, "pause", 1920, 1980),
    (11, "boredom", 1980, 2280),
    (12, "flow", 2280, 2300),             # too short (20 s): must be skipped
]
LEVEL = {"rest": 0.5, "pause": 0.5, "flow": 0.8, "boredom": 0.3, "overload": 0.4}
FSS = {"flow": 6.0, "boredom": 2.0, "overload": 3.0}
DEMAND = {"flow": 2, "boredom": 1, "overload": 3}


def fill(p, band_level=None, band_every=60.0):
    """Feed on_protocol events, then pin the block times to the synthetic timeline and fill
    the histories with per-condition levels (a small per-block offset keeps ranks unique)."""
    base = time.time() - 3000.0
    band_level = band_level or LEVEL
    for i, cond, _, _ in SCHEDULE:
        p.on_protocol({"phase": "block_start", "block_index": i, "condition": cond, "baseline": i == 0})
        p.on_protocol({"phase": "block_end", "block_index": i, "condition": cond})
    for b, (i, cond, s, e) in zip(p.blocks, SCHEDULE):
        b["t0"], b["t1"] = base + s, base + e
    for i, cond, s, e in SCHEDULE:
        off = 0.004 * i
        for t in np.arange(s + 2, e, 2.0):
            p.eeg_hist.append((base + t, LEVEL[cond] + off))
        for t in np.arange(s + band_every / 2, e, band_every):
            sample = {"t": base + t, "flow": band_level[cond] + off, "held": False, "settling": False,
                      "asleep": False, "poor_contact": False}
            p.band_hist.append((sample["t"], sample["flow"], sample))
        if cond in FSS and e - s >= 30:
            items = {k: FSS[cond] for k in ("challenge", "fluent", "time", "concentrate", "clear",
                                            "absorbed", "automatic", "know", "control", "lost")}
            items["demand"] = DEMAND[cond]
            items["flow_overall"] = FSS[cond] / 7
            p.ratings.append({"t": base + e + 5, "block": i, "condition": cond, "items": items})
    return base


class TestBlockEvents(unittest.TestCase):
    def test_on_protocol_builds_block_table(self):
        p, board = make_pipeline()
        self.assertIn(p._on_samples, board.buffer.listeners)
        self.assertEqual(p.blocks, [])
        p.on_protocol({"phase": "block_start", "block_index": 0, "condition": "rest", "baseline": True})
        self.assertEqual(len(p.blocks), 1)
        b = p.blocks[0]
        self.assertEqual((b["index"], b["condition"], b["baseline"]), (0, "rest", True))
        self.assertIsNone(b["t1"])
        self.assertAlmostEqual(b["t0"], time.time(), delta=2)
        p.on_protocol({"phase": "rating_due", "block_index": 0})        # ignored
        self.assertIsNone(p.blocks[0]["t1"])
        p.on_protocol({"phase": "block_end", "block_index": 0, "condition": "rest"})
        self.assertIsNotNone(p.blocks[0]["t1"])
        p.on_protocol({"phase": "block_end", "block_index": 0})         # a second end does not move t1
        t1 = p.blocks[0]["t1"]
        p.on_protocol({"phase": "block_start", "block_index": 1, "condition": "flow", "baseline": False})
        p.on_protocol({"phase": "stopped", "name": "x"})
        self.assertEqual(p.blocks[0]["t1"], t1)
        self.assertIsNotNone(p.blocks[1]["t1"])
        self.assertFalse(p.blocks[1]["baseline"])

    def test_empty_block_stats(self):
        p, _ = make_pipeline()
        st = p._block_stats()
        self.assertEqual(st, {"rows": [], "conditions": [], "ordering": {}, "block_rho": None, "n_blocks": 0})
        self.assertIsNotNone(p.stats)
        self.assertIsNone(p.stats["blocks"])      # computed lazily on marks / band samples


class TestBlockStats(unittest.TestCase):
    def test_grouping_means_ordering_and_rho(self):
        p, _ = make_pipeline()
        fill(p)
        st = p._block_stats()
        rows = {r["index"]: r for r in st["rows"]}
        # pause blocks and the 20 s block are skipped; rest is kept
        self.assertEqual(sorted(rows), [0, 1, 3, 5, 7, 9, 11])
        for r in st["rows"]:
            self.assertNotEqual(r["condition"], "pause")
        for i in (1, 7):
            self.assertAlmostEqual(rows[i]["eeg_mean"], 0.8 + 0.004 * i, places=3)
            self.assertAlmostEqual(rows[i]["band_mean"], 0.8 + 0.004 * i, places=3)
            self.assertAlmostEqual(rows[i]["fss_mean"], 6.0, places=3)
            self.assertAlmostEqual(rows[i]["felt"], 6.0 / 7, places=6)
        self.assertEqual(rows[1]["n_eeg"], 149)
        self.assertEqual(rows[1]["n_band"], 5)
        self.assertIsNone(rows[0]["fss_mean"])
        self.assertIsNone(rows[0]["felt"])
        self.assertAlmostEqual(rows[3]["eeg_mean"], 0.3 + 0.012, places=3)
        self.assertAlmostEqual(rows[5]["band_mean"], 0.4 + 0.02, places=3)
        conds = {c["condition"]: c for c in st["conditions"]}
        self.assertEqual(sorted(conds), ["boredom", "flow", "overload", "rest"])
        self.assertEqual(conds["flow"]["n_blocks"], 2)
        self.assertEqual(conds["rest"]["n_blocks"], 1)
        self.assertAlmostEqual(conds["flow"]["eeg_mean"], 0.8 + 0.004 * 4, places=3)
        self.assertAlmostEqual(conds["boredom"]["fss_mean"], 2.0, places=3)
        self.assertAlmostEqual(conds["overload"]["felt_mean"], 3.0 / 7, places=3)
        self.assertIsNone(conds["rest"]["fss_mean"])
        self.assertEqual(st["ordering"], {"eeg": "flow>overload>boredom", "band": "flow>overload>boredom",
                                          "fss": "flow>overload>boredom"})
        self.assertEqual(st["n_blocks"], 6)            # task blocks only: rest is excluded from rho
        self.assertAlmostEqual(st["block_rho"], 1.0, places=3)
        self.assertEqual(set(st["rows"][0]), {"index", "condition", "eeg_mean", "band_mean", "fss_mean", "felt",
                                              "n_eeg", "n_band"})

    def test_band_reversed_gives_negative_rho_and_other_ordering(self):
        p, _ = make_pipeline()
        fill(p, band_level={"rest": 0.5, "pause": 0.5, "flow": 0.2, "boredom": 0.8, "overload": 0.6})
        st = p._block_stats()
        self.assertEqual(st["ordering"]["band"], "boredom>overload>flow")
        self.assertEqual(st["ordering"]["eeg"], "flow>overload>boredom")
        self.assertLess(st["block_rho"], -0.5)

    def test_unusable_band_samples_are_skipped(self):
        p, _ = make_pipeline()
        base = fill(p)
        flow_block = [b for b in p.blocks if b["index"] == 1][0]
        for flag in ("held", "settling", "asleep", "poor_contact"):
            s = {"t": flow_block["t0"] + 100, "flow": 0.0, "held": False, "settling": False,
                 "asleep": False, "poor_contact": False, flag: True}
            p.band_hist.append((s["t"], s["flow"], s))
        p.band_hist.append((flow_block["t0"] + 120, None, {"t": flow_block["t0"] + 120, "flow": None}))
        rows = {r["index"]: r for r in p._block_stats()["rows"]}
        self.assertEqual(rows[1]["n_band"], 5)
        self.assertAlmostEqual(rows[1]["band_mean"], 0.804, places=3)
        self.assertGreater(base, 0)

    def test_open_block_uses_now_and_missing_data_gives_none(self):
        p, _ = make_pipeline()
        p.on_protocol({"phase": "block_start", "block_index": 0, "condition": "flow", "baseline": False})
        p.blocks[0]["t0"] = time.time() - 100          # still running, 100 s so far
        st = p._block_stats()
        self.assertEqual(len(st["rows"]), 1)
        r = st["rows"][0]
        self.assertIsNone(r["eeg_mean"])
        self.assertIsNone(r["band_mean"])
        self.assertEqual((r["n_eeg"], r["n_band"]), (0, 0))
        self.assertEqual(st["conditions"][0]["n_blocks"], 1)
        self.assertEqual(st["ordering"], {})
        self.assertIsNone(st["block_rho"])
        self.assertEqual(st["n_blocks"], 0)
        # fewer than 4 blocks with both means -> no rho
        p.eeg_hist.append((time.time() - 50, 0.7))
        p.band_hist.append((time.time() - 40, 0.6, {"flow": 0.6}))
        st = p._block_stats()
        self.assertEqual(st["n_blocks"], 1)
        self.assertIsNone(st["block_rho"])

    def test_stats_include_blocks(self):
        p, _ = make_pipeline()
        fill(p)
        p.mark("note", "x")
        self.assertEqual(p.stats["blocks"]["ordering"]["eeg"], "flow>overload>boredom")
        self.assertEqual(p.stats["n_pairs"], 0)


class TestRatingMark(unittest.TestCase):
    def test_rating_mark_populates_ratings(self):
        p, _ = make_pipeline()
        items = {"challenge": 6.0, "fluent": 5.0, "demand": 2.0, "flow_overall": 0.8}
        m = p.mark("rating", json.dumps({"block": 3, "condition": "flow", **items}), note="ok")
        self.assertEqual(m["kind"], "rating")
        self.assertEqual(len(p.marks), 1)
        self.assertEqual(len(p.ratings), 1)
        r = p.ratings[0]
        self.assertEqual((r["block"], r["condition"]), (3, "flow"))
        self.assertEqual(r["items"], items)
        self.assertAlmostEqual(r["t"], m["t"])
        # a dict value works too; garbage is ignored but still a mark
        p.mark("rating", {"block": 4, "condition": "boredom", "demand": 1})
        self.assertEqual(p.ratings[1]["items"], {"demand": 1})
        p.mark("rating", "not json")
        p.mark("felt_flow", 0.5)
        self.assertEqual(len(p.ratings), 2)
        self.assertEqual(len(p.marks), 4)

    def test_rating_feeds_block_fss(self):
        p, _ = make_pipeline()
        p.on_protocol({"phase": "block_start", "block_index": 0, "condition": "flow", "baseline": False})
        p.on_protocol({"phase": "block_end", "block_index": 0, "condition": "flow"})
        p.blocks[0]["t0"] = p.blocks[0]["t1"] - 300
        p.mark("rating", json.dumps({"block": 0, "condition": "flow", "challenge": 7, "fluent": 5, "demand": 2,
                                     "flow_overall": 0.9}))
        row = p._block_stats()["rows"][0]
        self.assertAlmostEqual(row["fss_mean"], 6.0, places=3)
        self.assertEqual(row["felt"], 0.9)


if __name__ == "__main__":
    unittest.main()
