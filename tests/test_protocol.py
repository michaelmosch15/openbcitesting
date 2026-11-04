"""ProtocolRunner: schedule building (seeded, counterbalanced, pauses), the timed loop with
tiny blocks, ratings, skip / stop, status, and error paths."""
import json
import threading
import time
import unittest

import _path  # noqa: F401
from protocol import ProtocolRunner, load_protocols

TINY = {
    "tiny": {
        "title": "Tiny",
        "description": "for tests",
        "blocks": [
            {"condition": "rest", "label": "Rest", "duration_s": 0.2, "baseline": True},
            {"shuffle": [
                {"condition": "boredom", "label": "Easy", "duration_s": 0.2, "task": "arithmetic_easy", "rating": True},
                {"condition": "flow", "label": "Matched", "duration_s": 0.2, "task": "arithmetic_adaptive", "rating": True},
            ], "repeat": 1,
                "between": {"condition": "pause", "label": "Pause", "duration_s": 0.2}},
            {"condition": "rest", "label": "Rest", "duration_s": 0.2},
        ],
    },
    "slow": {
        "title": "Slow",
        "blocks": [
            {"condition": "rest", "label": "Rest", "duration_s": 30, "baseline": True},
            {"condition": "flow", "label": "Flow", "duration_s": 30, "task": "arithmetic_adaptive", "rating": True},
            {"condition": "boredom", "label": "Easy", "duration_s": 30, "task": "arithmetic_easy", "rating": True},
        ],
    },
}


class Sink:
    """Collects events and marks from the runner's callbacks (called from its thread)."""

    def __init__(self):
        self.events, self.marks = [], []
        self.lock = threading.Lock()

    def on_event(self, ev):
        with self.lock:
            self.events.append(dict(ev))

    def on_mark(self, kind, value, note=""):
        with self.lock:
            self.marks.append((kind, value, note))

    def phases(self):
        with self.lock:
            return [(e["phase"], e.get("block_index")) for e in self.events]

    def wait_for(self, pred, timeout=5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                if pred(self.events, self.marks):
                    return True
            time.sleep(0.02)
        return False


def make(protocols=None):
    sink = Sink()
    runner = ProtocolRunner(protocols or TINY, sink.on_event, sink.on_mark, log=lambda *a: None)
    return runner, sink


def task_conditions(blocks):
    return [b["condition"] for b in blocks if b["condition"] != "pause"]


class TestBuild(unittest.TestCase):
    def setUp(self):
        self.real = load_protocols()
        self.assertIn("difficulty_ladder", self.real)

    def test_seed_is_reproducible_and_logged(self):
        r, _ = make(self.real)
        a = r.build("difficulty_ladder", seed=42)
        self.assertEqual(r.order_seed, 42)
        b = r.build("difficulty_ladder", seed=42)
        self.assertEqual([x["condition"] for x in a], [x["condition"] for x in b])
        self.assertEqual([x.get("rep") for x in a], [x.get("rep") for x in b])
        orders = {tuple(x["condition"] for x in r.build("difficulty_ladder", seed=s)) for s in range(30)}
        self.assertGreater(len(orders), 1)          # the seed really shuffles
        r.build("difficulty_ladder")                  # no seed -> clock seed
        self.assertIsInstance(r.order_seed, int)

    def test_layout_rest_pauses_and_no_trailing_pause(self):
        r, _ = make(self.real)
        proto = self.real["difficulty_ladder"]["blocks"]
        fixed = [b for b in proto if "shuffle" not in b]
        shuffle = next(b for b in proto if "shuffle" in b)
        group, reps = len(shuffle["shuffle"]), int(shuffle.get("repeat", 1))
        n_task = group * reps
        blocks = r.build("difficulty_ladder", seed=1)
        conds = [b["condition"] for b in blocks]
        self.assertEqual(len(blocks), len(fixed) + n_task + (n_task - 1))      # pauses between, none trailing
        self.assertEqual(conds[0], "rest")
        self.assertEqual(conds[-1], "rest")
        first = conds.index(next(c for c in conds if c in ("boredom", "flow", "overload")))
        region = conds[first: first + 2 * n_task - 1]
        self.assertEqual(region[1::2], ["pause"] * (n_task - 1))                # between the task blocks ...
        self.assertNotIn("pause", region[0::2])
        self.assertNotEqual(conds[first + 2 * n_task - 1], "pause")             # ... not after the last one
        self.assertEqual(conds.count("pause"), n_task - 1)
        self.assertEqual(sorted(c for c in conds if c in ("boredom", "flow", "overload")),
                         sorted(["boredom", "flow", "overload"] * reps))
        self.assertEqual([b["rep"] for b in blocks if b["condition"] in ("boredom", "flow", "overload")],
                         [k + 1 for k in range(reps) for _ in range(group)])
        # the eyes-closed rest (if present) sits before the task blocks, untouched by the shuffle
        if any(b["condition"] == "rest_ec" for b in fixed):
            self.assertLess(conds.index("rest_ec"), first)

    def test_no_condition_repeats_back_to_back(self):
        r, _ = make(self.real)
        for name in ("difficulty_ladder", "sim_quick"):
            for seed in range(200):
                conds = task_conditions(r.build(name, seed=seed))
                for a, b in zip(conds, conds[1:]):
                    self.assertNotEqual(a, b, f"{name} seed {seed}: {conds}")

    def test_indices_and_defaults(self):
        r, _ = make(self.real)
        blocks = r.build("difficulty_ladder", seed=3)
        for i, b in enumerate(blocks):
            self.assertEqual(b["index"], i)
            for k in ("rating", "baseline", "task", "condition", "duration_s"):
                self.assertIn(k, b)
        self.assertTrue(blocks[0]["baseline"])
        self.assertFalse(blocks[0]["rating"])
        self.assertIsNone(blocks[0]["task"])
        for b in blocks:
            if b["condition"] == "pause":
                self.assertFalse(b["rating"])
                self.assertIsNone(b["task"])
            elif b["condition"] in ("boredom", "flow", "overload"):
                self.assertTrue(b["rating"])
                self.assertTrue(b["task"].startswith("arithmetic_"))
        # building does not mutate the protocol definition
        proto = self.real["difficulty_ladder"]["blocks"]
        self.assertNotIn("index", proto[0])
        self.assertNotIn("index", next(b for b in proto if "shuffle" in b)["shuffle"][0])

    def test_unknown_protocol(self):
        r, _ = make()
        with self.assertRaises(ValueError):
            r.build("nope")
        with self.assertRaises(ValueError):
            r.start("nope")
        self.assertFalse(r.active)

    def test_status_available_total_min_matches_schedule(self):
        r, _ = make(self.real)
        st = r.status()
        avail = {a["name"]: a for a in st["available"]}
        self.assertEqual(set(avail), set(self.real))
        for name in self.real:
            total = sum(float(b.get("duration_s", 0)) for b in r.build(name, seed=0)) / 60
            self.assertAlmostEqual(avail[name]["total_min"], round(total, 1), places=6, msg=name)
            self.assertEqual(avail[name]["title"], self.real[name]["title"])
        # independent recount straight from the JSON (fixed blocks + group x repeat + pauses between)
        proto = self.real["difficulty_ladder"]["blocks"]
        total = 0.0
        for item in proto:
            if "shuffle" in item:
                reps = int(item.get("repeat", 1))
                n = len(item["shuffle"]) * reps
                total += sum(float(b["duration_s"]) for b in item["shuffle"]) * reps
                total += float(item["between"]["duration_s"]) * (n - 1)
            else:
                total += float(item["duration_s"])
        self.assertAlmostEqual(avail["difficulty_ladder"]["total_min"], round(total / 60, 1), places=6)
        self.assertEqual(avail["sim_quick"]["total_min"], round((60 + 6 * 60 + 5 * 20 + 30) / 60, 1))
        # a preview never changes the runner's own seed
        r.build("sim_quick", seed=77)
        r.status()
        self.assertEqual(r.order_seed, 77)
        self.assertFalse(st["active"])
        self.assertIsNone(st["block"])
        self.assertIsNone(st["block_index"])


class TestRun(unittest.TestCase):
    def test_tiny_protocol_runs_through(self):
        r, sink = make()
        st = r.start("tiny", seed=5)
        self.assertTrue(st["active"])
        self.assertEqual(st["name"], "tiny")
        self.assertEqual(st["total_blocks"], 5)
        self.assertEqual(st["order_seed"], 5)
        self.assertEqual(len(st["schedule"]), 5)
        self.assertTrue(sink.wait_for(lambda ev, mk: any(e["phase"] == "done" for e in ev)))
        r._thread.join(2.0)
        self.assertFalse(r.active)
        self.assertEqual(r.index, 5)
        conds = [b["condition"] for b in r.blocks]
        self.assertEqual(conds[0], "rest")
        self.assertEqual(conds[2], "pause")
        self.assertEqual(conds[-1], "rest")
        # events: block_start / (rating_due) / block_end per block, then done
        expected = []
        for b in r.blocks:
            expected.append(("block_start", b["index"]))
            if b["rating"]:
                expected.append(("rating_due", b["index"]))
            expected.append(("block_end", b["index"]))
        expected.append(("done", None))
        self.assertEqual(sink.phases(), expected)
        first = sink.events[0]
        for k in ("block_index", "condition", "label", "instruction", "duration_s", "task", "rating", "baseline", "rep"):
            self.assertIn(k, first)
        self.assertTrue(first["baseline"])
        # marks: protocol_start, then block_start/block_end pairs, then protocol_end
        kinds = [m[0] for m in sink.marks]
        self.assertEqual(kinds[0], "protocol_start")
        self.assertEqual(kinds[-1], "protocol_end")
        self.assertEqual(kinds[1:-1], ["block_start", "block_end"] * 5)
        self.assertEqual(sink.marks[0][1], "tiny")
        self.assertEqual(sink.marks[0][2], "seed=5 order=" + ",".join(conds))
        starts = [m for m in sink.marks if m[0] == "block_start"]
        for i, (b, m) in enumerate(zip(r.blocks, starts)):
            self.assertEqual(m[1], b["condition"])
            self.assertTrue(m[2].startswith(f"block={i} "), m[2])
        ends = [m for m in sink.marks if m[0] == "block_end"]
        self.assertEqual([m[2] for m in ends], [f"block={i}" for i in range(5)])
        # the last rated block is still awaiting its self-report
        rated = [b["index"] for b in r.blocks if b["rating"]]
        self.assertEqual(r.pending_rating, rated[-1])
        # done -> status shows no current block, schedule intact
        st = r.status()
        self.assertFalse(st["active"])
        self.assertIsNone(st["block"])
        self.assertIsNone(st["remaining_s"])
        self.assertEqual(len(st["schedule"]), 5)

    def test_rate_clears_pending_and_emits_json_mark(self):
        r, sink = make()
        r.start("tiny", seed=1)
        self.assertTrue(sink.wait_for(lambda ev, mk: any(e["phase"] == "done" for e in ev)))
        r._thread.join(2.0)
        pending = r.pending_rating
        self.assertIsNotNone(pending)
        items = {"challenge": 6, "fluent": 5, "time": 6, "concentrate": 5, "clear": 6, "absorbed": 6,
                 "automatic": 5, "know": 6, "control": 5, "lost": 4, "demand": 2, "flow_overall": 0.8}
        rec = r.rate(pending, items, note="felt good")
        self.assertIsNone(r.pending_rating)
        self.assertEqual(rec["block_index"], pending)
        self.assertEqual(rec["condition"], r.blocks[pending]["condition"])
        self.assertEqual(rec["items"], items)
        self.assertEqual(r.ratings, [rec])
        mark = [m for m in sink.marks if m[0] == "rating"]
        self.assertEqual(len(mark), 1)
        d = json.loads(mark[0][1])
        self.assertEqual(d["block"], pending)
        self.assertEqual(d["condition"], r.blocks[pending]["condition"])
        for k, v in items.items():
            self.assertEqual(d[k], v)
        self.assertEqual(mark[0][2], "felt good")
        self.assertIn(("rated", pending), sink.phases())
        # rating another (already not pending) block keeps pending None, out-of-range -> condition None
        rec2 = r.rate(99, {"demand": 1})
        self.assertIsNone(rec2["condition"])
        self.assertEqual(json.loads([m for m in sink.marks if m[0] == "rating"][-1][1])["condition"], None)

    def test_next_skips_block(self):
        r, sink = make()
        r.start("slow", seed=0)
        self.assertTrue(sink.wait_for(lambda ev, mk: any(e["phase"] == "block_start" for e in ev)))
        self.assertEqual(r.status()["block_index"], 0)
        self.assertGreater(r.status()["remaining_s"], 20)
        r.next()
        self.assertTrue(sink.wait_for(lambda ev, mk: ("block_start", 1) in [(e["phase"], e.get("block_index")) for e in ev]))
        self.assertEqual(r.status()["block_index"], 1)
        self.assertEqual(r.status()["block"]["condition"], "flow")
        self.assertEqual([m[2] for m in sink.marks if m[0] == "block_end"], ["block=0"])
        r.stop()
        r._thread.join(2.0)

    def test_stop_aborts(self):
        r, sink = make()
        r.start("slow", seed=0)
        self.assertTrue(sink.wait_for(lambda ev, mk: any(e["phase"] == "block_start" for e in ev)))
        r.next()
        self.assertTrue(sink.wait_for(lambda ev, mk: ("rating_due", 1) in [(e["phase"], e.get("block_index")) for e in ev]
                        or ("block_start", 1) in [(e["phase"], e.get("block_index")) for e in ev]))
        st = r.stop()
        r._thread.join(2.0)
        self.assertFalse(st["active"])
        self.assertFalse(r.active)
        self.assertIsNone(st["block"])
        self.assertIsNone(st["block_index"])
        ends = [m for m in sink.marks if m[0] == "block_end"]
        self.assertEqual(ends[-1][1], "flow")
        self.assertEqual(ends[-1][2], "block=1 aborted")
        self.assertEqual(sink.marks[-1][0], "protocol_stop")
        phases = sink.phases()
        self.assertEqual(phases[-1], ("stopped", None))
        self.assertEqual(phases[-2], ("block_end", 1))
        self.assertNotIn(("rating_due", 1), phases)      # an aborted block asks for no rating
        self.assertNotIn(("done", None), phases)
        self.assertIsNone(r.pending_rating)
        # a second stop is a no-op
        n = len(sink.marks)
        r.stop()
        self.assertEqual(len(sink.marks), n)

    def test_double_start_rejected_and_restart_after_done(self):
        r, sink = make()
        r.start("slow", seed=0)
        with self.assertRaises(ValueError):
            r.start("slow")
        with self.assertRaises(ValueError):
            r.start("tiny")
        r.stop()
        r._thread.join(2.0)
        sink.events.clear()
        r.start("tiny", seed=2)
        self.assertTrue(sink.wait_for(lambda ev, mk: any(e["phase"] == "done" for e in ev)))
        r._thread.join(2.0)
        self.assertEqual(r.name, "tiny")
        self.assertEqual(r.ratings, [])


if __name__ == "__main__":
    unittest.main()
