"""astra.py without the network: key loading, the SSE chunk parser, the two snapshot builders
and their text renderings, the live analyst's note / history / file behaviour with a fake chat
function, the setup helper's transcript, and the facade's command dispatch."""
from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from collections import deque
from pathlib import Path

import _path  # noqa: F401

import astra


class FakeFlow:
    baseline_state = "rest-frozen"
    baseline_ready = True
    n_epochs = 120
    n_artifact = 3
    last = {"frontal_sites": "Fz,F3,F4"}


class FakePipeline:
    def __init__(self, now):
        self.labels = ["Fz", "F3", "F4", "Cz", "C3", "C4", "P3", "P4"]
        self.n_ch = 8
        self.enabled = [True] * 8
        self.enabled[7] = False
        self.eeg_hist = deque((now - 300 + 2 * i, 0.4 + 0.001 * i) for i in range(150))
        self.band_hist = deque()
        for i in range(5):
            t = now - 290 + 60 * i
            self.band_hist.append((t, 0.5 + 0.02 * i, {"t": t, "flow": 0.5 + 0.02 * i, "hr": 70, "held": i == 4}))
        self.pairs = deque((now - 290 + 60 * i, 0.5 + 0.02 * i, 0.45 + 0.01 * i) for i in range(4))
        self.stats = {"n_pairs": 30, "r": 0.41, "r_ci": [0.1, 0.6], "r_last_5min": 0.5, "mae": 0.1,
                      "band_minus_eeg_mean": 0.05, "best_lag_s": 60, "r_at_best_lag": 0.55, "pair_window_s": 60,
                      "blocks": {"rows": [{"index": 2, "condition": "boredom", "eeg_mean": 0.44, "band_mean": 0.3,
                                           "fss_mean": 3.1, "felt": 0.3, "n_eeg": 170, "n_band": 6}],
                                 "conditions": [], "ordering": {"eeg": "flow>boredom"}, "block_rho": None, "n_blocks": 1}}
        self.trials = [{"block_index": 4, "correct": True, "level": 5}, {"block_index": 4, "correct": False, "level": 4}]
        self.marks = [{"t": now - 100, "kind": "felt_flow", "value": 0.7, "note": ""},
                      {"t": now - 50, "kind": "trial", "value": None, "note": ""}]
        self.last_analysis = {"quality": [{"state": "good", "pct": 10.0, "std": 8.0}] * 7 + [{"state": "railed", "pct": 95.0, "std": 1.0}],
                              "bands": {"rel": {"theta": [0.2] * 8, "alpha": [0.3] * 8}},
                              "flow_eeg": {"flow_eeg": 0.55, "artifact": False}}
        self.flow = FakeFlow()
        self.last_raw_peak = [20.0] * 7 + [4000.0]

    def band_view(self):
        s = self.band_hist[-1][2]
        return {"t": s["t"], "flow": s["flow"], "hr": 70, "mode": 0, "held": s["held"], "band": "BioProxal4",
                "source": "supabase", "age_s": 12.0}

    def flow_status(self):
        return {"baseline_state": "rest-frozen", "epochs": 120, "artifact_epochs": 3, "frontal_sites": "Fz,F3,F4"}


def fake_status():
    return {"board": {"kind": "cyton", "port": "COM5", "n_ch": 8, "fs": 250.0, "daisy": False, "streaming": True,
                      "packets": 1000, "dropped": 0, "last_packet_age_s": 0.01},
            "band": {"source": "supabase", "connected": True, "unit": "BioProxal4", "error": None, "seen": []},
            "recorder": {"active": False, "dir": None},
            "filter": {"notch": 60, "bp_lo": 1, "bp_hi": 50},
            "flow_index": {"baseline_state": "rest-frozen", "epochs": 120, "artifact_epochs": 3, "frontal_sites": "Fz,F3,F4"},
            "protocol": {"active": True, "name": "difficulty_ladder", "block_index": 4, "total_blocks": 12,
                         "block": {"condition": "flow", "task": "arithmetic_adaptive"}, "remaining_s": 200.0,
                         "pending_rating": None, "available": [{"name": "difficulty_ladder"}, {"name": "sim_quick"}]}}


class TestKeyAndTransport(unittest.TestCase):
    def test_find_env_walks_up_and_load_key_strips_quotes_and_space(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / ".env").write_text("OTHER=1\nOPENAI_KEY ='sk-test-123'\n", "utf-8")
            deep = root / "a" / "b"
            deep.mkdir(parents=True)
            self.assertEqual(astra.find_env(deep), root / ".env")
            self.assertEqual(astra.load_key(root / ".env"), "sk-test-123")
            (root / ".env").write_text("OPENAI_KEY=\n", "utf-8")
            with self.assertRaises(RuntimeError):
                astra.load_key(root / ".env")
            self.assertIsNone(astra.find_env(Path(d) / "a", name="NOPE_KEY"))

    def test_parse_sse_chunk(self):
        line = b'data: {"choices":[{"delta":{"content":"hel"}}]}\n'
        self.assertEqual(astra.parse_sse_chunk(line), "hel")
        self.assertEqual(astra.parse_sse_chunk('data: {"choices":[{"delta":{}}]}'), "")
        self.assertIsNone(astra.parse_sse_chunk("data: [DONE]"))
        self.assertIsNone(astra.parse_sse_chunk(": keepalive"))
        self.assertIsNone(astra.parse_sse_chunk("data: not json"))


class TestSnapshots(unittest.TestCase):
    def setUp(self):
        self.now = 1_800_000_000.0
        self.p = FakePipeline(self.now)
        self.st = fake_status()

    def test_live_snapshot_fields(self):
        s = astra.live_snapshot(self.p, self.st, 300.0, now=self.now)
        self.assertEqual(s["protocol"]["condition"], "flow")
        self.assertEqual(len(s["eeg_index_10s"]), 30)          # 150 epochs at 2 s -> 30 ten-second means
        self.assertEqual(len(s["band"]), 5)
        self.assertEqual(s["band"][4]["flags"], ["held"])
        self.assertEqual(len(s["pairs_window"]), 4)
        self.assertIsNotNone(s["r_window"])
        self.assertAlmostEqual(s["r_window"], 1.0, places=3)   # band and eeg both linear in i
        self.assertEqual(s["task_block"]["trials"], 2)
        self.assertEqual(s["task_block"]["accuracy"], 0.5)
        self.assertEqual([m["kind"] for m in s["marks"]], ["felt_flow"])   # trial marks hidden
        self.assertEqual(s["quality"][7]["state"], "railed")
        self.assertFalse(s["quality"][7]["enabled"])
        self.assertEqual(s["blocks"][0]["condition"], "boredom")
        json.dumps(s)   # serialisable

    def test_render_live_mentions_the_essentials(self):
        txt = astra.render_live(astra.live_snapshot(self.p, self.st, 300.0, now=self.now))
        for needle in ("condition=flow", "rest-frozen", "EEG flow index", "band flow", "r_window=1.00;",
                       "blocks so far", "8 P4 railed 95% OFF", "accuracy=0.50", "felt_flow=0.7", "recording=no"):
            self.assertIn(needle, txt)

    def test_live_snapshot_empty_pipeline(self):
        p = FakePipeline(self.now)
        p.eeg_hist.clear(); p.band_hist.clear(); p.pairs.clear(); p.marks.clear(); p.trials.clear()
        p.last_analysis = None
        p.band_view = lambda: None
        st = fake_status(); st["protocol"] = {"active": False, "available": []}
        s = astra.live_snapshot(p, st, 300.0, now=self.now)
        txt = astra.render_live(s)
        self.assertIn("no clean epochs", txt)
        self.assertIn("no samples in window", txt)
        self.assertIn("protocol: none running", txt)

    def test_setup_snapshot_and_render(self):
        s = astra.setup_snapshot(self.p, self.st, self.p.last_raw_peak)
        self.assertEqual(len(s["channels"]), 8)
        self.assertEqual(s["channels"][7]["raw_peak_uv"], 4000.0)
        self.assertEqual(s["montage"][0], "Fz")
        txt = astra.render_setup(s)
        self.assertIn("8 P4 railed 95.0% 1.0 4000 0.20 0.30 (disabled)", txt)
        self.assertIn("frontal sites used by the index: Fz,F3,F4", txt)
        self.assertIn("source=supabase connected=True", txt)
        self.assertIn("protocols: difficulty_ladder, sim_quick", txt)


class FakeRecorder:
    def __init__(self, d):
        self.active = True
        self.dir = d


class TestLiveAnalyst(unittest.TestCase):
    def setUp(self):
        self.now = time.time()
        self.p = FakePipeline(self.now)
        self.events = []
        self.calls = []

        def fake_chat(messages, model, max_tokens, on_delta=None):
            self.calls.append(messages)
            text = "[WATCH] theta flat in flow block; band link fine."
            if on_delta:
                on_delta(text[:7]); on_delta(text[7:])
            return text
        self.fake_chat = fake_chat

    def make(self, recorder=None, cfg=None):
        a = astra.LiveAnalyst(self.p, fake_status, cfg or {"live_interval_s": 60, "window_min": 5},
                              self.events.append, log=lambda *_: None, chat_fn=self.fake_chat, recorder=recorder)
        self.addCleanup(a.stop)
        return a

    def test_tick_emits_stream_and_remembers(self):
        a = self.make()
        note = a.tick()
        self.assertEqual(note["kind"], "note")
        self.assertEqual(note["block"], 4)
        self.assertEqual(note["condition"], "flow")
        kinds = [e["kind"] for e in self.events]
        self.assertEqual(kinds, ["note", "delta", "delta", "note"])
        self.assertFalse(self.events[0]["done"]); self.assertTrue(self.events[-1]["done"])
        self.assertEqual(self.events[-1]["text"], note["text"])
        # second tick carries the first exchange in the history
        a.tick()
        msgs = self.calls[1]
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user"); self.assertIn("LIVE WINDOW", msgs[1]["content"])
        self.assertEqual(msgs[2]["role"], "assistant"); self.assertIn("[WATCH]", msgs[2]["content"])
        self.assertEqual(len(a.notes), 2)
        self.assertEqual(a.status()["n_notes"], 2)

    def test_history_is_bounded(self):
        a = self.make(cfg={"memory_turns": 2})
        for _ in range(5):
            a.tick()
        self.assertEqual(len(a.history), 4)

    def test_analyze_and_ask(self):
        a = self.make()
        r = a.analyze()
        self.assertEqual(r["kind"], "analysis")
        self.assertIn("interim analysis", self.calls[-1][-1]["content"])
        q = a.ask("is the band lagging?")
        self.assertEqual(q["kind"], "reply")
        self.assertEqual(q["question"], "is the band lagging?")
        self.assertIn("OPERATOR QUESTION: is the band lagging?", self.calls[-1][-1]["content"])
        self.assertTrue(any(e.get("question") == "is the band lagging?" and e["done"] is False for e in self.events))

    def test_notes_written_while_recording(self):
        with tempfile.TemporaryDirectory() as d:
            a = self.make(recorder=FakeRecorder(Path(d)))
            a.tick()
            a.ask("why?")
            md = (Path(d) / "astra_notes.md").read_text("utf-8")
            self.assertTrue(md.startswith("# Astra live notes"))
            self.assertIn("[WATCH]", md)
            self.assertIn("Q&A Q: why? --", md)
            lines = (Path(d) / "astra_notes.jsonl").read_text("utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["kind"], "note")

    def test_no_file_without_recording(self):
        with tempfile.TemporaryDirectory() as d:
            rec = FakeRecorder(Path(d)); rec.active = False
            a = self.make(recorder=rec)
            a.tick()
            self.assertFalse((Path(d) / "astra_notes.md").exists())

    def test_error_is_reported_and_clears_busy(self):
        def boom(messages, model, max_tokens, on_delta=None):
            raise RuntimeError("HTTP 401: bad key")
        a = astra.LiveAnalyst(self.p, fake_status, {}, self.events.append, log=lambda *_: None, chat_fn=boom)
        self.addCleanup(a.stop)
        with self.assertRaises(RuntimeError):
            a.tick()
        self.assertFalse(a.busy)
        self.assertEqual(a.status()["error"], "HTTP 401: bad key")
        self.assertEqual(self.events[-1]["kind"], "error")

    def test_live_loop_ticks_on_interval(self):
        a = self.make(cfg={"live_interval_s": 20})
        a.set_live(True, interval_s=20)      # the minimum; last_tick reset so the first tick is immediate
        deadline = time.time() + 3
        while not a.notes and time.time() < deadline:
            time.sleep(0.05)
        self.assertEqual(len(a.notes), 1)
        a.set_live(False)
        self.assertFalse(a.status()["enabled"])


class TestSetupHelper(unittest.TestCase):
    def test_ask_and_check(self):
        p = FakePipeline(time.time())
        events, calls = [], []

        def fake_chat(messages, model, max_tokens, on_delta=None):
            calls.append(messages)
            on_delta("ch8 (P4) is railed: ")
            on_delta("check the electrode.")
            return "ch8 (P4) is railed: check the electrode."
        h = astra.SetupHelper(p, fake_status, {}, events.append, log=lambda *_: None, chat_fn=fake_chat,
                              raw_peak_fn=lambda: p.last_raw_peak, reference="REF-DOCS")
        r = h.ask("is ch8 ok?")
        self.assertIn("railed", r["text"])
        self.assertIn("REF-DOCS", calls[0][0]["content"])
        self.assertIn("SETUP STATE", calls[0][-1]["content"]); self.assertIn("QUESTION: is ch8 ok?", calls[0][-1]["content"])
        self.assertEqual([e["kind"] for e in events], ["setup", "delta", "delta", "setup"])
        self.assertEqual(events[0]["question"], "is ch8 ok?")
        self.assertEqual([m["who"] for m in h.transcript], ["user", "astra"])
        h.check()
        self.assertIn("READY or NOT READY", calls[1][-1]["content"])
        # history holds the bare question, not the state dump
        self.assertEqual(calls[1][1]["content"], "is ch8 ok?")
        self.assertEqual(h.status()["n_turns"], 2)
        h.clear()
        self.assertEqual(h.transcript, [])
        with self.assertRaises(ValueError):
            h.ask("   ")


class TestFacade(unittest.TestCase):
    def test_commands_dispatch_in_background(self):
        p = FakePipeline(time.time())
        events = []
        done = threading.Event()

        def fake_chat(messages, model, max_tokens, on_delta=None):
            done.set()
            return "[OK] fine"
        a = astra.Astra(p, fake_status, {"live_interval_s": 60}, lambda ev, d: events.append((ev, d)),
                        log=lambda *_: None, chat_fn=fake_chat, reference="")
        self.addCleanup(a.analyst.stop)
        a.available = True
        self.assertEqual(a.command("astra_analyze", {})["started"], True)
        self.assertTrue(done.wait(3))
        done.clear()
        self.assertEqual(a.command("astra_ask", {"role": "setup", "text": "where is Fz?"})["started"], True)
        self.assertTrue(done.wait(3))
        deadline = time.time() + 3
        while a.setup.busy and time.time() < deadline:
            time.sleep(0.02)
        self.assertEqual(a.setup.transcript[-1]["text"], "[OK] fine")
        self.assertEqual(a.command("astra_live", {"enabled": True, "interval_s": 45})["astra"]["live"]["interval_s"], 45.0)
        a.command("astra_live", {"enabled": False})
        self.assertEqual(a.command("astra_clear", {}), {"cleared": True})
        with self.assertRaises(ValueError):
            a.command("astra_ask", {"role": "setup", "text": ""})
        with self.assertRaises(ValueError):
            a.command("nope", {})
        self.assertTrue(all(ev == "astra" for ev, _ in events))
        st = a.state()
        self.assertIn("notes", st); self.assertIn("setup_transcript", st)

    def test_without_key_everything_but_live_off_is_refused(self):
        p = FakePipeline(time.time())
        a = astra.Astra(p, fake_status, {}, lambda ev, d: None, log=lambda *_: None, chat_fn=lambda *a, **k: "", reference="")
        self.addCleanup(a.analyst.stop)
        a.available = False
        with self.assertRaises(ValueError):
            a.command("astra_check", {})
        with self.assertRaises(ValueError):
            a.command("astra_live", {"enabled": True})
        self.assertFalse(a.command("astra_live", {"enabled": False})["astra"]["live"]["enabled"])


class TestReference(unittest.TestCase):
    def test_load_reference_has_troubleshooting_and_hardware_checks(self):
        ref = astra.load_reference()
        self.assertIn("## Troubleshooting", ref)
        self.assertIn("M-3", ref)
        self.assertLessEqual(len(ref), 16000)


if __name__ == "__main__":
    unittest.main()
