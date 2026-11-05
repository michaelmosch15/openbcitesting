"""band_link.py line parsing and the USB link's line handling, without a band: the 1-s status
line (normal, no reading, HIGH, sleep, warming), the 2-s [pub] line, [vbat], the tools-dir
finder, and BandLink._handle_line fan-out (listeners, console ring, on_line, pub de-dup)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import _path  # noqa: F401

import band_link as B

STATUS = ("[    56s] mode=LOW  phase=1 | HR= 97.9 bpm  SpO2= 89.4%  RMSSD=123.6 ms  SDNN=170.2 ms  PI=1.16%  "
          "RR=15.0 br/min* | anom=0.000 risk=0.000 flow=0.62 baseHR= 79.1 flowraw=0.59 flow_open5=0.55 flow_end5=0.40")
NO_READING = ("[    64s] mode=LOW  phase=1 | HR= 99.0 bpm  SpO2= 77.6%  RMSSD=123.6 ms  SDNN=166.0 ms  PI=2.58%  "
              "RR=15.0 br/min* | anom=0.000 risk=0.000 flow=0.00 baseHR= 79.1 flowraw=-1.00 flow_open5=0.00 flow_end5=0.00")
SETTLING = ("[    58s] mode=LOW  phase=1 | HR= 97.9 bpm  SpO2= 87.0%  RMSSD=123.6 ms  SDNN=170.2 ms  PI=1.95%  "
            "RR=15.0 br/min | anom=0.000 risk=0.000 flow=0.00 baseHR= 79.1 flowraw=0.29 flow_open5=0.55 flow_end5=0.40")
HIGH = ("[  1200s] mode=HIGH phase=2 | HR=110.2 bpm  SpO2= 95.0%  RMSSD= 20.1 ms  SDNN= 30.0 ms  PI=0.80%  "
        "RR=18.0 br/min | anom=0.712 risk=0.310 flow=0.41 baseHR= 79.1 flowraw=0.33")
SLEEP = "[  3000s] mode=LOW  phase=1 | OFF-WRIST SLEEP (probe in 30 s)"
WARM = "[     5s] mode=LOW  phase=1 | warming up  PI=0.40%"
PUB = "[pub] live  hr= 71.2 flow=0.62 anom=0.120 mlp=0.080 zrms=0.300 fhi=0.55 settling"
VBAT = "[vbat] battery 3127 mV (A0 1532 mV x 2.041), calibrated (BLE level + low-battery tiers on)"


class TestStatusLine(unittest.TestCase):
    def test_full_line(self):
        s = B.parse_status_line(STATUS, t=100.0)
        self.assertEqual(s["t"], 100.0)
        self.assertEqual((s["uptime_s"], s["mode"], s["phase"]), (56, "LOW", 1))
        self.assertEqual(s["hr"], 97.9); self.assertEqual(s["spo2"], 89.4)
        self.assertEqual(s["rmssd"], 123.6); self.assertEqual(s["sdnn"], 170.2)
        self.assertEqual(s["pi"], 1.16); self.assertEqual(s["rr"], 15.0); self.assertTrue(s["rr_modeled"])
        self.assertEqual(s["anom"], 0.0); self.assertEqual(s["risk"], 0.0)
        self.assertEqual(s["flow"], 0.62); self.assertEqual(s["flow_raw"], 0.59); self.assertEqual(s["base_hr"], 79.1)
        self.assertEqual(s["flow_open5"], 0.55); self.assertEqual(s["flow_end5"], 0.40)
        self.assertFalse(s["held"]); self.assertFalse(s["settling"]); self.assertFalse(s["asleep"])
        self.assertEqual(s["source_line"], "status")

    def test_no_reading_and_settling(self):
        s = B.parse_status_line(NO_READING)
        self.assertIsNone(s["flow"]); self.assertEqual(s["flow_raw"], -1.0); self.assertFalse(s["settling"])
        s = B.parse_status_line(SETTLING)
        self.assertIsNone(s["flow"]); self.assertEqual(s["flow_raw"], 0.29); self.assertTrue(s["settling"])
        self.assertFalse(s["rr_modeled"])

    def test_high_mode_is_held(self):
        s = B.parse_status_line(HIGH)
        self.assertEqual(s["mode"], "HIGH"); self.assertTrue(s["held"]); self.assertEqual(s["flow"], 0.41)
        self.assertEqual(s["anom"], 0.712); self.assertEqual(s["risk"], 0.31); self.assertIsNone(s["flow_open5"])

    def test_sleep_and_warming(self):
        s = B.parse_status_line(SLEEP)
        self.assertTrue(s["asleep"]); self.assertIsNone(s["hr"]); self.assertIsNone(s["flow"])
        w = B.parse_status_line(WARM)
        self.assertTrue(w["warming"]); self.assertEqual(w["pi"], 0.4); self.assertIsNone(w["hr"]); self.assertIsNone(w["flow"])

    def test_other_lines(self):
        for ln in (VBAT, "[fifo] polls=1 ring_would_drop=0", "", "[ 64495][I][orchestrator.cpp:2291] step(): [orch] anomaly: NO SCORE"):
            self.assertIsNone(B.parse_status_line(ln))


class TestPubLine(unittest.TestCase):
    def test_pub(self):
        s = B.parse_pub_line(PUB, t=5.0)
        self.assertEqual(s["source_line"], "pub"); self.assertEqual(s["hr"], 71.2); self.assertEqual(s["flow"], 0.62)
        self.assertEqual(s["mlp"], 0.08); self.assertEqual(s["zrms"], 0.3); self.assertEqual(s["fhi"], 0.55)
        self.assertTrue(s["settling"]); self.assertFalse(s["held"])
        self.assertTrue(B.parse_pub_line("[pub] HELD  hr= 70.0 flow=0.50 anom=0.100")["held"])
        self.assertIsNone(B.parse_pub_line(STATUS))


class TestToolsDir(unittest.TestCase):
    def test_configured_and_walk_up(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            proj = root / "a" / "b"
            proj.mkdir(parents=True)
            self.assertIsNone(B.find_tools_dir(None, start=proj))
            tools = root / "worktrees" / "v0629" / "tools"
            tools.mkdir(parents=True)
            (tools / "offload.py").write_text("# stub\n", "utf-8")
            self.assertEqual(B.find_tools_dir(None, start=proj), tools)
            self.assertEqual(B.find_tools_dir(str(tools), start=proj), tools)
            self.assertEqual(B.find_tools_dir("../../worktrees/v0629/tools", start=proj), tools)
            self.assertEqual(B.find_tools_dir("nope/tools", start=proj), tools)      # bad config -> walk-up


class TestV2Row(unittest.TestCase):
    def test_decode_row(self):
        import struct
        row = struct.pack("<IhhhhhhHHHHBBBBHBB", 1234, 865, 970, 500, 420, 150, 123, 6641, 0, 4200, 3552,
                          0, 1, 0x01, 7, 1000, 0x80 | 60, 1 << 5)
        s = B.decode_v2_row(row, t=10.0)
        self.assertEqual(s["uptime_s"], 1234); self.assertEqual(s["hr"], 86.5); self.assertEqual(s["spo2"], 97.0)
        self.assertEqual(s["sdnn"], 50.0); self.assertEqual(s["rmssd"], 42.0); self.assertEqual(s["rr"], 15.0)
        self.assertEqual(s["pi"], 1.23); self.assertEqual(s["anom"], 0.6641); self.assertEqual(s["flow"], 0.42)
        self.assertEqual(s["battery_mv"], 3552); self.assertEqual(s["mode"], "LOW"); self.assertFalse(s["held"])
        self.assertTrue(s["hrv_ok"]); self.assertEqual(s["ir_dc"], 8000); self.assertEqual(s["led"], 60); self.assertEqual(s["rate_hz"], 100)
        row2 = struct.pack("<IhhhhhhHHHHBBBBHBB", 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3000, 1, 2, 0x20 | 0x08 | 0x04, 0, 0, 0, 0)
        s2 = B.decode_v2_row(row2)
        self.assertIsNone(s2["flow"]); self.assertIsNone(s2["hr"]); self.assertEqual(s2["mode"], "HIGH"); self.assertTrue(s2["held"])
        self.assertTrue(s2["asleep"]); self.assertTrue(s2["poor_contact"]); self.assertEqual(s2["rate_hz"], 25)

    def test_find_tools_dir_v2(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            proj = root / "x" / "y"; proj.mkdir(parents=True)
            self.assertIsNone(B.find_tools_dir_v2(None, start=proj))
            t = root / "firmware_v2" / ".claude" / "worktrees" / "v2-build" / "tools"; t.mkdir(parents=True)
            (t / "live_usb.py").write_text("#", "utf-8"); (t / "offload.py").write_text("#", "utf-8")
            self.assertEqual(B.find_tools_dir_v2(None, start=proj), t)
            self.assertEqual(B.find_tools_dir_v2(str(t), start=proj), t)


class TestHandleLine(unittest.TestCase):
    def test_fanout_console_and_dedup(self):
        got, lines = [], []
        link = B.BandLink(None, log=lambda *_: None, on_line=lambda t, ln: lines.append(ln))
        link.listeners.append(got.append)
        link._handle_line(VBAT)
        self.assertEqual(link.state["battery_mv"], 3127); self.assertEqual(got, [])
        link._handle_line(STATUS)
        self.assertEqual(len(got), 1); self.assertEqual(got[0]["battery_mv"], 3127); self.assertEqual(got[0]["flow"], 0.62)
        link._handle_line(PUB)                      # within 0.5 s of the status sample: dropped
        self.assertEqual(len(got), 1)
        link._last_sample_t -= 5
        link._handle_line(PUB)                      # alone: kept
        self.assertEqual(len(got), 2); self.assertEqual(got[1]["source_line"], "pub")
        link._handle_line("[fifo] polls=1")
        self.assertEqual(len(got), 2)
        self.assertEqual(lines, [VBAT, STATUS, PUB, PUB, "[fifo] polls=1"])
        self.assertEqual([ln for _, ln in link.console], lines)
        tail = link.console_tail(2)
        self.assertEqual(tail[-1][1], "[fifo] polls=1")
        st = link.status()
        self.assertEqual(st["lines"], 5); self.assertIsNotNone(st["last_line_age_s"]); self.assertEqual(st["transport"], "usb")

    def test_listener_error_is_contained(self):
        link = B.BandLink(None, log=lambda *_: None)
        link.listeners.append(lambda s: 1 / 0)
        link._handle_line(STATUS)          # must not raise
        self.assertIsNotNone(link.latest)


if __name__ == "__main__":
    unittest.main()
