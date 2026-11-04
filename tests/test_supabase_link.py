"""supabase_link: timestamp parsing, row decoding, wearer grouping and the polling link
(all with a fake client - nothing touches the network)."""
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

import _path  # noqa: F401
import supabase_link
from supabase_link import (SupabaseBandLink, SupabaseClient, epoch_to_iso, iso_to_epoch,
                           load_env, row_to_sample)

T0 = datetime(2026, 11, 3, 14, 2, 11, tzinfo=timezone.utc).timestamp()


def row(sampled_at, created_at=None, user_id="54af48b8-1111-2222-3333-444444444444",
        band="BioProxal4", flow=0.62, mode=0, fs=3, **extra):
    r = {"user_id": user_id, "sampled_at": sampled_at, "created_at": created_at or sampled_at,
         "flow_score": flow, "flow_state": fs, "anomaly": 0.12, "risk": 0.03, "mode": mode,
         "phase": 2, "hrv_valid": True, "poor_contact": False, "asleep": False, "usable": True,
         "battery_mv": 3900, "uptime_s": 1234, "band": band}
    r.update(extra)
    return r


class TestIso(unittest.TestCase):
    def test_fractions_and_z(self):
        self.assertAlmostEqual(iso_to_epoch("2026-11-03T14:02:11Z"), T0)
        self.assertAlmostEqual(iso_to_epoch("2026-11-03T14:02:11.123Z"), T0 + 0.123, places=6)
        self.assertAlmostEqual(iso_to_epoch("2026-11-03T14:02:11.123456Z"), T0 + 0.123456, places=6)
        self.assertAlmostEqual(iso_to_epoch("2026-11-03T14:02:11.5+00:00"), T0 + 0.5, places=6)
        self.assertAlmostEqual(iso_to_epoch("2026-11-03T15:02:11+01:00"), T0)

    def test_round_trip(self):
        s = epoch_to_iso(T0 + 0.25)
        self.assertTrue(s.endswith("Z"))
        self.assertAlmostEqual(iso_to_epoch(s), T0 + 0.25, places=6)


class TestRowToSample(unittest.TestCase):
    def test_realistic_row(self):
        s = row_to_sample(row("2026-11-03T14:02:11.123456+00:00", "2026-11-03T14:02:40.5+00:00"))
        self.assertAlmostEqual(s["t"], T0 + 0.123456, places=5)
        self.assertAlmostEqual(s["t_created"], T0 + 29.5, places=5)
        self.assertAlmostEqual(s["flow"], 0.62)
        self.assertEqual(s["flow_state"], 3)
        self.assertEqual(s["window"], "open")
        self.assertIsNone(s["hr"])
        self.assertAlmostEqual(s["anom"], 0.12)
        self.assertAlmostEqual(s["risk"], 0.03)
        self.assertEqual(s["mode"], 0)
        self.assertFalse(s["held"])
        self.assertFalse(s["settling"])
        self.assertFalse(s["asleep"])
        self.assertFalse(s["poor_contact"])
        self.assertEqual(s["band"], "BioProxal4")
        self.assertEqual(s["user_id"][:8], "54af48b8")
        self.assertEqual(s["battery_mv"], 3900)
        for k in ("fhi", "fcal", "mlp", "zrms"):
            self.assertIsNone(s[k])

    def test_nulls_and_held(self):
        s = row_to_sample(row("2026-11-03T14:02:11Z", flow=None, mode=1, fs=None, anomaly=None,
                              risk=None, created_at=None))
        self.assertIsNone(s["flow"])
        self.assertTrue(s["held"])
        self.assertEqual(s["flow_state"], 0)
        self.assertEqual(s["window"], "unavailable")
        self.assertIsNone(s["anom"])
        self.assertIsNone(s["risk"])
        self.assertAlmostEqual(s["t_created"], T0)      # created_at defaults to sampled_at in row()
        s2 = row_to_sample({"sampled_at": "2026-11-03T14:02:11Z"})
        self.assertIsNone(s2["t_created"])
        self.assertIsNone(s2["flow"])
        self.assertIsNone(s2["band"])


class TestWearers(unittest.TestCase):
    def test_grouping_order_and_rows_last_hour(self):
        now = time.time()
        a = "54af48b8-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        b = "220929c4-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
        rows = [row(epoch_to_iso(now - 50), user_id=b, band="BioProxal7"),
                row(epoch_to_iso(now - 10), user_id=a),
                row(epoch_to_iso(now - 100), user_id=a),
                row(epoch_to_iso(now - 5000), user_id=a),
                row(epoch_to_iso(now - 7000), user_id=a, band=None)]
        rows.sort(key=lambda r: r["sampled_at"], reverse=True)       # as the REST query orders
        calls = []
        client = SupabaseClient("https://x.supabase.co", "key")
        client.get = lambda table, params: calls.append((table, params)) or rows
        out = client.wearers(hours=24)
        self.assertEqual(calls[0][0], "wearable_samples")
        self.assertEqual(calls[0][1]["order"], "sampled_at.desc")
        self.assertTrue(calls[0][1]["sampled_at"].startswith("gte."))
        since = iso_to_epoch(calls[0][1]["sampled_at"][4:])
        self.assertAlmostEqual(now - since, 24 * 3600, delta=5)
        self.assertEqual([(w["short"], w["band"]) for w in out],
                         [("54af48b8", "BioProxal4"), ("220929c4", "BioProxal7"), ("54af48b8", None)])
        self.assertEqual(out[0]["rows_last_hour"], 2)
        self.assertEqual(out[1]["rows_last_hour"], 1)
        self.assertEqual(out[2]["rows_last_hour"], 0)
        self.assertEqual(out[0]["user_id"], a)
        self.assertAlmostEqual(out[0]["age_s"], 10, delta=3)
        self.assertAlmostEqual(out[1]["age_s"], 50, delta=3)
        self.assertEqual(out[0]["last_sampled_at"], rows[0]["sampled_at"])

    def test_no_rows(self):
        client = SupabaseClient("https://x.supabase.co", "key")
        client.get = lambda table, params: []
        self.assertEqual(client.wearers(), [])


class FakeClient:
    """Returns the canned rows whose created_at is > the link's filter, in created_at order,
    and stops the link after `stop_after` calls so `_run` returns on its own."""

    def __init__(self, rows, link, stop_after=2, fail_on=()):
        self.rows = sorted(rows, key=lambda r: r["created_at"])
        self.link = link
        self.stop_after = stop_after
        self.fail_on = set(fail_on)
        self.calls = []

    def get(self, table, params):
        self.calls.append((table, dict(params)))
        n = len(self.calls)
        if n >= self.stop_after:
            self.link._stop.set()
        if n in self.fail_on:
            raise OSError("network down")
        since = params["created_at"][3:]
        out = [r for r in self.rows if iso_to_epoch(r["created_at"]) > iso_to_epoch(since)]
        return out[:int(params.get("limit", 500))]


class TestSupabaseBandLink(unittest.TestCase):
    def test_backfill_then_incremental(self):
        now = time.time()
        link = SupabaseBandLink(user_id="54af48b8-aaaa-aaaa-aaaa-aaaaaaaaaaaa", band="BioProxal4",
                                poll_s=0.01, backfill_min=30, log=lambda *a: None)
        rows = [row(epoch_to_iso(now - 600), epoch_to_iso(now - 570), flow=0.4),
                row(epoch_to_iso(now - 540), epoch_to_iso(now - 510), flow=0.5),
                row(epoch_to_iso(now - 3600), epoch_to_iso(now - 3570), flow=0.9),   # before backfill
                row(epoch_to_iso(now - 60), epoch_to_iso(now - 30), flow=0.6)]
        client = FakeClient(rows[:3], link, stop_after=2)
        link.client = client
        got = []
        link.listeners.append(got.append)

        # first poll returns rows[0], rows[1]; then "new" rows appear before the second poll
        orig_get = client.get

        def get(table, params):
            res = orig_get(table, params)
            if len(client.calls) == 1:
                client.rows.append(rows[3])
            return res
        client.get = get

        link._run()                                     # synchronous: stops after 2 polls
        self.assertEqual(len(client.calls), 2)
        p1, p2 = client.calls[0][1], client.calls[1][1]
        self.assertEqual(client.calls[0][0], "wearable_samples")
        self.assertEqual(p1["user_id"], "eq.54af48b8-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
        self.assertEqual(p1["band"], "eq.BioProxal4")
        self.assertEqual(p1["order"], "created_at.asc")
        self.assertTrue(p1["created_at"].startswith("gt."))
        self.assertAlmostEqual(now - iso_to_epoch(p1["created_at"][3:]), 1800, delta=5)
        # second poll asks only for rows created after the last one seen
        self.assertEqual(p2["created_at"], "gt." + rows[1]["created_at"])
        self.assertEqual([s["flow"] for s in got], [0.4, 0.5, 0.6])
        self.assertTrue(all(s["source"] == "supabase" for s in got))
        self.assertEqual(link._last_created, rows[3]["created_at"])
        self.assertEqual(link.latest["flow"], 0.6)
        self.assertEqual(link.state["lines"], 3)
        self.assertEqual(link.state["unit"], "BioProxal4")
        self.assertEqual(link.state["user_id_short"], "54af48b8")
        self.assertIsNone(link.state["error"])
        self.assertFalse(link.state["connected"])       # cleared on exit
        st = link.status()
        self.assertEqual(st["transport"], "supabase")
        self.assertIsNotNone(st["last_line_age_s"])

    def test_listener_in_thread(self):
        now = time.time()
        link = SupabaseBandLink(poll_s=0.01, log=lambda *a: None)
        link.client = FakeClient([row(epoch_to_iso(now - 60))], link, stop_after=3)
        got = []
        link.listeners.append(got.append)
        link.start()
        link._thread.join(2.0)
        self.assertFalse(link._thread.is_alive())
        self.assertEqual(len(got), 1)
        self.assertEqual(len(link.client.calls), 3)

    def test_error_sets_state(self):
        link = SupabaseBandLink(poll_s=0.01, log=lambda *a: None)
        link.client = FakeClient([], link, stop_after=1, fail_on={1})
        link._run()
        self.assertIn("network down", link.state["error"] or "")
        self.assertFalse(link.state["connected"])

    def test_error_then_recovery(self):
        link = SupabaseBandLink(poll_s=0.01, log=lambda *a: None)
        link.client = FakeClient([], link, stop_after=2, fail_on={1})
        link._run()
        self.assertEqual(len(link.client.calls), 2)       # kept polling after the failure
        self.assertIsNone(link.state["error"])            # cleared by the successful poll

    def test_filters_without_user_or_band(self):
        link = SupabaseBandLink()
        p = link._filters()
        self.assertNotIn("user_id", p)
        self.assertNotIn("band", p)
        self.assertEqual(p["select"], "*")
        self.assertIsNone(link.state["user_id_short"])

    def test_listener_exception_is_contained(self):
        now = time.time()
        logs = []
        link = SupabaseBandLink(poll_s=0.01, log=logs.append)
        link.client = FakeClient([row(epoch_to_iso(now - 60))], link, stop_after=1)
        link.listeners.append(lambda s: 1 / 0)
        link._run()
        self.assertEqual(link.state["lines"], 1)
        self.assertTrue(any("listener error" in m for m in logs))


class TestLoadEnv(unittest.TestCase):
    def test_parse(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "supabase.env"
            p.write_text('# creds\nSUPABASE_URL="https://x.supabase.co/"\nSUPABASE_PUBLISHABLE_KEY=pk\n', "utf-8")
            env = load_env(p)
            self.assertEqual(env, {"url": "https://x.supabase.co", "key": "pk"})
            p.write_text("SUPABASE_URL=https://x\nSUPABASE_SECRET_KEY='sk'\nSUPABASE_PUBLISHABLE_KEY=pk\n", "utf-8")
            self.assertEqual(load_env(p)["key"], "sk")
            p.write_text("SUPABASE_URL=https://x\n", "utf-8")
            with self.assertRaises(RuntimeError):
                load_env(p)
            with self.assertRaises(RuntimeError):
                load_env(Path(d) / "missing.env")
        self.assertTrue(str(supabase_link.ENV_PATH).endswith("supabase.env"))


if __name__ == "__main__":
    unittest.main()
