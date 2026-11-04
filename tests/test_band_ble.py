"""decode_state on hand-packed 28/29-byte STATE frames."""
import struct
import unittest

import _path  # noqa: F401
from band_ble import decode_state

HEAD = "<4B6h3HIH"          # version mode phase flags | hr spo2 sdnn rmssd rr pi | anom risk batt | up | flow_q


def frame(version=3, mode=0, phase=2, flags=0x30, hr=712, spo2=975, sdnn=420, rmssd=380, rr=150,
          pi=250, anom=1200, risk=300, batt=3900, up=12345, flow_q=6200, learn=45):
    data = struct.pack(HEAD, version, mode, phase, flags, hr, spo2, sdnn, rmssd, rr, pi,
                       anom, risk, batt, up, flow_q)
    assert len(data) == 28
    if learn is not None:
        data += bytes([learn])
    return data


class TestDecodeState(unittest.TestCase):
    def test_live_frame(self):
        d = decode_state(frame())
        self.assertEqual(d["version"], 3)
        self.assertEqual(d["mode"], 0)
        self.assertEqual(d["phase"], 2)
        self.assertEqual(d["flags"], 0x30)
        self.assertAlmostEqual(d["hr"], 71.2)
        self.assertAlmostEqual(d["spo2"], 97.5)
        self.assertAlmostEqual(d["sdnn"], 42.0)
        self.assertAlmostEqual(d["rmssd"], 38.0)
        self.assertAlmostEqual(d["resp"], 15.0)
        self.assertAlmostEqual(d["pi"], 2.5)
        self.assertAlmostEqual(d["anom"], 0.12)
        self.assertAlmostEqual(d["risk"], 0.03)
        self.assertEqual(d["battery_mv"], 3900)
        self.assertEqual(d["uptime_s"], 12345)
        self.assertAlmostEqual(d["flow"], 0.62)
        self.assertFalse(d["flow_no_reading"])
        self.assertFalse(d["held"])
        self.assertFalse(d["settling"])
        self.assertFalse(d["asleep"])
        self.assertFalse(d["poor_contact"])
        self.assertFalse(d["held_features"])
        self.assertEqual(d["window"], "open")
        self.assertEqual(d["learn_pct"], 45)
        self.assertFalse(d["flow_agent_only"])
        self.assertFalse(d["low_battery"])
        self.assertGreater(d["t"], 0)
        for k in ("fhi", "fcal", "mlp", "zrms"):
            self.assertIsNone(d[k])

    def test_bit7_no_reading_gives_flow_none_and_settling(self):
        d = decode_state(frame(flags=0x80 | 0x30, flow_q=5000))
        self.assertIsNone(d["flow"])
        self.assertTrue(d["flow_no_reading"])
        self.assertTrue(d["settling"])          # mode 0, hr > 0, not poor contact
        self.assertFalse(d["held"])

    def test_no_reading_with_poor_contact_is_not_settling(self):
        d = decode_state(frame(flags=0x80 | 0x08 | 0x30))
        self.assertIsNone(d["flow"])
        self.assertTrue(d["poor_contact"])
        self.assertFalse(d["settling"])

    def test_no_reading_without_hr_is_not_settling(self):
        d = decode_state(frame(flags=0x80 | 0x30, hr=0))
        self.assertFalse(d["settling"])

    def test_mode1_is_held(self):
        d = decode_state(frame(mode=1))
        self.assertTrue(d["held"])
        self.assertAlmostEqual(d["flow"], 0.62)
        self.assertFalse(d["settling"])

    def test_mode1_with_no_reading_is_not_held(self):
        d = decode_state(frame(mode=1, flags=0x80 | 0x30))
        self.assertFalse(d["held"])
        self.assertIsNone(d["flow"])

    def test_low_risk_window_unavailable_is_held(self):
        d = decode_state(frame(mode=0, flags=0x00))
        self.assertEqual(d["window"], "unavailable")
        self.assertTrue(d["held"])

    def test_window_states(self):
        for code, name in {0: "unavailable", 1: "closed", 2: "opening", 3: "open", 4: "ending"}.items():
            self.assertEqual(decode_state(frame(flags=code << 4))["window"], name)
        self.assertEqual(decode_state(frame(flags=7 << 4))["window"], "7")

    def test_flag_bits(self):
        d = decode_state(frame(flags=0x02 | 0x04 | 0x08 | 0x30))
        self.assertTrue(d["held_features"])
        self.assertTrue(d["asleep"])
        self.assertTrue(d["poor_contact"])

    def test_28_byte_frame_has_no_learn(self):
        data = frame(learn=None)
        self.assertEqual(len(data), 28)
        d = decode_state(data)
        self.assertIsNone(d["learn_pct"])
        self.assertIsNone(d["flow_agent_only"])
        self.assertFalse(d["low_battery"])
        self.assertAlmostEqual(d["flow"], 0.62)

    def test_learn_byte_variants(self):
        d = decode_state(frame(learn=0x80 | 30))
        self.assertEqual(d["learn_pct"], 30)
        self.assertTrue(d["flow_agent_only"])
        d = decode_state(frame(learn=126))
        self.assertIsNone(d["learn_pct"])       # > 100 is a status code, not a percentage
        self.assertTrue(d["low_battery"])
        d = decode_state(frame(learn=100))
        self.assertEqual(d["learn_pct"], 100)
        self.assertFalse(d["low_battery"])

    def test_negative_signed_fields(self):
        d = decode_state(frame(hr=-10, sdnn=-5))
        self.assertAlmostEqual(d["hr"], -1.0)
        self.assertAlmostEqual(d["sdnn"], -0.5)

    def test_flow_full_scale(self):
        self.assertAlmostEqual(decode_state(frame(flow_q=10000))["flow"], 1.0)
        self.assertAlmostEqual(decode_state(frame(flow_q=0))["flow"], 0.0)

    def test_too_short_raises(self):
        with self.assertRaises(ValueError):
            decode_state(frame()[:27])
        with self.assertRaises(ValueError):
            decode_state(b"")

    def test_longer_frame_is_tolerated(self):
        d = decode_state(frame() + b"\x00\x00")
        self.assertEqual(d["learn_pct"], 45)


if __name__ == "__main__":
    unittest.main()
