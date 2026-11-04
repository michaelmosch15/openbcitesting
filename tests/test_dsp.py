"""Filters, PSD, band powers and the railed/flat quality check."""
import unittest

import numpy as np

import _path  # noqa: F401
import dsp
from dsp import BANDS, Biquad, FilterChain, band_powers, psd, quality, relative_powers

FS = 250.0


def sine(f_hz, seconds, fs=FS, amp=1.0):
    t = np.arange(int(seconds * fs)) / fs
    return (amp * np.sin(2 * np.pi * f_hz * t))[:, None]


def gain_db(chain, f_hz, seconds, settle_s):
    x = sine(f_hz, seconds)
    y = chain.process(x)
    tail = int(settle_s * FS)
    rms_in = np.sqrt((x[tail:] ** 2).mean())
    rms_out = np.sqrt((y[tail:] ** 2).mean())
    return 20 * np.log10(rms_out / rms_in)


class TestNotch(unittest.TestCase):
    def test_60hz_attenuated_more_than_20db(self):
        chain = FilterChain(FS, 1, notch_hz=60, bp_lo=0, bp_hi=0)
        self.assertLess(gain_db(chain, 60, 10, 8), -20)

    def test_10hz_passes_within_1db(self):
        chain = FilterChain(FS, 1, notch_hz=60, bp_lo=0, bp_hi=0)
        self.assertLess(abs(gain_db(chain, 10, 10, 8)), 1.0)

    def test_single_biquad_notch(self):
        bq = Biquad.notch(60, FS, 1)
        y = bq.process(sine(60, 10))
        self.assertLess(np.abs(y[-500:]).max(), 0.1)

    def test_notch_off_has_no_notch_stages(self):
        chain = FilterChain(FS, 1, notch_hz=0, bp_lo=0, bp_hi=0)
        self.assertEqual(chain.stages, [])


class TestBandPass(unittest.TestCase):
    def test_passes_10hz(self):
        chain = FilterChain(FS, 1, notch_hz=0, bp_lo=1, bp_hi=50)
        self.assertLess(abs(gain_db(chain, 10, 20, 10)), 1.0)

    def test_attenuates_0_2hz(self):
        chain = FilterChain(FS, 1, notch_hz=0, bp_lo=1, bp_hi=50)
        self.assertLess(gain_db(chain, 0.2, 30, 20), -20)

    def test_attenuates_100hz(self):
        chain = FilterChain(FS, 1, notch_hz=0, bp_lo=1, bp_hi=50)
        self.assertLess(gain_db(chain, 100, 10, 5), -20)

    def test_state_persists_between_calls(self):
        """Processing in two chunks must equal processing in one go."""
        x = sine(10, 4) + sine(60, 4, amp=0.3)
        a = FilterChain(FS, 1).process(x)
        c = FilterChain(FS, 1)
        b = np.vstack([c.process(x[:300]), c.process(x[300:])])
        np.testing.assert_allclose(a, b, atol=1e-9)

    def test_multi_channel_independent(self):
        x = np.hstack([sine(10, 4), sine(60, 4)])
        y = FilterChain(FS, 2, notch_hz=60, bp_lo=0, bp_hi=0).process(x)
        self.assertGreater(np.abs(y[-250:, 0]).max(), 0.9)
        self.assertLess(np.abs(y[-250:, 1]).max(), 0.05)


class TestPsd(unittest.TestCase):
    def test_peak_at_10hz(self):
        f, P = psd(sine(10, 2, amp=20), FS)
        self.assertEqual(P.shape[1], 1)
        self.assertAlmostEqual(f[P[:, 0].argmax()], 10.0)
        self.assertAlmostEqual(f[1] - f[0], 0.5)

    def test_integral_matches_variance(self):
        x = sine(10, 2, amp=20)
        f, P = psd(x, FS)
        self.assertAlmostEqual(P[:, 0].sum() * (f[1] - f[0]), x.var(), delta=0.02 * x.var())

    def test_short_window_returns_empty(self):
        f, P = psd(np.zeros((4, 3)), FS)
        self.assertEqual(len(f), 0)
        self.assertEqual(P.shape, (0, 3))


class TestBandPowers(unittest.TestCase):
    def test_flat_psd_integrates_bandwidths(self):
        f = np.arange(0, 125.5, 0.5)
        P = np.ones((len(f), 2))
        bp = band_powers(f, P)
        for name, (lo, hi) in BANDS.items():
            np.testing.assert_allclose(bp[name], hi - lo)

    def test_relative_sums_to_one(self):
        f = np.arange(0, 125.5, 0.5)
        rng = np.random.default_rng(0)
        P = rng.uniform(0.1, 5, (len(f), 4))
        rel = relative_powers(band_powers(f, P))
        total = sum(rel[b] for b in BANDS)
        np.testing.assert_allclose(total, 1.0, atol=1e-6)
        self.assertEqual(list(rel), dsp.BAND_ORDER)

    def test_empty_psd(self):
        bp = band_powers(np.zeros(0), np.zeros((0, 3)))
        for b in BANDS:
            self.assertEqual(bp[b].shape, (3,))
        rel = relative_powers(bp)
        for b in BANDS:
            np.testing.assert_allclose(rel[b], 0.0)


class TestQuality(unittest.TestCase):
    FULL = 187500.0

    def test_states(self):
        n = 500
        railed = np.full(n, 0.95 * self.FULL)
        near = np.full(n, 0.80 * self.FULL) * np.sign(np.sin(np.arange(n)))
        flat = np.zeros(n)
        good = 20 * np.sin(2 * np.pi * 10 * np.arange(n) / FS)
        q = quality(np.stack([railed, near, flat, good], axis=1), self.FULL)
        self.assertEqual([c["state"] for c in q], ["railed", "near", "flat", "good"])
        self.assertAlmostEqual(q[0]["pct"], 95.0, places=1)
        self.assertAlmostEqual(q[3]["std"], 20 / np.sqrt(2), delta=0.1)

    def test_railed_boundary_at_90pct(self):
        x = np.full((10, 2), 0.90 * self.FULL)
        x[:, 1] = 0.899 * self.FULL
        q = quality(x, self.FULL)
        self.assertEqual(q[0]["state"], "railed")
        self.assertEqual(q[1]["state"], "near")

    def test_negative_rail(self):
        q = quality(np.full((10, 1), -0.99 * self.FULL), self.FULL)
        self.assertEqual(q[0]["state"], "railed")

    def test_empty_window_unknown(self):
        q = quality(np.zeros((0, 2)), self.FULL)
        self.assertEqual([c["state"] for c in q], ["unknown", "unknown"])


if __name__ == "__main__":
    unittest.main()
