"""FlowIndex: direction of response, warm-up, channel exclusion."""
import unittest

import numpy as np

import _path  # noqa: F401
from dsp import BAND_ORDER
from flow_index import FlowIndex, RollingZ, sigmoid

LABELS = ["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"]
N = len(LABELS)
FRONTAL = [0, 1]
POSTERIOR = [4, 5, 6, 7]


def epoch(theta=0.2, alpha=0.3, beta=0.2, delta=0.2, gamma=0.1, **per_channel):
    """Relative powers (constant across channels unless overridden per band with an
    8-vector) and matching absolute powers (relative x 100 uV^2)."""
    rel = {"delta": delta, "theta": theta, "alpha": alpha, "beta": beta, "gamma": gamma}
    out = {}
    for b in BAND_ORDER:
        v = per_channel.get(b, rel[b])
        out[b] = np.full(N, float(v)) if np.isscalar(v) else np.asarray(v, float)
    abs_pow = {b: out[b] * 100.0 for b in BAND_ORDER}
    return out, abs_pow


def feed_constant(fi, n=40, good=None, **kw):
    good = good or [True] * N
    rec = None
    for _ in range(n):
        rel, ab = epoch(**kw)
        rec = fi.update(rel, ab, good)
    return rec


class TestRollingZ(unittest.TestCase):
    def test_warmup_and_z(self):
        z = RollingZ(n=10, warmup=3)
        self.assertEqual(z.push(1.0), 0.0)
        self.assertFalse(z.ready)
        self.assertEqual(z.push(1.0), 0.0)
        self.assertTrue(z.push(1.0) == 0.0)      # constant -> sd 0 -> z 0
        self.assertTrue(z.ready)
        self.assertGreater(z.push(5.0), 1.0)
        self.assertLess(z.push(-5.0), -1.0)


class TestFlowIndex(unittest.TestCase):
    def test_baseline_ready_after_warmup(self):
        fi = FlowIndex(LABELS, warmup_epochs=15)
        good = [True] * N
        for i in range(1, 30):
            rel, ab = epoch()
            rec = fi.update(rel, ab, good)
            self.assertEqual(rec["baseline_ready"], i >= 15, f"epoch {i}")

    def test_constant_input_gives_neutral_flow(self):
        fi = FlowIndex(LABELS)
        rec = feed_constant(fi)
        # z = 0 everywhere -> x = 0 -> sigmoid(0) = 0.5 (neutral)
        self.assertAlmostEqual(rec["flow_eeg"], sigmoid(0.0), places=3)
        self.assertAlmostEqual(rec["flow_eeg"], 0.5, places=3)
        self.assertEqual(rec["z_theta"], 0.0)
        self.assertEqual(rec["n_good"], N)
        self.assertAlmostEqual(rec["theta_frontal_rel"], 0.2, places=4)
        self.assertAlmostEqual(rec["alpha_posterior_rel"], 0.3, places=4)
        self.assertAlmostEqual(rec["beta_rel"], 0.2, places=4)
        for b in BAND_ORDER:
            self.assertIn(f"{b}_rel_mean", rec)

    def test_frontal_theta_up_raises_flow(self):
        fi = FlowIndex(LABELS)
        base = feed_constant(fi)
        theta = np.full(N, 0.2)
        theta[FRONTAL] = 0.4
        rel, ab = epoch(theta=theta)
        rec = fi.update(rel, ab, [True] * N)
        self.assertGreater(rec["z_theta"], 1.0)
        self.assertGreater(rec["flow_eeg"], base["flow_eeg"])
        self.assertAlmostEqual(rec["theta_frontal_rel"], 0.4, places=4)

    def test_posterior_only_theta_does_not_raise_frontal_theta(self):
        fi = FlowIndex(LABELS)
        feed_constant(fi)
        theta = np.full(N, 0.2)
        theta[POSTERIOR] = 0.5
        rel, ab = epoch(theta=theta)
        rec = fi.update(rel, ab, [True] * N)
        self.assertAlmostEqual(rec["theta_frontal_rel"], 0.2, places=4)
        self.assertEqual(rec["z_theta"], 0.0)

    def test_beta_up_lowers_flow(self):
        fi = FlowIndex(LABELS)
        base = feed_constant(fi)
        rel, ab = epoch(beta=0.5)
        rec = fi.update(rel, ab, [True] * N)
        self.assertGreater(rec["z_beta"], 1.0)
        self.assertLess(rec["flow_eeg"], base["flow_eeg"])

    def test_alpha_deviation_either_way_lowers_flow(self):
        for alpha in (0.6, 0.05):
            fi = FlowIndex(LABELS)
            base = feed_constant(fi)
            rel, ab = epoch(alpha=alpha)
            rec = fi.update(rel, ab, [True] * N)
            self.assertLess(rec["flow_eeg"], base["flow_eeg"], f"alpha {alpha}")

    def test_weights_change_response(self):
        fi0 = FlowIndex(LABELS, weights={"theta": 0.0, "alpha": 0.0, "beta": 0.0})
        feed_constant(fi0)
        theta = np.full(N, 0.2)
        theta[FRONTAL] = 0.4
        rel, ab = epoch(theta=theta)
        self.assertAlmostEqual(fi0.update(rel, ab, [True] * N)["flow_eeg"], 0.5, places=3)

    def test_disabled_channel_excluded(self):
        fi = FlowIndex(LABELS)
        good = [True] * N
        good[0] = False                     # Fp1 off
        theta = np.full(N, 0.2)
        theta[0], theta[1] = 0.9, 0.1
        rel, ab = epoch(theta=theta)
        rec = fi.update(rel, ab, good)
        self.assertEqual(rec["n_good"], N - 1)
        self.assertAlmostEqual(rec["theta_frontal_rel"], 0.1, places=4)
        # with Fp1 on, the frontal mean includes it
        rec2 = FlowIndex(LABELS).update(rel, ab, [True] * N)
        self.assertAlmostEqual(rec2["theta_frontal_rel"], 0.5, places=4)

    def test_no_frontal_channels_left_falls_back_to_all_good(self):
        fi = FlowIndex(LABELS)
        good = [True] * N
        good[0] = good[1] = False
        theta = np.full(N, 0.3)
        rel, ab = epoch(theta=theta)
        rec = fi.update(rel, ab, good)
        self.assertEqual(rec["n_good"], 6)
        self.assertAlmostEqual(rec["theta_frontal_rel"], 0.3, places=4)

    def test_all_bad_falls_back_to_all_channels(self):
        fi = FlowIndex(LABELS)
        rel, ab = epoch()
        rec = fi.update(rel, ab, [False] * N)
        self.assertEqual(rec["n_good"], N)
        self.assertTrue(np.isfinite(rec["flow_eeg"]))

    def test_engagement_and_theta_alpha(self):
        fi = FlowIndex(LABELS)
        rel, ab = epoch(theta=0.2, alpha=0.3, beta=0.2)
        rec = fi.update(rel, ab, [True] * N)
        self.assertAlmostEqual(rec["engagement_index"], 20 / (30 + 20), places=3)
        self.assertAlmostEqual(rec["theta_alpha_ratio"], 20 / 30, places=3)

    def test_last_is_kept(self):
        fi = FlowIndex(LABELS)
        self.assertIsNone(fi.last)
        rel, ab = epoch()
        rec = fi.update(rel, ab, [True] * N)
        self.assertIs(fi.last, rec)


if __name__ == "__main__":
    unittest.main()
