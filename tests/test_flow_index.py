"""FlowIndex v0.2: Baseline modes (rest / session / rolling), warm-up, direction of
response (frontal theta is the primary marker; alpha / beta only through weights or the
exploratory flow_eeg_alt), artifact epochs, `extra` sub-band inputs, frontal /
central-parietal site selection, baseline_state strings."""
import math
import unittest

import numpy as np

import _path  # noqa: F401
from dsp import BAND_ORDER
from flow_index import Baseline, FlowIndex, sigmoid

LABELS_8 = ["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"]
LABELS_16 = LABELS_8 + ["F7", "F8", "F3", "F4", "T7", "T8", "P3", "P4"]
FRONTAL_16 = ["F3", "F4", "F7", "F8"]
CP_16 = ["C3", "C4", "P3", "P4"]
ALL_WEIGHTS = {"theta": 0.5, "alpha": 0.25, "beta": 0.25}      # the pre-v0.2 composite


def idx(labels, names):
    return [labels.index(n) for n in names]


def epoch(labels, theta=0.2, alpha=0.3, beta=0.2, delta=0.2, gamma=0.1, **per_channel):
    """Relative powers (constant across channels unless a band is given as a per-channel
    vector) and matching absolute powers (relative x 100 uV^2)."""
    n = len(labels)
    rel = {"delta": delta, "theta": theta, "alpha": alpha, "beta": beta, "gamma": gamma}
    out = {}
    for b in BAND_ORDER:
        v = per_channel.get(b, rel[b])
        out[b] = np.full(n, float(v)) if np.isscalar(v) else np.asarray(v, float)
    abs_pow = {b: out[b] * 100.0 for b in BAND_ORDER}
    return out, abs_pow


def feed(fi, labels, n=40, good=None, artifact=False, extra=None, **kw):
    good = good or [True] * len(labels)
    rec = None
    for _ in range(n):
        rel, ab = epoch(labels, **kw)
        rec = fi.update(rel, ab, good, artifact=artifact, extra=extra)
    return rec


def jitter(n, base, step=0.005):
    """A per-channel pattern that repeats epoch to epoch (so the baseline sd stays > 0 only
    where we add epoch-to-epoch variation) - used to give the baseline a non-degenerate sd."""
    return np.full(n, base) + step * (np.arange(n) % 2)


def feed_varied(fi, labels, n=40, good=None, seed=0, **kw):
    """Feed n epochs with small Gaussian jitter so every baseline has a non-zero sd."""
    rng = np.random.default_rng(seed)
    good = good or [True] * len(labels)
    rec = None
    for _ in range(n):
        k = dict(kw)
        for b in ("theta", "alpha", "beta"):
            v = k.get(b, {"theta": 0.2, "alpha": 0.3, "beta": 0.2}[b])
            k[b] = v * (1 + rng.normal(0, 0.03))
        rel, ab = epoch(labels, **k)
        rec = fi.update(rel, ab, good)
    return rec


class TestBaseline(unittest.TestCase):
    def test_warmup_returns_zero_until_enough_values(self):
        for mode in ("rest", "session", "rolling"):
            b = Baseline(mode, window=20, warmup=5)
            for i in range(4):
                self.assertEqual(b.push(float(i)), 0.0, f"{mode} value {i}")
                self.assertFalse(b.ready, mode)
            b.push(4.0)
            self.assertTrue(b.ready, mode)

    def test_rest_collects_then_freezes(self):
        b = Baseline("rest", window=50, warmup=5)
        self.assertEqual(b.state(), "session-until-rest")
        self.assertFalse(b.frozen)
        b.start_rest()
        self.assertEqual(b.state(), "rest-collecting")
        rest_values = [10.0, 12.0, 8.0, 11.0, 9.0, 10.0, 12.0, 8.0]
        for v in rest_values:
            b.push(v)
        b.end_rest()
        self.assertTrue(b.frozen)
        self.assertTrue(b.ready)
        self.assertEqual(b.state(), "rest-frozen")
        mean, std = float(np.mean(rest_values)), float(np.std(rest_values))
        self.assertAlmostEqual(b.mean, mean, places=9)
        self.assertAlmostEqual(b.std, std, places=9)
        # later values are z-scored against the frozen stats and do not move them
        for v in (50.0, 60.0, 70.0):
            self.assertAlmostEqual(b.push(v), (v - mean) / std, places=9)
        self.assertAlmostEqual(b.mean, mean, places=9)
        self.assertAlmostEqual(b.std, std, places=9)

    def test_rest_too_short_does_not_freeze(self):
        b = Baseline("rest", warmup=10)
        b.start_rest()
        for v in (1.0, 2.0, 3.0):
            b.push(v)
        b.end_rest()
        self.assertFalse(b.frozen)
        self.assertFalse(b.collecting)
        self.assertEqual(b.state(), "session-until-rest")

    def test_rest_before_frozen_behaves_like_session(self):
        rest, sess = Baseline("rest", warmup=3), Baseline("session", warmup=3)
        vals = [1.0, 2.0, 3.0, 10.0, -4.0, 7.0]
        for v in vals:
            self.assertAlmostEqual(rest.push(v), sess.push(v), places=12)

    def test_rest_constant_block_gets_unit_std(self):
        b = Baseline("rest", warmup=3)
        b.start_rest()
        for _ in range(5):
            b.push(2.0)
        b.end_rest()
        self.assertTrue(b.frozen)
        self.assertEqual(b.std, 1.0)
        self.assertAlmostEqual(b.push(3.0), 1.0)

    def test_session_expands_and_never_forgets(self):
        b = Baseline("session", window=5, warmup=3)
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        zs = [b.push(v) for v in vals]
        self.assertEqual(zs[:2], [0.0, 0.0])
        # after each push the stats are over everything seen so far (population std)
        self.assertAlmostEqual(b.mean, float(np.mean(vals)), places=9)
        self.assertAlmostEqual(b.std, float(np.std(vals)), places=9)
        self.assertAlmostEqual(zs[-1], (8.0 - np.mean(vals)) / np.std(vals), places=9)
        self.assertEqual(b.state(), "session")

    def test_rolling_forgets(self):
        b = Baseline("rolling", window=5, warmup=3)
        for _ in range(10):
            b.push(100.0)
        for _ in range(5):
            z = b.push(1.0)
        # the window now holds only 1.0 -> the old level is gone
        self.assertAlmostEqual(b.mean, 1.0, places=9)
        self.assertEqual(b.std, 1.0)          # degenerate sd -> 1
        self.assertAlmostEqual(z, 0.0, places=9)
        self.assertEqual(b.state(), "rolling")
        self.assertTrue(b.ready)

    def test_rolling_vs_session_diverge_after_a_shift(self):
        roll, sess = Baseline("rolling", window=10, warmup=3), Baseline("session", warmup=3)
        for _ in range(20):
            roll.push(0.0)
            sess.push(0.0)
        for _ in range(20):
            zr, zs = roll.push(5.0), sess.push(5.0)
        self.assertAlmostEqual(zr, 0.0, places=9)     # rolling has adapted to the new level
        self.assertGreater(zs, 0.5)                   # session still remembers the old level


class TestFlowIndexBasics(unittest.TestCase):
    def test_default_weights_are_theta_only(self):
        fi = FlowIndex(LABELS_8)
        self.assertEqual((fi.w_theta, fi.w_alpha, fi.w_beta), (1.0, 0.0, 0.0))
        fi2 = FlowIndex(LABELS_8, weights={"beta": 0.25})
        self.assertEqual((fi2.w_theta, fi2.w_alpha, fi2.w_beta), (1.0, 0.0, 0.25))

    def test_warmup_gives_neutral_half(self):
        fi = FlowIndex(LABELS_8, warmup_epochs=15)
        good = [True] * 8
        for i in range(1, 30):
            rel, ab = epoch(LABELS_8, theta=0.2 + 0.01 * (i % 3))
            rec = fi.update(rel, ab, good)
            self.assertEqual(rec["baseline_ready"], i >= 15, f"epoch {i}")
            if i < 15:
                self.assertEqual(rec["flow_eeg"], 0.5, f"epoch {i}")
                self.assertEqual(rec["flow_eeg_alt"], 0.5, f"epoch {i}")
                self.assertEqual((rec["z_theta"], rec["z_alpha"], rec["z_beta"]), (0.0, 0.0, 0.0))

    def test_constant_input_gives_neutral_flow_and_components(self):
        fi = FlowIndex(LABELS_8)
        rec = feed(fi, LABELS_8)
        self.assertAlmostEqual(rec["flow_eeg"], sigmoid(0.0), places=3)
        self.assertAlmostEqual(rec["flow_eeg_alt"], 0.5, places=3)
        self.assertEqual(rec["z_theta"], 0.0)
        self.assertEqual(rec["n_good"], 8)
        self.assertFalse(rec["artifact"])
        self.assertAlmostEqual(rec["theta_frontal_rel"], 0.2, places=4)
        self.assertAlmostEqual(rec["alpha_cp_rel"], 0.3, places=4)
        self.assertAlmostEqual(rec["beta_rel"], 0.2, places=4)
        self.assertAlmostEqual(rec["engagement_index"], 20 / (30 + 20), places=3)
        self.assertAlmostEqual(rec["theta_alpha_ratio"], 20 / 30, places=3)
        for b in BAND_ORDER:
            self.assertIn(f"{b}_rel_mean", rec)
        for old in ("alpha_posterior_rel", "alpha_fc_rel"):
            self.assertNotIn(old, rec)
        self.assertEqual(fi.n_epochs, 40)
        self.assertEqual(fi.n_artifact, 0)

    def test_frontal_theta_up_raises_flow(self):
        fi = FlowIndex(LABELS_16)
        base = feed_varied(fi, LABELS_16)
        theta = np.full(16, 0.2)
        theta[idx(LABELS_16, FRONTAL_16)] = 0.4
        rel, ab = epoch(LABELS_16, theta=theta)
        rec = fi.update(rel, ab, [True] * 16)
        self.assertGreater(rec["z_theta"], 1.0)
        self.assertGreater(rec["flow_eeg"], base["flow_eeg"])
        self.assertGreater(rec["flow_eeg"], 0.7)
        self.assertGreater(rec["flow_eeg_alt"], base["flow_eeg_alt"])
        self.assertAlmostEqual(rec["theta_frontal_rel"], 0.4, places=4)
        # and down with frontal theta below baseline
        theta[idx(LABELS_16, FRONTAL_16)] = 0.1
        rec = fi.update(*epoch(LABELS_16, theta=theta), [True] * 16)
        self.assertLess(rec["z_theta"], -1.0)
        self.assertLess(rec["flow_eeg"], 0.3)

    def test_theta_elsewhere_does_not_count_as_frontal(self):
        fi = FlowIndex(LABELS_16)
        feed(fi, LABELS_16)
        theta = np.full(16, 0.2)
        theta[idx(LABELS_16, ["Fp1", "Fp2", "P7", "P8", "O1", "O2"])] = 0.5
        rel, ab = epoch(LABELS_16, theta=theta)
        rec = fi.update(rel, ab, [True] * 16)
        self.assertAlmostEqual(rec["theta_frontal_rel"], 0.2, places=4)
        self.assertEqual(rec["z_theta"], 0.0)

    def test_beta_only_matters_with_a_beta_weight(self):
        fi = FlowIndex(LABELS_8)                                  # default: theta only
        feed_varied(fi, LABELS_8)
        ref = fi.update(*epoch(LABELS_8), [True] * 8)             # same theta as the probe below
        rec = fi.update(*epoch(LABELS_8, beta=0.5), [True] * 8)
        self.assertGreater(rec["z_beta"], 1.0)                    # the component is still exported
        self.assertAlmostEqual(rec["flow_eeg"], ref["flow_eeg"], places=2)
        fi = FlowIndex(LABELS_8, weights=ALL_WEIGHTS)
        feed_varied(fi, LABELS_8)
        ref = fi.update(*epoch(LABELS_8), [True] * 8)
        rec = fi.update(*epoch(LABELS_8, beta=0.5), [True] * 8)
        self.assertLess(rec["flow_eeg"], ref["flow_eeg"] - 0.1)

    def test_alpha_deviation_lowers_alt_index_either_way(self):
        for alpha in (0.6, 0.05):
            fi = FlowIndex(LABELS_8)
            feed_varied(fi, LABELS_8)
            ref = fi.update(*epoch(LABELS_8), [True] * 8)        # same theta as the probe below
            rec = fi.update(*epoch(LABELS_8, alpha=alpha), [True] * 8)
            self.assertNotEqual(rec["z_alpha"], 0.0)
            self.assertLess(rec["flow_eeg_alt"], ref["flow_eeg_alt"] - 0.1, f"alpha {alpha}")
            self.assertAlmostEqual(rec["flow_eeg"], ref["flow_eeg"], places=2)     # primary ignores alpha
            # alt = sigmoid(z_theta - 0.5 |z_alpha|)
            self.assertAlmostEqual(rec["flow_eeg_alt"], sigmoid(rec["z_theta"] - 0.5 * abs(rec["z_alpha"])), places=2)
        fi = FlowIndex(LABELS_8, weights=ALL_WEIGHTS)
        feed_varied(fi, LABELS_8)
        ref = fi.update(*epoch(LABELS_8), [True] * 8)
        rec = fi.update(*epoch(LABELS_8, alpha=0.6), [True] * 8)
        self.assertLess(rec["flow_eeg"], ref["flow_eeg"] - 0.1)

    def test_primary_is_sigmoid_of_weighted_z(self):
        fi = FlowIndex(LABELS_16, weights={"theta": 0.5, "alpha": 0.25, "beta": 0.25})
        feed_varied(fi, LABELS_16)
        theta = np.full(16, 0.2)
        theta[idx(LABELS_16, FRONTAL_16)] = 0.3
        rec = fi.update(*epoch(LABELS_16, theta=theta, beta=0.3, alpha=0.4), [True] * 16)
        x = 0.5 * rec["z_theta"] - 0.25 * abs(rec["z_alpha"]) - 0.25 * rec["z_beta"]
        self.assertAlmostEqual(rec["flow_eeg"], sigmoid(x), places=2)

    def test_weights_zero_pins_flow_at_half(self):
        fi0 = FlowIndex(LABELS_8, weights={"theta": 0.0, "alpha": 0.0, "beta": 0.0})
        feed(fi0, LABELS_8)
        rel, ab = epoch(LABELS_8, theta=0.6, beta=0.5)
        self.assertAlmostEqual(fi0.update(rel, ab, [True] * 8)["flow_eeg"], 0.5, places=3)
        self.assertEqual((fi0.w_theta, fi0.w_alpha, fi0.w_beta), (0.0, 0.0, 0.0))

    def test_extra_sub_bands_feed_theta_and_alpha(self):
        fi = FlowIndex(LABELS_16)
        n = 16
        extra = {"theta_fm": np.full(n, 0.11), "alpha_hi": np.full(n, 0.07)}
        rec = fi.update(*epoch(LABELS_16, theta=0.2, alpha=0.3), [True] * n, extra=extra)
        self.assertAlmostEqual(rec["theta_frontal_rel"], 0.11, places=4)
        self.assertAlmostEqual(rec["alpha_cp_rel"], 0.07, places=4)
        self.assertAlmostEqual(rec["theta_rel_mean"], 0.2, places=4)          # broad bands unchanged
        self.assertAlmostEqual(rec["alpha_rel_mean"], 0.3, places=4)
        # None / missing keys fall back to the broad bands
        rec = fi.update(*epoch(LABELS_16, theta=0.2, alpha=0.3), [True] * n, extra=None)
        self.assertAlmostEqual(rec["theta_frontal_rel"], 0.2, places=4)
        self.assertAlmostEqual(rec["alpha_cp_rel"], 0.3, places=4)
        rec = fi.update(*epoch(LABELS_16, theta=0.2, alpha=0.3), [True] * n, extra={})
        self.assertAlmostEqual(rec["theta_frontal_rel"], 0.2, places=4)
        # the z-scores follow the extra values, not the broad band
        fi = FlowIndex(LABELS_16)
        for _ in range(40):
            fi.update(*epoch(LABELS_16), [True] * n,
                      extra={"theta_fm": jitter(n, 0.1), "alpha_hi": jitter(n, 0.1)})
        rec = fi.update(*epoch(LABELS_16), [True] * n,
                        extra={"theta_fm": np.full(n, 0.25), "alpha_hi": jitter(n, 0.1)})
        self.assertGreater(rec["z_theta"], 1.0)
        self.assertGreater(rec["flow_eeg"], 0.7)

    def test_last_is_kept(self):
        fi = FlowIndex(LABELS_8)
        self.assertIsNone(fi.last)
        rel, ab = epoch(LABELS_8)
        rec = fi.update(rel, ab, [True] * 8)
        self.assertIs(fi.last, rec)


class TestArtifacts(unittest.TestCase):
    def test_artifact_epoch_has_no_index_and_does_not_touch_baseline(self):
        fi = FlowIndex(LABELS_8)
        feed(fi, LABELS_8, n=20)
        n_before = {k: b._session_n for k, b in fi.baselines.items()}
        rel, ab = epoch(LABELS_8, theta=0.9, beta=0.9)      # wild values that would shift the stats
        rec = fi.update(rel, ab, [True] * 8, artifact=True)
        self.assertIsNone(rec["flow_eeg"])
        self.assertIsNone(rec["flow_eeg_alt"])
        self.assertTrue(rec["artifact"])
        self.assertEqual((rec["z_theta"], rec["z_alpha"], rec["z_beta"]), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(rec["theta_frontal_rel"], 0.9, places=4)   # components still reported
        self.assertEqual(fi.n_artifact, 1)
        self.assertEqual(fi.n_epochs, 21)
        self.assertEqual({k: b._session_n for k, b in fi.baselines.items()}, n_before)
        self.assertIs(fi.last, rec)
        # the next clean epoch is unaffected by the artifact
        rel, ab = epoch(LABELS_8)
        clean = fi.update(rel, ab, [True] * 8)
        self.assertAlmostEqual(clean["flow_eeg"], 0.5, places=3)

    def test_artifact_during_rest_is_not_collected(self):
        fi = FlowIndex(LABELS_8, warmup_epochs=5)
        fi.start_rest()
        feed(fi, LABELS_8, n=10)
        feed(fi, LABELS_8, n=10, artifact=True, theta=0.9)
        fi.end_rest()
        self.assertEqual(len(fi.baselines["theta"]._rest), 10)
        self.assertTrue(fi.baseline_ready)
        self.assertEqual(fi.baseline_state, "rest-frozen")
        self.assertAlmostEqual(fi.baselines["theta"].mean, math.log(0.2), places=6)


class TestRestProtocol(unittest.TestCase):
    def test_rest_block_freezes_and_later_epochs_are_scored_against_it(self):
        fi = FlowIndex(LABELS_16, baseline_mode="rest", warmup_epochs=10)
        self.assertEqual(fi.baseline_state, "session-until-rest")
        fi.start_rest()
        self.assertEqual(fi.baseline_state, "rest-collecting")
        rng = np.random.default_rng(0)
        for _ in range(30):
            rel, ab = epoch(LABELS_16, theta=0.2 + rng.normal(0, 0.01), alpha=0.3 + rng.normal(0, 0.01),
                            beta=0.2 + rng.normal(0, 0.01))
            rec = fi.update(rel, ab, [True] * 16)
        self.assertEqual(rec["baseline_state"], "rest-collecting")
        fi.end_rest()
        self.assertEqual(fi.baseline_state, "rest-frozen")
        self.assertTrue(fi.baseline_ready)
        frozen = {k: (b.mean, b.std) for k, b in fi.baselines.items()}
        # a long run of a very different state would drag a session/rolling baseline along;
        # against the frozen rest it keeps reading as high theta -> high flow
        theta = np.full(16, 0.2)
        theta[idx(LABELS_16, FRONTAL_16)] = 0.35
        recs = [fi.update(*epoch(LABELS_16, theta=theta), [True] * 16) for _ in range(200)]
        self.assertGreater(recs[0]["z_theta"], 2.0)
        self.assertGreater(recs[-1]["z_theta"], 2.0)
        self.assertAlmostEqual(recs[0]["flow_eeg"], recs[-1]["flow_eeg"], places=3)
        self.assertGreater(recs[-1]["flow_eeg"], 0.7)
        self.assertEqual({k: (b.mean, b.std) for k, b in fi.baselines.items()}, frozen)
        self.assertEqual(recs[-1]["baseline_state"], "rest-frozen")

    def test_session_and_rolling_states(self):
        self.assertEqual(FlowIndex(LABELS_8, baseline_mode="session").baseline_state, "session")
        self.assertEqual(FlowIndex(LABELS_8, baseline_mode="rolling").baseline_state, "rolling")
        fi = FlowIndex(LABELS_8, baseline_mode="rolling", baseline_epochs=10, warmup_epochs=3)
        feed(fi, LABELS_8, n=20, theta=0.2)
        rec = feed(fi, LABELS_8, n=20, theta=0.4)
        self.assertAlmostEqual(rec["flow_eeg"], 0.5, places=3)      # rolling forgot the old level

    def test_update_uses_configured_modes(self):
        fi = FlowIndex(LABELS_8, baseline_mode="session", baseline_epochs=7, warmup_epochs=4)
        for b in fi.baselines.values():
            self.assertEqual((b.mode, b.window, b.warmup), ("session", 7, 4))
        rel, ab = epoch(LABELS_8)
        for i in range(1, 6):
            self.assertEqual(fi.update(rel, ab, [True] * 8)["baseline_ready"], i >= 4)


class TestSites(unittest.TestCase):
    def test_16ch_picks_true_frontal_sites(self):
        fi = FlowIndex(LABELS_16)
        rec = fi.update(*epoch(LABELS_16), [True] * 16)
        self.assertEqual(set(rec["frontal_sites"].split(",")), set(FRONTAL_16))
        s = fi.sites([True] * 16)
        self.assertEqual(sorted(LABELS_16[i] for i in s["frontal"]), sorted(FRONTAL_16))
        self.assertEqual(s["all"], list(range(16)))

    def test_16ch_theta_and_alpha_means_use_their_sites(self):
        fi = FlowIndex(LABELS_16)
        theta = np.full(16, 0.1)
        theta[idx(LABELS_16, FRONTAL_16)] = 0.3
        theta[idx(LABELS_16, ["Fp1", "Fp2"])] = 0.9
        alpha = np.full(16, 0.1)
        alpha[idx(LABELS_16, CP_16)] = 0.5             # central / parietal: C3 C4 P3 P4
        rec = fi.update(*epoch(LABELS_16, theta=theta, alpha=alpha), [True] * 16)
        self.assertAlmostEqual(rec["theta_frontal_rel"], 0.3, places=4)
        self.assertAlmostEqual(rec["alpha_cp_rel"], 0.5, places=4)
        # frontal alpha does not enter the central-parietal mean
        alpha = np.full(16, 0.1)
        alpha[idx(LABELS_16, FRONTAL_16 + ["Fp1", "Fp2", "O1", "O2"])] = 0.9
        rec = fi.update(*epoch(LABELS_16, alpha=alpha), [True] * 16)
        self.assertAlmostEqual(rec["alpha_cp_rel"], 0.1, places=4)

    def test_8ch_falls_back_to_fp_for_theta_and_c3_c4_for_alpha(self):
        fi = FlowIndex(LABELS_8)
        theta = np.full(8, 0.1)
        theta[idx(LABELS_8, ["Fp1", "Fp2"])] = 0.3
        alpha = np.full(8, 0.1)
        alpha[idx(LABELS_8, ["C3", "C4"])] = 0.5
        rec = fi.update(*epoch(LABELS_8, theta=theta, alpha=alpha), [True] * 8)
        self.assertEqual(rec["frontal_sites"], "Fp1,Fp2")
        self.assertAlmostEqual(rec["theta_frontal_rel"], 0.3, places=4)
        self.assertAlmostEqual(rec["alpha_cp_rel"], 0.5, places=4)
        # with C3/C4 bad the central-parietal alpha falls back to every good channel
        good = [True] * 8
        good[2] = good[3] = False
        rec = fi.update(*epoch(LABELS_8, theta=theta, alpha=alpha), good)
        self.assertEqual(rec["n_good"], 6)
        self.assertAlmostEqual(rec["alpha_cp_rel"], 0.1, places=4)
        self.assertEqual(rec["frontal_sites"], "Fp1,Fp2")

    def test_16ch_bad_frontal_falls_back_to_fp(self):
        fi = FlowIndex(LABELS_16)
        good = [True] * 16
        for n in FRONTAL_16:
            good[LABELS_16.index(n)] = False
        theta = np.full(16, 0.1)
        theta[idx(LABELS_16, ["Fp1", "Fp2"])] = 0.4
        rec = fi.update(*epoch(LABELS_16, theta=theta), good)
        self.assertEqual(rec["n_good"], 12)
        self.assertEqual(rec["frontal_sites"], "Fp1,Fp2")
        self.assertAlmostEqual(rec["theta_frontal_rel"], 0.4, places=4)

    def test_disabled_channel_excluded_from_frontal_mean(self):
        fi = FlowIndex(LABELS_8)
        good = [True] * 8
        good[0] = False                     # Fp1 off
        theta = np.full(8, 0.2)
        theta[0], theta[1] = 0.9, 0.1
        rec = fi.update(*epoch(LABELS_8, theta=theta), good)
        self.assertEqual(rec["n_good"], 7)
        self.assertEqual(rec["frontal_sites"], "Fp2")
        self.assertAlmostEqual(rec["theta_frontal_rel"], 0.1, places=4)
        rec2 = FlowIndex(LABELS_8).update(*epoch(LABELS_8, theta=theta), [True] * 8)
        self.assertAlmostEqual(rec2["theta_frontal_rel"], 0.5, places=4)

    def test_no_frontal_channels_left_falls_back_to_all_good(self):
        fi = FlowIndex(LABELS_8)
        good = [True] * 8
        good[0] = good[1] = False
        rec = fi.update(*epoch(LABELS_8, theta=0.3), good)
        self.assertEqual(rec["n_good"], 6)
        self.assertEqual(rec["frontal_sites"], "C3,C4,P7,P8,O1,O2")
        self.assertAlmostEqual(rec["theta_frontal_rel"], 0.3, places=4)

    def test_all_bad_falls_back_to_all_channels(self):
        fi = FlowIndex(LABELS_8)
        rec = fi.update(*epoch(LABELS_8), [False] * 8)
        self.assertEqual(rec["n_good"], 8)
        self.assertEqual(rec["frontal_sites"], "Fp1,Fp2")
        self.assertTrue(np.isfinite(rec["flow_eeg"]))


if __name__ == "__main__":
    unittest.main()
