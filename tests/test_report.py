"""report.py on a synthetic session with a planted band = lagged noisy copy of the EEG index."""
import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import _path  # noqa: F401
import report
from report import analyze, bootstrap_ci, cohen_kappa, pearson, rankdata, render_markdown, run, spearman

LABELS = ["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"]
T0 = 1762178400.0
EPOCH_S = 2.0
EPOCH_HEADER = (["t", "flow_eeg", "theta_frontal_rel", "alpha_posterior_rel", "beta_rel",
                 "engagement_index", "theta_alpha_ratio", "z_theta", "z_alpha", "z_beta",
                 "baseline_ready", "n_good"]
                + [f"{b}_rel_mean" for b in ("delta", "theta", "alpha", "beta", "gamma")]
                + [f"theta_abs_{l}" for l in LABELS] + [f"alpha_abs_{l}" for l in LABELS]
                + [f"beta_abs_{l}" for l in LABELS] + [f"q_{l}" for l in LABELS])


def hidden_flow(t):
    """Slow latent state with a faster ripple so the lag peak is sharp."""
    return 0.55 + 0.3 * np.sin(2 * np.pi * t / 420.0) + 0.1 * np.sin(2 * np.pi * t / 95.0)


def write_session(d: Path, duration=1800.0, band_every=10.0, lag=20.0, noise=0.02, seed=1,
                  warmup_epochs=15, held_every=None, felt=True, band_flow_missing_every=None):
    rng = np.random.default_rng(seed)
    te = T0 + np.arange(0, duration, EPOCH_S)
    rel = hidden_flow(te - T0)
    fe = np.clip(rel + rng.normal(0, 0.01, len(te)), 0, 1)
    with open(d / "epochs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(EPOCH_HEADER)
        for i, (t, v, x) in enumerate(zip(te, fe, rel)):
            theta = 0.15 + 0.2 * x                 # tracks flow
            beta = 0.35 - 0.2 * x                  # anti-tracks flow
            alpha = 0.3
            row = [f"{t:.3f}", f"{v:.4f}", f"{theta:.5f}", f"{alpha:.5f}", f"{beta:.5f}",
                   f"{beta / (alpha + theta):.4f}", f"{theta / alpha:.4f}", "0.1", "0.0", "-0.1",
                   str(i >= warmup_epochs), "8", "0.2", f"{theta:.5f}", "0.3", f"{beta:.5f}", "0.05"]
            row += ["12.5"] * 8 + ["20.0"] * 8 + ["8.0"] * 8 + ["good"] * 8
            w.writerow(row)
    # band: EEG mean over [t - lag - W, t - lag] + noise, W = 10 s for a 10 s cadence
    W = 60.0 if band_every > 20 else 10.0
    tb = T0 + np.arange(band_every * 3, duration, band_every)
    with open(d / "band.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "flow", "hr", "anom", "held", "settling", "fhi"])
        for k, t in enumerate(tb):
            m = (te >= t - lag - W) & (te <= t - lag)
            v = float(fe[m].mean()) + rng.normal(0, noise) if m.any() else 0.5
            held = int(bool(held_every) and k % held_every == 0)
            if held:
                v = 0.0                               # a held sample carries a stale value
            flow = "" if (band_flow_missing_every and k % band_flow_missing_every == 0) else f"{v:.4f}"
            w.writerow([f"{t:.3f}", flow, "" if k % 2 else "71.5", "0.12", held, 0, ""])
    with open(d / "marks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "kind", "value", "note"])
        if felt:
            for t in T0 + np.array([300, 600, 900, 1200, 1500.0]):
                if t - T0 >= duration:
                    continue
                m = (te >= t - 60) & (te <= t)
                w.writerow([f"{t:.3f}", "felt_flow", f"{fe[m].mean() + rng.normal(0, 0.03):.2f}", "slider"])
            w.writerow([f"{T0 + 100:.3f}", "flow_start", "", ""])
            w.writerow([f"{T0 + 200:.3f}", "note", "coffee", ""])
    (d / "session.json").write_text(json.dumps({"board": {"kind": "sim"}, "labels": LABELS,
                                                "started": "2026-11-03_14-00-00"}), "utf-8")
    return {"te": te, "fe": fe, "tb": tb, "W": W, "lag": lag}


class TestStats(unittest.TestCase):
    def test_pearson_edge_cases(self):
        self.assertIsNone(pearson([1, 2], [1, 2]))
        self.assertIsNone(pearson([1, 1, 1], [1, 2, 3]))
        self.assertAlmostEqual(pearson([1, 2, 3, 4], [2, 4, 6, 8]), 1.0)
        self.assertAlmostEqual(pearson([1, 2, 3, 4], [8, 6, 4, 2]), -1.0)
        self.assertAlmostEqual(pearson([1, 2, 3, np.nan], [2, 4, 6, 1]), 1.0)   # nan pair dropped
        self.assertIsNone(pearson([], []))

    def test_rankdata_ties(self):
        np.testing.assert_allclose(rankdata([10, 20, 20, 30]), [1, 2.5, 2.5, 4])
        np.testing.assert_allclose(rankdata([3, 1, 2]), [3, 1, 2])
        self.assertEqual(len(rankdata([])), 0)

    def test_spearman(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 100, 1000, 10000]), 1.0)
        self.assertIsNone(spearman([1, 1, 1], [1, 2, 3]))

    def test_bootstrap_ci(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=60)
        y = x + rng.normal(scale=0.5, size=60)
        ci = bootstrap_ci(x, y)
        r = pearson(x, y)
        self.assertLess(ci[0], r)
        self.assertGreater(ci[1], r)
        self.assertGreater(ci[0], 0.5)
        self.assertIsNone(bootstrap_ci(x[:5], y[:5]))
        self.assertIsNone(bootstrap_ci(np.ones(30), y[:30]))
        self.assertEqual(bootstrap_ci(x, y), ci)          # seeded -> deterministic

    def test_kappa(self):
        a = np.array([1, 1, 0, 0, 1, 0], bool)
        self.assertAlmostEqual(cohen_kappa(a, a), 1.0)
        self.assertAlmostEqual(cohen_kappa(a, ~a), -1.0)
        self.assertIsNone(cohen_kappa(np.ones(5, bool), np.ones(5, bool)))
        self.assertIsNone(cohen_kappa([], []))


class TestAnalyze(unittest.TestCase):
    def test_planted_relationship_is_recovered(self):
        with tempfile.TemporaryDirectory() as d:
            sd = Path(d) / "2026-11-03_14-00-00_synthetic"
            sd.mkdir()
            planted = write_session(sd)
            res = run(sd)
            self.assertEqual(res["pair_window_s"], 10.0)
            self.assertGreaterEqual(res["n_pairs"], 150)
            self.assertGreater(res["r"], 0.8)
            self.assertGreater(res["r_ci"][0], 0.8)
            self.assertLessEqual(res["r_ci"][1], 1.0)
            self.assertGreater(res["spearman"], 0.8)
            self.assertLess(res["mae"], 0.12)         # lag-0 pairs carry the planted 20 s lag
            self.assertLess(abs(res["bias"]), 0.03)
            self.assertLessEqual(abs(res["best_lag_s"] - planted["lag"]), 10)
            self.assertGreater(res["r_at_best_lag"], 0.95)
            self.assertGreater(res["r_at_best_lag"], res["r"])
            lags = dict(res["lags"])
            self.assertEqual(sorted(lags), list(range(-180, 181, 10)))
            self.assertGreater(lags[planted["lag"]], lags[-planted["lag"]])
            self.assertGreater(res["components"]["theta_frontal_rel"]["r"], 0.8)
            self.assertLess(res["components"]["beta_rel"]["r"], -0.8)
            self.assertGreater(res["agreement"]["accuracy"], 0.8)
            self.assertGreater(res["agreement"]["kappa"], 0.5)
            self.assertEqual(res["felt"]["n_marks"], 5)
            self.assertEqual(res["felt"]["n_eeg"], 5)
            self.assertEqual(res["felt"]["n_band"], 5)
            self.assertGreater(res["felt"]["r_eeg"], 0.8)
            self.assertEqual(len(res["other_marks"]), 2)
            self.assertEqual(res["verdict"], "PASS")
            cov = res["coverage"]
            self.assertEqual(cov["epochs"], 900)
            self.assertEqual(cov["epochs_warmup"], 15)
            self.assertAlmostEqual(cov["duration_s"], 1798.0, delta=1)
            self.assertGreater(cov["pct_time_both"], 90)
            self.assertEqual(res["meta"]["labels"], LABELS)
            # files written and JSON-clean
            self.assertTrue((sd / "report.md").exists())
            j = json.loads((sd / "report.json").read_text("utf-8"))
            self.assertEqual(j["verdict"], "PASS")
            self.assertNotIn("markdown", j)
            self.assertEqual(j["summary"]["n_pairs"], res["n_pairs"])

    def test_markdown_contents(self):
        with tempfile.TemporaryDirectory() as d:
            sd = Path(d) / "s"
            sd.mkdir()
            write_session(sd, duration=900)
            res = analyze(sd)
            md = render_markdown(res)
            for s in ("# Flow validation report: s", "## Verdict: **PASS**", "## Headline",
                      "| n pairs |", "| Pearson r [95 % CI] |", "| Spearman rho |", "| MAE |",
                      "| mean bias (band - EEG) |", "| best lag |", "## EEG components vs band flow",
                      "| theta_frontal_rel |", "## Lag sweep", "| +30 |", "| -180 |", "| +0 |",
                      "## Agreement", "Cohen's kappa", "## Felt-flow marks", "## Coverage",
                      "## How to read this", "proxy", "not ground truth", "Pre-registered gates"):
                self.assertIn(s, md, s)
            self.assertNotIn("| +10 |", md)          # lag table is every 30 s
            self.assertIn(f"| n pairs | {res['n_pairs']} |", md)

    def test_supabase_cadence_uses_60s_window(self):
        with tempfile.TemporaryDirectory() as d:
            sd = Path(d) / "s"
            sd.mkdir()
            write_session(sd, duration=2400, band_every=60.0, lag=0.0, noise=0.01)
            res = analyze(sd)
            self.assertEqual(res["pair_window_s"], 60.0)
            self.assertIn("60.0 s", res["pair_window_reason"])
            self.assertGreater(res["r"], 0.9)
            self.assertLessEqual(abs(res["best_lag_s"]), 10)

    def test_held_samples_skipped_unless_included(self):
        with tempfile.TemporaryDirectory() as d:
            sd = Path(d) / "s"
            sd.mkdir()
            write_session(sd, held_every=4, band_flow_missing_every=7)
            res = analyze(sd)
            cov = res["coverage"]
            self.assertGreater(cov["band_skipped_held"], 0)
            self.assertEqual(cov["band_samples"], cov["band_with_flow"] + (cov["band_samples"] - cov["band_with_flow"]))
            self.assertLess(cov["band_with_flow"], cov["band_samples"])
            self.assertEqual(res["n_pairs"], cov["band_with_flow"] - cov["band_skipped_held"])
            self.assertGreater(res["r"], 0.8)
            res2 = analyze(sd, include_held=True)
            self.assertEqual(res2["coverage"]["band_skipped_held"], 0)
            self.assertEqual(res2["n_pairs"], cov["band_with_flow"])
            self.assertLess(res2["r"], res["r"])        # the stale zeros hurt

    def test_fail_on_uncorrelated_band(self):
        with tempfile.TemporaryDirectory() as d:
            sd = Path(d) / "s"
            sd.mkdir()
            write_session(sd, noise=0.3, lag=0.0, felt=False)
            rng = np.random.default_rng(5)
            rows = list(csv.reader(open(sd / "band.csv", newline="", encoding="utf-8")))
            for r in rows[1:]:
                r[1] = f"{rng.uniform(0, 1):.4f}"
            with open(sd / "band.csv", "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(rows)
            res = analyze(sd)
            self.assertEqual(res["verdict"], "FAIL")
            self.assertLess(abs(res["r"]), 0.3)
            self.assertEqual(res["felt"]["n_marks"], 0)
            self.assertIn("No felt-flow marks", render_markdown(res))

    def test_too_few_pairs_is_inconclusive(self):
        with tempfile.TemporaryDirectory() as d:
            sd = Path(d) / "s"
            sd.mkdir()
            write_session(sd, duration=150, band_every=10.0)
            res = analyze(sd)
            self.assertLess(res["n_pairs"], 20)
            self.assertEqual(res["verdict"], "INCONCLUSIVE")
            self.assertIsNone(res["r_ci"]) if res["n_pairs"] < 8 else self.assertIsNotNone(res["r_ci"])
            render_markdown(res)

    def test_empty_and_missing_files(self):
        with tempfile.TemporaryDirectory() as d:
            sd = Path(d) / "empty"
            sd.mkdir()
            res = run(sd)
            self.assertEqual(res["n_pairs"], 0)
            self.assertIsNone(res["r"])
            self.assertIsNone(res["best_lag_s"])
            self.assertEqual(res["verdict"], "INCONCLUSIVE")
            self.assertEqual(res["coverage"]["duration_s"], 0.0)
            self.assertTrue((sd / "report.md").exists())
            json.loads((sd / "report.json").read_text("utf-8"))
            # header-only files and a band with no readings
            (sd / "epochs.csv").write_text(",".join(EPOCH_HEADER) + "\n", "utf-8")
            (sd / "band.csv").write_text("t,flow,hr,anom,held,settling,fhi\n1.000,,,0.1,0,1,\n", "utf-8")
            (sd / "marks.csv").write_text("", "utf-8")
            res = analyze(sd)
            self.assertEqual(res["n_pairs"], 0)
            self.assertEqual(res["coverage"]["band_samples"], 1)
            self.assertEqual(res["coverage"]["band_with_flow"], 0)
            self.assertTrue(any("no sample with a flow" in n for n in res["notes"]))
            render_markdown(res)

    def test_constant_series_gives_null_r(self):
        with tempfile.TemporaryDirectory() as d:
            sd = Path(d) / "s"
            sd.mkdir()
            write_session(sd, duration=600)
            rows = list(csv.reader(open(sd / "band.csv", newline="", encoding="utf-8")))
            for r in rows[1:]:
                r[1] = "0.5000"
            with open(sd / "band.csv", "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(rows)
            res = analyze(sd)
            self.assertGreater(res["n_pairs"], 20)
            self.assertIsNone(res["r"])
            self.assertIsNone(res["spearman"])
            self.assertIsNone(res["r_ci"])
            self.assertEqual(res["verdict"], "FAIL")
            self.assertIn("n/a", render_markdown(res))


class TestCli(unittest.TestCase):
    def test_all_and_json(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()) as out, \
                contextlib.redirect_stderr(io.StringIO()):
            root = Path(d) / "sessions"
            a = root / "2026-11-03_14-00-00_a"
            b = root / "2026-11-03_15-00-00_b"
            a.mkdir(parents=True)
            b.mkdir()
            write_session(a, duration=600)
            write_session(b, duration=600)
            (b / "report.md").write_text("old", "utf-8")
            (root / "not_a_session").mkdir()
            self.assertEqual(report.main(["--all", str(root), "--quiet"]), 0)
            self.assertIn("2026-11-03_14-00-00_a: PASS", out.getvalue())
            self.assertTrue((a / "report.json").exists())
            self.assertEqual((b / "report.md").read_text("utf-8"), "old")       # skipped: has a report
            self.assertEqual(report.main(["--all", str(root), "--force", "--quiet"]), 0)
            self.assertNotEqual((b / "report.md").read_text("utf-8"), "old")
            self.assertEqual(report.main([str(a), "--json", "--include-held"]), 0)
            self.assertTrue(json.loads((a / "report.json").read_text("utf-8"))["include_held"])
            self.assertEqual(report.main([str(root / "missing")]), 2)


if __name__ == "__main__":
    unittest.main()
