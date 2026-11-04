"""report.py on synthetic sessions: a free recording with a planted band = lagged noisy copy
of the EEG index (epoch-level design), and protocol sessions (block-level design) with
block / rating / task data planted so that flow blocks score high."""
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
from report import (analyze, block_analysis, bootstrap_ci, cohen_kappa, parse_protocol, pearson, rankdata,
                    render_markdown, run, spearman, welch_t)

LABELS = ["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"]
T0 = 1762178400.0
EPOCH_S = 2.0
EPOCH_HEADER = (["t", "flow_eeg", "flow_eeg_alt", "artifact", "theta_frontal_rel", "alpha_cp_rel", "beta_rel",
                 "engagement_index", "theta_alpha_ratio", "z_theta", "z_alpha", "z_beta",
                 "baseline_ready", "baseline_state", "n_good", "frontal_sites"]
                + [f"{b}_rel_mean" for b in ("delta", "theta", "alpha", "beta", "gamma")]
                + ["p2p_uv", "n_segments_used"]
                + [f"theta_abs_{l}" for l in LABELS] + [f"alpha_abs_{l}" for l in LABELS]
                + [f"beta_abs_{l}" for l in LABELS] + [f"q_{l}" for l in LABELS])
FSS_ITEMS = ["challenge", "fluent", "time", "concentrate", "clear", "absorbed", "automatic", "know", "control", "lost"]


def read_rows(p: Path):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def write_rows(p: Path, rows):
    with open(p, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


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
            row = [f"{t:.3f}", f"{v:.4f}", f"{v:.4f}", "False", f"{theta:.5f}", f"{alpha:.5f}", f"{beta:.5f}",
                   f"{beta / (alpha + theta):.4f}", f"{theta / alpha:.4f}", "0.1", "0.0", "-0.1",
                   str(i >= warmup_epochs), "session", "8", "Fp1,Fp2",
                   "0.2", f"{theta:.5f}", "0.3", f"{beta:.5f}", "0.05", "40.0", "9"]
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


# --- protocol session ----------------------------------------------------------------------
EEG_LEVEL = {"rest": 0.5, "pause": 0.5, "flow": 0.75, "boredom": 0.35, "overload": 0.45}
FSS_LEVEL = {"flow": 6.0, "boredom": 2.0, "overload": 3.0}
DEMAND = {"flow": 2, "boredom": 1, "overload": 3}
FELT = {"flow": 0.8, "boredom": 0.2, "overload": 0.3}
TASK = {"flow": ("arithmetic_adaptive", 0.78, 5.0, 3000.0), "boredom": ("arithmetic_easy", 1.0, 1.0, 1500.0),
        "overload": ("arithmetic_hard", 0.2, 12.0, 5000.0)}


def protocol_schedule(rounds=2):
    """(index, condition, start_s, end_s, label) like protocols.json difficulty_ladder."""
    seq = ["flow", "boredom", "overload", "flow", "overload", "boredom"][: 3 * rounds]
    blocks = [("rest", 180.0, "Rest, eyes open")]
    for k, c in enumerate(seq):
        if k:
            blocks.append(("pause", 60.0, "Pause"))
        blocks.append((c, 300.0, {"flow": "Matched arithmetic", "boredom": "Easy arithmetic",
                                  "overload": "Overload arithmetic"}[c]))
    blocks.append(("rest", 120.0, "Rest, eyes open"))
    out, t = [], 0.0
    for i, (c, dur, label) in enumerate(blocks):
        out.append((i, c, t, t + dur, label))
        t += dur
    return out


def write_protocol_session(d: Path, band="tracking", fss="ok", rounds=2, seed=3, band_lag=30.0,
                           artifact_every=25, with_band=True, with_task=True, stop_after=None):
    """A difficulty-ladder session: EEG index high in flow blocks (with artifact rows that
    carry no index), band = lagged 60-s mean of the EEG (or flat), FSS / demand / task data
    per the induced condition (or FSS flat = the induction failed)."""
    rng = np.random.default_rng(seed)
    sched = protocol_schedule(rounds)
    if stop_after is not None:
        sched = sched[: stop_after + 1]
    total = sched[-1][3]
    te = T0 + np.arange(2.0, total, EPOCH_S)

    def cond_at(t):
        for i, c, s, e, _ in sched:
            if s <= t - T0 < e:
                return i, c
        return sched[-1][0], sched[-1][1]

    level = np.array([EEG_LEVEL[cond_at(t)[1]] + 0.003 * cond_at(t)[0] for t in te])
    fe = np.clip(level + rng.normal(0, 0.03, len(te)), 0, 1)
    art = np.zeros(len(te), bool)
    if artifact_every:
        art[::artifact_every] = True
    with open(d / "epochs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(EPOCH_HEADER)
        for i, (t, v, a) in enumerate(zip(te, fe, art)):
            state = "rest-collecting" if t - T0 < 180 else "rest-frozen"
            row = [f"{t:.3f}", "" if a else f"{v:.4f}", "" if a else f"{min(1, v + 0.02):.4f}", str(bool(a)),
                   "0.2", "0.3", "0.2", "0.4", "0.67", "0.1", "0.0", "-0.1", str(i >= 15), state, "8", "Fp1,Fp2",
                   "0.2", "0.2", "0.3", "0.2", "0.05", "40.0", "9" if not a else "3"]
            row += ["12.5"] * 8 + ["20.0"] * 8 + ["8.0"] * 8 + ["good"] * 8
            w.writerow(row)
    tb = T0 + np.arange(60.0, total, 60.0)
    with open(d / "band.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "flow", "hr", "anom", "held", "settling", "fhi"])
        if with_band:
            for t in tb:
                if band == "tracking":
                    m = (te >= t - band_lag - 60) & (te <= t - band_lag) & ~art
                    v = float(fe[m].mean()) + rng.normal(0, 0.02) if m.any() else 0.5
                elif band == "flat":
                    v = 0.5 + rng.normal(0, 0.005)
                else:
                    v = float(band)
                w.writerow([f"{t:.3f}", f"{min(1, max(0, v)):.4f}", "", "0.1", 0, 0, ""])
    with open(d / "marks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "kind", "value", "note"])
        w.writerow([f"{T0:.3f}", "protocol_start", "difficulty_ladder",
                    "seed=42 order=" + ",".join(c for _, c, _, _, _ in sched)])
        w.writerow([f"{T0 + 30:.3f}", "note", "coffee", ""])
        seen: dict = {}
        for i, c, s, e, label in sched:
            w.writerow([f"{T0 + s:.3f}", "block_start", c, f"block={i} {label}"])
            w.writerow([f"{T0 + e:.3f}", "block_end", c, f"block={i}"])
            if c in FSS_LEVEL:
                base = 4.0 if fss == "flat" else FSS_LEVEL[c]
                k = seen.get(c, 0)                       # the 2nd block of a condition rates one point higher
                seen[c] = k + 1
                items = {it: float(min(7, max(1, round(base + (j % 3 - 1) + k)))) for j, it in enumerate(FSS_ITEMS)}
                items["demand"] = float(DEMAND[c])
                items["flow_overall"] = FELT[c]
                w.writerow([f"{T0 + e + 5:.3f}", "rating", json.dumps({"block": i, "condition": c, **items}), ""])
        w.writerow([f"{T0 + total:.3f}", "protocol_end", "difficulty_ladder", ""])
    with open(d / "task.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "block_index", "condition", "task", "level", "problem", "answer", "response", "correct", "rt_ms"])
        if with_task:
            for i, c, s, e, _ in sched:
                if c not in TASK:
                    continue
                task, acc, lvl, rt = TASK[c]
                n = 50
                for k in range(n):
                    correct = int(k < round(acc * n))
                    w.writerow([f"{T0 + s + 12 + k * 5:.3f}", i, c, task, int(lvl + (k % 3) - 1), "23 + 45", 68,
                                68 if correct else 66, correct, f"{rt + 100 * (k % 5):.0f}"])
    (d / "session.json").write_text(json.dumps({"board": {"kind": "sim"}, "labels": LABELS,
                                                "started": "2026-11-03_14-00-00"}), "utf-8")
    return {"schedule": sched, "te": te, "fe": fe, "art": art, "tb": tb}


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
            # no protocol: old epoch-level design, said so, block fields empty
            self.assertIsNone(res["protocol"])
            self.assertEqual(res["design"], "epoch")
            self.assertEqual(res["epoch_verdict"], res["verdict"])
            self.assertIn("No protocol was run", md)
            self.assertTrue(any("No protocol was run" in n for n in res["notes"]))
            self.assertNotIn("## Per-block table", md)
            self.assertIn("Katahira", md)
            self.assertIn("Pope", md)
            s = res["summary"]
            self.assertEqual(s["verdict"], "PASS")
            self.assertIsNone(s["block_rho"])
            self.assertEqual(s["n_blocks"], 0)
            self.assertEqual(s["ordering"], {"eeg": None, "band": None, "fss": None})
            self.assertIsNone(s["manipulation_ok"])
            self.assertIsNone(s["protocol"])

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
            rows = read_rows(sd / "band.csv")
            for r in rows[1:]:
                r[1] = f"{rng.uniform(0, 1):.4f}"
            write_rows(sd / "band.csv", rows)
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
            rows = read_rows(sd / "band.csv")
            for r in rows[1:]:
                r[1] = "0.5000"
            write_rows(sd / "band.csv", rows)
            res = analyze(sd)
            self.assertGreater(res["n_pairs"], 20)
            self.assertIsNone(res["r"])
            self.assertIsNone(res["spearman"])
            self.assertIsNone(res["r_ci"])
            self.assertEqual(res["verdict"], "FAIL")
            self.assertIn("n/a", render_markdown(res))


class TestProtocolParsing(unittest.TestCase):
    def test_parse_protocol(self):
        marks = [
            {"t": 100.0, "kind": "protocol_start", "value": float("nan"), "raw_value": "sim_quick",
             "note": "seed=7 order=rest,flow,pause,boredom"},
            {"t": 100.0, "kind": "block_start", "value": float("nan"), "raw_value": "rest", "note": "block=0 Rest, eyes open"},
            {"t": 160.0, "kind": "block_end", "value": float("nan"), "raw_value": "rest", "note": "block=0"},
            {"t": 160.0, "kind": "block_start", "value": float("nan"), "raw_value": "flow", "note": "block=1 Matched"},
            {"t": 220.0, "kind": "block_end", "value": float("nan"), "raw_value": "flow", "note": "block=1"},
            {"t": 225.0, "kind": "rating", "value": float("nan"),
             "raw_value": json.dumps({"block": 1, "condition": "flow", "challenge": 6, "fluent": 5.0, "demand": 2,
                                      "flow_overall": 0.8}), "note": "ok"},
            {"t": 226.0, "kind": "rating", "value": float("nan"), "raw_value": "not json", "note": ""},
            {"t": 227.0, "kind": "rating", "value": float("nan"), "raw_value": "[1, 2]", "note": ""},
            {"t": 220.0, "kind": "block_start", "value": float("nan"), "raw_value": "pause", "note": "block=2 Pause"},
            {"t": 240.0, "kind": "block_end", "value": float("nan"), "raw_value": "pause", "note": "block=2"},
            {"t": 240.0, "kind": "block_start", "value": float("nan"), "raw_value": "boredom", "note": "block=3 Easy"},
            {"t": 270.0, "kind": "block_end", "value": float("nan"), "raw_value": "boredom", "note": "block=3 aborted"},
            {"t": 270.0, "kind": "protocol_stop", "value": float("nan"), "raw_value": "sim_quick", "note": ""},
        ]
        p = parse_protocol(marks)
        self.assertEqual(p["name"], "sim_quick")
        self.assertEqual(p["seed"], 7)
        self.assertEqual(p["planned_order"], ["rest", "flow", "pause", "boredom"])
        self.assertEqual((p["t_start"], p["t_end"], p["stopped"]), (100.0, 270.0, True))
        self.assertEqual([b["index"] for b in p["blocks"]], [0, 1, 2, 3])
        self.assertEqual([b["condition"] for b in p["blocks"]], ["rest", "flow", "pause", "boredom"])
        self.assertEqual(p["blocks"][0]["label"], "Rest, eyes open")
        self.assertEqual((p["blocks"][1]["t0"], p["blocks"][1]["t1"], p["blocks"][1]["aborted"]), (160.0, 220.0, False))
        self.assertTrue(p["blocks"][3]["aborted"])
        self.assertEqual(len(p["ratings"]), 1)
        r = p["ratings"][0]
        self.assertEqual((r["block"], r["condition"], r["note"]), (1, "flow", "ok"))
        self.assertEqual(r["items"], {"challenge": 6.0, "fluent": 5.0, "demand": 2.0, "flow_overall": 0.8})
        # no protocol marks at all -> None; block marks without a protocol_start still parse
        self.assertIsNone(parse_protocol([m for m in marks if m["kind"] in ("rating",)]))
        self.assertIsNone(parse_protocol([]))
        q = parse_protocol([m for m in marks if m["kind"].startswith("block")])
        self.assertIsNone(q["name"])
        self.assertEqual(len(q["blocks"]), 4)
        # an open block (no end mark) keeps t1 None
        q = parse_protocol(marks[:2])
        self.assertIsNone(q["blocks"][0]["t1"])

    def test_welch_t(self):
        tt = welch_t([6.0, 5.5], [2.0, 2.5])
        self.assertGreater(tt["t"], 5)
        self.assertEqual((tt["n_a"], tt["n_b"]), (2, 2))
        self.assertAlmostEqual(tt["df"], 2.0, places=6)
        self.assertLess(welch_t([2.0, 2.5], [6.0, 5.5])["t"], -5)
        self.assertIsNone(welch_t([6.0], [2.0, 2.5]))
        self.assertIsNone(welch_t([6.0, 6.0], [2.0, 2.0]))           # no variance at all
        self.assertIsNone(welch_t([6.0, None], [2.0, 2.5]))
        tt = welch_t([5, 6, 7, 8], [1, 2, 3, 4])
        self.assertAlmostEqual(tt["t"], 4 / np.sqrt(2 * (5 / 3) / 4), places=6)

    def test_block_analysis_band_fallback_after_block_end(self):
        proto = {"name": "x", "seed": 1, "planned_order": None, "t_start": 0.0, "t_end": 1000.0, "stopped": False,
                 "blocks": [{"index": 0, "condition": "flow", "label": "F", "t0": 100.0, "t1": 400.0, "aborted": False},
                            {"index": 1, "condition": "boredom", "label": "B", "t0": 400.0, "t1": 420.0, "aborted": False},
                            {"index": 2, "condition": "pause", "label": "P", "t0": 420.0, "t1": 500.0, "aborted": False},
                            {"index": 3, "condition": "overload", "label": "O", "t0": 500.0, "t1": None, "aborted": False}],
                 "ratings": [{"t": 405.0, "block": 0, "condition": "flow", "items": {"challenge": 6.0, "fluent": 4.0,
                                                                                     "demand": 2.0, "flow_overall": 0.7},
                              "note": ""}]}
        te = np.arange(100.0, 400.0, 2.0)
        ep = {"t": te, "flow": np.full(len(te), 0.7), "flow_alt": np.full(len(te), 0.72),
              "ready": np.ones(len(te), bool), "artifact": np.zeros(len(te), bool)}
        ep["artifact"][2] = True                                   # t = 104, inside the transition: not counted
        ep["artifact"][50] = True
        bd = {"t": np.array([50.0, 105.0, 430.0, 450.0, 455.0]), "flow": np.array([0.1, 0.2, 0.6, 0.9, 0.8]),
              "held": np.array([False, False, False, True, False]), "settling": np.zeros(5, bool)}
        task = [{"t": 110.0, "block": 0, "condition": "flow", "task": "a", "level": 4.0, "correct": True, "rt_ms": 2000.0},
                {"t": 120.0, "block": 0, "condition": "flow", "task": "a", "level": 6.0, "correct": False, "rt_ms": 3000.0},
                {"t": 130.0, "block": 0, "condition": "flow", "task": "a", "level": 5.0, "correct": True, "rt_ms": float("nan")}]
        P = block_analysis(proto, ep, bd, task, include_held=False)
        rows = {r["index"]: r for r in P["blocks"]}
        self.assertEqual(sorted(rows), [0, 1, 2])                   # the open block 3 is dropped with a note
        self.assertTrue(any("block 3" in n for n in P["notes"]))
        r0 = rows[0]
        self.assertTrue(r0["scored"] and r0["is_task"])
        self.assertAlmostEqual(r0["eeg_mean"], 0.7)
        self.assertEqual(r0["eeg_n"], int(((te >= 110) & (te <= 400)).sum()) - 1)
        self.assertEqual(r0["eeg_artifacts"], 1)
        self.assertAlmostEqual(r0["eeg_alt_mean"], 0.72)
        # the only band sample inside [t0+10, t1] is none (105 < 110) -> samples within 60 s after t1
        self.assertTrue(r0["band_after_end"])
        self.assertEqual(r0["band_n"], 2)                           # 430, 455 (450 is held)
        self.assertAlmostEqual(r0["band_mean"], 0.7)
        self.assertTrue(any("after the block end" in n for n in P["notes"]))
        self.assertAlmostEqual(r0["fss"], 5.0)
        self.assertEqual(r0["fss_n_items"], 2)
        self.assertEqual((r0["demand"], r0["flow_overall"]), (2.0, 0.7))
        self.assertEqual(r0["n_trials"], 3)
        self.assertAlmostEqual(r0["accuracy"], 2 / 3)
        self.assertAlmostEqual(r0["rt_median_ms"], 2500.0)
        self.assertAlmostEqual(r0["level_mean"], 5.0)
        self.assertFalse(rows[1]["scored"])                         # 20 s block
        self.assertFalse(rows[2]["scored"])                         # pause
        self.assertEqual(P["n_task_blocks"], 1)
        self.assertEqual(P["verdict"], "INCONCLUSIVE")
        self.assertIn("only 1 usable task blocks", P["verdict_reason"])
        self.assertIsNone(P["manipulation"]["ok"])
        self.assertIn("boredom, overload", P["manipulation"]["reason"])
        self.assertEqual([c["condition"] for c in P["conditions"]], ["flow"])
        self.assertEqual(P["primary"]["ordering"]["eeg"], None)
        # include_held keeps the held sample
        P2 = block_analysis(proto, ep, bd, task, include_held=True)
        self.assertEqual({r["index"]: r for r in P2["blocks"]}[0]["band_n"], 3)


class TestProtocolSession(unittest.TestCase):
    def test_pass_when_band_tracks_the_induction(self):
        with tempfile.TemporaryDirectory() as d:
            sd = Path(d) / "2026-11-03_14-00-00_difficulty_ladder"
            sd.mkdir()
            planted = write_protocol_session(sd)
            res = run(sd)
            self.assertEqual(res["design"], "block")
            P = res["protocol"]
            self.assertIsNotNone(P)
            self.assertEqual((P["name"], P["seed"], P["stopped"]), ("difficulty_ladder", 42, False))
            self.assertEqual(P["planned_order"][:3], ["rest", "flow", "pause"])
            self.assertEqual(P["n_blocks_total"], 13)
            self.assertEqual(P["n_scored"], 8)                       # 6 task + 2 rest
            self.assertEqual(P["n_task_blocks"], 6)
            self.assertEqual(P["n_ratings"], 6)
            self.assertEqual(P["n_trials"], 300)
            rows = {r["index"]: r for r in P["blocks"]}
            self.assertEqual(len(rows), 13)
            for i, c, s, e, label in planted["schedule"]:
                r = rows[i]
                self.assertEqual(r["condition"], c)
                self.assertEqual(r["label"], label)
                self.assertAlmostEqual(r["duration_s"], e - s)
                self.assertEqual(r["scored"], c != "pause")
                if c == "pause":
                    continue
                self.assertAlmostEqual(r["eeg_mean"], EEG_LEVEL[c] + 0.003 * i, delta=0.02)
                self.assertGreater(r["eeg_artifacts"], 0)             # artifact rows were there ...
                self.assertLess(r["eeg_n"], (e - s - 10) / 2 + 1)      # ... and are not in the mean
                self.assertGreaterEqual(r["band_n"], 4 if c in FSS_LEVEL else 1)   # 1/min cadence
                self.assertFalse(r["band_after_end"])
                if c in FSS_LEVEL:
                    self.assertAlmostEqual(r["fss"], FSS_LEVEL[c] + 0.4, delta=0.6)      # 5.9 / 6.6 for flow etc.
                    self.assertEqual(r["fss_n_items"], 10)
                    self.assertEqual(r["demand"], DEMAND[c])
                    self.assertEqual(r["flow_overall"], FELT[c])
                    self.assertEqual(r["n_trials"], 50)
                    self.assertAlmostEqual(r["accuracy"], TASK[c][1], delta=0.02)
                    self.assertAlmostEqual(r["level_mean"], TASK[c][2], delta=0.1)
                    self.assertAlmostEqual(r["rt_median_ms"], TASK[c][3] + 200, delta=100)
                else:
                    self.assertIsNone(r["fss"])
                    self.assertEqual(r["n_trials"], 0)
            conds = {c["condition"]: c for c in P["conditions"]}
            self.assertEqual(list(conds), ["rest", "flow", "boredom", "overload"])
            self.assertEqual(conds["flow"]["n_blocks"], 2)
            self.assertEqual(conds["rest"]["n_blocks"], 2)
            self.assertAlmostEqual(conds["flow"]["eeg_mean"]["mean"], 0.75 + 0.003 * 4, delta=0.02)
            self.assertIsNotNone(conds["flow"]["eeg_mean"]["sd"])
            self.assertAlmostEqual(conds["boredom"]["accuracy"]["mean"], 1.0)
            self.assertIsNone(conds["rest"]["fss"]["mean"])
            M = P["manipulation"]
            self.assertTrue(M["ok"])
            self.assertGreater(M["fss"]["diff_flow_boredom"], 3)
            self.assertGreater(M["fss"]["diff_flow_overload"], 2)
            self.assertGreater(M["fss"]["t_flow_boredom"]["t"], 3)
            self.assertGreater(M["fss"]["t_flow_overload"]["t"], 3)
            self.assertEqual(M["fss"]["n"], {"flow": 2, "boredom": 2, "overload": 2})
            self.assertTrue(M["demand"]["ok"])
            self.assertTrue(M["accuracy"]["ok"])
            R = P["primary"]
            self.assertEqual(R["ordering"]["eeg"], "flow>overload>boredom")
            self.assertEqual(R["ordering"]["band"], "flow>overload>boredom")
            self.assertEqual(R["ordering"]["fss"], "flow>overload>boredom")
            self.assertTrue(R["eeg_flow_highest"] and R["band_flow_highest"] and R["ordering_ok"])
            self.assertEqual(R["n_blocks"], 6)
            self.assertGreaterEqual(R["block_rho"], 0.5)
            self.assertTrue(R["rho_ok"])
            Q = P["reference"]
            self.assertTrue(Q["ok"])
            self.assertTrue(Q["eeg_flow_highest"])
            self.assertGreater(Q["rho_eeg_fss"], 0.5)
            self.assertGreater(Q["rho_band_fss"], 0.5)
            self.assertEqual(Q["n_eeg_fss"], 6)
            self.assertEqual(res["verdict"], "PASS")
            self.assertIn("manipulation check passed", res["verdict_reason"])
            # epoch-level still there as secondary, with artifacts left out
            self.assertGreater(res["n_pairs"], 20)
            self.assertGreater(res["r"], 0.5)
            self.assertEqual(res["coverage"]["epochs_artifact"], int(planted["art"].sum()))
            ready = np.arange(len(planted["te"])) >= 15
            self.assertEqual(res["coverage"]["epochs_ready"], int((ready & ~planted["art"]).sum()))
            self.assertEqual(res["coverage"]["task_trials"], 300)
            self.assertEqual(res["other_marks"], [{"t": T0 + 30, "kind": "note", "value": "coffee", "note": ""}])
            s = res["summary"]
            self.assertEqual(s["verdict"], "PASS")
            self.assertEqual(s["design"], "block")
            self.assertEqual(s["protocol"], "difficulty_ladder")
            self.assertEqual(s["block_rho"], R["block_rho"])
            self.assertEqual(s["n_blocks"], 6)
            self.assertEqual(s["ordering"], R["ordering"])
            self.assertTrue(s["manipulation_ok"])
            self.assertTrue(s["reference_ok"])
            self.assertIn("epoch_verdict", s)
            for k in ("n_pairs", "r", "r_ci", "spearman", "mae", "bias", "best_lag_s", "r_at_best_lag", "pair_window_s"):
                self.assertIn(k, s)
            j = json.loads((sd / "report.json").read_text("utf-8"))
            self.assertEqual(j["protocol"]["verdict"], "PASS")
            self.assertEqual(j["summary"]["block_rho"], s["block_rho"])

    def test_markdown_sections_in_order(self):
        with tempfile.TemporaryDirectory() as d:
            sd = Path(d) / "s"
            sd.mkdir()
            write_protocol_session(sd)
            res = analyze(sd)
            md = render_markdown(res)
            order = ["**Verdict: PASS**", "## Protocol", "## Per-block table", "| block | condition |",
                     "## Per-condition table", "## Manipulation check", "## Primary result (block level)",
                     "## Reference check", "## Headline (epoch-level pairs, secondary)", "## EEG components vs band flow",
                     "## Lag sweep", "## Agreement", "## Felt-flow marks", "## Coverage", "## Verdict: **PASS**",
                     "## How to read this", "Katahira", "Pope"]
            pos = [md.index(s) for s in order]
            self.assertEqual(pos, sorted(pos), "sections out of order")
            self.assertIn("| 1 | flow |", md)
            self.assertIn("| 2 | pause (not scored) |", md)
            self.assertIn("flow>overload>boredom", md)
            self.assertIn("proxy", md)
            self.assertIn("not ground truth", md)
            self.assertIn("Welch", md)
            self.assertIn("No felt-flow marks", md)
            self.assertIn("coffee", md)
            self.assertNotIn("block_start", md)                       # protocol marks are not listed as "other"
            self.assertIn("task trials | 300", md)

    def test_flat_band_fails(self):
        with tempfile.TemporaryDirectory() as d:
            sd = Path(d) / "s"
            sd.mkdir()
            write_protocol_session(sd, band="flat")
            res = analyze(sd)
            P = res["protocol"]
            self.assertTrue(P["manipulation"]["ok"])
            self.assertTrue(P["reference"]["ok"])
            self.assertTrue(P["primary"]["eeg_flow_highest"])
            self.assertEqual(res["verdict"], "FAIL")
            self.assertFalse(P["primary"]["ordering_ok"] and P["primary"]["rho_ok"])
            self.assertTrue("band flow not highest" in res["verdict_reason"] or "block rho" in res["verdict_reason"])
            self.assertNotIn("reference failed", res["verdict_reason"])
            md = render_markdown(res)
            self.assertIn("## Verdict: **FAIL**", md)
            self.assertEqual(res["summary"]["manipulation_ok"], True)

    def test_band_anti_tracking_fails_with_rho(self):
        with tempfile.TemporaryDirectory() as d:
            sd = Path(d) / "s"
            sd.mkdir()
            write_protocol_session(sd)
            rows = read_rows(sd / "band.csv")
            for r in rows[1:]:
                r[1] = f"{1 - float(r[1]):.4f}"
            write_rows(sd / "band.csv", rows)
            res = analyze(sd)
            self.assertEqual(res["verdict"], "FAIL")
            self.assertLess(res["protocol"]["primary"]["block_rho"], -0.5)
            self.assertEqual(res["protocol"]["primary"]["ordering"]["band"], "boredom>overload>flow")
            self.assertIn("band flow not highest in flow", res["verdict_reason"])
            self.assertIn("block rho = -", res["verdict_reason"])

    def test_failed_induction_is_inconclusive(self):
        with tempfile.TemporaryDirectory() as d:
            sd = Path(d) / "s"
            sd.mkdir()
            write_protocol_session(sd, fss="flat")
            res = analyze(sd)
            P = res["protocol"]
            self.assertFalse(P["manipulation"]["ok"])
            self.assertIn("not rated highest", P["manipulation"]["reason"])
            self.assertTrue(P["primary"]["ordering_ok"])              # EEG and band would have passed ...
            self.assertTrue(P["primary"]["rho_ok"])
            self.assertEqual(res["verdict"], "INCONCLUSIVE")           # ... but the induction did not work
            self.assertIn("induction did not work", res["verdict_reason"])
            self.assertIn("cannot validate the band", res["verdict_reason"])
            self.assertEqual(res["summary"]["manipulation_ok"], False)
            md = render_markdown(res)
            self.assertIn("**FAILED**", md)
            self.assertIn("## Verdict: **INCONCLUSIVE**", md)

    def test_too_few_blocks_and_no_band_are_inconclusive(self):
        with tempfile.TemporaryDirectory() as d:
            sd = Path(d) / "s"
            sd.mkdir()
            write_protocol_session(sd, rounds=1)
            res = analyze(sd)
            self.assertEqual(res["protocol"]["n_task_blocks"], 3)
            self.assertEqual(res["verdict"], "INCONCLUSIVE")
            self.assertIn("only 3 usable task blocks", res["verdict_reason"])
            render_markdown(res)
            sd2 = Path(d) / "s2"
            sd2.mkdir()
            write_protocol_session(sd2, with_band=False)
            res = analyze(sd2)
            self.assertEqual(res["verdict"], "INCONCLUSIVE")
            self.assertIn("no usable band sample", res["verdict_reason"])
            self.assertEqual(res["protocol"]["primary"]["n_blocks"], 0)
            self.assertTrue(res["protocol"]["reference"]["ok"])      # the EEG side is still checked
            md = render_markdown(res)
            self.assertIn("| n/a |", md)

    def test_missing_ratings_and_task(self):
        with tempfile.TemporaryDirectory() as d:
            sd = Path(d) / "s"
            sd.mkdir()
            write_protocol_session(sd, with_task=False)
            rows = read_rows(sd / "marks.csv")
            rows = [r for r in rows if r[1] != "rating"]
            write_rows(sd / "marks.csv", rows)
            res = analyze(sd)
            P = res["protocol"]
            self.assertEqual(P["n_ratings"], 0)
            self.assertIsNone(P["manipulation"]["ok"])
            self.assertIsNone(P["manipulation"]["demand"]["ok"])
            self.assertIsNone(P["manipulation"]["accuracy"]["ok"])
            self.assertEqual(res["verdict"], "INCONCLUSIVE")
            self.assertIn("manipulation check not possible", res["verdict_reason"])
            self.assertTrue(any("without a self-report" in n for n in res["notes"]))
            self.assertIsNone(P["reference"]["rho_eeg_fss"])
            self.assertTrue(P["reference"]["ok"])                    # ordering alone still holds
            self.assertEqual(P["primary"]["ordering"]["fss"], None)
            self.assertEqual(res["coverage"]["task_trials"], 0)
            render_markdown(res)

    def test_stopped_early_session(self):
        with tempfile.TemporaryDirectory() as d:
            sd = Path(d) / "s"
            sd.mkdir()
            write_protocol_session(sd, stop_after=7)                 # rest + 4 task blocks
            rows = read_rows(sd / "marks.csv")
            rows[-1][1] = "protocol_stop"
            rows[-3][3] += " aborted"                                 # last block_end
            write_rows(sd / "marks.csv", rows)
            res = analyze(sd)
            P = res["protocol"]
            self.assertTrue(P["stopped"])
            self.assertTrue(P["blocks"][-1]["aborted"])
            self.assertEqual(P["n_task_blocks"], 4)
            self.assertEqual(res["verdict"], "INCONCLUSIVE")
            self.assertIn("aborted", render_markdown(res))

    def test_cli_quiet_line_for_protocol_session(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()) as out:
            sd = Path(d) / "2026-11-03_14-00-00_ladder"
            sd.mkdir()
            write_protocol_session(sd)
            self.assertEqual(report.main([str(sd), "--quiet"]), 0)
            line = out.getvalue().strip()
            self.assertTrue(line.startswith("2026-11-03_14-00-00_ladder: PASS (blocks = 6, rho = "), line)


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
