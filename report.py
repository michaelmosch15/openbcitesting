"""Offline validation report for one recorded session: how well does the BioProxal band's
flow score track the EEG flow index (the literature-based reference proxy)?

    python report.py sessions/<name>              # writes report.md + report.json into the folder
    python report.py sessions/<name> --json       # also print the result as JSON
    python report.py --all [sessions]             # every session folder without a report.md
    python report.py --all --force                # regenerate all
    python report.py sessions/<name> --include-held   # keep band samples flagged "held"

Function API (used by the server for GET /api/report):
    analyze(session_dir) -> dict        the numbers (JSON-serialisable, None = n/a)
    render_markdown(result) -> str
    run(session_dir) -> dict            analyze + write report.md / report.json

Pre-registered gates (do not move them after looking at the data):
    PASS          n >= 20 pairs, Pearson r >= 0.5 and the bootstrap 95 % CI lower bound > 0
    INCONCLUSIVE  fewer than 20 pairs
    FAIL          otherwise - reported as such, honestly

numpy + stdlib only (no pandas / scipy on this laptop).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

VERSION = "0.1.0"
COMPONENTS = ["theta_frontal_rel", "alpha_posterior_rel", "beta_rel", "engagement_index", "theta_alpha_ratio"]
LAGS = list(range(-180, 181, 10))
FLOW_THRESHOLD = 0.6
FELT_EEG_WINDOW_S = 60.0
FELT_BAND_WINDOW_S = 90.0
GATE_MIN_PAIRS = 20
GATE_MIN_R = 0.5
N_BOOT = 1000
CI_MIN_PAIRS = 8


# --- loading ------------------------------------------------------------------------
def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    """Header + rows as dicts (strings). Missing / empty file -> ([], [])."""
    if not path.exists():
        return [], []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            header = list(r.fieldnames or [])
            rows = [row for row in r if row]
    except (OSError, csv.Error):
        return [], []
    return header, rows


def _f(v) -> float:
    """float or nan."""
    if v is None:
        return math.nan
    try:
        x = float(str(v).strip())
    except ValueError:
        return math.nan
    return x if math.isfinite(x) else math.nan


def _bool(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "t", "y")


def _col(rows: list[dict], name: str) -> np.ndarray:
    return np.array([_f(r.get(name)) for r in rows], float)


def load_epochs(session_dir: Path) -> dict:
    header, rows = _read_csv(session_dir / "epochs.csv")
    t = _col(rows, "t")
    keep = np.isfinite(t)
    order = np.argsort(t[keep], kind="stable")
    rows = [r for r, k in zip(rows, keep) if k]
    rows = [rows[i] for i in order]
    out = {"header": header, "n_rows": len(rows), "t": t[keep][order],
           "flow": _col(rows, "flow_eeg") if rows else np.zeros(0)}
    if "baseline_ready" in header:
        out["ready"] = np.array([_bool(r.get("baseline_ready")) for r in rows], bool)
    else:
        out["ready"] = np.ones(len(rows), bool)
    out["components"] = {c: (_col(rows, c) if c in header else np.full(len(rows), math.nan)) for c in COMPONENTS}
    return out


def load_band(session_dir: Path) -> dict:
    header, rows = _read_csv(session_dir / "band.csv")
    t = _col(rows, "t")
    keep = np.isfinite(t)
    order = np.argsort(t[keep], kind="stable")
    rows = [r for r, k in zip(rows, keep) if k]
    rows = [rows[i] for i in order]
    return {"header": header, "n_rows": len(rows), "t": t[keep][order],
            "flow": _col(rows, "flow") if rows else np.zeros(0),
            "hr": _col(rows, "hr") if rows else np.zeros(0),
            "held": np.array([_bool(r.get("held")) for r in rows], bool),
            "settling": np.array([_bool(r.get("settling")) for r in rows], bool)}


def load_marks(session_dir: Path) -> list[dict]:
    _, rows = _read_csv(session_dir / "marks.csv")
    out = []
    for r in rows:
        t = _f(r.get("t"))
        if not math.isfinite(t):
            continue
        out.append({"t": t, "kind": (r.get("kind") or "").strip(), "value": _f(r.get("value")),
                    "raw_value": r.get("value"), "note": r.get("note") or ""})
    out.sort(key=lambda m: m["t"])
    return out


# --- numerics -----------------------------------------------------------------------
def _py(v):
    """numpy scalar -> plain Python; nan -> None."""
    if v is None:
        return None
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    v = float(v)
    return v if math.isfinite(v) else None


def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return None
    xc = x - x.mean()
    yc = y - y.mean()
    sx = math.sqrt(float((xc * xc).sum()))
    sy = math.sqrt(float((yc * yc).sum()))
    if sx < 1e-12 or sy < 1e-12:
        return None
    r = float((xc * yc).sum()) / (sx * sy)
    return max(-1.0, min(1.0, r))


def rankdata(a) -> np.ndarray:
    """Average ranks (1-based), ties share the mean rank - like scipy.stats.rankdata."""
    a = np.asarray(a, float)
    n = len(a)
    if n == 0:
        return np.zeros(0)
    order = np.argsort(a, kind="stable")
    s = a[order]
    uniq, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    start = cum - counts
    avg = (start + 1 + cum) / 2.0
    ranks = np.empty(n)
    ranks[order] = avg[inv]
    return ranks


def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return None
    return pearson(rankdata(x[m]), rankdata(y[m]))


def bootstrap_ci(x, y, n_boot: int = N_BOOT, seed: int = 0):
    """95 % percentile bootstrap CI of Pearson r; None when n < CI_MIN_PAIRS or degenerate."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < CI_MIN_PAIRS:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    xs = x[idx]
    ys = y[idx]
    xc = xs - xs.mean(axis=1, keepdims=True)
    yc = ys - ys.mean(axis=1, keepdims=True)
    den = np.sqrt((xc * xc).sum(axis=1) * (yc * yc).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 1e-12, (xc * yc).sum(axis=1) / np.where(den > 1e-12, den, 1.0), np.nan)
    r = r[np.isfinite(r)]
    if len(r) < n_boot // 2:
        return None
    lo, hi = np.percentile(r, [2.5, 97.5])
    return [float(max(-1.0, lo)), float(min(1.0, hi))]


def cohen_kappa(a, b):
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    n = len(a)
    if n == 0:
        return None
    po = float((a == b).mean())
    pa, pb = float(a.mean()), float(b.mean())
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 1.0 - 1e-12:
        return None
    return (po - pe) / (1 - pe)


def window_means(te: np.ndarray, values: np.ndarray, t_end: np.ndarray, width: float,
                 shift: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Mean of `values` (indexed by sorted times te) over [t_end - shift - width, t_end - shift]
    for every t_end; nan where the window holds no epoch. Returns (means, counts)."""
    t_end = np.asarray(t_end, float)
    if len(te) == 0 or len(t_end) == 0:
        return np.full(len(t_end), math.nan), np.zeros(len(t_end), int)
    v = np.where(np.isfinite(values), values, 0.0)
    ok = np.isfinite(values).astype(float)
    cs = np.concatenate([[0.0], np.cumsum(v)])
    cn = np.concatenate([[0.0], np.cumsum(ok)])
    lo = np.searchsorted(te, t_end - shift - width, side="left")
    hi = np.searchsorted(te, t_end - shift, side="right")
    n = cn[hi] - cn[lo]
    means = np.full(len(t_end), math.nan)
    m = n > 0
    means[m] = (cs[hi[m]] - cs[lo[m]]) / n[m]
    return means, n.astype(int)


# --- analysis -----------------------------------------------------------------------
def analyze(session_dir: Path | str, include_held: bool = False) -> dict:
    session_dir = Path(session_dir)
    ep = load_epochs(session_dir)
    bd = load_band(session_dir)
    marks = load_marks(session_dir)
    meta = {}
    sj = session_dir / "session.json"
    if sj.exists():
        try:
            meta = json.loads(sj.read_text("utf-8"))
        except (OSError, ValueError):
            meta = {}

    notes: list[str] = []
    # reference series: baseline-ready epochs with a finite flow index
    e_ok = ep["ready"] & np.isfinite(ep["flow"])
    te = ep["t"][e_ok]
    fe = ep["flow"][e_ok]
    comp = {c: ep["components"][c][e_ok] for c in COMPONENTS}
    n_warm = int((~ep["ready"]).sum())
    if n_warm:
        notes.append(f"{n_warm} EEG epochs before the baseline was ready were left out of the pairing.")

    # band cadence -> pairing window
    tb_all = bd["t"]
    if len(tb_all) >= 2:
        med_int = float(np.median(np.diff(tb_all)))
    else:
        med_int = None
    W = 60.0 if (med_int is None or med_int > 20.0) else 10.0
    window_reason = (f"median band interval {med_int:.1f} s" if med_int is not None
                     else "fewer than 2 band samples (default)")

    has_flow = np.isfinite(bd["flow"])
    held = bd["held"]
    use = has_flow & (np.ones(len(held), bool) if include_held else ~held)
    n_skipped_held = int((has_flow & held).sum()) if not include_held else 0
    tb = tb_all[use]
    fb = bd["flow"][use]

    # pairs at lag 0
    eeg_mean, n_ep = window_means(te, fe, tb, W)
    pm = np.isfinite(eeg_mean)
    px, py = fb[pm], eeg_mean[pm]          # band, eeg
    pt = tb[pm]
    n_pairs = int(pm.sum())

    r = pearson(px, py)
    r_ci = bootstrap_ci(px, py)
    rho = spearman(px, py)
    diff = px - py
    mae = float(np.abs(diff).mean()) if n_pairs else None
    bias = float(diff.mean()) if n_pairs else None
    sd_band = float(px.std()) if n_pairs >= 2 else None
    sd_eeg = float(py.std()) if n_pairs >= 2 else None

    # components vs band flow (same windows)
    components = {}
    for c in COMPONENTS:
        cm, _ = window_means(te, comp[c], tb, W)
        components[c] = {"r": pearson(fb, cm), "n": int((np.isfinite(cm) & np.isfinite(fb)).sum())}

    # lag sweep: positive lag = band lags EEG (band at t compared with EEG ending at t - lag)
    lags = []
    for lag in LAGS:
        m_lag, _ = window_means(te, fe, tb, W, shift=float(lag))
        lags.append({"lag_s": lag, "r": pearson(fb, m_lag),
                     "n": int((np.isfinite(m_lag) & np.isfinite(fb)).sum())})
    valid = [l for l in lags if l["r"] is not None]
    if valid:
        best = max(valid, key=lambda l: (l["r"], -abs(l["lag_s"])))
        best_lag, r_best = best["lag_s"], best["r"]
    else:
        best_lag, r_best = None, None

    # agreement on "in flow"
    if n_pairs:
        a = px >= FLOW_THRESHOLD
        b = py >= FLOW_THRESHOLD
        agreement = {"threshold": FLOW_THRESHOLD, "n": n_pairs, "accuracy": float((a == b).mean()),
                     "kappa": cohen_kappa(a, b),
                     "both": int((a & b).sum()), "band_only": int((a & ~b).sum()),
                     "eeg_only": int((~a & b).sum()), "neither": int((~a & ~b).sum())}
    else:
        agreement = {"threshold": FLOW_THRESHOLD, "n": 0, "accuracy": None, "kappa": None,
                     "both": 0, "band_only": 0, "eeg_only": 0, "neither": 0}

    # felt-flow marks
    felt_marks = [m for m in marks if m["kind"] == "felt_flow" and math.isfinite(m["value"])]
    felt_rows = []
    for m in felt_marks:
        em, _ = window_means(te, fe, np.array([m["t"]]), FELT_EEG_WINDOW_S)
        eeg_v = _py(em[0])
        band_v = None
        band_dt = None
        if len(tb):
            j = int(np.argmin(np.abs(tb - m["t"])))
            if abs(tb[j] - m["t"]) <= FELT_BAND_WINDOW_S:
                band_v, band_dt = float(fb[j]), float(tb[j] - m["t"])
        felt_rows.append({"t": m["t"], "felt": float(m["value"]), "eeg": eeg_v, "band": band_v,
                          "band_dt_s": band_dt, "note": m["note"]})
    fv = np.array([f["felt"] for f in felt_rows], float)
    fe_v = np.array([math.nan if f["eeg"] is None else f["eeg"] for f in felt_rows], float)
    fb_v = np.array([math.nan if f["band"] is None else f["band"] for f in felt_rows], float)
    felt = {"n_marks": len(felt_rows), "n_eeg": int(np.isfinite(fe_v).sum()),
            "n_band": int(np.isfinite(fb_v).sum()),
            "r_eeg": pearson(fv, fe_v), "r_band": pearson(fv, fb_v),
            "mae_eeg": _py(np.nanmean(np.abs(fv - fe_v))) if np.isfinite(fe_v).any() else None,
            "mae_band": _py(np.nanmean(np.abs(fv - fb_v))) if np.isfinite(fb_v).any() else None,
            "marks": felt_rows}
    other_marks = [{"t": m["t"], "kind": m["kind"], "value": m["raw_value"], "note": m["note"]}
                   for m in marks if m["kind"] != "felt_flow"]

    # coverage
    ts = [x for x in (ep["t"], tb_all, np.array([m["t"] for m in marks])) if len(x)]
    if ts:
        t0 = float(min(x.min() for x in ts))
        t1 = float(max(x.max() for x in ts))
    else:
        t0 = t1 = None
    duration = (t1 - t0) if ts else 0.0
    pct_both = None
    pct_eeg = None
    pct_band = None
    if ts and duration > 0:
        step = 10.0
        edges = np.arange(t0, t1 + step, step)
        starts, ends = edges[:-1], edges[1:]
        te_all = ep["t"]
        has_e = (np.searchsorted(te_all, ends, "right") - np.searchsorted(te_all, starts, "left")) > 0
        tb_f = tb_all[has_flow]
        has_b = (np.searchsorted(tb_f, ends, "right") - np.searchsorted(tb_f, starts - W, "left")) > 0
        pct_both = float((has_e & has_b).mean() * 100.0)
        pct_eeg = float(has_e.mean() * 100.0)
        pct_band = float(has_b.mean() * 100.0)
    coverage = {"t_start": t0, "t_end": t1, "duration_s": float(duration),
                "epochs": int(ep["n_rows"]), "epochs_ready": int(e_ok.sum()), "epochs_warmup": n_warm,
                "band_samples": int(bd["n_rows"]), "band_with_flow": int(has_flow.sum()),
                "band_held": int((has_flow & held).sum()), "band_skipped_held": n_skipped_held,
                "band_settling": int(bd["settling"].sum()),
                "band_median_interval_s": med_int, "marks": len(marks), "felt_marks": len(felt_rows),
                "pct_time_eeg": pct_eeg, "pct_time_band": pct_band, "pct_time_both": pct_both}
    if not ep["n_rows"]:
        notes.append("epochs.csv is missing or empty: no EEG reference.")
    if not bd["n_rows"]:
        notes.append("band.csv is missing or empty: no band samples.")
    elif not has_flow.any():
        notes.append("band.csv has no sample with a flow value (all 'no reading').")

    # verdict against the pre-registered gates
    gates = {"min_pairs": GATE_MIN_PAIRS, "min_r": GATE_MIN_R, "ci_lower_gt": 0.0}
    if n_pairs < GATE_MIN_PAIRS:
        verdict = "INCONCLUSIVE"
        reason = f"only {n_pairs} pairs (need >= {GATE_MIN_PAIRS})"
    elif r is None:
        verdict = "FAIL"
        reason = "Pearson r undefined (constant series)"
    elif r >= GATE_MIN_R and r_ci is not None and r_ci[0] > 0:
        verdict = "PASS"
        reason = f"r = {r:.2f} >= {GATE_MIN_R} and CI lower bound {r_ci[0]:.2f} > 0 with n = {n_pairs}"
    else:
        verdict = "FAIL"
        parts = []
        if r < GATE_MIN_R:
            parts.append(f"r = {r:.2f} < {GATE_MIN_R}")
        if r_ci is None:
            parts.append("no bootstrap CI")
        elif r_ci[0] <= 0:
            parts.append(f"CI lower bound {r_ci[0]:.2f} <= 0")
        reason = "; ".join(parts) or "gates not met"

    result = {
        "session": session_dir.name, "dir": str(session_dir.resolve()),
        "report_version": VERSION, "generated_at": datetime.now().isoformat(timespec="seconds"),
        "include_held": include_held,
        "pair_window_s": W, "pair_window_reason": window_reason,
        "n_pairs": n_pairs, "r": r, "r_ci": r_ci, "spearman": rho, "mae": mae, "bias": bias,
        "sd_band": sd_band, "sd_eeg": sd_eeg,
        "best_lag_s": best_lag, "r_at_best_lag": r_best, "lags": [[l["lag_s"], l["r"]] for l in lags],
        "lag_table": lags,
        "components": components,
        "agreement": agreement,
        "felt": felt,
        "other_marks": other_marks,
        "coverage": coverage,
        "gates": gates, "verdict": verdict, "verdict_reason": reason,
        "notes": notes,
        "meta": {k: meta.get(k) for k in ("board", "labels", "started", "fs", "band_source", "config")
                 if k in meta},
        "pairs": [[float(a), float(b), float(c)] for a, b, c in zip(pt, px, py)],
    }
    result["summary"] = {k: result[k] for k in ("n_pairs", "r", "r_ci", "spearman", "mae", "bias",
                                                 "best_lag_s", "r_at_best_lag", "pair_window_s",
                                                 "verdict", "verdict_reason")}
    return _clean(result)


def _clean(o):
    """Recursively convert numpy scalars / nan to JSON-friendly Python values."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_clean(v) for v in o.tolist()]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return _py(o)
    return o


# --- markdown -----------------------------------------------------------------------
def _n(v, nd=3):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    return f"{v:.{nd}f}"


def _ci(ci):
    return "n/a" if ci is None else f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def _hms(s):
    if s is None:
        return "n/a"
    s = int(round(s))
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _ts(t):
    if t is None:
        return "n/a"
    try:
        return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")
    except (OverflowError, OSError, ValueError):
        return f"{t:.0f}"


def render_markdown(res: dict) -> str:
    cov = res["coverage"]
    agr = res["agreement"]
    felt = res["felt"]
    L = []
    L.append(f"# Flow validation report: {res['session']}")
    L.append("")
    L.append(f"Generated {res['generated_at']} by report.py {res['report_version']}. "
             f"Session {_ts(cov['t_start'])} to {_ts(cov['t_end'])} ({_hms(cov['duration_s'])}).")
    L.append("")
    L.append(f"## Verdict: **{res['verdict']}**")
    L.append("")
    L.append(f"{res['verdict_reason']}.")
    L.append("")
    L.append("Pre-registered gates (fixed before looking at the data): **PASS** if Pearson r >= "
             f"{res['gates']['min_r']} and the bootstrap 95 % CI lower bound > 0 with n >= "
             f"{res['gates']['min_pairs']} pairs; **INCONCLUSIVE** if n < {res['gates']['min_pairs']}; "
             "otherwise **FAIL**. A FAIL is reported as a FAIL.")
    L.append("")
    L.append("## Headline")
    L.append("")
    L.append("| metric | value |")
    L.append("|---|---|")
    L.append(f"| n pairs | {res['n_pairs']} |")
    L.append(f"| Pearson r [95 % CI] | {_n(res['r'])} {_ci(res['r_ci'])} |")
    L.append(f"| Spearman rho | {_n(res['spearman'])} |")
    L.append(f"| MAE | {_n(res['mae'])} |")
    L.append(f"| mean bias (band - EEG) | {_n(res['bias'], 3)} |")
    L.append(f"| best lag | {_n(res['best_lag_s'], 0)} s (r = {_n(res['r_at_best_lag'])}) |")
    L.append(f"| pairing window | {res['pair_window_s']:.0f} s ({res['pair_window_reason']}) |")
    L.append(f"| held band samples | {'included' if res['include_held'] else 'excluded'} "
             f"({cov['band_skipped_held']} skipped) |")
    L.append("")
    L.append("## EEG components vs band flow")
    L.append("")
    L.append("Pearson r of each raw EEG component (mean over the same pairing window) against the band's flow score.")
    L.append("")
    L.append("| component | r | n |")
    L.append("|---|---|---|")
    for c in COMPONENTS:
        L.append(f"| {c} | {_n(res['components'][c]['r'])} | {res['components'][c]['n']} |")
    L.append("")
    L.append("## Lag sweep")
    L.append("")
    L.append("Band series shifted against the EEG index; positive lag = band lags EEG "
             f"(band at t vs EEG ending at t - lag). Best lag {_n(res['best_lag_s'], 0)} s "
             f"(r = {_n(res['r_at_best_lag'])}); full sweep -180..+180 s in 10 s steps in report.json.")
    L.append("")
    L.append("| lag (s) | r | n |")
    L.append("|---|---|---|")
    for row in res["lag_table"]:
        if row["lag_s"] % 30 == 0:
            L.append(f"| {row['lag_s']:+d} | {_n(row['r'])} | {row['n']} |")
    L.append("")
    L.append(f"## Agreement on \"in flow\" (>= {agr['threshold']})")
    L.append("")
    L.append("| metric | value |")
    L.append("|---|---|")
    L.append(f"| n | {agr['n']} |")
    L.append(f"| accuracy | {_n(agr['accuracy'])} |")
    L.append(f"| Cohen's kappa | {_n(agr['kappa'])} |")
    L.append(f"| both in flow / band only / EEG only / neither | {agr['both']} / {agr['band_only']} / "
             f"{agr['eeg_only']} / {agr['neither']} |")
    L.append("")
    L.append("## Felt-flow marks")
    L.append("")
    if felt["n_marks"]:
        L.append(f"{felt['n_marks']} felt-flow marks. EEG = mean index over the {FELT_EEG_WINDOW_S:.0f} s "
                 f"before the mark; band = nearest sample within {FELT_BAND_WINDOW_S:.0f} s.")
        L.append("")
        L.append("| metric | vs EEG | vs band |")
        L.append("|---|---|---|")
        L.append(f"| n | {felt['n_eeg']} | {felt['n_band']} |")
        L.append(f"| Pearson r | {_n(felt['r_eeg'])} | {_n(felt['r_band'])} |")
        L.append(f"| MAE | {_n(felt['mae_eeg'])} | {_n(felt['mae_band'])} |")
        L.append("")
        L.append("| time | felt | EEG | band | note |")
        L.append("|---|---|---|---|---|")
        for m in felt["marks"]:
            L.append(f"| {_ts(m['t'])} | {_n(m['felt'], 2)} | {_n(m['eeg'], 2)} | {_n(m['band'], 2)} | {m['note']} |")
    else:
        L.append("No felt-flow marks in this session.")
    if res["other_marks"]:
        L.append("")
        L.append("Other marks: " + "; ".join(f"{_ts(m['t'])} {m['kind']}"
                                             + (f" ({m['value']})" if m['value'] not in (None, "") else "")
                                             for m in res["other_marks"]))
    L.append("")
    L.append("## Coverage")
    L.append("")
    L.append("| metric | value |")
    L.append("|---|---|")
    L.append(f"| session duration | {_hms(cov['duration_s'])} |")
    L.append(f"| EEG epochs (baseline ready / warm-up) | {cov['epochs']} ({cov['epochs_ready']} / {cov['epochs_warmup']}) |")
    L.append(f"| band samples (with flow / held / settling) | {cov['band_samples']} ({cov['band_with_flow']} / "
             f"{cov['band_held']} / {cov['band_settling']}) |")
    L.append(f"| median band interval | {_n(cov['band_median_interval_s'], 1)} s |")
    L.append(f"| time with EEG / band / both | {_n(cov['pct_time_eeg'], 0)} % / {_n(cov['pct_time_band'], 0)} % / "
             f"{_n(cov['pct_time_both'], 0)} % |")
    L.append(f"| marks (felt-flow) | {cov['marks']} ({cov['felt_marks']}) |")
    if res["notes"]:
        L.append("")
        for n in res["notes"]:
            L.append(f"- {n}")
    L.append("")
    L.append("## How to read this")
    L.append("")
    L.append("The EEG flow index is a literature-based proxy, not ground truth: it rises with frontal "
             "theta, prefers moderate posterior alpha and falls with beta (Katahira 2018; Ewing 2016; "
             "Pope 1995 engagement index), each z-scored against the wearer's own rolling baseline. "
             "A high r means the band's score moves with that proxy over this session; it does not by "
             "itself prove either signal measures flow. A low r can mean the band is wrong, the proxy "
             "is wrong, the electrodes were poor, or the session simply had no flow to find - check "
             "coverage, the component table and the felt-flow marks before concluding. The lag sweep "
             "shows whether the band trails the EEG (positive lag) as its 60 s averaging would predict. "
             "Pairs are formed at the band's cadence (EEG index averaged over the pairing window before "
             "each band sample), band samples flagged held are skipped unless --include-held, and "
             "epochs before the EEG baseline was ready are not used.")
    L.append("")
    return "\n".join(L)


# --- entry points -------------------------------------------------------------------
def run(session_dir: Path | str, include_held: bool = False) -> dict:
    session_dir = Path(session_dir)
    res = analyze(session_dir, include_held=include_held)
    md = render_markdown(res)
    (session_dir / "report.md").write_text(md, "utf-8")
    (session_dir / "report.json").write_text(json.dumps(res, indent=2), "utf-8")
    res["markdown"] = md
    return res


def _session_dirs(root: Path, force: bool) -> list[Path]:
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if not any((d / f).exists() for f in ("epochs.csv", "band.csv", "session.json")):
            continue
        if force or not (d / "report.md").exists():
            out.append(d)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="*", help="session folder(s), e.g. sessions/2026-11-03_14-02-11_desk")
    ap.add_argument("--all", nargs="?", const="sessions", metavar="ROOT",
                    help="run every session under ROOT (default sessions/) that has no report.md")
    ap.add_argument("--force", action="store_true", help="with --all: regenerate existing reports too")
    ap.add_argument("--json", action="store_true", help="print the result as JSON instead of markdown")
    ap.add_argument("--include-held", action="store_true", help="keep band samples flagged held")
    ap.add_argument("--quiet", action="store_true", help="only print the verdict line per session")
    a = ap.parse_args(argv)

    dirs = [Path(s) for s in a.session]
    if a.all is not None:
        found = _session_dirs(Path(a.all), a.force)
        dirs += found
        if not found and not a.session:
            print(f"[report] no session under {Path(a.all)} needs a report")
            return 0
    if not dirs:
        ap.error("give a session folder or --all")
    rc = 0
    for d in dirs:
        if not d.is_dir():
            print(f"[report] not a folder: {d}", file=sys.stderr)
            rc = 2
            continue
        res = run(d, include_held=a.include_held)
        if a.json:
            out = {k: v for k, v in res.items() if k != "markdown"}
            print(json.dumps(out, indent=2))
        elif a.quiet:
            print(f"{d.name}: {res['verdict']} (n = {res['n_pairs']}, r = {_n(res['r'])}, "
                  f"window {res['pair_window_s']:.0f} s)")
        else:
            print(f"[report] {d.name}: pairing window {res['pair_window_s']:.0f} s ({res['pair_window_reason']})")
            print(res["markdown"])
            print(f"[report] wrote {d / 'report.md'} and report.json")
    return rc


if __name__ == "__main__":
    sys.exit(main())
