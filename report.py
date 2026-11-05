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

Two designs, picked by what the session contains:

Protocol session (marks.csv has protocol_start / block_start / block_end / rating marks from
protocol.py): the PRIMARY result is at the block level, blocks being the independent units.
Pre-registered gates (do not move them after looking at the data):
    manipulation check  FSS (Flow Short Scale) mean is higher in flow blocks than in boredom
                        AND than in overload blocks (demand item and task accuracy reported
                        as supporting checks). If this fails the induction did not work and
                        the session cannot validate the band -> INCONCLUSIVE (not FAIL).
    (a) ordering        EEG index block mean highest in flow (vs boredom AND overload) AND
                        band flow block mean highest in flow (vs both)
    (b) block rho       Spearman rho between EEG and band block means >= 0.5 over >= 6 task
                        blocks (boredom / flow / overload / work; rest and pause excluded)
    PASS          manipulation check passed and (a) and (b) hold
    FAIL          manipulation check passed and (a) or (b) fails
    INCONCLUSIVE  fewer than 6 usable task blocks, no band data, or the manipulation
                  check failed / could not be run
The epoch-level correlation (below) is kept as a secondary, descriptive result: epochs are
autocorrelated, so its CI is not inferential.

Session without protocol marks (free recording): the old epoch-level gates apply:
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
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

VERSION = "0.2.0"
COMPONENTS = ["theta_frontal_rel", "alpha_cp_rel", "beta_rel", "engagement_index", "theta_alpha_ratio"]
COMPONENT_ALIASES = {"alpha_cp_rel": ("alpha_fc_rel", "alpha_posterior_rel")}     # older sessions
LAGS = list(range(-180, 181, 10))
FLOW_THRESHOLD = 0.6
FELT_EEG_WINDOW_S = 60.0
FELT_BAND_WINDOW_S = 90.0
GATE_MIN_PAIRS = 20
GATE_MIN_R = 0.5
N_BOOT = 1000
CI_MIN_PAIRS = 8

# block level (protocol sessions)
FSS_ITEMS = ["challenge", "fluent", "time", "concentrate", "clear", "absorbed", "automatic", "know", "control", "lost"]
TASK_CONDITIONS = ("boredom", "flow", "overload", "work")
BLOCK_TRANSITION_S = 10.0        # the first seconds of a block are a transition and are skipped
BLOCK_BAND_AFTER_S = 60.0        # a block with no band sample inside uses samples up to this long after it
BLOCK_MIN_S = 30.0               # shorter blocks are not scored
BLOCK_MIN_N = 6                  # task blocks with both an EEG and a band mean
BLOCK_MIN_RHO = 0.5
ACCURACY_FLOW_RANGE = (0.55, 0.9)   # the adaptive task targets ~70-85 % correct


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
           "flow": _col(rows, "flow_eeg") if rows else np.zeros(0),
           "flow_alt": (_col(rows, "flow_eeg_alt") if ("flow_eeg_alt" in header and rows)
                        else np.full(len(rows), math.nan))}
    if "baseline_ready" in header:
        out["ready"] = np.array([_bool(r.get("baseline_ready")) for r in rows], bool)
    else:
        out["ready"] = np.ones(len(rows), bool)
    if "artifact" in header:
        out["artifact"] = np.array([_bool(r.get("artifact")) for r in rows], bool)
    else:
        out["artifact"] = np.zeros(len(rows), bool)
    comps = {}
    for c in COMPONENTS:
        name = c if c in header else next((a for a in COMPONENT_ALIASES.get(c, ()) if a in header), None)
        comps[c] = _col(rows, name) if name else np.full(len(rows), math.nan)
    out["components"] = comps
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
            "settling": np.array([_bool(r.get("settling")) for r in rows], bool),
            # v0.2 sessions also record these; older files simply have no such column
            "asleep": np.array([_bool(r.get("asleep")) for r in rows], bool),
            "poor_contact": np.array([_bool(r.get("poor_contact")) for r in rows], bool)}


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


def load_task(session_dir: Path) -> list[dict]:
    """task.csv rows (one arithmetic trial each); rows without a block index are dropped."""
    _, rows = _read_csv(session_dir / "task.csv")
    out = []
    for r in rows:
        bi = _f(r.get("block_index"))
        if not math.isfinite(bi):
            continue
        out.append({"t": _f(r.get("t")), "block": int(bi), "condition": (r.get("condition") or "").strip(),
                    "task": (r.get("task") or "").strip(), "level": _f(r.get("level")),
                    "correct": _bool(r.get("correct")), "rt_ms": _f(r.get("rt_ms"))})
    return out


def _block_index_from_note(note: str, default):
    m = re.search(r"block=(\d+)", note or "")
    return int(m.group(1)) if m else default


def parse_protocol(marks: list[dict]) -> dict | None:
    """The protocol run recorded in marks.csv: blocks (t0 / t1 from block_start / block_end,
    condition = the mark's value), ratings (kind rating, value = JSON), seed and planned
    order from the protocol_start note. None when the session had no protocol."""
    start = next((m for m in marks if m["kind"] == "protocol_start"), None)
    blocks: dict[int, dict] = {}
    order: list[int] = []
    for m in marks:
        if m["kind"] == "block_start":
            i = _block_index_from_note(m["note"], len(order))
            label = re.sub(r"^\s*block=\d+\s*", "", m["note"] or "").strip()
            blocks[i] = {"index": i, "condition": (m["raw_value"] or "").strip() or "?", "label": label,
                         "t0": m["t"], "t1": None, "aborted": False}
            order.append(i)
        elif m["kind"] == "block_end":
            i = _block_index_from_note(m["note"], order[-1] if order else -1)
            b = blocks.get(i)
            if b is not None and b["t1"] is None:
                b["t1"] = m["t"]
                b["aborted"] = "aborted" in (m["note"] or "")
    ratings = []
    for m in marks:
        if m["kind"] != "rating":
            continue
        try:
            d = json.loads(m["raw_value"] or "")
        except (TypeError, ValueError):
            continue
        if not isinstance(d, dict):
            continue
        bi = _f(d.get("block"))
        items = {str(k): _f(v) for k, v in d.items() if k not in ("block", "condition")}
        ratings.append({"t": m["t"], "block": int(bi) if math.isfinite(bi) else None,
                        "condition": d.get("condition"), "items": items, "note": m["note"]})
    if start is None and not blocks:
        return None
    seed = planned = None
    if start is not None:
        ms = re.search(r"seed=(-?\d+)", start["note"])
        seed = int(ms.group(1)) if ms else None
        mo = re.search(r"order=(\S+)", start["note"])
        planned = mo.group(1).split(",") if mo else None
    ends = [m for m in marks if m["kind"] in ("protocol_end", "protocol_stop")
            and (start is None or m["t"] >= start["t"])]
    end = ends[-1] if ends else None
    return {"name": (start["raw_value"] or "").strip() if start else None, "seed": seed, "planned_order": planned,
            "t_start": start["t"] if start else None, "t_end": end["t"] if end else None,
            "stopped": (end["kind"] == "protocol_stop") if end else None,
            "blocks": [blocks[i] for i in sorted(blocks)], "ratings": ratings}


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


def welch_t(a, b):
    """Welch's t for mean(a) - mean(b) with the Welch-Satterthwaite df; None when either
    group has fewer than 2 values or both are constant. No p-value (no scipy): read |t|
    against the df, e.g. |t| > 4.3 at df 2, > 2.8 at df 4 for two-sided 5 %."""
    a = np.asarray([v for v in a if v is not None], float)
    b = np.asarray([v for v in b if v is not None], float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
    se2 = va / na + vb / nb
    if se2 < 1e-18:
        return None
    t = float(a.mean() - b.mean()) / math.sqrt(se2)
    num = se2 * se2
    den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    df = num / den if den > 0 else None
    return {"t": t, "df": df, "n_a": na, "n_b": nb}


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


def _mean_sd(vals) -> dict:
    v = np.asarray([x for x in vals if x is not None], float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"mean": None, "sd": None, "n": 0}
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) >= 2 else None, "n": int(len(v))}


# --- block-level analysis (protocol sessions) --------------------------------------------
def block_analysis(proto: dict, ep: dict, bd: dict, task_rows: list[dict], include_held: bool) -> dict:
    """Per-block / per-condition scores, manipulation check, primary gates and the EEG
    reference check. Blocks are the units; pause blocks and blocks < BLOCK_MIN_S are not
    scored; rest blocks are tabled but excluded from ordering and rho."""
    te, fe, falt = ep["t"], ep["flow"], ep["flow_alt"]
    e_ok = ep["ready"] & np.isfinite(fe) & ~ep["artifact"]
    tb, fb = bd["t"], bd["flow"]
    b_ok = (np.isfinite(fb) & ~bd["settling"] & ~bd.get("asleep", np.zeros(len(fb), bool))
            & ~bd.get("poor_contact", np.zeros(len(fb), bool))
            & (np.ones(len(fb), bool) if include_held else ~bd["held"]))
    notes: list[str] = []
    rows = []
    for b in proto["blocks"]:
        i, cond = b["index"], b["condition"]
        if b["t1"] is None:
            notes.append(f"block {i} ({cond}) has no end mark and was not scored.")
            continue
        t0, t1 = b["t0"], b["t1"]
        dur = t1 - t0
        row = {"index": i, "condition": cond, "label": b["label"], "t0": t0, "t1": t1, "duration_s": dur,
               "aborted": b["aborted"], "is_task": cond in TASK_CONDITIONS,
               "scored": cond != "pause" and dur >= BLOCK_MIN_S}
        m = e_ok & (te >= t0 + BLOCK_TRANSITION_S) & (te <= t1)
        row["eeg_mean"] = float(fe[m].mean()) if m.any() else None
        row["eeg_sd"] = float(fe[m].std()) if m.sum() >= 2 else None
        row["eeg_n"] = int(m.sum())
        row["eeg_artifacts"] = int((ep["artifact"] & (te >= t0 + BLOCK_TRANSITION_S) & (te <= t1)).sum())
        ma = m & np.isfinite(falt)
        row["eeg_alt_mean"] = float(falt[ma].mean()) if ma.any() else None
        mb = b_ok & (tb >= t0 + BLOCK_TRANSITION_S) & (tb <= t1)
        after = False
        if not mb.any():
            mb = b_ok & (tb > t1) & (tb <= t1 + BLOCK_BAND_AFTER_S)
            after = bool(mb.any())
        row["band_mean"] = float(fb[mb].mean()) if mb.any() else None
        row["band_sd"] = float(fb[mb].std()) if mb.sum() >= 2 else None
        row["band_n"] = int(mb.sum())
        row["band_after_end"] = after
        rating = next((r for r in proto["ratings"] if r["block"] == i), None)
        row["fss"] = row["demand"] = row["flow_overall"] = None
        row["fss_n_items"] = 0
        row["rated"] = rating is not None
        if rating is not None:
            core = [rating["items"][k] for k in FSS_ITEMS if k in rating["items"] and math.isfinite(rating["items"][k])]
            row["fss"] = float(np.mean(core)) if core else None
            row["fss_n_items"] = len(core)
            row["demand"] = _py(rating["items"].get("demand"))
            row["flow_overall"] = _py(rating["items"].get("flow_overall"))
        trials = [x for x in task_rows if x["block"] == i]
        rts = np.array([x["rt_ms"] for x in trials], float)
        lv = np.array([x["level"] for x in trials], float)
        row["n_trials"] = len(trials)
        row["accuracy"] = float(np.mean([x["correct"] for x in trials])) if trials else None
        row["rt_median_ms"] = float(np.median(rts[np.isfinite(rts)])) if np.isfinite(rts).any() else None
        row["level_mean"] = float(lv[np.isfinite(lv)].mean()) if np.isfinite(lv).any() else None
        rows.append(row)
    scored = [r for r in rows if r["scored"]]
    task = [r for r in scored if r["is_task"]]
    if any(r["band_after_end"] for r in scored):
        ids = ", ".join(str(r["index"]) for r in scored if r["band_after_end"])
        notes.append(f"block(s) {ids} had no usable band sample inside; samples up to "
                     f"{BLOCK_BAND_AFTER_S:.0f} s after the block end were used (the band's 1/min cadence).")
    unrated = [r["index"] for r in task if not r["rated"]]
    if unrated:
        notes.append("task block(s) without a self-report: " + ", ".join(map(str, unrated)) + ".")

    # per condition
    keys = ("eeg_mean", "eeg_alt_mean", "band_mean", "fss", "demand", "flow_overall", "accuracy", "rt_median_ms", "level_mean")
    conditions = []
    seen = []
    for r in scored:
        if r["condition"] not in seen:
            seen.append(r["condition"])
    for cond in seen:
        rs = [r for r in scored if r["condition"] == cond]
        c = {"condition": cond, "n_blocks": len(rs), "blocks": [r["index"] for r in rs]}
        for k in keys:
            c[k] = _mean_sd([r[k] for r in rs])
        conditions.append(c)
    by = {c["condition"]: c for c in conditions}

    def cmean(cond, key):
        c = by.get(cond)
        return None if c is None else c[key]["mean"]

    def ordering(key):
        have = [(c["condition"], c[key]["mean"]) for c in conditions
                if c["condition"] in TASK_CONDITIONS and c[key]["mean"] is not None]
        if len(have) < 2:
            return None
        return ">".join(c for c, _ in sorted(have, key=lambda x: -x[1]))

    def flow_highest(key):
        f, bo, ov = cmean("flow", key), cmean("boredom", key), cmean("overload", key)
        if f is None or bo is None or ov is None:
            return None
        return bool(f > bo and f > ov)

    orderings = {"eeg": ordering("eeg_mean"), "band": ordering("band_mean"), "fss": ordering("fss"),
                 "eeg_alt": ordering("eeg_alt_mean")}

    # manipulation check: did the induction work? (FSS is the criterion; demand / accuracy support)
    fss_f, fss_b, fss_o = cmean("flow", "fss"), cmean("boredom", "fss"), cmean("overload", "fss")
    vals = {cond: [r["fss"] for r in task if r["condition"] == cond and r["fss"] is not None]
            for cond in ("flow", "boredom", "overload")}
    manip = {"fss": {"flow": fss_f, "boredom": fss_b, "overload": fss_o,
                     "diff_flow_boredom": None if None in (fss_f, fss_b) else fss_f - fss_b,
                     "diff_flow_overload": None if None in (fss_f, fss_o) else fss_f - fss_o,
                     "t_flow_boredom": welch_t(vals["flow"], vals["boredom"]),
                     "t_flow_overload": welch_t(vals["flow"], vals["overload"]),
                     "n": {k: len(v) for k, v in vals.items()}},
             "demand": {"flow": cmean("flow", "demand"), "boredom": cmean("boredom", "demand"),
                        "overload": cmean("overload", "demand"), "ok": None},
             "accuracy": {"flow": cmean("flow", "accuracy"), "boredom": cmean("boredom", "accuracy"),
                          "overload": cmean("overload", "accuracy"), "ok": None},
             "ok": None, "reason": ""}
    d = manip["demand"]
    if None not in (d["flow"], d["boredom"], d["overload"]):
        d["ok"] = bool(d["boredom"] < d["flow"] < d["overload"])
    a = manip["accuracy"]
    if None not in (a["flow"], a["boredom"], a["overload"]):
        a["ok"] = bool(a["boredom"] > a["flow"] > a["overload"]
                       and ACCURACY_FLOW_RANGE[0] <= a["flow"] <= ACCURACY_FLOW_RANGE[1])
    if fss_f is None or fss_b is None or fss_o is None:
        missing = [c for c, v in (("flow", fss_f), ("boredom", fss_b), ("overload", fss_o)) if v is None]
        manip["reason"] = "no FSS self-report for " + ", ".join(missing) + " blocks"
    else:
        manip["ok"] = bool(fss_f > fss_b and fss_f > fss_o)
        manip["reason"] = (f"FSS flow {fss_f:.2f} vs boredom {fss_b:.2f} vs overload {fss_o:.2f}"
                           + ("" if manip["ok"] else " - flow is not rated highest"))

    # primary gates
    pairs = [(r["eeg_mean"], r["band_mean"]) for r in task if r["eeg_mean"] is not None and r["band_mean"] is not None]
    rho = spearman([p[0] for p in pairs], [p[1] for p in pairs]) if len(pairs) >= 3 else None
    n_blocks = len(pairs)
    eeg_hi, band_hi = flow_highest("eeg_mean"), flow_highest("band_mean")
    primary = {"ordering": orderings, "eeg_flow_highest": eeg_hi, "band_flow_highest": band_hi,
               "ordering_ok": None if (eeg_hi is None or band_hi is None) else bool(eeg_hi and band_hi),
               "block_rho": rho, "n_blocks": n_blocks, "n_task_blocks": len(task),
               "rho_ok": None if rho is None else bool(rho >= BLOCK_MIN_RHO and n_blocks >= BLOCK_MIN_N),
               "block_r": pearson([p[0] for p in pairs], [p[1] for p in pairs]) if len(pairs) >= 3 else None}

    # reference check: is the EEG index itself behaving as the induction says it should?
    ef = [(r["eeg_mean"], r["fss"]) for r in task if r["eeg_mean"] is not None and r["fss"] is not None]
    bf = [(r["band_mean"], r["fss"]) for r in task if r["band_mean"] is not None and r["fss"] is not None]
    af = [(r["eeg_alt_mean"], r["fss"]) for r in task if r["eeg_alt_mean"] is not None and r["fss"] is not None]
    rho_ef = spearman([p[0] for p in ef], [p[1] for p in ef]) if len(ef) >= 3 else None
    rho_bf = spearman([p[0] for p in bf], [p[1] for p in bf]) if len(bf) >= 3 else None
    rho_af = spearman([p[0] for p in af], [p[1] for p in af]) if len(af) >= 3 else None
    ref_ok = None if eeg_hi is None else bool(eeg_hi and (rho_ef is None or rho_ef > 0))
    reference = {"eeg_flow_highest": eeg_hi, "ordering": orderings["eeg"], "rho_eeg_fss": rho_ef, "n_eeg_fss": len(ef),
                 "rho_band_fss": rho_bf, "n_band_fss": len(bf),
                 "alt_flow_highest": flow_highest("eeg_alt_mean"), "alt_ordering": orderings["eeg_alt"],
                 "rho_alt_fss": rho_af, "ok": ref_ok}
    reference["text"] = ("n/a: not every condition has an EEG mean" if ref_ok is None else
                         ("the EEG index orders the conditions as the induction predicts"
                          + (f" and tracks the FSS across blocks (rho = {rho_ef:.2f})" if rho_ef is not None else "")
                          if ref_ok else
                          "the EEG index does NOT behave as the induction predicts, so the reference is not "
                          "trustworthy for this session"))

    # verdict (pre-registered)
    if len(task) < BLOCK_MIN_N:
        verdict, reason = "INCONCLUSIVE", f"only {len(task)} usable task blocks (need >= {BLOCK_MIN_N})"
    elif n_blocks == 0:
        verdict, reason = "INCONCLUSIVE", "no usable band sample in any task block"
    elif manip["ok"] is None:
        verdict, reason = "INCONCLUSIVE", "manipulation check not possible: " + manip["reason"]
    elif not manip["ok"]:
        verdict = "INCONCLUSIVE"
        reason = ("the induction did not work (" + manip["reason"]
                  + "), so this session cannot validate the band")
    elif n_blocks < BLOCK_MIN_N:
        verdict = "INCONCLUSIVE"
        reason = f"only {n_blocks} task blocks with both an EEG and a band mean (need >= {BLOCK_MIN_N})"
    elif primary["ordering_ok"] and primary["rho_ok"]:
        verdict = "PASS"
        reason = (f"EEG and band both highest in flow ({orderings['eeg']} / {orderings['band']}) and "
                  f"block rho = {rho:.2f} >= {BLOCK_MIN_RHO} over {n_blocks} blocks; manipulation check passed")
    else:
        verdict = "FAIL"
        parts = []
        if not primary["ordering_ok"]:
            if not eeg_hi:
                parts.append(f"EEG index not highest in flow ({orderings['eeg']})")
            if not band_hi:
                parts.append(f"band flow not highest in flow ({orderings['band']})")
        if not primary["rho_ok"]:
            parts.append("block rho undefined" if rho is None else f"block rho = {rho:.2f} < {BLOCK_MIN_RHO}")
        reason = "; ".join(parts) or "gates not met"
        if ref_ok is False:
            reason += " (note: the EEG reference failed its own check, so this may reflect the reference, not the band)"
    return {"name": proto["name"], "seed": proto["seed"], "planned_order": proto["planned_order"],
            "t_start": proto["t_start"], "t_end": proto["t_end"], "stopped": proto["stopped"],
            "n_blocks_total": len(proto["blocks"]), "n_scored": len(scored), "n_task_blocks": len(task),
            "n_ratings": len(proto["ratings"]), "n_trials": len(task_rows),
            "blocks": rows, "conditions": conditions, "manipulation": manip, "primary": primary,
            "reference": reference, "verdict": verdict, "verdict_reason": reason, "notes": notes,
            "gates": {"min_task_blocks": BLOCK_MIN_N, "min_rho": BLOCK_MIN_RHO, "transition_s": BLOCK_TRANSITION_S,
                      "band_after_s": BLOCK_BAND_AFTER_S, "min_block_s": BLOCK_MIN_S}}


# --- analysis -----------------------------------------------------------------------
def analyze(session_dir: Path | str, include_held: bool = False) -> dict:
    session_dir = Path(session_dir)
    ep = load_epochs(session_dir)
    bd = load_band(session_dir)
    marks = load_marks(session_dir)
    task_rows = load_task(session_dir)
    proto = parse_protocol(marks)
    meta = {}
    sj = session_dir / "session.json"
    if sj.exists():
        try:
            meta = json.loads(sj.read_text("utf-8"))
        except (OSError, ValueError):
            meta = {}

    notes: list[str] = []
    # reference series: baseline-ready, non-artifact epochs with a finite flow index
    e_ok = ep["ready"] & np.isfinite(ep["flow"]) & ~ep["artifact"]
    te = ep["t"][e_ok]
    fe = ep["flow"][e_ok]
    comp = {c: ep["components"][c][e_ok] for c in COMPONENTS}
    n_warm = int((~ep["ready"]).sum())
    n_art = int(ep["artifact"].sum())
    if n_warm:
        notes.append(f"{n_warm} EEG epochs before the baseline was ready were left out of the pairing.")
    if n_art:
        notes.append(f"{n_art} artifact epochs (no index value) were left out.")

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
    # exploratory alt index at the same windows
    alt_ok = e_ok & np.isfinite(ep["flow_alt"])
    alt_mean, _ = window_means(ep["t"][alt_ok], ep["flow_alt"][alt_ok], tb, W)
    r_alt = pearson(fb, alt_mean)

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
    protocol_kinds = ("felt_flow", "protocol_start", "protocol_end", "protocol_stop", "block_start", "block_end",
                      "rating", "trial")
    other_marks = [{"t": m["t"], "kind": m["kind"], "value": m["raw_value"], "note": m["note"]}
                   for m in marks if m["kind"] not in protocol_kinds]

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
                "epochs_artifact": n_art,
                "band_samples": int(bd["n_rows"]), "band_with_flow": int(has_flow.sum()),
                "band_held": int((has_flow & held).sum()), "band_skipped_held": n_skipped_held,
                "band_settling": int(bd["settling"].sum()),
                "band_median_interval_s": med_int, "marks": len(marks), "felt_marks": len(felt_rows),
                "task_trials": len(task_rows),
                "pct_time_eeg": pct_eeg, "pct_time_band": pct_band, "pct_time_both": pct_both}
    if not ep["n_rows"]:
        notes.append("epochs.csv is missing or empty: no EEG reference.")
    if not bd["n_rows"]:
        notes.append("band.csv is missing or empty: no band samples.")
    elif not has_flow.any():
        notes.append("band.csv has no sample with a flow value (all 'no reading').")

    # epoch-level verdict against the old pre-registered gates (primary without a protocol)
    gates = {"min_pairs": GATE_MIN_PAIRS, "min_r": GATE_MIN_R, "ci_lower_gt": 0.0}
    if n_pairs < GATE_MIN_PAIRS:
        e_verdict = "INCONCLUSIVE"
        e_reason = f"only {n_pairs} pairs (need >= {GATE_MIN_PAIRS})"
    elif r is None:
        e_verdict = "FAIL"
        e_reason = "Pearson r undefined (constant series)"
    elif r >= GATE_MIN_R and r_ci is not None and r_ci[0] > 0:
        e_verdict = "PASS"
        e_reason = f"r = {r:.2f} >= {GATE_MIN_R} and CI lower bound {r_ci[0]:.2f} > 0 with n = {n_pairs}"
    else:
        e_verdict = "FAIL"
        parts = []
        if r < GATE_MIN_R:
            parts.append(f"r = {r:.2f} < {GATE_MIN_R}")
        if r_ci is None:
            parts.append("no bootstrap CI")
        elif r_ci[0] <= 0:
            parts.append(f"CI lower bound {r_ci[0]:.2f} <= 0")
        e_reason = "; ".join(parts) or "gates not met"

    # block level (protocol sessions) decides the verdict
    protocol = block_analysis(proto, ep, bd, task_rows, include_held) if proto else None
    if protocol is not None:
        verdict, reason = protocol["verdict"], protocol["verdict_reason"]
        notes.extend(protocol["notes"])
        design = "block"
    else:
        verdict, reason = e_verdict, e_reason
        design = "epoch"
        notes.append("No protocol was run in this session: the verdict is the epoch-level gate.")

    result = {
        "session": session_dir.name, "dir": str(session_dir.resolve()),
        "report_version": VERSION, "generated_at": datetime.now().isoformat(timespec="seconds"),
        "include_held": include_held,
        "design": design,
        "pair_window_s": W, "pair_window_reason": window_reason,
        "n_pairs": n_pairs, "r": r, "r_ci": r_ci, "spearman": rho, "mae": mae, "bias": bias,
        "sd_band": sd_band, "sd_eeg": sd_eeg, "r_alt": r_alt,
        "best_lag_s": best_lag, "r_at_best_lag": r_best, "lags": [[l["lag_s"], l["r"]] for l in lags],
        "lag_table": lags,
        "components": components,
        "agreement": agreement,
        "felt": felt,
        "other_marks": other_marks,
        "coverage": coverage,
        "gates": gates, "epoch_verdict": e_verdict, "epoch_verdict_reason": e_reason,
        "protocol": protocol,
        "verdict": verdict, "verdict_reason": reason,
        "notes": notes,
        "meta": {k: meta.get(k) for k in ("board", "labels", "started", "fs", "band_source", "config")
                 if k in meta},
        "pairs": [[float(a), float(b), float(c)] for a, b, c in zip(pt, px, py)],
    }
    summary = {k: result[k] for k in ("n_pairs", "r", "r_ci", "spearman", "mae", "bias",
                                      "best_lag_s", "r_at_best_lag", "pair_window_s",
                                      "verdict", "verdict_reason", "design", "epoch_verdict")}
    if protocol is not None:
        summary.update({"protocol": protocol["name"], "block_rho": protocol["primary"]["block_rho"],
                        "n_blocks": protocol["primary"]["n_blocks"], "ordering": protocol["primary"]["ordering"],
                        "manipulation_ok": protocol["manipulation"]["ok"],
                        "reference_ok": protocol["reference"]["ok"]})
    else:
        summary.update({"protocol": None, "block_rho": None, "n_blocks": 0,
                        "ordering": {"eeg": None, "band": None, "fss": None}, "manipulation_ok": None,
                        "reference_ok": None})
    result["summary"] = summary
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


def _pm(ms: dict, nd=2):
    """mean +- sd from a _mean_sd dict."""
    if not ms or ms.get("mean") is None:
        return "n/a"
    s = f"{ms['mean']:.{nd}f}"
    if ms.get("sd") is not None:
        s += f" ± {ms['sd']:.{nd}f}"
    return s


def _pct(v):
    return "n/a" if v is None else f"{100 * v:.0f} %"


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


def _yn(v):
    return "n/a" if v is None else ("yes" if v else "no")


def _t(tt):
    if tt is None:
        return "n/a"
    return f"t = {tt['t']:.2f} (Welch df {_n(tt['df'], 1)}, n = {tt['n_a']} vs {tt['n_b']})"


def _render_protocol(P: dict) -> list[str]:
    L = []
    L.append("## Protocol")
    L.append("")
    L.append(f"`{P['name']}` (seed {P['seed']}), {_ts(P['t_start'])} to {_ts(P['t_end'])}"
             + (", stopped early" if P["stopped"] else "")
             + f". {P['n_blocks_total']} blocks ({P['n_scored']} scored, {P['n_task_blocks']} task blocks), "
             f"{P['n_ratings']} self-reports, {P['n_trials']} task trials.")
    if P["planned_order"]:
        L.append("")
        L.append("Planned order: " + ", ".join(P["planned_order"]) + ".")
    L.append("")
    L.append("## Per-block table")
    L.append("")
    L.append(f"EEG = mean flow index over non-artifact epochs from {P['gates']['transition_s']:.0f} s after the block "
             "start to its end; band = mean flow over usable band samples (not held / settling) in the same window "
             f"(samples up to {P['gates']['band_after_s']:.0f} s after the end when none fell inside, marked *); "
             "FSS = mean of the 10 Flow Short Scale items (1-7); demand 1 = too low, 2 = just right, 3 = too high; "
             "accuracy / level / median RT from the arithmetic task. Pause blocks and blocks shorter than "
             f"{P['gates']['min_block_s']:.0f} s are not scored.")
    L.append("")
    L.append("| block | condition | start | dur | EEG | band | FSS | demand | accuracy | level | RT ms | n EEG / band |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in P["blocks"]:
        name = r["condition"] + ("" if r["scored"] else " (not scored)") + (" aborted" if r["aborted"] else "")
        band = _n(r["band_mean"], 2) + ("*" if r["band_after_end"] else "")
        L.append(f"| {r['index']} | {name} | {_ts(r['t0'])} | {_hms(r['duration_s'])} | {_n(r['eeg_mean'], 2)} | {band} | "
                 f"{_n(r['fss'], 2)} | {_n(r['demand'], 0)} | {_pct(r['accuracy'])} | {_n(r['level_mean'], 1)} | "
                 f"{_n(r['rt_median_ms'], 0)} | {r['eeg_n']} / {r['band_n']} |")
    L.append("")
    L.append("## Per-condition table")
    L.append("")
    L.append("Mean ± sd across blocks (n = blocks). Rest is shown for reference; ordering and rho use task blocks only.")
    L.append("")
    L.append("| condition | n | EEG | band | FSS | demand | accuracy | level | RT ms | EEG alt |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in P["conditions"]:
        acc = c["accuracy"]
        acc_s = "n/a" if acc["mean"] is None else (f"{100 * acc['mean']:.0f} %"
                                                   + (f" ± {100 * acc['sd']:.0f}" if acc["sd"] is not None else ""))
        L.append(f"| {c['condition']} | {c['n_blocks']} | {_pm(c['eeg_mean'])} | {_pm(c['band_mean'])} | {_pm(c['fss'])} | "
                 f"{_pm(c['demand'], 1)} | {acc_s} | {_pm(c['level_mean'], 1)} | {_pm(c['rt_median_ms'], 0)} | "
                 f"{_pm(c['eeg_alt_mean'])} |")
    M = P["manipulation"]
    L.append("")
    L.append("## Manipulation check")
    L.append("")
    L.append(f"Did the induction work? **{'passed' if M['ok'] else ('FAILED' if M['ok'] is False else 'not possible')}** "
             f"- {M['reason']}. Criterion: FSS higher in flow than in boredom AND than in overload "
             "(the demand item and the task accuracy are supporting checks and do not gate).")
    L.append("")
    L.append("| check | flow | boredom | overload | flow - boredom | flow - overload | ok |")
    L.append("|---|---|---|---|---|---|---|")
    F = M["fss"]
    L.append(f"| FSS (1-7) | {_n(F['flow'], 2)} | {_n(F['boredom'], 2)} | {_n(F['overload'], 2)} | "
             f"{_n(F['diff_flow_boredom'], 2)} | {_n(F['diff_flow_overload'], 2)} | {_yn(M['ok'])} |")
    D = M["demand"]
    L.append(f"| demand (expect 2 / 1 / 3) | {_n(D['flow'], 1)} | {_n(D['boredom'], 1)} | {_n(D['overload'], 1)} | | | {_yn(D['ok'])} |")
    A = M["accuracy"]
    L.append(f"| task accuracy (expect ~70-85 % / ~100 % / low) | {_pct(A['flow'])} | {_pct(A['boredom'])} | "
             f"{_pct(A['overload'])} | | | {_yn(A['ok'])} |")
    L.append("")
    L.append(f"FSS flow vs boredom: {_t(F['t_flow_boredom'])}; flow vs overload: {_t(F['t_flow_overload'])} "
             "(Welch t over blocks, numpy; no p-value without scipy - with 2 blocks per condition treat it as descriptive).")
    R = P["primary"]
    L.append("")
    L.append("## Primary result (block level)")
    L.append("")
    L.append("| gate | value | ok |")
    L.append("|---|---|---|")
    L.append(f"| (a) EEG index highest in flow (vs boredom and overload) | {R['ordering']['eeg'] or 'n/a'} | {_yn(R['eeg_flow_highest'])} |")
    L.append(f"| (a) band flow highest in flow (vs boredom and overload) | {R['ordering']['band'] or 'n/a'} | {_yn(R['band_flow_highest'])} |")
    L.append(f"| (b) block-level Spearman rho EEG vs band (>= {P['gates']['min_rho']}, n >= {P['gates']['min_task_blocks']}) | "
             f"{_n(R['block_rho'], 2)} over {R['n_blocks']} blocks (Pearson {_n(R['block_r'], 2)}) | {_yn(R['rho_ok'])} |")
    L.append(f"| FSS ordering (self-report) | {R['ordering']['fss'] or 'n/a'} | |")
    L.append("")
    L.append("(c) The epoch-level correlation further down is a secondary, descriptive result: epochs are "
             "autocorrelated, so its CI is not inferential. Blocks are the independent units.")
    Q = P["reference"]
    L.append("")
    L.append("## Reference check")
    L.append("")
    L.append("The EEG index is a proxy, so it is checked against the induction too: " + Q["text"] + ".")
    L.append("")
    L.append("| check | value |")
    L.append("|---|---|")
    L.append(f"| EEG index ordering (expect flow highest) | {Q['ordering'] or 'n/a'} ({_yn(Q['eeg_flow_highest'])}) |")
    L.append(f"| Spearman rho EEG block mean vs FSS | {_n(Q['rho_eeg_fss'], 2)} (n = {Q['n_eeg_fss']}) |")
    L.append(f"| Spearman rho band block mean vs FSS | {_n(Q['rho_band_fss'], 2)} (n = {Q['n_band_fss']}) |")
    L.append(f"| exploratory alt index (theta + moderate alpha) ordering | {Q['alt_ordering'] or 'n/a'} "
             f"({_yn(Q['alt_flow_highest'])}), rho vs FSS {_n(Q['rho_alt_fss'], 2)} |")
    L.append(f"| reference trustworthy for this session | {_yn(Q['ok'])} |")
    return L


def render_markdown(res: dict) -> str:
    cov = res["coverage"]
    agr = res["agreement"]
    felt = res["felt"]
    P = res.get("protocol")
    L = []
    L.append(f"# Flow validation report: {res['session']}")
    L.append("")
    L.append(f"Generated {res['generated_at']} by report.py {res['report_version']}. "
             f"Session {_ts(cov['t_start'])} to {_ts(cov['t_end'])} ({_hms(cov['duration_s'])}).")
    L.append("")
    L.append(f"**Verdict: {res['verdict']}** - {res['verdict_reason']}. "
             + ("Block-level design (protocol session); details below, full verdict section at the end."
                if P else "No protocol was run in this session: epoch-level design (the only one available)."))
    L.append("")
    if ((res.get("meta") or {}).get("board") or {}).get("kind") == "sim":
        L.append("**Simulated session.** Synthetic board and band (`--sim`), ratings may be auto-generated: "
                 "this demonstrates that the pipeline, protocol and gates work end to end. It says nothing "
                 "about a real BioProxal band.")
        L.append("")
    if P:
        L.extend(_render_protocol(P))
        L.append("")
    L.append("## Headline (epoch-level pairs" + (", secondary)" if P else ")"))
    L.append("")
    if P:
        L.append(f"Epoch-level gate for reference: **{res['epoch_verdict']}** ({res['epoch_verdict_reason']}).")
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
    L.append(f"| exploratory alt index r | {_n(res.get('r_alt'))} |")
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
    L.append(f"| EEG epochs (usable / warm-up / artifact) | {cov['epochs']} ({cov['epochs_ready']} / "
             f"{cov['epochs_warmup']} / {cov['epochs_artifact']}) |")
    L.append(f"| band samples (with flow / held / settling) | {cov['band_samples']} ({cov['band_with_flow']} / "
             f"{cov['band_held']} / {cov['band_settling']}) |")
    L.append(f"| median band interval | {_n(cov['band_median_interval_s'], 1)} s |")
    L.append(f"| time with EEG / band / both | {_n(cov['pct_time_eeg'], 0)} % / {_n(cov['pct_time_band'], 0)} % / "
             f"{_n(cov['pct_time_both'], 0)} % |")
    L.append(f"| marks (felt-flow) | {cov['marks']} ({cov['felt_marks']}) |")
    L.append(f"| task trials | {cov['task_trials']} |")
    if res["notes"]:
        L.append("")
        for n in res["notes"]:
            L.append(f"- {n}")
    L.append("")
    L.append(f"## Verdict: **{res['verdict']}**")
    L.append("")
    L.append(f"{res['verdict_reason']}.")
    L.append("")
    if P:
        g = P["gates"]
        L.append("Pre-registered gates (fixed before looking at the data), block level: the manipulation check "
                 "(FSS higher in flow than in boredom and overload) must pass, else the induction did not work and the "
                 "session is **INCONCLUSIVE** (it cannot validate the band either way). Then **PASS** if (a) the EEG "
                 "index and the band flow are both highest in the flow blocks (vs boredom and vs overload) and (b) the "
                 f"block-level Spearman rho between EEG and band block means is >= {g['min_rho']} over >= "
                 f"{g['min_task_blocks']} task blocks; **FAIL** if (a) or (b) fails; **INCONCLUSIVE** with fewer than "
                 f"{g['min_task_blocks']} usable task blocks or no band data. A FAIL is reported as a FAIL.")
    else:
        L.append("Pre-registered gates (fixed before looking at the data): **PASS** if Pearson r >= "
                 f"{res['gates']['min_r']} and the bootstrap 95 % CI lower bound > 0 with n >= "
                 f"{res['gates']['min_pairs']} pairs; **INCONCLUSIVE** if n < {res['gates']['min_pairs']}; "
                 "otherwise **FAIL**. A FAIL is reported as a FAIL. No protocol was run in this session, so only "
                 "this epoch-level design is available; it is descriptive (epochs are autocorrelated).")
    L.append("")
    L.append("## How to read this")
    L.append("")
    L.append("The EEG flow index is a literature-based proxy, not ground truth. Its primary form is frontal "
             "theta alone (4-7 Hz at F3/F4/F7/F8/Fz, Fp1/Fp2 as fallback), the one marker that replicates "
             "across flow studies: higher in demand-matched flow than in boredom (Katahira et al. 2018; Ewing "
             "et al. 2016). The exploratory alt index adds a penalty for central-parietal alpha away from "
             "baseline (flow between boredom and overload, Katahira et al. 2018); beta and the Pope et al. 1995 "
             "engagement index beta/(alpha+theta) are exported as covariates (they track load, not flow). Each "
             "component is z-scored against the wearer's own eyes-open rest block, so 0.5 = at rest. "
             "A block-level PASS means the band's score ordered the induced conditions like the EEG proxy and the "
             "self-report did, with the blocks as independent units; it does not by itself prove either signal "
             "measures flow. A FAIL can mean the band is wrong, the proxy is wrong (see the reference check), "
             "the electrodes were poor, or the induction was weak (see the manipulation check). The epoch-level "
             "r, lag sweep and felt-flow marks are secondary: pairs are formed at the band's cadence (EEG index "
             "averaged over the pairing window before each band sample), band samples flagged held are skipped "
             "unless --include-held, and epochs before the baseline was ready or with artifacts are not used.")
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
            if res.get("protocol"):
                pr = res["protocol"]["primary"]
                print(f"{d.name}: {res['verdict']} (blocks = {pr['n_blocks']}, rho = {_n(pr['block_rho'])}, "
                      f"epoch n = {res['n_pairs']}, r = {_n(res['r'])})")
            else:
                print(f"{d.name}: {res['verdict']} (n = {res['n_pairs']}, r = {_n(res['r'])}, "
                      f"window {res['pair_window_s']:.0f} s)")
        else:
            print(f"[report] {d.name}: pairing window {res['pair_window_s']:.0f} s ({res['pair_window_reason']})")
            print(res["markdown"])
            print(f"[report] wrote {d / 'report.md'} and report.json")
    return rc


if __name__ == "__main__":
    sys.exit(main())
