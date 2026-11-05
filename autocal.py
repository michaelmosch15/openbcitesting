"""Auto-calibration of electrode placement from the signals themselves.

The headset as built has all eight combs across the forehead; which channel sits where is
unknown until someone taps each comb. This module recovers the assignment from a short
guided run (rest, blinks, left/right eye movements, a jaw clench) matched against a template
of how frontal 10-10 sites behave, built from public PhysioNet EEG by
tools/build_autocal_template.py (autocal_template.json):

  blink   blink amplitude is largest next to the eyes (Fp row) and falls with distance -
          separates the lower row from the upper row
  heog    a horizontal eye movement deflects left-hemisphere sites one way and right-
          hemisphere sites the other, more so toward the temples - separates left / right and
          lateral / medial (the overall sign is arbitrary and is solved as a free variable)
  corr    neighbouring sites correlate more than distant ones (1-30 Hz, ear-referenced) -
          a pairwise term that ties the whole arrangement together
  emg     a jaw clench adds 20-45 Hz muscle power that is strongest at the temples (live
          only; the template has no clench, so this is a lateral-vs-medial prior from geometry)

Solver: channels are assigned injectively to the frontal site set (17 candidates by default,
or the slots the operator chose) minimising unary mismatches + pairwise correlation
mismatches, by a Hungarian start on the unary costs followed by 2-opt swaps / moves with
random restarts. Confidence per channel = agreement among the near-optimal solutions.
Nothing here touches the index, thresholds or the protocol: it only proposes labels, and the
operator applies them (`set_labels`), which is recorded as a mark.
"""
from __future__ import annotations

import json
import math
import threading
import time
from itertools import permutations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMPLATE_PATH = HERE / "autocal_template.json"

FRONTAL_SITES = ("Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
                 "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8")
# 10-10 positions on the unit disc, x right, y forward (nasion at y = 1)
SITE_XY = {
    "Fp1": (-0.31, 0.95), "Fpz": (0.0, 1.0), "Fp2": (0.31, 0.95),
    "AF7": (-0.59, 0.81), "AF3": (-0.33, 0.81), "AFz": (0.0, 0.81), "AF4": (0.33, 0.81), "AF8": (0.59, 0.81),
    "F7": (-0.81, 0.59), "F5": (-0.68, 0.58), "F3": (-0.50, 0.55), "F1": (-0.26, 0.56), "Fz": (0.0, 0.55),
    "F2": (0.26, 0.56), "F4": (0.50, 0.55), "F6": (0.68, 0.58), "F8": (0.81, 0.59),
}
PHASES = ("rest", "blink", "look", "clench")
DEFAULT_WEIGHTS = {"blink": 1.0, "heog": 1.0, "corr": 1.0, "emg": 0.5}


# --- signal helpers (numpy only) ---------------------------------------------------------
def bandpass(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    """Zero-phase FFT band-pass of [n, ch] with mirror padding (no scipy)."""
    x = np.asarray(x, np.float64)
    n = len(x)
    if n < 8:
        return x.copy()
    pad = min(n, int(fs * 2))
    xp = np.concatenate([x[pad:0:-1], x, x[-2:-pad - 2:-1]], axis=0) if pad > 1 else x
    X = np.fft.rfft(xp, axis=0)
    f = np.fft.rfftfreq(len(xp), 1.0 / fs)
    mask = (f >= lo) & (f <= hi)
    X[~mask] = 0
    y = np.fft.irfft(X, n=len(xp), axis=0)
    return y[pad - 1: pad - 1 + n] if pad > 1 else y


def detect_events(sig: np.ndarray, fs: float, min_gap_s: float = 0.5, k_mad: float = 4.0,
                  positive_only: bool = True) -> list[int]:
    """Sample indices of peaks above median + k * MAD (|x| when not positive_only), at
    least min_gap_s apart (the larger peak wins)."""
    s = np.asarray(sig, np.float64)
    v = s if positive_only else np.abs(s)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826 + 1e-9
    thr = med + k_mad * mad
    above = v > thr
    if not above.any():
        return []
    gap = max(1, int(min_gap_s * fs))
    idx = np.flatnonzero(above)
    peaks = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i - prev > 1:
            seg = np.arange(start, prev + 1)
            peaks.append(int(seg[np.argmax(v[seg])]))
            start = i
        prev = i
    seg = np.arange(start, prev + 1)
    peaks.append(int(seg[np.argmax(v[seg])]))
    out = []
    for p in peaks:
        if out and p - out[-1] < gap:
            if v[p] > v[out[-1]]:
                out[-1] = p
        else:
            out.append(p)
    return out


# --- template ---------------------------------------------------------------------------
def geometry_template(sites=FRONTAL_SITES) -> dict:
    """A template from geometry alone (used when autocal_template.json is missing):
    correlation falls with distance, blink falls with distance from the eyes, heog grows with
    |x| and carries the sign of x."""
    xy = np.array([SITE_XY[s] for s in sites], float)
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    corr = np.exp(-d / 0.45)
    eyes = np.array([[-0.25, 1.05], [0.25, 1.05]])
    de = np.min(np.sqrt(((xy[:, None, :] - eyes[None, :, :]) ** 2).sum(-1)), axis=1)
    blink = np.exp(-(de - de.min()) / 0.35)
    heog = -xy[:, 0] / 0.81 * (0.4 + 0.6 * (1 - xy[:, 1]) / 0.45)   # positive left, bigger toward the temples
    return {"sites": list(sites), "corr": corr.tolist(), "blink": blink.tolist(), "heog": (heog / np.abs(heog).max()).tolist(),
            "geometry": {s: list(SITE_XY[s]) for s in sites}, "source": "geometry prior (no template file)"}


def load_template(path: Path = TEMPLATE_PATH) -> dict:
    if path.exists():
        t = json.loads(path.read_text("utf-8"))
        if t.get("blink") is None or t.get("heog") is None:
            g = geometry_template(tuple(t["sites"]))
            t["blink"] = t.get("blink") or g["blink"]
            t["heog"] = t.get("heog") or g["heog"]
        return t
    return geometry_template()


# --- live features ------------------------------------------------------------------------
class Capture:
    """Raw EEG chunks tagged by calibration phase, filled from the board thread."""

    def __init__(self, fs: float, n_ch: int, max_s: float = 180.0):
        self.fs, self.n_ch = float(fs), int(n_ch)
        self.max_n = int(max_s * fs)
        self.phase: str | None = None
        self.chunks: dict[str, list] = {p: [] for p in PHASES}
        self.counts: dict[str, int] = {p: 0 for p in PHASES}
        self.started = time.time()
        self.lock = threading.Lock()

    def set_phase(self, phase: str | None) -> None:
        if phase is not None and phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r}")
        with self.lock:
            self.phase = phase

    def push(self, eeg: np.ndarray) -> None:
        with self.lock:
            p = self.phase
            if p is None or self.counts[p] >= self.max_n:
                return
            room = self.max_n - self.counts[p]
            chunk = np.asarray(eeg, np.float32)[:room].copy()
            self.chunks[p].append(chunk)
            self.counts[p] += len(chunk)

    def data(self, phase: str) -> np.ndarray | None:
        with self.lock:
            ch = self.chunks.get(phase) or []
            return np.concatenate(ch, axis=0) if ch else None

    def status(self) -> dict:
        with self.lock:
            return {"phase": self.phase, "seconds": {p: round(self.counts[p] / self.fs, 1) for p in PHASES},
                    "since_s": round(time.time() - self.started, 1)}


def features(cap: Capture, min_rest_s: float = 10.0) -> dict:
    """Per-channel blink / heog / emg vectors and the rest correlation matrix, or raise."""
    fs, n = cap.fs, cap.n_ch
    rest = cap.data("rest")
    if rest is None or len(rest) < min_rest_s * fs:
        raise ValueError(f"need at least {min_rest_s:.0f} s of rest; have {0 if rest is None else len(rest) / fs:.1f} s")
    rest = rest - rest.mean(axis=0)
    eeg = bandpass(rest, fs, 1.0, 30.0)
    corr = np.corrcoef(eeg.T)
    corr = np.nan_to_num(corr, nan=0.0)
    out = {"corr": corr, "n_ch": n, "fs": fs, "notes": []}
    # blink: peaks of the channel-mean slow signal; amplitude per channel at those peaks
    blink = cap.data("blink")
    if blink is not None and len(blink) >= 3 * fs:
        slow = bandpass(blink - blink.mean(axis=0), fs, 0.5, 8.0)
        drive = slow.mean(axis=1)
        pk = detect_events(drive, fs, 0.5, 3.0, positive_only=True)
        if len(pk) < 2:                                 # blinks may be negative-going on this reference
            pk = detect_events(-drive, fs, 0.5, 3.0, positive_only=True)
        if len(pk) >= 2:
            amp = np.stack([slow[t, :] for t in pk])      # [n_blink, n_ch]
            sgn = np.sign(np.median(amp.mean(axis=1))) or 1.0
            vec = np.median(amp * sgn, axis=0)
            out["blink"] = vec / (np.abs(vec).max() + 1e-9)
            out["n_blink"] = len(pk)
        else:
            out["notes"].append("no blinks detected")
    else:
        out["notes"].append("no blink phase")
    # heog: first principal component of the slow signal during the look phase
    look = cap.data("look")
    if look is not None and len(look) >= 3 * fs:
        slow = bandpass(look - look.mean(axis=0), fs, 0.3, 8.0)
        u, s, vt = np.linalg.svd(slow, full_matrices=False)
        load = vt[0]
        out["heog"] = load / (np.abs(load).max() + 1e-9)
        out["heog_var"] = float(s[0] ** 2 / (s ** 2).sum())
    else:
        out["notes"].append("no look phase")
    # emg: clench 20-45 Hz power relative to rest
    clench = cap.data("clench")
    if clench is not None and len(clench) >= 2 * fs:
        hi_c = bandpass(clench - clench.mean(axis=0), fs, 20.0, min(45.0, fs / 2 - 1)).std(axis=0)
        hi_r = bandpass(rest, fs, 20.0, min(45.0, fs / 2 - 1)).std(axis=0) + 1e-9
        ratio = np.log(hi_c / hi_r + 1e-9)
        ratio = ratio - ratio.min()
        out["emg"] = ratio / (ratio.max() + 1e-9)
    else:
        out["notes"].append("no clench phase")
    return out


# --- solver ------------------------------------------------------------------------------
def _hungarian(cost: np.ndarray) -> list[int]:
    """Minimum-cost assignment of rows to distinct columns (rows <= cols), brute force for
    small sizes, else a simple greedy + 2-opt (sizes here are 8 x 17)."""
    n, m = cost.shape
    if n <= 6 and m <= 8:
        best, best_c = None, math.inf
        for perm in permutations(range(m), n):
            c = sum(cost[i, perm[i]] for i in range(n))
            if c < best_c:
                best, best_c = list(perm), c
        return best
    used = set()
    assign = [-1] * n
    for i in np.argsort(cost.min(axis=1)):
        order = np.argsort(cost[i])
        for j in order:
            if j not in used:
                assign[i] = int(j); used.add(int(j)); break
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(m):
                if j in used:
                    k = assign.index(j)
                    if k != i and cost[i, j] + cost[k, assign[i]] < cost[i, assign[i]] + cost[k, j] - 1e-12:
                        assign[i], assign[k] = j, assign[i]; improved = True
                elif cost[i, j] < cost[i, assign[i]] - 1e-12:
                    used.discard(assign[i]); assign[i] = j; used.add(j); improved = True
    return assign


def _unary(feat: dict, tpl: dict, sites: list[str], sign: float, w: dict) -> np.ndarray:
    n = feat["n_ch"]
    S = [tpl["sites"].index(s) for s in sites]
    cost = np.zeros((n, len(sites)))
    if "blink" in feat:
        tb = np.array(tpl["blink"], float)[S]
        tb = tb / (np.abs(tb).max() + 1e-9)
        cost += w["blink"] * (feat["blink"][:, None] - tb[None, :]) ** 2
    if "heog" in feat:
        th = np.array(tpl["heog"], float)[S]
        th = th / (np.abs(th).max() + 1e-9)
        cost += w["heog"] * (sign * feat["heog"][:, None] - th[None, :]) ** 2
    if "emg" in feat:
        xy = np.array([SITE_XY[s] for s in sites], float)
        lat = np.abs(xy[:, 0]) / 0.81
        cost += w["emg"] * (feat["emg"][:, None] - lat[None, :]) ** 2
    return cost


def _pair_cost(assign: list[int], corr_live: np.ndarray, corr_tpl: np.ndarray) -> float:
    n = len(assign)
    c = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            c += (corr_live[i, j] - corr_tpl[assign[i], assign[j]]) ** 2
    return c


def _total(assign, unary, corr_live, corr_tpl, w) -> float:
    return float(sum(unary[i, assign[i]] for i in range(len(assign))) + w["corr"] * _pair_cost(assign, corr_live, corr_tpl))


def _local_search(assign, unary, corr_live, corr_tpl, w, m: int) -> tuple[list[int], float]:
    n = len(assign)
    cur = list(assign)
    best = _total(cur, unary, corr_live, corr_tpl, w)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                cand = list(cur); cand[i], cand[j] = cand[j], cand[i]
                c = _total(cand, unary, corr_live, corr_tpl, w)
                if c < best - 1e-12:
                    cur, best, improved = cand, c, True
            used = set(cur)
            for s in range(m):
                if s in used:
                    continue
                cand = list(cur); cand[i] = s
                c = _total(cand, unary, corr_live, corr_tpl, w)
                if c < best - 1e-12:
                    cur, best, improved = cand, c, True
                    used = set(cur)
    return cur, best


def solve(feat: dict, tpl: dict | None = None, slots: list[str] | None = None, weights: dict | None = None,
          restarts: int = 24, seed: int = 0) -> dict:
    """Assign channels to sites. Returns labels, per-channel confidence, cost, sign, runner-ups."""
    tpl = tpl or load_template()
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    sites = [s for s in (slots or tpl["sites"]) if s in tpl["sites"]]
    n = feat["n_ch"]
    if len(sites) < n:
        raise ValueError(f"{len(sites)} candidate sites for {n} channels")
    S = [tpl["sites"].index(s) for s in sites]
    corr_tpl = np.array(tpl["corr"], float)[np.ix_(S, S)]
    corr_live = feat["corr"]
    rng = np.random.default_rng(seed)
    solutions = []
    for sign in ((1.0, -1.0) if "heog" in feat else (1.0,)):
        unary = _unary(feat, tpl, sites, sign, w)
        starts = [_hungarian(unary)]
        for _ in range(restarts):
            starts.append(list(rng.choice(len(sites), size=n, replace=False)))
        for st in starts:
            a, c = _local_search(st, unary, corr_live, corr_tpl, w, len(sites))
            solutions.append((c, sign, tuple(a)))
    solutions.sort(key=lambda x: x[0])
    best_c, best_sign, best_a = solutions[0]
    distinct = []
    for c, sg, a in solutions:
        if all(a != d[2] for d in distinct):
            distinct.append((c, sg, a))
    # confidence: agreement among the near-optimal distinct solutions (within 10 % of the best
    # cost, at least the top three) - a channel that every good solution places the same way is
    # a sure thing, one that flips between neighbours is not
    near = [d for d in distinct if d[0] <= best_c * 1.10 + 0.01]
    if len(near) < 3:
        near = distinct[:3]
    conf = []
    for i in range(n):
        agree = sum(1 for d in near if d[2][i] == best_a[i])
        conf.append(round(agree / len(near), 2))
    runner = [{"labels": [sites[k] for k in a], "cost": round(c, 4)} for c, sg, a in distinct[1:4]]
    margin = (distinct[1][0] - best_c) / (abs(best_c) + 1e-9) if len(distinct) > 1 else 1.0
    return {"labels": [sites[k] for k in best_a], "confidence": conf, "cost": round(best_c, 4),
            "margin": round(float(margin), 3), "heog_sign": best_sign, "runner_ups": runner,
            "n_candidates": len(sites), "template": tpl.get("source", "?"),
            "features": {k: (np.round(v, 3).tolist() if isinstance(v, np.ndarray) else v)
                         for k, v in feat.items() if k in ("blink", "heog", "emg", "n_blink", "heog_var", "notes")}}


# --- synthetic data (tests + --sim) ----------------------------------------------------
def synthesize(assignment: list[str], tpl: dict, fs: float = 250.0, seed: int = 1, noise: float = 0.3) -> dict:
    """Phase -> [n, ch] signals for channels sitting at `assignment`, following the template:
    correlated rest EEG, blinks with the template's amplitude ratios, saccades with the
    template's signed ratios, a clench with lateral EMG."""
    rng = np.random.default_rng(seed)
    S = [tpl["sites"].index(s) for s in assignment]
    corr = np.array(tpl["corr"], float)[np.ix_(S, S)]
    corr = corr + np.eye(len(S)) * 1e-3
    L = np.linalg.cholesky(corr)

    def eeg(n):
        white = rng.normal(0, 1, (n, len(S)))
        x = white @ L.T
        return bandpass(x, fs, 1.0, 30.0) * 10.0 + rng.normal(0, noise, (n, len(S)))
    out = {}
    n_rest = int(30 * fs)
    out["rest"] = eeg(n_rest)
    n_b = int(15 * fs)
    xb = eeg(n_b)
    blink = np.array(tpl["blink"], float)[S]
    tt = np.arange(int(0.3 * fs)) / fs
    shape = np.sin(np.pi * tt / 0.3) ** 2
    for k in range(8):
        t0 = int((1.0 + k * 1.6) * fs)
        xb[t0:t0 + len(shape), :] += 120.0 * shape[:, None] * blink[None, :]
    out["blink"] = xb
    n_l = int(15 * fs)
    xl = eeg(n_l)
    heog = np.array(tpl["heog"], float)[S]
    for k in range(10):
        t0 = int((0.8 + k * 1.4) * fs)
        step = np.ones(int(0.6 * fs)) * (1 if k % 2 == 0 else -1)
        xl[t0:t0 + len(step), :] += 60.0 * step[:, None] * heog[None, :]
    out["look"] = xl
    n_c = int(5 * fs)
    xc = eeg(n_c)
    lat = np.abs(np.array([SITE_XY[s][0] for s in assignment])) / 0.81
    xc += bandpass(rng.normal(0, 1, (n_c, len(S))), fs, 20, 45) * (5.0 + 40.0 * lat)[None, :]
    out["clench"] = xc
    return out


def capture_from_synthetic(assignment: list[str], tpl: dict | None = None, fs: float = 250.0, seed: int = 1, noise: float = 0.3) -> Capture:
    tpl = tpl or load_template()
    cap = Capture(fs, len(assignment))
    for phase, x in synthesize(assignment, tpl, fs, seed, noise).items():
        cap.set_phase(phase)
        cap.push(x)
    cap.set_phase(None)
    return cap
