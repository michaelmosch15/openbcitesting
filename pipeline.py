"""The analysis pipeline: EEG samples -> display filter -> 2-s epochs -> PSD / band powers /
quality -> EEG flow index; band samples -> pairs at the band's cadence -> validation stats.
Everything the browser sees is assembled here (docs/API.md); the server only transports it.
"""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from cyton import FULL_SCALE_UV
from dsp import BAND_ORDER, FilterChain, band_powers, psd, quality, relative_powers
from flow_index import FlowIndex

LABELS_8 = ["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"]
LABELS_16 = LABELS_8 + ["F7", "F8", "F3", "F4", "T7", "T8", "P3", "P4"]
CHANNEL_COLORS = ["#818181", "#7C4B8D", "#36579E", "#317159", "#DDB20D", "#FD5E34", "#E0382D", "#A25231"]
# 10-20 positions on the unit circle (x right, y up), for the head plot
POSITIONS = {
    "Fp1": (-0.31, 0.95), "Fp2": (0.31, 0.95), "F7": (-0.81, 0.59), "F8": (0.81, 0.59),
    "F3": (-0.50, 0.55), "F4": (0.50, 0.55), "Fz": (0.0, 0.55), "C3": (-0.55, 0.0), "C4": (0.55, 0.0),
    "Cz": (0.0, 0.0), "T7": (-1.0, 0.0), "T8": (1.0, 0.0), "P3": (-0.50, -0.55), "P4": (0.50, -0.55),
    "Pz": (0.0, -0.55), "P7": (-0.81, -0.59), "P8": (0.81, -0.59), "O1": (-0.31, -0.95), "O2": (0.31, -0.95),
}
LAGS = list(range(-180, 181, 10))


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3:
        return None
    sx, sy = x.std(), y.std()
    if sx < 1e-12 or sy < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 500, seed: int = 0):
    if len(x) < 8:
        return None
    rng = np.random.default_rng(seed)
    rs = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r = pearson(x[idx], y[idx])
        if r is not None:
            rs.append(r)
    if len(rs) < 50:
        return None
    return [round(float(np.percentile(rs, 2.5)), 3), round(float(np.percentile(rs, 97.5)), 3)]


class Pipeline:
    def __init__(self, board, config: dict, recorder=None, log=print):
        self.board = board
        self.cfg = config
        self.recorder = recorder
        self.log = log
        self.fs = board.fs
        self.n_ch = board.n_ch
        labels = config.get("labels_16" if self.n_ch == 16 else "labels_8") or (LABELS_16 if self.n_ch == 16 else LABELS_8)
        self.labels = list(labels)[: self.n_ch]
        self.colors = [CHANNEL_COLORS[i % 8] for i in range(self.n_ch)]
        self.enabled = [True] * self.n_ch
        # display filter defaults follow the OpenBCI GUI: 60 Hz notch, 5-50 Hz band-pass
        self.filter_cfg = {"notch": float(config.get("notch_hz", 60)), "bp_lo": float(config.get("bp_lo", 5.0)),
                           "bp_hi": float(config.get("bp_hi", 50.0))}
        self.chain = FilterChain(self.fs, self.n_ch, self.filter_cfg["notch"], self.filter_cfg["bp_lo"], self.filter_cfg["bp_hi"])
        self.epoch_s = float(config.get("epoch_s", 2.0))
        fi = config.get("flow_index", {})
        self.flow_weights = {"theta": fi.get("theta", 0.5), "alpha": fi.get("alpha", 0.25), "beta": fi.get("beta", 0.25)}
        self.flow = FlowIndex(self.labels, self.flow_weights, int(fi.get("baseline_epochs", 150)), int(fi.get("warmup_epochs", 15)))
        hist_s = float(config.get("history_hours", 4)) * 3600
        self.max_hist = int(hist_s / self.epoch_s)
        # filtered sample queue for the browser (drained by the SSE loop)
        self.lock = threading.Lock()
        self._pending = []            # list of (t0, filtered [k, n_ch], raw_peak [n_ch])
        # epoch state
        self._since_epoch = 0
        self.last_analysis = None
        self.eeg_hist: deque = deque(maxlen=self.max_hist)      # (t, flow_eeg)
        self.band_hist: deque = deque(maxlen=self.max_hist)     # (t, flow or None, sample)
        self.pairs: deque = deque(maxlen=self.max_hist)         # (t, band_flow, eeg_flow)
        self.marks: list = []
        self.band_source = "none"
        self.pair_window_s = 60.0
        self.latest_band = None
        self.stats = self._empty_stats()
        board.buffer.listeners.append(self._on_samples)

    # --- configuration ----------------------------------------------------------
    def set_filter(self, notch: float, bp_lo: float, bp_hi: float) -> None:
        self.filter_cfg = {"notch": float(notch), "bp_lo": float(bp_lo), "bp_hi": float(bp_hi)}
        with self.lock:
            self.chain = FilterChain(self.fs, self.n_ch, notch, bp_lo, bp_hi)

    def set_channel(self, index: int, enabled: bool) -> None:
        if 0 <= index < self.n_ch:
            self.enabled[index] = bool(enabled)

    def set_flow_weights(self, theta: float, alpha: float, beta: float) -> None:
        self.flow_weights = {"theta": float(theta), "alpha": float(alpha), "beta": float(beta)}
        self.flow.w_theta, self.flow.w_alpha, self.flow.w_beta = float(theta), float(alpha), float(beta)

    def set_band_source(self, source: str) -> None:
        self.band_source = source
        pw = self.cfg.get("pair_window_s", {})
        self.pair_window_s = float(pw.get(source, 60 if source == "supabase" else 10))

    # --- EEG path (board thread) ------------------------------------------------
    def _on_samples(self, eeg: np.ndarray, accel: np.ndarray, t: np.ndarray, count0: int) -> None:
        with self.lock:
            filt = self.chain.process(eeg)
            self._pending.append((float(t[0]), filt.astype(np.float32), np.abs(eeg).max(axis=0)))
            self._since_epoch += len(eeg)
            due = self._since_epoch >= int(self.fs * self.epoch_s)
            if due:
                self._since_epoch = 0
        if self.recorder is not None:
            self.recorder.eeg(eeg, accel, t, count0)
        if due:
            try:
                self._epoch()
            except Exception as e:  # noqa: BLE001
                self.log(f"[pipeline] epoch error: {e!r}")

    def drain_samples(self) -> dict | None:
        with self.lock:
            if not self._pending:
                return None
            items, self._pending = self._pending, []
        t0 = items[0][0]
        filt = np.concatenate([it[1] for it in items], axis=0)
        peak = np.max(np.stack([it[2] for it in items]), axis=0)
        return {"t0": round(t0, 4), "fs": self.fs, "n": int(len(filt)),
                "ch": [np.round(filt[:, c], 2).tolist() for c in range(self.n_ch)],
                "raw_peak": np.round(peak, 2).tolist()}

    def _epoch(self) -> None:
        n = int(self.fs * self.epoch_s)
        raw, t = self.board.buffer.last(n)
        if len(raw) < n // 2:
            return
        t_end = float(t[-1])
        q = quality(raw, FULL_SCALE_UV)
        # the spectrum uses a FIXED analysis filter (mains notch + 1-55 Hz), independent of the
        # display band-pass, so a 5-50 Hz display setting never removes theta from the flow index
        chain = FilterChain(self.fs, self.n_ch, self.filter_cfg["notch"], 1.0, min(55.0, self.fs / 2 - 1))
        pad = np.concatenate([raw[::-1][: n // 2], raw], axis=0)           # warm the filter with a mirror
        filt = chain.process(pad)[n // 2:]
        f, P = psd(filt, self.fs)
        keep = f <= 60.0
        f, P = f[keep], P[keep]
        ab = band_powers(f, P)
        rel = relative_powers(ab)
        good = [self.enabled[i] and q[i]["state"] == "good" for i in range(self.n_ch)]
        rec = self.flow.update(rel, ab, good)
        rec = {"t": round(t_end, 3), **rec}
        self.eeg_hist.append((t_end, rec["flow_eeg"]))
        analysis = {
            "t": round(t_end, 3),
            "fft": {"freqs": np.round(f, 3).tolist(), "psd": [np.round(P[:, c], 5).tolist() for c in range(self.n_ch)]},
            "bands": {"order": BAND_ORDER,
                      "abs": {b: np.round(ab[b], 4).tolist() for b in BAND_ORDER},
                      "rel": {b: np.round(rel[b], 4).tolist() for b in BAND_ORDER}},
            "quality": q,
            "enabled": list(self.enabled),
            "flow_eeg": rec,
            "band": self.band_view(),
            "stats": self.stats,
        }
        self.last_analysis = analysis
        if self.recorder is not None:
            epoch_rec = dict(rec)
            for i, lab in enumerate(self.labels):
                for b in ("theta", "alpha", "beta"):
                    epoch_rec[f"{b}_abs_{lab}"] = round(float(ab[b][i]), 4)
                epoch_rec[f"q_{lab}"] = q[i]["state"]
            self.recorder.epoch(epoch_rec)

    # --- band path (band-link thread) -------------------------------------------
    def on_band(self, s: dict) -> None:
        s = dict(s)
        s.setdefault("source", self.band_source)
        self.latest_band = s
        self.band_hist.append((s["t"], s.get("flow"), s))
        if self.recorder is not None:
            self.recorder.band({"t": s["t"], "flow": "" if s.get("flow") is None else s["flow"],
                                "hr": "" if s.get("hr") is None else s["hr"], "anom": s.get("anom", ""),
                                "held": bool(s.get("held")), "settling": bool(s.get("settling")), "fhi": s.get("fhi")})
        usable = (s.get("flow") is not None and not s.get("held") and not s.get("asleep")
                  and not s.get("settling") and not s.get("poor_contact"))
        if usable:
            eeg = self._eeg_mean(s["t"] - self.pair_window_s, s["t"])
            if eeg is not None:
                self.pairs.append((s["t"], float(s["flow"]), eeg))
                self.stats = self._compute_stats()

    def _eeg_mean(self, t0: float, t1: float) -> float | None:
        vals = [v for (t, v) in self.eeg_hist if t0 <= t <= t1]
        return float(np.mean(vals)) if vals else None

    def band_view(self) -> dict | None:
        s = self.latest_band
        if s is None:
            return None
        v = {k: s.get(k) for k in ("t", "flow", "hr", "anom", "mode", "held", "settling", "asleep", "poor_contact",
                                   "flow_state", "window", "band", "source", "risk", "battery_mv", "learn_pct",
                                   "low_battery", "user_id")}
        v["band"] = v.get("band") or s.get("unit")
        v["user_id_short"] = (s.get("user_id") or "")[:8] or None
        v.pop("user_id", None)
        v["age_s"] = round(time.time() - s["t"], 1)
        return v

    # --- marks ----------------------------------------------------------------
    def mark(self, kind: str, value, note: str = "") -> dict:
        m = {"t": time.time(), "kind": kind, "value": value, "note": note}
        self.marks.append(m)
        if self.recorder is not None:
            self.recorder.mark(kind, value, note)
        self.stats = self._compute_stats()
        return m

    # --- stats ----------------------------------------------------------------
    def _empty_stats(self) -> dict:
        return {"n_pairs": 0, "r": None, "r_ci": None, "r_last_5min": None, "mae": None,
                "band_minus_eeg_mean": None, "best_lag_s": None, "r_at_best_lag": None, "lags": [],
                "felt": {"n": 0, "r_band": None, "r_eeg": None}, "pair_window_s": self.pair_window_s}

    def _compute_stats(self) -> dict:
        st = self._empty_stats()
        if not self.pairs:
            return st
        arr = np.array([(t, b, e) for (t, b, e) in self.pairs], float)
        t, b, e = arr[:, 0], arr[:, 1], arr[:, 2]
        st["n_pairs"] = int(len(arr))
        r = pearson(b, e)
        st["r"] = None if r is None else round(r, 3)
        st["r_ci"] = bootstrap_ci(b, e)
        recent = t >= t[-1] - 300
        rr = pearson(b[recent], e[recent])
        st["r_last_5min"] = None if rr is None else round(rr, 3)
        st["mae"] = round(float(np.mean(np.abs(b - e))), 3)
        st["band_minus_eeg_mean"] = round(float(np.mean(b - e)), 3)
        # lag sweep: EEG re-averaged at band time shifted by lag (positive = band lags EEG)
        if len(arr) >= 8 and self.eeg_hist:
            eh = np.array(list(self.eeg_hist), float)
            lags = []
            for lag in LAGS:
                ev = []
                for ti in t:
                    t1 = ti - lag
                    m = (eh[:, 0] >= t1 - self.pair_window_s) & (eh[:, 0] <= t1)
                    ev.append(eh[m, 1].mean() if m.any() else np.nan)
                ev = np.array(ev)
                ok = ~np.isnan(ev)
                rl = pearson(b[ok], ev[ok]) if ok.sum() >= 8 else None
                lags.append([lag, None if rl is None else round(rl, 3)])
            st["lags"] = lags
            valid = [(lg, v) for lg, v in lags if v is not None]
            if valid:
                best = max(valid, key=lambda x: x[1])
                st["best_lag_s"], st["r_at_best_lag"] = best
        # felt-flow marks
        felt = [m for m in self.marks if m["kind"] == "felt_flow" and m.get("value") is not None]
        if felt:
            fb, fe, fv = [], [], []
            for m in felt:
                try:
                    v = float(m["value"])
                except (TypeError, ValueError):
                    continue
                eeg = self._eeg_mean(m["t"] - 60, m["t"])
                near = [(abs(tb - m["t"]), fl) for (tb, fl, _) in self.band_hist if fl is not None and abs(tb - m["t"]) <= 90]
                if eeg is not None and near:
                    fv.append(v)
                    fe.append(eeg)
                    fb.append(min(near)[1])
            st["felt"] = {"n": len(fv), "r_band": None, "r_eeg": None}
            if len(fv) >= 3:
                rb, re_ = pearson(np.array(fv), np.array(fb)), pearson(np.array(fv), np.array(fe))
                st["felt"]["r_band"] = None if rb is None else round(rb, 3)
                st["felt"]["r_eeg"] = None if re_ is None else round(re_, 3)
        return st

    # --- history for the browser ----------------------------------------------
    def history(self) -> dict:
        return {"pairs": [[round(t, 2), round(b, 4), round(e, 4)] for (t, b, e) in self.pairs],
                "eeg": [[round(t, 2), v] for (t, v) in self.eeg_hist],
                "band": [[round(t, 2), fl, bool(s.get("held"))] for (t, fl, s) in self.band_hist],
                "marks": [[round(m["t"], 2), m["kind"], m["value"]] for m in self.marks]}

    def hello(self) -> dict:
        return {"labels": self.labels, "fs": self.fs, "n_ch": self.n_ch, "channel_colors": self.colors,
                "positions": {lab: list(POSITIONS.get(lab, (0.0, 0.0))) for lab in self.labels}}
