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
from dsp import (BAND_ORDER, INDEX_BANDS, TOTAL_BAND, FilterChain, band_powers, band_powers_of, psd, quality,
                 relative_powers, welch)
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
# the band's per-second vitals (USB status line / sim); recorded in band.csv and shown in the panel
BAND_EXTRA = ("spo2", "pi", "rmssd", "sdnn", "rr", "flow_raw", "base_hr", "phase", "uptime_s", "flow_open5", "flow_end5")


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3:
        return None
    sx, sy = x.std(), y.std()
    if sx < 1e-12 or sy < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _avg_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties given their average rank (Spearman convention)."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


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
        from flow_index import DEFAULT_WEIGHTS
        self.flow_weights = {k: float(fi.get(k, DEFAULT_WEIGHTS[k])) for k in ("theta", "alpha", "beta")}
        self.flow = FlowIndex(self.labels, self.flow_weights, str(fi.get("baseline_mode", "rest")),
                              int(fi.get("baseline_epochs", 150)), int(fi.get("warmup_epochs", 15)))
        self.artifact_p2p_uv = float(fi.get("artifact_p2p_uv", 150.0))
        self.artifact_abs_uv = float(fi.get("artifact_abs_uv", 100.0))
        self.analysis_hi_hz = float(fi.get("analysis_hi_hz", 45.0))
        self.index_window_s = float(fi.get("index_window_s", 10.0))      # Welch window for the index
        self.index_seg_s = float(fi.get("index_seg_s", 2.0))
        self.index_min_clean = float(fi.get("index_min_clean_fraction", 0.5))
        self.blocks: list[dict] = []          # [{index, condition, t0, t1, baseline}], from protocol marks
        self.ratings: list[dict] = []         # [{t, block, condition, items}]
        self.trials: list[dict] = []
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
        self.last_raw_peak = None     # |uV| max per channel of the last chunk (setup helper)
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
            peak = np.abs(eeg).max(axis=0)
            self.last_raw_peak = peak.tolist()
            self._pending.append((float(t[0]), filt.astype(np.float32), peak))
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
        n_idx = int(self.fs * self.index_window_s)
        raw_long, t = self.board.buffer.last(max(n, n_idx))
        if len(raw_long) < n // 2:
            return
        t_end = float(t[-1])
        raw = raw_long[-n:]
        q = quality(raw, FULL_SCALE_UV)
        # the spectra use a FIXED analysis filter (mains notch + 1-45 Hz), independent of the
        # display band-pass, so a 5-50 Hz display setting never removes theta from the flow index
        chain = FilterChain(self.fs, self.n_ch, self.filter_cfg["notch"], 1.0, min(self.analysis_hi_hz, self.fs / 2 - 1))
        pad = np.concatenate([raw_long[::-1][: n // 2], raw_long], axis=0)   # warm the filter with a mirror
        filt_long = chain.process(pad)[n // 2:]
        filt = filt_long[-n:]
        # display spectrum: the last epoch
        f, P = psd(filt, self.fs)
        keep = f <= 60.0
        f, P = f[keep], P[keep]
        ab = band_powers(f, P)
        rel = relative_powers(ab)
        good = [self.enabled[i] and q[i]["state"] == "good" for i in range(self.n_ch)]
        used = [i for i in range(self.n_ch) if good[i]] or list(range(self.n_ch))
        p2p = float(np.max(filt[:, used].max(axis=0) - filt[:, used].min(axis=0)))

        # index spectrum: Welch over the last index_window_s, dropping artifact segments
        def reject(seg: np.ndarray) -> bool:
            s = seg[:, used]
            return bool(np.max(s.max(axis=0) - s.min(axis=0)) > self.artifact_p2p_uv
                        or np.abs(s).max() > self.artifact_abs_uv)

        fi, Pi, n_used, n_total = welch(filt_long, self.fs, self.index_seg_s, 0.5, reject)
        artifact = n_total == 0 or n_used < max(1, int(np.ceil(self.index_min_clean * n_total)))
        if not artifact:
            idx_abs = band_powers_of(fi, Pi, {**INDEX_BANDS, "total": TOTAL_BAND})
            total = idx_abs.pop("total")
            extra = relative_powers(idx_abs, total)
            idx_disp = band_powers_of(fi, Pi, {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)})
            rel_idx = {**rel, **relative_powers(idx_disp, total)}      # theta/alpha/beta from the index window
            abs_idx = {**ab, **idx_disp}
        else:
            extra, rel_idx, abs_idx = None, rel, ab
        rec = self.flow.update(rel_idx, abs_idx, good, artifact=artifact, extra=extra)
        rec = {"t": round(t_end, 3), **rec, "p2p_uv": round(p2p, 1), "n_segments_used": int(n_used)}
        if rec["flow_eeg"] is not None:
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
            rec = {"t": s["t"], "flow": "" if s.get("flow") is None else s["flow"],
                   "hr": "" if s.get("hr") is None else s["hr"], "anom": s.get("anom", ""),
                   "held": bool(s.get("held")), "settling": bool(s.get("settling")), "fhi": s.get("fhi"),
                   "asleep": bool(s.get("asleep")), "poor_contact": bool(s.get("poor_contact"))}
            for k in BAND_EXTRA:
                rec[k] = s.get(k)
            self.recorder.band(rec)
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
                                   "low_battery", "user_id", *BAND_EXTRA)}
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
        if kind == "rating":
            try:
                import json as _json
                d = _json.loads(value) if isinstance(value, str) else dict(value)
                self.ratings.append({"t": m["t"], "block": d.get("block"), "condition": d.get("condition"),
                                     "items": {k: v for k, v in d.items() if k not in ("block", "condition")}})
            except (ValueError, TypeError):
                pass
        self.stats = self._compute_stats()
        return m

    def on_protocol(self, ev: dict) -> None:
        """Keep the block table (for per-condition stats) from the runner's transitions."""
        phase = ev.get("phase")
        if phase == "block_start":
            self.blocks.append({"index": ev.get("block_index"), "condition": ev.get("condition"),
                                "t0": time.time(), "t1": None, "baseline": bool(ev.get("baseline"))})
        elif phase in ("block_end",) and self.blocks and self.blocks[-1]["t1"] is None:
            self.blocks[-1]["t1"] = time.time()
        elif phase in ("stopped",) and self.blocks and self.blocks[-1]["t1"] is None:
            self.blocks[-1]["t1"] = time.time()

    def trial(self, rec: dict) -> None:
        self.trials.append(rec)
        if self.recorder is not None:
            self.recorder.trial(rec)

    def flow_status(self) -> dict:
        return {"baseline_mode": self.flow.baselines["theta"].mode, "baseline_state": self.flow.baseline_state,
                "baseline_ready": self.flow.baseline_ready, "epochs": self.flow.n_epochs,
                "artifact_epochs": self.flow.n_artifact,
                "artifact_p2p_uv": self.artifact_p2p_uv, "artifact_abs_uv": self.artifact_abs_uv,
                "weights": self.flow_weights,
                "frontal_sites": self.flow.last.get("frontal_sites") if self.flow.last else None}

    # --- per-condition (block-level) statistics ------------------------------------
    def _block_stats(self) -> dict:
        """Means of the EEG index, the band's flow and the FSS score per block, grouped by
        condition; the primary validation evidence (blocks are the independent units)."""
        rows = []
        for b in self.blocks:
            if b["condition"] in ("pause",) or b["t0"] is None:
                continue
            t0, t1 = b["t0"], b["t1"] or time.time()
            if t1 - t0 < 30:
                continue
            eeg = [v for (t, v) in self.eeg_hist if t0 <= t <= t1 and v is not None]
            band = [fl for (t, fl, s) in self.band_hist
                    if t0 <= t <= t1 and fl is not None and not s.get("held") and not s.get("asleep")
                    and not s.get("settling") and not s.get("poor_contact")]
            fss = None
            for r in self.ratings:
                if r.get("block") == b["index"]:
                    core = [v for k, v in r["items"].items() if k not in ("demand", "flow_overall")]
                    fss = float(np.mean(core)) if core else None
                    felt = r["items"].get("flow_overall")
                    break
            else:
                felt = None
            rows.append({"index": b["index"], "condition": b["condition"],
                         "eeg_mean": round(float(np.mean(eeg)), 4) if eeg else None,
                         "band_mean": round(float(np.mean(band)), 4) if band else None,
                         "fss_mean": None if fss is None else round(fss, 3),
                         "felt": felt, "n_eeg": len(eeg), "n_band": len(band)})
        by: dict[str, list] = {}
        for r in rows:
            by.setdefault(r["condition"], []).append(r)
        conds = []
        for c, rs in by.items():
            def m(key):
                vals = [r[key] for r in rs if r[key] is not None]
                return round(float(np.mean(vals)), 4) if vals else None
            conds.append({"condition": c, "n_blocks": len(rs), "eeg_mean": m("eeg_mean"),
                          "band_mean": m("band_mean"), "fss_mean": m("fss_mean"), "felt_mean": m("felt")})
        non_task = ("rest", "rest_ec", "pause")
        ordering = {}
        for key in ("eeg_mean", "band_mean", "fss_mean"):
            have = [(c["condition"], c[key]) for c in conds if c[key] is not None and c["condition"] not in non_task]
            if len(have) >= 2:
                ordering[key.split("_")[0]] = ">".join(c for c, _ in sorted(have, key=lambda x: -x[1]))
        # block-level Spearman between EEG and band means over TASK blocks (the pre-registered
        # block test; tie-aware average ranks, like report.py)
        pairs = [(r["eeg_mean"], r["band_mean"]) for r in rows
                 if r["eeg_mean"] is not None and r["band_mean"] is not None and r["condition"] not in non_task]
        rho = None
        if len(pairs) >= 4:
            a = np.array(pairs)
            rho = pearson(_avg_ranks(a[:, 0]), _avg_ranks(a[:, 1]))
        return {"rows": rows, "conditions": conds, "ordering": ordering,
                "block_rho": None if rho is None else round(rho, 3), "n_blocks": len(pairs)}

    # --- stats ----------------------------------------------------------------
    def _empty_stats(self) -> dict:
        return {"n_pairs": 0, "r": None, "r_ci": None, "r_last_5min": None, "mae": None,
                "band_minus_eeg_mean": None, "best_lag_s": None, "r_at_best_lag": None, "lags": [],
                "felt": {"n": 0, "r_band": None, "r_eeg": None}, "pair_window_s": self.pair_window_s,
                "blocks": None}

    def _compute_stats(self) -> dict:
        st = self._empty_stats()
        try:
            st["blocks"] = self._block_stats()
        except Exception as e:  # noqa: BLE001
            self.log(f"[pipeline] block stats error: {e!r}")
            st["blocks"] = None
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
                "vitals": self.vitals(),
                "marks": [[round(m["t"], 2), m["kind"], m["value"]] for m in self.marks]}

    def vitals(self, max_n: int = 1800) -> list:
        """[t, hr, spo2, pi, flow_raw, rmssd] per band sample, newest `max_n` (the band panel chart)."""
        out = []
        for (t, fl, s) in list(self.band_hist)[-max_n:]:
            out.append([round(t, 2), s.get("hr"), s.get("spo2"), s.get("pi"), s.get("flow_raw"), s.get("rmssd")])
        return out

    def hello(self) -> dict:
        return {"labels": self.labels, "fs": self.fs, "n_ch": self.n_ch, "channel_colors": self.colors,
                "positions": {lab: list(POSITIONS.get(lab, (0.0, 0.0))) for lab in self.labels}}
