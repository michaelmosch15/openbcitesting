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
                 relative_powers, welch, welch_masked)
from flow_index import FRONTAL_FALLBACK, FRONTAL_PREFERRED, Baseline, FlowIndex, sigmoid

LABELS_8 = ["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"]
LABELS_16 = LABELS_8 + ["F7", "F8", "F3", "F4", "T7", "T8", "P3", "P4"]
CHANNEL_COLORS = ["#818181", "#7C4B8D", "#36579E", "#317159", "#DDB20D", "#FD5E34", "#E0382D", "#A25231"]
# 10-20 positions on the unit circle (x right, y up), for the head plot
POSITIONS = {
    "Fp1": (-0.31, 0.95), "Fpz": (0.0, 1.0), "Fp2": (0.31, 0.95), "F7": (-0.81, 0.59), "F8": (0.81, 0.59),
    "AF7": (-0.59, 0.81), "AF3": (-0.33, 0.81), "AFz": (0.0, 0.81), "AF4": (0.33, 0.81), "AF8": (0.59, 0.81),
    "F5": (-0.68, 0.58), "F1": (-0.26, 0.56), "F2": (0.26, 0.56), "F6": (0.68, 0.58),
    "F3": (-0.50, 0.55), "F4": (0.50, 0.55), "Fz": (0.0, 0.55), "C3": (-0.55, 0.0), "C4": (0.55, 0.0),
    "Cz": (0.0, 0.0), "T7": (-1.0, 0.0), "T8": (1.0, 0.0), "P3": (-0.50, -0.55), "P4": (0.50, -0.55),
    "Pz": (0.0, -0.55), "P7": (-0.81, -0.59), "P8": (0.81, -0.59), "O1": (-0.31, -0.95), "O2": (0.31, -0.95),
}
LAGS = list(range(-180, 181, 10))
# the band's per-second vitals and flags (USB status line / v2 row / sim); recorded in band.csv, shown in the panel
from recorder import BAND_EXTRA  # noqa: E402


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


# --- live helmet flow (display only; the evidence for these choices is tools/live_eval/) -------
LIVE_HP_HZ = 4.0            # extra high-pass before the artifact test and the spectrum: eye / head movement is below it
LIVE_THETA, LIVE_TOTAL = (5.0, 8.0), (5.0, 30.0)   # clear of the high-pass skirt and blink leakage; no EMG / mains above 30
LIVE_P2P_UV, LIVE_ABS_UV = 100.0, 60.0             # artifact limits on the high-passed 2-s segment (EEG there is ~10 uV rms)
LIVE_HORIZON_S = 60.0       # clean segments are pooled over this long (10 s of EEG cannot pin a 3-Hz band down)
LIVE_MIN_SEGMENTS = 10      # clean 2-s segments (50 % overlap: ~11 s of signal) a channel needs in the horizon
LIVE_WARM_S = 2.0           # start of the freshly filtered window: high-pass warm-up on the DC offset, never used
LIVE_SD_FLOOR = 0.15        # log units, ~2x the estimator noise at 60 s: noise alone moves the number ~0.4-0.6


_LIVE_HP_KERNELS: dict = {}


def _live_hp_kernel(fs: float) -> np.ndarray:
    """The LIVE_HP_HZ high-pass (the same 4th-order Butterworth sections as FilterChain) as a 1-s
    FIR kernel: its impulse response has died out by then, and a convolution is ~100x faster
    than the per-sample biquad loop."""
    k = _LIVE_HP_KERNELS.get(fs)
    if k is None:
        imp = np.zeros((int(fs), 1))
        imp[0, 0] = 1.0
        k = _LIVE_HP_KERNELS[fs] = FilterChain(fs, 1, 0, LIVE_HP_HZ, 0).process(imp)[:, 0]
    return k


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
        # LIVE helmet flow (display / exploratory only, never paired or reported; see _live_flow): theta
        # 5-8 Hz as a share of 5-30 Hz on a 4 Hz high-passed copy, clean 2-s segments pooled over 60 s,
        # z vs the session until the protocol's rest block freezes a resting baseline
        self.live_base = Baseline("rest", warmup=15)
        self._recent_corr: deque = deque(maxlen=15)             # last 15 epochs of mean |corr| (reference gate)
        self.eeg_smooth_hist: deque = deque(maxlen=20000)       # chart line: (t, 60-s median of the strict flow_eeg, or None)
        self.artifact_p2p_uv = float(fi.get("artifact_p2p_uv", 150.0))
        self.artifact_abs_uv = float(fi.get("artifact_abs_uv", 100.0))
        self.analysis_hi_hz = float(fi.get("analysis_hi_hz", 45.0))
        self.index_window_s = float(fi.get("index_window_s", 10.0))      # Welch window for the index
        self.index_seg_s = float(fi.get("index_seg_s", 2.0))
        self.index_min_clean = float(fi.get("index_min_clean_fraction", 0.5))
        # "channel": each channel keeps its own clean segments and the index uses the channels
        # that stay >= index_min_clean_fraction clean (needs min_clean_channels); "epoch": a
        # segment is dropped for everyone when any used channel is noisy (the v0.2 rule)
        self.artifact_scope = str(fi.get("artifact_scope", "channel"))
        self.min_clean_channels = int(fi.get("min_clean_channels", 2))
        self.readiness: dict = {"ready": False, "reasons": ["no data"], "clean_channels": []}
        self._recent_artifact: deque = deque(maxlen=15)
        # the index is withheld (flow_eeg null, `gated`) while the good channels are one shared
        # signal (|corr| >= 0.9: no reference clip, or the headset is not on a head) or fewer
        # than min_clean_channels are clean - a quiet desk looks "clean" to the artifact rule
        self.gate_on_readiness = bool(fi.get("gate_on_readiness", True))
        self.gate_reasons = tuple(fi.get("gate_reasons", ["reference", "few"]))
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
        self.autocal = None           # autocal.Capture while a placement calibration runs
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

    def set_labels(self, labels: list[str]) -> list[str]:
        """Relabel the channels live (auto-calibration or a manual fix): the index's site
        sets, the recorder's header for the NEXT session, the head plot. Returns the labels."""
        labels = [str(x)[:8] for x in labels]
        if len(labels) != self.n_ch:
            raise ValueError(f"{len(labels)} labels for {self.n_ch} channels")
        if len(set(labels)) != len(labels):
            raise ValueError("labels must be distinct")
        self.labels = list(labels)
        self.flow.labels = self.labels
        if self.recorder is not None:
            self.recorder.labels = self.labels
        return self.labels

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
            meta = getattr(self.board.buffer, "push_meta", None)      # (packet index, raw rx time) from a Cyton
            if meta is not None:
                self.recorder.eeg(eeg, accel, t, count0, pkt=meta[0], rx=meta[1])
            else:
                self.recorder.eeg(eeg, accel, t, count0)
        cap = self.autocal
        if cap is not None:
            cap.push(eeg)
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
        # mains-hum share of the raw signal (58-62 Hz over 1-100 Hz): a comb on hair, not scalp, reads ~0.9+
        try:
            xr = raw - raw.mean(axis=0)
            S = np.abs(np.fft.rfft(xr * np.hanning(len(xr))[:, None], axis=0)) ** 2
            fr_ = np.fft.rfftfreq(len(xr), 1.0 / self.fs)
            tot = S[(fr_ >= 1) & (fr_ < 100)].sum(axis=0)
            hum = S[(fr_ >= 58) & (fr_ < 62)].sum(axis=0) / np.where(tot > 0, tot, 1.0)
            for i in range(self.n_ch):
                q[i]["hum"] = round(float(hum[i]), 2)
        except Exception:  # noqa: BLE001
            pass
        # the spectra use a FIXED analysis filter (mains notch + 1-45 Hz), independent of the
        # display band-pass, so a 5-50 Hz display setting never removes theta from the flow index
        chain = FilterChain(self.fs, self.n_ch, self.filter_cfg["notch"], 1.0, min(self.analysis_hi_hz, self.fs / 2 - 1))
        # a fresh filter per epoch: remove the electrode DC offset first (often -100..-200 mV; a step
        # that size rings through the 1 Hz high-pass for seconds and fails the artifact test), then
        # warm it with the window's own START mirrored, so the pad joins the data without a jump
        rr = raw_long - np.median(raw_long, axis=0)
        pad = np.concatenate([rr[: n // 2][::-1], rr], axis=0)
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
        good_epoch = list(good)
        clean_channels: list[int] = [i for i in range(self.n_ch) if good[i]]
        if self.artifact_scope == "channel":
            def bad(seg: np.ndarray) -> np.ndarray:
                m = (seg.max(axis=0) - seg.min(axis=0) > self.artifact_p2p_uv) | (np.abs(seg).max(axis=0) > self.artifact_abs_uv)
                m[[i for i in range(self.n_ch) if not good[i]]] = True
                return m
            fi, Pi, used_per_ch, n_total = welch_masked(filt_long, self.fs, self.index_seg_s, 0.5, bad)
            need = max(1, int(np.ceil(self.index_min_clean * n_total))) if n_total else 1
            clean_channels = [i for i in range(self.n_ch) if good[i] and used_per_ch[i] >= need]
            good_epoch = [i in clean_channels for i in range(self.n_ch)]
            artifact = n_total == 0 or len(clean_channels) < self.min_clean_channels
            n_used = int(np.median(used_per_ch[clean_channels])) if clean_channels else 0
            if not artifact:
                Pi = np.nan_to_num(Pi, nan=0.0)
        else:
            def reject(seg: np.ndarray) -> bool:
                s = seg[:, used]
                return bool(np.max(s.max(axis=0) - s.min(axis=0)) > self.artifact_p2p_uv
                            or np.abs(s).max() > self.artifact_abs_uv)

            fi, Pi, n_used, n_total = welch(filt_long, self.fs, self.index_seg_s, 0.5, reject)
            artifact = n_total == 0 or n_used < max(1, int(np.ceil(self.index_min_clean * n_total)))
            if artifact:
                clean_channels = []
        if not artifact:
            idx_abs = band_powers_of(fi, Pi, {**INDEX_BANDS, "total": TOTAL_BAND})
            total = idx_abs.pop("total")
            extra = relative_powers(idx_abs, total)
            idx_disp = band_powers_of(fi, Pi, {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)})
            rel_idx = {**rel, **relative_powers(idx_disp, total)}      # theta/alpha/beta from the index window
            abs_idx = {**ab, **idx_disp}
        else:
            extra, rel_idx, abs_idx = None, rel, ab
        self._recent_artifact.append(bool(artifact))
        self.readiness = self._readiness(q, good, filt, raw, clean_channels)
        gated = [r for r in self.readiness["reasons"] if r in self.gate_reasons] if self.gate_on_readiness else []
        if gated and not artifact:
            artifact = True                     # withheld: no index, no baseline update, no pair
            extra, rel_idx, abs_idx = None, rel, ab
        rec = self.flow.update(rel_idx, abs_idx, good_epoch, artifact=artifact, extra=extra)
        rec = {"t": round(t_end, 3), **rec, "rest_block_needed": self.rest_block_needed(),
               "p2p_uv": round(p2p, 1), "n_segments_used": int(n_used),
               "clean_channels": [self.labels[i] for i in clean_channels], "gated": ",".join(gated) or None}
        rec.update(self._live_flow(filt_long, good))
        if rec["flow_eeg"] is not None:
            self.eeg_hist.append((t_end, rec["flow_eeg"]))
        # chart line: median of the valid strict values in the last 60 s (None when there were none), so
        # the helmet trace is continuous across short artifact gaps. Display only; pairs/report use eeg_hist.
        recent = [v for (tt, v) in list(self.eeg_hist)[-40:] if t_end - tt <= 60.0]
        rec["flow_eeg_60s"] = round(float(np.median(recent)), 4) if recent else None
        rec["flow_eeg_60s_n"] = len(recent)
        # the helmet LINE is the strict index only. The live index (flow_live, _live_flow) is exploratory:
        # shown as a labelled number when the strict one is withheld, never plotted as the helmet line.
        # (11-19 epochs.csv files have flow_helmet / flow_helmet_60s = flow_live; those keys are gone.)
        self.eeg_smooth_hist.append((t_end, rec["flow_eeg_60s"]))
        analysis = {
            "t": round(t_end, 3),
            "fft": {"freqs": np.round(f, 3).tolist(), "psd": [np.round(P[:, c], 5).tolist() for c in range(self.n_ch)]},
            "bands": {"order": BAND_ORDER,
                      "abs": {b: np.round(ab[b], 4).tolist() for b in BAND_ORDER},
                      "rel": {b: np.round(rel[b], 4).tolist() for b in BAND_ORDER}},
            "quality": q,
            "enabled": list(self.enabled),
            "readiness": self.readiness,
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

    def _live_flow(self, filt_long: np.ndarray, good: list) -> dict:
        """Live helmet flow: frontal theta (5-8 Hz) as a share of 5-30 Hz, from artifact-free 2-s
        segments of a 4 Hz high-passed copy, pooled over the last LIVE_HORIZON_S, z vs the session
        (the rest block once frozen). Not the pre-registered index (flow_eeg); never paired with
        the band or used by report.py. Never raises: a failure is one logged line and no number."""
        none = {"flow_live": None, "live_channels": None, "live_z": None}
        try:
            return self._live_flow_inner(filt_long, good, none)
        except Exception as e:  # noqa: BLE001
            # rate-limited, not once-only: a fault that comes back later is still seen
            now = time.time()
            self._live_err_n = getattr(self, "_live_err_n", 0) + 1
            if now - getattr(self, "_live_err_t", 0.0) >= 60.0:
                extra = f" ({self._live_err_n - 1} more since the last report)" if self._live_err_n > 1 else ""
                self.log(f"[pipeline] live flow error: {e!r}{extra}")
                self._live_err_t, self._live_err_n = now, 0
            return none

    def _live_flow_inner(self, filt_long: np.ndarray, good: list, none: dict) -> dict:
        if "reference" in self.readiness.get("reasons", []):          # not on a head / no ear clip
            return none
        fr = [i for i, lab in enumerate(self.labels) if good[i] and lab in FRONTAL_PREFERRED]
        fr = fr or [i for i, lab in enumerate(self.labels) if good[i] and lab in FRONTAL_FALLBACK]
        if not fr:
            return none
        fs = self.fs
        seg = int(fs * self.index_seg_s)
        step = max(1, seg // 2)
        segs = getattr(self, "_live_segs", None)     # (abs start, theta [n_ch], total [n_ch], clean [n_ch]) per segment
        if segs is None:                             # made here, not in __init__, so a hot reload keeps working
            segs = self._live_segs = deque()
            self._live_next = 0
        end = int(self.board.buffer.count)           # absolute index one past the newest sample of filt_long
        start = end - len(filt_long)
        if end < self._live_next:                    # the sample counter restarted (new board)
            segs.clear()
            self._live_next = 0
        # each segment is analysed once, on a fixed grid of absolute sample positions, never inside
        # the filter warm-up at the start of the window
        s = -(-max(self._live_next, start + int(LIVE_WARM_S * fs)) // step) * step
        if s + seg <= end:
            # eye and head movement live below ~4 Hz and are far larger than the EEG: on a 4 Hz
            # high-passed copy they neither fail the amplitude test nor leak into the theta band
            k = _live_hp_kernel(fs)
            a = s - start - (len(k) - 1)             # >= 0: LIVE_WARM_S is longer than the kernel
            hp = np.zeros((end - s, self.n_ch))
            for i in fr:
                hp[:, i] = np.convolve(filt_long[a:, i], k, "valid")
            in_fr = np.zeros(self.n_ch, bool)
            in_fr[fr] = True
            while s + seg <= end:
                x = hp[s - (end - len(hp)):s - (end - len(hp)) + seg]
                f, P = psd(x, fs)
                bp = band_powers_of(f, P, {"theta": LIVE_THETA, "total": LIVE_TOTAL})
                clean = (in_fr & (x.max(axis=0) - x.min(axis=0) <= LIVE_P2P_UV) & (np.abs(x).max(axis=0) <= LIVE_ABS_UV)
                         & (bp["theta"] > 0) & (bp["total"] > 0))
                segs.append((s, bp["theta"], bp["total"], clean))
                s += step
            self._live_next = s
        while segs and segs[0][0] < end - int(LIVE_HORIZON_S * fs):
            segs.popleft()
        if not segs:
            return {**none, "live_channels": "no clean signal"}
        th, tot, ok = (np.asarray([e[k] for e in segs]) for k in (1, 2, 3))
        vals = {}
        for i in fr:                                 # per channel: pooled power over its own clean segments
            m = ok[:, i]
            if m.sum() >= LIVE_MIN_SEGMENTS:
                vals[i] = float(np.log(th[m, i].sum() / tot[m, i].sum()))
        if len(vals) < min(2, len(fr)):              # one electrode alone is not trusted when there are others
            return {**none, "live_channels": "no clean signal"}
        names = " ".join(self.labels[i] for i in vals)
        v = float(np.median(list(vals.values())))
        self.live_base.push(v)
        if not self.live_base.ready:
            return {**none, "live_channels": names}
        # the spread is floored: over a steady stretch the session SD is only estimator noise, and
        # dividing by it would stretch that noise over the whole 0-1 range
        z = (v - self.live_base.mean) / max(self.live_base.std, LIVE_SD_FLOOR)
        z = max(-20.0, min(20.0, z))
        return {"flow_live": round(sigmoid(z), 4), "live_channels": names, "live_z": round(z, 3)}

    def _readiness(self, q: list, good: list, filt: np.ndarray, raw: np.ndarray, clean_channels: list) -> dict:
        """Why the index is (not) usable right now, in operator terms. Reasons:
        contact (railed / near / flat channels), reference (the good channels are one shared
        signal: |corr| >= 0.9 - ear clip), settling (raw offsets above 100 mV, still drifting
        after a put-on), artifact (>= 80 % of the last 15 epochs rejected), few (fewer clean
        channels than min_clean_channels)."""
        reasons, detail = [], {}
        bad = [self.labels[i] for i in range(self.n_ch) if self.enabled[i] and q[i]["state"] != "good"]
        if bad:
            reasons.append("contact"); detail["bad_channels"] = bad
        gi = [i for i in range(self.n_ch) if good[i]]
        if len(gi) >= 2:
            x = filt[:, gi]
            sd = x.std(axis=0)
            ok = sd > 1e-6
            if ok.sum() >= 2:
                c = np.corrcoef(x[:, ok].T)
                m = np.abs(c[np.triu_indices(len(c), 1)]).mean()
                detail["common_mode_corr"] = round(float(m), 2)
                # a blink / eye movement makes every forehead channel agree for one 2-s epoch; a missing
                # reference (ear clip off, headset on a desk) keeps them agreeing. Gate on the median of the
                # last 15 epochs (30 s), so only the sustained case withholds the index.
                self._recent_corr.append(float(m))
                ms = float(np.median(self._recent_corr))
                detail["common_mode_corr_30s"] = round(ms, 2)
                if ms >= 0.9:
                    reasons.append("reference")
        off = [self.labels[i] for i in gi if np.abs(raw[:, i]).max() > 100000.0]
        if off:
            reasons.append("settling"); detail["offset_channels"] = off
        if len(self._recent_artifact) >= 5:
            fr = sum(self._recent_artifact) / len(self._recent_artifact)
            detail["artifact_fraction"] = round(fr, 2)
            if fr >= 0.8:
                reasons.append("artifact")
        if len(clean_channels) < self.min_clean_channels:
            reasons.append("few")
        return {"ready": not reasons, "reasons": reasons, "clean_channels": [self.labels[i] for i in clean_channels],
                "n_clean": len(clean_channels), **detail}

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
                  and not s.get("settling") and not s.get("poor_contact") and s.get("usable") is not False)
        if usable:
            eeg = self._eeg_mean(s["t"] - self.pair_window_s, s["t"])
            if eeg is not None:
                self.pairs.append((s["t"], float(s["flow"]), eeg))
                self.stats = self._compute_stats()

    def _eeg_mean(self, t0: float, t1: float) -> float | None:
        vals = [v for (t, v) in list(self.eeg_hist) if t0 <= t <= t1]
        return float(np.mean(vals)) if vals else None

    def band_view(self) -> dict | None:
        s = self.latest_band
        if s is None:
            return None
        v = {k: s.get(k) for k in ("t", "flow", "hr", "anom", "mode", "held", "settling", "asleep", "poor_contact",
                                   "flow_state", "window", "band", "source", "risk", "battery_mv", "learn_pct",
                                   "low_battery", "user_id", "usable", "no_reading", *BAND_EXTRA)}
        v["band"] = v.get("band") or s.get("unit")
        v["user_id_short"] = (s.get("user_id") or "")[:8] or None
        v.pop("user_id", None)
        v["age_s"] = round(time.time() - s["t"], 1)
        return v

    # --- marks ----------------------------------------------------------------
    MARK_DEDUP_S = 2.0
    EDITABLE_MARKS = ("felt_flow", "flow_start", "flow_end", "distracted", "note")

    def mark(self, kind: str, value, note: str = "", client_id: str | None = None) -> dict:
        """Record a mark once. A retry is recognised by its client id, or by the same kind + value
        (+ note) within MARK_DEDUP_S of an earlier mark; the earlier mark comes back with
        duplicate=True and nothing is written twice."""
        now = time.time()
        # only the operator's marks are de-duplicated; protocol / trial marks come once from the runner
        for old in (reversed(self.marks[-50:]) if kind in self.EDITABLE_MARKS else ()):
            if client_id and old.get("client_id") == client_id:
                return {**old, "duplicate": True}
            if (now - old["t"] <= self.MARK_DEDUP_S and old["kind"] == kind and old.get("value") == value
                    and (old.get("note") or "") == (note or "")):
                return {**old, "duplicate": True}
        import uuid
        m = {"t": now, "kind": kind, "value": value, "note": note, "id": uuid.uuid4().hex[:12]}
        if client_id:
            m["client_id"] = str(client_id)[:64]
        self.marks.append(m)
        if self.recorder is not None:
            self.recorder.mark(kind, value, note, mid=m["id"], t=now)
        if kind == "rating":
            try:
                import json as _json
                d = _json.loads(value) if isinstance(value, str) else dict(value)
                self.ratings.append({"t": m["t"], "block": d.get("block"), "condition": d.get("condition"),
                                     "items": {k: v for k, v in d.items() if k not in ("block", "condition")}})
            except (ValueError, TypeError):
                pass
        self._refresh_stats()
        return m

    def edit_mark(self, mark_id: str, value=None, note: str | None = None, delete: bool = False) -> dict:
        """Correct (new value / note) or delete one of the operator's marks. The original row in
        marks.csv stays; an edit row (kind mark_edit / mark_delete, edit_of = id) is appended."""
        m = next((x for x in self.marks if x.get("id") == mark_id), None)
        if m is None:
            raise ValueError(f"no mark {mark_id!r}")
        if m["kind"] not in self.EDITABLE_MARKS:
            raise ValueError(f"{m['kind']} marks come from the protocol and cannot be edited here")
        if delete:
            self.marks.remove(m)
            out = {**m, "deleted": True}
        else:
            if value is not None:
                m["value"] = value
            if note is not None:
                m["note"] = note
            out = dict(m)
        if self.recorder is not None:
            self.recorder.mark("mark_delete" if delete else "mark_edit", "" if delete or value is None else value,
                               "" if note is None else note, edit_of=mark_id)
        self._refresh_stats()
        return out

    def _refresh_stats(self) -> None:
        try:
            self.stats = self._compute_stats()
        except Exception as e:  # noqa: BLE001
            self.log(f"[pipeline] stats error: {e!r}")

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

    def rest_block_needed(self) -> bool:
        """True while a 'rest' baseline mode has no frozen rest block yet: the numbers are then
        z-scores against the session so far, not against the wearer's rest."""
        b = self.flow.baselines["theta"]
        return b.mode == "rest" and not b.frozen

    # --- baseline persistence (a restart must not lose the wearer's rest block) ----------
    def baseline_snapshot(self) -> dict:
        return {"saved_t": round(time.time(), 3), "labels": list(self.labels), "flow": self.flow.state_dict(),
                "live": self.live_base.to_dict(), "state": self.flow.baseline_state}

    def save_baseline(self, path) -> None:
        import json as _json
        from pathlib import Path as _Path
        p = _Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(_json.dumps(self.baseline_snapshot(), indent=1), "utf-8")
        tmp.replace(p)

    def load_baseline(self, path) -> bool:
        """Restore a FROZEN rest baseline saved by save_baseline (same channel labels only).
        Returns True when it was applied."""
        import json as _json
        from pathlib import Path as _Path
        try:
            d = _json.loads(_Path(path).read_text("utf-8"))
        except (OSError, ValueError):
            return False
        if list(d.get("labels") or []) != list(self.labels):
            return False
        fl = d.get("flow") or {}
        if not all((fl.get(k) or {}).get("frozen") for k in ("theta", "alpha", "beta")):
            return False
        self.flow.load_state(fl)
        for b in self.flow.baselines.values():
            b.collecting = False
        if (d.get("live") or {}).get("frozen"):
            self.live_base.load_dict(d["live"])
            self.live_base.collecting = False
        self.baseline_restored_from = str(path)
        return True

    def flow_status(self) -> dict:
        return {"baseline_mode": self.flow.baselines["theta"].mode, "baseline_state": self.flow.baseline_state,
                "baseline_ready": self.flow.baseline_ready, "epochs": self.flow.n_epochs,
                "baseline_restored_from": getattr(self, "baseline_restored_from", None),
                "rest_block_needed": self.rest_block_needed(),
                "artifact_epochs": self.flow.n_artifact,
                "artifact_p2p_uv": self.artifact_p2p_uv, "artifact_abs_uv": self.artifact_abs_uv,
                "artifact_scope": self.artifact_scope, "min_clean_channels": self.min_clean_channels,
                "readiness": self.readiness,
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
            eeg = [v for (t, v) in list(self.eeg_hist) if t0 <= t <= t1 and v is not None]
            band = [fl for (t, fl, s) in list(self.band_hist)
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
        arr = np.array([(t, b, e) for (t, b, e) in list(self.pairs)], float)
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
        felt = [m for m in list(self.marks) if m["kind"] == "felt_flow" and m.get("value") is not None]
        if felt:
            fb, fe, fv = [], [], []
            for m in felt:
                try:
                    v = float(m["value"])
                except (TypeError, ValueError):
                    continue
                eeg = self._eeg_mean(m["t"] - 60, m["t"])
                near = [(abs(tb - m["t"]), fl) for (tb, fl, _) in list(self.band_hist) if fl is not None and abs(tb - m["t"]) <= 90]
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
        return {"pairs": [[round(t, 2), round(b, 4), round(e, 4)] for (t, b, e) in list(self.pairs)],
                "eeg": [[round(t, 2), v] for (t, v) in (list(self.eeg_smooth_hist) or list(self.eeg_hist))],
                "eeg_raw": [[round(t, 2), v] for (t, v) in list(self.eeg_hist)],
                "band": [[round(t, 2), fl, bool(s.get("held"))] for (t, fl, s) in list(self.band_hist)],
                "vitals": self.vitals(),
                "marks": [[round(m["t"], 2), m["kind"], m["value"], m.get("id")] for m in list(self.marks)]}

    def vitals(self, max_n: int = 1800) -> list:
        """[t, hr, spo2, pi, flow_raw, rmssd] per band sample, newest `max_n` (the band panel chart)."""
        out = []
        for (t, fl, s) in list(self.band_hist)[-max_n:]:
            out.append([round(t, 2), s.get("hr"), s.get("spo2"), s.get("pi"), s.get("flow_raw"), s.get("rmssd")])
        return out

    def hello(self) -> dict:
        return {"labels": self.labels, "fs": self.fs, "n_ch": self.n_ch, "channel_colors": self.colors,
                "positions": {lab: list(POSITIONS.get(lab, (0.0, 0.0))) for lab in self.labels}}
