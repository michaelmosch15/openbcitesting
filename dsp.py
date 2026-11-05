"""Signal processing for the EEG widgets: display filters (notch + band-pass, the same
defaults as the OpenBCI GUI: 60 Hz notch, 5-50 Hz... here 1-50 Hz), FFT / PSD, band
powers and the GUI-style "railed" signal-quality check. numpy only."""
from __future__ import annotations

import math

import numpy as np

BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
         "beta": (13.0, 30.0), "gamma": (30.0, 50.0)}
BAND_ORDER = list(BANDS)
# the bands the flow index is built on (docs/EEG_FLOW_INDEX.md): frontal-midline theta as
# used by Katahira 2018 / Ewing 2016, and upper alpha for the exploratory index
INDEX_BANDS = {"theta_fm": (4.0, 7.0), "alpha_hi": (10.0, 13.0)}
TOTAL_BAND = (1.0, 45.0)


def welch(window: np.ndarray, fs: float, seg_s: float = 2.0, overlap: float = 0.5,
          reject=None) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Welch PSD: Hann segments of `seg_s` with `overlap`, averaged. `reject(segment)` -> True
    drops a segment (artifact). Returns (freqs, PSD [m, n_ch], n_used, n_total)."""
    n = len(window)
    seg = int(fs * seg_s)
    step = max(1, int(seg * (1 - overlap)))
    if n < seg:
        f, P = psd(window, fs)
        return f, P, 1, 1
    acc = None
    used = total = 0
    for start in range(0, n - seg + 1, step):
        s = window[start:start + seg]
        total += 1
        if reject is not None and reject(s):
            continue
        f, P = psd(s, fs)
        acc = P if acc is None else acc + P
        used += 1
    if used == 0:
        f, P = psd(window[:seg], fs)
        return f, P * np.nan, 0, total
    return f, acc / used, used, total


def welch_masked(window: np.ndarray, fs: float, seg_s: float = 2.0, overlap: float = 0.5,
                 bad=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Welch PSD with PER-CHANNEL segment rejection: `bad(segment)` -> bool array [n_ch]
    (True = drop that channel's segment). Each channel's PSD is the mean over its own clean
    segments (NaN where it has none). Returns (freqs, PSD [m, n_ch], n_used [n_ch], n_total)."""
    n, n_ch = window.shape
    seg = int(fs * seg_s)
    step = max(1, int(seg * (1 - overlap)))
    if n < seg:
        f, P = psd(window, fs)
        return f, P, np.ones(n_ch, int), 1
    acc = None
    used = np.zeros(n_ch, int)
    total = 0
    f = None
    for start in range(0, n - seg + 1, step):
        s = window[start:start + seg]
        total += 1
        mask = np.zeros(n_ch, bool) if bad is None else np.asarray(bad(s), bool)
        if mask.all():
            continue
        f, P = psd(s, fs)
        if acc is None:
            acc = np.zeros_like(P)
        keep = ~mask
        acc[:, keep] += P[:, keep]
        used[keep] += 1
    if acc is None:
        f, P = psd(window[:seg], fs)
        return f, P * np.nan, used, total
    out = np.full_like(acc, np.nan)
    ok = used > 0
    out[:, ok] = acc[:, ok] / used[ok]
    return f, out, used, total


def band_powers_of(f: np.ndarray, P: np.ndarray, bands: dict) -> dict[str, np.ndarray]:
    out = {}
    if len(f) == 0:
        return {b: np.zeros(P.shape[1] if P.ndim == 2 else 0) for b in bands}
    df = f[1] - f[0] if len(f) > 1 else 1.0
    for name, (lo, hi) in bands.items():
        m = (f >= lo) & (f < hi)
        out[name] = P[m].sum(axis=0) * df
    return out


class Biquad:
    """Direct-form-II-transposed biquad applied per channel with state (RBJ cookbook)."""

    def __init__(self, b, a, n_ch: int):
        self.b = np.asarray(b, float) / a[0]
        self.a = np.asarray(a, float) / a[0]
        self.z1 = np.zeros(n_ch)
        self.z2 = np.zeros(n_ch)

    @classmethod
    def notch(cls, f0: float, fs: float, n_ch: int, q: float = 30.0):
        w = 2 * math.pi * f0 / fs
        alpha = math.sin(w) / (2 * q)
        c = math.cos(w)
        return cls([1, -2 * c, 1], [1 + alpha, -2 * c, 1 - alpha], n_ch)

    @classmethod
    def lowpass(cls, fc: float, fs: float, n_ch: int, q: float = 0.7071):
        w = 2 * math.pi * fc / fs
        alpha = math.sin(w) / (2 * q)
        c = math.cos(w)
        return cls([(1 - c) / 2, 1 - c, (1 - c) / 2], [1 + alpha, -2 * c, 1 - alpha], n_ch)

    @classmethod
    def highpass(cls, fc: float, fs: float, n_ch: int, q: float = 0.7071):
        w = 2 * math.pi * fc / fs
        alpha = math.sin(w) / (2 * q)
        c = math.cos(w)
        return cls([(1 + c) / 2, -(1 + c), (1 + c) / 2], [1 + alpha, -2 * c, 1 - alpha], n_ch)

    def process(self, x: np.ndarray) -> np.ndarray:
        """x [k, n_ch] -> y [k, n_ch]; keeps state between calls."""
        b0, b1, b2 = self.b
        _, a1, a2 = self.a
        y = np.empty_like(x, dtype=float)
        z1, z2 = self.z1, self.z2
        for i in range(len(x)):
            xi = x[i]
            yi = b0 * xi + z1
            z1 = b1 * xi - a1 * yi + z2
            z2 = b2 * xi - a2 * yi
            y[i] = yi
        self.z1, self.z2 = z1, z2
        return y


class FilterChain:
    """Notch (0 = off) + band-pass (two cascaded 2nd-order sections each side)."""

    def __init__(self, fs: float, n_ch: int, notch_hz: float = 60.0,
                 bp_lo: float = 1.0, bp_hi: float = 50.0):
        self.fs, self.n_ch = fs, n_ch
        self.notch_hz, self.bp_lo, self.bp_hi = notch_hz, bp_lo, bp_hi
        self.stages = []
        if notch_hz:
            self.stages.append(Biquad.notch(notch_hz, fs, n_ch))
            if 2 * notch_hz < fs / 2:
                self.stages.append(Biquad.notch(2 * notch_hz, fs, n_ch))
        if bp_lo:
            self.stages += [Biquad.highpass(bp_lo, fs, n_ch, 0.5412), Biquad.highpass(bp_lo, fs, n_ch, 1.3066)]
        if bp_hi and bp_hi < fs / 2:
            self.stages += [Biquad.lowpass(bp_hi, fs, n_ch, 0.5412), Biquad.lowpass(bp_hi, fs, n_ch, 1.3066)]

    def process(self, x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, float)
        for s in self.stages:
            y = s.process(y)
        return y


def psd(window: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed one-sided PSD (uV^2/Hz). window [n, n_ch] -> (freqs [m], P [m, n_ch])."""
    n = len(window)
    if n < 8:
        return np.zeros(0), np.zeros((0, window.shape[1]))
    x = window - window.mean(axis=0)
    w = np.hanning(n)[:, None]
    X = np.fft.rfft(x * w, axis=0)
    scale = 1.0 / (fs * (w ** 2).sum())
    P = (np.abs(X) ** 2) * scale
    if n % 2 == 0:
        P[1:-1] *= 2
    else:
        P[1:] *= 2
    f = np.fft.rfftfreq(n, 1.0 / fs)
    return f, P


def band_powers(f: np.ndarray, P: np.ndarray) -> dict[str, np.ndarray]:
    """Absolute power per band (uV^2) per channel, integrated from the PSD."""
    out = {}
    if len(f) == 0:
        return {b: np.zeros(P.shape[1] if P.ndim == 2 else 0) for b in BANDS}
    df = f[1] - f[0] if len(f) > 1 else 1.0
    for name, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        out[name] = P[m].sum(axis=0) * df
    return out


def relative_powers(abs_pow: dict[str, np.ndarray], total: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Band / total power. `total` defaults to the sum of the five display bands (1-50 Hz);
    pass the 1-45 Hz total for the index bands."""
    if total is None:
        total = sum(abs_pow[b] for b in BANDS if b in abs_pow)
    total = total + 1e-12
    return {b: abs_pow[b] / total for b in abs_pow}


def quality(raw_window: np.ndarray, full_scale_uv: float) -> list[dict]:
    """OpenBCI GUI-style per-channel signal check on RAW (unfiltered) data:
    railed = |x| above 90 % of full scale for most of the window, near-railed above 75 %,
    flat = no variation at all (open input / unplugged), else good."""
    out = []
    for c in range(raw_window.shape[1]):
        x = raw_window[:, c]
        if len(x) == 0:
            out.append({"state": "unknown", "pct": 0.0, "std": 0.0})
            continue
        amax = np.abs(x).max()
        pct = float(amax / full_scale_uv * 100.0)
        std = float(x.std())
        if pct >= 90:
            st = "railed"
        elif pct >= 75:
            st = "near"
        elif std < 0.05:
            st = "flat"
        else:
            st = "good"
        out.append({"state": st, "pct": round(pct, 1), "std": round(std, 2)})
    return out
