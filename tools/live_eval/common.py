"""Shared helpers for the offline evaluation of the live helmet flow (`flow_live`).

Loads a session's eeg_raw.csv (raw uV) and applies the SAME fixed analysis filter as
pipeline._epoch (mains notch + 1-45 Hz). Read-only on sessions/.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(HERE))
from dsp import FilterChain, psd  # noqa: E402

FS = 250.0
RAIL_UV = 187500.0


def newest_sessions(prefix: str = "2026-11-19", min_bytes: int = 2_000_000) -> list[Path]:
    out = [d for d in sorted((HERE / "sessions").glob(prefix + "*"))
           if (d / "eeg_raw.csv").exists() and (d / "eeg_raw.csv").stat().st_size >= min_bytes]
    return out


def load_raw(session: Path):
    """-> t [n], x [n, n_ch] raw uV, labels. Tolerates a half-written last line (live recording)."""
    with open(session / "eeg_raw.csv", newline="", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        cols = [i for i, h in enumerate(header) if h.startswith("ch")]
        labels = [header[i].split("_", 1)[1] for i in cols]
        rows = []
        for line in f:
            p = line.split(",")
            if len(p) < len(header):
                continue
            try:
                rows.append([float(p[0])] + [float(p[i]) for i in cols])
            except ValueError:
                continue
    a = np.asarray(rows, float)
    return a[:, 0], a[:, 1:], labels


def analysis_filter(x: np.ndarray, fs: float = FS, lo: float = 1.0, hi: float = 45.0) -> np.ndarray:
    chain = FilterChain(fs, x.shape[1], 60.0, lo, hi)
    n = int(fs)
    pad = np.concatenate([x[::-1][:n], x], axis=0)
    return chain.process(pad)[n:]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def segments(x: np.ndarray, fs: float = FS, seg_s: float = 2.0, step_s: float = 1.0):
    n, st = int(fs * seg_s), int(fs * step_s)
    return [(s, x[s:s + n]) for s in range(0, len(x) - n + 1, st)]


def seg_psd(x: np.ndarray, fs: float = FS, seg_s: float = 2.0, step_s: float = 1.0):
    """-> freqs [m], P [k, m, n_ch], starts [k] for every Hann segment (no rejection)."""
    Ps, st = [], []
    f = None
    for s, seg in segments(x, fs, seg_s, step_s):
        f, P = psd(seg, fs)
        Ps.append(P); st.append(s)
    return f, np.asarray(Ps), np.asarray(st)
