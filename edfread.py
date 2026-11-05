"""Minimal EDF / EDF+ reader (header + int16 data records), numpy only.

Enough for the public PhysioNet EEG Motor Movement/Imagery recordings (64 channels, 160 Hz,
one data record per second) that autocal.py's placement template is built from. Signals come
back in physical units (uV) as a float32 array [n_samples, n_channels]; annotation channels
("EDF Annotations") are skipped. Labels are returned cleaned (`Fp1.` -> `Fp1`, `Fcz` -> `FCz`).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_CANON = {"fp1": "Fp1", "fpz": "Fpz", "fp2": "Fp2", "af7": "AF7", "af3": "AF3", "afz": "AFz", "af4": "AF4", "af8": "AF8",
          "f7": "F7", "f5": "F5", "f3": "F3", "f1": "F1", "fz": "Fz", "f2": "F2", "f4": "F4", "f6": "F6", "f8": "F8",
          "ft7": "FT7", "fc5": "FC5", "fc3": "FC3", "fc1": "FC1", "fcz": "FCz", "fc2": "FC2", "fc4": "FC4", "fc6": "FC6", "ft8": "FT8",
          "t7": "T7", "c5": "C5", "c3": "C3", "c1": "C1", "cz": "Cz", "c2": "C2", "c4": "C4", "c6": "C6", "t8": "T8",
          "tp7": "TP7", "cp5": "CP5", "cp3": "CP3", "cp1": "CP1", "cpz": "CPz", "cp2": "CP2", "cp4": "CP4", "cp6": "CP6", "tp8": "TP8",
          "p7": "P7", "p5": "P5", "p3": "P3", "p1": "P1", "pz": "Pz", "p2": "P2", "p4": "P4", "p6": "P6", "p8": "P8",
          "po7": "PO7", "po3": "PO3", "poz": "POz", "po4": "PO4", "po8": "PO8", "o1": "O1", "oz": "Oz", "o2": "O2", "iz": "Iz",
          "t9": "T9", "t10": "T10", "ft9": "FT9", "ft10": "FT10", "tp9": "TP9", "tp10": "TP10"}


def clean_label(raw: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]", "", raw.strip())
    return _CANON.get(s.lower(), s)


def read_edf(path: str | Path):
    """-> (signals [n, n_ch] float32 uV, labels, fs_hz)."""
    path = Path(path)
    with open(path, "rb") as f:
        hdr = f.read(256)
        if len(hdr) < 256:
            raise ValueError(f"{path}: not an EDF file (short header)")
        n_hdr = int(hdr[184:192])
        n_rec = int(hdr[236:244])
        rec_dur = float(hdr[244:252])
        n_sig = int(hdr[252:256])
        sh = f.read(256 * n_sig)
        if len(sh) < 256 * n_sig:
            raise ValueError(f"{path}: truncated signal header")

        def field(off: int, width: int) -> list[str]:
            base = off * n_sig
            return [sh[base + i * width: base + (i + 1) * width].decode("ascii", "replace").strip() for i in range(n_sig)]
        labels = field(0, 16)
        phys_min = [float(x) for x in field(16 + 80 + 8, 8)]
        phys_max = [float(x) for x in field(16 + 80 + 8 + 8, 8)]
        dig_min = [int(float(x)) for x in field(16 + 80 + 8 + 8 + 8, 8)]
        dig_max = [int(float(x)) for x in field(16 + 80 + 8 + 8 + 8 + 8, 8)]
        n_samp = [int(x) for x in field(16 + 80 + 8 + 8 + 8 + 8 + 8 + 80, 8)]
        f.seek(n_hdr)
        data = np.fromfile(f, dtype="<i2")
    rec_len = sum(n_samp)
    if n_rec < 0:
        n_rec = len(data) // rec_len
    data = data[: n_rec * rec_len].reshape(n_rec, rec_len)
    keep = [i for i, lab in enumerate(labels) if "annotation" not in lab.lower()]
    if not keep:
        raise ValueError(f"{path}: no signal channels")
    ns = n_samp[keep[0]]
    if any(n_samp[i] != ns for i in keep):
        raise ValueError(f"{path}: mixed sampling rates are not supported")
    fs = ns / rec_dur
    out = np.empty((n_rec * ns, len(keep)), np.float32)
    off = 0
    offsets = {}
    for i in range(n_sig):
        offsets[i] = off
        off += n_samp[i]
    for c, i in enumerate(keep):
        raw = data[:, offsets[i]: offsets[i] + ns].reshape(-1).astype(np.float32)
        gain = (phys_max[i] - phys_min[i]) / float(dig_max[i] - dig_min[i] or 1)
        out[:, c] = (raw - dig_min[i]) * gain + phys_min[i]
    return out, [clean_label(labels[i]) for i in keep], fs
