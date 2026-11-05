"""Build autocal_template.json from the public PhysioNet EEG Motor Movement/Imagery Dataset
(Schalk et al. 2004, Goldberger et al. 2000; 64-channel 10-10 EEG, 160 Hz; CC-BY-like ODC-BY
licence). Baseline runs only: R01 eyes open, R02 eyes closed.

    python tools/build_autocal_template.py <folder with SxxxR01.edf / SxxxR02.edf> [--out autocal_template.json]

What goes into the template, for every frontal 10-10 site (Fp1 Fpz Fp2 AF7 AF3 AFz AF4 AF8
F7 F5 F3 F1 Fz F2 F4 F6 F8), all re-referenced to the mean of T9 and T10 (the closest thing
in the dataset to the Cyton's earlobe reference):

  corr    the inter-site correlation matrix of 1-30 Hz EEG (Fisher-z averaged over subjects
          and both runs) - neighbours correlate more than distant sites
  blink   the amplitude of a blink at each site relative to the Fp1/Fp2 mean (median over
          blinks detected on eyes-open runs) - largest next to the eyes, falling with distance
  heog    the signed deflection at each site during a horizontal eye movement relative to the
          F7-F8 difference (median over detected saccades) - positive left, negative right,
          growing toward the temples
  geometry 2-D 10-10 coordinates on the unit disc (x right, y forward) for the distance prior

autocal.py matches a live 8-channel forehead recording against these vectors to recover which
channel sits where. The template carries its provenance (files, subjects, n blinks / saccades).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from autocal import FRONTAL_SITES, SITE_XY, bandpass, detect_events  # noqa: E402
from edfread import read_edf  # noqa: E402


def _z(r: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(r, -0.999, 0.999))


def process_file(path: Path, eyes_open: bool):
    x, labels, fs = read_edf(path)
    idx = {lab: i for i, lab in enumerate(labels)}
    missing = [s for s in FRONTAL_SITES if s not in idx]
    if missing or "T9" not in idx or "T10" not in idx:
        raise ValueError(f"{path.name}: missing {missing or ['T9/T10']}")
    ref = (x[:, idx["T9"]] + x[:, idx["T10"]]) / 2.0
    fr = np.stack([x[:, idx[s]] - ref for s in FRONTAL_SITES], axis=1)
    fr = fr - fr.mean(axis=0)
    eeg = bandpass(fr, fs, 1.0, 30.0)
    corr = np.corrcoef(eeg.T)
    out = {"corr_z": _z(corr), "n": len(eeg)}
    if eyes_open:
        slow = bandpass(fr, fs, 0.5, 8.0)
        fp = (slow[:, FRONTAL_SITES.index("Fp1")] + slow[:, FRONTAL_SITES.index("Fp2")]) / 2.0
        blinks = detect_events(fp, fs, min_gap_s=0.5, k_mad=4.0, positive_only=True)
        if len(blinks) >= 3:
            amp = np.stack([slow[t, :] for t in blinks])            # [n_blink, n_site]
            ref_amp = amp[:, [FRONTAL_SITES.index("Fp1"), FRONTAL_SITES.index("Fp2")]].mean(axis=1, keepdims=True)
            ok = ref_amp[:, 0] > 1e-6
            out["blink"] = np.median(amp[ok] / ref_amp[ok], axis=0)
            out["n_blink"] = int(ok.sum())
        heog = slow[:, FRONTAL_SITES.index("F7")] - slow[:, FRONTAL_SITES.index("F8")]
        sacc = detect_events(heog, fs, min_gap_s=0.4, k_mad=4.0, positive_only=False)
        if len(sacc) >= 3:
            d = np.stack([slow[t, :] for t in sacc])
            ref_d = heog[sacc][:, None]
            ok = np.abs(ref_d[:, 0]) > 1e-6
            out["heog"] = np.median(d[ok] / ref_d[ok], axis=0)
            out["n_sacc"] = int(ok.sum())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--out", default=str(HERE / "autocal_template.json"))
    args = ap.parse_args(argv)
    files = sorted(Path(args.folder).glob("S*R0[12].edf"))
    files = [f for f in files if f.stat().st_size > 1000]
    if not files:
        print("no EDF files found", file=sys.stderr)
        return 2
    zs, blinks, heogs, used, n_blink, n_sacc = [], [], [], [], 0, 0
    for f in files:
        try:
            r = process_file(f, eyes_open=f.name.endswith("R01.edf"))
        except ValueError as e:
            print(f"skip {e}", file=sys.stderr)
            continue
        zs.append(r["corr_z"])
        if "blink" in r:
            blinks.append(r["blink"]); n_blink += r["n_blink"]
        if "heog" in r:
            heogs.append(r["heog"]); n_sacc += r["n_sacc"]
        used.append(f.name)
        print(f"{f.name}: {r['n']} samples" + (f", {r.get('n_blink', 0)} blinks, {r.get('n_sacc', 0)} saccades" if f.name.endswith("R01.edf") else ""))
    corr = np.tanh(np.mean(zs, axis=0))
    np.fill_diagonal(corr, 1.0)
    tpl = {
        "version": 1, "built": datetime.now().isoformat(timespec="seconds"),
        "source": "PhysioNet EEG Motor Movement/Imagery Dataset (eegmmidb 1.0.0), baseline runs R01 (eyes open) + R02 (eyes closed), "
                  "64-ch 10-10 at 160 Hz, re-referenced to mean(T9, T10)",
        "files": used, "n_subjects": len({f[:4] for f in used}), "n_blinks": n_blink, "n_saccades": n_sacc,
        "sites": list(FRONTAL_SITES),
        "geometry": {s: list(SITE_XY[s]) for s in FRONTAL_SITES},
        "corr": np.round(corr, 4).tolist(),
        "blink": np.round(np.median(blinks, axis=0), 4).tolist() if blinks else None,
        "heog": np.round(np.median(heogs, axis=0), 4).tolist() if heogs else None,
    }
    Path(args.out).write_text(json.dumps(tpl, indent=1), "utf-8")
    print(f"wrote {args.out}: {len(used)} files, {tpl['n_subjects']} subjects, {n_blink} blinks, {n_sacc} saccades")
    if tpl["blink"]:
        print("blink ratio:", {s: v for s, v in zip(FRONTAL_SITES, tpl["blink"])})
    if tpl["heog"]:
        print("heog ratio :", {s: v for s, v in zip(FRONTAL_SITES, tpl["heog"])})
    return 0


if __name__ == "__main__":
    sys.exit(main())
