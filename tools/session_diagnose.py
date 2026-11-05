"""Diagnose a recorded session's EEG and band data offline.

    python tools/session_diagnose.py sessions/<folder> [--reference srb2|car] [--seconds 120]

Reads eeg_raw.csv (unfiltered uV) and band.csv and reports, per channel: DC offset, std of
the 1-45 Hz signal, fraction of 2-s segments passing the artifact rule (150 uV p2p / 100 uV
abs), the correlation with the other channels (a block of 0.95+ means one shared signal:
the reference / bias clip, not eight brains), relative theta / alpha / beta, and the same
after a common-average re-reference (CAR). Then the band columns: how many rows carry a
flow, HR, PI, HRV. Prints a verdict on what blocked the index. Numpy only, no network.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from dsp import FilterChain, band_powers, psd, relative_powers  # noqa: E402


def load_eeg(path: Path, max_s: float | None):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        ch_cols = [i for i, h in enumerate(header) if h.startswith("ch")]
        labels = [header[i].split("_", 1)[1] if "_" in header[i] else header[i] for i in ch_cols]
        rows = []
        for row in r:
            try:
                rows.append([float(row[0])] + [float(row[i]) for i in ch_cols])
            except (ValueError, IndexError):
                continue
    a = np.array(rows, float)
    t, x = a[:, 0], a[:, 1:]
    if max_s:
        keep = t <= t[0] + max_s
        t, x = t[keep], x[keep]
    fs = 1.0 / np.median(np.diff(t)) if len(t) > 10 else 250.0
    return t, x, labels, round(fs)


def segment_pass(x: np.ndarray, fs: float, seg_s: float = 2.0, p2p=150.0, absv=100.0) -> np.ndarray:
    n = int(fs * seg_s)
    k = len(x) // n
    out = np.zeros((k, x.shape[1]), bool)
    for i in range(k):
        s = x[i * n:(i + 1) * n]
        out[i] = (s.max(axis=0) - s.min(axis=0) <= p2p) & (np.abs(s).max(axis=0) <= absv)
    return out


def analyse(x: np.ndarray, fs: float, labels, title: str):
    chain = FilterChain(fs, x.shape[1], 60.0, 1.0, 45.0)
    pad = np.concatenate([x[::-1][: int(fs)], x], axis=0)
    f = chain.process(pad)[int(fs):]
    std = f.std(axis=0)
    ok = segment_pass(f, fs)
    corr = np.corrcoef(f.T)
    np.fill_diagonal(corr, np.nan)
    print(f"\n== {title} ==")
    print(f"{'ch':>3} {'site':>5} {'dc_mV':>7} {'std_uV':>8} {'clean%':>7} {'corr_oth':>8} {'theta':>6} {'alpha':>6} {'beta':>6}")
    fr, P = psd(f, fs)
    keep = fr <= 60
    ab = band_powers(fr[keep], P[keep])
    rel = relative_powers(ab)
    for i, lab in enumerate(labels):
        print(f"{i + 1:>3} {lab:>5} {x[:, i].mean() / 1000:7.1f} {std[i]:8.1f} {100 * ok[:, i].mean():6.0f}% "
              f"{np.nanmean(corr[i]):8.2f} {rel['theta'][i]:6.2f} {rel['alpha'][i]:6.2f} {rel['beta'][i]:6.2f}")
    all_clean = ok.all(axis=1).mean()
    any2 = (ok.sum(axis=1) >= 2).mean()
    print(f"segments clean on ALL channels: {100 * all_clean:.0f}%   on >= 2 channels: {100 * any2:.0f}%   mean |corr| {np.nanmean(np.abs(corr)):.2f}")
    return ok, corr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--seconds", type=float, default=None, help="analyse only the first N seconds")
    ap.add_argument("--drop", default="", help="comma-separated 1-based channels to leave out (railed)")
    args = ap.parse_args(argv)
    d = Path(args.session)
    t, x, labels, fs = load_eeg(d / "eeg_raw.csv", args.seconds)
    print(f"{d.name}: {len(x)} samples, {len(x) / fs / 60:.1f} min at {fs} Hz, channels {labels}")
    drop = {int(s) - 1 for s in args.drop.split(",") if s.strip()}
    railed = {i for i in range(x.shape[1]) if np.abs(x[:, i]).max() > 180000 and x[:, i].std() < 1}
    if railed:
        print(f"railed the whole time (open input): {[labels[i] for i in sorted(railed)]}")
    use = [i for i in range(x.shape[1]) if i not in drop and i not in railed]
    xs, ls = x[:, use], [labels[i] for i in use]
    ok1, corr1 = analyse(xs, fs, ls, "as recorded (SRB2 ear reference)")
    car = xs - xs.mean(axis=1, keepdims=True)
    ok2, corr2 = analyse(car, fs, ls, "common-average re-reference (CAR) over the same channels")
    # band
    bp = d / "band.csv"
    if bp.exists():
        with open(bp, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        n = len(rows)
        if n:
            flows = [r["flow"] for r in rows]
            n_flow = sum(1 for v in flows if v not in ("", None))
            n_nonzero = sum(1 for v in flows if v not in ("", None) and float(v) > 0)
            pis = [float(r["pi"]) for r in rows if r.get("pi") not in ("", None)]
            hrs = [float(r["hr"]) for r in rows if r.get("hr") not in ("", None)]
            print(f"\n== band.csv: {n} rows ==")
            print(f"flow present {n_flow} ({100 * n_flow / n:.0f}%), flow > 0: {n_nonzero}; HR rows {len(hrs)} "
                  f"(mean {np.mean(hrs):.0f} bpm)" if hrs else f"flow present {n_flow}; no HR")
            if pis:
                print(f"PI mean {np.mean(pis):.2f}% (0 = no pulse seen), rows with PI > 0.3: {sum(1 for p in pis if p > 0.3)}")
            print("example rows:", rows[0], rows[-1], sep="\n  ")
    print("\n== verdict ==")
    if np.nanmean(np.abs(corr1)) > 0.9:
        print("- the recorded channels are one shared signal (|corr| > 0.9): the SRB2 reference / BIAS clip was not "
              "making contact; what the index saw was reference noise, so every epoch failed the artifact rule")
    if ok2.all(axis=1).mean() > ok1.all(axis=1).mean() + 0.1:
        print(f"- CAR recovers usable segments: {100 * ok1.all(axis=1).mean():.0f}% -> {100 * ok2.all(axis=1).mean():.0f}% clean "
              "(the differential signal between combs is brain + skin, the shared part cancels)")
    else:
        print("- CAR does not rescue it: the electrodes themselves were not making stable contact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
