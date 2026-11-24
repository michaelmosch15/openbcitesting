import csv, datetime as dt, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as md
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gapfill import continuous, plot_continuous, measured, plot_measured, SIM_COLOR
# Copy of sessions/2026-11-19_summary/charts_docx.py with honest gaps. Default --gaps gaps: where a device gave no
# reading the line BREAKS. --gaps simulated: the old modelled fill, drawn grey and dashed and labelled
# "simulated" (never as a solid measured line). Writes to --out (default sessions/2026-11-19_summary/honest_gaps/),
# so the saved 11-19 PNG / PDF / DOCX outputs are never overwritten.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ap = argparse.ArgumentParser()
_ap.add_argument("--gaps", choices=["gaps", "simulated"], default="gaps")
_ap.add_argument("--root", default=ROOT, help="the folder holding sessions/ (default: this checkout)")
_ap.add_argument("--out", default=None, help="output folder (default <root>/sessions/2026-11-19_summary/honest_gaps)")
ARGS = _ap.parse_args()
OUT = ARGS.out or os.path.join(ARGS.root, "sessions", "2026-11-19_summary", "honest_gaps")
if os.path.abspath(OUT) == os.path.abspath(os.path.join(ARGS.root, "sessions", "2026-11-19_summary")):
    sys.exit("refusing to write over the saved 11-19 outputs; pick another --out")
os.makedirs(OUT, exist_ok=True)


def draw(ax, pts, t0, t1, kind, seed, color, lw, key):
    """One device line: measured only (gaps) or measured + grey dashed simulated fill."""
    if ARGS.gaps == "gaps":
        g, v = measured(pts, t0, t1, kind)
        SHARE.setdefault(key, []).append(round(float(np.isnan(v).mean()), 3))
        plot_measured(ax, g, v, color, lw, t0)
    else:
        g, v, real = continuous(pts, t0, t1, kind, seed=seed)
        SHARE.setdefault(key, []).append(round(float(1 - real.mean()), 3))
        plot_continuous(ax, g, v, real, color, lw, origin=t0)
SHARE = {}


def num(v):
    try:
        return float(v)
    except Exception:
        return None


D = lambda t: dt.datetime.fromtimestamp(t)


def load(folder, helm_col, rate_file, band_col, helm_rate_col):
    S_ = os.path.join(ARGS.root, "sessions", folder)
    ep = [(num(r["t_unix"]), num(r[helm_col])) for r in csv.DictReader(open(os.path.join(S_, "helmet_epochs.csv")))]
    bd = [(num(r["t_unix"]), num(r["band_flow"])) for r in csv.DictReader(open(os.path.join(S_, "band.csv")))]
    rt = [(num(r["t_unix"]), num(r["felt_flow"]), num(r[band_col]), num(r[helm_rate_col]))
          for r in csv.DictReader(open(os.path.join(S_, rate_file)))]
    return ep, bd, rt


lily = load("2026-11-19_lily_combined", "helmet_strict", "lily_ratings.csv", "band_flow_nearest", "helmet_strict_last30s")
mich = load("2026-11-19_michael_combined", "helmet_new_method", "michael_ratings.csv", "band_flow_within_15s", "helmet_new_last30s")
# one rating left out of the one-pager at the wearer's request (42 at 15:34); it stays in the raw recording
mich = (mich[0], mich[1], [r for r in mich[2] if not (abs(r[1] - 0.42) < 1e-6 and abs(r[0] - 1795120443.4) < 5)])
# two band readings left out of the BAND figures as outliers at the wearer's request (0.46 at 15:11:09, 0.48 at 15:50:08);
# the ratings, the EEG comparison and the plotted band line are unchanged
mich = (mich[0], mich[1], [(r[0], r[1], None, r[3]) if min(abs(r[0] - 1795119069.4), abs(r[0] - 1795121408.6)) < 5 else r for r in mich[2]])


def roll_median(pts, win):
    v = [(t, x) for t, x in pts if x is not None]
    out = []
    for t, x in pts:
        if x is None:
            out.append((t, None)); continue
        out.append((t, float(np.median([b for a, b in v if t - win <= a <= t]))))
    return out


def series(pts, gap):
    xs, ys, last = [], [], None
    for t, v in pts:
        if v is None or (last is not None and t - last > gap):
            xs.append(D(t - 0.001)); ys.append(np.nan)
        if v is not None:
            xs.append(D(t)); ys.append(v); last = t
    return xs, ys


def stats(rt, i):
    a = np.array([(r[1], r[i]) for r in rt if r[i] is not None])
    d = np.abs(a[:, 0] - a[:, 1])
    return {"n": len(a), "within": int((d <= 0.10).sum()), "gap": round(float(d.mean()), 3),
            "r": round(float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), 2)}


BAND, EEG = "#089981", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})


def chart(data, helm_label, smooth, locator, h, path):
    ep, bd, rt = data
    fig = plt.figure(figsize=(7.3, h), facecolor=SURF)
    YL = 1 + 0.16 / (h - 0.70)                      # legend row a fixed 0.16 in above the plot
    ax = fig.add_axes([0.075, 0.40 / h, 0.915, 1 - 0.40 / h - 0.30 / h])   # fixed inches for labels / legend
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.axhline(0.5, color="#c9c8c1", lw=0.7, ls=(0, (4, 4)))
    hh = roll_median(ep, smooth) if smooth else ep
    ts = [t for t, x in hh if x is not None] + [t for t, x in bd if x is not None] + [r[0] for r in rt]
    t0, t1 = min(ts), max(ts) + 10                 # minute 0 = first reading of the session
    X = lambda t: (t - t0) / 60.0
    draw(ax, hh, t0, t1, "smooth", 11 + len(rt), EEG, 1.5, "eeg")
    draw(ax, bd, t0, t1, "step", 23 + len(rt), BAND, 2.4, "band")
    ax.scatter([X(r[0]) for r in rt], [r[1] for r in rt], s=58, marker="D", color=INK, edgecolor=SURF, linewidth=1.5, zorder=5)
    prev_t, below = None, False
    for r in rt:
        below = (prev_t is not None and r[0] - prev_t < 80 and not below)
        prev_t = r[0]
        ax.annotate(f"{round(r[1] * 100)}", (X(r[0]), r[1]), textcoords="offset points", xytext=(0, -14 if below else 8),
                    ha="center", fontsize=7.6, fontweight="bold", color=INK)
    ax.set_ylim(0, 1); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.tick_params(labelsize=8, length=0)
    ax.set_ylabel("Flow (0-1)", fontsize=8.5)
    import matplotlib.ticker as mt
    ax.xaxis.set_major_locator(mt.MultipleLocator(locator)); ax.xaxis.set_major_formatter(mt.FormatStrFormatter("%d"))
    ax.set_xlim(0, X(t1)); ax.set_xlabel("Minutes from start of session", fontsize=8.5, labelpad=2)
    lx = 0.0
    for lab, kw, w in (("Band flow", dict(color=BAND, lw=2.4), 0.115), (helm_label, dict(color=EEG, lw=1.5), 0.225)):
        ax.plot([lx, lx + 0.035], [YL, YL], transform=ax.transAxes, clip_on=False, **kw)
        ax.text(lx + 0.043, YL, lab, transform=ax.transAxes, fontsize=8, va="center", color=INK)
        lx += 0.043 + w
    ax.scatter([lx + 0.012], [YL], transform=ax.transAxes, s=34, marker="D", color=INK, clip_on=False)
    ax.text(lx + 0.03, YL, "Felt flow (self-rating, value labelled)", transform=ax.transAxes, fontsize=8, va="center", color=INK)
    if ARGS.gaps == "simulated":
        ax.plot([0.0, 0.035], [YL - 0.09, YL - 0.09], transform=ax.transAxes, clip_on=False, color=SIM_COLOR, lw=1.2, ls=(0, (3, 2)))
        ax.text(0.043, YL - 0.09, "simulated (no reading; modelled, not measured)", transform=ax.transAxes, fontsize=7.4, va="center", color=INK2)
    fig.savefig(path, dpi=220, facecolor=SURF)


out = OUT + os.sep
chart(lily, "EEG flow (30-s median)", 30, 2, 3.0, out + "one_pager_chart_lily.png")
chart(mich, "EEG flow (60-s method)", 0, 5, 1.75, out + "one_pager_chart_michael.png")
S = {"lily": {"band": stats(lily[2], 2), "eeg": stats(lily[2], 3)}, "michael": {"band": stats(mich[2], 2), "eeg": stats(mich[2], 3)}}
S["gap_rendering"] = ARGS.gaps
S["no_reading_share" if ARGS.gaps == "gaps" else "modelled_share"] = {"lily": {"eeg": SHARE["eeg"][0], "band": SHARE["band"][0]}, "michael": {"eeg": SHARE["eeg"][1], "band": SHARE["band"][1]}}
json.dump(S, open(out + "one_pager_stats.json", "w"), indent=1)
print(json.dumps(S))
