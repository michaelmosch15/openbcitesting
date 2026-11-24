import csv, datetime as dt, textwrap
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as md
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gapfill import continuous, plot_continuous, measured, plot_measured, SIM_COLOR
# Copy of sessions/2026-11-19_summary/make_one_pager.py with honest gaps. Default --gaps gaps: where a device gave no
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
from matplotlib.patches import Rectangle, FancyBboxPatch


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
# one rating left out of the one-pager at the wearer's request (a 42, about 30 minutes in); it stays in the raw recording
mich = (mich[0], mich[1], [r for r in mich[2] if not (abs(r[1] - 0.42) < 1e-6 and abs(r[0] - 1795120443.4) < 5)])
# two band readings left out of the BAND figures as outliers at the wearer's request (0.46 at 15:11:09, 0.48 at 15:50:08);
# the ratings, the EEG comparison and the plotted band line are unchanged
mich = (mich[0], mich[1], [(r[0], r[1], None, r[3]) if min(abs(r[0] - 1795119069.4), abs(r[0] - 1795121408.6)) < 5 else r for r in mich[2]])


def roll_median(pts, win):
    v = [(t, x) for t, x in pts if x is not None]
    out = []
    for t, x in pts:
        if x is None:
            out.append((t, None))
            continue
        w = [b for a, b in v if t - win <= a <= t]
        out.append((t, float(np.median(w))))
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
    return len(a), int((d <= 0.10).sum()), float(d.mean()), float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])


SL = {"band": stats(lily[2], 2), "eeg": stats(lily[2], 3)}
SM = {"band": stats(mich[2], 2), "eeg": stats(mich[2], 3)}
nb = SL["band"][0] + SM["band"][0]; wb = SL["band"][1] + SM["band"][1]
ne = SL["eeg"][0] + SM["eeg"][0]; we = SL["eeg"][1] + SM["eeg"][1]
gb = (SL["band"][2] * SL["band"][0] + SM["band"][2] * SM["band"][0]) / nb
ge = (SL["eeg"][2] * SL["eeg"][0] + SM["eeg"][2] * SM["eeg"][0]) / ne
print(SL, SM, nb, wb, ne, we, round(gb, 3), round(ge, 3))

TEAL, MINT, BAND, EEG = "#0A5853", "#D3E9E2", "#089981", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig = plt.figure(figsize=(8.5, 11), facecolor=SURF)
L, R = 0.07, 0.93

# header
fig.add_artist(Rectangle((0, 0.915), 1, 0.085, transform=fig.transFigure, color=TEAL, lw=0))
fig.text(L, 0.962, "The BioProxal band tracked felt flow more closely than EEG", color="white", fontsize=16.5,
         fontweight="bold", va="center")
fig.text(L, 0.933, "Head-to-head session, 19 November 2026  ·  wrist band vs 8-channel EEG headset vs spoken self-ratings",
         color=MINT, fontsize=9.2, va="center")

# lede
fig.text(L, 0.895, "Two people wore a BioProxal band and an OpenBCI EEG headset at the same time and called out how \"in flow\" they felt\n"
                   "(0-100). In both sessions the band's flow score stayed close to those ratings. The EEG flow score did not.",
         fontsize=9.3, color=INK, va="top", linespacing=1.5)

# stat tiles
tiles = [(f"{wb} of {nb}", "band readings within 0.10\nof the wearer's rating", BAND),
         (f"{we} of {ne}", "EEG readings within 0.10\nof the wearer's rating", EEG),
         (f"{gb:.2f} vs {ge:.2f}", "average gap to felt flow,\nband vs EEG (lower is better)", TEAL)]
tw = (R - L - 0.04) / 3
for k, (big, small, col) in enumerate(tiles):
    x0 = L + k * (tw + 0.02)
    fig.add_artist(FancyBboxPatch((x0, 0.778), tw, 0.064, boxstyle="round,pad=0,rounding_size=0.006",
                                  transform=fig.transFigure, fc="#f4f8f6", ec=MINT, lw=1))
    fig.add_artist(Rectangle((x0, 0.778), 0.006, 0.064, transform=fig.transFigure, color=col, lw=0))
    fig.text(x0 + 0.018, 0.822, big, fontsize=16, fontweight="bold", color=INK, va="center")
    fig.text(x0 + 0.018, 0.794, small, fontsize=7.2, color=INK2, va="center", linespacing=1.3)


def chart(ax, data, who, helm_label, smooth, locator, sub):
    ep, bd, rt = data
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.axhline(0.5, color="#c9c8c1", lw=0.7, ls=(0, (4, 4)))
    h = roll_median(ep, smooth) if smooth else ep
    ts = [t for t, x in h if x is not None] + [t for t, x in bd if x is not None] + [r[0] for r in rt]
    t0, t1 = min(ts), max(ts) + 10                 # minute 0 = first reading of the session
    X = lambda t: (t - t0) / 60.0
    draw(ax, h, t0, t1, "smooth", 11 + len(rt), EEG, 1.5, "eeg")
    draw(ax, bd, t0, t1, "step", 23 + len(rt), BAND, 2.4, "band")
    ax.scatter([X(r[0]) for r in rt], [r[1] for r in rt], s=62, marker="D", color=INK, edgecolor=SURF, linewidth=1.5, zorder=5)
    prev_t, below = None, False
    for r in rt:
        below = (prev_t is not None and r[0] - prev_t < 80 and not below)   # alternate when ratings are close together
        prev_t = r[0]
        ax.annotate(f"{round(r[1] * 100)}", (X(r[0]), r[1]), textcoords="offset points", xytext=(0, -14 if below else 8),
                    ha="center", fontsize=7.6, fontweight="bold", color=INK)
    ax.set_ylim(0, 1); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.tick_params(labelsize=8, length=0)
    ax.set_ylabel("Flow (0-1)", fontsize=8.5)
    import matplotlib.ticker as mt
    ax.xaxis.set_major_locator(mt.MultipleLocator(locator)); ax.xaxis.set_major_formatter(mt.FormatStrFormatter("%d"))
    ax.set_xlim(0, X(t1)); ax.set_xlabel("Minutes from start of session", fontsize=8.5, labelpad=2)
    ax.text(0, 1.20, who, transform=ax.transAxes, fontsize=11.5, fontweight="bold", color=TEAL)
    ax.text(0, 1.115, sub, transform=ax.transAxes, fontsize=8.2, color=INK2)
    # legend row: coloured mark beside ink-coloured label
    lx = 0.0
    for lab, kw, w in (("Band flow", dict(color=BAND, lw=2.4), 0.115), (helm_label, dict(color=EEG, lw=1.5), 0.225)):
        ax.plot([lx, lx + 0.035], [1.04, 1.04], transform=ax.transAxes, clip_on=False, **kw)
        ax.text(lx + 0.043, 1.04, lab, transform=ax.transAxes, fontsize=8, va="center", color=INK)
        lx += 0.043 + w
    ax.scatter([lx + 0.012], [1.04], transform=ax.transAxes, s=36, marker="D", color=INK, clip_on=False)
    ax.text(lx + 0.03, 1.04, "Felt flow (self-rating, value labelled)", transform=ax.transAxes, fontsize=8, va="center", color=INK)
    if ARGS.gaps == "simulated":
        ax.plot([0.0, 0.035], [1.04 - 0.09, 1.04 - 0.09], transform=ax.transAxes, clip_on=False, color=SIM_COLOR, lw=1.2, ls=(0, (3, 2)))
        ax.text(0.043, 1.04 - 0.09, "simulated (no reading; modelled, not measured)", transform=ax.transAxes, fontsize=7.4, va="center", color=INK2)


ax1 = fig.add_axes([L + 0.04, 0.540, R - L - 0.04, 0.174])
chart(ax1, lily, "Lily  ·  7 ratings over 14 minutes", "EEG flow (30-s median)", 30, 2,
      f"Band: within 0.10 on {SL['band'][1]} of {SL['band'][0]}, average gap {SL['band'][2]:.2f}, r = {SL['band'][3]:.2f}."
      f"     EEG: {SL['eeg'][1]} of {SL['eeg'][0]}, average gap {SL['eeg'][2]:.2f}, r = {SL['eeg'][3]:.2f}.")
ax2 = fig.add_axes([L + 0.04, 0.310, R - L - 0.04, 0.147])
chart(ax2, mich, "Michael  ·  15 ratings over 48 minutes (supporting session)", "EEG flow (60-s method)", 0, 5,
      f"Band: within 0.10 on {SM['band'][1]} of {SM['band'][0]}, average gap {SM['band'][2]:.2f}, r = {SM['band'][3]:.2f}."
      f"     EEG: {SM['eeg'][1]} of {SM['eeg'][0]}, average gap {SM['eeg'][2]:.2f}, r = {SM['eeg'][3]:.2f}.")

# takeaways
fig.add_artist(Rectangle((L, 0.254), R - L, 0.0012, transform=fig.transFigure, color=MINT, lw=0))
fig.text(L, 0.243, "What the graphs show", fontsize=10, fontweight="bold", color=TEAL, va="top")
pts = [f"The band line stays near the black rating diamonds in both sessions: {wb} of {nb} readings within 0.10, and it moves "
       f"with the ratings (r = {SL['band'][3]:.2f} and {SM['band'][3]:.2f}).",
       f"The EEG line swings more widely between ratings and lands near them less often ({we} of {ne}); its link to "
       f"felt flow is weaker (r = {SL['eeg'][3]:.2f} and {SM['eeg'][3]:.2f}).",
       "Michael's session repeats Lily's result on a second person, over a wider range of ratings (44 to 72) and with all "
       "8 EEG electrodes in contact."]
y = 0.222
for p in pts:
    lines = textwrap.wrap(p, 120)
    fig.text(L, y, "•", fontsize=9, color=BAND, va="top", fontweight="bold")
    fig.text(L + 0.016, y, "\n".join(lines), fontsize=8.4, va="top", linespacing=1.45)
    y -= 0.0152 * len(lines) + 0.005
note = ("Read with care: felt flow (a spoken self-rating) is the reference here, not a clinical measure. Small sample "
        f"({nb} band and {ne} EEG comparisons, two wearers), free sitting with no task protocol. The EEG was a dry-electrode "
        "forehead headset with no resting baseline, scored on frontal theta; a different EEG setup or a structured task could "
        "perform differently. Ratings with no device reading at that moment are left out. Exploratory result, not a validation study."
        + ((" Where a device gave no reading the line is broken (Lily: no reading for "
            f"{SHARE['eeg'][0]:.0%} of the EEG line and {SHARE['band'][0]:.0%} of the band line; Michael: {SHARE['eeg'][1]:.0%} and "
            f"{SHARE['band'][1]:.0%}).") if ARGS.gaps == "gaps" else
           (" Grey dashed stretches are SIMULATED: no reading there, filled with modelled values that follow that wearer's own level "
            f"and variation (Lily: {SHARE['eeg'][0]:.0%} of the EEG line and {SHARE['band'][0]:.0%} of the band line; Michael: "
            f"{SHARE['eeg'][1]:.0%} and {SHARE['band'][1]:.0%}).")) + " All figures use measured readings only. One of Michael's ratings (a 42, about 30 minutes in) is left out at his request. Two of Michael's band readings (46 and 48, each about 17 points below his rating) are left out of the band figures as outliers at his request; with them the band is within 0.10 on 18 of 21 readings and its average gap is 0.05.")
fig.text(L, 0.114, "\n".join(textwrap.wrap(note, 144)), fontsize=6.7, color=INK2, va="top", linespacing=1.45)
fig.text(R, 0.010, "BioProxal  ·  internal", fontsize=7, color=INK2, ha="right")
out = os.path.join(OUT, f"BioProxal_band_vs_EEG_one_pager_2026-11-19_{ARGS.gaps}")
fig.savefig(out + ".pdf", facecolor=SURF)
fig.savefig(out + ".png", dpi=200, facecolor=SURF)
print("saved", out)
