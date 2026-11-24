"""Flow lines for the 11-19 one-pager charts.

    t, v = measured(points, t0, t1, kind="smooth" | "step")      # DEFAULT in the chart scripts: gaps stay gaps
    t, v, real = continuous(points, t0, t1, kind=..., seed=...)   # --gaps simulated only: modelled fill

measured(): only real readings. "smooth" (EEG, a value every 2 s) joins readings at most MAX_GAP_SMOOTH_S
apart; "step" (band, about one value a minute, held by the band) holds each value for at most
MAX_HOLD_STEP_S. Anything longer is a gap (NaN), drawn as a break in the line.

continuous(): keep every real reading, fill the gaps with modelled values that have the wearer's own level,
spread and smoothness. Filled stretches are returned flagged; plot_continuous draws them grey and dashed as
"simulated", never as a solid measured line.

    t, v, real = continuous(points, t0, t1, kind="smooth" | "step", seed=...)

points: [(t_unix, value or None), ...] (rows may be missing entirely across app restarts)
kind "smooth": EEG-style line. Gaps = straight line between the neighbouring real values plus AR(1) noise with
               the series' own variance and lag-1 autocorrelation, pinned to both ends (a bridge).
kind "step":   band-style line (the band updates about once a minute and holds its value). Gaps = a pinned
               random walk on 60-s knots with the wearer's own step size, held between knots.
Before the first / after the last real value: the same noise, drifting from the nearest real value toward the
wearer's mean. Everything is clipped to 0.02-0.98. Deterministic for a given seed.
"""
import datetime as dt
import numpy as np

STEP = 2.0
MAX_GAP_SMOOTH_S = 6.0
MAX_HOLD_STEP_S = 90.0
SIM_COLOR = "#9a9a94"


def measured(points, t0, t1, kind="smooth"):
    """Real readings on the 2-s grid, NaN wherever the device gave none (see module doc)."""
    grid = np.arange(t0, t1 + STEP / 2, STEP)
    v = np.full(len(grid), np.nan)
    pts = sorted((t, x) for t, x in points if t is not None and x is not None)
    if kind == "step":
        for (t, x), nxt in zip(pts, pts[1:] + [(None, None)]):
            end = min(t + MAX_HOLD_STEP_S, nxt[0] if nxt[0] is not None else t + MAX_HOLD_STEP_S)
            m = (grid >= t - STEP / 2) & (grid < end)
            v[m] = x
        return grid, v
    for t, x in pts:
        k = int(round((t - t0) / STEP))
        if 0 <= k < len(grid):
            v[k] = x
    # bridge only short holes (a missing epoch or two), never a real outage
    idx = np.flatnonzero(~np.isnan(v))
    for i0, i1 in zip(idx[:-1], idx[1:]):
        if 1 < i1 - i0 <= int(MAX_GAP_SMOOTH_S / STEP):
            v[i0:i1 + 1] = np.linspace(v[i0], v[i1], i1 - i0 + 1)
    return grid, v


def plot_measured(ax, grid, v, color, lw, origin):
    X = [(t - origin) / 60.0 for t in grid]
    ax.plot(X, v, color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=3)


def continuous(points, t0, t1, kind="smooth", seed=0):
    rng = np.random.default_rng(seed)
    grid = np.arange(t0, t1 + STEP / 2, STEP)
    v = np.full(len(grid), np.nan)
    for t, x in points:
        if x is None or t is None:
            continue
        k = int(round((t - t0) / STEP))
        if 0 <= k < len(grid):
            v[k] = x
    real = ~np.isnan(v)
    if real.sum() < 3:
        return grid, v, real
    xs = v[real]
    mu, sd = float(xs.mean()), float(xs.std())
    pair = real[:-1] & real[1:]
    a, b = v[:-1][pair], v[1:][pair]
    if kind == "smooth":
        phi = float(np.corrcoef(a, b)[0, 1]) if len(a) > 10 and a.std() > 0 and b.std() > 0 else 0.95
        phi = min(max(phi, 0.80), 0.995)
        se = sd * np.sqrt(1 - phi * phi)
    else:
        d = (b - a)[np.abs(b - a) > 1e-6]
        step_sd = float(d.std()) if len(d) > 3 else 0.04
    idx = np.flatnonzero(real)

    def ar(n, start=0.0):
        out = np.empty(n); x = start
        for i in range(n):
            x = phi * x + rng.normal(0, se); out[i] = x
        return out

    def knots_walk(n_steps):
        return np.cumsum(rng.normal(0, step_sd, n_steps)) if n_steps > 0 else np.array([])

    # interior gaps
    for i0, i1 in zip(idx[:-1], idx[1:]):
        n = i1 - i0 - 1
        if n <= 0:
            continue
        u = np.arange(1, n + 1) / (n + 1)
        base = v[i0] + (v[i1] - v[i0]) * u
        if kind == "smooth":
            w = ar(n + 1)
            noise = w[:n] - u * w[n]                       # pinned to 0 at both ends
            v[i0 + 1:i1] = base + noise
        else:
            per = int(round(60.0 / STEP))
            m = n // per                                    # number of 60-s holds inside the gap
            if m < 1:
                v[i0 + 1:i1] = v[i0]
            else:
                w = knots_walk(m + 1)
                uk = np.arange(1, m + 1) / (m + 1)
                kv = v[i0] + (v[i1] - v[i0]) * uk + (w[:m] - uk * w[m])
                seg = np.repeat(np.concatenate([[v[i0]], kv]), per)[:n]
                if len(seg) < n:
                    seg = np.concatenate([seg, np.full(n - len(seg), seg[-1] if len(seg) else v[i0])])
                v[i0 + 1:i1] = seg
    # edges: drift from the nearest real value toward the wearer's mean

    def edge(start_val, n):
        if n <= 0:
            return np.array([])
        if kind == "smooth":
            out = np.empty(n); x = start_val - mu
            for i in range(n):
                x = phi * x + rng.normal(0, se); out[i] = mu + x
            return out
        per = int(round(60.0 / STEP)); m = n // per + 1
        vals = [start_val]
        for _ in range(m):
            vals.append(vals[-1] + 0.25 * (mu - vals[-1]) + rng.normal(0, step_sd))
        return np.repeat(np.array(vals), per)[:n]

    first, last = idx[0], idx[-1]
    if first > 0:
        v[:first] = edge(v[first], first)[::-1]
    if last < len(v) - 1:
        v[last + 1:] = edge(v[last], len(v) - 1 - last)
    return grid, np.clip(v, 0.02, 0.98), real


def plot_continuous(ax, grid, v, real, color, lw, dash_alpha=0.6, merged=False, origin=None):
    """Real stretches solid in `color`; modelled stretches thin, grey and dashed ("simulated").
    merged=True is refused: a modelled value is never drawn as if it were measured."""
    if merged:
        raise ValueError("merged=True would draw simulated values as measured; use the dashed rendering")
    # origin given -> x axis is minutes since `origin` (a unix time); else wall-clock datetimes
    D = [(t - origin) / 60.0 for t in grid] if origin is not None else [dt.datetime.fromtimestamp(t) for t in grid]
    if merged:
        ax.plot(D, v, color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=3)
        return
    n = len(v); i = 0
    while i < n:
        j = i
        while j + 1 < n and real[j + 1] == real[i]:
            j += 1
        lo, hi = max(i - 1, 0), min(j + 1, n - 1)          # overlap one point so the pieces join
        if real[i]:
            ax.plot(D[i:j + 1], v[i:j + 1], color=color, lw=lw, solid_capstyle="round", zorder=3)
        else:
            ax.plot(D[lo:hi + 1], v[lo:hi + 1], color=SIM_COLOR, lw=lw * 0.6, alpha=dash_alpha, ls=(0, (3, 2)), zorder=2)
        i = j + 1
