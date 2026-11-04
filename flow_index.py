"""EEG flow index: a transparent proxy for "being in flow" computed from scalp EEG band
powers, used as the reference the BioProxal band's flow score is compared against.

It is a PROXY, not ground truth. The literature on EEG correlates of flow (Katahira et
al. 2018; Ewing et al. 2016; Nacke & Lindley 2008 and the Pope 1995 engagement index) is
consistent on three points, and the index is built from exactly those:

  1. frontal theta (4-8 Hz, midline/frontal sites) rises in flow vs. boredom AND overload;
  2. alpha (8-13 Hz, parietal/occipital) is moderate - neither the high alpha of idling
     nor the suppressed alpha of strain;
  3. beta (13-30 Hz) is lower than in overload / anxious effort (engagement index
     beta / (alpha + theta) peaks in overload, not in flow).

Every 2-s epoch each component is z-scored against a rolling baseline of the wearer's own
session (the band also judges against a personal baseline), and combined:

    x        = w_theta * z(theta_frontal_rel)
             - w_alpha * |z(alpha_rel)|                # "moderate" alpha: any departure costs
             - w_beta  * z(beta_rel)
    flow_eeg = sigmoid(1.5 * x)                        # 0.5 while every z is 0 (warm-up)

with default weights 0.5 / 0.25 / 0.25 (config.json "flow_index"). The raw components are
exported too, so report.py can tell which of them (if any) tracks the band's score."""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from dsp import BAND_ORDER


class RollingZ:
    """Mean / std over the last `n` values (default 150 epochs = 5 min at 2 s)."""

    def __init__(self, n: int = 150, warmup: int = 15):
        self.q = deque(maxlen=n)
        self.warmup = warmup

    def push(self, v: float) -> float:
        self.q.append(v)
        if len(self.q) < self.warmup:
            return 0.0
        a = np.asarray(self.q)
        sd = a.std()
        return float((v - a.mean()) / sd) if sd > 1e-9 else 0.0

    @property
    def ready(self) -> bool:
        return len(self.q) >= self.warmup


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


FRONTAL_PREFIXES = ("Fp", "F", "AF", "Fz")
POSTERIOR_PREFIXES = ("O", "P", "PO", "Pz")
CENTRAL_PREFIXES = ("C", "Cz", "T")


def _pick(labels: list[str], prefixes: tuple[str, ...], good: list[bool]) -> list[int]:
    idx = [i for i, lab in enumerate(labels)
           if good[i] and any(lab.startswith(p) for p in prefixes)]
    return idx


class FlowIndex:
    def __init__(self, labels: list[str], weights: dict | None = None,
                 baseline_epochs: int = 150, warmup_epochs: int = 15):
        self.labels = labels
        w = weights or {}
        self.w_theta = float(w.get("theta", 0.5))
        self.w_alpha = float(w.get("alpha", 0.25))
        self.w_beta = float(w.get("beta", 0.25))
        self.z_theta = RollingZ(baseline_epochs, warmup_epochs)
        self.z_alpha = RollingZ(baseline_epochs, warmup_epochs)
        self.z_beta = RollingZ(baseline_epochs, warmup_epochs)
        self.last = None

    def update(self, rel: dict[str, np.ndarray], abs_pow: dict[str, np.ndarray],
               good: list[bool]) -> dict:
        """rel/abs_pow: band -> per-channel power for one epoch. Returns the epoch record."""
        n = len(self.labels)
        if not any(good):
            good = [True] * n
        frontal = _pick(self.labels, FRONTAL_PREFIXES, good) or [i for i in range(n) if good[i]]
        posterior = _pick(self.labels, POSTERIOR_PREFIXES, good) or [i for i in range(n) if good[i]]
        allgood = [i for i in range(n) if good[i]]

        theta_f = float(np.mean(rel["theta"][frontal]))
        alpha_p = float(np.mean(rel["alpha"][posterior]))
        beta_a = float(np.mean(rel["beta"][allgood]))
        engagement = float(np.mean(abs_pow["beta"][allgood])
                           / (np.mean(abs_pow["alpha"][allgood]) + np.mean(abs_pow["theta"][allgood]) + 1e-9))
        theta_alpha = float(np.mean(abs_pow["theta"][frontal]) / (np.mean(abs_pow["alpha"][frontal]) + 1e-9))

        zt = self.z_theta.push(theta_f)
        za = self.z_alpha.push(alpha_p)
        zb = self.z_beta.push(beta_a)
        x = self.w_theta * zt - self.w_alpha * abs(za) - self.w_beta * zb
        flow = sigmoid(1.5 * x)
        rec = {
            "flow_eeg": round(flow, 4),
            "theta_frontal_rel": round(theta_f, 5),
            "alpha_posterior_rel": round(alpha_p, 5),
            "beta_rel": round(beta_a, 5),
            "engagement_index": round(engagement, 4),
            "theta_alpha_ratio": round(theta_alpha, 4),
            "z_theta": round(zt, 3), "z_alpha": round(za, 3), "z_beta": round(zb, 3),
            "baseline_ready": self.z_theta.ready,
            "n_good": len(allgood),
        }
        for b in BAND_ORDER:
            rec[f"{b}_rel_mean"] = round(float(np.mean(rel[b][allgood])), 5)
        self.last = rec
        return rec
