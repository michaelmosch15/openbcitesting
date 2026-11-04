"""EEG flow index: the reference the BioProxal band's flow score is validated against.

What the literature supports (docs/EEG_FLOW_LITERATURE.md, docs/EEG_FLOW_INDEX.md):

  * The ONE marker that replicates across independent labs and paradigms: frontal /
    fronto-midline theta (4-7 Hz; Fz, F3/F4 region) is HIGHER in the demand-matched "flow"
    condition than in the too-easy "boredom" condition (Katahira et al. 2018; Ewing,
    Fairclough & Gilleade 2016; Fairclough & Ewing 2017; Eschmann et al. 2022 neurofeedback;
    Knierim et al. 2025 with a Cyton+Daisy). Theta vs overload is unresolved (equal in
    Katahira, inverted-U in Ewing).
  * Alpha: "flow sits between boredom and overload" is the only point of agreement; the sign
    and the sites differ between studies (right-central in Katahira; frontal/parietal upper
    alpha falling with demand in Ewing; nulls elsewhere).
  * Beta: inconsistent direction across studies. The Pope et al. 1995 engagement index
    beta/(alpha+theta) is monotonic with demand (vigilance), never validated against flow.

So the PRIMARY index is frontal theta alone, and the alpha term is kept as a separately
reported EXPLORATORY index; beta is exported as a covariate only.

Per 2 s, over the last 10 s (Welch, 2-s Hann segments, 50 % overlap, artifact segments
dropped), relative power = band / 1-45 Hz total (so impedance differences cancel), log, then
z-scored against the wearer's OWN eyes-open resting baseline (the protocol's rest block,
frozen when it ends; `baseline_mode` "rest"; an expanding session baseline until then):

    z_theta  = z(log theta_fm_rel at frontal sites)      # F3 F4 F7 F8 Fz (fallback Fp1 Fp2)
    z_alpha  = z(log alpha_hi_rel at central/parietal)   # C3 C4 P3 P4 Cz Pz (fallback all)
    z_beta   = z(log beta_rel at all good sites)

    flow_eeg     = sigmoid( w_theta * z_theta - w_alpha * |z_alpha| - w_beta * z_beta )
                   with default weights 1.0 / 0.0 / 0.0  -> frontal theta alone
    flow_eeg_alt = sigmoid( z_theta - 0.5 * |z_alpha| )   # exploratory "theta + moderate alpha"

0.5 = at the wearer's resting baseline. Artifact epochs (fewer than half the Welch segments
clean) carry no index value and never touch the baselines.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from dsp import BAND_ORDER

FRONTAL_PREFERRED = ("Fz", "F3", "F4", "F7", "F8", "AFz", "AF3", "AF4", "FCz", "FC1", "FC2")
FRONTAL_FALLBACK = ("Fp1", "Fp2", "Fpz")
CENTRAL_PARIETAL = ("C3", "C4", "Cz", "P3", "P4", "Pz", "CP1", "CP2", "CP5", "CP6")
DEFAULT_WEIGHTS = {"theta": 1.0, "alpha": 0.0, "beta": 0.0}
ALT_ALPHA_WEIGHT = 0.5


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Baseline:
    """Per-component mean / std for z-scoring, in one of three modes:
    rest     - collect while `collecting` is True (the rest block), freeze at the end;
               until frozen, behave like `session`;
    session  - expanding mean/std over everything seen so far (never forgets);
    rolling  - the last `window` values only (discouraged: it subtracts out the slow
               changes the validation is looking for).
    `warmup` values are needed before any z is non-zero."""

    def __init__(self, mode: str = "rest", window: int = 150, warmup: int = 15):
        self.mode = mode
        self.window, self.warmup = int(window), int(warmup)
        self.collecting = False
        self.frozen = False
        self._rest: list[float] = []
        self._session_n, self._session_sum, self._session_sq = 0, 0.0, 0.0
        self._roll: deque = deque(maxlen=self.window)
        self.mean, self.std = 0.0, 1.0

    def start_rest(self) -> None:
        self.collecting, self.frozen, self._rest = True, False, []

    def end_rest(self) -> None:
        self.collecting = False
        if len(self._rest) >= self.warmup:
            a = np.asarray(self._rest)
            sd = float(a.std())
            self.mean, self.std = float(a.mean()), (sd if sd > 1e-9 else 1.0)
            self.frozen = True

    def push(self, v: float) -> float:
        if self.collecting:
            self._rest.append(v)
        self._session_n += 1
        self._session_sum += v
        self._session_sq += v * v
        self._roll.append(v)
        if self.mode == "rest" and self.frozen:
            pass
        elif self.mode == "rolling":
            if len(self._roll) < self.warmup:
                return 0.0
            a = np.asarray(self._roll)
            sd = float(a.std())
            self.mean, self.std = float(a.mean()), (sd if sd > 1e-9 else 1.0)
        else:                                   # session (also rest-before-frozen)
            if self._session_n < self.warmup:
                return 0.0
            m = self._session_sum / self._session_n
            var = max(self._session_sq / self._session_n - m * m, 0.0)
            self.mean, self.std = m, (math.sqrt(var) if var > 1e-18 else 1.0)
        return float((v - self.mean) / self.std)

    @property
    def ready(self) -> bool:
        if self.mode == "rest" and self.frozen:
            return True
        n = len(self._roll) if self.mode == "rolling" else self._session_n
        return n >= self.warmup

    def state(self) -> str:
        if self.mode == "rest":
            return "rest-frozen" if self.frozen else ("rest-collecting" if self.collecting else "session-until-rest")
        return self.mode


def _pick(labels: list[str], names: tuple[str, ...], good: list[bool]) -> list[int]:
    return [i for i, lab in enumerate(labels) if good[i] and lab in names]


class FlowIndex:
    def __init__(self, labels: list[str], weights: dict | None = None, baseline_mode: str = "rest",
                 baseline_epochs: int = 150, warmup_epochs: int = 15):
        self.labels = labels
        w = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.w_theta, self.w_alpha, self.w_beta = float(w["theta"]), float(w["alpha"]), float(w["beta"])
        self.baselines = {k: Baseline(baseline_mode, baseline_epochs, warmup_epochs) for k in ("theta", "alpha", "beta")}
        self.last = None
        self.n_artifact = 0
        self.n_epochs = 0

    # --- baseline control (driven by the protocol's rest block) ---------------------
    def start_rest(self) -> None:
        for b in self.baselines.values():
            b.start_rest()

    def end_rest(self) -> None:
        for b in self.baselines.values():
            b.end_rest()

    @property
    def baseline_state(self) -> str:
        return self.baselines["theta"].state()

    @property
    def baseline_ready(self) -> bool:
        return all(b.ready for b in self.baselines.values())

    # --- sites ---------------------------------------------------------------------
    def sites(self, good: list[bool]) -> dict[str, list[int]]:
        n = len(self.labels)
        allgood = [i for i in range(n) if good[i]] or list(range(n))
        frontal = _pick(self.labels, FRONTAL_PREFERRED, good) or _pick(self.labels, FRONTAL_FALLBACK, good) or allgood
        cp = _pick(self.labels, CENTRAL_PARIETAL, good) or allgood
        return {"frontal": frontal, "central_parietal": cp, "all": allgood}

    # --- the index -------------------------------------------------------------------
    def update(self, rel: dict[str, np.ndarray], abs_pow: dict[str, np.ndarray],
               good: list[bool], artifact: bool = False, extra: dict | None = None) -> dict:
        """rel/abs_pow: band -> per-channel power for one epoch (display bands). `extra`: the
        index bands' relative powers {"theta_fm", "alpha_hi"} (fall back to theta / alpha).
        Returns the epoch record; `flow_eeg` is None for an artifact epoch and nothing is
        pushed into the baselines then."""
        n = len(self.labels)
        if not any(good):
            good = [True] * n
        s = self.sites(good)
        extra = extra or {}
        theta_rel = np.asarray(extra.get("theta_fm", rel["theta"]), float)
        alpha_rel = np.asarray(extra.get("alpha_hi", rel["alpha"]), float)
        theta_f = float(np.mean(theta_rel[s["frontal"]]))
        alpha_cp = float(np.mean(alpha_rel[s["central_parietal"]]))
        beta_a = float(np.mean(rel["beta"][s["all"]]))
        engagement = float(np.mean(abs_pow["beta"][s["all"]])
                           / (np.mean(abs_pow["alpha"][s["all"]]) + np.mean(abs_pow["theta"][s["all"]]) + 1e-9))
        theta_alpha = float(np.mean(abs_pow["theta"][s["frontal"]]) / (np.mean(abs_pow["alpha"][s["frontal"]]) + 1e-9))
        self.n_epochs += 1
        rec = {
            "flow_eeg": None,
            "flow_eeg_alt": None,
            "artifact": bool(artifact),
            "theta_frontal_rel": round(theta_f, 5),
            "alpha_cp_rel": round(alpha_cp, 5),
            "beta_rel": round(beta_a, 5),
            "engagement_index": round(engagement, 4),
            "theta_alpha_ratio": round(theta_alpha, 4),
            "z_theta": 0.0, "z_alpha": 0.0, "z_beta": 0.0,
            "baseline_ready": self.baseline_ready,
            "baseline_state": self.baseline_state,
            "n_good": len([i for i in range(n) if good[i]]),
            "frontal_sites": ",".join(self.labels[i] for i in s["frontal"]),
        }
        for b in BAND_ORDER:
            rec[f"{b}_rel_mean"] = round(float(np.mean(rel[b][s["all"]])), 5)
        if artifact or not np.isfinite(theta_f) or not np.isfinite(alpha_cp) or not np.isfinite(beta_a):
            rec["artifact"] = True
            self.n_artifact += 1
            self.last = rec
            return rec
        zt = self.baselines["theta"].push(math.log(max(theta_f, 1e-6)))
        za = self.baselines["alpha"].push(math.log(max(alpha_cp, 1e-6)))
        zb = self.baselines["beta"].push(math.log(max(beta_a, 1e-6)))
        x = self.w_theta * zt - self.w_alpha * abs(za) - self.w_beta * zb
        rec.update({"flow_eeg": round(sigmoid(x), 4),
                    "flow_eeg_alt": round(sigmoid(zt - ALT_ALPHA_WEIGHT * abs(za)), 4),
                    "z_theta": round(zt, 3), "z_alpha": round(za, 3), "z_beta": round(zb, 3),
                    "baseline_ready": self.baseline_ready, "baseline_state": self.baseline_state})
        self.last = rec
        return rec
