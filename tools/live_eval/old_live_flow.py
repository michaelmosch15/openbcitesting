"""Frozen copy of pipeline.Pipeline._live_flow as it was on 2026-11-19 ~14:55 (the "before" of the
offline comparison in replay_live.py). Not used by the app."""
from __future__ import annotations

import numpy as np

from dsp import band_powers_of, welch_masked
from flow_index import FRONTAL_FALLBACK, FRONTAL_PREFERRED, sigmoid


def old_live_flow(self, filt_long: np.ndarray, good: list) -> dict:
    """Live helmet flow: a number whenever the helmet is on a head, like the band. Not the
    pre-registered index (flow_eeg); never paired with the band or used by report.py."""
    none = {"flow_live": None, "live_channels": None, "live_z": None}
    if "reference" in self.readiness.get("reasons", []):          # not on a head / no ear clip
        return none
    fr = [i for i, lab in enumerate(self.labels) if good[i] and lab in FRONTAL_PREFERRED]
    fr = fr or [i for i, lab in enumerate(self.labels) if good[i] and lab in FRONTAL_FALLBACK]
    if not fr:
        return none
    # brain signal only: per channel, keep just the 2-s segments inside the same artifact limits as
    # the strict index (p2p <= artifact_p2p_uv, |x| <= artifact_abs_uv); no clean segment -> no number
    def bad(seg: np.ndarray) -> np.ndarray:
        m = (seg.max(axis=0) - seg.min(axis=0) > self.artifact_p2p_uv) | (np.abs(seg).max(axis=0) > self.artifact_abs_uv)
        m[[i for i in range(self.n_ch) if i not in fr]] = True
        return m
    fi, Pi, used_per_ch, n_total = welch_masked(filt_long, self.fs, self.index_seg_s, 0.5, bad)
    if not n_total:
        return none
    need = max(1, int(np.ceil(self.index_min_clean * n_total)))      # same bar as the strict index, per channel
    fr = [i for i in fr if used_per_ch[i] >= need]
    if not fr:
        return {**none, "live_channels": "no clean signal"}
    bp = band_powers_of(fi, np.nan_to_num(Pi, nan=0.0), {"theta": (4.0, 7.0), "total": (4.0, 45.0)})
    rel = [bp["theta"][i] / bp["total"][i] for i in fr if bp["total"][i] > 0]
    if not rel:
        return none
    v = float(np.log(max(float(np.median(rel)), 1e-9)))
    z = self.live_base.push(v)
    if not self.live_base.ready:
        return {**none, "live_channels": " ".join(self.labels[i] for i in fr)}
    self._live_recent.append(sigmoid(max(-20.0, min(20.0, z))))
    return {"flow_live": round(float(np.median(self._live_recent)), 4),
            "live_channels": " ".join(self.labels[i] for i in fr), "live_z": round(z, 3)}
