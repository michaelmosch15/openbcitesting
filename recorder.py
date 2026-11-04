"""Session recording: one folder per session under sessions/, plain CSV, written from a
queue thread so the acquisition threads never block on disk.

  eeg_raw.csv   t_host, sample, ch1..chN (uV, unfiltered), acc_x, acc_y, acc_z
  epochs.csv    t, flow_eeg + components, per-band mean relative power, per-channel
                alpha/theta/beta absolute power, n_good
  band.csv      t, flow, hr, anom, held, settling, fhi   (BioProxal [pub] lines)
  marks.csv     t, kind, value, note                      (felt-flow slider, buttons)
  session.json  board / montage / config snapshot
"""
from __future__ import annotations

import csv
import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np


class Recorder:
    def __init__(self, root: Path, labels: list[str], meta: dict):
        self.root = Path(root)
        self.labels = labels
        self.meta = meta
        self.dir: Path | None = None
        self.q: queue.Queue = queue.Queue()
        self._thread = None
        self._files = {}
        self._writers = {}
        self.active = False
        self.started_at = None
        self.counts = {"eeg": 0, "epochs": 0, "band": 0, "marks": 0}

    def start(self, label: str = "") -> Path:
        if self.active:
            return self.dir
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = stamp + (f"_{label}" if label else "")
        self.dir = self.root / name
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "session.json").write_text(json.dumps(
            {**self.meta, "labels": self.labels, "started": stamp}, indent=2), "utf-8")
        self._open("eeg_raw", ["t_host", "sample"] + [f"ch{i + 1}_{l}" for i, l in enumerate(self.labels)]
                   + ["acc_x", "acc_y", "acc_z"])
        self._open("band", ["t", "flow", "hr", "anom", "held", "settling", "fhi", "asleep", "poor_contact"])
        self._open("marks", ["t", "kind", "value", "note"])
        self._open("task", ["t", "block_index", "condition", "task", "level", "problem", "answer", "response", "correct", "rt_ms"])
        self._epoch_header = None
        self.counts = {"eeg": 0, "epochs": 0, "band": 0, "marks": 0, "trials": 0}
        self.active = True
        self.started_at = time.time()
        self._thread = threading.Thread(target=self._writer, name="recorder", daemon=True)
        self._thread.start()
        return self.dir

    def _open(self, name: str, header: list[str]) -> None:
        f = open(self.dir / f"{name}.csv", "w", newline="", encoding="utf-8")
        w = csv.writer(f)
        w.writerow(header)
        self._files[name], self._writers[name] = f, w

    def stop(self) -> Path | None:
        if not self.active:
            return None
        self.active = False
        self.q.put(None)
        if self._thread:
            self._thread.join(5.0)
        for f in self._files.values():
            try:
                f.close()
            except Exception:  # noqa: BLE001
                pass
        d, self._files, self._writers = self.dir, {}, {}
        return d

    # --- producers (any thread) ---------------------------------------------------
    def eeg(self, samples: np.ndarray, accel: np.ndarray, t: np.ndarray, count0: int) -> None:
        if self.active:
            self.q.put(("eeg", samples.copy(), accel.copy(), t.copy(), count0))

    def epoch(self, rec: dict) -> None:
        if self.active:
            self.q.put(("epoch", dict(rec)))

    def band(self, s: dict) -> None:
        if self.active:
            self.q.put(("band", dict(s)))

    def mark(self, kind: str, value, note: str = "") -> None:
        if self.active:
            self.q.put(("mark", {"t": time.time(), "kind": kind, "value": value, "note": note}))

    def trial(self, rec: dict) -> None:
        if self.active:
            self.q.put(("trial", dict(rec)))

    # --- consumer -----------------------------------------------------------------
    def _writer(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                return
            try:
                kind = item[0]
                if kind == "eeg":
                    _, samples, accel, t, count0 = item
                    w = self._writers["eeg_raw"]
                    for i in range(len(samples)):
                        w.writerow([f"{t[i]:.4f}", count0 + i] + [f"{v:.2f}" for v in samples[i]]
                                   + [f"{v:.3f}" for v in accel[i]])
                    self.counts["eeg"] += len(samples)
                elif kind == "epoch":
                    rec = item[1]
                    if self._epoch_header is None:
                        self._epoch_header = list(rec.keys())
                        self._open("epochs", self._epoch_header)
                    self._writers["epochs"].writerow([rec.get(k, "") for k in self._epoch_header])
                    self.counts["epochs"] += 1
                elif kind == "band":
                    s = item[1]
                    self._writers["band"].writerow([f"{s['t']:.3f}", s["flow"], s["hr"], s["anom"],
                                                    int(s["held"]), int(s["settling"]),
                                                    "" if s.get("fhi") is None else s["fhi"],
                                                    int(bool(s.get("asleep"))), int(bool(s.get("poor_contact")))])
                    self.counts["band"] += 1
                elif kind == "mark":
                    m = item[1]
                    self._writers["marks"].writerow([f"{m['t']:.3f}", m["kind"], m["value"], m["note"]])
                    self.counts["marks"] += 1
                elif kind == "trial":
                    r = item[1]
                    self._writers["task"].writerow([f"{float(r.get('t', time.time())):.3f}", r.get("block_index", ""),
                                                    r.get("condition", ""), r.get("task", ""), r.get("level", ""),
                                                    r.get("problem", ""), r.get("answer", ""), r.get("response", ""),
                                                    int(bool(r.get("correct"))), r.get("rt_ms", "")])
                    self.counts["trials"] = self.counts.get("trials", 0) + 1
                if self.q.qsize() == 0:
                    for f in self._files.values():
                        f.flush()
            except Exception as e:  # noqa: BLE001
                print(f"[recorder] write error: {e!r}")

    def status(self) -> dict:
        return {"active": self.active, "dir": str(self.dir) if self.dir else None,
                "since_s": round(time.time() - self.started_at, 1) if self.active and self.started_at else None,
                "counts": dict(self.counts)}
