"""Session recording: one folder per session under sessions/, plain CSV, written from a
queue thread so the acquisition threads never block on disk.

  eeg_raw.csv   t_host, sample, ch1..chN (uV, unfiltered), acc_x, acc_y, acc_z, pkt_idx, t_rx
                (t_host = smoothed board clock, sample = running count; pkt_idx = the board's own
                0-255 packet counter and t_rx = raw host receive time, so drops and stalls show)
  epochs.csv    t, flow_eeg + components, per-band mean relative power, per-channel
                alpha/theta/beta absolute power, n_good
  band.csv      t, flow, hr, anom, held, settling, fhi   (BioProxal [pub] lines)
  marks.csv     t, kind, value, note, id, edit_of         (felt-flow slider, buttons; append-and-close
                per row, so a hand edit while recording is safe. A correction / deletion is a later
                row of kind mark_edit / mark_delete with edit_of = the mark's id: apply_mark_edits)
  events.csv    t, source, event, detail                  (board reconnects / stalls, band link)
  session.json  board / montage / config snapshot; at stop also `stopped` + `at_stop`
                (board dropped / reconnects / stalls, band unit / MAC, counts)
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

# the band's per-second vitals and decode flags, after the fixed band.csv columns (pipeline.py imports this);
# the flag columns explain a blank flow: flags_raw = firmware flags byte, no_reading_flag = its bit 7
BAND_EXTRA = ("spo2", "pi", "rmssd", "sdnn", "rr", "flow_raw", "base_hr", "phase", "uptime_s", "flow_open5", "flow_end5",
              "flags_raw", "window", "no_reading_flag", "no_reading", "hrv_ok", "mode", "battery_mv",
              "quality", "ir_dc", "led", "rate_hz")


def _cell(v):
    if v is None:
        return ""
    return int(v) if isinstance(v, bool) else v


def apply_mark_edits(rows: list[dict]) -> list[dict]:
    """marks.csv rows (dicts) -> the marks as they stand after the operator's corrections:
    mark_edit rows replace the value / note of the mark they point at, mark_delete rows remove
    it; the edit rows themselves are dropped. Files without the id / edit_of columns pass through."""
    edits = [r for r in rows if (r.get("kind") or "") in ("mark_edit", "mark_delete") and r.get("edit_of")]
    if not edits:
        return [r for r in rows if (r.get("kind") or "") not in ("mark_edit", "mark_delete")]
    by_id = {}
    out = []
    for r in rows:
        if (r.get("kind") or "") in ("mark_edit", "mark_delete"):
            continue
        r = dict(r)
        out.append(r)
        if r.get("id"):
            by_id[r["id"]] = r
    gone = set()
    for e in edits:
        tgt = by_id.get(e["edit_of"])
        if tgt is None:
            continue
        if e["kind"] == "mark_delete":
            gone.add(e["edit_of"])
        else:
            if (e.get("value") or "") != "":
                tgt["value"] = e["value"]
            if (e.get("note") or "") != "":
                tgt["note"] = e["note"]
    return [r for r in out if r.get("id") not in gone]


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
        self.stop_meta_fn = None      # callable() -> dict merged into session.json["at_stop"] at stop

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
                   + ["acc_x", "acc_y", "acc_z", "pkt_idx", "t_rx"])
        self._open("events", ["t", "source", "event", "detail"])
        self._open("band", ["t", "flow", "hr", "anom", "held", "settling", "fhi", "asleep", "poor_contact"] + list(BAND_EXTRA))
        self._console = open(self.dir / "band_console.log", "a", encoding="utf-8")
        # marks.csv is opened, appended and closed per row (never held open): it is the one file people
        # hand-edit, and an editor saving it under a held handle left NUL bytes on 11-19
        with open(self.dir / "marks.csv", "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["t", "kind", "value", "note", "id", "edit_of"])
        self._open("task", ["t", "block_index", "condition", "task", "level", "problem", "answer", "response", "correct", "rt_ms"])
        self._epoch_header = None
        self.counts = {"eeg": 0, "epochs": 0, "band": 0, "marks": 0, "trials": 0, "events": 0}
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
        self._write_stop_meta()
        for f in list(self._files.values()) + [getattr(self, "_console", None)]:
            try:
                if f is not None:
                    f.close()
            except Exception:  # noqa: BLE001
                pass
        self._console = None
        d, self._files, self._writers = self.dir, {}, {}
        return d

    def _write_stop_meta(self) -> None:
        """Add the stop time and the end-of-session board / band state to session.json."""
        try:
            p = self.dir / "session.json"
            meta = json.loads(p.read_text("utf-8")) if p.exists() else {}
            extra = {}
            if self.stop_meta_fn is not None:
                try:
                    extra = dict(self.stop_meta_fn() or {})
                except Exception as e:  # noqa: BLE001
                    extra = {"error": repr(e)}
            meta["stopped"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            meta["stopped_t"] = round(time.time(), 3)
            meta["at_stop"] = {**extra, "counts": dict(self.counts)}
            p.write_text(json.dumps(meta, indent=2, default=str), "utf-8")
        except Exception as e:  # noqa: BLE001
            print(f"[recorder] session.json at stop: {e!r}")

    # --- producers (any thread) ---------------------------------------------------
    def eeg(self, samples: np.ndarray, accel: np.ndarray, t: np.ndarray, count0: int, pkt=None, rx=None) -> None:
        if self.active:
            self.q.put(("eeg", samples.copy(), accel.copy(), t.copy(), count0,
                        None if pkt is None else np.array(pkt), None if rx is None else np.array(rx)))

    def event(self, source: str, event: str, detail: str = "") -> None:
        """A link event (board reconnect / stall, band connect / error) -> events.csv."""
        if self.active:
            self.q.put(("event", time.time(), str(source), str(event), str(detail)[:300]))

    def epoch(self, rec: dict) -> None:
        if self.active:
            self.q.put(("epoch", dict(rec)))

    def band(self, s: dict) -> None:
        if self.active:
            self.q.put(("band", dict(s)))

    def mark(self, kind: str, value, note: str = "", mid: str | None = None, edit_of: str | None = None,
             t: float | None = None) -> None:
        if self.active:
            self.q.put(("mark", {"t": time.time() if t is None else float(t), "kind": kind, "value": value, "note": note,
                                 "id": mid or "", "edit_of": edit_of or ""}))

    def trial(self, rec: dict) -> None:
        if self.active:
            self.q.put(("trial", dict(rec)))

    def console(self, t: float, line: str) -> None:
        """Every line the band prints on USB, as it came (band_console.log)."""
        if self.active:
            self.q.put(("console", float(t), str(line)))

    # --- consumer -----------------------------------------------------------------
    def _writer(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                return
            try:
                kind = item[0]
                if kind == "eeg":
                    _, samples, accel, t, count0, pkt, rx = item
                    w = self._writers["eeg_raw"]
                    has_meta = pkt is not None and rx is not None and len(pkt) == len(samples) == len(rx)
                    for i in range(len(samples)):
                        w.writerow([f"{t[i]:.4f}", count0 + i] + [f"{v:.2f}" for v in samples[i]]
                                   + [f"{v:.3f}" for v in accel[i]]
                                   + ([int(pkt[i]), f"{rx[i]:.4f}"] if has_meta else ["", ""]))
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
                                                    int(bool(s.get("asleep"))), int(bool(s.get("poor_contact")))]
                                                   + [_cell(s.get(k)) for k in BAND_EXTRA])
                    self.counts["band"] += 1
                elif kind == "event":
                    _, t, source, ev, detail = item
                    self._writers["events"].writerow([f"{t:.3f}", source, ev, detail])
                    self.counts["events"] = self.counts.get("events", 0) + 1
                elif kind == "console":
                    _, t, line = item
                    if getattr(self, "_console", None) is not None:
                        self._console.write(f"{t:.3f} {line}\n")
                        self.counts["console"] = self.counts.get("console", 0) + 1
                elif kind == "mark":
                    m = item[1]
                    with open(self.dir / "marks.csv", "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([f"{m['t']:.3f}", m["kind"], m["value"], m["note"],
                                                m.get("id", ""), m.get("edit_of", "")])
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
                    if getattr(self, "_console", None) is not None:
                        self._console.flush()
            except Exception as e:  # noqa: BLE001
                print(f"[recorder] write error: {e!r}")

    def status(self) -> dict:
        return {"active": self.active, "dir": str(self.dir) if self.dir else None,
                "since_s": round(time.time() - self.started_at, 1) if self.active and self.started_at else None,
                "counts": dict(self.counts)}
