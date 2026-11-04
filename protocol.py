"""Experiment protocol runner: a timed sequence of blocks (rest baseline, boredom, flow,
overload ...) that induces known flow / non-flow periods, so the helmet's EEG flow index and
the band's flow score can both be checked against the condition each block was designed to
produce (Katahira et al. 2018 difficulty-manipulation paradigm; docs/VALIDATION_PROTOCOL.md).

protocols.json holds the schedules. A session's block order is counterbalanced from a seed
(the session start time) and logged. Every transition is written as a mark
(kind block_start / block_end, value = condition) and every self-report as a mark of kind
rating; task trials go to the session's task.csv (server.py -> Pipeline.trial), so
report.py can score per block.
"""
from __future__ import annotations

import json
import random
import threading
import time
from pathlib import Path

HERE = Path(__file__).parent


def load_protocols(path: Path | None = None) -> dict:
    p = path or HERE / "protocols.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text("utf-8"))


class ProtocolRunner:
    def __init__(self, protocols: dict, on_event, on_mark, log=print):
        """on_event(dict) broadcasts to the browser; on_mark(kind, value, note) records a mark."""
        self.protocols = protocols
        self.on_event, self.on_mark, self.log = on_event, on_mark, log
        self.active = False
        self.name = None
        self.blocks: list[dict] = []
        self.index = -1
        self.block_started = 0.0
        self.order_seed = None
        self.pending_rating = None        # block index awaiting a self-report
        self.ratings: list[dict] = []
        self._stop = threading.Event()
        self._advance = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    # --- schedule ----------------------------------------------------------------
    def build(self, name: str, seed: int | None = None) -> list[dict]:
        proto = self.protocols.get(name)
        if not proto:
            raise ValueError(f"unknown protocol {name!r}")
        seed = int(time.time()) if seed is None else int(seed)
        rng = random.Random(seed)
        blocks: list[dict] = []
        for item in proto["blocks"]:
            if "shuffle" in item:
                group = [dict(b) for b in item["shuffle"]]
                reps = int(item.get("repeat", 1))
                between = item.get("between")
                for r in range(reps):
                    g = list(group)
                    rng.shuffle(g)
                    last_cond = next((b["condition"] for b in reversed(blocks) if b.get("condition") != (between or {}).get("condition")), None)
                    if g and last_cond == g[0].get("condition") and len(g) > 1:
                        g[0], g[1] = g[1], g[0]           # never the same condition twice in a row
                    for b in g:
                        b = dict(b)
                        b["rep"] = r + 1
                        blocks.append(b)
                        if between:
                            blocks.append(dict(between))
                if between and blocks and blocks[-1].get("condition") == between.get("condition"):
                    blocks.pop()                           # no trailing pause
            else:
                blocks.append(dict(item))
        for i, b in enumerate(blocks):
            b["index"] = i
            b.setdefault("rating", False)
            b.setdefault("baseline", False)
            b.setdefault("task", None)
        self.order_seed = seed
        return blocks

    # --- control -------------------------------------------------------------------
    def start(self, name: str, seed: int | None = None) -> dict:
        with self._lock:
            if self.active:
                raise ValueError("a protocol is already running")
            self.blocks = self.build(name, seed)
            self.name = name
            self.index = -1
            self.ratings = []
            self.pending_rating = None
            self.active = True
            self._stop.clear()
            self._advance.clear()
        self.on_mark("protocol_start", name, f"seed={self.order_seed} order=" + ",".join(b["condition"] for b in self.blocks))
        self._thread = threading.Thread(target=self._run, name="protocol", daemon=True)
        self._thread.start()
        return self.status()

    def stop(self) -> dict:
        with self._lock:
            was = self.active
            self.active = False
            self._stop.set()
        if was:
            self._end_block(aborted=True)
            self.on_mark("protocol_stop", self.name, "")
            self.on_event({"phase": "stopped", "name": self.name})
        return self.status()

    def next(self) -> dict:
        """Skip to the next block now."""
        self._advance.set()
        return self.status()

    def rate(self, block_index: int, items: dict, note: str = "") -> dict:
        b = self.blocks[block_index] if 0 <= block_index < len(self.blocks) else None
        rec = {"t": time.time(), "block_index": block_index, "condition": b["condition"] if b else None,
               "items": items, "note": note}
        self.ratings.append(rec)
        if self.pending_rating == block_index:
            self.pending_rating = None
        self.on_mark("rating", json.dumps({"block": block_index, "condition": rec["condition"], **items}), note)
        self.on_event({"phase": "rated", "block_index": block_index})
        return rec

    # --- loop ----------------------------------------------------------------------
    def _run(self) -> None:
        for i, b in enumerate(self.blocks):
            if self._stop.is_set():
                break
            with self._lock:
                self.index = i
                self.block_started = time.time()
            self.on_mark("block_start", b["condition"], f"block={i} {b.get('label', '')}")
            self.on_event({"phase": "block_start", **self._block_view(b)})
            self._advance.clear()
            dur = float(b.get("duration_s", 60))
            while not self._stop.is_set() and not self._advance.is_set():
                if time.time() - self.block_started >= dur:
                    break
                self._advance.wait(0.25)
            if self._stop.is_set():
                break
            self._end_block()
        if not self._stop.is_set():
            with self._lock:
                self.active = False
                self.index = len(self.blocks)
            self.on_mark("protocol_end", self.name, "")
            self.on_event({"phase": "done", "name": self.name})

    def _end_block(self, aborted: bool = False) -> None:
        i = self.index
        if not (0 <= i < len(self.blocks)):
            return
        b = self.blocks[i]
        self.on_mark("block_end", b["condition"], f"block={i}" + (" aborted" if aborted else ""))
        if b.get("rating") and not aborted:
            self.pending_rating = i
            self.on_event({"phase": "rating_due", **self._block_view(b)})
        self.on_event({"phase": "block_end", **self._block_view(b)})

    # --- views ---------------------------------------------------------------------
    def _block_view(self, b: dict) -> dict:
        return {"block_index": b["index"], "condition": b["condition"], "label": b.get("label", b["condition"]),
                "instruction": b.get("instruction", ""), "duration_s": b.get("duration_s"),
                "task": b.get("task"), "rating": bool(b.get("rating")), "baseline": bool(b.get("baseline")),
                "rep": b.get("rep")}

    def current(self) -> dict | None:
        if 0 <= self.index < len(self.blocks):
            return self.blocks[self.index]
        return None

    def status(self) -> dict:
        b = self.current()
        rem = None
        if self.active and b is not None:
            rem = max(0.0, float(b.get("duration_s", 0)) - (time.time() - self.block_started))
        return {"active": self.active, "name": self.name, "block_index": self.index if self.active else None,
                "total_blocks": len(self.blocks), "block": self._block_view(b) if (self.active and b) else None,
                "remaining_s": None if rem is None else round(rem, 1),
                "pending_rating": self.pending_rating, "order_seed": self.order_seed,
                "schedule": [self._block_view(x) for x in self.blocks],
                "available": [{"name": k, "title": v.get("title", k), "description": v.get("description", ""),
                               "total_min": round(sum(float(x.get("duration_s", 0)) for x in self._preview(k)) / 60, 1)}
                              for k, v in self.protocols.items()]}

    def _preview(self, name: str) -> list[dict]:
        """The schedule as build() would lay it out, without touching the runner's state."""
        keep = self.order_seed
        try:
            return self.build(name, seed=0)
        finally:
            self.order_seed = keep
