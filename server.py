"""Localhost HTTP + Server-Sent-Events server (stdlib only). Routes in docs/API.md."""
from __future__ import annotations

import json
import mimetypes
import re
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

STATIC = Path(__file__).parent / "static"
SERVER_VERSION = "0.2.0"
SAFE_LABEL = re.compile(r"^[A-Za-z0-9_-]{0,32}$")


class App:
    """What the handlers talk to: the pipeline, the board, the band link and the recorder,
    plus the command dispatcher. Built by app.py."""

    def __init__(self, pipeline, board, recorder, band_manager, config: dict, log=print, protocol=None):
        self.pipeline, self.board, self.recorder, self.bands, self.cfg, self.log = pipeline, board, recorder, band_manager, config, log
        self.protocol = protocol            # ProtocolRunner, wired by app.py (may be None)
        self.astra = None                   # astra.Astra, wired by app.py after construction (may be None)
        self.subscribers: list = []
        self.sub_lock = threading.Lock()
        self.sessions_dir = Path(__file__).parent / "sessions"
        self._ticker = threading.Thread(target=self._tick, name="sse-tick", daemon=True)
        self._ticker.start()

    # --- broadcast ----------------------------------------------------------------
    def broadcast(self, event: str, data) -> None:
        msg = f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'), allow_nan=False)}\n\n".encode("utf-8")
        with self.sub_lock:
            subs = list(self.subscribers)
        for q in subs:
            try:
                q.append(msg)
            except Exception:  # noqa: BLE001
                pass

    def _tick(self) -> None:
        last_analysis_id = None
        last_status = 0.0
        while True:
            time.sleep(0.1)
            try:
                chunk = self.pipeline.drain_samples()
                if chunk is not None:
                    self.broadcast("samples", chunk)
                a = self.pipeline.last_analysis
                if a is not None and id(a) != last_analysis_id:
                    last_analysis_id = id(a)
                    a["band"] = self.pipeline.band_view()
                    a["stats"] = self.pipeline.stats
                    self.broadcast("analysis", a)
                if time.time() - last_status >= 1.0:
                    last_status = time.time()
                    self.broadcast("status", self.status())
            except Exception as e:  # noqa: BLE001
                self.log(f"[sse] tick error: {e!r}")

    def on_band(self, s: dict) -> None:
        self.pipeline.on_band(s)
        self.broadcast("band", self.pipeline.band_view())

    # --- snapshots ----------------------------------------------------------------
    def status(self) -> dict:
        return {"board": self.board.status(), "band": self.bands.status(), "recorder": self.recorder.status(),
                "filter": self.pipeline.filter_cfg, "flow_weights": self.pipeline.flow_weights,
                "flow_index": self.pipeline.flow_status(),
                "protocol": self.protocol.status() if self.protocol else None,
                "astra": self.astra.status() if self.astra else None,
                "autocal": self.pipeline.autocal.status() if self.pipeline.autocal is not None else None,
                "labels": list(self.pipeline.labels),
                "clock": round(time.time(), 3)}

    # --- protocol runner callbacks ------------------------------------------------
    def on_protocol_event(self, ev: dict) -> None:
        """Block transitions drive the EEG baseline: the rest block marked `baseline` collects
        the wearer's resting statistics and freezes them when it ends."""
        phase = ev.get("phase")
        if phase == "block_start" and ev.get("baseline"):
            self.pipeline.flow.start_rest()
            self.log("[protocol] collecting the resting baseline")
        elif phase == "block_end" and ev.get("baseline"):
            self.pipeline.flow.end_rest()
            self.log(f"[protocol] baseline {self.pipeline.flow.baseline_state}")
        self.pipeline.on_protocol(ev)
        if hasattr(self.board, "set_condition"):            # the simulator follows the condition
            if phase == "block_start":
                self.board.set_condition(ev.get("condition"))
            elif phase in ("done", "stopped"):
                self.board.set_condition(None)
            elif phase == "rating_due" and self.cfg.get("sim", {}).get("auto_rate", True):
                # simulator only: the "participant" rates the block the way the induced state
                # predicts, so an unattended --sim run reaches a verdict (marked simulated)
                threading.Timer(3.0, self._sim_rate, args=(ev.get("block_index"), ev.get("condition"))).start()
        self.broadcast("protocol", ev)

    SIM_RATINGS = {"boredom": (2.5, 1, 0.25), "flow": (6.0, 2, 0.85), "overload": (3.0, 3, 0.30), "work": (4.5, 2, 0.55)}

    def _sim_rate(self, block_index, condition) -> None:
        if self.protocol is None or self.protocol.pending_rating != block_index:
            return
        import random
        core, demand, overall = self.SIM_RATINGS.get(condition or "", (4.0, 2, 0.5))
        items = {k: float(min(7, max(1, round(core + random.uniform(-0.6, 0.6)))))
                 for k in ("challenge", "fluent", "time", "concentrate", "clear", "absorbed", "automatic", "know", "control", "lost")}
        items["demand"] = float(demand)
        items["flow_overall"] = round(min(1.0, max(0.0, overall + random.uniform(-0.05, 0.05))), 2)
        self.protocol.rate(int(block_index), items, "simulated rating (sim.auto_rate)")
        self.log(f"[sim] auto-rated block {block_index} ({condition})")

    def on_protocol_mark(self, kind: str, value, note: str = "") -> None:
        m = self.pipeline.mark(kind, value, note)
        self.broadcast("mark", m)

    def hello(self) -> dict:
        return {"server_version": SERVER_VERSION, **self.pipeline.hello(), "history": self.pipeline.history()}

    def sessions(self) -> list[dict]:
        out = []
        if not self.sessions_dir.exists():
            return out
        for d in sorted(self.sessions_dir.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            def count(name):
                p = d / name
                if not p.exists():
                    return 0
                with open(p, "rb") as f:
                    return max(0, sum(1 for _ in f) - 1)
            out.append({"name": d.name, "dir": str(d), "epochs": count("epochs.csv"), "band": count("band.csv"),
                        "marks": count("marks.csv"), "has_report": (d / "report.md").exists()})
        return out

    def report(self, name: str) -> dict:
        if not re.match(r"^[A-Za-z0-9_.-]+$", name):
            raise ValueError("bad session name")
        d = self.sessions_dir / name
        if not d.is_dir():
            raise ValueError(f"no session {name}")
        import report  # local module, written alongside
        result = report.run(d)
        md = (d / "report.md").read_text("utf-8") if (d / "report.md").exists() else report.render_markdown(result)
        return {"markdown": md, "summary": result.get("summary", result)}

    # --- commands -----------------------------------------------------------------
    def command(self, body: dict) -> dict:
        action = body.get("action")
        p = self.pipeline
        if action == "start":
            self.board.start()
            return {"streaming": True}
        if action == "stop":
            self.board.stop()
            return {"streaming": False}
        if action == "record_start":
            label = str(body.get("label") or "")
            if not SAFE_LABEL.match(label):
                raise ValueError("label: letters, digits, _ and - only (max 32)")
            d = self.recorder.start(label)
            self.log(f"[rec] recording to {d}")
            return {"dir": str(d)}
        if action == "record_stop":
            d = self.recorder.stop()
            self.log(f"[rec] stopped ({d})")
            return {"dir": str(d) if d else None}
        if action == "mark":
            kind = str(body.get("kind") or "note")
            if kind not in ("felt_flow", "flow_start", "flow_end", "distracted", "note"):
                raise ValueError("unknown mark kind")
            value = body.get("value")
            if kind == "felt_flow":
                value = float(value)
                if not 0 <= value <= 1:
                    raise ValueError("felt_flow value must be 0..1")
            m = p.mark(kind, value, str(body.get("note") or "")[:200])
            self.broadcast("mark", m)
            return {"mark": m}
        if action == "set_filter":
            notch = float(body.get("notch", p.filter_cfg["notch"]))
            lo = float(body.get("bp_lo", p.filter_cfg["bp_lo"]))
            hi = float(body.get("bp_hi", p.filter_cfg["bp_hi"]))
            if notch not in (0, 50, 60) or not (0 <= lo < hi <= p.fs / 2):
                raise ValueError("filter: notch 0/50/60, 0 <= lo < hi <= fs/2")
            p.set_filter(notch, lo, hi)
            return {"filter": p.filter_cfg}
        if action == "set_channel":
            p.set_channel(int(body.get("index", -1)), bool(body.get("enabled", True)))
            return {"enabled": p.enabled}
        if action == "set_flow_weights":
            p.set_flow_weights(float(body.get("theta", 0.5)), float(body.get("alpha", 0.25)), float(body.get("beta", 0.25)))
            return {"flow_weights": p.flow_weights}
        if action == "set_band_source":
            src = str(body.get("source") or "none")
            self.bands.switch(src, user_id=body.get("user_id"), band=body.get("band"),
                              address=body.get("address"), port=body.get("port"))
            return {"band": self.bands.status()}
        if action in ("protocol_start", "protocol_stop", "protocol_next", "rating", "trial"):
            if self.protocol is None:
                raise ValueError("no protocol runner")
            if action == "protocol_start":
                name = str(body.get("name") or "")
                seed = body.get("seed")
                if not self.recorder.active:
                    self.recorder.start(re.sub(r"[^A-Za-z0-9_-]", "", name)[:32])
                    self.log(f"[rec] recording to {self.recorder.dir} (protocol {name})")
                return {"protocol": self.protocol.start(name, None if seed in (None, "") else int(seed))}
            if action == "protocol_stop":
                return {"protocol": self.protocol.stop()}
            if action == "protocol_next":
                return {"protocol": self.protocol.next()}
            if action == "rating":
                items = body.get("items") or {}
                if not isinstance(items, dict):
                    raise ValueError("items must be an object")
                clean = {}
                for k, v in items.items():
                    k = str(k)[:24]
                    try:
                        clean[k] = float(v)
                    except (TypeError, ValueError):
                        continue
                rec = self.protocol.rate(int(body.get("block_index", -1)), clean, str(body.get("note") or "")[:300])
                return {"rating": rec}
            if action == "trial":
                rec = {k: body.get(k) for k in ("block_index", "task", "level", "problem", "answer", "response", "correct", "rt_ms", "t")}
                rec["t"] = float(rec["t"]) if rec.get("t") else time.time()
                blk = self.protocol.current()
                rec["condition"] = blk["condition"] if blk else None
                p.trial(rec)
                return {"ok": True}
        if action == "set_labels":
            labels = p.set_labels(list(body.get("labels") or []))
            note = str(body.get("note") or "")[:200]
            m = p.mark("note", "labels " + " ".join(labels), note)
            self.broadcast("mark", m)
            self.broadcast("labels", {"labels": labels, "positions": p.hello()["positions"]})
            self.log(f"[labels] {' '.join(labels)}" + (f" ({note})" if note else ""))
            return {"labels": labels}
        if action in ("autocal_start", "autocal_phase", "autocal_solve", "autocal_stop", "autocal_sim"):
            import autocal
            if action == "autocal_start":
                p.autocal = autocal.Capture(p.fs, p.n_ch)
                return {"autocal": p.autocal.status()}
            if action == "autocal_sim":
                # synthetic run for the --sim board: a known hidden placement to recover
                truth = list(body.get("truth") or ["AF7", "Fp1", "Fp2", "AF8", "F7", "F3", "F4", "F8"])[: p.n_ch]
                p.autocal = autocal.capture_from_synthetic(truth, fs=p.fs, seed=int(body.get("seed") or 1))
                return {"autocal": p.autocal.status(), "truth": truth}
            if p.autocal is None:
                raise ValueError("no calibration running (autocal_start first)")
            if action == "autocal_phase":
                p.autocal.set_phase(body.get("phase"))
                return {"autocal": p.autocal.status()}
            if action == "autocal_stop":
                p.autocal = None
                return {"autocal": None}
            feat = autocal.features(p.autocal)
            slots = body.get("slots") or (self.cfg.get("autocal") or {}).get("slots") or None
            res = autocal.solve(feat, slots=slots)
            res["capture"] = p.autocal.status()
            res["current_labels"] = list(p.labels)
            self.broadcast("autocal", res)
            return {"autocal": res}
        if action in ("astra_live", "astra_analyze", "astra_ask", "astra_check", "astra_clear",
                      "astra_fix", "astra_revert", "astra_restart"):
            if self.astra is None:
                raise ValueError("Astra is not wired (start without --astra off)")
            return self.astra.command(action, body)
        raise ValueError(f"unknown action {action!r}")


class Handler(BaseHTTPRequestHandler):
    app: App = None  # set by serve()
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet
        pass

    # --- helpers --------------------------------------------------------------
    def _json(self, obj, code: int = 200) -> None:
        body = json.dumps(obj, allow_nan=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path) -> None:
        try:
            path = path.resolve()
            if STATIC.resolve() not in path.parents and path != STATIC.resolve():
                raise FileNotFoundError
            data = path.read_bytes()
        except (FileNotFoundError, OSError):
            self._json({"ok": False, "error": "not found"}, 404)
            return
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    # --- routes ---------------------------------------------------------------
    def do_GET(self):
        u = urlparse(self.path)
        app = self.app
        try:
            if u.path == "/":
                return self._file(STATIC / "index.html")
            if u.path.startswith("/static/"):
                return self._file(STATIC / u.path[len("/static/"):])
            if u.path == "/events":
                return self._events()
            if u.path == "/api/state":
                return self._json({"status": app.status(), "analysis": app.pipeline.last_analysis})
            if u.path == "/api/history":
                return self._json(app.pipeline.history())
            if u.path == "/api/wearers":
                return self._json({"wearers": app.bands.wearers()})
            if u.path == "/api/protocol":
                return self._json(app.protocol.status() if app.protocol else {"active": False, "available": []})
            if u.path == "/api/sessions":
                return self._json({"sessions": app.sessions()})
            if u.path == "/api/band_console":
                n = int(parse_qs(u.query).get("n", ["60"])[0])
                return self._json({"lines": app.bands.console_tail(max(1, min(n, 300))), "vitals": app.pipeline.vitals(600)})
            if u.path == "/api/astra":
                return self._json(app.astra.state() if app.astra else {"available": False, "notes": [], "setup_transcript": [], "fixes": []})
            if u.path == "/api/report":
                name = parse_qs(u.query).get("session", [""])[0]
                return self._json({"ok": True, **app.report(name)})
            self._json({"ok": False, "error": "not found"}, 404)
        except ValueError as e:
            self._json({"ok": False, "error": str(e)}, 400)
        except Exception as e:  # noqa: BLE001
            app.log(f"[http] {self.path}: {e!r}\n{traceback.format_exc()}")
            self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        app = self.app
        try:
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b"{}"
            body = json.loads(raw.decode("utf-8") or "{}")
            if u.path == "/api/cmd":
                return self._json({"ok": True, **app.command(body)})
            self._json({"ok": False, "error": "not found"}, 404)
        except ValueError as e:
            self._json({"ok": False, "error": str(e)}, 400)
        except Exception as e:  # noqa: BLE001
            app.log(f"[http] {self.path}: {e!r}\n{traceback.format_exc()}")
            self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)

    def _events(self):
        app = self.app
        from collections import deque
        q: deque = deque(maxlen=400)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        hello = f"event: hello\ndata: {json.dumps(app.hello(), separators=(',', ':'), allow_nan=False)}\n\n".encode("utf-8")
        self.wfile.write(hello)
        self.wfile.flush()
        with app.sub_lock:
            app.subscribers.append(q)
        try:
            last = time.time()
            while True:
                if q:
                    while q:
                        self.wfile.write(q.popleft())
                    self.wfile.flush()
                    last = time.time()
                else:
                    time.sleep(0.05)
                    if time.time() - last > 15:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        last = time.time()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            with app.sub_lock:
                if q in app.subscribers:
                    app.subscribers.remove(q)


def serve(app: App, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    Handler.app = app
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    t = threading.Thread(target=httpd.serve_forever, name="http", daemon=True)
    t.start()
    return httpd
