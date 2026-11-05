"""OpenBCI x BioProxal flow validation - entry point.

    python app.py                  # auto-detect the OpenBCI dongle, band flow from Supabase
    python app.py --sim            # no hardware: synthetic helmet + synthetic band
    python app.py --band ble       # band straight to this laptop over Bluetooth (close the phone app)
    python app.py --band usb       # band on a USB cable (firmware tools + master key)
    python app.py --source udp     # mirror the real OpenBCI GUI's Networking widget (UDP 12345)

Then open http://127.0.0.1:8765 (opens by itself unless --no-browser).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from cyton import CytonBoard, SimBoard, find_dongle, list_candidate_ports  # noqa: E402
from pipeline import Pipeline  # noqa: E402
from recorder import Recorder  # noqa: E402
from server import App, serve  # noqa: E402


from collections import deque  # noqa: E402

LOG_RING: deque = deque(maxlen=400)      # the last log lines, for Astra's fixer context


def log(msg: str) -> None:
    line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
    LOG_RING.append(line)
    print(line, flush=True)


def load_config(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text("utf-8"))
        except ValueError as e:
            log(f"[config] {path} is not valid JSON ({e}); using defaults")
    return {}


class BandManager:
    """Owns the one active band link (supabase / ble / usb / sim / none) and swaps it."""

    def __init__(self, cfg: dict, on_sample, sim_board=None, on_line=None):
        self.cfg, self.on_sample, self.sim_board, self.on_line = cfg, on_sample, sim_board, on_line
        self.source = "none"
        self.link = None
        self._lock = threading.Lock()
        self._sb_client = None

    def switch(self, source: str, user_id=None, band=None, address=None, port=None) -> None:
        with self._lock:
            if self.link is not None:
                try:
                    self.link.stop()
                except Exception:  # noqa: BLE001
                    pass
                self.link = None
            self.source = source
            sb = self.cfg.get("supabase", {})
            if source == "supabase":
                from supabase_link import SupabaseBandLink
                self.link = SupabaseBandLink(user_id=user_id or sb.get("user_id"), band=band or sb.get("band"),
                                             poll_s=float(sb.get("poll_s", 5)), backfill_min=float(sb.get("backfill_min", 30)), log=log)
            elif source == "ble":
                from band_ble import BleBandLink
                b = self.cfg.get("ble", {})
                self.link = BleBandLink(address=address or b.get("address"), name_prefix=b.get("name_prefix", "BioProxal"), log=log)
            elif source == "usb":
                from band_link import BandLink
                u = self.cfg.get("usb", {})
                self.link = BandLink(u.get("bioproxal_tools_dir"), port=port or u.get("port"), log=log,
                                     on_line=self.on_line)
                if self.link.tools_dir:
                    log(f"[band] firmware tools: {self.link.tools_dir}")
            elif source == "sim":
                from band_link import SimBandLink
                if self.sim_board is None:
                    from cyton import SimBoard as _SB
                    self.sim_board = _SB(8, log=lambda *_: None)
                    self.sim_board._t0 = time.time()
                self.link = SimBandLink(self.sim_board, log=log)
            elif source == "none":
                self.link = None
            else:
                raise ValueError(f"unknown band source {source!r}")
            if self.link is not None:
                self.link.listeners.append(self.on_sample)
                self.link.start()
            log(f"[band] source = {source}")

    def status(self) -> dict:
        st = {"source": self.source, "connected": False, "unit": None, "mac": None, "user_id_short": None,
              "error": None, "last_line_age_s": None, "lines": 0, "seen": []}
        if self.link is not None:
            try:
                st.update(self.link.status())
            except Exception as e:  # noqa: BLE001
                st["error"] = repr(e)
        st["source"] = self.source
        return st

    def console_tail(self, n: int = 40) -> list:
        link = self.link
        return link.console_tail(n) if link is not None and hasattr(link, "console_tail") else []

    def wearers(self) -> list[dict]:
        from supabase_link import SupabaseClient, load_env
        if self._sb_client is None:
            env = load_env()
            self._sb_client = SupabaseClient(env["url"], env["key"])
        return self._sb_client.wearers(24.0)


def open_board(args, cfg: dict):
    if args.sim:
        n = 16 if str(args.channels) == "16" else 8
        return SimBoard(n, log=log)
    source = args.source or cfg.get("source", "serial")
    if source == "udp":
        from openbci_udp import UdpBoard
        n = 16 if str(args.channels) == "16" else 8
        return UdpBoard(int(args.udp_port or cfg.get("udp_port", 12345)), n, log=log)
    port = args.board or cfg.get("board", "auto")
    if port in (None, "", "auto"):
        port = find_dongle()
        if port is None:
            ports = list_candidate_ports()
            log("[cyton] no OpenBCI dongle found. Serial ports present:")
            for p in ports:
                log(f"         {p['device']}: {p['description']}" + ("  (BioProxal band)" if p["band"] else ""))
            log("[cyton] plug the dongle in and restart, pass --board COMx, or use --sim")
            sys.exit(2)
    want_16 = {"8": False, "16": True}.get(str(args.channels or cfg.get("channels", "auto")))
    return CytonBoard(port, want_16=want_16, daisy_even_is_upper=bool(cfg.get("daisy_even_is_upper", True)), log=log)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="synthetic helmet + synthetic band, no hardware")
    ap.add_argument("--port", type=int, help="HTTP port (default from config.json, 8765)")
    ap.add_argument("--board", help="COM port of the OpenBCI dongle (default: auto-detect)")
    ap.add_argument("--channels", choices=["auto", "8", "16"], help="force 8 or 16 channels")
    ap.add_argument("--source", choices=["serial", "udp"], help="EEG source: the dongle, or the OpenBCI GUI's UDP stream")
    ap.add_argument("--udp-port", type=int, help="UDP port for --source udp (default 12345)")
    ap.add_argument("--band", choices=["supabase", "ble", "usb", "sim", "none"], help="where the band's flow comes from")
    ap.add_argument("--user-id", help="Supabase user_id (full uuid) to follow")
    ap.add_argument("--band-name", help="Supabase band name to follow, e.g. BioProxal4")
    ap.add_argument("--ble-address", help="band BLE address (default: first BioProxal* found)")
    ap.add_argument("--montage", choices=["default", "flow", "frontal"], default="default",
                    help="8-ch electrode labels: 'default' = Mark IV factory sites (Fp1 Fp2 C3 C4 P7 P8 O1 O2); "
                         "'flow' = the re-sited flow montage in config.json labels_8_flow (Fz F3 F4 Cz C3 C4 P3 P4); "
                         "'frontal' = all eight combs across the forehead band, labels_8_frontal (Fp1 Fp2 AF3 AF4 F3 F4 F7 F8)")
    ap.add_argument("--astra", choices=["on", "off", "live"], default="on",
                    help="Astra panels (OpenAI gpt-6-astra, key from BioProxal2026/.env): on = ready, "
                         "live = start the live analyst at once, off = no Astra")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-stream", action="store_true", help="do not start streaming automatically")
    ap.add_argument("--config", default=str(HERE / "config.json"))
    args = ap.parse_args(argv)

    cfg = load_config(Path(args.config))
    if args.montage in ("flow", "frontal") and cfg.get(f"labels_8_{args.montage}"):
        cfg["labels_8"] = cfg[f"labels_8_{args.montage}"]
        log(f"[montage] 8-ch {args.montage} montage: {' '.join(cfg['labels_8'])}")
    port = args.port or int(cfg.get("port", 8765))

    board = open_board(args, cfg)
    recorder = Recorder(HERE / "sessions", labels=[], meta={})
    pipeline = Pipeline(board, cfg, recorder=recorder, log=log)
    recorder.labels = pipeline.labels
    from server import SERVER_VERSION
    recorder.meta = {"board": board.status(), "config": cfg, "server_version": SERVER_VERSION}

    bands = BandManager(cfg, on_sample=lambda s: app.on_band(s), sim_board=board if args.sim else None,
                        on_line=lambda t, line: recorder.console(t, line))
    from protocol import ProtocolRunner, load_protocols
    runner = ProtocolRunner(load_protocols(HERE / "protocols.json"),
                            on_event=lambda e: app.on_protocol_event(e),
                            on_mark=lambda k, v, n="": app.on_protocol_mark(k, v, n), log=log)
    app = App(pipeline, board, recorder, bands, cfg, log=log, protocol=runner)
    held = {"httpd": None}          # filled after serve(); used by the restart and post-reload hooks

    def shutdown_all() -> None:
        try:
            recorder.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            board.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            if bands.link:
                bands.link.stop()
        except Exception:  # noqa: BLE001
            pass

    def restart_server() -> None:
        """Astra's in-flight fixer: start a fresh process with the same arguments (no new browser
        tab) once the listening socket is closed, then leave. The page reconnects by itself."""
        import subprocess
        log("[app] restarting")
        argv = [sys.executable] + sys.argv
        if "--no-browser" not in argv:
            argv.append("--no-browser")
        shutdown_all()
        if held["httpd"] is not None:
            try:
                held["httpd"].shutdown()
                held["httpd"].server_close()
            except Exception:  # noqa: BLE001
                pass
        subprocess.Popen(argv, cwd=str(HERE), close_fds=True)
        os._exit(0)

    def post_reload(name: str, info: dict) -> None:
        if name == "server" and held["httpd"] is not None:
            import server as _server
            _server.Handler.app = app
            held["httpd"].RequestHandlerClass = _server.Handler

    def protocol_reload() -> None:
        if runner.active:
            raise ValueError("protocol running; re-read after it ends")
        runner.protocols = load_protocols(HERE / "protocols.json")

    if args.astra != "off":
        from astra import Astra
        acfg = dict(cfg.get("astra") or {})
        if args.astra == "live":
            acfg["auto_live"] = True
        try:
            app.astra = Astra(pipeline, app.status, acfg, app.broadcast, log=log, recorder=recorder,
                              raw_peak_fn=lambda: pipeline.last_raw_peak,
                              log_tail_fn=lambda n: list(LOG_RING)[-n:],
                              roots_fn=lambda: [app, pipeline, board, recorder, bands, runner, app.astra],
                              restart_fn=restart_server, protocol_reload_fn=protocol_reload, post_reload_fn=post_reload)
        except Exception as e:  # noqa: BLE001
            log(f"[astra] disabled: {e}")
    band_source = args.band or ("sim" if args.sim else cfg.get("band_source", "supabase"))
    pipeline.set_band_source(band_source)
    try:
        bands.switch(band_source, user_id=args.user_id, band=args.band_name, address=args.ble_address)
    except Exception as e:  # noqa: BLE001
        log(f"[band] could not start source {band_source}: {e}")

    httpd = serve(app, "127.0.0.1", port)
    held["httpd"] = httpd
    url = f"http://127.0.0.1:{port}/"
    log(f"[http] dashboard at {url}")
    if not args.no_stream:
        board.start()
        log("[board] streaming")
    if not args.no_browser and cfg.get("open_browser", True):
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    stop = threading.Event()

    def on_sig(*_):
        stop.set()

    signal.signal(signal.SIGINT, on_sig)
    try:
        signal.signal(signal.SIGTERM, on_sig)
    except (AttributeError, ValueError):
        pass
    try:
        while not stop.is_set():
            stop.wait(0.5)
    finally:
        log("[app] shutting down")
        try:
            recorder.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            board.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            if bands.link:
                bands.link.stop()
        except Exception:  # noqa: BLE001
            pass
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
