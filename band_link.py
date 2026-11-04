"""Live BioProxal band flow score over USB.

The prod firmware locks its USB console (docs/SECURITY.md in the firmware repo): nothing
health-bearing is printed until the host proves the band's key. This module reuses the
firmware repo's own tools (tools/offload.py + tools/bpx_keys.py, path in config.json
"bioproxal_tools_dir") to open the port without resetting the band, unlock it with this
laptop's master key (%USERPROFILE%\\.bioproxal\\master.key), keep the session alive, and
then just reads the 2-s "[pub]" lines the band prints for every BLE publish:

    [pub] live  hr= 71.2 flow=0.62 anom=0.120 mlp=0.080 zrms=0.300 [fcal=...] [fhi=..] [settling]

Read-only: no baseline reset, no log clear, no flash. Closing the port sends LOCK.

If the firmware tools or the key are missing the link reports why and the rest of the app
runs EEG-only. With --sim the "band" is a noisy copy of the simulator's hidden flow state.
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time
from pathlib import Path

PUB_RE = re.compile(
    r"\[pub\]\s+(HELD|live)\s+hr=\s*([-\d.]+)\s+flow=\s*([-\d.]+)\s+anom=\s*([-\d.]+)"
    r"(?:\s+mlp=\s*([-\d.]+))?(?:\s+zrms=\s*([-\d.]+))?(?P<rest>.*)")


class BandLink:
    def __init__(self, tools_dir: str | Path | None, port: str | None = None, log=print,
                 keepalive_s: float = 60.0):
        self.tools_dir = Path(tools_dir).expanduser() if tools_dir else None
        self.port = port
        self.log = log
        self.keepalive_s = keepalive_s
        self.state = {"connected": False, "port": None, "unit": None, "mac": None,
                      "error": None, "last_line_age_s": None, "lines": 0}
        self.latest = None            # dict of the newest [pub] sample
        self.listeners = []           # callables(sample dict)
        self._stop = threading.Event()
        self._thread = None
        self._offload = None
        self._last_line_t = 0.0

    # --- firmware tool import -----------------------------------------------------
    def _import_tools(self):
        if self._offload is not None:
            return self._offload
        if not self.tools_dir or not (self.tools_dir / "offload.py").exists():
            raise RuntimeError(f"firmware tools not found at {self.tools_dir} "
                               f"(set bioproxal_tools_dir in config.json)")
        key = Path(os.environ.get("USERPROFILE", "~")).expanduser() / ".bioproxal" / "master.key"
        if not key.exists():
            raise RuntimeError(f"no USB master key at {key}: the band's [pub] lines stay locked")
        if str(self.tools_dir) not in sys.path:
            sys.path.insert(0, str(self.tools_dir))
        import offload  # noqa: WPS433  (firmware repo tool)
        self._offload = offload
        return offload

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="band", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(3.0)

    # --- main loop ----------------------------------------------------------------
    def _run(self) -> None:
        backoff = 2.0
        while not self._stop.is_set():
            try:
                offload = self._import_tools()
                port = self.port
                if port is None:
                    ports = offload.band_ports()
                    if not ports:
                        raise RuntimeError("no BioProxal band on USB (ESP32-S3 port not found)")
                    port = ports[0][0]
                ser = offload.open_band(port)             # unlocks; raises AuthError with a reason
                info = {}
                try:
                    info = offload.parse_identify(offload.request(ser, "I")) if hasattr(offload, "request") else {}
                except Exception:  # noqa: BLE001
                    info = {}
                self.state.update({"connected": True, "port": port, "error": None,
                                   "unit": info.get("unit") or info.get("name"),
                                   "mac": info.get("mac")})
                self.log(f"[band] {port} unlocked" + (f" ({self.state['unit']})" if self.state['unit'] else ""))
                lines = offload.BandLines(ser, keepalive_s=self.keepalive_s)
                backoff = 2.0
                try:
                    while not self._stop.is_set():
                        raw = lines.readline()
                        if not raw:
                            if self._last_line_t and time.time() - self._last_line_t > 30:
                                raise RuntimeError("no [pub] line for 30 s (band asleep / off-wrist / BLE suspended?)")
                            continue
                        self._handle_line(raw.decode("utf-8", "replace").strip())
                finally:
                    try:
                        ser.close()                         # offload wraps close() to send LOCK
                    except Exception:  # noqa: BLE001
                        pass
            except Exception as e:  # noqa: BLE001
                self.state.update({"connected": False, "error": f"{type(e).__name__}: {e}"})
                self.log(f"[band] {e}")
                self._stop.wait(backoff)
                backoff = min(backoff * 1.5, 20.0)
        self.state["connected"] = False

    def _handle_line(self, line: str) -> None:
        self._last_line_t = time.time()
        m = PUB_RE.search(line)
        if not m:
            return
        rest = m.group("rest") or ""
        s = {
            "t": time.time(),
            "held": m.group(1) == "HELD",
            "hr": float(m.group(2)),
            "flow": float(m.group(3)),
            "anom": float(m.group(4)),
            "mlp": float(m.group(5)) if m.group(5) else None,
            "zrms": float(m.group(6)) if m.group(6) else None,
            "settling": "settling" in rest,
            "fhi": None,
            "fcal": None,
        }
        mh = re.search(r"fhi=\s*([-\d.]+)", rest)
        if mh:
            s["fhi"] = float(mh.group(1))
        mc = re.search(r"fcal=(\S+)", rest)
        if mc:
            s["fcal"] = mc.group(1)
        self.latest = s
        self.state["lines"] += 1
        for fn in list(self.listeners):
            try:
                fn(s)
            except Exception as e:  # noqa: BLE001
                self.log(f"[band] listener error: {e!r}")

    def status(self) -> dict:
        st = dict(self.state)
        st["last_line_age_s"] = round(time.time() - self._last_line_t, 1) if self._last_line_t else None
        return st


class SimBandLink:
    """Stands in for the band with --sim: publishes the simulator's hidden flow (+ noise,
    + a 20 s lag, the way a wrist signal trails the brain) every 2 s."""

    def __init__(self, sim_board, log=print, lag_s: float = 20.0):
        self.sim = sim_board
        self.log = log
        self.lag_s = lag_s
        self.latest = None
        self.listeners = []
        self.state = {"connected": True, "port": "SIM", "unit": "BioProxalSIM", "mac": None,
                      "error": None, "lines": 0}
        self._stop = threading.Event()
        self._thread = None
        self._last_t = 0.0
        import numpy as np
        self._rng = np.random.default_rng(7)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="simband", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(2.0):
            f = self.sim.hidden_flow(time.time() - self.lag_s) + float(self._rng.normal(0, 0.06))
            f = min(1.0, max(0.0, f))
            s = {"t": time.time(), "held": False, "hr": 70 + 10 * f + float(self._rng.normal(0, 1)),
                 "flow": round(f, 2), "anom": 0.1, "mlp": None, "zrms": None, "settling": False,
                 "fhi": None, "fcal": None}
            self.latest = s
            self._last_t = s["t"]
            self.state["lines"] += 1
            for fn in list(self.listeners):
                fn(s)

    def status(self) -> dict:
        st = dict(self.state)
        st["last_line_age_s"] = round(time.time() - self._last_t, 1) if self._last_t else None
        return st
