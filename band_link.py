"""Live BioProxal band telemetry over USB.

The prod firmware locks its USB console (docs/SECURITY.md in the firmware repo): nothing
health-bearing is printed until the host proves the band's key. This module reuses the
firmware repo's own tools (tools/offload.py + tools/bpx_keys.py, path in config.json
"bioproxal_tools_dir", or found by walking up to <...>/worktrees/v0629/tools) to open the
port without resetting the band, unlock it with this laptop's master key
(%USERPROFILE%\\.bioproxal\\master.key), keep the session alive, and read every line the
band prints. Two line kinds carry data:

  the 1-s status line (always, phone or not):
    [    56s] mode=LOW  phase=1 | HR= 97.9 bpm  SpO2= 89.4%  RMSSD=123.6 ms  SDNN=170.2 ms
    PI=1.16%  RR=15.0 br/min* | anom=0.000 risk=0.000 flow=0.00 baseHR= 79.1 flowraw=0.29
    flow_open5=0.55 flow_end5=0.40
  the 2-s publish line (only while a BLE client is connected):
    [pub] live  hr= 71.2 flow=0.62 anom=0.120 mlp=0.080 zrms=0.300 [fcal=...] [fhi=..] [settling]

plus `[vbat] battery 3127 mV ...`, sleep lines (`OFF-WRIST SLEEP`, `USELESS-CONTACT SLEEP`)
and `warming up`. The band's raw red/IR PPG is not available from a production image; what
this reads is every derived value the band computes, once a second.

Read-only: no baseline reset, no log clear, no flash. Closing the port sends LOCK.
With --sim the "band" is a noisy copy of the simulator's hidden flow state.
"""
from __future__ import annotations

import os
import re
import struct
import sys
import threading
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
NUM = r"(-?\d+(?:\.\d+)?)"
PUB_RE = re.compile(
    r"\[pub\]\s+(HELD|live)\s+hr=\s*([-\d.]+)\s+flow=\s*([-\d.]+)\s+anom=\s*([-\d.]+)"
    r"(?:\s+mlp=\s*([-\d.]+))?(?:\s+zrms=\s*([-\d.]+))?(?P<rest>.*)")
STATUS_HEAD_RE = re.compile(r"^\[\s*(\d+)s\]\s+mode=(\w+)\s+phase=(\d+)")
FIELD_RES = {
    "hr": re.compile(r"\bHR=\s*" + NUM), "spo2": re.compile(r"SpO2=\s*" + NUM),
    "rmssd": re.compile(r"RMSSD=\s*" + NUM), "sdnn": re.compile(r"SDNN=\s*" + NUM),
    "pi": re.compile(r"\bPI=\s*" + NUM), "rr": re.compile(r"\bRR=\s*" + NUM),
    "anom": re.compile(r"anom=\s*" + NUM), "risk": re.compile(r"risk=\s*" + NUM),
    "flow": re.compile(r"\bflow=\s*" + NUM), "base_hr": re.compile(r"baseHR=\s*" + NUM),
    "flow_raw": re.compile(r"flowraw=\s*" + NUM), "flow_open5": re.compile(r"flow_open5=\s*" + NUM),
    "flow_end5": re.compile(r"flow_end5=\s*" + NUM),
}
VBAT_RE = re.compile(r"\[vbat\]\s+battery\s+(\d+)\s*mV")
CONSOLE_KEEP = 300


def find_tools_dir(configured: str | None, start: Path = HERE) -> Path | None:
    """The firmware tools folder: the configured path (absolute, or relative to the project),
    else the first <ancestor>/worktrees/v0629/tools (or any worktrees/*/tools) with offload.py."""
    if configured:
        p = Path(configured).expanduser()
        if not p.is_absolute():
            p = (start / p).resolve()
        if (p / "offload.py").exists():
            return p
    for d in [start, *start.parents]:
        cand = d / "worktrees" / "v0629" / "tools"
        if (cand / "offload.py").exists():
            return cand
        wt = d / "worktrees"
        if wt.is_dir():
            for sub in sorted(wt.iterdir(), reverse=True):
                if (sub / "tools" / "offload.py").exists():
                    return sub / "tools"
    return None


def _grab(rx: re.Pattern, text: str):
    m = rx.search(text)
    return float(m.group(1)) if m else None


def parse_status_line(line: str, t: float | None = None) -> dict | None:
    """The band's 1-s status line -> a sample dict, or None for any other line.
    `flow` is the PUBLISHED reading (None while the band shows no reading: flowraw < 0, or
    0.00 while settling); `flow_raw` is the window score before the display rules."""
    m = STATUS_HEAD_RE.search(line)
    if not m:
        return None
    t = time.time() if t is None else t
    uptime_s, mode, phase = int(m.group(1)), m.group(2).upper(), int(m.group(3))
    s = {"t": t, "uptime_s": uptime_s, "mode": mode, "phase": phase, "source_line": "status",
         "asleep": "SLEEP" in line.upper(), "warming": "warming up" in line}
    for k, rx in FIELD_RES.items():
        s[k] = _grab(rx, line)
    s["rr_modeled"] = bool(re.search(r"br/min\*", line))
    flow, raw = s.get("flow"), s.get("flow_raw")
    s["held"] = mode == "HIGH"
    s["settling"] = bool(raw is not None and raw >= 0 and (flow is None or flow <= 0) and mode == "LOW")
    if flow is None or flow <= 0 or (raw is not None and raw < 0):
        s["flow"] = None
    s["hr"] = s["hr"] if s.get("hr") else None
    s["poor_contact"] = bool(s.get("pi") is not None and s["pi"] < 0.3 and not s["hr"])
    return s


def parse_pub_line(line: str, t: float | None = None) -> dict | None:
    m = PUB_RE.search(line)
    if not m:
        return None
    rest = m.group("rest") or ""
    s = {"t": time.time() if t is None else t, "source_line": "pub",
         "held": m.group(1) == "HELD", "hr": float(m.group(2)), "flow": float(m.group(3)),
         "anom": float(m.group(4)), "mlp": float(m.group(5)) if m.group(5) else None,
         "zrms": float(m.group(6)) if m.group(6) else None, "settling": "settling" in rest,
         "fhi": None, "fcal": None}
    mh = re.search(r"fhi=\s*([-\d.]+)", rest)
    if mh:
        s["fhi"] = float(mh.group(1))
    mc = re.search(r"fcal=(\S+)", rest)
    if mc:
        s["fcal"] = mc.group(1)
    return s


class BandLink:
    """USB link: unlocks the band, reads its console, turns status / pub lines into samples
    (listeners get every sample; `console` keeps the last lines for the UI and the recorder)."""

    def __init__(self, tools_dir: str | Path | None, port: str | None = None, log=print,
                 keepalive_s: float = 60.0, on_line=None):
        self.tools_dir = find_tools_dir(str(tools_dir) if tools_dir else None)
        self.port = port
        self.log = log
        self.keepalive_s = keepalive_s
        self.on_line = on_line            # callable(t, line) for every console line (recorder)
        self.state = {"connected": False, "port": None, "unit": None, "mac": None,
                      "error": None, "last_line_age_s": None, "lines": 0, "battery_mv": None,
                      "transport": "usb"}
        self.latest = None            # newest sample
        self.listeners = []           # callables(sample dict)
        self.console: deque = deque(maxlen=CONSOLE_KEEP)   # (t, line)
        self._stop = threading.Event()
        self._thread = None
        self._offload = None
        self._last_line_t = 0.0
        self._last_sample_t = 0.0

    # --- firmware tool import -----------------------------------------------------
    def _import_tools(self):
        if self._offload is not None:
            return self._offload
        if not self.tools_dir or not (self.tools_dir / "offload.py").exists():
            raise RuntimeError("firmware tools not found (set usb.bioproxal_tools_dir in config.json to "
                               "<firmware repo>/tools, or keep a worktrees/v0629/tools above this folder)")
        key = Path(os.environ.get("USERPROFILE", "~")).expanduser() / ".bioproxal" / "master.key"
        if not key.exists():
            raise RuntimeError(f"no USB master key at {key}: the band's console stays locked")
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
                self.state.update({"connected": True, "port": port, "error": None})
                self.log(f"[band] {port} unlocked, identifying...")
                try:
                    # short marker timeout: the band prints a status line every second and
                    # identify must not hold the link for the default 30 s
                    info = offload.parse_identify(offload.request(ser, "I", marker_timeout=5.0)) if hasattr(offload, "request") else {}
                except Exception as e:  # noqa: BLE001
                    self.log(f"[band] identify skipped: {e}")
                    info = {}
                self.state.update({"unit": info.get("unit") or info.get("name") or info.get("device"),
                                   "mac": info.get("mac"), "build": info.get("build"), "fw": info.get("fw") or info.get("version")})
                self.log(f"[band] {port} ready" + (f" ({self.state['unit']}, {self.state.get('build')})" if self.state['unit'] else ""))
                lines = offload.BandLines(ser, keepalive_s=self.keepalive_s)
                backoff = 2.0
                try:
                    while not self._stop.is_set():
                        raw = lines.readline()
                        if not raw:
                            if self._last_line_t and time.time() - self._last_line_t > 30:
                                raise RuntimeError("no console line for 30 s (band asleep / off-wrist / USB suspended?)")
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
        if not line:
            return
        t = time.time()
        self._last_line_t = t
        self.state["lines"] += 1
        self.console.append((t, line))
        if self.on_line is not None:
            try:
                self.on_line(t, line)
            except Exception:  # noqa: BLE001
                pass
        mv = VBAT_RE.search(line)
        if mv:
            self.state["battery_mv"] = int(mv.group(1))
            return
        s = parse_status_line(line, t) or parse_pub_line(line, t)
        if s is None:
            return
        if s["source_line"] == "pub" and self._last_sample_t and t - self._last_sample_t < 0.5:
            return                        # the status line already covered this second
        s["battery_mv"] = self.state.get("battery_mv")
        self.latest = s
        self._last_sample_t = t
        for fn in list(self.listeners):
            try:
                fn(s)
            except Exception as e:  # noqa: BLE001
                self.log(f"[band] listener error: {e!r}")

    def status(self) -> dict:
        st = dict(self.state)
        st["last_line_age_s"] = round(time.time() - self._last_line_t, 1) if self._last_line_t else None
        return st

    def console_tail(self, n: int = 40) -> list:
        return [[round(t, 3), ln] for (t, ln) in list(self.console)[-n:]]


def find_tools_dir_v2(configured: str | None, start: Path = HERE) -> Path | None:
    """The firmware_v2 tools folder (live_usb.py + its offload.py): configured, else
    <ancestor>/firmware_v2/.claude/worktrees/*/tools or <ancestor>/firmware_v2/tools."""
    if configured:
        p = Path(configured).expanduser()
        if not p.is_absolute():
            p = (start / p).resolve()
        if (p / "live_usb.py").exists():
            return p
    for d in [start, *start.parents]:
        fw = d / "firmware_v2"
        if not fw.is_dir():
            continue
        cands = [fw / "tools"]
        wt = fw / ".claude" / "worktrees"
        if wt.is_dir():
            cands = [sub / "tools" for sub in sorted(wt.iterdir(), reverse=True)] + cands
        for c in cands:
            if (c / "live_usb.py").exists() and (c / "offload.py").exists():
                return c
    return None


def _load_module(path: Path, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod


DATALOG_ROW = struct.Struct("<IhhhhhhHHHHBBBBHBB")      # firmware_v2 DatalogRow, 32 B (tools/live_usb.py)


def decode_v2_row(row: bytes, t: float | None = None) -> dict:
    """One firmware_v2 datalog row -> the same sample shape the status line gives."""
    (up, hr, spo2, sdnn, rmssd, resp, pi, an, rk, fl, mv, mode, phase, flags, qual, dcq, led, cfg) = DATALOG_ROW.unpack(row[:32])
    no_flow = bool(flags & 0x20)
    hrv_ok = bool(flags & 0x01)
    mode_s = "HIGH" if mode == 1 else "LOW"
    # a 0.0000 published while HRV is not yet valid (cold start, first minutes on the wrist) is
    # "not computed", not a low flow: the dashboard shows "no reading" and nothing is paired
    no_reading = None
    if no_flow:
        no_reading = "band reports no flow reading"
    elif fl == 0 and (not hrv_ok or sdnn == 0):
        no_reading = "flow not computed yet (HRV not valid)"
    s = {"t": time.time() if t is None else t, "uptime_s": up, "mode": mode_s, "phase": phase, "source_line": "v2row",
         "hr": hr / 10 if hr > 0 else None, "spo2": spo2 / 10 if spo2 > 0 else None, "sdnn": sdnn / 10, "rmssd": rmssd / 10,
         "rr": resp / 10, "pi": pi / 100, "anom": an / 10000, "risk": rk / 10000,
         "flow": None if no_reading else fl / 10000, "flow_raw": fl / 10000, "base_hr": None, "no_reading": no_reading,
         "battery_mv": mv, "held": mode == 1, "asleep": bool(flags & 0x04), "poor_contact": bool(flags & 0x08),
         "settling": False, "hrv_ok": hrv_ok, "quality": qual, "ir_dc": dcq * 8, "led": led & 0x7F,
         "rate_hz": 100 if (cfg >> 5) & 1 else 25, "flow_open5": None, "flow_end5": None}
    return s


class BandLinkV2:
    """USB link to a firmware_v2 band: authenticates with the firmware_v2 tools, then polls
    the newest datalog row ('T' frame, n = 1) every `poll_s`; the band writes a row every 2 s.
    Same read-only promise: nothing is cleared, flashed or reset."""

    def __init__(self, tools_dir: str | Path | None, port: str | None = None, log=print, poll_s: float = 2.0, on_line=None):
        self.tools_dir = find_tools_dir_v2(str(tools_dir) if tools_dir else None)
        self.port, self.log, self.poll_s, self.on_line = port, log, float(poll_s), on_line
        self.state = {"connected": False, "port": None, "unit": None, "mac": None, "error": None,
                      "last_line_age_s": None, "lines": 0, "battery_mv": None, "transport": "usb2"}
        self.latest = None
        self.listeners = []
        self.console: deque = deque(maxlen=CONSOLE_KEEP)
        self._stop = threading.Event()
        self._thread = None
        self._mod = None
        self._last_t = 0.0
        self._last_up = -1

    def _tools(self):
        if self._mod is None:
            if not self.tools_dir:
                raise RuntimeError("firmware_v2 tools not found (set usb.firmware_v2_tools_dir to <firmware_v2>/tools)")
            self._mod = _load_module(self.tools_dir / "offload.py", "bpx2_offload")
        return self._mod

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="band-v2", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(3.0)

    def _run(self) -> None:
        backoff = 2.0
        while not self._stop.is_set():
            ser = None
            try:
                O = self._tools()
                port = self.port
                if port is None:
                    from serial.tools import list_ports
                    ports = [p.device for p in list_ports.comports() if (p.vid or 0) == 0x303A]
                    if not ports:
                        raise RuntimeError("no BioProxal band on USB (ESP32-S3 port not found)")
                    port = ports[0]
                ser = O.open_port(port)
                st = O.authenticate(ser, quiet=True)
                self.state.update({"connected": True, "port": port, "error": None, "mac": st.get("mac"),
                                   "unit": f"band {st.get('mac', '')[-5:]}" if st.get("mac") else "band"})
                self.log(f"[band] {port} firmware_v2 session open ({self.state['unit']}), polling the datalog every {self.poll_s:.0f} s")
                backoff = 2.0
                while not self._stop.is_set():
                    try:
                        payload = O.request(ser, "T", struct.pack("<H", 1), marker_timeout=5.0)
                    except Exception as e:  # noqa: BLE001
                        if "locked" in str(e).lower():
                            O.authenticate(ser, quiet=True)
                            continue
                        raise
                    if len(payload) >= 32:
                        s = decode_v2_row(payload[-32:])
                        self._last_t = s["t"]
                        if s["uptime_s"] != self._last_up:
                            self._last_up = s["uptime_s"]
                            self.state["lines"] += 1
                            self.state["battery_mv"] = s["battery_mv"]
                            line = (f"[v2 {s['uptime_s']}s] mode={s['mode']} phase={s['phase']} HR={s['hr']} SpO2={s['spo2']} "
                                    f"RMSSD={s['rmssd']} SDNN={s['sdnn']} PI={s['pi']} RR={s['rr']} anom={s['anom']} risk={s['risk']} "
                                    f"flow={s['flow']} batt={s['battery_mv']} ir_dc={s['ir_dc']} led={s['led']}")
                            self.console.append((s["t"], line))
                            if self.on_line is not None:
                                try:
                                    self.on_line(s["t"], line)
                                except Exception:  # noqa: BLE001
                                    pass
                            self.latest = s
                            for fn in list(self.listeners):
                                try:
                                    fn(s)
                                except Exception as e:  # noqa: BLE001
                                    self.log(f"[band] listener error: {e!r}")
                    self._stop.wait(self.poll_s)
            except Exception as e:  # noqa: BLE001
                self.state.update({"connected": False, "error": f"{type(e).__name__}: {e}"})
                self.log(f"[band] {e}")
                self._stop.wait(backoff)
                backoff = min(backoff * 1.5, 20.0)
            finally:
                if ser is not None:
                    try:
                        ser.close()
                    except Exception:  # noqa: BLE001
                        pass
        self.state["connected"] = False

    def status(self) -> dict:
        st = dict(self.state)
        st["last_line_age_s"] = round(time.time() - self._last_t, 1) if self._last_t else None
        return st

    def console_tail(self, n: int = 40) -> list:
        return [[round(t, 3), ln] for (t, ln) in list(self.console)[-n:]]


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
            hr = 70 + 10 * f + float(self._rng.normal(0, 1))
            s = {"t": time.time(), "held": False, "hr": hr, "flow": round(f, 2), "anom": 0.1, "mlp": None,
                 "zrms": None, "settling": False, "fhi": None, "fcal": None, "flow_raw": round(f, 2),
                 "spo2": round(97 + float(self._rng.normal(0, 0.5)), 1), "pi": round(1.5 + 0.5 * f, 2),
                 "rmssd": round(40 + 20 * f, 1), "sdnn": round(50 + 20 * f, 1), "rr": 15.0,
                 "mode": "LOW", "phase": 1, "battery_mv": 3900, "source_line": "sim"}
            self.latest = s
            self._last_t = s["t"]
            self.state["lines"] += 1
            for fn in list(self.listeners):
                fn(s)

    def status(self) -> dict:
        st = dict(self.state)
        st["last_line_age_s"] = round(time.time() - self._last_t, 1) if self._last_t else None
        return st

    def console_tail(self, n: int = 40) -> list:
        return []
