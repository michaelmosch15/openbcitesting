"""BioProxal band over BLE, straight to this laptop (no phone, no USB cable).

The band advertises as "BioProxal<unit>" with service 5dc2a000-...; the STATE characteristic
(5dc2a001-...) notifies a 28/29-byte packed frame every 2 s (docs/BLE_GATT.md in the
firmware repo). This is exactly what the Amplifly app reads; nothing is written to the band.

The band allows ONE BLE connection: close the Amplifly app (or turn the phone's Bluetooth
off) while the laptop is connected, else the scan finds nothing / the connect fails.

Requires `pip install bleak` (Windows: uses the system Bluetooth adapter through WinRT).
"""
from __future__ import annotations

import asyncio
import struct
import threading
import time

SERVICE_UUID = "5dc2a000-c3a9-4d8f-9b3f-bf1e3e0a1001"
STATE_UUID = "5dc2a001-c3a9-4d8f-9b3f-bf1e3e0a1001"
NAME_PREFIX = "BioProxal"

WINDOW_STATES = {0: "unavailable", 1: "closed", 2: "opening", 3: "open", 4: "ending"}


def decode_state(data: bytes) -> dict:
    """Decode one STATE frame (28 or 29 bytes) into app-level fields."""
    if len(data) < 28:
        raise ValueError(f"STATE frame too short: {len(data)} bytes")
    version, mode, phase, flags = struct.unpack_from("<4B", data, 0)
    hr, spo2, sdnn, rmssd, rr, pi = struct.unpack_from("<6h", data, 4)
    anom, risk, batt = struct.unpack_from("<3H", data, 16)
    (up,) = struct.unpack_from("<I", data, 22)
    (flow_q,) = struct.unpack_from("<H", data, 26)
    learn = data[28] if len(data) >= 29 else None
    no_reading = bool(flags & 0x80)
    held_feat = bool(flags & 0x02)
    asleep = bool(flags & 0x04)
    poor = bool(flags & 0x08)
    win = (flags >> 4) & 0x07
    # "held" the way the GUI should draw it: a HIGH_RISK display hold, or LOW_RISK with
    # window state unavailable (last reading, not current) - BLE_GATT.md "No fake zero".
    held = (mode == 1 and not no_reading) or (mode == 0 and win == 0 and not no_reading)
    settling = mode == 0 and no_reading and hr > 0 and not poor
    return {
        "t": time.time(),
        "version": version, "mode": mode, "phase": phase, "flags": flags,
        "hr": hr / 10.0, "spo2": spo2 / 10.0, "sdnn": sdnn / 10.0, "rmssd": rmssd / 10.0,
        "resp": rr / 10.0, "pi": pi / 100.0,
        "anom": anom / 10000.0, "risk": risk / 10000.0, "battery_mv": batt, "uptime_s": up,
        "flow": None if no_reading else flow_q / 10000.0,
        "flow_no_reading": no_reading, "held": held, "held_features": held_feat,
        "asleep": asleep, "poor_contact": poor, "settling": settling,
        "window": WINDOW_STATES.get(win, str(win)),
        "learn_pct": (None if learn is None or (learn & 0x7F) > 100 else learn & 0x7F),
        "low_battery": learn is not None and (learn & 0x7F) == 126,
        "flow_agent_only": bool(learn & 0x80) if learn is not None else None,
        "fhi": None, "fcal": None, "mlp": None, "zrms": None,
    }


class BleBandLink:
    """Runs a bleak event loop in a background thread; auto-scans, connects, subscribes to
    STATE and reconnects forever. `latest` is the newest decoded frame; listeners get each."""

    def __init__(self, address: str | None = None, name_prefix: str = NAME_PREFIX, log=print):
        # accept a bare MAC or a "BioProxal4 (AA:BB:CC:DD:EE:FF)" entry from the scan list
        if address:
            import re
            m = re.search(r"([0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5})", address)
            address = m.group(1).upper() if m else address.strip()
        self.address = address or None
        self.name_prefix = name_prefix
        self.log = log
        self.latest = None
        self.listeners = []
        self.state = {"connected": False, "port": "ble", "unit": None, "mac": address,
                      "error": None, "lines": 0, "scanning": False, "seen": []}
        self._stop = threading.Event()
        self._thread = None
        self._last_t = 0.0
        self._loop = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._thread_main, name="band-ble", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(5.0)

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except Exception as e:  # noqa: BLE001
            self.state.update({"connected": False, "error": f"{type(e).__name__}: {e}"})
            self.log(f"[band-ble] {e}")

    async def _find(self):
        from bleak import BleakScanner
        self.state["scanning"] = True
        try:
            if self.address:
                return await BleakScanner.find_device_by_address(self.address, timeout=12.0)
            seen = []
            found = None
            devices = await BleakScanner.discover(timeout=8.0)
            for d in devices:
                if d.name:
                    seen.append(f"{d.name} ({d.address})")
                if d.name and d.name.startswith(self.name_prefix):
                    found = d
                    break
            self.state["seen"] = seen[:30]
            return found
        finally:
            self.state["scanning"] = False

    async def _run(self) -> None:
        from bleak import BleakClient
        backoff = 2.0
        while not self._stop.is_set():
            try:
                dev = await self._find()
                if dev is None:
                    raise RuntimeError(f"no '{self.name_prefix}*' band advertising - is it on, "
                                       f"worn, and NOT connected to the phone app?")
                self.log(f"[band-ble] found {dev.name} @ {dev.address}, connecting")
                disconnected = asyncio.Event()

                def on_disc(_c):
                    disconnected.set()

                async with BleakClient(dev, disconnected_callback=on_disc, timeout=20.0) as client:
                    self.state.update({"connected": True, "unit": dev.name, "mac": dev.address,
                                       "error": None})
                    backoff = 2.0
                    try:
                        first = await client.read_gatt_char(STATE_UUID)
                        self._on_state(None, first)
                    except Exception:  # noqa: BLE001
                        pass
                    await client.start_notify(STATE_UUID, self._on_state)
                    self.log(f"[band-ble] subscribed to STATE on {dev.name}")
                    while not self._stop.is_set() and not disconnected.is_set():
                        await asyncio.sleep(0.5)
                        if self._last_t and time.time() - self._last_t > 45:
                            raise RuntimeError("no STATE notify for 45 s (band asleep / off-wrist?)")
                    try:
                        await client.stop_notify(STATE_UUID)
                    except Exception:  # noqa: BLE001
                        pass
                if not self._stop.is_set():
                    raise RuntimeError("band disconnected")
            except Exception as e:  # noqa: BLE001
                self.state.update({"connected": False, "error": f"{type(e).__name__}: {e}"})
                self.log(f"[band-ble] {e}")
                try:
                    await asyncio.wait_for(asyncio.get_event_loop().run_in_executor(None, self._stop.wait, backoff),
                                           timeout=backoff + 1)
                except Exception:  # noqa: BLE001
                    pass
                backoff = min(backoff * 1.5, 20.0)
        self.state["connected"] = False

    def _on_state(self, _handle, data: bytearray) -> None:
        try:
            s = decode_state(bytes(data))
        except Exception as e:  # noqa: BLE001
            self.log(f"[band-ble] bad STATE frame: {e}")
            return
        self._last_t = s["t"]
        self.latest = s
        self.state["lines"] += 1
        for fn in list(self.listeners):
            try:
                fn(s)
            except Exception as e:  # noqa: BLE001
                self.log(f"[band-ble] listener error: {e!r}")

    def status(self) -> dict:
        st = dict(self.state)
        st["last_line_age_s"] = round(time.time() - self._last_t, 1) if self._last_t else None
        st["transport"] = "ble"
        return st
