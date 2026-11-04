"""OpenBCI Cyton (v3, 32-bit board) serial driver + a synthetic board for bench use.

Talks to the Cyton through its RFduino USB dongle (FTDI, VID 0x0403) with the plain
OpenBCI serial protocol - no BrainFlow, no native DLLs (Windows Smart App Control blocks
unsigned DLLs on this laptop), only pyserial + numpy.

Protocol (docs.openbci.com, "Cyton Data Format"):
  'v'  soft reset  -> text banner ending in "$$$" ("OpenBCI V3 8-16 channel", "On Daisy
                      ADS1299 Device ID" when a Daisy is attached)
  'c'/'C'  8 / 16 channel mode     'd'  default channel settings
  'b'/'s'  start / stop streaming
  Packet (33 bytes): A0 | sample# | 8 x int24 big-endian EEG | 3 x int16 accel | C0..CF
  Scale: 4.5 V / gain 24 / (2^23 - 1) per count  ->  0.02235 uV per count.
  Daisy: the board alternates packets; by default odd sample numbers carry channels 1-8
  (Cyton) and even ones channels 9-16 (Daisy); 125 Hz per channel. Flip
  `daisy_even_is_upper` in config.json if the labels look swapped on your headset.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np

try:
    import serial
    from serial.tools import list_ports
except ImportError:  # pragma: no cover
    serial = None
    list_ports = None

FTDI_VID = 0x0403
ESP32S3_VID = 0x303A          # the BioProxal band - never open it as a Cyton
ADS_GAIN = 24.0
UV_PER_COUNT = 4.5 / ADS_GAIN / (2 ** 23 - 1) * 1e6
ACCEL_G_PER_COUNT = 0.002 / 16
FULL_SCALE_UV = (2 ** 23 - 1) * UV_PER_COUNT      # ~187 500 uV
PACKET = 33


class SampleBuffer:
    """Thread-safe ring of the newest `seconds` of samples, [n, n_ch] float32 uV."""

    def __init__(self, n_ch: int, fs: float, seconds: float = 30.0):
        self.n_ch, self.fs = n_ch, fs
        self.size = int(fs * seconds)
        self.data = np.zeros((self.size, n_ch), np.float32)
        self.accel = np.zeros((self.size, 3), np.float32)
        self.t = np.zeros(self.size, np.float64)
        self.pos = 0
        self.count = 0           # total samples ever written
        self.lock = threading.Lock()
        self.listeners = []      # callables(samples_uv [k,n_ch], accel [k,3], t [k], count0)

    def push(self, eeg: np.ndarray, accel: np.ndarray, t: np.ndarray) -> None:
        k = len(eeg)
        with self.lock:
            idx = (self.pos + np.arange(k)) % self.size
            self.data[idx] = eeg
            self.accel[idx] = accel
            self.t[idx] = t
            self.pos = (self.pos + k) % self.size
            count0 = self.count
            self.count += k
        for fn in list(self.listeners):
            try:
                fn(eeg, accel, t, count0)
            except Exception as e:  # noqa: BLE001
                print(f"[buffer] listener error: {e!r}")

    def last(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """(samples [m, n_ch], t [m]) oldest-first, m = min(n, available)."""
        with self.lock:
            n = min(n, self.count, self.size)
            if n == 0:
                return np.zeros((0, self.n_ch), np.float32), np.zeros(0)
            idx = (self.pos - n + np.arange(n)) % self.size
            return self.data[idx].copy(), self.t[idx].copy()


def find_dongle(exclude: set[str] | None = None) -> str | None:
    """COM port of the OpenBCI dongle: an FTDI port, preferring one that calls itself
    OpenBCI / FT231X. The BioProxal band (ESP32-S3) is never a candidate."""
    if list_ports is None:
        return None
    exclude = exclude or set()
    cands = []
    for p in list_ports.comports():
        if p.device in exclude or (p.vid or 0) == ESP32S3_VID:
            continue
        desc = f"{p.description or ''} {p.manufacturer or ''} {p.product or ''}".lower()
        score = 0
        if (p.vid or 0) == FTDI_VID:
            score += 2
        if "openbci" in desc or "ft231" in desc or "rfduino" in desc:
            score += 3
        if "usb serial port" in desc:
            score += 1
        if score:
            cands.append((score, p.device))
    cands.sort(reverse=True)
    return cands[0][1] if cands else None


def list_candidate_ports() -> list[dict]:
    if list_ports is None:
        return []
    out = []
    for p in list_ports.comports():
        out.append({"device": p.device, "description": p.description, "vid": p.vid, "pid": p.pid,
                    "band": (p.vid or 0) == ESP32S3_VID, "ftdi": (p.vid or 0) == FTDI_VID})
    return out


class CytonBoard:
    """Streams a real Cyton into a SampleBuffer from a background thread."""

    def __init__(self, port: str, buffer_seconds: float = 30.0,
                 want_16: bool | None = None, daisy_even_is_upper: bool = True,
                 log=print):
        if serial is None:
            raise RuntimeError("pyserial is required: pip install pyserial")
        self.port = port
        self.log = log
        self.want_16 = want_16
        self.daisy_even_is_upper = daisy_even_is_upper
        self.ser = serial.Serial()
        self.ser.port, self.ser.baudrate, self.ser.timeout = port, 115200, 0.05
        self.ser.open()
        time.sleep(2.0)                      # dongle/board settle
        self.ser.reset_input_buffer()
        self.banner = self._soft_reset()
        low = self.banner.lower()
        self.daisy = "on daisy" in low and "device id" in low
        if want_16 is not False:
            # Ask for 16 channels; the reply settles it (Cyton SDK): "16$$$" / "daisy attached16$$$"
            # = Daisy present, "no daisy to attach!8$$$" = not present.
            reply = self._cmd(b"C", wait=1.5).lower()
            if "no daisy" in reply:
                self.daisy = False
            elif "16" in reply or "daisy attached" in reply:
                self.daisy = True
            if reply.strip():
                self.log("[cyton] C -> " + reply.strip().replace("\n", " "))
        self.n_ch = 16 if (self.daisy and want_16 is not False) else 8
        self.fs = 125.0 if self.n_ch == 16 else 250.0
        if self.n_ch == 8:
            self._cmd(b"c", wait=1.0)        # 8-channel mode ("daisy removed$$$" if one is attached)
        self._cmd(b"d", wait=0.5)            # default channel settings (gain 24, SRB2 on)
        self.buffer = SampleBuffer(self.n_ch, self.fs, buffer_seconds)
        self.streaming = False
        self.packets = 0
        self.dropped = 0
        self.last_packet_t = 0.0
        self._stop = threading.Event()
        self._thread = None
        self.log(f"[cyton] {port}: {'16-ch (Daisy)' if self.n_ch == 16 else '8-ch'} at {self.fs:.0f} Hz")

    # --- commands -----------------------------------------------------------------
    def _cmd(self, b: bytes, wait: float = 0.3) -> str:
        self.ser.write(b)
        self.ser.flush()
        time.sleep(wait)
        return self.ser.read(self.ser.in_waiting or 0).decode("ascii", "replace")

    def _soft_reset(self) -> str:
        self.ser.write(b"s")
        time.sleep(0.3)
        self.ser.reset_input_buffer()
        self.ser.write(b"v")
        self.ser.flush()
        out = b""
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline and b"$$$" not in out:
            out += self.ser.read(256)
        text = out.decode("ascii", "replace")
        if "$$$" not in text:
            raise RuntimeError(f"no '$$$' banner from the board on {self.port} - is the board "
                               f"switched to PC and the dongle paired? got: {text!r}")
        for line in text.strip().splitlines():
            self.log("[cyton] " + line.strip())
        return text

    def start(self) -> None:
        if self.streaming:
            return
        self.ser.reset_input_buffer()
        self.ser.write(b"b")
        self.ser.flush()
        self.streaming = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader, name="cyton", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(2.0)
        try:
            self.ser.write(b"s")
            self.ser.flush()
        except Exception:  # noqa: BLE001
            pass
        self.streaming = False

    def close(self) -> None:
        self.stop()
        try:
            self.ser.close()
        except Exception:  # noqa: BLE001
            pass

    # --- packet reader ------------------------------------------------------------
    def _reader(self) -> None:
        buf = bytearray()
        last_idx = None
        pending_half = None       # Daisy mode: (is_upper, 8 channels) waiting for the other half
        t_est = None
        dt = 1.0 / 250.0
        while not self._stop.is_set():
            chunk = self.ser.read(PACKET * 8)
            if not chunk:
                continue
            buf += chunk
            now = time.time()
            while len(buf) >= PACKET:
                if buf[0] != 0xA0 or not (0xC0 <= buf[PACKET - 1] <= 0xCF):
                    del buf[0]
                    continue
                pkt = bytes(buf[:PACKET])
                del buf[:PACKET]
                idx = pkt[1]
                eeg = np.array([int.from_bytes(pkt[2 + 3 * i:5 + 3 * i], "big", signed=True)
                                for i in range(8)], np.float32) * UV_PER_COUNT
                acc = np.array([int.from_bytes(pkt[26 + 2 * i:28 + 2 * i], "big", signed=True)
                                for i in range(3)], np.float32) * ACCEL_G_PER_COUNT
                if last_idx is not None:
                    gap = (idx - last_idx) % 256
                    if gap != 1:
                        self.dropped += gap - 1
                last_idx = idx
                self.packets += 1
                self.last_packet_t = now
                # timestamp: board-rate clock, re-anchored gently to host time
                if t_est is None:
                    t_est = now
                else:
                    t_est += dt
                    t_est += 0.01 * (now - t_est)
                if self.n_ch == 16:
                    # Daisy: even sample numbers carry channels 9-16, odd ones 1-8 (docs.openbci.com
                    # Cyton Data Format; BrainFlow cyton_daisy.cpp). Either half may arrive first;
                    # a 16-channel sample is emitted as soon as both halves are in hand.
                    upper = (idx % 2 == 0) == self.daisy_even_is_upper
                    if pending_half is not None and pending_half[0] != upper:
                        other = pending_half[1]
                        pending_half = None
                        lower, high = (other, eeg) if upper else (eeg, other)
                        sample = np.concatenate([lower, high])[None, :]
                        self.buffer.push(sample, acc[None, :], np.array([t_est]))
                    else:
                        pending_half = (upper, eeg)
                else:
                    self.buffer.push(eeg[None, :], acc[None, :], np.array([t_est]))

    def status(self) -> dict:
        age = time.time() - self.last_packet_t if self.last_packet_t else None
        return {"kind": "cyton", "port": self.port, "n_ch": self.n_ch, "fs": self.fs,
                "daisy": self.daisy, "streaming": self.streaming, "packets": self.packets,
                "dropped": self.dropped, "last_packet_age_s": age}


class SimBoard:
    """Synthetic 8- or 16-channel EEG at 250/125 Hz with a slow hidden 'flow' state, so the
    whole pipeline (widgets, flow index, correlation report) can be exercised with nothing
    plugged in. `hidden_flow()` is what the simulated BioProxal band reports (plus noise)."""

    def __init__(self, n_ch: int = 8, buffer_seconds: float = 30.0, log=print):
        self.n_ch = n_ch
        self.fs = 125.0 if n_ch == 16 else 250.0
        self.buffer = SampleBuffer(n_ch, self.fs, buffer_seconds)
        self.streaming = False
        self.packets = 0
        self.dropped = 0
        self.last_packet_t = 0.0
        self.port = "SIM"
        self.daisy = n_ch == 16
        self._stop = threading.Event()
        self._thread = None
        self._t0 = time.time()
        self._rng = np.random.default_rng(1)
        self.log = log
        self.log(f"[sim] synthetic {n_ch}-ch board at {self.fs:.0f} Hz")

    def hidden_flow(self, t: float | None = None) -> float:
        t = (time.time() if t is None else t) - self._t0
        return 0.5 + 0.35 * math.sin(2 * math.pi * t / 240.0) + 0.1 * math.sin(2 * math.pi * t / 53.0)

    def start(self) -> None:
        if self.streaming:
            return
        self.streaming = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._gen, name="sim", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(1.0)
        self.streaming = False

    def close(self) -> None:
        self.stop()

    def _gen(self) -> None:
        n = self.n_ch
        k = max(1, int(self.fs / 25))               # 40 ms blocks
        dt = 1.0 / self.fs
        t = time.time()
        phase = self._rng.uniform(0, 2 * np.pi, n)
        while not self._stop.is_set():
            flow = self.hidden_flow(t)
            tt = t + np.arange(k) * dt
            # frontal theta rises with flow, beta falls with it, posterior alpha stays put:
            # a clean monotonic signature so --sim demonstrates the whole pipeline end to end
            theta_amp = 3 + 16 * flow
            alpha_amp = 9.0
            beta_amp = 4 + 14 * (1 - flow)
            out = np.zeros((k, n), np.float32)
            for c in range(n):
                frontal = c in (0, 1, 8, 9, 10, 11)
                occ = c in (4, 5, 6, 7, 14, 15)
                s = (theta_amp * (1.0 if frontal else 0.4)) * np.sin(2 * np.pi * 6.0 * tt + phase[c])
                s += (alpha_amp * (1.0 if occ else 0.5)) * np.sin(2 * np.pi * 10.0 * tt + phase[c] * 2)
                s += beta_amp * 0.6 * np.sin(2 * np.pi * 20.0 * tt + phase[c] * 3)
                s += 3.0 * np.sin(2 * np.pi * 60.0 * tt)                       # mains
                s += self._rng.normal(0, 4, k)
                out[:, c] = s
            if self._rng.random() < 0.02:                                       # blink
                blink = 120 * np.exp(-((np.arange(k) - k / 2) ** 2) / (2 * (k / 4) ** 2))
                out[:, 0] += blink
                out[:, 1] += blink
            acc = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (k, 1))
            self.buffer.push(out, acc, tt)
            self.packets += k
            self.last_packet_t = time.time()
            t += k * dt
            lag = t - time.time()
            if lag > 0:
                time.sleep(lag)
            elif lag < -2.0:
                t = time.time()          # the laptop slept or the loop stalled: re-anchor, don't race

    def status(self) -> dict:
        return {"kind": "sim", "port": "SIM", "n_ch": self.n_ch, "fs": self.fs, "daisy": self.daisy,
                "streaming": self.streaming, "packets": self.packets, "dropped": 0,
                "last_packet_age_s": time.time() - self.last_packet_t if self.last_packet_t else None}
