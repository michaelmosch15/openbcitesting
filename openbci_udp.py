"""Alternative EEG source: the real OpenBCI GUI's Networking widget over UDP.

In the OpenBCI GUI choose Networking -> Protocol UDP -> Data Type "TimeSeriesRaw"
(or "TimeSeriesFilt"), IP 127.0.0.1, port 12345, Start. Each datagram is JSON:
    {"type": "timeSeriesRaw", "data": [[ch1 samples...], [ch2 ...], ...]}   (uV)
Then run:  python app.py --source udp [--udp-port 12345] [--channels 8|16]
The GUI keeps the board; this app only mirrors its stream into the BioProxal layer.
"""
from __future__ import annotations

import json
import socket
import threading
import time

import numpy as np

from cyton import SampleBuffer


class UdpBoard:
    def __init__(self, port: int = 12345, n_ch: int = 8, fs: float | None = None,
                 buffer_seconds: float = 30.0, log=print):
        self.port = f"udp:{port}"
        self.udp_port = port
        self.n_ch = n_ch
        self.fs = fs or (125.0 if n_ch == 16 else 250.0)
        self.daisy = n_ch == 16
        self.buffer = SampleBuffer(n_ch, self.fs, buffer_seconds)
        self.streaming = False
        self.packets = 0
        self.dropped = 0
        self.last_packet_t = 0.0
        self.log = log
        self._stop = threading.Event()
        self._thread = None
        self._t_est = None

    def start(self) -> None:
        if self.streaming:
            return
        self.streaming = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="udp", daemon=True)
        self._thread.start()
        self.log(f"[udp] listening on 127.0.0.1:{self.udp_port} for OpenBCI GUI time series")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(2.0)
        self.streaming = False

    def close(self) -> None:
        self.stop()

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.5)
        sock.bind(("127.0.0.1", self.udp_port))
        dt = 1.0 / self.fs
        try:
            while not self._stop.is_set():
                try:
                    data, _ = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                try:
                    msg = json.loads(data.decode("utf-8", "replace"))
                except ValueError:
                    continue
                if not str(msg.get("type", "")).lower().startswith("timeseries"):
                    continue
                chans = msg.get("data") or []
                if not chans:
                    continue
                k = min(len(c) for c in chans)
                n = min(len(chans), self.n_ch)
                block = np.zeros((k, self.n_ch), np.float32)
                for c in range(n):
                    block[:, c] = np.asarray(chans[c][:k], np.float32)
                now = time.time()
                if self._t_est is None:
                    self._t_est = now - k * dt
                t = self._t_est + np.arange(1, k + 1) * dt
                self._t_est = t[-1] + 0.02 * (now - t[-1])
                acc = np.zeros((k, 3), np.float32)
                self.buffer.push(block, acc, t)
                self.packets += k
                self.last_packet_t = now
        finally:
            sock.close()

    def status(self) -> dict:
        return {"kind": "udp", "port": self.port, "n_ch": self.n_ch, "fs": self.fs, "daisy": self.daisy,
                "streaming": self.streaming, "packets": self.packets, "dropped": 0,
                "last_packet_age_s": time.time() - self.last_packet_t if self.last_packet_t else None}
