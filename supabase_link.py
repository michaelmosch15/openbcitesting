"""Band flow from Supabase: wearable -> Amplifly app (BLE) -> Supabase -> this dashboard.

The app stores one `wearable_samples` row per minute while connected (columns: user_id,
sampled_at, flow_score, flow_state, anomaly, risk, mode, phase, hrv_valid, poor_contact,
asleep, usable, battery_mv, uptime_s, band). This link polls the REST endpoint every few
seconds for rows newer than the last one seen and hands them to the pipeline like any other
band sample. Read-only (select only).

Credentials: %USERPROFILE%\\.bioproxal\\supabase.env (SUPABASE_URL, SUPABASE_SECRET_KEY or
SUPABASE_PUBLISHABLE_KEY). The key never leaves this process; the browser only sees decoded
samples over localhost.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ENV_PATH = Path(os.environ.get("USERPROFILE", "~")).expanduser() / ".bioproxal" / "supabase.env"
WINDOW_STATES = {0: "unavailable", 1: "closed", 2: "opening", 3: "open", 4: "ending"}


def load_env(path: Path = ENV_PATH) -> dict:
    env = {}
    if not path.exists():
        raise RuntimeError(f"no Supabase credentials at {path}")
    for line in path.read_text("utf-8").splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    if "SUPABASE_URL" not in env:
        raise RuntimeError(f"{path} has no SUPABASE_URL")
    key = env.get("SUPABASE_SECRET_KEY") or env.get("SUPABASE_PUBLISHABLE_KEY")
    if not key:
        raise RuntimeError(f"{path} has no SUPABASE_SECRET_KEY / SUPABASE_PUBLISHABLE_KEY")
    return {"url": env["SUPABASE_URL"].rstrip("/"), "key": key}


def iso_to_epoch(s: str) -> float:
    s = s.replace("Z", "+00:00")
    # Postgres can emit 1-6 fractional digits; fromisoformat on 3.11+ copes with all of them.
    return datetime.fromisoformat(s).timestamp()


def epoch_to_iso(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat().replace("+00:00", "Z")


class SupabaseClient:
    def __init__(self, url: str, key: str, timeout: float = 15.0):
        self.url, self.key, self.timeout = url, key, timeout

    def get(self, table: str, params: dict) -> list[dict]:
        # "+" must be percent-encoded: a created_at of ...+00:00 sent raw is decoded as a space
        # by PostgREST ("invalid input syntax for type timestamp", seen live 2026-11-05)
        q = urllib.parse.urlencode(params, safe="*,.:()-")
        req = urllib.request.Request(f"{self.url}/rest/v1/{table}?{q}",
                                     headers={"apikey": self.key, "Authorization": f"Bearer {self.key}",
                                              "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def wearers(self, hours: float = 24.0) -> list[dict]:
        """Who has uploaded samples recently: one row per (user, band), newest first."""
        since = epoch_to_iso(time.time() - hours * 3600)
        rows = self.get("wearable_samples", {
            "select": "user_id,band,sampled_at", "sampled_at": f"gte.{since}",
            "order": "sampled_at.desc", "limit": "5000"})
        now = time.time()
        out: dict[tuple, dict] = {}
        for r in rows:
            k = (r["user_id"], r.get("band") or "")
            t = iso_to_epoch(r["sampled_at"])
            w = out.get(k)
            if w is None:
                w = out[k] = {"user_id": r["user_id"], "short": r["user_id"][:8], "band": r.get("band"),
                              "last_sampled_at": r["sampled_at"], "age_s": round(now - t), "rows_last_hour": 0}
            if now - t <= 3600:
                w["rows_last_hour"] += 1
        return sorted(out.values(), key=lambda w: w["age_s"])


def row_to_sample(r: dict) -> dict:
    fs = int(r.get("flow_state") or 0)
    flow = r.get("flow_score")
    hrv_valid = r.get("hrv_valid")
    no_reading = None
    if flow is not None and float(flow) == 0.0 and hrv_valid is False:
        no_reading = "flow not computed yet (HRV not valid)"     # a cold-start 0, not a low flow
    return {
        "t": iso_to_epoch(r["sampled_at"]),
        "t_created": iso_to_epoch(r["created_at"]) if r.get("created_at") else None,
        "flow": None if (flow is None or no_reading) else float(flow),
        "no_reading": no_reading, "hrv_valid": hrv_valid,
        "flow_state": fs,
        "window": WINDOW_STATES.get(fs, str(fs)),
        "hr": None,
        "anom": float(r["anomaly"]) if r.get("anomaly") is not None else None,
        "risk": float(r["risk"]) if r.get("risk") is not None else None,
        "mode": int(r.get("mode") or 0),
        "phase": r.get("phase"),
        "held": int(r.get("mode") or 0) == 1,
        "settling": False,
        "asleep": bool(r.get("asleep")),
        "poor_contact": bool(r.get("poor_contact")),
        "usable": r.get("usable"),
        "battery_mv": r.get("battery_mv"),
        "band": r.get("band"),
        "user_id": r.get("user_id"),
        "fhi": None, "fcal": None, "mlp": None, "zrms": None,
    }


class SupabaseBandLink:
    def __init__(self, user_id: str | None = None, band: str | None = None,
                 poll_s: float = 5.0, backfill_min: float = 30.0, log=print, env_path: Path = ENV_PATH):
        self.user_id, self.band = user_id, band
        self.poll_s, self.backfill_min = poll_s, backfill_min
        self.log = log
        self.env_path = env_path
        self.latest = None
        self.listeners = []
        self.state = {"connected": False, "port": "supabase", "unit": band, "mac": None,
                      "user_id_short": (user_id or "")[:8] or None, "error": None, "lines": 0}
        self._stop = threading.Event()
        self._thread = None
        self._last_created = None
        self._last_t = 0.0
        self.client: SupabaseClient | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="band-supabase", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(3.0)

    def _filters(self) -> dict:
        p = {"select": "*", "order": "created_at.asc", "limit": "500"}
        if self.user_id:
            p["user_id"] = f"eq.{self.user_id}"
        if self.band:
            p["band"] = f"eq.{self.band}"
        return p

    def _run(self) -> None:
        backoff = self.poll_s
        while not self._stop.is_set():
            try:
                if self.client is None:
                    env = load_env(self.env_path)
                    self.client = SupabaseClient(env["url"], env["key"])
                if self._last_created is None:
                    since = epoch_to_iso(time.time() - self.backfill_min * 60)
                else:
                    since = self._last_created
                p = self._filters()
                p["created_at"] = f"gt.{since}"
                rows = self.client.get("wearable_samples", p)
                if not self.state["connected"]:
                    self.log(f"[supabase] polling wearable_samples every {self.poll_s:.0f} s"
                             + (f" for user {self.user_id[:8]}" if self.user_id else "")
                             + (f" band {self.band}" if self.band else ""))
                self.state.update({"connected": True, "error": None})
                for r in rows:
                    self._last_created = r["created_at"]
                    s = row_to_sample(r)
                    s["source"] = "supabase"
                    self.latest = s
                    self._last_t = time.time()
                    self.state["lines"] += 1
                    self.state["unit"] = s.get("band") or self.state["unit"]
                    self.state["user_id_short"] = (s.get("user_id") or "")[:8] or self.state["user_id_short"]
                    for fn in list(self.listeners):
                        try:
                            fn(s)
                        except Exception as e:  # noqa: BLE001
                            self.log(f"[supabase] listener error: {e!r}")
                backoff = self.poll_s
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")[:200]
                self.state.update({"connected": False, "error": f"HTTP {e.code}: {body}"})
                self.log(f"[supabase] HTTP {e.code}: {body}")
                backoff = min(backoff * 2, 60.0)
            except Exception as e:  # noqa: BLE001
                self.state.update({"connected": False, "error": f"{type(e).__name__}: {e}"})
                self.log(f"[supabase] {e}")
                backoff = min(backoff * 2, 60.0)
            self._stop.wait(backoff)
        self.state["connected"] = False

    def status(self) -> dict:
        st = dict(self.state)
        st["last_line_age_s"] = round(time.time() - self._last_t, 1) if self._last_t else None
        st["last_sample_age_s"] = round(time.time() - self.latest["t"], 1) if self.latest else None
        st["transport"] = "supabase"
        return st
