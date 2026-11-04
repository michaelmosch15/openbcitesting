# Interface contract: server <-> browser

The Python server (`app.py` / `server.py`) owns every device and all the math. The browser
(`static/`) only renders. This file is the contract both sides are built against; change it
here first.

Server: `http://127.0.0.1:8765` (port in `config.json`, `--port` overrides). Localhost only.

## Static

| Route | Serves |
|---|---|
| `GET /` | `static/index.html` |
| `GET /static/<file>` | files under `static/` (no caching headers, `Cache-Control: no-store`) |

## JSON API

All responses `application/json`. Errors: `{"ok": false, "error": "<message>"}` with HTTP 400/500.

| Route | Returns |
|---|---|
| `GET /api/state` | the full `status` object below plus the latest `analysis` object (`{"status": ..., "analysis": ...}`) |
| `GET /api/history` | `{"pairs": [[t, band_flow, eeg_flow], ...], "eeg": [[t, flow_eeg], ...], "band": [[t, flow], ...], "marks": [[t, kind, value], ...]}` - the whole session so far (server keeps up to 4 h) |
| `GET /api/wearers` | Supabase only: `{"wearers": [{"user_id": "<full uuid>", "short": "54af48b8", "band": "BioProxal4", "last_sampled_at": "<iso>", "age_s": 42, "rows_last_hour": 55}, ...]}` ordered newest first (last 24 h) |
| `GET /api/sessions` | `{"sessions": [{"name": "2026-11-03_14-02-11_desk", "dir": "<abs path>", "epochs": 900, "band": 30, "marks": 4, "has_report": true}, ...]}` |
| `GET /api/report?session=<name>` | runs `report.py` on that session, writes `report.md` into it, returns `{"ok": true, "markdown": "...", "summary": {...}}` |
| `POST /api/cmd` | body `{"action": "...", ...}`; returns `{"ok": true, ...}` |

### `POST /api/cmd` actions

| action | fields | effect |
|---|---|---|
| `start` | - | start EEG streaming (sends `b` to the Cyton) |
| `stop` | - | stop streaming |
| `record_start` | `label` (optional, `[A-Za-z0-9_-]{0,32}`) | new session folder under `sessions/`, all CSVs start |
| `record_stop` | - | close the session; response has `dir` |
| `mark` | `kind` one of `felt_flow` (value 0-1 from the slider), `flow_start`, `flow_end`, `distracted`, `note` (value = text), `note` (optional) | writes `marks.csv` when recording, always appended to history |
| `set_filter` | `notch` 0/50/60, `bp_lo` Hz, `bp_hi` Hz | rebuilds the display filter chain |
| `set_channel` | `index` 0-based, `enabled` bool | an excluded channel is not drawn in FFT/band/head widgets and is left out of the EEG flow index |
| `set_band_source` | `source` one of `supabase`, `ble`, `usb`, `sim`, `none`; `user_id` (full uuid) and/or `band` (e.g. `BioProxal4`) for supabase; `address` for ble; `port` for usb | restarts the band link |
| `set_flow_weights` | `theta`, `alpha`, `beta` (floats) | EEG flow index weights (see `flow_index.py`); takes effect on the next epoch |

## Server-sent events: `GET /events`

`text/event-stream`. Each message has `event:` and a one-line JSON `data:`.

### `event: hello` (once, on connect)
```json
{"server_version": "0.1.0", "labels": ["Fp1","Fp2","C3","C4","P7","P8","O1","O2"],
 "fs": 250, "n_ch": 8, "channel_colors": ["#818181","#7C4B8D","#36579E","#317159","#DDB20D","#FD5E34","#E0382D","#A25231"],
 "positions": {"Fp1": [-0.31, 0.95], "Fp2": [0.31, 0.95], "...": [0, 0]},
 "history": { ...same object as GET /api/history... }}
```

### `event: samples` (every 100 ms while streaming)
Filtered (notch + band-pass) samples that arrived since the previous event, per channel, in uV.
```json
{"t0": 1762178471.123, "fs": 250, "n": 25,
 "ch": [[...25 floats...], [...], ...],          // n_ch arrays, oldest first
 "raw_peak": [12.3, 15.0, ...]}                   // per channel, max |raw uV| in this chunk
```
The browser keeps its own ring (5 / 10 / 20 s window, like the OpenBCI GUI time-series widget).

### `event: analysis` (once per 2-s epoch while streaming)
```json
{"t": 1762178471.6,
 "fft": {"freqs": [0.5, 1.0, ..., 60.0], "psd": [[...per freq...], ...n_ch]},   // uV^2/Hz, 2 s Hann window
 "bands": {"order": ["delta","theta","alpha","beta","gamma"],
           "abs": {"delta": [..n_ch..], "theta": [...], "alpha": [...], "beta": [...], "gamma": [...]},
           "rel": {"delta": [...], ...}},
 "quality": [{"state": "good|near|railed|flat|unknown", "pct": 1.2, "std": 8.4}, ...n_ch],
 "enabled": [true, true, ...n_ch],
 "flow_eeg": {"flow_eeg": 0.61, "theta_frontal_rel": 0.21, "alpha_posterior_rel": 0.30, "beta_rel": 0.18,
              "engagement_index": 0.52, "theta_alpha_ratio": 0.8, "z_theta": 0.4, "z_alpha": -0.1, "z_beta": -0.3,
              "baseline_ready": true, "n_good": 8, "delta_rel_mean": 0.2, "...": 0},    // null until the first epoch
 "band": {"t": 1762178440.0, "flow": 0.58, "hr": 71.5, "anom": 0.12, "mode": 0, "held": false, "settling": false,
          "asleep": false, "poor_contact": false, "flow_state": 3, "window": "open", "band": "BioProxal4",
          "source": "supabase", "age_s": 31.6},                      // null when no band sample yet; hr null on supabase
 "stats": {"n_pairs": 31, "r": 0.62, "r_ci": [0.32, 0.81], "r_last_5min": 0.7, "mae": 0.11,
           "band_minus_eeg_mean": -0.04, "best_lag_s": -60, "r_at_best_lag": 0.66,
           "lags": [[-180, 0.2], [-120, 0.4], ..., [180, 0.1]],
           "felt": {"n": 4, "r_band": 0.5, "r_eeg": 0.7},
           "pair_window_s": 60}}
```
`pairs` are formed at the band's cadence: for every band sample the EEG flow index is averaged
over the `pair_window_s` seconds before the band's timestamp (60 s for Supabase, 10 s for
BLE/USB/sim). `r_ci` is a bootstrap 95 % interval; null until `n_pairs >= 8`.

### `event: status` (every 1 s)
```json
{"board": {"kind": "cyton|sim|udp", "port": "COM7", "n_ch": 8, "fs": 250, "daisy": false,
           "streaming": true, "packets": 12345, "dropped": 2, "last_packet_age_s": 0.01},
 "band": {"source": "supabase|ble|usb|sim|none", "connected": true, "unit": "BioProxal4", "mac": null,
          "user_id_short": "54af48b8", "error": null, "last_line_age_s": 12.0, "lines": 40,
          "seen": ["..."]},                                   // seen: BLE scan names when not connected
 "recorder": {"active": false, "dir": null, "since_s": null, "counts": {"eeg": 0, "epochs": 0, "band": 0, "marks": 0}},
 "filter": {"notch": 60, "bp_lo": 1, "bp_hi": 50},
 "flow_weights": {"theta": 0.5, "alpha": 0.25, "beta": 0.25},
 "clock": 1762178471.9}
```

### `event: band` (whenever a new band sample arrives)
The `band` object above, alone.

### `event: mark` (whenever a mark is added)
`{"t": ..., "kind": "felt_flow", "value": 0.7, "note": ""}`

## Session folder layout (`sessions/<stamp>_<label>/`)

See `recorder.py` docstring: `session.json`, `eeg_raw.csv`, `epochs.csv`, `band.csv`,
`marks.csv`, and `report.md` + `report.json` once `report.py` has run.
