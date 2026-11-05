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
{"server_version": "0.2.0", "labels": ["Fp1","Fp2","C3","C4","P7","P8","O1","O2"],
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
 "flow_eeg": {"t": 1762178471.6, "flow_eeg": 0.61, "flow_eeg_alt": 0.58, "artifact": false,
              "theta_frontal_rel": 0.21, "alpha_cp_rel": 0.30, "beta_rel": 0.18,
              "engagement_index": 0.52, "theta_alpha_ratio": 0.8, "z_theta": 0.4, "z_alpha": -0.1, "z_beta": -0.3,
              "baseline_ready": true, "baseline_state": "rest-frozen", "n_good": 8, "frontal_sites": "F3,F4",
              "delta_rel_mean": 0.2, "...": 0, "p2p_uv": 42.1, "n_segments_used": 9},
              // null until the first epoch; flow_eeg/flow_eeg_alt null on an artifact epoch (artifact true)
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
 "recorder": {"active": false, "dir": null, "since_s": null, "counts": {"eeg": 0, "epochs": 0, "band": 0, "marks": 0, "trials": 0}},
 "flow_index": {"baseline_mode": "rest", "baseline_state": "rest-frozen", "baseline_ready": true, "epochs": 420,
                "artifact_epochs": 31, "artifact_p2p_uv": 150.0, "artifact_abs_uv": 100.0,
                "weights": {"theta": 1.0, "alpha": 0.0, "beta": 0.0}, "frontal_sites": "F3,F4"},
 "protocol": { ...see "Experiment protocol and task (v0.2)" below... },
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

## Experiment protocol and task (v0.2)

The protocol runner (`protocol.py`, schedules in `protocols.json`) plays a timed block sequence
(rest baseline, boredom / flow / overload arithmetic blocks, pauses) so both flow scores can be
checked against the condition each block was built to induce. The browser renders the current
block, runs the arithmetic task, and collects the self-report; the server keeps the clock.

| Route | Returns |
|---|---|
| `GET /api/protocol` | the `protocol` status object below (schedule, current block, available protocols) |

### `POST /api/cmd` additions

| action | fields | effect |
|---|---|---|
| `protocol_start` | `name` (key in protocols.json, e.g. `difficulty_ladder`), `seed` (optional int) | builds a counterbalanced schedule and starts block 0; auto-starts recording if not recording (label = protocol name) |
| `protocol_stop` | - | aborts the protocol (recording keeps running) |
| `protocol_next` | - | ends the current block now and starts the next |
| `rating` | `block_index`, `items` (object of item id -> integer 1..7, plus `demand` 1..3 and `flow_overall` 0..1), `note` | stores the block's self-report (mark kind `rating`) |
| `trial` | `block_index`, `task`, `level`, `problem`, `answer`, `response`, `correct` (bool), `rt_ms`, `t` | logs one task trial (mark kind `trial`, and `task.csv` when recording) |

### `protocol` object (in `event: status` as `status.protocol`, and `GET /api/protocol`)
```json
{"active": true, "name": "difficulty_ladder", "block_index": 3, "total_blocks": 14,
 "block": {"block_index": 3, "condition": "flow", "label": "Matched arithmetic",
           "instruction": "Answer as fast and accurately as you can...", "duration_s": 360,
           "task": "arithmetic_adaptive", "rating": true, "baseline": false, "rep": 1},
 "remaining_s": 212.4, "pending_rating": null, "order_seed": 1762190000,
 "schedule": [ {...block view...}, ... ],
 "available": [{"name": "difficulty_ladder", "title": "...", "description": "...", "total_min": 47.0}, ...]}
```
`condition` values: `rest`, `pause`, `boredom`, `flow`, `overload`, `work`. `task` values:
`arithmetic_easy`, `arithmetic_adaptive`, `arithmetic_hard`, or null (no task: show the
instruction and, for `rest`, a fixation dot).

### `event: protocol` (on every transition)
`{"phase": "block_start" | "block_end" | "rating_due" | "rated" | "done" | "stopped", ...block view...}`
On `rating_due` the browser opens the self-report dialog for `block_index`; on `block_start`
it switches the task widget to `task`; on `done`/`stopped` it clears the task.

### The arithmetic task (browser-side, `static/app.js`)
Mental arithmetic like Katahira et al. 2018 (where flow was induced by difficulty matched to
the player). One problem at a time, typed answer, Enter submits, immediate feedback tick.
- `arithmetic_easy` (boredom): single-digit additions (2 + 3), a new problem only every 4 s
  (the answer field is disabled until then), no time pressure.
- `arithmetic_adaptive` (flow): levels 1..12 (L1 one-digit add; L3 two-digit add; L5 two-digit
  add/sub mixed; L7 two-digit x one-digit; L9 three-digit add/sub; L11 two-digit x two-digit),
  staircase: 2 correct in a row -> level up, 1 wrong -> level down (targets ~70-80 % correct),
  soft time limit 10 s per problem (counts as wrong). Starts at level 4.
- `arithmetic_hard` (overload): fixed level 12 problems (two-digit x two-digit, three-digit
  sums of three numbers) with a hard 6 s limit each; wrong/timeouts simply move on.
Every answer posts a `trial` command. The widget shows the running accuracy and level.

### Self-report dialog (Flow Short Scale, Rheinberg, Vollmeyer & Engeser 2003; 7-point)
Items (ids): `challenge` "I feel just the right amount of challenge", `fluent` "My thoughts/
activities run fluidly and smoothly", `time` "I don't notice time passing", `concentrate` "I
have no difficulty concentrating", `clear` "My mind is completely clear", `absorbed` "I am
totally absorbed in what I am doing", `automatic` "The right thoughts/movements occur of their
own accord", `know` "I know what I have to do each step of the way", `control` "I feel that I
have everything under control", `lost` "I am completely lost in thought" (1 = not at all,
7 = very much). Manipulation check `demand`: "For me personally, the demands were" 1 = too
low, 2 = just right, 3 = too high. Overall `flow_overall`: the existing 0-1 felt-flow slider.
The dialog posts `rating` with all items; it can be dismissed (no rating stored).

### Session folder additions
`task.csv` (t, block_index, condition, task, level, problem, answer, response, correct, rt_ms)
and `ratings` inside `marks.csv` (kind `rating`, value = JSON). `report.py` scores per block.

## Astra (v0.3)

Two panels backed by the owner's OpenAI reviewer model (`astra.py`, model `gpt-6-astra`, key
`OPENAI_KEY` read from the first `.env` above the project folder, i.e. `BioProxal2026/.env`;
never sent to the browser). The server builds every prompt from its own state; the browser
only sends questions and shows text. Requests run on background threads; results arrive as
`event: astra`. One request at a time per panel (a second one gets `400 "Astra is still ..."`).

| Route | Returns |
|---|---|
| `GET /api/astra` | `{"available": bool, "model": "gpt-6-astra", "live": {...}, "setup": {...}, "notes": [note, ...] (last 200), "setup_transcript": [{"id", "t", "who": "user"|"astra", "text"}, ...] (last 60)}` |

### `POST /api/cmd` additions

| action | fields | effect |
|---|---|---|
| `astra_live` | `enabled` bool, `interval_s` (>= 20, default `config.json` `astra.live_interval_s` 60) | start / stop the live analyst; the first note is written at once |
| `astra_analyze` | - | one structured interim analysis (data quality, blocks so far, agreement, toward the gates, watch next) over the last 15 min; response `{"started": true}` |
| `astra_ask` | `role` `analyst` (default) or `setup`, `text` (<= 2000 chars) | a question to the live analyst (answered from the live window + its earlier notes) or to the setup helper (answered from the live setup state + the placement reference) |
| `astra_check` | - | the setup helper's channel-by-channel readiness check, ending in READY / NOT READY |
| `astra_clear` | - | forget the setup conversation |

### `status.astra` (in `event: status`)
```json
{"available": true, "model": "gpt-6-astra",
 "live": {"enabled": false, "interval_s": 60.0, "window_s": 300.0, "busy": false, "model": "gpt-6-astra",
          "n_notes": 0, "error": null, "next_in_s": null},
 "setup": {"busy": false, "model": "gpt-6-astra", "n_turns": 0, "error": null}}
```
`available` is false when no `.env` with `OPENAI_KEY` was found; every `astra_*` command but
`astra_live {enabled: false}` is then refused with 400. `status.astra` is `null` under
`--astra off`.

### `event: astra`
One stream for both panels; `kind` tells them apart.

| kind | fields | meaning |
|---|---|---|
| `note` / `analysis` / `reply` | `id`, `t`, `role: "analyst"`, `text`, `done`, `block`, `condition`, `question` (reply only) | a live-analyst entry: first with `done: false` and empty `text` (open a pending entry), then after the deltas with `done: true` and the full text |
| `setup` | `id`, `t`, `question`, `text`, `done` | a setup-helper answer, same two-phase shape |
| `delta` | `id`, `text` | a streamed piece of the entry with that `id`; append |
| `error` | `id`, `t`, `role` (`analyst` / `setup`), `text`, `done: true` | the request failed (HTTP error, timeout, no key) |

Live-analyst notes start with `[OK]`, `[WATCH]` or `[ISSUE]`; the browser colours the left
border from that tag. The analyst sees, per tick: the protocol block, baseline state, the EEG
index as 10-s means over the window, every band sample with its flags, the pairs in the window
and their Pearson r, the session statistics, the block table, ordering and block rho, the
current block's task accuracy, marks, per-channel quality, recording state, plus its own last
`astra.memory_turns` exchanges. Its system prompt forbids proposing changes to weights,
thresholds, gates or schedule (pre-registration).

### Session folder additions
While recording, every live-analyst entry is appended to `astra_notes.md` (one bullet per
note with clock, block and condition) and `astra_notes.jsonl` (the note objects). Nothing is
written when not recording; the setup conversation is never written.
