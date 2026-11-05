# Architecture

## Components

```
 OpenBCI Cyton (+Daisy)                       BioProxal band (ESP32-S3)
   | RFduino dongle, 115200 baud                 |            |              |
   v                                             | USB        | BLE          | BLE
 cyton.py  CytonBoard ----+                      v            v              v
 cyton.py  SimBoard ------+--> SampleBuffer   band_link.py  band_ble.py   Amplifly app
 openbci_udp.py UdpBoard -+     (30 s ring)   BandLink      BleBandLink      |
   (OpenBCI GUI UDP)              |                |            |          Supabase
                                  | listener       |            |      wearable_samples
                                  v                |            |              |
                        pipeline.py Pipeline       |            |    supabase_link.py
                          FilterChain (dsp.py)     |            |    SupabaseBandLink
                          2-s epochs: display psd, |            |              |
                          10-s Welch index window, +------------+--------------+
                          artifact rule, quality                | on_band()
                          FlowIndex (flow_index.py)             |
                          block table + block stats  <----------+
                          pairs + stats + history
                                  |   ^         ^
                                  |   |         | on_protocol(ev)      protocol.py
                 recorder.py      |   |         +--------------------- ProtocolRunner
                 Recorder <-------+   | start_rest / end_rest           (timer thread,
                 (writer thread)  |   | (App.on_protocol_event)          protocols.json)
                 eeg / epoch /    |   |                                     |  ^
                 band / mark /    v   |                                     |  | rate()
                 trial          server.py App + ThreadingHTTPServer  <------+  |
                                  GET /events (SSE), GET /api/*, POST /api/cmd |
                                  |                                            |
                                  v                                            |
                        static/ index.html, app.js, style.css  ----------------+
                          (render; the arithmetic task and the FSS dialog run here)
```

`app.py` wires it together: loads `config.json` (substituting `labels_8_flow` for
`labels_8` under `--montage flow`), picks the board (`find_dongle()` / `--board` /
`SimBoard` / `UdpBoard`), builds `Recorder`, `Pipeline`, `BandManager`, `ProtocolRunner`,
`App` and (unless `--astra off`) `astra.Astra`, starts the server on 127.0.0.1 and opens the
browser. `report.py` runs offline on a session folder.

`astra.py` (v0.3) sits beside the pipeline and reads it: `LiveAnalyst` renders a text
snapshot of the last 5 min (`live_snapshot` / `render_live`: EEG index as 10-s means, band
samples with flags, pairs and their r, block table, task accuracy, quality) every
`live_interval_s`, sends it with its own previous notes to the OpenAI model `gpt-6-astra`
(stdlib `urllib`, streamed SSE, key from `BioProxal2026/.env`) and broadcasts the answer as
`event: astra`; while recording, notes are appended to `astra_notes.md` / `.jsonl` in the
session folder. `SetupHelper` answers placement / wiring / band-link questions from a
`setup_snapshot` (board, per-channel quality and raw peaks, montage, frontal sites, band
link) plus a fixed reference (10-20 sites, Mark IV handling, Cyton switches, quality flags,
the README troubleshooting table and the manual hardware checks). Astra never writes to the
pipeline, the recorder or the protocol; it is observational and its prompts forbid tuning
suggestions.

## Data flow

### EEG path

1. `CytonBoard._reader` reads 33-byte packets, decodes 8 x int24 EEG to uV and 3 x int16
   accelerometer to g, counts dropped sample numbers, merges Daisy halves, and timestamps
   each sample with a board-rate clock gently re-anchored to host time (1 % per sample).
2. `SampleBuffer.push` stores the raw sample in the 30-s ring and calls every listener.
3. `Pipeline._on_samples` (still on the board thread) runs the stateful display
   `FilterChain` (notch + band-pass) on the new samples, appends them to the pending list
   for the browser, forwards the raw block to the `Recorder`, and when `fs * epoch_s`
   samples have accumulated calls `_epoch()`.
4. `_epoch()` takes the last 10 s of raw data (`max(epoch_s, index_window_s)`) from the
   buffer; computes `quality()` on the raw last 2 s; filters a mirror-padded copy with a
   fresh, stateless analysis chain (selected notch + 1-45 Hz); for the display, a Hann
   `psd()` of the last 2 s up to 60 Hz with `band_powers()` / `relative_powers()`; for the
   index, `welch()` over the 10 s in 2-s Hann segments at 50 % overlap (9 segments),
   dropping any segment whose enabled-and-good channels exceed 150 uV peak-to-peak or
   100 uV absolute. If fewer than 5 of 9 segments survive the epoch is an artifact. The
   index bands (`theta_fm` 4-7, `alpha_hi` 10-13 Hz) and the covariate bands are
   integrated from the Welch PSD and divided by the 1-45 Hz total, then
   `FlowIndex.update()` is called with the artifact flag. The result becomes
   `analysis.flow_eeg`, an `epochs.csv` row (with per-channel theta/alpha/beta absolute
   power and the quality state), and - for a non-artifact epoch - an `eeg_hist` entry.
   The full definition is `docs/EEG_FLOW_INDEX.md`.
5. `App._tick` (SSE thread, every 100 ms) drains the pending filtered samples into a
   `samples` event, sends `analysis` when a new epoch exists, and `status` once a second
   (`status.protocol` carries the runner's status).

### Band path

1. A band link thread produces one sample dict per band reading (`[pub]` line, BLE STATE
   frame, or Supabase row) and calls its listeners.
2. `Pipeline.on_band` stores it in `band_hist`, writes `band.csv`, and - if `flow` is not
   null and the sample is not `held`, `asleep`, `settling` or `poor_contact` - averages
   `eeg_hist` over `[t - pair_window_s, t]` to form one pair, then recomputes `stats`
   (block stats, then the secondary `pearson`, `bootstrap_ci`, lag sweep, felt-flow
   correlations).
3. The server broadcasts the `band` event; the next `analysis` carries the new `stats`.

### Protocol path (v0.2)

1. `POST /api/cmd {"action": "protocol_start", "name": ...}` -> `App`: starts a recording
   if none is active (label = protocol name), then `ProtocolRunner.start(name, seed)`.
   `build()` lays out the schedule from `protocols.json`: `shuffle` groups permuted per
   repetition with `random.Random(seed)` (seed = session start time unless given), the
   first block of a round swapped if it would repeat the previous condition, a `between`
   pause after every task block except the last, and `index` / `rating` / `baseline` /
   `task` defaults on every block. The order and seed go into the `protocol_start` mark.
2. The `protocol` daemon thread (`_run`) steps through the blocks: for each, a
   `block_start` mark (value = condition, note `block=<i> <label>`) and a `block_start`
   event; then it waits on the `_advance` event in 0.25-s slices until `duration_s`
   elapses, `next()` sets `_advance`, or `stop()` sets `_stop`. `_end_block()` writes the
   `block_end` mark, emits `rating_due` if the block has `rating` (not when aborted) and
   sets `pending_rating`, then emits `block_end`. After the last block: `protocol_end` mark
   and a `done` event. `stop()` ends the current block with `aborted`, marks
   `protocol_stop` and emits `stopped`.
3. Every runner callback lands in `App`: `on_protocol_mark` -> `Pipeline.mark` (history,
   `marks.csv`, SSE `mark`); `on_protocol_event` does four things in order:
   - **baseline control**: on `block_start` of the block flagged `baseline` it calls
     `FlowIndex.start_rest()` (every `Baseline` starts collecting); on that block's
     `block_end` it calls `FlowIndex.end_rest()`, which freezes mean / SD per component if
     >= 15 clean epochs were collected (`baseline_state` -> `rest-frozen`; otherwise it
     stays `session-until-rest`);
   - `Pipeline.on_protocol(ev)` keeps the block table `pipeline.blocks` (`index`,
     `condition`, `t0`, `t1`, `baseline`) from `block_start` / `block_end` / `stopped`
     using the server clock, the same clock as the epochs;
   - with `--sim`, `SimBoard.set_condition(condition)` so the synthetic hidden state eases
     to the condition's level (and back to drift on `done` / `stopped`);
   - broadcast as SSE `protocol`.
4. The browser, on `block_start`, shows the instruction, starts the countdown and switches
   the task widget to the block's `task` (`arithmetic_easy` / `_adaptive` / `_hard` or
   none, with a fixation dot for `rest`). Every answer is a `POST /api/cmd {"action":
   "trial", ...}`; `App` stamps the current block's condition and calls `Pipeline.trial()`,
   which appends to `pipeline.trials` and writes `task.csv` (t, block_index, condition,
   task, level, problem, answer, response, correct, rt_ms). Trials are not marks.
5. On `rating_due` the browser opens the Flow Short Scale dialog; submitting posts
   `rating` with `block_index` and `items`. `App` coerces every item to float (keys cut to
   24 chars) and calls `ProtocolRunner.rate()`, which stores the rating, clears
   `pending_rating`, writes a `rating` mark whose value is JSON (`block`, `condition`,
   items) and emits `rated`. `Pipeline.mark` parses that JSON into `pipeline.ratings`
   (`t`, `block`, `condition`, `items`), which is where block stats and `report.py` (via
   `marks.csv`) read the FSS from.

### Block-level statistics (`Pipeline._block_stats`)

Recomputed on every mark and every usable band sample (`_compute_stats`), published as
`stats.blocks`. For each block in the table except `pause` and blocks shorter than 30 s
(an open block runs to now): `eeg_mean` over `eeg_hist` within `[t0, t1]` (artifact
epochs are never in `eeg_hist`), `band_mean` over usable `band_hist` samples in the block,
`fss_mean` = mean of the block's rating items except `demand` and `flow_overall`, `felt` =
`flow_overall`, with `n_eeg` / `n_band`. Then per-condition means and `n_blocks`, an
`ordering` string per measure over the non-rest conditions (e.g. `flow>overload>boredom`),
and `block_rho` = Spearman (Pearson of ranks) between `eeg_mean` and `band_mean` over the
rows that have both, null below 4 rows. The live rho includes the rest rows when they
have a band mean; `report.py` restricts the gate to task blocks and is the record.

Marks: `Pipeline.mark` appends to `marks`, writes `marks.csv`, recomputes stats, and the
server broadcasts a `mark` event.

## Threads

| Thread (name) | Owner | Does |
|---|---|---|
| `cyton` / `sim` / `udp` | board object | reads packets or generates samples; pushes into `SampleBuffer`; runs the pipeline's filter and epoch work via the listener |
| `band` / `band-ble` / `band-supabase` / `simband` | band link | one at a time, swapped by `BandManager.switch()`; reconnect loops with back-off |
| `protocol` | `ProtocolRunner` | one per protocol run; steps the schedule, sleeps in 0.25-s slices on `_advance`, calls `on_mark` / `on_event` (which run the baseline control and block table on this thread) |
| `recorder` | `Recorder` | drains a `queue.Queue` of eeg/epoch/band/mark/trial items to CSV; flushes when the queue is empty |
| `sse-tick` | `App` | the single transport loop: every 100 ms drains samples and pushes `samples`, `analysis`, `status` to all SSE subscribers |
| `http` + one thread per connection | `ThreadingHTTPServer` | JSON API and the long-lived `/events` connections (keepalive comment every 15 s); `rating` / `trial` / `protocol_*` commands run here |
| `astra-live` | `astra.LiveAnalyst` | wakes every second; when live and due, builds the snapshot and makes one streamed OpenAI call (up to ~3 min with retries), emitting `astra` events; `astra_analyze` / `astra_ask` / `astra_check` each run on a one-off daemon thread, serialised per panel by `LiveAnalyst._lock` / `SetupHelper._lock` |
| main | `app.py` | waits for Ctrl-C, then stops board, band, recorder, server |

Shared state is protected by `SampleBuffer.lock` (ring), `Pipeline.lock` (filter chain +
pending list), `App.sub_lock` (subscriber list), `BandManager._lock` (link swap) and
`ProtocolRunner._lock` (active / index). `deque(maxlen)` histories are appended from one
thread and read from another; a snapshot is taken (`list(...)`, `np.array(...)`) before
iterating. `pipeline.blocks` and `pipeline.ratings` are plain lists appended on the
protocol / http threads and read on the board and band threads; the readers tolerate a
row appearing mid-iteration because every row is complete when appended.

## Timing

| Item | Cadence / size | Where set |
|---|---|---|
| Cyton packets | 250/s; 8 ch at 250 Hz or 16 ch at 125 Hz per channel | board |
| `SampleBuffer` | 30 s ring | `cyton.py` |
| Filtered samples to browser | `samples` event every 100 ms (~25 samples) | `server.py` `_tick` |
| Epoch (display PSD, quality, index, block stats not included) | every `epoch_s` = 2 s | `config.json` |
| Index window | 10 s Welch, 2-s Hann segments, 50 % overlap = 9 segments; a 10-s moving window stepped every 2 s | `config.json` `flow_index.index_window_s` / `index_seg_s` |
| Analysis filter | selected notch (+ 2nd harmonic below Nyquist), 1-45 Hz; mirror-padded 1 s warm-up | `Pipeline._epoch`, `analysis_hi_hz` |
| Artifact rule | segment: 150 uV p2p or 100 uV abs on enabled good channels; epoch: < 50 % clean segments | `artifact_p2p_uv`, `artifact_abs_uv`, `index_min_clean_fraction` |
| Flow index baseline | eyes-open rest block (3 min = 90 epochs in `difficulty_ladder`), frozen at block end if >= 15 clean epochs; expanding session mean / SD until then; z = 0 for the first 15 epochs | `config.json` `flow_index.baseline_mode` / `warmup_epochs`; `protocols.json` `baseline` flag |
| `analysis` event | on each new epoch | `server.py` |
| `status` event | every 1 s (carries `status.protocol`) | `server.py` |
| Protocol block | `duration_s` per block, ended within 0.25 s; `difficulty_ladder` 180 + 60 + 6 x 360 + 5 x 60 + 120 s = 47 min; `sim_quick` 60 + 6 x 60 + 5 x 20 + 30 s = 9.2 min | `protocols.json`, `ProtocolRunner._run` |
| Rating pause | 60 s (`difficulty_ladder`), 20 s (`sim_quick`); the dialog stays open until submitted or dismissed | `protocols.json` `between` |
| Block stats | recomputed on every mark and usable band sample; blocks < 30 s skipped; rho from 4 rows | `pipeline.py` `_block_stats` |
| Band `[pub]` (USB) / STATE notify (BLE) | every 2 s | firmware |
| Supabase rows | 1 per minute per wearer (5-6 per 6-min block); polled every `poll_s` = 5 s; initial backfill 30 min | `config.json` `supabase` |
| Sim band | every 2 s, 20 s lag, sigma 0.06 noise; hidden state eases to the condition level over ~20 s | `band_link.py`, `cyton.SimBoard` |
| Pair window (secondary) | 60 s Supabase, 10 s BLE/USB/sim | `config.json` `pair_window_s` |
| Bootstrap CI (secondary) | 500 resamples, from 8 pairs | `pipeline.py` |
| Lag sweep (secondary) | -180 to +180 s, 10-s steps, from 8 pairs; positive = band lags EEG | `pipeline.py` |
| Felt-flow pairing (secondary) | EEG mean over 60 s before the mark; nearest band sample within 90 s | `pipeline.py` |
| History kept in memory | `history_hours` = 4 h of epochs, band samples and pairs | `config.json` |
| USB link timeouts | keepalive 60 s, no `[pub]` for 30 s -> reconnect; back-off 2 s x1.5 up to 20 s | `band_link.py` |
| BLE link timeouts | scan 8 s (12 s by address), no notify for 45 s -> reconnect; same back-off | `band_ble.py` |
| Supabase back-off | x2 per error up to 60 s | `supabase_link.py` |
| SSE keepalive | comment every 15 s | `server.py` |

## The EEG flow index

Specified in `docs/EEG_FLOW_INDEX.md` (operational definition, numbers, limits,
pre-registration lock) on the basis of `docs/EEG_FLOW_LITERATURE.md`. In one paragraph:

`flow_eeg = sigmoid(z_theta)` where `z_theta` is the natural log of frontal-midline theta
(4-7 Hz) relative power over the 1-45 Hz total, at the frontal set (Fz F3 F4 ... ;
Fp1/Fp2 fallback on the factory 8-ch montage), from the 10-s Welch spectrum, z-scored
against the mean and SD of the eyes-open rest block. Weights theta 1.0 / alpha 0.0 / beta
0.0 are fixed (`config.json` `flow_index`). 0.5 = at rest, 0.73 = +1 SD, 0.27 = -1 SD.
`flow_eeg_alt = sigmoid(z_theta - 0.5 * |z_alpha|)` (upper alpha 10-13 Hz at Cz C3 C4 P3
P4) is exploratory. `beta_rel`, `engagement_index` (Pope 1995 beta / (alpha + theta)),
`theta_alpha_ratio`, `z_beta` and the five `<band>_rel_mean` values are covariates.

Evidence (DOIs in the literature file): frontal theta higher in a demand-matched block than
in boredom - Katahira 2018, Ewing 2016, Fairclough & Ewing 2017, Knierim 2025, Rácz 2025,
Lin 2025; causal link Eschmann 2022. Nulls: Knierim 2018, Sugino 2024; hypofrontality in
expert improvisation: Rosen 2024. Alpha and beta have no replicated direction.

Implementation details the index file states and the code must keep:

- The index spectrum comes from the fixed analysis chain (selected notch + 1-45 Hz), not
  from the display band-pass, so the GUI-style 5-50 Hz display default never removes
  theta (`Pipeline._epoch`).
- Site sets (`FlowIndex.sites`): frontal = good channels in `FRONTAL_PREFERRED`, else in
  `FRONTAL_FALLBACK` (Fp), else all good; central-parietal = good channels in
  `CENTRAL_PARIETAL`, else all good; if no channel is good, all are used. The sets in use
  are exported as `frontal_sites` / `n_good`.
- An artifact epoch has `flow_eeg` null, is written to `epochs.csv`, is not in
  `eeg_hist`, and does not touch the baselines.
- `Pipeline.__init__` takes any weight missing from `config.json` from
  `flow_index.DEFAULT_WEIGHTS` (1.0 / 0.0 / 0.0); `session.json` records what ran.

Limits (detail in the index file, section 4): it is a proxy with a ceiling of about r 0.5-0.6
against self-report (Lin 2025); it inherits the difficulty confound (Sugino 2024 null);
the sign may reverse in expert creative tasks (Rosen 2024); on Fp1/Fp2 it is
blink-sensitive; no study validated a wrist device against it.

## The band's flow score (facts used here)

- The band publishes a flow score every 2 s together with HR, anomaly and flags (USB
  `[pub]` line; BLE STATE frame, 28/29 bytes, `docs/BLE_GATT.md` in the firmware repo).
- The Amplifly app stores one `wearable_samples` row per minute while connected
  (`flow_score`, `flow_state`, `anomaly`, `risk`, `mode`, `phase`, `poor_contact`, `asleep`,
  `usable`, `battery_mv`, `band`, `user_id`, `sampled_at`). HR is not stored, so `hr` is
  null with the Supabase source.
- Semantics carried into every sample dict: `held` = the band is showing a held value
  (HIGH_RISK display hold, or LOW_RISK with window state "unavailable"), not a fresh
  reading; `settling` = the band has HR but no flow yet (post-start / post-contact
  settling); `flow == null` = no reading ("no fake zero"). `window` is one of
  unavailable / closed / opening / open / ending.
- Held, settling, asleep, poor-contact and no-reading samples are stored and recorded but
  never paired and never enter a block mean.

## Design decisions (summary; details in `docs/DECISIONS.md`)

- stdlib `http.server` + Server-Sent Events, not websockets (ADR-3).
- Pure-Python Cyton parser, not BrainFlow: native DLLs are blocked on this laptop (ADR-1).
- Polling Supabase REST, not Realtime (ADR-2).
- One y-axis per chart, 0-1 for both flow scores (ADR-7).
- Analysis in the acquisition thread's listener, epochs by sample count (ADR-6).
- Frontal theta as the sole primary marker, alpha exploratory, beta a covariate (ADR-8).
- Fixed eyes-open rest baseline driven by the protocol, not a rolling one (ADR-9).
- Block-level inference with the manipulation check as a precondition (ADR-10).
- In-browser arithmetic task; the server keeps the clock (ADR-11).

## Risks and limitations

- Validity: both scores are indirect. A PASS says the band ordered the three conditions
  like the self-report did and co-varied with the EEG reference across blocks; a FAIL
  means the task induced the states (self-report says so) and the band did not track
  them; INCONCLUSIVE means the task did not work or the session lost too much data.
- Few blocks: six task blocks per session make the block rho coarse; conclusions come
  from the series, not one session.
- Cadence: with Supabase the band contributes 5-6 values per block; a dropped minute is
  visible in `n_band`. BLE gives 180 per block but takes the phone's connection.
- Baseline quality: a rest block with blinks or movement leaves the baseline unfrozen or
  biased; the exclusion rules catch the first, the artifact count hints at the second.
- Montage: the factory 8-ch montage measures theta at Fp1/Fp2 only (blink-sensitive);
  use `--montage flow`.
- Artefacts: thresholding only (no ICA / ASR / EMG gate); jaw tension in overload blocks
  raises beta and the 1-45 Hz total.
- Task: the staircase starts at a fixed level; per-block accuracy shows whether the flow
  block landed in 70-85 %. The arithmetic task may not induce flow in every participant;
  those sessions are INCONCLUSIVE by design, not FAIL.
- Clock alignment: EEG timestamps are host time re-anchored at 1 %/sample; Supabase rows
  carry the phone's `sampled_at`; block marks are server time. Keep phone and laptop on
  network time.
- Baseline drift of the band: a drifted band baseline shows as a steady offset
  (`band_minus_eeg_mean`); ordering and rank statistics are unaffected, level is not gated.
- Single BLE connection: `--band ble` means the phone app is not uploading that session.
- Privacy: raw EEG, task answers, self-reports and HR-derived scores are health data;
  sessions stay local.
