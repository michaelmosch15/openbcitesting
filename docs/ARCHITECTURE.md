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
                          2-s epochs: psd,         |            |              |
                          band_powers, quality     +------------+--------------+
                          FlowIndex (flow_index.py)             | on_band()
                          pairs + stats + history  <------------+
                                  |         ^
                                  |         | BandManager.switch() (app.py)
                 recorder.py      |         |
                 Recorder <-------+  eeg / epoch / band / mark
                 (writer thread)  |
                                  v
                        server.py App + ThreadingHTTPServer
                          GET /events (SSE), GET /api/*, POST /api/cmd
                                  |
                                  v
                        static/ index.html, app.js, style.css (render only)
```

`app.py` wires it together: loads `config.json`, picks the board (`find_dongle()` / `--board`
/ `SimBoard` / `UdpBoard`), builds `Recorder`, `Pipeline`, `BandManager` and `App`, starts
the server on 127.0.0.1 and opens the browser. `report.py` runs offline on a session folder.

## Data flow

EEG path:

1. `CytonBoard._reader` reads 33-byte packets, decodes 8 x int24 EEG to uV and 3 x int16
   accelerometer to g, counts dropped sample numbers, merges Daisy halves, and timestamps
   each sample with a board-rate clock gently re-anchored to host time (1 % per sample).
2. `SampleBuffer.push` stores the raw sample in the 30 s ring and calls every listener.
3. `Pipeline._on_samples` (still on the board thread) runs the stateful `FilterChain`
   (notch + band-pass) on the new samples, appends them to the pending list for the
   browser, forwards the raw block to the `Recorder`, and when `fs * epoch_s` samples have
   accumulated calls `_epoch()`.
4. `_epoch()` takes the last 2 s of raw data from the buffer, computes `quality()` on the
   raw window, filters a mirror-padded copy with a fresh stateless chain, computes the Hann
   `psd()` up to 60 Hz, `band_powers()` and `relative_powers()`, marks channels good only if
   enabled and `quality == good`, and calls `FlowIndex.update()`. The result becomes the
   `analysis` object and an `epochs.csv` row (with per-channel theta/alpha/beta absolute
   power and the quality state).
5. `App._tick` (SSE thread, every 100 ms) drains the pending filtered samples into a
   `samples` event, sends `analysis` when a new epoch exists, and `status` once a second.

Band path:

1. A band link thread produces one sample dict per band reading (`[pub]` line, BLE STATE
   frame, or Supabase row) and calls its listeners.
2. `Pipeline.on_band` stores it in `band_hist`, writes `band.csv`, and - if `flow` is not
   null, not `held` and not `asleep` - averages `eeg_hist` over `[t - pair_window_s, t]` to
   form one pair, then recomputes `stats` (`pearson`, `bootstrap_ci`, lag sweep, felt-flow
   correlations).
3. The server broadcasts the `band` event; the next `analysis` carries the new `stats`.

Marks: `Pipeline.mark` appends to `marks`, writes `marks.csv`, recomputes stats, and the
server broadcasts a `mark` event.

## Threads

| Thread (name) | Owner | Does |
|---|---|---|
| `cyton` / `sim` / `udp` | board object | reads packets or generates samples; pushes into `SampleBuffer`; runs the pipeline's filter and epoch work via the listener |
| `band` / `band-ble` / `band-supabase` / `simband` | band link | one at a time, swapped by `BandManager.switch()`; reconnect loops with back-off |
| `recorder` | `Recorder` | drains a `queue.Queue` of eeg/epoch/band/mark items to CSV; flushes when the queue is empty |
| `sse-tick` | `App` | the single analysis/transport loop: every 100 ms drains samples and pushes `samples`, `analysis`, `status` to all SSE subscribers |
| `http` + one thread per connection | `ThreadingHTTPServer` | JSON API and the long-lived `/events` connections (keepalive comment every 15 s) |
| main | `app.py` | waits for Ctrl-C, then stops board, band, recorder, server |

Shared state is protected by `SampleBuffer.lock` (ring), `Pipeline.lock` (filter chain +
pending list), `App.sub_lock` (subscriber list) and `BandManager._lock` (link swap).
`deque(maxlen)` histories are appended from one thread and read from another; a snapshot
is taken (`list(...)`, `np.array(...)`) before iterating.

## Timing

| Item | Cadence / size | Where set |
|---|---|---|
| Cyton packets | 250/s; 8 ch at 250 Hz or 16 ch at 125 Hz per channel | board |
| `SampleBuffer` | 30 s ring | `cyton.py` |
| Filtered samples to browser | `samples` event every 100 ms (~25 samples) | `server.py` `_tick` |
| Epoch (PSD, band powers, quality, flow index) | every `epoch_s` = 2 s | `config.json` |
| `analysis` event | on each new epoch (API.md says every 500 ms; it is gated by the epoch) | `server.py` |
| `status` event | every 1 s | `server.py` |
| Flow index baseline | rolling 150 epochs (5 min), warm-up 15 epochs (30 s) | `config.json` `flow_index` |
| Band `[pub]` (USB) / STATE notify (BLE) | every 2 s | firmware |
| Supabase rows | 1 per minute per wearer (written by the app); polled every `poll_s` = 5 s; initial backfill 30 min | `config.json` `supabase` |
| Sim band | every 2 s, 20 s lag, sigma 0.06 noise | `band_link.py` |
| Pair window | 60 s Supabase, 10 s BLE/USB/sim | `config.json` `pair_window_s` |
| Bootstrap CI | 500 resamples, from 8 pairs | `pipeline.py` |
| Lag sweep | -180 to +180 s, 10-s steps, from 8 pairs; positive = band lags EEG | `pipeline.py` |
| `r_last_5min` | pairs in the last 300 s | `pipeline.py` |
| Felt-flow pairing | EEG mean over 60 s before the mark; nearest band sample within 90 s | `pipeline.py` |
| History kept in memory | `history_hours` = 4 h of epochs, band samples and pairs | `config.json` |
| USB link timeouts | keepalive 60 s, no `[pub]` for 30 s -> reconnect; back-off 2 s x1.5 up to 20 s | `band_link.py` |
| BLE link timeouts | scan 8 s (12 s by address), no notify for 45 s -> reconnect; same back-off | `band_ble.py` |
| Supabase back-off | x2 per error up to 60 s | `supabase_link.py` |
| SSE keepalive | comment every 15 s | `server.py` |

## The EEG flow index

From `flow_index.py`:

> It is a PROXY, not ground truth. The literature on EEG correlates of flow (Katahira et
> al. 2018; Ewing et al. 2016; Nacke & Lindley 2008 and the Pope 1995 engagement index) is
> consistent on three points, and the index is built from exactly those:
>
> 1. frontal theta (4-8 Hz, midline/frontal sites) rises in flow vs. boredom AND overload;
> 2. alpha (8-13 Hz, parietal/occipital) is moderate - neither the high alpha of idling
>    nor the suppressed alpha of strain;
> 3. beta (13-30 Hz) is lower than in overload / anxious effort (engagement index
>    beta / (alpha + theta) peaks in overload, not in flow).
>
> Every 2-s epoch each component is z-scored against a rolling baseline of the wearer's own
> session (the band also judges against a personal baseline), and combined:
>
>     x        = w_theta * z(theta_frontal_rel)
>              - w_alpha * |z(alpha_rel)|                # "moderate" alpha: any departure costs
>              - w_beta  * z(beta_rel)
>     flow_eeg = sigmoid(1.5 * x)                        # 0.5 while every z is 0 (warm-up)
>
> with default weights 0.5 / 0.25 / 0.25 (config.json "flow_index"). The raw components are
> exported too, so report.py can tell which of them (if any) tracks the band's score.

Implementation details that the docstring does not say:

- The spectrum behind the index comes from a FIXED analysis filter (the selected mains notch
  + 1-55 Hz), not from the display band-pass, so the GUI-style 5-50 Hz display default never
  removes theta from the index (`Pipeline._epoch`).
- `theta_frontal_rel` is the mean relative theta over good channels whose label starts with
  `Fp`, `F`, `AF` or `Fz`; `alpha_posterior_rel` over `O`, `P`, `PO`, `Pz`; `beta_rel` over
  all good channels. If a group has no good channel it falls back to all good channels; if
  no channel is good, all are used.
- `engagement_index` = mean beta / (mean alpha + mean theta) in absolute power (Pope 1995);
  `theta_alpha_ratio` = frontal theta / frontal alpha in absolute power. Both are exported
  but not part of `flow_eeg`.
- During warm-up (first 15 epochs) all z are 0, so `flow_eeg = sigmoid(0) = 0.5`.

References: Katahira K. et al. (2018) "EEG correlates of the flow state: a combination of
increased frontal theta and moderate frontocentral alpha rhythm in the mental arithmetic
task", Front. Psychol. 9:300. Ewing K.C., Fairclough S.H., Gilleade K. (2016) "Evaluation
of an adaptive game that uses EEG measures validated during the design process as inputs to
a biocybernetic loop", Front. Hum. Neurosci. 10:223. Nacke L., Lindley C.A. (2008) "Flow and
immersion in first-person shooters", FuturePlay. Pope A.T., Bogart E.H., Bartolome D.S.
(1995) "Biocybernetic system evaluates indices of operator engagement in automated task",
Biol. Psychol. 40:187-195.

Limits:

- It is a proxy. There is no ground-truth flow signal; agreement between the band and this
  index is evidence that both track the same thing, not proof that either measures flow.
- Blink and eye-movement artefacts land on Fp1/Fp2 in the 1-8 Hz range and inflate
  "frontal theta". The quality check only catches railed/flat channels, not blinks.
- The default 8-channel Ultracortex montage (Fp1, Fp2, C3, C4, P7, P8, O1, O2) has only
  the Fp sites as "frontal". Use the 16-channel Daisy (adds F3, F4, F7, F8) or re-site two
  electrodes to F3/F4 and edit `labels_8` in `config.json`. Report the montage in every
  session (`session.json` stores the labels).
- The rolling 5-min baseline makes the index relative to the recent past: a long steady
  state drifts toward 0.5-0.6 even if the wearer stays in flow. The band has the same kind
  of personal baseline, which is one reason to compare them, but it also means the two can
  disagree about slow drifts.
- Weights are a prior, not fitted. See `docs/VALIDATION_PROTOCOL.md` for the rule on
  changing them.

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
- Held, asleep and no-reading samples are stored and recorded but never paired.

## Design decisions (summary; details in `docs/DECISIONS.md`)

- stdlib `http.server` + Server-Sent Events, not websockets: one-way streaming to one
  local browser needs no extra dependency and SSE reconnects by itself.
- Pure-Python Cyton parser, not BrainFlow: BrainFlow ships native DLLs that Smart App
  Control blocks on this laptop; the protocol is 33-byte packets and three commands.
- Polling Supabase REST, not Realtime: rows arrive once a minute; a 5-s poll with
  `created_at > last` is simpler than a websocket client and uses the same path the
  dashboard uses.
- One y-axis per chart, 0-1 for both flow scores, so the eye compares the shapes directly.
- Analysis in the acquisition thread's listener rather than a separate timer: the epoch is
  triggered by sample count, so it stays aligned to the board clock.

## Risks and limitations

- Validity: both scores are indirect. A PASS says they co-vary within a session; a FAIL
  may mean the band is wrong, the EEG proxy is wrong, or the task did not produce flow.
- Cadence mismatch: with Supabase the band contributes one value a minute, so a 20-min
  session yields ~20 pairs and the CI is wide. Prefer BLE for short sessions.
- Lag: wrist-derived state trails the brain; the lag sweep reports it but the primary
  gate uses lag 0 by pre-registration.
- Montage: the 8-channel headset is weak on frontal theta (above).
- Artefacts: blinks, jaw clench, electrode pops. Quality flags catch rails, not these.
- Clock alignment: EEG timestamps are host time re-anchored at 1 %/sample; Supabase rows
  carry the phone's `sampled_at`. A phone clock offset shifts pairs; keep phone and laptop
  on network time.
- Baseline drift: the band's own baseline can drift with noisy HR (seen in earlier
  sessions); a drifted band baseline shows as a steady offset in `band_minus_eeg_mean`.
- Single BLE connection: using `--band ble` means the phone app is not recording that
  session to Supabase.
- Privacy: raw EEG and HR-derived scores are health data; sessions stay local.
