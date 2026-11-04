# Decisions

Short architecture decision records. Status: accepted unless noted.

## ADR-1: Pure-Python Cyton driver instead of BrainFlow

Context: The Cyton is normally read through BrainFlow or the OpenBCI GUI. BrainFlow ships
native shared libraries; on this laptop Windows Smart App Control blocks unsigned DLLs
(it already blocks `pio.exe` and a pandas DLL). The protocol is small: `v` for a banner
ending in `$$$`, `b`/`s` to start/stop, `c`/`C` for 8/16 channels, `d` for defaults, and
33-byte packets `A0 | n | 8 x int24 | 3 x int16 | C0..CF`.

Decision: `cyton.py` implements the protocol with pyserial and numpy only (`CytonBoard`,
`SampleBuffer`, `find_dongle`). Daisy support merges alternating packets with a
configurable `daisy_even_is_upper`.

Consequences: no DLL risk, full control over timestamps and drop counting, easy to unit
test with byte strings. We lose BrainFlow's board catalogue (Ganglion, WiFi shield) and its
impedance-check helpers; neither is needed here. The `--source udp` path exists as a fallback
that lets the real OpenBCI GUI keep the board.

## ADR-2: Supabase polling as the default band source

Context: In deployment the band's score reaches us as band -> Amplifly app (BLE) -> Supabase
`wearable_samples`, one row per minute per wearer. Direct BLE from the laptop gives 2-s
samples but steals the band's single BLE connection from the phone, so nothing is uploaded
during the session. USB gives 2-s samples but needs the firmware tools, the master key and
a cable, which is a bench setup, not a wearer's day.

Decision: default `band_source` is `supabase`, polling `GET /rest/v1/wearable_samples` with
`created_at > last_seen` every 5 s, 30 min backfill on start. The 1/min cadence is accepted:
pairs form at that cadence with a 60-s EEG average, and the protocol asks for 25-40 min
sessions to reach 20 pairs. BLE and USB stay available for finer-grained sessions.

Consequences: validation runs on the same path and the same numbers users see; sessions are
longer; `hr` is null from Supabase. Polling, not Realtime (ADR-3's spirit): no websocket
client, and the dashboard site already reads the table the same way.

## ADR-3: stdlib HTTP + Server-Sent Events, not websockets

Context: One local browser needs a stream of samples (every 100 ms), analysis (every
epoch), status (1 s) and band/mark events, plus a handful of commands.

Decision: `server.py` uses `http.server.ThreadingHTTPServer`; `/events` is a `text/event-stream`
with named events; commands are `POST /api/cmd` JSON. No third-party web framework.

Consequences: zero extra dependencies; SSE reconnects by itself in the browser; one thread
per connection is fine for one or two tabs. Upstream traffic goes through separate POSTs,
which is acceptable at the command rate here. Not suitable for many remote clients, which
is not a goal (localhost only).

## ADR-4: The EEG flow index is a transparent proxy with exported components

Context: There is no agreed EEG ground truth for flow. Several studies agree on frontal
theta up, alpha moderate, beta not elevated (Katahira 2018; Ewing 2016; Nacke & Lindley
2008; Pope 1995 engagement index as the contrast).

Decision: `flow_index.py` computes exactly those three components, z-scores each against a
rolling personal baseline (150 epochs), combines them with fixed weights 0.5/0.25/0.25 and a
sigmoid, and exports every intermediate (`z_theta`, `z_alpha`, `z_beta`, `engagement_index`,
`theta_alpha_ratio`, relative powers, `n_good`). The formula is printed in the docs and the
weights are stored in `session.json`.

Consequences: anyone can see why the index moved; `report.py` can show which component the
band tracks even when the composite fails. The weights are a prior, not a fit; changing them
is governed by the protocol's honesty rules (re-run earlier sessions, say so).

## ADR-5: Pre-registered gates in `report.py`

Context: The owner's rule set: pre-register studies and report FAIL honestly. A live
dashboard invites reading the best-looking stretch.

Decision: The verdict is produced only by `report.py` from the session CSVs, with gates
fixed in code and in `docs/VALIDATION_PROTOCOL.md`: r >= 0.5, bootstrap 95 % CI lower bound
> 0, n >= 20 pairs; below 20 pairs the verdict is INCONCLUSIVE. Secondary metrics are
reported and never change the verdict. The report lists exclusions it applied.

Consequences: the live `stats` in the browser are for monitoring only; the same numbers
recomputed offline are the record. A gate change is a version change (CHANGELOG) and
invalidates comparisons across the change.

## ADR-6: Analysis runs on the acquisition thread, epochs by sample count

Context: An epoch timer could drift against the board clock; the board's own sample
counter cannot.

Decision: `Pipeline._on_samples` is the `SampleBuffer` listener; after `fs * epoch_s`
samples it runs `_epoch()` inline. The SSE tick thread only transports.

Consequences: epochs are aligned to the board; a slow epoch would delay packet reads, so
the epoch work is kept to one PSD and a few means (well under 2 s). Stats are recomputed on
the band thread when a pair forms.

## ADR-7: One y-axis per chart

Context: Twin axes make any two series look correlated.

Decision: The flow comparison draws the band score and the EEG index on one 0-1 axis; band
power and FFT charts use one axis per chart with a shared scale across channels.

Consequences: offsets between the two scores are visible rather than hidden by auto-scaling;
`band_minus_eeg_mean` is reported.
