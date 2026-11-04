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

Status: the transparency decision stands; the weights, the baseline and the citation list
are superseded by ADR-8 and ADR-9 (v0.2.0).

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

Status: the principle stands; the epoch-level gates (r >= 0.5, CI > 0, n >= 20 pairs) are
superseded by the block-level gates of ADR-10 (v0.2.0) and remain as secondary output.

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

## ADR-8: Frontal theta is the sole primary marker (v0.2.0)

Context: The literature review (`docs/EEG_FLOW_LITERATURE.md`) found exactly one EEG marker
of flow that replicates across independent labs: frontal / fronto-midline theta (4-7 Hz)
higher in a demand-matched block than in a boring one - Katahira 2018
(10.3389/fpsyg.2018.00300), Ewing 2016 (10.3389/fnhum.2016.00223), Knierim 2025
(10.1145/3706598.3713512), Rácz 2025 (10.1038/s41598-025-95647-x), Lin 2025
(10.1007/s11571-024-10205-x), with a causal link from Eschmann 2022
(10.1007/s41465-021-00236-1). Alpha's direction depends on site, sub-band and baseline
(rises in Katahira, falls in Ewing and Knierim 2018, null in Knierim 2025, flips with the
rest condition in Rácz); only "flow is intermediate" recurs. Beta's direction is
contradictory across five studies and beta is the band most contaminated by EMG in an
overload block. The v0.1 index (0.5 theta / 0.25 moderate alpha / 0.25 low beta, citing
Nacke & Lindley 2008) mixed one replicated marker with two that are not, and Nacke &
Lindley is not in the reviewed evidence base.

Decision: `flow_eeg = sigmoid(z_theta)`: weights 1.0 / 0.0 / 0.0 (`config.json`
`flow_index`, `flow_index.DEFAULT_WEIGHTS`), where `z_theta` is the log relative power of
4-7 Hz over the 1-45 Hz total at the frontal set (Fz F3 F4 ... with Fp fallback), z-scored
against the rest baseline (ADR-9). The alpha term survives only as the exploratory
`flow_eeg_alt = sigmoid(z_theta - 0.5 * |z_alpha|)` with upper alpha 10-13 Hz at central /
parietal sites; beta and the Pope 1995 engagement index are exported covariates. The
sigmoid gain is 1 (was 1.5). The full specification is `docs/EEG_FLOW_INDEX.md`; its
section 6 freezes these numbers before data collection.

Consequences: the primary index is a monotone transform of one quantity, so every rank
statistic is a statistic of frontal theta, and a reviewer can audit it in one table. It is
expected to separate flow from boredom and not reliably from overload (Katahira: equal;
Ewing: lower), which the protocol accepts (H2 is flow > boredom only). It inherits the
difficulty confound and the Sugino 2024 null in expert gamers. On the factory 8-ch montage
it is measured at Fp1/Fp2 and is blink-sensitive, hence the `--montage flow` preset.
`flow_eeg_alt` may not be promoted to primary post hoc within a series.

## ADR-9: Fixed eyes-open rest baseline instead of a rolling one (v0.2.0)

Context: v0.1 z-scored every component against the last 150 epochs (5 min). A rolling
baseline subtracts exactly the slow between-block shifts the validation is looking for: a
wearer who stays in flow for 6 min drifts back toward 0.5, and the index of a flow block
is then relative to the preceding pause rather than to rest. Katahira 2018 and Eschmann
2022 normalised to rest; Rácz 2025 showed the boredom-vs-flow alpha result flips sign
between an eyes-closed and an eyes-open reference, so the reference must be fixed and
never mixed.

Decision: `flow_index.Baseline` mode `rest`: the protocol block flagged `baseline` (the
3-min eyes-open rest at the start of `difficulty_ladder`) collects the mean / SD of each
log component via `FlowIndex.start_rest()` / `end_rest()`, called by `App.on_protocol_event`
on that block's `block_start` / `block_end`; the statistics freeze at block end if at least
15 clean epochs were collected. Until then (and in sessions without a protocol) an
expanding session mean / SD is used. Artifact epochs never enter the baseline. The
`rolling` mode is kept in code, unused, for comparison runs only. The eyes-closed minute
is a reactivity check and does not enter the baseline.

Consequences: a session is only interpretable after a good rest block; a rest block with
too many artifacts leaves the baseline unfrozen and the session INCONCLUSIVE. The index no
longer drifts during a long steady state. The index is per session; cross-session values
are not comparable and the report compares blocks within a session only. The band's own
baseline is personal and adaptive, so a steady offset between the two is expected and is
not a gate.

## ADR-10: Block-level inference with the manipulation check as a precondition (v0.2.0)

Context: v0.1 gated on Pearson r over epoch pairs. Those pairs are not independent (a
10-s index window stepped every 2 s; a 60-s average per Supabase row), so n was inflated
and the CI too narrow; and with a free task nothing guaranteed that any flow occurred.
Every study in the literature file that found an effect scored per block within
participants after a per-person difficulty manipulation, and Durcan 2024
(10.1038/s44271-024-00115-3, items 6a, 7a) requires the manipulation check to pass before
any physiological claim.

Decision: the block is the unit. `Pipeline._block_stats` (live, for monitoring) and
`report.py` (the record) compute per-block means of the EEG index, the band's usable
samples and the Flow Short Scale, per-condition means, their ordering, and the Spearman
rho between the band's and the EEG's block means. Gates (`docs/VALIDATION_PROTOCOL.md`
section 7): G1 manipulation (FSS flow > boredom and > overload; `demand` ordered) is a
precondition - if it fails the session is INCONCLUSIVE, not FAIL; G2 band condition mean
highest in flow vs both others; G3 block rho >= 0.5 with >= 6 blocks. PASS = G1, G2, G3;
FAIL = G1 holds and G2 or G3 fails. The EEG reference check (H2) is reported, not gated.
Epoch-level r, lag sweep and felt-flow correlations stay as secondary output.

Consequences: one session yields 6 task blocks, so a single-session rho is coarse (with
n = 6 only strong agreement reaches 0.5) and the series-level criteria aggregate across
sessions and participants. The self-report is the criterion for both instruments; the EEG
index is a convergent reference. A FAIL now means "the task induced the states and the
band did not track them", which is the claim the owner wants tested.

## ADR-11: In-browser mental-arithmetic task (v0.2.0)

Context: the paradigm needs a task whose difficulty is objectively controllable per person,
has one clear goal, little head and jaw movement, and immediate feedback (Durcan items 2a,
3a-c). Katahira 2018 and Knierim 2018 used adaptive two-number addition; Tetris (Ewing,
Rácz) gives stronger flow reports but more movement and needs external software.

Decision: the task runs in the browser (`static/app.js`), chosen by the block's `task`
field: `arithmetic_easy` (single-digit sums, one every 4 s), `arithmetic_adaptive`
(levels 1-12 from level 4; two correct -> up, one wrong -> down; 10-s soft limit; target
70-85 % correct), `arithmetic_hard` (level 12, 6-s hard limit). Every answer is posted as
`trial`; the server stamps the current condition and writes `task.csv`. The server keeps
the clock (`ProtocolRunner`); the browser only renders the block and the task.

Consequences: no extra software or licence; the same task for every participant; accuracy
per block is logged, so H1c (about 100 % / 70-85 % / low) is checked from data rather than
assumed. The staircase starts at a fixed level 4 instead of a pre-test level (Katahira
used the last 25 % of a practice run); the 2-min practice in the set-up checklist and the
first minute of the block absorb the difference, and the per-block accuracy shows whether
it did. Typing adds small hand movement (less than a game controller). If the arithmetic
task fails the manipulation check for a participant group, a Tetris-style task is a new
protocol entry, not a change to this one.
