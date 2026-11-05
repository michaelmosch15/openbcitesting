# openbci-flow-validation

A localhost experiment runner and dashboard that streams an OpenBCI Cyton v3 (Ultracortex
headset, 8 or 16 channels, RFduino USB dongle) into the browser, runs a difficulty-ladder
protocol (boredom / flow / overload mental arithmetic with a Flow Short Scale self-report
after every block), computes a literature-grounded EEG flow index (frontal theta against
an eyes-open rest baseline), and scores the BioProxal wrist band's flow score against both
per block, with pre-registered gates and an honest report.

Python 3.12+, numpy + pyserial (+ bleak for direct BLE). No pandas, no scipy, no BrainFlow,
no native DLLs (Windows Smart App Control blocks them on this laptop).

## Quick start

1. Plug the OpenBCI dongle into USB. Dongle switch to **GPIO6**, Cyton board switch to
   **PC**, board on (blue LED). Headset on the participant, ear clips (SRB2, BIAS) on.
   Band on the participant's wrist for at least 10 min, phone app connected and uploading.
2. From this folder: `python app.py --montage flow` (or `run.ps1` / `run.bat`, which also
   check `numpy` and `serial`). The browser opens http://127.0.0.1:8765.
3. Press **Start**. Every channel must read `good` (not railed / near / flat); re-seat
   electrodes until it does. Check the FFT has no mains peak (notch 60, or 50).
4. In the band panel **pick the wearer** (Supabase) and wait for a fresh sample.
5. In the experiment panel pick `difficulty_ladder` and press **START PROTOCOL**.
   Recording starts by itself. Follow the screen: 3 min rest eyes open (the baseline),
   1 min eyes closed, then six 6-min arithmetic blocks in a counterbalanced order with a
   self-report dialog and a 60-s pause after each, then 2 min rest. About 47 min.
6. Press Stop recording. Open **Sessions**, pick the session, press **Report**. Copy the
   verdict into the results log in `docs/VALIDATION_PROTOCOL.md`.

No hardware? `python app.py --sim` runs a synthetic board and band that follow the
protocol's conditions; use the `sim_quick` protocol (about 9 min).

Command line (defaults from `config.json`):

```
python app.py [--sim] [--montage default|flow] [--port 8765] [--board COMx]
              [--channels auto|8|16] [--source serial|udp] [--udp-port 12345]
              [--band supabase|ble|usb|sim|none] [--user-id <uuid>] [--band-name BioProxal4]
              [--ble-address <mac>] [--astra on|live|off] [--no-browser] [--no-stream]
              [--config path/to/config.json]
```

Install once: `pip install -r requirements.txt` (`bleak` only for `--band ble`).

## Montage: put electrodes where the index needs them

The index is frontal theta (4-7 Hz). The Ultracortex factory 8-channel montage (Fp1 Fp2
C3 C4 P7 P8 O1 O2) has only the prefrontal Fp sites, which blinks dominate. Move the
electrodes to **Fz F3 F4 Cz C3 C4 P3 P4** and start with `--montage flow` (the labels in
`config.json` `labels_8_flow`): theta is then read at Fz/F3/F4 and the upper-alpha and
engagement covariates at Cz/C3/C4/P3/P4. With a 16-channel Daisy the frontal set is
F7 F8 F3 F4. The headset as built on 2026-11-05 has all eight combs in two rows across the
forehead band: start with `--montage frontal` (`labels_8_frontal`: Fp1 Fp2 AF3 AF4 F3 F4 F7
F8); every channel then counts as frontal for the index, and the exact channel-to-position
order is confirmed by tapping each comb while watching the time series. The montage is
recorded in `session.json` and in the protocol header; it is part of the pre-registration.
Details: `docs/EEG_FLOW_INDEX.md` section 5.

## What the EEG index is

`flow_eeg = sigmoid(z_theta)`: relative power of 4-7 Hz frontal theta over the 1-45 Hz
total, from a 10-s Welch spectrum (2-s Hann segments, 50 % overlap, segments above
150 uV peak-to-peak or 100 uV absolute dropped), log, z-scored against the mean and SD of
the 3-min eyes-open rest block, updated every 2 s. 0.5 = at rest, 0.73 = one resting SD
above, 0.27 = one below. Weights are theta 1.0 / alpha 0 / beta 0 and are frozen before
data collection; `flow_eeg_alt` (theta minus 0.5 x |upper-alpha deviation|) is exploratory.
Why only theta, what the index can and cannot claim, and every number: `docs/EEG_FLOW_INDEX.md`;
the research review: `docs/EEG_FLOW_LITERATURE.md`.

## Band sources

| Source | Path | Cadence | Use when |
|---|---|---|---|
| `supabase` (default) | band -> Amplifly phone app (BLE) -> Supabase `wearable_samples` -> polled every 5 s (`supabase_link.py`) | 1 row/min (5-6 per 6-min block) | The normal deployment path. Needs `%USERPROFILE%\.bioproxal\supabase.env` with `SUPABASE_URL` and a key. Pick the wearer in the band panel. |
| `ble` | band -> this laptop with `bleak` (`band_ble.py`) | every 2 s | Finer resolution. The band allows one BLE connection: close the Amplifly app first. Nothing is written to the band. |
| `usb` | band on a cable through the firmware repo's tools (`band_link.py`) | every 1 s | The band's whole console once unlocked: HR, SpO2, RMSSD, SDNN, perfusion, respiration, anomaly, risk, published flow, base HR, raw flow, forecast heads, battery, mode / phase. Needs `%USERPROFILE%\.bioproxal\master.key`; the firmware tools are found at `<ancestor>/worktrees/v0629/tools` or set `usb.bioproxal_tools_dir`. Read-only; every line is kept (`band_console.log`, band panel console). Raw red/IR PPG is not exported by a production band. |
| `sim` | synthetic | every 2 s | With `--sim`: the simulator's hidden state, which follows the protocol's conditions, plus noise and a 20-s lag. |
| `none` | - | - | EEG and self-report only. |

## Astra: live analyst and setup helper

Astra is the owner's OpenAI reviewer persona (model `gpt-6-astra`; the key is `OPENAI_KEY` in
`BioProxal2026/.env`, the folder above this project, same as the firmware's
`tools/astra_review.py`). Two panels above Sessions:

- **Live analyst** - press *start live* (or run `--astra live`). Every 60 s (configurable)
  Astra reads the last 5 minutes - EEG index, band flow with its flags, the pairs and their
  correlation, block means, ordering and block rho, task accuracy, channel quality, baseline
  state - compares it with what it said before, and writes one `[OK]` / `[WATCH]` / `[ISSUE]`
  note. *Run analysis* gives a structured interim read toward gates G1-G3. Ask it questions in
  the box. While recording, every note lands in `astra_notes.md` and `astra_notes.jsonl` in
  the session folder. Astra is observational: its prompt forbids suggesting changes to the
  index, thresholds, gates or schedule mid-session.
- **Setup helper** - a live channel table (quality, % of full scale, raw peak, relative theta
  and alpha) plus a chat that knows 10-20 placement, the Mark IV, the flow montage, the Cyton
  switches, the quality flags and the band links, and sees the table with every question.
  *Check my setup* walks channel by channel and ends READY / NOT READY.

- **In-flight fixes** - describe what is wrong while the app runs ("the band panel shows age
  0 s when the link is stale"). Astra reads the source it needs, sends a minimal patch, the
  server runs `py_compile`, `node --check` and the unit tests, keeps before/after copies under
  `astra_fixes/`, and hot-swaps the changed module into the running process (static files are
  served fresh on the next page load; `app.py` / `config.json` changes show a *restart server*
  button, refused while a protocol or recording runs). A failed verification is rolled back
  and Astra gets one more try; *revert last fix* restores the previous files. Pre-registered
  parameters (protocol durations, index weights, artifact thresholds, gates) stay locked
  unless you tick "may change pre-registered parameters" for that job.

No `.env` with the key: the panels say so and stay disabled; everything else works.
`--astra off` removes them. Settings in `config.json` `astra` (`live_interval_s`,
`window_min`, `max_tokens`, `memory_turns`, `auto_live`). Nothing but the prompt text (numbers
from the dashboard, your questions) leaves the machine; raw EEG never does.

## Reading the report

`report.py` scores the session per block from `epochs.csv`, `band.csv`, `marks.csv`
(block and `rating` marks) and `task.csv`, then applies the pre-registered gates of
`docs/VALIDATION_PROTOCOL.md` section 7:

| Gate | Threshold |
|---|---|
| G1 manipulation check | Flow Short Scale condition mean flow > boredom and flow > overload; `demand` item boredom < flow < overload |
| G2 band ordering | band condition mean highest in flow vs both other conditions |
| G3 block agreement | Spearman rho between the band's and the EEG index's block means >= 0.5 over >= 6 task blocks |

**PASS** = G1, G2 and G3. **FAIL** = G1 holds, G2 or G3 does not (the task worked, the
band did not track it). **INCONCLUSIVE** = G1 fails or a session-level exclusion applies
(baseline never frozen, fewer than 6 usable blocks, missing ratings). The EEG reference
check (theta index flow > boredom) is printed next to the verdict and does not change it.
Epoch-level r, the lag sweep and felt-flow correlations from v0.1 remain as secondary
numbers. A FAIL is reported as FAIL.

## Widgets

Experiment panel (protocol picker, START / NEXT / STOP, schedule strip, instruction, task,
self-report status); time series (5/10/20 s, quality flags); FFT; band power; head plot;
flow comparison (band and EEG index on one 0-1 axis, block means per condition, block rho,
secondary pair statistics); band panel (source, unit, HR on BLE/USB, held / settling /
asleep / poor-contact, wearer picker); felt-flow marks; sessions + report. Channel
exclusion (`set_channel`) removes a channel from the widgets and from the index; filter
controls change only the display, never the index.

## Troubleshooting

| Symptom | Check |
|---|---|
| "no dongle found" / wrong COM | `find_dongle()` wants an FTDI port (VID 0x0403). `python -c "import cyton; print(cyton.list_candidate_ports())"` lists ports. The band's ESP32-S3 port (VID 0x303A) is never a candidate. Force with `--board COM7`. |
| "no '$$$' banner from the board" | Board switch **PC**, dongle switch **GPIO6**, board on, dongle paired to this board. Close the OpenBCI GUI, replug the dongle, retry. |
| Channel **railed** / **near** / **flat** | Electrode not on the scalp or ear clips off (railed / near); open input (flat). Re-seat. |
| 60 Hz peak in the FFT | Set the notch to 50 in a 50-Hz country (filter controls or `notch_hz`). |
| 16-ch labels look swapped | Flip `daisy_even_is_upper` in `config.json`. |
| Supabase `HTTP 401` / no rows | Key in `%USERPROFILE%\.bioproxal\supabase.env`; project paused or phone app not uploading - check `/api/wearers`. |
| BLE "no 'BioProxal*' band advertising" | The band is connected to the phone; close the Amplifly app or turn the phone's Bluetooth off. |
| Flow index stuck at 0.5 | First 15 epochs (30 s), or every epoch is an artifact (`artifact_epochs` in the flow widget rising): blinks, movement, bad contact. |
| `baseline_state` never `rest-frozen` | The rest block had fewer than 15 clean epochs. The session is INCONCLUSIVE; fix contact / stillness and run again. |
| `frontal_sites` shows `Fp1,Fp2` | Factory montage labels. Move the electrodes and start with `--montage flow`, or accept an Fp-only session and label it so. |
| Report says INCONCLUSIVE "manipulation failed" | The ratings did not put flow highest. The task did not induce the states for this person; not a band result. |
| `ImportError` / DLL blocked | Smart App Control blocks native DLLs; this tool needs none. Use `python -m pip`, `python -m unittest`. |

## Folder map

```
app.py              entry point, CLI (--montage), BandManager, wires the protocol runner
server.py           stdlib HTTP + SSE server (App, serve); protocol events drive the baseline
protocol.py         ProtocolRunner: counterbalanced block schedule, timer thread, ratings
protocols.json      difficulty_ladder, sim_quick, free_work
pipeline.py         Pipeline: filters, 10-s Welch index spectrum, artifact rule, pairs, block stats
flow_index.py       FlowIndex, Baseline (rest / session / rolling), site sets
dsp.py              Biquad, FilterChain, psd, welch, band powers, quality
cyton.py            CytonBoard, SampleBuffer, SimBoard (follows protocol conditions), find_dongle
openbci_udp.py      UdpBoard: mirror the OpenBCI GUI Networking widget
band_link.py        BandLink (USB), SimBandLink;  band_ble.py  BleBandLink;  supabase_link.py
recorder.py         Recorder: session folders, CSVs (eeg_raw, epochs, band, marks, task)
report.py           offline per-block report with the pre-registered gates
astra.py            Astra (gpt-6-astra): LiveAnalyst notes / analysis / Q&A, SetupHelper chat
fixer.py            Astra in-flight fixes: read / patch / verify loop, records, revert, restart
hotreload.py        swap a reloaded module into live instances and sibling imports
static/             index.html, app.js (task + rating dialog + Astra panels), style.css
config.json         defaults: port, board, labels_8 / labels_8_flow / labels_16, filters,
                    flow_index (weights, baseline, artifact thresholds, Welch window)
docs/               EEG_FLOW_LITERATURE, EEG_FLOW_INDEX, VALIDATION_PROTOCOL, API,
                    ARCHITECTURE, REQUIREMENTS, TEST_PLAN, DECISIONS
tests/              python -m unittest discover -s tests -v
sessions/           recordings (git-ignored)
```

## Privacy

Sessions hold raw EEG, heart-rate-derived scores, task answers and self-reports of real
people. `sessions/` is git-ignored and stays on this machine. Credentials live in
`%USERPROFILE%\.bioproxal\`, never in the repo; the browser never sees them. The server
binds 127.0.0.1 only. This tool never flashes, resets or writes to the band.
