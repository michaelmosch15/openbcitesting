# openbci-flow-validation

A localhost dashboard that streams an OpenBCI Cyton v3 (Ultracortex headset, 8 or 16
channels, RFduino USB dongle) into the browser the way the OpenBCI GUI does, computes an
EEG flow index from the band powers, and compares it live with the BioProxal wrist band's
flow score. The point is to find out, with pre-registered gates and an honest report,
whether the band's score tracks a literature-based EEG proxy of flow.

Python 3.14, numpy + pyserial (+ bleak for direct BLE). No pandas, no scipy, no BrainFlow,
no native DLLs (Windows Smart App Control blocks them on this laptop).

## Quick start (60 seconds)

1. Plug the OpenBCI dongle into USB. Set the dongle switch to **GPIO6** and the Cyton
   board switch to **PC**. Turn the board on (blue LED).
2. From this folder run `run.ps1` (PowerShell), `run.bat` (cmd), or `python app.py`.
   The scripts check `import numpy, serial` and print `pip install -r requirements.txt`
   if either is missing.
3. The server starts on http://127.0.0.1:8765 and the browser opens. Press **Start** in the
   header. The time series should move; the status bar shows the COM port, 8/16 ch and the
   packet count.
4. No hardware? `python app.py --sim` runs a synthetic 8-channel board plus a synthetic band
   (`SimBoard` in `cyton.py`, `SimBandLink` in `band_link.py`).

Command line (defaults come from `config.json`):

```
python app.py [--sim] [--port 8765] [--board COMx] [--channels auto|8|16]
              [--source serial|udp] [--udp-port 12345]
              [--band supabase|ble|usb|sim|none] [--user-id <uuid>] [--band-name BioProxal4]
              [--ble-address <mac>] [--no-browser] [--no-stream] [--config path/to/config.json]
```

`--no-stream` starts the server without sending `b` to the board; press Start in the browser.

Install once: `pip install -r requirements.txt` (`bleak` is only needed for `--band ble`).

## Band sources

The band's flow score can reach this tool three ways. Pick with `--band` or the band panel.

| Source | Path | Cadence | Use when |
|---|---|---|---|
| `supabase` (default) | band -> Amplifly phone app (BLE) -> Supabase `wearable_samples` -> this server polls REST every 5 s (`supabase_link.py`) | 1 row/min | The normal deployment path; the phone app stays open. Needs `%USERPROFILE%\.bioproxal\supabase.env` with `SUPABASE_URL` and `SUPABASE_SECRET_KEY` or `SUPABASE_PUBLISHABLE_KEY`. Pick the wearer in the band panel (`GET /api/wearers`). |
| `ble` | band -> this laptop over BLE with `bleak` (`band_ble.py`) | every 2 s | Finer time resolution. The band allows one BLE connection: close the Amplifly app (or turn the phone's Bluetooth off) first. Nothing is written to the band. |
| `usb` | band on a USB cable, console unlocked with the firmware repo's tools (`band_link.py`) | every 2 s (`[pub]` lines) | Bench use. Needs `usb.bioproxal_tools_dir` in `config.json` pointing at the firmware repo's `tools/` and the master key at `%USERPROFILE%\.bioproxal\master.key`. Read-only: no baseline reset, no log clear, no flash. |
| `sim` | synthetic | every 2 s | With `--sim`. The sim band is the simulator's hidden flow state plus noise and a 20 s lag. |
| `none` | - | - | EEG only. |

Pairs form at the band's cadence: Supabase averages the EEG index over the 60 s before each
row (`pair_window_s.supabase`); BLE/USB/sim use 10 s.

## Widgets

- **Time series** - filtered uV per channel, 5/10/20 s window, OpenBCI GUI channel colours,
  a railed / near-railed / flat flag per channel from the raw signal.
- **FFT** - 2 s Hann PSD per channel, uV^2/Hz, 0.5-60 Hz.
- **Band power** - delta/theta/alpha/beta/gamma, absolute and relative, per channel.
- **Head plot** - 10-20 positions for the configured labels, coloured by the selected band.
- **Flow comparison** - EEG flow index (every 2 s) and band flow (at the band's cadence) on
  one 0-1 axis, plus `n_pairs`, `r`, the 95 % CI, r over the last 5 min, MAE, best lag and
  the lag curve.
- **Band panel** - source, unit, connection state, last sample age, HR (BLE/USB only),
  HELD / settling / asleep / poor-contact flags, window state; wearer picker for Supabase;
  BLE scan list while not connected.
- **Felt-flow marks** - a 0-1 slider (`felt_flow`) and buttons `flow_start`, `flow_end`,
  `distracted`, `note`. Marks go to history and, while recording, to `marks.csv`.
- **Sessions + report** - list `sessions/`, run `report.py` on one, read `report.md` in place.

Channel exclusion (`set_channel`) removes a channel from the FFT/band/head widgets and from
the EEG flow index. Filter controls (`set_filter`) rebuild the notch + band-pass chain.

## Validation workflow

1. **Start** the stream; check every channel reads "good" (not railed/flat).
2. **Pick the wearer** in the band panel (Supabase) or connect BLE/USB. Wait until the band
   panel shows a fresh sample and the flow widget shows `baseline_ready` (about 30 s after
   the stream starts; the index sits at 0.5 until then).
3. **Record** with a short label (`[A-Za-z0-9_-]{0,32}`), for example `desk`.
4. **Work** on a real focused task for at least 20 min. Rate felt flow with the slider every
   ~5 min and at transitions; press `distracted` when you were.
5. **Stop** recording. Open **Sessions**, pick the session, press **Report**.

With the default Supabase source a 20-min session gives about 20 pairs (one per band row).
The pre-registered minimum is 20 usable pairs, so plan for 25-40 min.

## Reading the report

`report.py` reads `epochs.csv`, `band.csv` and `marks.csv` from the session folder, forms
pairs at the band's cadence (EEG flow averaged over `pair_window_s` before each band
timestamp, held/settling/no-reading rows excluded), and writes `report.md` + `report.json`.

Pre-registered primary gates, all three required for **PASS**:

| Gate | Value |
|---|---|
| Pearson r, band flow vs EEG flow index | r >= 0.5 |
| Bootstrap 95 % CI on r | lower bound > 0 |
| Usable pairs | n >= 20 |

Below 20 pairs the verdict is **INCONCLUSIVE**, not FAIL. A FAIL is reported as FAIL. The
secondary metrics (correlation of the band score with each exported component - `z_theta`,
`z_alpha`, `z_beta`, `engagement_index`, `theta_alpha_ratio` - best lag, in-flow agreement,
felt-flow correlations) are there to diagnose a result, not to change the verdict. See
`docs/VALIDATION_PROTOCOL.md`.

## Troubleshooting

| Symptom | Check |
|---|---|
| "no dongle found" / wrong COM | `find_dongle()` wants an FTDI port (VID 0x0403), preferring one that calls itself OpenBCI / FT231X. `python -c "import cyton; print(cyton.list_candidate_ports())"` lists what it sees. The band's own ESP32-S3 port (VID 0x303A) is never a candidate. Force with `--board COM7`. |
| "no '$$$' banner from the board" | Board switch to **PC**, dongle switch to **GPIO6**, board on, dongle paired to this board. Close the OpenBCI GUI if it holds the port, unplug/replug the dongle, retry. |
| Channel flagged **railed** / **near** | Raw signal above 90 % / 75 % of full scale (about 187 500 uV): electrode not on the scalp, or the SRB2/BIAS ear clips off. Press the electrode down; the flag should go to good. **flat** = open input. |
| 60 Hz peak in the FFT | The notch defaults to 60 Hz (`notch_hz`). In a 50 Hz country set the notch to 50 in the filter controls or `config.json`. |
| 16-ch: the top and bottom 8 labels look swapped | Flip `daisy_even_is_upper` in `config.json` (which half of the alternating Daisy packets carries channels 9-16). |
| Supabase `HTTP 401` | Key missing or wrong in `%USERPROFILE%\.bioproxal\supabase.env`. It never goes in the repo. |
| Supabase connects, no rows | Project paused (free tier) or the phone app is not uploading. Open the Amplifly app; check `/api/wearers` for a fresh `last_sampled_at`. |
| BLE "no 'BioProxal*' band advertising" | The band is connected to the phone. Close the Amplifly app or turn the phone's Bluetooth off; the band allows one connection. Also: band on, worn, laptop Bluetooth on. |
| USB "firmware tools not found" / "no USB master key" | Set `usb.bioproxal_tools_dir` to the firmware repo's `tools/`; the key must be at `%USERPROFILE%\.bioproxal\master.key`. |
| `ImportError` / DLL blocked | Windows Smart App Control blocks unsigned native DLLs (pandas, scipy, BrainFlow). This tool needs none of them. Run `python -m pip`, `python -m unittest` rather than `.exe` shims. |
| Flow index stuck at 0.5 | Baseline warm-up (first 15 epochs = 30 s): all z-scores are 0 until then. |

## Folder map

```
app.py              entry point, CLI, BandManager (swaps band sources), opens the browser
server.py           stdlib HTTP + SSE server (App, serve), SSE tick thread
pipeline.py         Pipeline: filters, 2-s epochs, PSD, flow index, pairing, stats, history
cyton.py            CytonBoard (serial driver), SampleBuffer, SimBoard, find_dongle
openbci_udp.py      UdpBoard: mirror the OpenBCI GUI Networking widget (UDP time series)
dsp.py              Biquad, FilterChain, psd, band_powers, relative_powers, quality
flow_index.py       FlowIndex, RollingZ: the EEG flow index and its components
band_link.py        BandLink (USB via firmware tools), SimBandLink
band_ble.py         BleBandLink, decode_state (BLE STATE frame)
supabase_link.py    SupabaseBandLink, SupabaseClient, row_to_sample
recorder.py         Recorder: session folders, CSVs from a writer thread
report.py           offline report with the pre-registered gates
static/             index.html, app.js, style.css (the browser only renders)
config.json         defaults (port, board, labels, filters, band source, weights, windows)
docs/               API.md, ARCHITECTURE.md, REQUIREMENTS.md, TEST_PLAN.md,
                    VALIDATION_PROTOCOL.md, DECISIONS.md
tests/              unittest modules: python -m unittest discover -s tests -v
sessions/           recordings (git-ignored)
run.ps1 / run.bat   launchers
```

## Privacy

Sessions hold raw EEG, heart-rate-derived scores and felt-flow ratings of real people.
`sessions/` is git-ignored and stays on this machine. Supabase and USB credentials live in
`%USERPROFILE%\.bioproxal\`, never in the repo; the browser never sees them (the server
talks to Supabase, the browser only gets decoded samples over localhost). The server binds
127.0.0.1 only. This tool never flashes, resets or writes to the band.
