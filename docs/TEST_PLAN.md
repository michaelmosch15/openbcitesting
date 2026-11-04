# Test plan

## 1. Unit tests

Run from the project folder:

```
python -m unittest discover -s tests -v
```

No hardware, network or credentials are needed; links are tested against fake serial,
fake BLE frames and fake HTTP responses.

| Module | Covers |
|---|---|
| `tests/test_cyton.py` | packet decode (`A0 ... C0`), uV scaling (`UV_PER_COUNT`), accelerometer scaling, dropped-count from sample-number gaps, resync after a stray byte, Daisy merge with `daisy_even_is_upper` both ways, banner parsing (Daisy detection, missing `$$$` error), `find_dongle` port scoring (FTDI preferred, ESP32-S3 excluded), `SampleBuffer.push/last` wrap-around, `SimBoard` output shape and rate |
| `tests/test_dsp.py` | `Biquad.notch` attenuates 60 Hz by > 20 dB and passes 10 Hz; `FilterChain` band edges; `psd` of a known-amplitude sine integrates to the right power; `band_powers`/`relative_powers` sum to 1; `quality` thresholds (railed 90 %, near 75 %, flat) |
| `tests/test_flow_index.py` | `RollingZ` warm-up returns 0 then z-scores; `FlowIndex.update` picks frontal/posterior channels by label and skips bad ones; rising frontal theta raises `flow_eeg`, rising beta lowers it; all exported keys present; weights from `set_flow_weights` take effect |
| `tests/test_band_ble.py` | `decode_state` on 28- and 29-byte frames: scaling of hr/spo2/anom/flow, `flow_no_reading` -> `flow is None`, `held` rules (mode 1; mode 0 + window unavailable), `settling`, `asleep`, `poor_contact`, window names, `learn_pct` / `low_battery`; short frame raises |
| `tests/test_supabase_link.py` | `load_env` parsing and errors; `iso_to_epoch` with 1-6 fractional digits and `Z`; `row_to_sample` (held = mode 1, null flow, window names); `SupabaseClient.wearers()` aggregation and ordering with a stubbed `get`; `SupabaseBandLink._filters` with/without user and band; polling with a fake client only requests `created_at > last` |
| `tests/test_recorder.py` | `start` creates the folder, `session.json` and headers; eeg/epoch/band/mark rows land in the right CSV with the right formatting; `epochs.csv` header is taken from the first record; `stop` closes files and returns the dir; writes after `stop` are dropped |
| `tests/test_report.py` | pairing from `epochs.csv` + `band.csv` at the pair window; held/settling/null rows excluded; `pearson` and `bootstrap_ci` on known data; gate logic: PASS, FAIL (low r / CI crossing 0) and INCONCLUSIVE (< 20 pairs); `report.md` and `report.json` are written; felt-flow section with and without marks |

`tests/_path.py` puts the project root on `sys.path` for the test modules. There is no unit
test for the USB `BandLink` because it imports the firmware repo's `offload.py`; it is
covered by M-7b.

## 2. Integration test with `--sim`

```
python app.py --sim
```

Check in the browser, in this order:

1. Status bar: board `sim`, 8 ch, 250 Hz, `packets` increasing by ~250/s, `dropped` 0.
2. Time series: 8 traces moving; a blink-shaped bump appears on Fp1/Fp2 now and then.
3. FFT: peaks at 6, 10 and 20 Hz; the 60 Hz line is suppressed with the notch on, visible
   with notch 0 (set via the filter control, then set it back to 60).
4. Band power: theta highest on channels 1-2 (frontal), alpha highest on 5-8 (posterior).
5. Quality: all channels `good`.
6. Flow comparison: `baseline_ready` after 30 s; the EEG index then follows a slow
   ~4-min oscillation; the band trace (every 2 s) follows the same shape about 20 s later.
7. Band panel: source `sim`, unit `BioProxalSIM`, connected, `age_s` under 3.
8. After ~10 min: `n_pairs` about 300, `r > 0.7`, CI lower bound > 0, `best_lag_s` about
   +20 (band lags EEG). The sim band is a lagged, noisy copy of the hidden state that
   drives the sim EEG, so a lower r means the pipeline, not the sim, is broken.
9. Marks: move the felt-flow slider three times; `felt.n` becomes 3 and `r_band`/`r_eeg`
   appear.
10. Record for 3 min with label `simtest`, stop, open Sessions, run Report. Expect
    INCONCLUSIVE if fewer than 20 pairs (3 min at 2 s = ~90 pairs, so expect PASS here),
    `report.md` shown in the page and present in the session folder.
11. Set channel 1 disabled: it disappears from FFT/band/head widgets, `n_good` drops to 7.
12. Switch band source to `none` and back to `sim`: the band panel updates, pairs resume.

## 3. Manual hardware checks

| ID | Check | Steps | Pass when |
|---|---|---|---|
| M-1 | Dongle detection | Dongle in, board off. `python app.py --no-browser`. | Log shows `[cyton] COMx:` with the chosen port; with the dongle unplugged the log says to plug it in / pass `--board` / use `--sim`, and the server still starts. |
| M-2 | Banner | Board on, switch PC, dongle GPIO6. Start the app. | Log shows the banner lines ("OpenBCI V3 8-16 channel", "ADS1299 Device ID", `$$$`); with the board switch OFF the error says no `$$$` banner and names the port. |
| M-3 | Railed check | Streaming, headset on the desk, then on the head. Touch one electrode to the scalp, let go. | Off-head: channels `railed` or `near`; on-head with ear clips: `good`; releasing the electrode flips it back. Flag colour matches the OpenBCI GUI convention. |
| M-4 | Mains notch | Streaming on-head. Set notch 0, then 60 (or 50). | FFT shows a 60 Hz peak at notch 0, flattened with notch 60. In a 50 Hz country the peak is at 50 and needs notch 50. |
| M-5 | Daisy order | 16-ch board. Tap the electrode labelled F3 (channel 11). | The F3 trace moves. If a trace in the first 8 moves instead, flip `daisy_even_is_upper` and retry. |
| M-6 | Supabase wearer pick | Phone app connected to the band, uploading. `--band supabase`. Open the band panel. | The wearer list shows the right short id and band name with `age_s` under 120; picking it makes `band` samples arrive about once a minute; `pair_window_s` is 60. With a wrong key the panel shows `HTTP 401` and the EEG keeps streaming. |
| M-7a | BLE direct | Close the Amplifly app. `--band ble`. | Log: `found BioProxalN @ mac, connecting` then `subscribed to STATE`; samples every 2 s with HR; `pair_window_s` 10. With the phone app open the error says no band advertising and lists what was seen. |
| M-7b | USB | Band on the cable, `usb.bioproxal_tools_dir` set, master key present. `--band usb`. | Log: `[band] COMx unlocked (BioProxalN)`; `[pub]` samples every 2 s; closing the app sends LOCK (firmware console locks again). No baseline reset or log clear happened (check the band's log count before/after). |
| M-8 | Record / report round trip | Any source. Record 5 min with label `m8`, add two felt-flow marks, stop, Report. | `sessions/<stamp>_m8/` has `session.json`, `eeg_raw.csv` (~75 000 rows for 5 min at 250 Hz), `epochs.csv` (~150 rows), `band.csv`, `marks.csv` (2 rows), `report.md`, `report.json`; the verdict is INCONCLUSIVE with Supabase (5 pairs) and the markdown shows the gates and n. |
| M-9 | UDP mirror | OpenBCI GUI streaming, Networking -> UDP, TimeSeriesRaw, 127.0.0.1:12345, Start. `python app.py --source udp`. | Status bar: board `udp`, packets increasing; time series matches the GUI. |
| M-10 | Localhost only | From another device on the LAN open `http://<laptop-ip>:8765`. | Connection refused. |

## 4. Release checklist

- [ ] `python -m unittest discover -s tests -v` passes on this laptop (3.14).
- [ ] CI green on Ubuntu + Windows, Python 3.12 (3.14 may be allowed to fail).
- [ ] `node --check static/app.js` passes.
- [ ] Integration test with `--sim` completed through step 12.
- [ ] M-1 to M-4 and M-8 done on the real headset; M-6 or M-7a done with the real band.
- [ ] `docs/API.md` matches the events and routes the browser uses.
- [ ] `config.json` defaults reviewed (notch for the country, labels for the montage).
- [ ] `CHANGELOG.md` updated; version in `server.py` `SERVER_VERSION` matches.
- [ ] `git status` shows no `sessions/`, `*.env`, `*.key` files staged.
- [ ] No flow-index weight change since the last validation sessions, or the sessions were
      re-run and the protocol results log says so.
