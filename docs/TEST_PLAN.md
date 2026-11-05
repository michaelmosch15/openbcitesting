# Test plan

## 1. Unit tests

Run from the project folder:

```
python -m unittest discover -s tests -v
```

No hardware, network or credentials are needed; links are tested against fake serial,
fake BLE frames and fake HTTP responses; the protocol runner is tested with sub-second
blocks.

| Module | Covers |
|---|---|
| `tests/test_cyton.py` | packet decode (`A0 ... C0`), uV scaling (`UV_PER_COUNT`), accelerometer scaling, dropped-count from sample-number gaps, resync after a stray byte, Daisy merge with `daisy_even_is_upper` both ways, banner parsing (Daisy detection, missing `$$$` error), `find_dongle` port scoring (FTDI preferred, ESP32-S3 excluded), `SampleBuffer.push/last` wrap-around, `SimBoard` output shape and rate |
| `tests/test_dsp.py` | `Biquad.notch` attenuates 60 Hz by > 20 dB and passes 10 Hz; `FilterChain` band edges; `psd` of a known-amplitude sine integrates to the right power; `band_powers` / `relative_powers` sum to 1; `quality` thresholds (railed 90 %, near 75 %, flat). **Gap (v0.2):** `welch` (segment count 9 for 10 s / 2 s / 50 %, averaging, the `reject` callback, the all-rejected NaN case) and `band_powers_of` with the 1-45 Hz total have no unit test yet; they are exercised only through the `--sim` integration ordering (section 2) and `TestArtifacts` in `test_flow_index.py` (which feeds the artifact flag directly). Add `TestWelch` before the first real series. |
| `tests/test_flow_index.py` | `Baseline`: warm-up returns 0 until 15 values; `rest` mode collects during `start_rest` / `end_rest`, freezes mean / SD, refuses to freeze on a too-short rest, behaves like `session` until frozen, unit SD on a constant block; `session` expands and never forgets; `rolling` forgets and diverges from `session` after a shift. `FlowIndex`: default weights are theta-only; warm-up gives 0.5; constant input gives 0.5 and neutral components; frontal theta up raises `flow_eeg`; theta elsewhere does not count; beta only matters with a beta weight; alpha deviation lowers `flow_eeg_alt` either way; primary = sigmoid of the weighted z; `theta_fm` / `alpha_hi` from `extra` feed theta and alpha; `last` kept. Artifacts: an artifact epoch has `flow_eeg` None and does not touch the baselines, also during rest. Rest protocol: the rest block freezes and later epochs are scored against it; `baseline_state` strings. Sites: 16-ch picks F7 F8 F3 F4 and C3 C4 P3 P4; 8-ch factory falls back to Fp1 Fp2 and C3 C4; bad frontal channels fall back to Fp; a disabled channel leaves the frontal mean; no frontal left -> all good; all bad -> all channels |
| `tests/test_protocol.py` | `build`: seed reproducible and logged, clock seed when none; `difficulty_ladder` layout (rest first and last, pauses between task blocks and not after the last, two of each condition, `rep` 1/2); no condition twice in a row over 200 seeds for `difficulty_ladder` and `sim_quick`; indices and defaults (`rating`, `baseline`, `task`); the protocol definition is not mutated; unknown protocol raises; `status().available` totals match the schedule and a preview does not change the runner's seed. Run (0.2-s blocks): the full event sequence `block_start` / `rating_due` / `block_end` ... `done`; marks `protocol_start` (seed + order) ... `protocol_end` with `block=<i>` notes; `pending_rating` after the last rated block; `rate` clears pending and emits a JSON `rating` mark and a `rated` event; `next` skips; `stop` aborts (`block=<i> aborted`, no `rating_due`, `stopped` event, second stop is a no-op); double start rejected and restart after done |
| `tests/test_pipeline_blocks.py` | `Pipeline.on_protocol` builds the block table from `block_start` / `block_end` / `stopped` (a second end does not move `t1`); empty `_block_stats`; grouping, per-block and per-condition means, FSS and `felt` from ratings, `ordering` strings, `block_rho` = 1 on a consistent synthetic timeline and < -0.5 with the band reversed; pauses and < 30-s blocks skipped; held / settling / asleep / poor-contact / null band samples skipped; an open block uses now; fewer than 4 blocks -> no rho; `stats["blocks"]` filled on a mark; `rating` marks populate `pipeline.ratings` (JSON string or dict; garbage ignored) and feed the block's FSS mean |
| `tests/test_band_ble.py` | `decode_state` on 28- and 29-byte frames: scaling of hr/spo2/anom/flow, `flow_no_reading` -> `flow is None`, `held` rules (mode 1; mode 0 + window unavailable), `settling`, `asleep`, `poor_contact`, window names, `learn_pct` / `low_battery`; short frame raises |
| `tests/test_supabase_link.py` | `load_env` parsing and errors; `iso_to_epoch` with 1-6 fractional digits and `Z`; `row_to_sample` (held = mode 1, null flow, window names); `SupabaseClient.wearers()` aggregation and ordering with a stubbed `get`; `SupabaseBandLink._filters` with/without user and band; polling with a fake client only requests `created_at > last` |
| `tests/test_recorder.py` | `start` creates the folder, `session.json` and headers (including `task.csv`); eeg/epoch/band/mark/trial rows land in the right CSV with the right formatting; `epochs.csv` header is taken from the first record; `stop` closes files and returns the dir; writes after `stop` are dropped |
| `tests/test_astra.py` | `find_env` walks up to the first `.env` with `OPENAI_KEY`, `load_key` strips quotes and the trailing space in the name; `parse_sse_chunk` on data / empty-delta / `[DONE]` / comment / garbage lines; `live_snapshot` (10-s means, band flags, window pairs and r, hidden trial marks, task accuracy, quality, blocks) and `render_live` (essentials present, empty pipeline wording); `setup_snapshot` / `render_setup` (channel rows with raw peak and disabled flag, frontal sites, band link, protocols); `LiveAnalyst` with a fake chat function: two-phase `note` events with `delta`s, history carried into the next call and bounded by `memory_turns`, `analyze` / `ask` prompts and the `question` field, `astra_notes.md` / `.jsonl` written only while recording, an error clears `busy` and emits `error`, the live loop ticks once at once when enabled; `SetupHelper`: state + question in the user turn, reference in the system prompt, bare question in history, transcript, `check`, `clear`, empty question refused; `Astra` facade: commands run in the background and serialise per panel, `astra_live` interval, refusals without a key; `load_reference` holds the troubleshooting table and M-checks under 16 k chars |
| `tests/test_fixer.py` | against a sandbox folder: `resolve_path` allows root `*.py` / `static` / `tests` / `docs` and refuses `.env`, `sessions/`, `astra_fixes/`, parent paths, drive paths, other folders and file types; `file_tree` / `read_file` (missing, truncation); `parse_action` on bare / fenced / prefixed JSON and garbage; `apply_edits` unique match, all-or-nothing, ambiguous / empty / too many edits, the pre-registration guard (`protocols.json`, `BLOCK_MIN_RHO`, `"theta":` rejected without the flag, unrelated lines in the same file fine, everything allowed with it), refused targets; `Fixer` with a scripted model: read -> patch -> done with context and named file in the first message, refused `.env` reported, module reloaded and static flagged, events `start read patch verify done`, record folder with before/after and `fix.md`, `revert` restores and re-swaps; patch-before-read rejected then fixed; failed verification rolled back and retried with the output; two failures give up with files restored; `answer` and unusable replies; pre-reg rejected / allowed; `config.json` sets `needs_restart` and `restart()` calls the hook; `verify` compile / json steps on real files, docs skip the tests |
| `tests/test_hotreload.py` | throw-away modules on `sys.path`: `project_modules` lists only root files; `reload_module` gives live instances (direct attribute, dict value, list item) the new method body while keeping attributes, rebinds `from x import y` names in the sibling module, new instances use the new class; unknown module raises; `reachable` filters by module |
| `tests/test_report.py` | block scoring from `marks.csv` block marks + `rating` marks + `epochs.csv` + `band.csv` + `task.csv`: per-condition FSS / demand / accuracy, EEG and band condition means and ordering, block-level Spearman with n; verdict logic PASS / FAIL / INCONCLUSIVE (manipulation failed, baseline never frozen, < 6 blocks); held / settling / null band rows and artifact epochs excluded; the secondary epoch-level `pearson` / `bootstrap_ci`; `report.md` and `report.json` written; felt-flow section with and without marks |

`tests/_path.py` puts the project root on `sys.path` for the test modules. There is no unit
test for the USB `BandLink` because it imports the firmware repo's `offload.py`; it is
covered by M-7b. The arithmetic task and the rating dialog live in `static/app.js` and are
covered by M-11 / M-12 and `node --check`.

## 2. Integration test with `--sim`

```
python app.py --sim
```

The simulator follows the protocol: `SimBoard.set_condition()` eases the hidden flow state
to rest 0.40 / boredom 0.25 / flow 0.85 / overload 0.30 over ~20 s, and the sim band
reports that state plus noise 20 s later. So the expected block ordering is
**flow > overload > boredom** for both the EEG index and the band, with a high block rho.

Check in the browser, in this order:

1. Status bar: board `sim`, 8 ch, 250 Hz, `packets` increasing by ~250/s, `dropped` 0.
2. Time series: 8 traces moving; a blink-shaped bump appears on Fp1/Fp2 now and then.
3. FFT: peaks at 6, 10 and 20 Hz; the 60 Hz line is suppressed with the notch on, visible
   with notch 0 (set via the filter control, then set it back to 60).
4. Band power: theta highest on channels 1-2 (frontal), alpha highest on 5-8 (posterior).
5. Quality: all channels `good`. Flow widget: `baseline_state` = `session-until-rest`,
   `flow_eeg` 0.5 for the first 30 s, then moving; `frontal_sites` = `Fp1,Fp2` (factory
   labels; the sim is 8-ch).
6. Band panel: source `sim`, unit `BioProxalSIM`, connected, `age_s` under 3.
7. Experiment panel: pick `sim_quick` (about 9 min) and press START PROTOCOL. Recording
   starts by itself (status bar shows the `sim_quick` session). The schedule strip shows
   rest, six task blocks with pauses, rest; the order differs between runs (seed in the
   `protocol_start` mark note).
8. Block 0 (rest, 60 s): the log says `[protocol] collecting the resting baseline`; at its
   end `[protocol] baseline rest-frozen` and the flow widget shows `rest-frozen`.
9. Task blocks: the task widget shows the sums; answer a few in each block (any accuracy;
   the sim EEG does not depend on answers). The level and running accuracy update. At each
   block end the Flow Short Scale dialog opens; fill it in and submit (set `demand` 1 / 2
   / 3 to match the block) - it must close and the self-report tile show "rated".
10. After the six blocks `stats.blocks` (flow comparison panel) shows per-condition means
    with `ordering.eeg` = `flow>overload>boredom`, `ordering.band` the same, and
    `block_rho > 0.8` with `n_blocks` >= 6. `ordering.fss` follows whatever you entered.
11. On `done` the task area clears and the final rest runs; press Stop recording. Open
    Sessions, run Report: the verdict section shows the manipulation check from your
    ratings, EEG ordering `flow>overload>boredom`, band ordering the same, rho > 0.8 with
    n >= 6, and PASS if the ratings you entered put flow highest (INCONCLUSIVE otherwise -
    that is the manipulation gate working). `report.md` is in the session folder next to
    `task.csv` (one row per answer) and `marks.csv` (block marks and `rating` rows).
12. Marks: move the felt-flow slider three times; `felt.n` becomes 3 and `r_band` /
    `r_eeg` appear (secondary).
13. Set channel 1 disabled: it disappears from FFT/band/head widgets, `n_good` drops to 7,
    `frontal_sites` becomes `Fp2`.
14. Switch band source to `none` and back to `sim`: the band panel updates, pairs resume.
15. `python app.py --sim --montage flow`: the log shows `[montage] 8-ch flow montage: Fz
    F3 F4 Cz C3 C4 P3 P4`, the head plot uses those positions and `frontal_sites` is
    `Fz,F3,F4`.

## 3. Manual hardware checks

| ID | Check | Steps | Pass when |
|---|---|---|---|
| M-1 | Dongle detection | Dongle in, board off. `python app.py --no-browser`. | Log shows `[cyton] COMx:` with the chosen port; with the dongle unplugged the log says to plug it in / pass `--board` / use `--sim`, and the server still starts. |
| M-2 | Banner | Board on, switch PC, dongle GPIO6. Start the app. | Log shows the banner lines ("OpenBCI V3 8-16 channel", "ADS1299 Device ID", `$$$`); with the board switch OFF the error says no `$$$` banner and names the port. |
| M-3 | Railed check | Streaming, headset on the desk, then on the head. Touch one electrode to the scalp, let go. | Off-head: channels `railed` or `near`; on-head with ear clips: `good`; releasing the electrode flips it back. Flag colour matches the OpenBCI GUI convention. |
| M-4 | Mains notch | Streaming on-head. Set notch 0, then 60 (or 50). | FFT shows a 60 Hz peak at notch 0, flattened with notch 60. In a 50 Hz country the peak is at 50 and needs notch 50. The index is unaffected by the display band-pass: set `bp_lo` to 5 and `theta_frontal_rel` does not drop. |
| M-5 | Daisy order | 16-ch board. Tap the electrode labelled F3 (channel 11). | The F3 trace moves. If a trace in the first 8 moves instead, flip `daisy_even_is_upper` and retry. |
| M-6 | Supabase wearer pick | Phone app connected to the band, uploading. `--band supabase`. Open the band panel. | The wearer list shows the right short id and band name with `age_s` under 120; picking it makes `band` samples arrive about once a minute; `pair_window_s` is 60. With a wrong key the panel shows `HTTP 401` and the EEG keeps streaming. |
| M-7a | BLE direct | Close the Amplifly app. `--band ble`. | Log: `found BioProxalN @ mac, connecting` then `subscribed to STATE`; samples every 2 s with HR; `pair_window_s` 10. With the phone app open the error says no band advertising and lists what was seen. |
| M-7b | USB | Band on the cable, `usb.bioproxal_tools_dir` set, master key present. `--band usb`. | Log: `[band] COMx unlocked (BioProxalN)`; `[pub]` samples every 2 s; closing the app sends LOCK (firmware console locks again). No baseline reset or log clear happened (check the band's log count before/after). |
| M-8 | Record / report round trip | Any source. Run `sim_quick` or record 5 min with label `m8`, add two felt-flow marks, stop, Report. | `sessions/<stamp>_.../` has `session.json`, `eeg_raw.csv` (~75 000 rows for 5 min at 250 Hz), `epochs.csv` (~150 rows), `band.csv`, `marks.csv`, `task.csv`, `report.md`, `report.json`; the markdown shows the gates, the per-block table and the exclusions; a free recording without block marks is INCONCLUSIVE ("no protocol blocks"). |
| M-9 | UDP mirror | OpenBCI GUI streaming, Networking -> UDP, TimeSeriesRaw, 127.0.0.1:12345, Start. `python app.py --source udp`. | Status bar: board `udp`, packets increasing; time series matches the GUI. |
| M-10 | Localhost only | From another device on the LAN open `http://<laptop-ip>:8765`. | Connection refused. |
| M-11 | Arithmetic task | Start `sim_quick` (or `difficulty_ladder`) on any source; use each task block. | `arithmetic_easy`: single-digit sums, the answer field is disabled for 4 s after each answer. `arithmetic_adaptive`: starts at level 4; two correct in a row raises the level, one wrong lowers it; an unanswered problem counts wrong after 10 s. `arithmetic_hard`: two-digit x two-digit / three-number sums, moves on after 6 s. Running accuracy and level shown. Every answer appears as a row in `task.csv` with the right `block_index`, `condition`, `task`, `level`, `correct`, `rt_ms`. During a block the panel shows only instruction, task and timer. |
| M-12 | Rating dialog | Let a task block end. | The Flow Short Scale dialog opens with 10 items (1-7), the `demand` item (1/2/3) and the `flow_overall` slider; submit writes a `rating` mark whose JSON holds `block`, `condition` and all items, the tile shows "rated" and `stats.blocks.rows[block].fss_mean` fills in. Dismissing stores nothing and the tile keeps "rating due" until the next block starts. |
| M-13 | Rest baseline freeze | Real headset on-head, all channels good. Start `difficulty_ladder`. | During block 0 `baseline_state` = `rest-collecting`; at its end the log says `[protocol] baseline rest-frozen` and `flow_eeg` near 0.5 for a still, eyes-open wearer. Blink hard ten times during rest: those epochs are flagged `artifact` and `artifact_epochs` in `flow_status` rises; with more than 75 artifact epochs out of 90 the baseline does not freeze and the state stays `session-until-rest` (the session is then INCONCLUSIVE). |
| M-15 | Astra panels | `python app.py --sim --astra live`, then START PROTOCOL `sim_quick`. Press "check my setup", ask "is ch3 ok?", press "run analysis" after two task blocks. Then repeat the check on the real headset with one electrode lifted. | The first live note appears within ~10 s and then every 60 s, tagged `[OK]` / `[WATCH]` / `[ISSUE]`, with the block and condition in the header and the text streaming in; the setup check lists ch1-8 with their quality and ends READY / NOT READY; the ch3 answer names the channel's current quality and site; the analysis has the five headings and quotes block means; `sessions/<sim_quick>/astra_notes.md` holds every note with clock, block and condition. With no `.env` the header reads "no key" and the buttons are disabled; with a wrong key the feed shows one `error` entry and the live loop keeps retrying without crashing the stream. On the real headset the lifted electrode's channel is called out as railed / near with the fix. |
| M-16 | In-flight fix | `python app.py --sim`. In the fixes panel type "the band panel's `age_s` should show one decimal, e.g. 12.3 s, not a rounded value" and press Fix it. Then press revert last fix. Then type an issue that would change `protocols.json` without the checkbox. | The log shows `start`, `read` (band panel / app.js or pipeline.py), `patch`, `verify` (tests pass), `done` with the diff; the page shows the "reload page" button (static) or the module is reloaded and the panel changes on the next event without a restart; `astra_fixes/<stamp>_.../fix.md` holds the diff. Revert restores the previous text and the panel reverts. The protocols.json job ends `rejected` ... `answer`/`failed` with "pre-registered" in the text and the file unchanged. Restart server is refused while `sim_quick` runs and works after it (stream resumes, page reconnects). |
| M-14 | Flow montage | Electrodes moved to Fz F3 F4 Cz C3 C4 P3 P4. `python app.py --montage flow`. Tap the electrode at Fz. | Log line `[montage] 8-ch flow montage: ...`; the Fz trace (channel 1) moves; `frontal_sites` = `Fz,F3,F4`; `session.json` stores the flow labels. Eyes closed for 20 s: alpha rises at P3/P4 in the band-power widget. |

## 4. Release checklist

- [ ] `python -m unittest discover -s tests -v` passes on this laptop (3.14).
- [ ] CI green on Ubuntu + Windows, Python 3.12 (3.14 may be allowed to fail).
- [ ] `node --check static/app.js` passes.
- [ ] Integration test with `--sim` completed through step 15 (`sim_quick` ordering
      `flow>overload>boredom`, block rho > 0.8).
- [ ] M-1 to M-4, M-8 and M-13 done on the real headset; M-6 or M-7a done with the real
      band; M-11 / M-12 done once with a real participant-style run.
- [ ] `docs/API.md` matches the events and routes the browser uses (including
      `protocol_*`, `rating`, `trial`, `GET /api/protocol`, `event: protocol`).
- [ ] `config.json` defaults reviewed: notch for the country, `labels_8` / `labels_8_flow`
      for the montage, `flow_index` weights 1.0 / 0.0 / 0.0, `baseline_mode` `rest`.
- [ ] `protocols.json` durations match `docs/VALIDATION_PROTOCOL.md` section 5 and
      `tests/test_protocol.py`'s totals.
- [ ] `CHANGELOG.md` updated; version in `server.py` `SERVER_VERSION`, `report.py`
      `VERSION` and the protocol header match.
- [ ] `git status` shows no `sessions/`, `*.env`, `*.key` files staged.
- [ ] No change to the index definition (`docs/EEG_FLOW_INDEX.md` section 6), the
      protocol schedule or the gates since the last validation series, or the series was
      re-run and the protocol's change table says so.
