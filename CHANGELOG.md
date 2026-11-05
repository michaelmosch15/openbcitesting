# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow SemVer.

## [Unreleased]

### Added

- `astra.py`: the owner's OpenAI reviewer persona Astra (`gpt-6-astra`, key `OPENAI_KEY` from
  `BioProxal2026/.env`, stdlib `urllib` streaming) as two dashboard panels.
  `LiveAnalyst`: every `astra.live_interval_s` (60 s) a compact snapshot of the last
  `astra.window_min` (5) minutes - EEG index as 10-s means, band samples with flags, pairs and
  their r, session stats, block table / ordering / block rho, task accuracy, marks, channel
  quality, baseline state - plus its own recent notes, one `[OK]` / `[WATCH]` / `[ISSUE]` note
  back; `analyze()` for a structured interim analysis toward the gates; `ask()` for questions;
  notes appended to `astra_notes.md` + `astra_notes.jsonl` in the session folder while
  recording. `SetupHelper`: electrode-placement / montage / Cyton switches / quality-flag /
  band-link coach that sees the live per-channel state with every question, `check()` for a
  READY / NOT READY walk-through. Observational only (prompts forbid tuning suggestions).
- API: `GET /api/astra`, commands `astra_live`, `astra_analyze`, `astra_ask`, `astra_check`,
  `astra_clear`, `status.astra`, `event: astra` with streamed `delta` pieces (`docs/API.md`).
- Browser: "Astra · live analyst" (feed, start/stop live, interval, run analysis, ask) and
  "Astra · setup helper" (live channel table with quality / % full scale / raw peak / rel
  theta / rel alpha, chat, check my setup, clear) row above Sessions.
- `app.py --astra on|live|off`; `config.json` `astra` block (`model`, `live_interval_s`,
  `window_min`, `max_tokens`, `memory_turns`, `auto_live`); `Pipeline.last_raw_peak`.
- `tests/test_astra.py` (key loading, SSE chunk parser, snapshots and renderings, analyst
  notes / history / files with a fake chat function, setup transcript, facade dispatch).
- Simulator demo: with `--sim` (`config.json` `sim.auto_rate`, default true) each
  `rating_due` is answered 3 s later with a Flow Short Scale rating that follows the induced
  condition (boredom low / flow high / overload low, demand 1 / 2 / 3), so an unattended
  `sim_quick` run reaches a verdict. `report.py` prints a "Simulated session" banner for
  sessions recorded from the sim board.

## [0.2.0] - 2026-11-03

The dashboard becomes an experiment: a protocol runner induces known flow / non-flow
blocks, the participant rates each block, and both flow scores are judged per block.

### Added

- `protocol.py`: `ProtocolRunner` - timed block schedules from `protocols.json`,
  counterbalanced from a logged seed, no condition twice in a row, pauses between task
  blocks, `start` / `next` / `stop` / `rate`, marks (`protocol_start`, `block_start`,
  `block_end`, `rating`, `protocol_end`, `protocol_stop`) and SSE `protocol` events
  (`block_start`, `rating_due`, `block_end`, `rated`, `done`, `stopped`).
- `protocols.json`: `difficulty_ladder` (3-min eyes-open rest baseline, 1-min eyes closed,
  2 rounds x {boredom, flow, overload} 6-min arithmetic blocks with 60-s rating pauses,
  2-min rest; 47 min), `sim_quick` (same shape, 60-s blocks), `free_work` (8 x 5-min
  rated work blocks, not gated).
- Browser experiment panel: protocol picker, START / NEXT / STOP, schedule strip,
  instruction and timer, in-browser mental-arithmetic task (`arithmetic_easy`,
  `arithmetic_adaptive` staircase, `arithmetic_hard`), Flow Short Scale self-report dialog
  (10 items 1-7, `demand` 1-3, `flow_overall` slider).
- API: `protocol_start` / `protocol_stop` / `protocol_next` / `rating` / `trial` commands,
  `GET /api/protocol`, `status.protocol`, `event: protocol` (`docs/API.md`).
- `recorder.py`: `task.csv` (one row per answer); `protocol_start` auto-starts recording.
- `flow_index.py`: `Baseline` with `rest` / `session` / `rolling` modes; the protocol's
  baseline block drives `start_rest()` / `end_rest()`; `flow_eeg_alt` exploratory index;
  site sets `FRONTAL_PREFERRED` / `FRONTAL_FALLBACK` / `CENTRAL_PARIETAL`; `artifact`,
  `baseline_state`, `frontal_sites` exported.
- `dsp.py`: `welch()` with per-segment rejection, `band_powers_of()`, `INDEX_BANDS`
  (theta_fm 4-7, alpha_hi 10-13 Hz), `TOTAL_BAND` 1-45 Hz.
- `pipeline.py`: 10-s Welch index window (2-s Hann, 50 %) every 2 s; artifact rule
  (150 uV p2p / 100 uV abs per segment, epoch artifact below 50 % clean segments);
  `on_protocol` block table, `_block_stats` (per-block and per-condition means, ordering,
  block-level Spearman rho) published as `stats.blocks`; `trial()`; `flow_status()`.
- `config.json`: `flow_index` block (weights, `baseline_mode`, `warmup_epochs`,
  `artifact_p2p_uv`, `artifact_abs_uv`, `analysis_hi_hz`, `index_window_s`,
  `index_seg_s`, `index_min_clean_fraction`); `labels_8_flow` montage preset;
  `app.py --montage flow`.
- `cyton.py`: `SimBoard.set_condition()` - the simulator's hidden flow state follows the
  protocol's conditions (rest 0.40, boredom 0.25, flow 0.85, overload 0.30).
- `report.py`: per-block scoring and the block-level gates (manipulation check, band
  ordering, block rho >= 0.5 with >= 6 blocks); the v0.1 epoch-level statistics kept as
  secondary output.
- Tests: `tests/test_protocol.py`, `tests/test_pipeline_blocks.py`;
  `tests/test_flow_index.py` rewritten for `Baseline`, artifacts, rest protocol and sites.
- Docs: `docs/EEG_FLOW_LITERATURE.md` (research review), `docs/EEG_FLOW_INDEX.md`
  (operational specification of the index), `docs/VALIDATION_PROTOCOL.md` rewritten as
  the pre-registration for the difficulty-ladder sessions, ADR-8 to ADR-11, README quick
  start for the protocol, updated ARCHITECTURE / REQUIREMENTS / TEST_PLAN.

### Changed

- EEG flow index: frontal theta is the sole primary marker. Weights 0.5 / 0.25 / 0.25 ->
  1.0 / 0.0 / 0.0; sigmoid gain 1.5 -> 1; theta band 4-8 -> 4-7 Hz (`theta_fm`); alpha
  term now upper alpha 10-13 Hz at central / parietal sites (`alpha_cp_rel`, was
  `alpha_posterior_rel`) and only in the exploratory index; relative power over 1-45 Hz
  (was 1-55 Hz analysis band and the 1-50 Hz display-band sum); 10-s Welch window (was the
  2-s epoch PSD); baseline rolling 150 epochs -> fixed eyes-open rest block.
- Verdict logic: epoch-pair Pearson gates (r >= 0.5, CI > 0, n >= 20) -> block-level gates
  with the manipulation check as a precondition (`docs/VALIDATION_PROTOCOL.md` section 7).
- Band pairing also excludes `settling` and `poor_contact` rows.
- `tests/test_recorder.py` covers `task.csv`; `tests/test_report.py` covers the block gates.

### Removed

- `RollingZ` (replaced by `Baseline`); the v0.1 free-task validation protocol (superseded;
  v0.1 sessions are not comparable with v0.2 verdicts).

## [0.1.0] - 2026-11-03

### Added

- `cyton.py`: pure-Python OpenBCI Cyton v3 driver over the RFduino dongle (`CytonBoard`),
  8/16-channel (Daisy) support with configurable half ordering, dongle auto-detection that
  never picks the band's ESP32-S3 port, `SampleBuffer` ring, `SimBoard` synthetic EEG with
  a hidden flow state.
- `openbci_udp.py`: `UdpBoard`, mirrors the OpenBCI GUI Networking widget's UDP time series.
- `dsp.py`: biquad notch (60/50 Hz + harmonic) and 1-50 Hz band-pass matching the GUI's
  display filters, Hann PSD, band powers, GUI-style railed/near/flat quality flags.
- `flow_index.py`: transparent EEG flow index (frontal theta, moderate posterior alpha, low
  beta; rolling personal baseline, z-scores, sigmoid) with all components exported.
- Band sources: `supabase_link.py` (Amplifly app -> Supabase `wearable_samples`, polled
  every 5 s, wearer picker), `band_ble.py` (direct BLE STATE frames with `bleak`),
  `band_link.py` (USB `[pub]` lines through the firmware repo's tools; `SimBandLink`).
- `pipeline.py`: filters, 2-s epochs, pairing at the band's cadence, Pearson r with
  bootstrap CI, lag sweep, felt-flow correlations, 4-h history.
- `server.py` / `app.py`: stdlib HTTP + SSE server on 127.0.0.1:8765, JSON API
  (`docs/API.md`), `BandManager` to swap band sources live, CLI flags.
- `static/`: OpenBCI-GUI-style widgets (time series, FFT, band power, head plot) plus the
  BioProxal layer (flow comparison, band panel, felt-flow marks, sessions and report).
- `recorder.py`: per-session CSV recording from a writer thread.
- `report.py`: offline report with pre-registered gates (r >= 0.5, CI lower bound > 0,
  n >= 20; INCONCLUSIVE below 20 pairs).
- Docs: README, API, ARCHITECTURE, REQUIREMENTS, TEST_PLAN, VALIDATION_PROTOCOL, DECISIONS.
- Tests under `tests/`, CI on Ubuntu + Windows with Python 3.12 and 3.14.
- `run.ps1` / `run.bat` launchers, `config.json` defaults.

[Unreleased]: https://github.com/michaelmosch15/openbci-flow-validation/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/michaelmosch15/openbci-flow-validation/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/michaelmosch15/openbci-flow-validation/releases/tag/v0.1.0
