# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow SemVer.

## [Unreleased]

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

[Unreleased]: https://github.com/michaelmosch15/openbci-flow-validation/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/michaelmosch15/openbci-flow-validation/releases/tag/v0.1.0
