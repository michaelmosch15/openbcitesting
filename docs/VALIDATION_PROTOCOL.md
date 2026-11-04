# Validation protocol: BioProxal band flow score vs EEG flow index

Pre-registration template. Fill in the header before the first session of a series and do
not change sections 2-8 afterwards without logging the change in section 10.

| Field | Value |
|---|---|
| Series name | |
| Registered on (date) | |
| Registered by | |
| Firmware version on the band(s) | |
| openbci-flow-validation version | 0.1.0 |
| Flow-index weights at registration | theta 0.5 / alpha 0.25 / beta 0.25 (sigmoid gain 1.5, baseline 150 epochs, warm-up 15) |
| Band source for the series | supabase / ble (one per series) |

## 1. Hypothesis

H1: During a focused task, the BioProxal band's flow score co-varies with the EEG flow index
(`flow_index.py`) of the same person over the same minutes, with Pearson r >= 0.5.

H0: r < 0.5 or its 95 % CI includes 0.

The EEG index is a literature-based proxy, not ground truth; a PASS supports "the band
tracks the same thing a frontal-theta / moderate-alpha / low-beta proxy tracks".

## 2. Participants

- The owner first (user id 54af48b8, male), minimum 3 sessions, before anyone else.
- Then women and men, aiming for a balanced set; each person wears one band for the whole
  series. If a band is swapped (battery, fault), note the old and new unit in the results
  log and treat the first session on the new band as a new baseline.
- Each participant consents to EEG and wrist data being recorded locally for this
  validation; the data stays on the laptop (section 9).

## 3. Montage

- Ultracortex Mark IV, Cyton: default 8-ch labels Fp1, Fp2, C3, C4, P7, P8, O1, O2, ear-clip
  SRB2 reference and BIAS. Preferred: Cyton + Daisy 16-ch (adds F7, F8, F3, F4, T7, T8, P3,
  P4) so frontal theta is not only Fp. If only 8 channels are available and two electrodes
  are re-sited to F3/F4, set `labels_8` accordingly before the session and record it here.
- Filters: notch 60 Hz (50 Hz where mains is 50); the flow index uses the fixed 1-55 Hz analysis filter (the display band-pass, default 5-50 Hz, does not affect it); epoch 2 s.
- Impedance/contact: all channels `good` in the quality flags before recording starts.

Montage used for this series: ________________________________

## 4. Session structure

| Phase | Duration | Instructions |
|---|---|---|
| Setup | - | Headset on, band on the same person, phone app open (Supabase) or closed (BLE). Stream on, all channels good, band sample fresh, `baseline_ready`. Then `record_start` with a label. |
| Rest baseline | 5 min | Sit quietly, eyes open, no task. Rate felt flow at the end (expected low). |
| Task | 20-40 min | A real focused task the participant chose (code, writing, music, a game). Rate felt flow every ~5 min and whenever it changes noticeably; press `distracted` when interrupted; `flow_start` / `flow_end` if the participant notices entering or leaving. |
| Rest | 5 min | As the baseline. Rate felt flow at the end. |
| Stop | - | `record_stop`. Run Report. Copy the verdict and numbers into the results log. |

Felt-flow rating: the 0-1 slider, "how much were you in flow over the last few minutes".
Marks are written to `marks.csv`.

## 5. Primary metric and gates

Primary metric: Pearson r between the band's flow score and the EEG flow index, paired at
the band's cadence (EEG index averaged over the `pair_window_s` seconds before each band
timestamp: 60 s for Supabase, 10 s for BLE), lag 0, task phase plus rest phases together
(the whole recording).

Gates, evaluated per session by `report.py`:

| Gate | Threshold |
|---|---|
| r | >= 0.5 |
| bootstrap 95 % CI on r (500 resamples) | lower bound > 0 |
| usable pairs n | >= 20 |

Verdict: PASS if all three hold; FAIL if n >= 20 and either of the first two fails;
INCONCLUSIVE if n < 20 (not counted as a FAIL, but also not as a PASS).

Series-level success: at least 3 of the first 4 sessions PASS, with no session FAIL
attributable to the band rather than to an exclusion-rule breach. Series-level criteria are
set here, before the first session.

## 6. Secondary metrics (reported, not gated)

- Correlation of the band score with each exported EEG component: `z_theta`, `z_alpha`,
  `z_beta`, `theta_frontal_rel`, `alpha_posterior_rel`, `beta_rel`, `engagement_index`,
  `theta_alpha_ratio`.
- Lag: best lag in the -180..+180 s sweep and r at that lag (positive = band lags EEG).
- In-flow agreement: Cohen's kappa between "band flow >= 0.6" and "EEG flow >= 0.6" per
  pair (threshold fixed here).
- Felt flow: r between felt-flow marks and the band score (nearest sample within 90 s) and
  between the marks and the EEG index (mean over the 60 s before the mark).
- Phase means: band and EEG index in rest-1, task, rest-2 (task should be highest for both).
- MAE and mean offset (`band_minus_eeg_mean`).

## 7. Exclusion rules

Applied by `report.py` where it can, by hand otherwise; every exclusion is listed in the
report.

- Epochs with any used channel not `good` (railed, near, flat) are dropped from the index
  before pairing; if more than 20 % of task epochs are dropped the session is INCONCLUSIVE.
- Band rows that are held, settling, asleep, poor-contact or have no reading (`flow`
  null) are never paired.
- Sessions where the band was off-wrist or the phone app disconnected for more than 5
  consecutive minutes during the task: INCONCLUSIVE.
- Blink-heavy segments: if Fp1/Fp2 carry more than 30 % of epochs with a > 100 uV
  excursion, note it; re-run the report with Fp1/Fp2 disabled and report both results, the
  pre-registered one first.
- Sessions shorter than 20 min of task or with fewer than 3 felt-flow marks are kept for
  the primary metric if n >= 20 but flagged as "short" in the log.

## 8. Analysis

`python report.py sessions/<session>` (or Report in the browser). The report is the
analysis; no other script, spreadsheet or notebook decides the verdict. Everything the
report needs is in `epochs.csv`, `band.csv`, `marks.csv` and `session.json`.

## 9. Honesty rules

- The flow-index weights (and the sigmoid gain, baseline and warm-up lengths) are fixed at
  registration. They are not tuned on the sessions of this series.
- If the weights are changed for any reason, the change is dated here, every earlier
  session in the series is re-run with the new weights, and both the old and new verdicts
  are kept in the results log with a note saying the index was changed after the data was
  seen. Such re-runs are exploratory, not confirmatory.
- The live dashboard is for monitoring; the verdict comes only from the offline report.
- A FAIL is logged as FAIL. Explanations go in the notes column, not in the verdict.
- Report every session that was started, including aborted ones (as INCONCLUSIVE with the
  reason).

## 10. Privacy

Raw EEG, wrist-derived scores and felt-flow ratings are health data of named people.
Session folders stay in `sessions/` on this laptop (git-ignored) and are not copied to the
cloud. The results log below holds only aggregate numbers and the participant's label
(initials or a code), never the Supabase user id in full.

## 11. Results log

| # | Date | Participant | Band unit | Source | Montage | Task | Task min | n pairs | r | 95 % CI | best lag s | kappa | felt r band / eeg | Verdict | Excluded / notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | | | | | | | | | | | | | | | |
| 2 | | | | | | | | | | | | | | | |
| 3 | | | | | | | | | | | | | | | |
| 4 | | | | | | | | | | | | | | | |

Protocol changes after registration:

| Date | Change | Sessions re-run | Why |
|---|---|---|---|
| | | | |
