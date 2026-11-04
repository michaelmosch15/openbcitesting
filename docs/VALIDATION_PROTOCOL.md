# Validation protocol: pre-registration for the difficulty-ladder sessions (v0.2.0)

This document is the pre-registration. Fill in the header before the first session of a
series; sections 1-9 are then frozen for that series. Any later change is dated in
section 11 and the affected sessions are re-run and labelled exploratory. The v0.1.0
protocol (free task, Pearson r on epoch pairs) is superseded; its sessions are not
comparable with these and are not entered in the results log below.

| Field | Value |
|---|---|
| Series name | |
| Registered on (date) | |
| Registered by | |
| Firmware version on the band(s) | |
| Amplifly app version on the phone | |
| openbci-flow-validation version | 0.2.0 |
| Protocol | `difficulty_ladder` (`protocols.json`) |
| EEG index definition | `docs/EEG_FLOW_INDEX.md` as of the date above; weights theta 1.0 / alpha 0.0 / beta 0.0; baseline mode `rest` |
| Montage for the series | `--montage flow` (Fz F3 F4 Cz C3 C4 P3 P4) / 16-ch (F7 F8 disabled: yes / no) / 8-ch factory Fp-only |
| Mains notch | 60 Hz / 50 Hz |
| Band source for the series | supabase / ble (one per series) |

## 1. Design in one paragraph

Within-participant difficulty manipulation after Katahira 2018 (10.3389/fpsyg.2018.00300)
and Ewing 2016 (10.3389/fnhum.2016.00223), following the Durcan 2024
(10.1038/s44271-024-00115-3) checklist items 3a-c, 6a and 7a: the same mental-arithmetic
task at three difficulty levels (boredom: trivial; flow: adaptive, matched to the person;
overload: too hard), six 6-min blocks (two per condition) in a counterbalanced order, with
a Flow Short Scale self-report after every block. Flow is expected only in the matched
block. The band's flow score, the EEG theta index and the self-report are all scored per
block, and the block is the unit of inference (ADR-10). Self-report is the criterion; the
EEG index is a convergent reference, not ground truth (`docs/EEG_FLOW_INDEX.md` section 4).

## 2. Hypotheses

**H1 - manipulation check (precondition for everything else).**
H1a: the Flow Short Scale block mean (10 core items, 1-7) is higher in flow blocks than in
boredom blocks and higher than in overload blocks (condition means over the two blocks
each). H1b: the `demand` item is ordered boredom (1 = too low) < flow (2 = just right) <
overload (3 = too high) in the condition means. H1c: task accuracy is about 100 % in
boredom, 70-85 % in flow (the staircase's target) and low (< 40 %) in overload. If H1a or
H1b fails the session is INCONCLUSIVE: the task did not induce the states, so nothing can
be said about either instrument.

**H2 - reference check.** The EEG theta index (`flow_eeg` block mean of non-artifact
epochs) is higher in flow blocks than in boredom blocks (condition means). Flow vs overload
is not a hypothesis (theta did not separate them in Katahira 2018). H2 is reported, not
gated: a session where H1 holds and H2 fails is a result about the EEG reference, and the
band verdict is still computed but flagged "reference failed".

**H3 - the band.** H3a: the band's block-mean flow score is highest in the flow condition,
i.e. flow > boredom and flow > overload in the condition means. H3b: the block-level
Spearman rho between the band's block means and the EEG index block means over the task
blocks is >= 0.5, with at least 6 task blocks contributing (both means present).

Expected pattern (literature file section 4.3):

| Measure | boredom | flow | overload |
|---|---|---|---|
| FSS block mean | low | high | medium-low |
| `demand` item | 1 | 2 | 3 |
| accuracy | ~100 % | 70-85 % | low |
| EEG `flow_eeg` | low | high | high or slightly lower than flow |
| band flow | low | high | medium |
| `engagement_index` (covariate) | low | mid | high |

## 3. Participants

- The owner first (user id prefix 54af48b8, male): at least 3 complete sessions on
  different days before anyone else is run. These sessions establish that the procedure
  works and are entered in the results log like any other.
- Then a group series: N >= 20 participants completing all six task blocks is required
  before any group-level claim is made (Ewing 2016 found demand effects at N = 20 with
  eta^2 0.4-0.7; Katahira at N = 16; Knierim 2018 at N = 7 found nothing). Women and men,
  aiming for balance; the band is personalised per wearer, so the series is not valid for
  one sex only.
- One session per participant per day. Each participant wears one band for the whole
  series; a swapped band (battery, fault) is noted with the old and new unit, and the
  first session on the new unit is marked "new band".
- Each participant consents to EEG, wrist data, task answers and self-reports being
  recorded locally for this validation; data stays on the laptop (section 10).

## 4. Set-up checklist (before START PROTOCOL)

1. Laptop and phone on network time (a phone clock offset shifts the Supabase pairs).
2. Band worn on the participant's wrist for at least 10 min before the protocol starts
   (the band settles after contact; `settling` rows are never used). Phone app (Amplifly)
   connected to the band and uploading - the band panel's wearer list shows the
   participant with `age_s` under 120. For a BLE series: app closed, `--band ble`.
3. `python app.py --montage flow` (or the montage named in the header). The log line
   `[montage] 8-ch flow montage: Fz F3 F4 Cz C3 C4 P3 P4` confirms it. `session.json` will
   store the labels.
4. Headset on; ear clips (SRB2, BIAS) on. Press Start. Every channel reads `good` in the
   time-series widget (no railed / near / flat) for at least 30 s. Press and re-seat any
   electrode that is not.
5. Notch set for the country (60 / 50 Hz); check the FFT has no mains peak.
6. Pick the wearer in the band panel (Supabase) and wait for a fresh band sample with
   `held` false and `settling` false.
7. The participant has read the task instructions and done the 2-min practice in the
   task widget (easy then adaptive) so that novelty is gone (Durcan 4b). Chair, screen and
   keyboard arranged so the head and jaw are still; no talking during blocks.
8. Pick `difficulty_ladder` and press START PROTOCOL. Recording starts by itself with the
   protocol name as label; the schedule strip shows the counterbalanced order and the
   seed is written to `marks.csv` (`protocol_start` note).

## 5. Session script

`difficulty_ladder` in `protocols.json`, 14 blocks, 47 min of protocol plus set-up.
The operator does not intervene except for safety or equipment failure; `NEXT` (skip) is
only for an aborted block and is logged as such.

| # | Block (`condition`) | Duration | Instruction on screen | What happens underneath |
|---|---|---|---|---|
| 0 | `rest`, eyes open, **baseline** | 3 min | sit still, eyes open, look at the fixation dot, breathe normally | `FlowIndex.start_rest()` at block start, `end_rest()` at block end: the rest mean / SD are frozen (`baseline_state` = `rest-frozen`). Needs >= 15 clean epochs; 90 are available. |
| 1 | `rest_ec`, eyes closed | 1 min | close your eyes, sit still | alpha-reactivity check only (posterior alpha should rise); not part of the baseline |
| 2-12 | six task blocks, 6 min each, with a 60-s `pause` between consecutive blocks | 6 x 6 min + 5 x 1 min | `boredom`: "Answer the sums. They are deliberately trivial and slow." `flow`: "Answer as fast and accurately as you can. Difficulty adapts." `overload`: "Answer as fast as you can. The problems are too hard on purpose; keep trying." | task widget runs `arithmetic_easy` / `arithmetic_adaptive` / `arithmetic_hard`; every answer is a `trial` row in `task.csv`. At block end the Flow Short Scale dialog opens (`rating_due`); the participant rates during the pause. Order: two rounds of the three conditions, shuffled per round from the seed, never the same condition twice in a row. |
| 13 | `rest`, eyes open | 2 min | sit still, eyes open | recovery; not a baseline, not a task block |

Then `protocol_end` is marked, the browser clears the task, recording keeps running until
the operator presses Stop recording. Open Sessions, pick the session, press Report, and
copy the verdict and the numbers into the results log (section 12).

Self-report after every task block (`docs/API.md`, "Self-report dialog"): the 10 Flow Short
Scale items (Rheinberg, Vollmeyer & Engeser 2003; English wording from the Finnish
validation, PMC9681728), 1 = not at all ... 7 = very much; the `demand` item 1 / 2 / 3;
and the 0-1 `flow_overall` slider. The dialog can be dismissed; a dismissed rating is a
missing FSS for that block (section 7, exclusions). Keep the item order fixed.

Task (browser side, `static/app.js`; spec in `docs/API.md`): `arithmetic_easy` single-digit
additions with a new problem only every 4 s; `arithmetic_adaptive` levels 1-12 starting at
level 4, two correct in a row -> up, one wrong -> down, 10-s soft limit, targeting 70-85 %
correct; `arithmetic_hard` fixed level 12 with a 6-s hard limit. Feedback (tick / cross)
in all conditions.

## 6. Exclusion rules

Applied by `report.py` where it can, by hand otherwise; every exclusion is listed in the
report and in the results-log notes.

| Unit | Rule | Effect |
|---|---|---|
| EEG epoch | `artifact = true` (fewer than 5 of 9 Welch segments clean, `docs/EEG_FLOW_INDEX.md` 2.4), or `baseline_state != rest-frozen` | not used in any block mean or pair |
| EEG epoch | any channel in the frontal set not `good` for that epoch | the set shrinks (`frontal_sites`); if the set fell back to Fp or to all channels in a session registered as `--montage flow` / 16-ch, note it and treat those epochs as artifact in a sensitivity run |
| Band row | `held`, `settling`, `asleep`, `poor_contact`, or `flow` null | never used (`Pipeline.on_band`, `_block_stats`, `report.py`) |
| Task block | fewer than 50 % clean EEG epochs | block excluded from H2 and from the block rho |
| Task block | no band row usable inside the block (Supabase: expect 5-6 per 6-min block) | block has no band mean; excluded from H3b |
| Task block | no FSS rating (dialog dismissed) | block excluded from H1; if more than one task block lacks a rating the session is INCONCLUSIVE |
| Task block | aborted with NEXT / Stop before 3 min | excluded entirely |
| Session | rest baseline not frozen (`baseline_state` never `rest-frozen`) | INCONCLUSIVE |
| Session | H1a or H1b fails | INCONCLUSIVE ("manipulation failed") |
| Session | fewer than 6 task blocks with both an EEG mean and a band mean | INCONCLUSIVE |
| Session | band off-wrist or phone app disconnected for more than 3 consecutive minutes inside a task block | that block excluded; if two or more blocks are lost, INCONCLUSIVE |
| Session | Fp-only montage (8-ch factory) | kept, verdict computed, labelled "Fp-only" in the log; not counted toward the series-level criterion |

A sensitivity run (reported after the pre-registered one, never instead of it): the same
report with `flow_eeg_alt` in place of `flow_eeg`, and with the first 60 s of every task
block dropped (ramp-in, as Katahira dropped trial 1). Neither changes the verdict.

## 7. Gates and verdict (`report.py`)

The verdict is computed only by `report.py` from `epochs.csv`, `band.csv`, `marks.csv`,
`task.csv` and `session.json` in the session folder; the live `stats.blocks` in the
browser (`Pipeline._block_stats`) is for monitoring. The report's sections are: Verdict,
Manipulation check (H1: FSS, demand, accuracy per condition), Reference check (H2: EEG
condition means and ordering), Band (H3: condition means, ordering, block-level Spearman
rho and n blocks, the per-block table), Covariates (`engagement_index`, `flow_eeg_alt`,
`z_alpha`, `z_beta`, HR where present), Exclusions, and How to read this.

| Gate | Threshold | From |
|---|---|---|
| G1 manipulation | FSS condition mean: flow > boredom and flow > overload; `demand` condition mean: boredom < flow < overload | H1a, H1b |
| G2 band ordering | band condition mean: flow > boredom and flow > overload | H3a |
| G3 block agreement | Spearman rho(band block means, EEG block means) >= 0.5 over the task blocks, with n blocks >= 6 | H3b |

Verdict:

- **PASS**: G1 and G2 and G3 all hold.
- **FAIL**: G1 holds and G2 or G3 does not (the task worked; the band did not track it).
- **INCONCLUSIVE**: G1 does not hold, or any session-level exclusion in section 6 applies
  (baseline not frozen, fewer than 6 usable blocks, missing ratings).

H2 (EEG flow > boredom) is printed next to the verdict as "reference ok" / "reference
failed" and does not change the verdict. A FAIL with "reference failed" is still a FAIL for
the band against the self-report ordering (G2) but the rho gate is then weak evidence,
and the notes say so.

H1c (accuracy) is reported as a check on the task; a flow block with accuracy outside
55-95 % is flagged in the notes but does not by itself fail G1.

Series-level criterion (set now): for the owner, at least 2 of the first 3 complete
sessions PASS. For the group series, the band passes when >= 60 % of participants with a
non-INCONCLUSIVE session PASS and the median block rho across those participants is
>= 0.5. A group series with fewer than 20 non-INCONCLUSIVE sessions is reported as
"underpowered", not as a pass.

## 8. Analysis

`python report.py sessions/<session>` (or Report in the browser) is the analysis. No other
script, spreadsheet or notebook decides the verdict. Means are per block, then per
condition; ordering is on condition means; Spearman is on block means with ties broken by
average rank. Epoch-level Pearson r, lag sweep and felt-flow correlations from v0.1 remain
in the report as secondary, descriptive numbers.

## 9. Honesty rules

- The index definition (`docs/EEG_FLOW_INDEX.md` section 6), the protocol schedule, the
  task parameters, the exclusion rules and the gates are fixed at registration and are not
  tuned on the sessions of this series.
- `flow_eeg_alt` is exploratory and may not be promoted to the primary index for this
  series after data has been seen (`docs/EEG_FLOW_INDEX.md` section 6).
- If anything in the previous two points is changed, the change is dated in section 11,
  every earlier session in the series is re-run with the new definition, both verdicts are
  kept in the results log with the note "index / protocol changed after data was seen",
  and the re-runs are labelled exploratory.
- The live dashboard is for monitoring; the verdict comes only from the offline report.
- A FAIL is logged as FAIL. Explanations go in the notes column, not in the verdict.
- Every session that was started is logged, including aborted ones (INCONCLUSIVE with the
  reason).
- The counterbalancing seed is the session start time and is logged; it is never chosen.
- The band firmware and app versions are logged; a firmware change starts a new series.

## 10. Privacy

Raw EEG, wrist-derived scores, task answers and self-reports are health data of named
people. Session folders stay in `sessions/` on this laptop (git-ignored) and are not copied
to the cloud. The results log holds aggregate numbers and a participant code, never the
full Supabase user id.

## 11. Protocol changes after registration

| Date | Change | Sessions re-run | Why |
|---|---|---|---|
| | | | |

## 12. Results log

One row per session started. `ordering` columns are the condition order of the means
(e.g. `flow>overload>boredom`). `rho` is the block-level Spearman with its n.

| # | Date | Participant | Band unit / fw | Source | Montage | Seed | Blocks usable | FSS ordering | demand ordering | accuracy b/f/o | EEG ordering (H2) | Band ordering (H3a) | rho (n) | Verdict | Excluded / notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | | | | | | | | | | | | | | | |
| 2 | | | | | | | | | | | | | | | |
| 3 | | | | | | | | | | | | | | | |
| 4 | | | | | | | | | | | | | | | |
