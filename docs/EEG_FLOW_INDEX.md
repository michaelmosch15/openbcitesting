# EEG flow index: operational specification (v0.2.0)

This is the auditable definition of the reference index the BioProxal band's flow score is
validated against. It says what the code computes, with the numbers, and why. The research
behind it is in `docs/EEG_FLOW_LITERATURE.md` (the literature file); the code is
`flow_index.py`, `dsp.py` and `pipeline.py`. Where this file and the code disagree, the code
is wrong and must be fixed to match this file, or this file must be amended with a dated
note in section 8 - never silently.

Status: pre-registered for the difficulty-ladder sessions (`docs/VALIDATION_PROTOCOL.md`).
Section 6 says what may and may not change.

## 1. Evidence base

Only one EEG marker of flow replicates across independent labs: frontal / fronto-midline
theta (4-7 Hz) is higher in a demand-matched block than in a too-easy block. Every other
candidate is contradictory, null, or an engagement/workload measure that was never tested
against flow self-report (literature file section 2).

| Marker | Expected direction (boredom / flow / overload) | Studies | Replication status |
|---|---|---|---|
| Frontal-midline theta, 4-7 Hz, Fz / F3 / F4 / AFz region | low / high / high-or-slightly-lower | Katahira 2018 (10.3389/fpsyg.2018.00300): frontal ROIs boredom < flow = overload. Ewing, Fairclough & Gilleade 2016 (10.3389/fnhum.2016.00223): AFz/Fz/FCz/F1/F2 theta inverted-U in demand, eta^2 = 0.71. Fairclough & Ewing 2017 (10.1016/j.ijpsycho.2017.01.007): curvilinear in n-back. Knierim 2025 (10.1145/3706598.3713512): relative theta inverted-U against the Flow Short Scale on a Cyton+Daisy. Rácz 2025 (10.1038/s41598-025-95647-x): Fp1/AF8 theta flow > frustration. Lin 2025 (10.1007/s11571-024-10205-x): higher theta in flow. Eschmann 2022 (10.1007/s41465-021-00236-1): Fz theta neurofeedback raises Flow State Scale-2 scores (causal, d = 0.30). | Replicated in > 2 independent labs for flow vs boredom. Nulls: Knierim 2018 (10.5220/0006926700420053, N = 7, Emotiv); Sugino 2024 (bioRxiv 10.1101/2024.07.11.603158, preprint): Fz relative theta uncorrelated with FSS in 6/7 expert gamers; Bombeke 2018 (10.1007/978-3-319-91470-1_10) could not induce the states; Rosen 2024 (10.1016/j.neuropsychologia.2024.108824): hypofrontality in expert jazz improvisation. Flow vs overload: unresolved (equal in Katahira; lower in Ewing). |
| Upper alpha, 10-13 Hz, central / parietal | monotonic in demand with flow in the middle; sign differs by study | Katahira 2018: right-central boredom < flow < overload (rises). Ewing 2016: frontal/central/parietal upper alpha falls with demand, eta^2 = 0.42. Knierim 2018: frontal hiAlpha boredom > flow/overload (falls). Rácz 2025: AF7 alpha flow > boredom against an eyes-closed baseline, null against eyes-open. Núñez Castellar 2019 (10.1111/psyp.13347): midfrontal alpha up in flow. Knierim 2025: null. Lin 2025: "moderate". | Direction not replicated; only "flow is intermediate" recurs. Sign flips with site, sub-band and baseline (Rácz). Not usable as a signed marker. |
| Beta, 13-30 Hz | no consistent direction | Berta 2013 (10.1109/TCIAIG.2013.2260340): low beta most discriminative, direction not given. Knierim 2018: higher in overload. Rácz 2025: higher in flow than boredom at prefrontal sites. Lin 2025: lower in flow. Knierim 2025: quadratic asymmetry, absolute beta null. Soltész 2012 (10.1556/MPSzle.67.2012.1.6): declines within flow blocks. | Contradictory across five studies; also the band most contaminated by facial/neck EMG in overload. |
| Engagement index beta / (alpha + theta), Cz Pz P3 P4 | monotonic with demand (low / mid / high) | Pope, Bogart & Bartolome 1995 (10.1016/0301-0511(95)05116-3); Freeman 1999 confirmation. | A task-engagement / vigilance index; never validated against flow self-report. Carries no inverted-U. |
| Hemispheric alpha asymmetry | - | Labonté-LeMoyne 2016 (10.1145/2851581.2892356) frontal; Wolf 2015 (10.1016/j.biopsycho.2015.01.007) temporal, in experts. | Not replicated by Knierim 2018 or Knierim 2025. |

Why theta is the primary and alpha/beta are not: a signed marker needs a replicated
direction. Theta has one (flow > boredom). Alpha has only "flow sits between the extremes",
which can be used as an unsigned term but requires the subject's own boredom and overload
blocks to define the extremes. Beta has no agreed direction and the strongest EMG risk.
Alameda 2022 (10.1016/j.cortex.2022.06.005), the only systematic review, calls the whole EEG
evidence "sparse and inconclusive"; Durcan 2024 (10.1038/s44271-024-00115-3) says the same
and asks for within-participant designs with per-person difficulty, which is what the
difficulty-ladder protocol does.

Consequence for the design: the primary index is frontal theta alone. The alpha term is
kept as a separately reported exploratory index (`flow_eeg_alt`). Beta is exported as a
covariate (`beta_rel`, `engagement_index`) and never enters the primary index.

## 2. Computation pipeline, step by step

All numbers come from `config.json` `"flow_index"` and the constants in `dsp.py` /
`flow_index.py`. Sampling rate `fs` is 250 Hz with 8 channels or 125 Hz per channel with
the 16-channel Daisy.

### 2.1 Analysis filter (`Pipeline._epoch`, `dsp.FilterChain`)

1. The last `max(2 s, 10 s)` = 10 s of raw samples are taken from the 30-s `SampleBuffer`
   (2500 samples at 250 Hz, 1250 at 125 Hz).
2. A fresh, stateless `FilterChain(fs, n_ch, notch, 1.0, min(45.0, fs/2 - 1))` is built
   every epoch. It is independent of the display band-pass (default 5-50 Hz), so what the
   operator sets in the filter controls never removes theta from the index. Only the notch
   frequency is shared with the display (`notch_hz`, 60 Hz default; 50 where mains is 50).
3. Stages, in order: biquad notch at the mains frequency (Q = 30) and at its second
   harmonic when that is below Nyquist (120 Hz at 250 Hz; none at 125 Hz); 1.0 Hz high-pass
   as two cascaded 2nd-order sections (Q 0.5412 and 1.3066, a 4th-order Butterworth); 45 Hz
   low-pass, same structure (`analysis_hi_hz` = 45).
4. The filter is warmed by prepending a 1-s mirror of the window (`n // 2` samples at the
   2-s epoch length) which is discarded after filtering.

### 2.2 Welch spectrum (`dsp.welch`, `dsp.psd`)

5. The 10-s filtered window is cut into 2-s Hann segments with 50 % overlap
   (`index_window_s` = 10, `index_seg_s` = 2): segment starts at 0, 1, ..., 8 s, so 9
   segments per epoch. Each segment has its mean removed, is Hann-windowed, and gives a
   one-sided PSD in uV^2/Hz with 0.5 Hz bins (`scale = 1 / (fs * sum(w^2))`, interior bins
   doubled).
6. The clean segments (2.4) are averaged. The epoch cadence is 2 s (`epoch_s`), so
   consecutive index values share 8 of 9 segments: the index is a 10-s moving window
   updated every 2 s.

### 2.3 Band powers, relative power, log

7. Band power = sum of PSD bins with `lo <= f < hi`, times 0.5 Hz (`dsp.band_powers_of`).
   Index bands (`dsp.INDEX_BANDS`): `theta_fm` = 4-7 Hz (bins 4.0 ... 6.5), `alpha_hi` =
   10-13 Hz (bins 10.0 ... 12.5). Total (`dsp.TOTAL_BAND`) = 1-45 Hz (bins 1.0 ... 44.5).
   Covariate bands from the same Welch spectrum: theta 4-8, alpha 8-13, beta 13-30 Hz.
8. Relative power = band / total(1-45 Hz) per channel (`dsp.relative_powers`). Relative
   power is used so that electrode impedance and gain differences between sessions and
   between gel and semi-dry contacts cancel (Sugino 2024 and Knierim 2025 used relative
   power).
9. Site means (section 2.6): `theta_frontal_rel` = mean of `theta_fm` relative power over
   the frontal set; `alpha_cp_rel` = mean of `alpha_hi` relative power over the
   central-parietal set; `beta_rel` = mean beta relative power over all good channels.
10. Natural log: `log(max(theta_frontal_rel, 1e-6))`, same for alpha and beta.

### 2.4 Artifact rule (`Pipeline._epoch` `reject`, `index_min_clean_fraction`)

11. A 2-s segment is rejected when, on any channel that is enabled and whose raw quality
    is `good` (not railed / near-railed / flat), the filtered peak-to-peak exceeds
    `artifact_p2p_uv` = 150 uV **or** the filtered absolute value exceeds
    `artifact_abs_uv` = 100 uV. Channels that are not `good` are already excluded from
    the index for that epoch (they do not count toward rejection either).
12. The epoch is an **artifact epoch** when fewer than `ceil(0.5 * 9)` = 5 of the 9
    segments are clean (`index_min_clean_fraction` = 0.5). An artifact epoch carries
    `flow_eeg = null`, `artifact = true`, is written to `epochs.csv` with its covariates
    but is not appended to `eeg_hist`, is not used in any pair or block mean, and never
    enters the baseline statistics. `n_segments_used` records how many segments survived.
13. Origin of the numbers: Katahira 2018 and Ewing 2016 rejected trials / segments above
    +-100 uV; the literature file (section 3.2) allows 150 uV on frontal channels when
    more than 20 % of epochs would otherwise be lost. The code applies the 150 uV
    peak-to-peak rule on all used channels together with the 100 uV absolute rule; with
    a 1-Hz high-pass (no DC) the absolute rule is the binding one for symmetric
    transients. The 50 % clean-segment rule is per epoch; the per-block rule (< 50 %
    clean epochs = block excluded) is in `docs/VALIDATION_PROTOCOL.md`. The literature
    file suggested 60 % per block; the pre-registered value is 50 %.
13b. **v0.3 change after the first worn session (2026-11-05, ADR-12):** the rejection is
    per channel (`artifact_scope` = `channel`): each channel averages its own clean
    segments, a channel is clean for the epoch when >= 50 % of its segments are clean, and
    the index is computed over the clean channels when at least `min_clean_channels` = 2
    remain (the frontal mean in 2.3 runs over those). The v0.2 behaviour (one noisy channel
    drops the segment for everyone) is `artifact_scope` = `epoch`. In the first worn session
    39 % of segments were clean on every channel but 92 % on two or more.
13c. **Readiness gate (`gate_on_readiness`, ADR-12):** the index is **withheld** (`flow_eeg`
    null, `artifact` true, `gated` = reason, nothing enters `eeg_hist`, pairs or the
    baseline) while the good channels are one shared signal (mean |corr| of the 2-s
    filtered epoch >= 0.9: the SRB2 / BIAS clip is not on, or the headset is on a desk - a
    quiet desk passes the amplitude rule) or fewer than `min_clean_channels` are clean.
    `analysis.readiness` lists every reason (`contact`, `reference`, `settling`, `artifact`,
    `few`) and the dashboard shows it. The gate changes what is scored, never how.
14. Not implemented (so not claimed): ICA / ASR blink removal, EMG gating by 20-40 Hz
    power, interpolation of a railed channel from its homologue. Blink energy that stays
    under 100 uV passes the rule; see 2.6 and 5.

### 2.5 Baseline and z-score (`flow_index.Baseline`, mode `"rest"`)

15. Every log component is z-scored against the wearer's own eyes-open resting baseline:
    `z = (x - mean_rest) / sd_rest`, with `mean_rest`, `sd_rest` computed over the
    non-artifact epochs of the protocol block flagged `"baseline": true` (the 3-min
    "Rest, eyes open" block of `difficulty_ladder`; 90 epochs at 2 s).
16. Control path: `protocol.py` emits `block_start` / `block_end` for that block; `App.
    on_protocol_event` calls `FlowIndex.start_rest()` at the start (collection begins)
    and `FlowIndex.end_rest()` at the end, which freezes mean and SD if at least
    `warmup_epochs` = 15 clean epochs were collected. If fewer were collected the
    baseline is not frozen and the index keeps using the expanding session statistics
    (state `session-until-rest`); the report must treat such a session as INCONCLUSIVE.
17. Until the rest block has ended (and in sessions without a protocol) the baseline is
    the expanding mean / SD of everything seen so far (`session` behaviour); for the
    first 15 epochs all z are 0. A `rolling` mode (last `baseline_epochs` = 150 epochs)
    exists but is not used: a rolling baseline subtracts the slow between-block shifts
    that are exactly the validation signal (literature file section 3.2). `rest` is the
    mode fixed by this specification (`config.json` `"baseline_mode": "rest"`).
18. If `sd_rest` is below 1e-9 (a constant input) the SD is set to 1.0.
19. Why eyes-open and not eyes-closed: Rácz 2025 showed the boredom-vs-flow alpha result
    flips sign depending on which rest is the reference. The eyes-closed minute in the
    protocol is only an alpha-reactivity sanity check and never enters the baseline.

### 2.6 Site selection (`flow_index.FlowIndex.sites`)

A channel is used when it is enabled (`set_channel`) and its raw quality is `good`. The
frontal set is the first non-empty of: `FRONTAL_PREFERRED` (Fz F3 F4 F7 F8 AFz AF3 AF4 FCz
FC1 FC2), then `FRONTAL_FALLBACK` (Fp1 Fp2 Fpz), then all good channels. The
central-parietal set is `CENTRAL_PARIETAL` (C3 C4 Cz P3 P4 Pz CP1 CP2 CP5 CP6) or all good
channels. The sets actually used are exported per epoch as `frontal_sites` and `n_good`.

| Montage (`config.json`) | Labels | Frontal set (theta) | Central-parietal set (alpha_hi) | Comment |
|---|---|---|---|---|
| 8-ch default `labels_8` (Ultracortex Mark IV factory) | Fp1 Fp2 C3 C4 P7 P8 O1 O2 | **Fp1 Fp2** (fallback) | C3 C4 | Prefrontal only; blink-prone (below). P7/P8/O1/O2 are not in any index set. |
| 8-ch flow montage `labels_8_flow` (`python app.py --montage flow`) | Fz F3 F4 Cz C3 C4 P3 P4 | Fz F3 F4 | Cz C3 C4 P3 P4 | The recommended 8-ch set (section 5). |
| 16-ch `labels_16` (Cyton + Daisy) | Fp1 Fp2 C3 C4 P7 P8 O1 O2 F7 F8 F3 F4 T7 T8 P3 P4 | F7 F8 F3 F4 | C3 C4 P3 P4 | F7/F8 enter at equal weight; the literature file rates them lateral-eye-movement-prone and suggests reduced weight or exclusion. Decide before the first session whether F7/F8 are disabled (`set_channel`) and record it in the protocol header; do not change it afterwards. If all four frontal channels go bad the set falls back to Fp1 Fp2. |

Fp fallback caveat: on the 8-ch default montage "frontal theta" is measured at Fp1/Fp2
only. Blink spectra peak below 4 Hz but leak into 4-7 Hz, and the 1-Hz high-pass plus the
100 uV rule remove only large blinks. Expect inflated and noisier `theta_frontal_rel`,
lower reliability of H2 and H3 (literature file section 3.4), and a rejection rate of
10-25 % rather than the 2 % of a 64-channel wet system. Report the montage in every session
(`session.json` stores the labels).

### 2.7 The index (`flow_index.FlowIndex.update`)

```
z_theta  = z( log theta_frontal_rel )          # frontal set, theta_fm 4-7 Hz / total 1-45 Hz
z_alpha  = z( log alpha_cp_rel )               # central-parietal set, alpha_hi 10-13 Hz / total
z_beta   = z( log beta_rel )                   # all good channels, beta 13-30 Hz / total

x            = w_theta * z_theta - w_alpha * |z_alpha| - w_beta * z_beta
flow_eeg     = sigmoid(x) = 1 / (1 + exp(-x))         weights 1.0 / 0.0 / 0.0  ->  sigmoid(z_theta)
flow_eeg_alt = sigmoid( z_theta - 0.5 * |z_alpha| )   exploratory, ALT_ALPHA_WEIGHT = 0.5
```

With the fixed weights, `flow_eeg` is a monotone transform of `z_theta` and nothing else.
The sigmoid is only there to put the index on the band's 0-1 axis; it does not change rank
order, so every Spearman statistic in the report is unaffected by it.

Covariates exported per epoch and never part of `flow_eeg`: `engagement_index` = mean
absolute beta / (mean absolute alpha + mean absolute theta) over all good channels, from
the index Welch spectrum (Pope 1995 form; not log, not z-scored); `theta_alpha_ratio` =
absolute theta / absolute alpha over the frontal set; `<band>_rel_mean` for delta, theta,
alpha, beta, gamma; `z_beta`; `beta_rel`; `p2p_uv`; `n_segments_used`; `baseline_state`.

`flow_eeg_alt` differs from the literature file's `FlowEEG_2`: the literature defines
"moderate alpha" as the distance from the midpoint of the subject's own boredom and
overload blocks, which can only be computed after the session; the code uses the distance
from the eyes-open rest baseline so that it can run live. This is one more reason it is
exploratory (section 6).

## 3. Reading the scale

`flow_eeg = sigmoid(z_theta)` with the rest baseline as zero:

| `flow_eeg` | `z_theta` | Meaning |
|---|---|---|
| 0.50 | 0 | frontal theta relative power equals the wearer's eyes-open resting mean |
| 0.73 | +1 | one resting SD above rest |
| 0.27 | -1 | one resting SD below rest |
| 0.88 / 0.12 | +2 / -2 | two SD above / below |

The scale is per wearer and per session. 0.73 on one person is not comparable to 0.73 on
another; only within-session contrasts (block means, ranks) are. Before the rest block
freezes, the value is 0.50 for the first 15 epochs and then relative to the expanding
session mean; those epochs are flagged `baseline_state != "rest-frozen"` and are not part
of any block test.

Expected block ordering (literature file section 4.3): boredom low, flow high, overload
high or slightly lower than flow. The index is expected to separate flow from boredom
reliably and from overload weakly or not at all.

## 4. Known limits

- **Ceiling for agreement with self-report.** The best published cross-validated figure
  for any EEG feature set predicting flow self-report is r = 0.57 (Lin 2025, Lasso);
  Katahira's single-item correlations are r = 0.34-0.50; Eschmann's R^2 = 0.26. A band
  score that reaches r ~ 0.3-0.5 with the FSS and with this index is in the same league as
  the EEG itself. Do not expect, or demand, r > 0.6.
- **The Sugino 2024 null.** In 920 difficulty-matched matches played by expert gamers, Fz
  relative theta did not correlate positively with the FSS in 6 of 7 people. Theta tracks
  controlled effort in demand-manipulation designs; whether it tracks flow in a
  well-practised task is open. This index inherits the difficulty confound: a high value
  means "frontal theta elevated relative to rest", which the literature labels flow only
  in demand-matched blocks.
- **Hypofrontality in experts.** Rosen 2024 reports reduced frontal activity in high-flow
  improvisation by expert musicians. For expert, intrinsically motivated creative tasks
  the sign of the frontal marker may reverse. The `free_work` protocol is therefore not
  gated and must not be used for a PASS/FAIL claim.
- **8-channel Fp-only montage.** With the factory 8-ch montage the frontal set is Fp1/Fp2
  (section 2.6). No study in the literature file measured flow theta at Fp sites without
  ICA blink removal; Rácz 2025 used Fp1/Fp2/AF7/AF8 with ICA. Results on this montage are
  weaker evidence and must be labelled as such.
- **No wrist-vs-EEG precedent.** No published study validates a wrist device against an
  EEG flow index; the closest is Rácz 2025 (prefrontal 4-channel plus armband HR/IMU).
  There is no benchmark to say what "good enough" agreement is; the gates in
  `docs/VALIDATION_PROTOCOL.md` are set by us, before data, and are stated as such.
- **Flow vs overload.** Theta alone did not separate them in Katahira 2018. The engagement
  index covariate (expected boredom < flow < overload) is reported so that a flow block
  and an overload block with equal theta can still be told apart descriptively; it does
  not enter any gate.
- **Artifact handling is thresholding only** (2.4): no ICA, ASR or EMG gate. Jaw and neck
  tension in overload blocks raises beta; this contaminates `beta_rel` and
  `engagement_index`, not `theta_fm` directly, but strong EMG also raises the 1-45 Hz
  total and so lowers relative theta.
- **Epoch count.** One 6-min block gives up to 180 epochs that are not independent (10-s
  window, 2-s step). The block is the unit of inference (ADR-10); epoch-level pairs are
  descriptive.

## 5. Montage recommendation

The index needs true frontal sites. In order of preference:

1. **8-ch, re-sited: `python app.py --montage flow`** with electrodes at Fz F3 F4 Cz C3 C4
   P3 P4 (`config.json` `labels_8_flow`). The Cyton does not care which 10-20 position a
   channel sits at; move the Ultracortex nodes (or use the spider/clip electrodes) from
   Fp1 Fp2 P7 P8 O1 O2 to Fz F3 F4 Cz P3 P4. Fz alone is what Sugino 2024, Eschmann 2022 and
   the Ewing 2016 loop used; Cz / P3 / P4 give the Pope 1995 sites for the engagement
   covariate. This montage has no occipital channel, so the eyes-closed alpha check is
   read at P3/P4 instead of O1/O2.
2. **16-ch Cyton + Daisy** (`labels_16`): frontal set F7 F8 F3 F4, alpha set C3 C4 P3 P4.
   Decide before the first session whether F7/F8 are disabled (see 2.6). Costs: 125 Hz per
   channel and twice the electrode preparation.
3. **8-ch factory montage** (Fp1 Fp2 C3 C4 P7 P8 O1 O2): only if neither of the above is
   possible. Fp-only theta, lowest reliability. Label every such session "Fp-only" in the
   results log.

Reference: ear-clip SRB2 (reference) and BIAS, as the Ultracortex ships. Do not use an
average reference with 8 channels. Record the montage in the protocol header; it is part
of the pre-registration and a montage change starts a new series.

## 6. Pre-registered: do not tune

The following are fixed before the first difficulty-ladder session and are not changed
after any data from the series has been seen:

- weights `theta 1.0 / alpha 0.0 / beta 0.0` (`config.json` `"flow_index"`,
  `flow_index.DEFAULT_WEIGHTS`); `set_flow_weights` exists for exploration only and any
  use of it during a series is logged in the protocol's change table;
- bands theta_fm 4-7, alpha_hi 10-13, total 1-45 Hz; Welch 10 s / 2 s Hann / 50 %;
  relative power; natural log; baseline mode `rest`; warm-up 15 epochs;
- artifact thresholds 150 uV peak-to-peak and 100 uV absolute; 50 % clean segments per
  epoch; 50 % clean epochs per block;
- site sets `FRONTAL_PREFERRED`, `FRONTAL_FALLBACK`, `CENTRAL_PARIETAL` and the montage
  named in the protocol header;
- the gates and the verdict logic in `docs/VALIDATION_PROTOCOL.md` section 7.

`flow_eeg_alt` (theta minus 0.5 x |upper-alpha deviation|) is **exploratory**. It is
reported next to the primary index in `report.py` but it is not a gate, and it may not be
promoted to primary for the series in which it was examined. If a later series wants it as
primary, that is a new pre-registration with a new series name, and the earlier series'
verdicts stand as recorded.

Any change to the items above is a version change (`CHANGELOG.md`), is dated in the
protocol's change table, and invalidates PASS/FAIL comparisons across the change. Earlier
sessions may be re-run with the new definition, and those re-runs are labelled exploratory.

## 7. Where each number lives

| Quantity | Value | Defined in |
|---|---|---|
| Analysis band-pass | 1.0 - 45.0 Hz | `Pipeline._epoch` (lo fixed at 1.0), `config.json` `analysis_hi_hz` |
| Notch | `notch_hz` (60; 50 for 50-Hz mains) + 2nd harmonic when < fs/2 | `config.json`, `dsp.FilterChain` |
| Index window / segment / overlap | 10 s / 2 s Hann / 50 % (9 segments) | `index_window_s`, `index_seg_s`, `dsp.welch` |
| Epoch cadence | 2 s | `epoch_s` |
| theta_fm, alpha_hi, total | 4-7, 10-13, 1-45 Hz | `dsp.INDEX_BANDS`, `dsp.TOTAL_BAND` |
| Covariate bands | theta 4-8, alpha 8-13, beta 13-30 (index spectrum); delta 1-4, gamma 30-50 (display spectrum) | `Pipeline._epoch`, `dsp.BANDS` |
| Artifact thresholds | 150 uV p2p, 100 uV abs, 50 % clean segments | `artifact_p2p_uv`, `artifact_abs_uv`, `index_min_clean_fraction` |
| Baseline | mode `rest`, warm-up 15 epochs, (rolling window 150, unused) | `baseline_mode`, `warmup_epochs`, `baseline_epochs` |
| Weights | theta 1.0, alpha 0.0, beta 0.0 | `config.json` `flow_index`, `flow_index.DEFAULT_WEIGHTS` |
| Exploratory alpha weight | 0.5 | `flow_index.ALT_ALPHA_WEIGHT` |
| Site sets | see 2.6 | `flow_index.FRONTAL_PREFERRED` / `FRONTAL_FALLBACK` / `CENTRAL_PARIETAL` |
| 8-ch flow montage | Fz F3 F4 Cz C3 C4 P3 P4 | `config.json` `labels_8_flow`, `app.py --montage flow` |

Note for implementers: `Pipeline.__init__` takes any weight missing from `config.json`
from `flow_index.DEFAULT_WEIGHTS` (1.0 / 0.0 / 0.0), so a config without a `"flow_index"`
block still runs the registered theta-only index. A session's `session.json` records the
weights actually used.

## 8. Amendments

| Date | Section | Change | Reason |
|---|---|---|---|
| 2026-11-03 | all | first version, registered with v0.2.0 | - |
