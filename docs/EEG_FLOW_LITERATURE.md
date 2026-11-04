# EEG correlates of the flow state: literature base for an OpenBCI Cyton reference index

Compiled 2026-11-03 for the BioProxal wrist-wearable validation study. Purpose: define a scalp-EEG "flow index" that can be computed from an 8-channel Cyton (Fp1 Fp2 C3 C4 P7 P8 O1 O2) or 16-channel Cyton+Daisy (adds F7 F8 F3 F4 T7 T8 P3 P4) and used as a convergent reference against which the wearable's flow score is validated.

How this was compiled: each paper was read via the publisher/PMC/Europe PMC full text or, where that was paywalled or bot-blocked, the abstract or an open repository copy. The "Source read" row in each table says which. Items are tagged:

- **[E]** evidence-backed: stated in the paper that was read.
- **[E-abs]** from the abstract only; full text not accessible by this tool.
- **[I]** inference/recommendation by the compiler, not a published finding.

Headline conclusion up front: the only EEG result replicated across more than two independent labs is **frontal theta (4-7/4-8 Hz, midline Fz/AFz/FCz region) higher during a demand-matched "flow" block than during a boredom/easy block**. Alpha is reported in *both* directions depending on site, sub-band and baseline. Several recent studies (Sugino 2024; Knierim 2018; Bombeke 2018) report null or non-monotonic frontal-theta results. No published study has validated an EEG flow index on OpenBCI hardware against a wrist wearable, and the one systematic review (Alameda 2022) calls the EEG evidence "sparse and inconclusive". The EEG index proposed in section 3 should therefore be treated as a *second* convergent reference alongside block-wise self-report, not as ground truth.

---

## 1. Study-by-study tables

### 1.1 Katahira et al. 2018 — mental arithmetic, boredom/flow/overload

| Field | Content |
|---|---|
| Citation | Katahira K, Yamazaki Y, Yamaoka C, Ozaki H, Nakagawa S, Nagata N (2018). EEG correlates of the flow state: a combination of increased frontal theta and moderate frontocentral alpha rhythm in the mental arithmetic task. *Front Psychol* 9:300. DOI 10.3389/fpsyg.2018.00300. PMC5855042. |
| Source read | Full text (Frontiers HTML + Europe PMC XML). |
| N | 16 right-handed students (10 M), mean age 21.9. [E] |
| Paradigm | Two-number addition on screen, trackball keypad, no accuracy feedback. Three conditions × 3 blocks each + 3 Rest (fixation) blocks; **every block 184 s, 8 trials (first trial excluded)**; two counterbalanced block orders. Pre-experiment: 5 min Boredom-level practice + 5 min Flow-level task to estimate skill. [E] |
| Difficulty manipulation | *Boredom*: fixed 100-109 + 1-9. *Flow*: start level = mean difficulty of last 25 % of the preliminary task; two consecutive correct → harder (alternately add a digit / add a summand), two consecutive errors → easier. *Overload*: starts 3 levels above Flow start; up if ≥3/5 correct, down if ≥4/5 wrong, never below start. Accuracy 99.7 / 54.4 / 16.9 %; RT 5.5 / 14.8 / 17.2 s. [E] |
| Flow measure | Not a standard scale. 9 items adapted from Ulrich et al. 2014 + 2 items (challenge, goal), 7-point "I do not agree at all … I completely agree", plus item 11 subjective time on a 0-100 VAS. Administered **after every block**. Item wording (Table 2): 1 "I would love to solve math calculations of that kind again"; 2 "I was strongly involved in the task"; 3 "I was thrilled"; 4 "The task was boring"; 5 "I had the necessary skill to solve the calculations successfully"; 6 "Task demands were well matched to my ability"; 7 "During the task, I was concerned with other task-irrelevant issues"; 8 "During the task, my consciousness was completely focused on solving the math calculations"; 9 "The task was challenging"; 10 "I was heading toward the goal"; 11 subjective time. Factor analysis → "level of challenge" and "challenge-skill balance". Items 1, 3, 6 highest in Flow but only 1 (Bor<Flow) and 6 (Flow>Over) significant. [E] |
| EEG | BioSemi 64-ch, 1024 Hz → 256 Hz, 1-100 Hz bandpass, ICA (manual), **surface Laplacian**, trials with blinks or any channel > ±100 µV rejected (mean 2.2 %). Morlet wavelets (7 cycles). Bands: delta 1-3, **theta 4-7, alpha 10-13 (narrow, effectively upper alpha)**, beta 14-30. 7 ROIs: fronto-central AFz/Fz/FCz; L-frontal AF3/F3/F7/FC3; R-frontal AF4/F4/F8/FC4; L-central C3/C5/CP3/CP5; R-central C4/C6/CP4/CP6; L-occipital P3/P5/PO3/PO7; R-occipital P4/P6/PO4/PO8. Epoch: 1 s after trial onset to condition mean RT (so epoch length differs by condition, 4.5-16 s). Power expressed as "amount of change based on the Rest blocks" — exact formula (ratio/log/dB) not stated. [E] |
| Statistics | Repeated-measures ANOVA F(2,30), Bonferroni pairwise; correlations pooled across conditions and subjects, FDR-corrected. No effect sizes reported. [E] |
| Result: theta | Frontal ROIs: **Boredom < Flow = Overload** (p < .05; Flow vs Overload n.s.). Correlates (fronto-central theta): r = .40 with "challenging", r = .42 with "heading toward goal". [E] |
| Result: alpha | Frontal ROIs: Boredom < Flow and Boredom < Overload; Flow vs Overload **not** reported as significant. **Right-central ROI: Boredom < Flow < Overload, all pairs significant** — this is the only site where the three conditions are fully ordered. Fronto-central alpha correlated −.50 with "I had the necessary skill" and +.46 with "strongly involved". Authors' text says alpha "increased gradually" with difficulty (Discussion) but the Conclusion says Flow showed "suppressed alpha" — internally inconsistent. "Moderate alpha" is never defined quantitatively; it means Flow sat between Boredom and Overload. [E] |
| Result: beta | Left-occipital beta Boredom > Flow; authors attribute to movement artefact. [E] |
| Caveats (authors + compiler) | Theta does not separate Flow from Overload. Epoch lengths and motor/visual load differ by condition. Small N, pooled correlations violate independence. Authors state results "insufficient to serve as objective indicators of flow strength". [E] |

### 1.2 Ewing, Fairclough & Gilleade 2016 — adaptive Tetris, biocybernetic loop (and Fairclough & Ewing 2017 n-back)

| Field | Content |
|---|---|
| Citation | Ewing KC, Fairclough SH, Gilleade K (2016). Evaluation of an adaptive game that uses EEG measures validated during the design process as inputs to a biocybernetic loop. *Front Hum Neurosci* 10:223. DOI 10.3389/fnhum.2016.00223. PMC4870503. Companion: Fairclough SH, Ewing KC (2017). The effect of task demand and incentive on neurophysiological and cardiovascular markers of effort. *Int J Psychophysiol* 119:58-66. DOI 10.1016/j.ijpsycho.2017.01.007 (abstract + LJMU preprint listing read; PDF not text-extractable). |
| Source read | 2016: full text. 2017: abstract. |
| N | Study 1: 20 (11 F), 19-36 y. Study 2 (loop): 10. 2017 n-back: 18. [E] |
| Paradigm | Tetris modified so game-over cannot occur; **180 s games**. Demand: low 2.5 sq/s (~22 pieces), high 6.7 (~66), excessive 20 (~217); crossed with coin incentive, within subjects. No flow scale; NASA-TLX and DSSQ motivation. Subjective motivation peaked at *high* demand (Motivational Intensity model). [E] |
| EEG | BioSemi 64-ch, 256 Hz, 0.05-40 Hz, visual inspection, blink/saccade topographic correction, segments > 100 µV (EMG) excluded. FFT 2-s windows, 50 % overlap, over the whole 180 s; **natural-log absolute power (µV²)**; no baseline correction in Study 1. Theta = individual 1-Hz window around each person's theta peak within 4-7 Hz at **AFz, Fz, FCz, F1, F2**. Lower alpha 7.5-10, upper alpha 10.5-13 at F3/F4, C3/C4, T7/T8, P3/P4, O1/O2. [E] |
| Result: frontal theta | **Quadratic in demand: low < high > excessive**; high > both others p < .01; excessive marginally > low (p = .05). Main effect F(2,18) = 21.9, η² = 0.71. No incentive effect. [E] |
| Result: upper alpha | **Linear decrease with demand** (F(1,19) = 13.6, η² = 0.42), present at frontal, central and parietal but not occipital/temporal sites; strongest suppression at excessive; parietal low-vs-excessive η² = 0.49. Also reduced by incentive (right hemisphere). [E] |
| Result: lower alpha | No main effect of demand; only site interactions (temporal, occipital). [E] |
| Loop (Study 2) | Two electrodes only: **theta at Fz = effort/motivation axis; upper alpha at P4 = demand/workload axis**; not a ratio. 2-30 Hz FIR, 2-s Hann FFT, evaluated **every 5 s**, expressed as % change from a baseline recorded while watching a relaxing video. Rules: boredom = theta ↓ and alpha ↑ vs baseline → speed up; overload = theta ↓ and alpha ↓ → slow down. Thresholds 100/150/200 %. Outcome: few significant experience differences (N = 10); energetic arousal higher for conservative loop. [E] |
| 2017 n-back (1/4/7-back) | Fronto-medial theta curvilinear with demand (peaks at hard, falls at very hard); upper-band parietal alpha also curvilinear; systolic BP responsive to demand and incentive. [E-abs] |
| Relevance | The clearest demonstration that frontal theta is an **inverted-U in demand** (so it can in principle distinguish flow from overload, which Katahira could not) and that upper alpha is a near-monotonic demand/workload index. Not a flow study per se (engagement/effort framing). [E] |

### 1.3 Pope, Bogart & Bartolome 1995 — engagement index β/(α+θ)

| Field | Content |
|---|---|
| Citation | Pope AT, Bogart EH, Bartolome DS (1995). Biocybernetic system evaluates indices of operator engagement in automated task. *Biol Psychol* 40:187-195. DOI 10.1016/0301-0511(95)05116-3. Follow-ups: Freeman FG, Mikulka PJ, Prinzel LJ, Scerbo MW (1999) *Biol Psychol* 50:61-76; Freeman et al. 2000 *Appl Psychophysiol Biofeedback*. |
| Source read | NASA NTRS abstract; OpenAlex record (no abstract); secondary reviews for sites. The 1995 full text was not accessible. |
| Paradigm | Closed-loop adaptive automation on a NASA multi-task battery (flight-deck style); the task-allocation (manual/auto) was switched by the index under positive vs negative feedback. Four candidate indices and three control definitions; judged by whether negative feedback produced stable operation and positive feedback unstable operation. [E-abs] |
| Result | **β/(α+θ) "reflected task engagement best"** among candidates. Freeman 1999 confirmed it against β/α and 1/α in a compensatory tracking task. [E-abs] |
| Sites / bands | Secondary sources (Freeman et al.; a 2020s review of EEG engagement indices) give: bands averaged over **Cz, Pz, P3, P4**; theta 4-8, alpha 8-13, beta 13-22 Hz; index updated on short (≈1.67 s) FFT windows with a sliding ~20-s average (window figures from compiler's prior knowledge of the NASA reports, **not verified in this pass**). [E-abs for sites; window figures unverified] |
| What it indexes | Task engagement / vigilance / arousal in sustained-attention and automation contexts. It was never validated against flow self-report and is expected to rise monotonically with demand/arousal (it carries no inverted-U). [E-abs + I] |

### 1.4 Alameda, Sanabria & Ciria 2022 — systematic review (Cortex)

| Field | Content |
|---|---|
| Citation | Alameda C, Sanabria D, Ciria LF (2022). The brain in flow: A systematic review on the neural basis of the flow state. *Cortex* 154:348-364. DOI 10.1016/j.cortex.2022.06.005. Preprint: hdl.handle.net/10481/78493. |
| Source read | Abstract (Europe PMC). Full text and preprint both blocked (paywall / bot protection). |
| Scope | PRISMA; Web of Science, Scopus, PsycINFO to March 2022; EEG, fMRI, fNIRS, tDCS studies that measured or manipulated flow. **25 studies, 471 participants** (secondary sources: 11 of them EEG). Quality assessed with **Cochrane RoB 2**. [E-abs] |
| Conclusions | Experimental studies point to attention, executive-function and reward systems, with anterior regions (DLPFC, MPFC, IFG) central — **but "the activity dynamics of these regions during flow were inconsistent across studies"**; evidence is "sparse and inconclusive", limiting theoretical debate. Main limitations: few studies, high heterogeneity, "important methodological constraints". Recommendation: future work must attend to experimental design and to **how flow is measured**. [E-abs] |
| Not retrievable | Per-study RoB ratings and the detailed recommendation list. A secondary summary (iMotions; Durcan 2024) attributes the heterogeneity to methodological factors: difficulty-level confounds, short sessions, unfamiliar tasks. [E-abs] |

### 1.5 Berta et al. 2013 — 4-electrode consumer EEG, games

| Field | Content |
|---|---|
| Citation | Berta R, Bellotti F, De Gloria A, Pranantha D, Schatten C (2013). Electroencephalogram and physiological signal analysis for assessing flow in games. *IEEE Trans Comput Intell AI Games* 5(2):164-175. DOI 10.1109/TCIAIG.2013.2260340. |
| Source read | Abstract (OpenAlex). Full text paywalled. |
| Paradigm | Game levels designed to induce boredom, flow, anxiety; 4-electrode commercial EEG + peripheral signals; SVM classification. [E-abs] |
| Result | "The most informative frequency bands for discriminating among gaming conditions are **around low beta**"; peripheral signals add marginal information; 3-class accuracy 66.4 % user-dependent vs 50.1 % user-independent; 7-8 min personal calibration. A 2017 review chapter summarises it as alpha, low-beta and mid-beta showing the greatest differences across flow/boredom/anxiety (secondary). No direction or site reported in the abstract. [E-abs] |
| Relevance | Shows that user-independent spectral classifiers generalise poorly (≈chance+17 %) — supports per-subject calibration blocks. [E-abs + I] |

### 1.6 Knierim et al. (Karlsruhe) — knowledge work, 2018 → 2025

| Field | Knierim, Nadj, Hariharan, Weinhardt 2018 (PhyCS) | Knierim, Stano, Kurz, Heusch, Wilson 2025 (CHI '25) |
|---|---|---|
| Citation | Flow neurophysiology in knowledge work: EEG observations from two cognitive tasks. PhyCS 2018, pp. 42-53. DOI 10.5220/0006926700420053. (Also: Knierim et al. 2018 *Exploring flow psychophysiology in knowledge work*, LNISO, DOI 10.1007/978-3-030-01087-4_29 — abstract not retrievable; secondary mention: 6-participant cEEGrid math study linking more intense flow to moderate temporal alpha.) | Exploring flow in real-world knowledge work using discreet cEEGrid sensors. CHI 2025. DOI 10.1145/3706598.3713512; arXiv 2501.19217. |
| Source read | Full text (SciTePress HTML). | Full text (arXiv HTML, first 100 k chars). |
| N | 12 recruited; **EEG analysis N = 7** right-handed males. | 21 recruited, 17 usable EEG. |
| Paradigm | Task 1: writing own thesis in Word, 3 × 7 min rounds. Task 2: Katahira-style addition, boredom (101+1/2/3) / flow (adaptive after 3 consecutive correct/incorrect) / overload, **5 min each**, 60-s washouts; baselines 5 min EO, 1 min EC. | Home-office day: 4 × 1-h self-selected project sessions (thesis or coding) + 2 × 3-min arithmetic (start/end). **Flow Short Scale probes every 10-20 min**. |
| Flow measure | **Flow Short Scale, 10 items** (Engeser & Rheinberg 2008) + 1 demand item; stress (5 items); SAM arousal. | FSS 10 items, α = .91. |
| EEG | Emotiv EPOC+ 14-ch, 256 Hz; frontal sites AF3 F3 F7 FC5 / FC6 F8 F4 AF4. 0.5-45 Hz, 50 Hz notch, manual rejection, ICA (EOG). 2-s Hann epochs, 50 % overlap, FFT, **ln absolute power**, median per phase, **change score = phase − eyes-open baseline**. theta 4-8, alpha 8-12 (lo 8-10, hi 10-12), beta 12-30. | Gold open-cEEGrid around-ear, **OpenBCI Cyton+Daisy 16-ch, 250 Hz**; 2-30 Hz, ZapLine, re-ref L5/R5, Riemannian ASR. 170-s windows; Welch 2-s, no overlap; **relative power (bin / total)**; theta 4-7, alpha 8-12, beta 13-30; asymmetry L−R. No baseline correction. |
| Statistics | Friedman + Wilcoxon (non-parametric); no effect sizes (median-difference bands). | Mixed models with quadratic terms, two-line tests. |
| Theta result | **No significant theta effect**; descriptively flow > boredom (ln diff 0.17). Authors: theta's role "not supported"/"reduced". | **Inverted-U between relative theta and FSS**: math quadratic p = .043; natural work trend p = .075; rising limb significant in both (p < .002), falling limb n.s. |
| Alpha result | Only **hiAlpha (10-12 Hz)** significant: **boredom > overload (sig), boredom > flow (trend, large diff 0.49)**; flow = overload. Authors read lower frontal upper alpha in flow as frontal *activation* (contra hypofrontality). No hemispheric asymmetry. | Alpha power: **null**. Alpha asymmetry: null (Labonté-LeMoyne 2016 asymmetry **not replicated**). |
| Beta result | Overload > boredom/flow descriptively (0.15-0.23). | **Beta asymmetry quadratic with flow in natural work** (p = .003), peak near zero asymmetry; absolute beta null. |
| Caveats | N = 7; Emotiv; no effect sizes. | Field data; day-to-day variance; few high-flow data points after inflection. |

### 1.7 Sugino, Ideriha, Yamazaki, Ushiyama 2024 — 7 gamers × >100 matches (null FMT)

| Field | Content |
|---|---|
| Citation | The flow state is not accompanied by frontal-midline theta activity: an EEG investigation of more than 700 video gameplay sessions. *bioRxiv* 2024.07.11.603158 (preprint, not peer-reviewed). |
| Source read | Full text. |
| N / design | 7 adults (6 M, mean 24 y), each 10 non-consecutive lab days playing their **own favourite, most-skilled** competitive game (fighting ×3, shooter ×2, racing, rhythm); difficulty matched by online rank (≈50 % win). 1,009 trials, 920 analysed. [E] |
| Flow measure | **Flow Short Scale (10 items, 7-point) after every match**; separate 9-point challenge-skill item (5 = optimal). [E] |
| EEG | 19-ch 10-20, Ag/AgCl, earlobe ref, 1000 Hz; 3-100 Hz, notches; 1-s epochs, Welch 1-s Hann; visual rejection + ICA. **FMT ratio = theta 4-7 Hz power / 1-49 Hz power at Fz** (relative). 2-min eyes-open rest daily. [E] |
| Statistics | Within-participant only: Mann-Whitney (gameplay vs rest; difficulty bins), Spearman (FSS vs FMT), BH-FDR per person. [E] |
| Result | Gameplay FMT > rest in 5/7. **FSS vs FMT: no significant positive correlation in 6/7** (one positive, one negative). Optimal-difficulty trials did not show higher FMT in anyone. Flow index correlated with RR-interval minimum (shorter RR = deeper flow) in 4/7 and was higher in won matches in 7/7. No other bands analysed. [E] |
| Authors' explanation | Prior positive findings used unfamiliar tasks, few trials, group statistics, and **compared optimal vs non-optimal difficulty — confounding difficulty with flow**. [E] |

### 1.8 Lin, Guo, Zhuang, Zhang, Wang 2025 — video game, Lasso prediction of flow

| Field | Content |
|---|---|
| Citation | Neural oscillations predict flow experience. *Cogn Neurodyn* 19(1). DOI 10.1007/s11571-024-10205-x (online Dec 2024). |
| Source read | Abstract (Europe PMC). Full text paywalled. |
| Paradigm | Video-game difficulty adjusted to induce flow vs non-flow; self-reported flow scores. [E-abs] |
| Result | Flow vs non-flow: **higher theta, "moderate" alpha, lower beta**. Lasso regression on neural features predicted self-reported flow, **r = 0.571, p < .01**. [E-abs] |
| Relevance | Independent replication of the Katahira pattern (theta ↑, alpha intermediate) plus a beta ↓; and the first published cross-validated prediction figure, which sets a realistic ceiling (~r 0.5-0.6) for how well *any* EEG index tracks flow self-report. Sites/bands/N not retrievable. [E-abs + I] |

### 1.9 Rácz et al. 2025 — Tetris boredom/flow/frustration with a 4-channel prefrontal wearable

| Field | Content |
|---|---|
| Citation | Rácz M, Becske M, Magyaródi T, Kitta G, Szuromi M, Márton G (2025). Physiological assessment of the psychological flow state using wearable devices. *Sci Rep* 15:11839. DOI 10.1038/s41598-025-95647-x. PMC11977251. |
| Source read | Full text (Europe PMC XML). |
| N / design | 28 recruited, 24 analysed. Modified Tetris; **10-min blocks**, order randomised; boredom (1 s/square, no drop), flow (normal, 0.5 s/square start), frustration (0.15 s/square, random rotation). Baseline 1 min EO + 1 min EC. Analysed 8-min segments in 1-min windows. [E] |
| Flow measure | PPL Flow State Questionnaire, 20 items, 5-point; subscales challenge-skill balance (11) and absorption (9); plus short PANAS. [E] |
| EEG | 4 channels **AF7 Fp1 Fp2 AF8**, 250 Hz, stainless-steel electrodes, ref behind ears; 0.5-30 Hz Butterworth; ICA for blinks; 1-min FFT; bands delta 0.5-4, theta 4-7, alpha 8-12, beta 13-30; log power, **baseline-corrected to EC and EO rest** (equation in image). One-sample t-tests per channel, no multiple-comparison correction, no effect sizes. [E] |
| Result (EC-baseline) | **Boredom < Flow: alpha at AF7; beta at AF7, Fp1, Fp2**. **Flow > Frustration: theta and alpha at Fp1; delta, theta, alpha at AF8**. Boredom > Frustration: delta AF7. [E] |
| Result (EO-baseline) | Flow > Frustration in delta (Fp2, AF8), theta and alpha (AF8); boredom > frustration similarly; **no boredom-vs-flow difference**. [E] |
| Peripheral | HR: boredom > flow < frustration (U-shape described, not formally tested); HR SD: flow < frustration (p = .0001). SpO2, GSR n.s. Head/arm motion lowest in flow. [E] |
| Relevance | Prefrontal-only wearable: the theta/alpha *elevation* in flow is relative to **frustration/overload**, the alpha/beta elevation is relative to **boredom**. Direction of alpha here (flow > boredom) is opposite to Knierim 2018 and Ewing 2016, consistent with Katahira frontal. Pattern depends on baseline choice. [E + I] |

### 1.10 Rosen, Oh, Chesebrough, Zhang, Kounios 2024 — jazz improvisation ("creative flow")

| Field | Content |
|---|---|
| Citation | Creative flow as optimized processing: evidence from brain oscillations during jazz improvisations by expert and non-expert musicians. *Neuropsychologia* 196:108824. DOI 10.1016/j.neuropsychologia.2024.108824 (CC BY-NC). |
| Source read | Abstract + Drexel repository page. ScienceDirect full text blocked. |
| N / design | 32 jazz guitarists (high vs low experience); 6 improvisations each to lead sheets; **Core Flow State Scale** after each; high-density EEG, SPM12 source reconstruction. [E-abs] |
| Result | High- vs low-flow improvisations: **"transient hypofrontality"** across all musicians; experts: reduced posterior default-mode activity + **left-hemisphere auditory/somatosensory flow-related activity**; novices: right-hemisphere deactivations, no DMN/FPCN modulation. Press coverage: frontal executive regions *less* active in intense flow. Band-specific directions not retrievable from the abstract. [E-abs] |
| Relevance | Supports hypofrontality for expert, intrinsically motivated creative tasks — the opposite of the frontal-theta-up story from effortful arithmetic. Flow phenomenology may have different EEG signatures by task class. [E-abs + I] |

### 1.11 Wolf et al. 2015 — table tennis, temporal alpha asymmetry

| Field | Content |
|---|---|
| Citation | Wolf S, Brölz E, Keune PM, Wesa B, Hautzinger M, Birbaumer N, Strehl U (2015). Motor skill failure or flow-experience? Functional brain asymmetry and brain connectivity in elite and amateur table tennis players. *Biol Psychol* 105:95-105. DOI 10.1016/j.biopsycho.2015.01.007. PMID 25616246. |
| Source read | Abstract (Europe PMC). |
| N / design | 14 elite vs 15 amateur players; motor imagery of returning serves shown on video; flow assessed (instrument not named in abstract). [E-abs] |
| Result | **Alpha (8-13) asymmetry at T3/T4**: experts show stronger shift toward *lower relative left-temporal activity* (i.e., higher left-temporal alpha), which correlates with more flow and higher world ranking; experts also show stronger right temporal-premotor (T4-Fz) theta coherence. **Not frontal alpha.** [E-abs] |

### 1.12 Núñez Castellar et al. 2019 (and Bombeke et al. 2018) — gameplay with auditory oddball

| Field | Content |
|---|---|
| Citation | Núñez Castellar EP, Antons J-N, Marinazzo D, Van Looy J (2019). Mapping attention during gameplay: assessment of behavioral and ERP markers in an auditory oddball task. *Psychophysiology* 56(7):e13347. DOI 10.1111/psyp.13347 (preprint PeerJ 2482, 2016). Bombeke K et al. (2018). Do not disturb: psychophysiological correlates of boredom, flow and frustration during VR gaming. HCII 2018, LNCS 10915:101-119. DOI 10.1007/978-3-319-91470-1_10. |
| Source read | 2019 abstract; 2018 abstract. |
| Paradigm | Game at boredom / flow / frustration difficulty with a secondary auditory novelty oddball. [E-abs] |
| Result | Flow: slower RT and more errors on the secondary task; delayed response-locked fronto-central negativity (medial frontal source); **only alpha showed a significant power increase in flow, maximal at midfrontal sites**, and higher alpha correlated with faster RTs — interpreted as inhibitory top-down control. **Bombeke 2018 (VR replication attempt): could not reliably induce boredom/flow/frustration and did not replicate the oddball effect**; found lower P300 and more high-frequency activity in VR vs monitor. [E-abs] |

### 1.13 Labonté-LeMoyne et al. 2016 — collaborative tennis game (CHI EA)

| Field | Content |
|---|---|
| Citation | Labonté-LeMoyne É, Léger P-M, Resseguier B, Bastarache-Roberge M-C, Fredette M, Sénécal S, Courtemanche F (2016). Are we in flow? Neurophysiological correlates of flow states in a collaborative game. CHI EA 2016:1980-1988. DOI 10.1145/2851581.2892356. |
| Source read | Abstract (OpenAlex). |
| Result | Preliminary doubles-tennis study; reports an interaction between one player's EEG activity and the partner's flow; secondary sources (Knierim 2025) cite it for a **frontal alpha asymmetry** flow correlate that Knierim 2025 then failed to replicate. Bands/sites not in abstract. [E-abs] |

### 1.14 Eschmann, Riedel & Mecklinger 2022 — causal test: Fz theta neurofeedback raises flow

| Field | Content |
|---|---|
| Citation | Theta neurofeedback training supports motor performance and flow experience. *J Cogn Enhanc* 6:434-450. DOI 10.1007/s41465-021-00236-1. PMC9360146. |
| Source read | Full text (PMC). |
| Design | N = 48 students; one 30-min session up-regulating **Fz theta 4-8 Hz** (earlobe ref; 1-s FFT; % change from 5-min rest). Flow: **Flow State Scale-2, 36 items, 5-point**, summed. Task: finger-tapping sequence. [E] |
| Result | Responders (n = 25, mean theta +4.1 %) vs non-responders (+0.2 %): higher flow at post-test (b = 9.18, p = .024); theta gain predicted flow gain (R² = .263, p < .001), surviving control for motor gain. Effect sizes small-medium (flow d = 0.30). No active control; mainly correlational. [E] |
| Relevance | The only interventional link between FMT and flow self-report; supports theta as the primary channel, with modest effect size. [E] |

### 1.15 Durcan, Holland & Bhattacharya 2024 — methodological framework

| Field | Content |
|---|---|
| Citation | A framework for neurophysiological experiments on flow states. *Commun Psychol* 2:66. DOI 10.1038/s44271-024-00115-3. PMC11332228. |
| Source read | Full text (Europe PMC XML; first ~100 k chars incl. the checklist). |
| Key points | Reviews of flow neuroscience "consistently highlight a lack of consensus". Confounds named: comparing challenge *levels* instead of challenge *types*; challenge not tailored to skill; pressured speed and fixed goals induce "clutch" not flow; unfamiliar tasks; movement-restricted labs; **short sessions (~12 min mean in game studies)**; experts may need ≥25 min to enter flow. [E] |
| 24-item checklist (Box 2) | 1a state which theoretical conditions are met. 2a-e manipulate challenge type (feedback, interaction speed, goal openness, number of goals, session duration). **3a set individual challenge from pre-test performance; 3b use dynamic difficulty adjustment; 3c keep relative challenge differences equal across conditions.** 4a-g ensure skill/self-efficacy (training, practice to remove novelty, pre-test skills). 5a-d ensure intrinsic motivation (choice of task, SIMS pre-measure). **6a within-participant design, all conditions; 6b multiple tasks.** 7a drop cases where manipulation check fails; 7b analyse flow dimensions separately, don't fold antecedent items into a composite. [E] |
| Self-report | Flow State Scale, Psychological Flow Scale, Core Flow; antecedent items (challenge-skill) used as manipulation checks, not as the flow score. Conservative multiple-comparison correction and G*Power sensitivity analyses recommended. [E] |

### 1.16 Other studies located (brief)

| Study | Design | EEG result | Read |
|---|---|---|---|
| Soltész P, Magyaródi T, Mózes T, Nagy H, Oláh A (2012). *Magyar Pszichológiai Szemle* 67:77-103. DOI 10.1556/MPSzle.67.2012.1.6 | 20 M, collision-avoidance game, speed individually adjusted → flow/boredom/anxiety | Across delta/theta/beta/gamma: **boredom < flow < anxiety**; in the second half flow shows a beta/gamma decline. Sites not in abstract. | E-abs |
| "The after-glow of flow" (2023). *Creativity Res J*. DOI 10.1080/10400419.2023.2277042 | Musicians, naturalistic performance, pre/post | After flow: higher **upper alpha 10-12 and beta 15-30, mainly frontal** | E-abs (search snippet only) |
| Bieri U et al. (2025). *Sci Rep* 15:40710. PMC12630671 | 20 surgeons, da Vinci VR, **Muse S (AF7/AF8 usable only)**, FSS 13 items | **AF8 theta higher in high performers**; FSS not directly linked to theta; HR null | E |
| Beiramvand M, Koivula R, Lipping T (2025). arXiv 2609.19737 / *Biomed Signal Process Control* | 45 participants, adaptive Tetris, Muse-S + Emotiv Insight, entropy features | RF 97 % random-split vs **75 % leave-one-subject-out** | E-abs |
| Eldaly et al. (2026). Neural symphony of flow experience. *NeuroImage* 338:122049 | 64-ch, video game, flow/boredom/anxiety | Flow = highest global connectivity, metastability, dimensionality; no band-power result | E-abs |
| Rosso G et al. (2025). arXiv 2506.16838 | Muse, athletes, proprietary "Flow State Index" | No N, bands or accuracy in abstract | E-abs; low weight |
| Chanel G, Rebetez C, Bétrancourt M, Pun T (2011). *IEEE SMC-A* 41:1052-1063 | Tetris 3 levels, EEG + peripheral | EEG > peripheral for short windows; fusion 63 % 3-class | E-abs |
| Missouri S&T MSc thesis (N = 44, 64-ch, Tetris flow/boredom/anxiety; scholarsmine.mst.edu) | — | Flow: lower occipital alpha vs rest; lower left-parietal theta vs rest; fronto-temporal theta flow > anxiety | grey literature, search snippets only |

No quantitative meta-analysis of EEG flow correlates was found (searched 2023-2026). The Alameda 2022 review is narrative.

---

## 2. Synthesis

### 2.1 What is consistent [E]

1. **Frontal/fronto-midline theta is higher in the demand-matched block than in the boring/easy block.** Katahira 2018 (frontal ROIs, Laplacian, 4-7 Hz), Ewing 2016 (AFz/Fz/FCz/F1/F2, individualised theta), Fairclough & Ewing 2017, Knierim 2025 (rising limb of the inverted-U, relative theta, around-ear), Rácz 2025 (Fp1/AF8 theta flow > frustration only), Lin 2025, Bieri 2025 (AF8 theta in high performers). Eschmann 2022 adds a (weak) causal link from Fz theta to FSS-2 flow.
2. **Theta alone does not separate flow from overload in arithmetic** (Katahira: Flow = Overload), but **does in Tetris/n-back when an "excessive" level is included** (Ewing: inverted-U; Knierim 2025: inverted-U vs FSS). The difference is probably that Katahira's overload was still *attempted* effortfully (17 % accuracy, 17-s trials) whereas Ewing's excessive level caused disengagement. [E + I]
3. **Upper alpha (≈10-13 Hz) over central/parietal cortex tracks task demand/workload**, and flow sits at an intermediate value between the easy and the hard condition. Direction of the monotonic trend differs (see 2.2), but "flow is in the middle" is reported by Katahira (right-central: Bor < Flow < Over), Ewing (Fz/P4: alpha falls with demand), Lin 2025 ("moderate alpha"), Soltész 2012 (all bands: boredom < flow < anxiety).
4. **Within-subject designs with per-participant adaptive difficulty and block-wise self-report are the standard paradigm** (Katahira, Knierim 2018, Rácz, Sugino, Ewing), and the systematic review + Durcan framework both demand it.
5. **User-independent classifiers generalise poorly** (Berta 50 %; Beiramvand 75 % LOSO vs 97 % random split), so any reference index needs per-subject calibration/normalisation.
6. **Effect sizes, where reported, are large for demand manipulations (η² 0.4-0.7 in Ewing) but modest for flow self-report prediction** (Lin r = .57 cross-validated; Eschmann R² = .26; Katahira item correlations r = .34-.50 on pooled data).

### 2.2 What is inconsistent or null [E]

1. **Alpha direction depends on site, sub-band, task and baseline.**
   - Alpha *rises* with difficulty/flow at frontal and right-central sites relative to a fixation rest (Katahira, 10-13 Hz); at prefrontal AF7/Fp1 relative to eyes-closed rest (Rácz, flow > boredom); midfrontal during flow with oddball (Núñez Castellar).
   - Alpha *falls* with demand at frontal/central/parietal sites in absolute log power (Ewing upper alpha; Knierim 2018 frontal hiAlpha boredom > flow/overload).
   - Alpha is *null* in relative power around the ear (Knierim 2025) and under an eyes-open baseline for boredom vs flow (Rácz).
   Katahira's own paper is internally inconsistent ("gradually increased" vs "suppressed").
2. **Frontal-midline theta is null or reversed in several studies**: Knierim 2018 (Emotiv, N = 7, no theta effect); Sugino 2024 (Fz relative theta, 920 well-matched matches in expert players, no positive correlation with FSS in 6/7); Bombeke 2018 could not induce the states at all; Rosen 2024 reports *hypofrontality* in expert creative flow.
3. **Beta** is reported as: most discriminative (Berta, low beta), higher in overload (Knierim 2018), higher in flow than boredom at prefrontal sites (Rácz), lower in flow (Lin 2025), quadratic asymmetry (Knierim 2025), declining within flow blocks (Soltész). No consistent direction.
4. **Hemispheric alpha asymmetry** (Labonté-LeMoyne 2016 frontal; Wolf 2015 temporal) was not replicated by Knierim 2018 or 2025.
5. **The difficulty confound**: every positive "flow" EEG finding except Sugino's and Rosen's comes from comparing difficulty levels, so "theta up in flow" is not separable from "theta up with controlled effort" (Durcan 2024; Sugino 2024; Alameda 2022). The index below inherits that limitation.

---

## 3. Recommended operational index for the Cyton

Everything in this section is **[I]** unless a bracketed source is given. The design principle: use only the one replicated marker (frontal theta) as the signed core, use alpha as an *unsigned* "intermediate demand" term calibrated per subject, keep the Pope engagement index as a separate monotonic demand covariate, and never fold the self-report-antecedent items into the criterion.

### 3.1 Channels

| Role | 16-ch Cyton+Daisy | 8-ch Cyton with the stated montage | Better 8-ch montage if electrodes can be moved |
|---|---|---|---|
| Frontal theta (primary) | mean of **F3, F4** (nearest to the Fz/F1/F2/AFz cluster used by Ewing/Katahira/Eschmann/Sugino) — optionally add F7, F8 at reduced weight; include Fp1/Fp2 only after blink-component removal | **Fp1, Fp2** are the only frontal sites; usable *only* with ICA/regression blink removal and strict epoch rejection, otherwise theta is dominated by blink energy (blink spectra peak < 4 Hz but leak into 4-7 Hz). C3/C4 theta is a poor substitute. | Reassign one or two of P7/P8/O1/O2 to **Fz** (and **Cz**) — the Cyton does not care which 10-20 site a channel is placed at. Fz alone is what Sugino, Eschmann and the Ewing loop used. |
| Upper alpha "demand" term | mean of **C3, C4, P3, P4** (Katahira right-central ROI; Ewing P4 / parietal) | **C3, C4** (closest to Katahira's right-central effect); P7/P8 are temporo-parietal and were outside Katahira's and Ewing's alpha-effect sites — use with reduced weight | add **Pz/P4** |
| Engagement index β/(α+θ) | **C3, C4, P3, P4** (Pope sites were Cz, Pz, P3, P4; C3/C4/P3/P4 is the closest available set) | C3, C4, P7, P8 (approximation) | Cz, Pz, P3, P4 (exact) |
| Visual/arousal control, artefact check | O1, O2 (occipital alpha blocking = eyes-open/engagement sanity check; not part of the flow index) | O1, O2 | O1 |
| Not used | T7/T8 (EMG-prone), F7/F8 (lateral eye movements) | P7/P8 | — |

Reference: earlobe or mastoid linked reference (Sugino, Eschmann); Cyton SRB2 to one earclip, BIAS to the other. Do not use an average reference with 8 channels.

### 3.2 Signal processing (per block and per rolling epoch)

- Sampling 250 Hz (Cyton native). Bandpass 1-40 Hz (Butterworth 4th order, zero-phase offline); notch 50 Hz (EU) or 60 Hz (US); optional ZapLine (Knierim 2025).
- **Windows: Welch PSD, 2-s Hann, 50 % overlap** (Ewing 2016; Knierim 2018/2025) → 0.5 Hz bins. Aggregate windows into **10-s epochs for the rolling index** (Ewing's loop used 5 s; 10 s halves variance and is still shorter than any wearable flow-score cadence) and into **block means for validation**.
- Bands: theta 4-7 Hz (Katahira, Sugino, Rácz), alpha split into lower 8-10 and **upper 10-13 Hz** (Ewing; Katahira's "alpha" was 10-13), beta 13-30 Hz; total 1-40 Hz. Optional: individualise theta as a 1-Hz window around the subject's theta peak at Fz/F3/F4 during the hardest block (Ewing) — only if the peak is clearly visible.
- **Power metric: relative power (band / total 1-40 Hz) and then natural log**, i.e. `log(P_band / P_total)`. Rationale: relative power removes electrode-impedance and gain differences between sessions and between gel/semi-dry contacts (Sugino, Knierim 2025 used relative; Ewing/Knierim 2018 used log absolute). Keep log-absolute as a secondary feature set; report both in the validation.
- **Baseline: 3-min eyes-open rest (fixation cross) at the start of every session, plus 1-min eyes-closed** (for the Berger-effect quality check only). Express every feature as a z-score against the mean and SD of the eyes-open rest epochs: `z = (x_epoch − mean_rest) / sd_rest`. Katahira and Eschmann normalised to rest; Ewing's loop to a relaxing-video baseline; Rácz showed the boredom-vs-flow alpha result flips sign depending on EC vs EO baseline — so **fix EO as the baseline and never mix**. A rolling baseline is **not** recommended for the reference index because it would remove slow between-block shifts, which are exactly the validation signal; a rolling baseline is only appropriate inside the wearable's own real-time scoring.
- **Artefact handling** (frontal gel or semi-dry electrodes, blinks at Fp1/Fp2):
  1. Impedance check < 20 kΩ gel / < 100 kΩ semi-dry before each block; log it.
  2. Epoch rejection if any channel exceeds **±100 µV** peak-to-peak after filtering (Katahira; Ewing) — relax to ±150 µV for Fp1/Fp2 only if more than 20 % of epochs are lost, and report the rejection rate (Katahira: 2.2 % with 64-ch wet; expect 10-25 % with Cyton frontal channels).
  3. Blinks: with 16 channels run extended-Infomax ICA on the whole session and remove EOG components (Katahira, Knierim 2018, Sugino, Rácz); with 8 channels ICA is unreliable, so use ASR (Knierim 2025 used Riemannian ASR on 16 channels) or a regression of Fp1/Fp2 against a blink template, and additionally drop any 2-s window where Fp1-Fp2 contain a deflection > 75 µV.
  4. EMG: reject windows with beta+gamma (20-40 Hz) relative power > 3 SD above the subject's rest value; this also protects the engagement index's beta numerator from jaw/neck tension in overload blocks.
  5. Flat/railed channel detection (Cyton rails at ±187 mV; a flat line indicates lost contact); interpolate from the homologous channel for the block, flag it.
  6. Minimum data rule: a block is only scored if ≥ 60 % of its epochs survive.

### 3.3 The index

Compute per epoch (and per block) for subject *s*:

```
T  = z( log relθ   at mean(F3,F4)          )      # frontal theta, signed +
A  = z( log relα_u at mean(C3,C4,P3,P4)    )      # upper alpha 10-13 Hz, central-parietal
EI = z( log( β / (α+θ) ) at mean(C3,C4,P3,P4) )   # Pope engagement index, demand covariate
```

**Primary reference index (pre-registered):**

```
FlowEEG_1 = T
```

Frontal theta alone is the only component with replicated direction. Expected ordering across validation blocks: **boredom < flow ≥ overload** (Katahira: flow = overload; Ewing: flow > excessive). The index is therefore expected to *separate flow from boredom reliably and from overload weakly*.

**Secondary index with a "moderate alpha" term (exploratory):**

```
A_mid      = ( A_boredom_block + A_overload_block ) / 2        # subject's own calibration midpoint
FlowEEG_2  = T  −  w · | A − A_mid |            with w = 0.5 as the pre-registered default
```

This is the only defensible operationalisation of "moderate alpha": the literature does not agree on whether alpha rises or falls with demand, but agrees that the flow block lies *between* the two extremes. Taking the absolute distance from the subject's own boredom/overload midpoint makes the term direction-agnostic and strictly requires the calibration blocks that the validation paradigm provides anyway. It cannot be computed in free-living use, which is a reason to keep FlowEEG_1 as the primary. Do **not** define "moderate" as a fixed µV² or fixed relative-power band — no paper gives one and Rácz shows the sign flips with baseline.

**Engagement index (not in the flow index):** report EI as a separate monotonic demand/vigilance covariate. Expected ordering **boredom < flow < overload** (inference from Pope/Freeman's engagement semantics; no flow study has reported EI across boredom/flow/overload). Its role is (a) a manipulation check that overload really raised arousal, (b) a way to tell "flow vs overload" apart when FlowEEG_1 cannot: high T with mid EI = flow; high T with top EI = overload.

**Why beta is not in the index:** beta direction is inconsistent across five studies (section 2.2) and beta at scalp sites is the band most contaminated by EMG from the facial/neck tension that overload produces. Keep beta inside EI only, where it is divided by alpha+theta and EMG-gated.

**Scaling for comparison with the wearable:** per subject, map FlowEEG to 0-100 by the within-session min-max over all scored epochs, or better, leave it as z and compare rank-order (Spearman) with the wearable score. Do not calibrate it to the FSS score — the FSS is the criterion for both.

### 3.4 What the index can and cannot claim

- It inherits the difficulty confound: a high FlowEEG_1 means "frontal theta is elevated relative to rest and relative to the easy block", which the literature labels flow in demand-matched blocks but which Sugino 2024 shows does not track trial-level flow in expert gamers. [E]
- The realistic ceiling for correlation between an EEG index and block-wise flow self-report is about r ≈ 0.4-0.6 (Lin 2025 r = .57 with a fitted Lasso; Katahira single-item correlations .34-.50). A wearable score that reaches r ≈ 0.3-0.5 with FSS and r ≈ 0.3-0.5 with FlowEEG_1 would be in the same league as the EEG itself. [E + I]
- With Fp1/Fp2 as the only frontal sites (8-ch montage as stated), expect substantially lower reliability for T than with F3/F4 or Fz; the compiler's recommendation is to re-site two electrodes to Fz and Cz before the study rather than accept prefrontal-only theta. [I]

---

## 4. Recommended validation paradigm

### 4.1 Design (within-subject, three-level difficulty, [E] where sourced)

| Element | Recommendation | Basis |
|---|---|---|
| Task | A continuous task with objective difficulty control and a single clear goal: adaptive mental arithmetic (Katahira/Knierim: two-number addition with digit-count levels) **or** speed-controlled Tetris (Ewing, Rácz, Chanel, Harmat). Arithmetic is easier to adapt per trial and has less hand/head movement; Tetris produces stronger flow reports and better matches "game" wearables. Choose one and keep it for all subjects. | Katahira 2018; Ewing 2016; Rácz 2025; Durcan item 2 |
| Skill calibration | 5-min practice at the easy level + 5-min adaptive run; take the mean level of the last 25 % as the Flow starting level (Katahira). Durcan 3a/3c: set challenge from pre-test performance and keep the relative distance between conditions equal for everyone. | Katahira 2018; Durcan 2024 |
| Conditions | **Boredom** (fixed, trivially easy, no adaptation); **Flow** (staircase: 2 consecutive correct → +1 level, 2 consecutive errors → −1 level; target ≈ 50-70 % accuracy); **Overload** (start +3 levels above Flow start, floor at that level, step rule as Katahira). Give accuracy feedback in all conditions (feedback is a flow antecedent; Katahira gave none and still got the effect, but Durcan 2a recommends it). | Katahira 2018; Durcan 2024 |
| Block length | **6 min per block** (between Katahira's 184 s and Rácz's 10 min; Knierim 2018 used 5 min). Longer than 3 min because Durcan notes short sessions (~12 min total) are a known weakness, shorter than 10 min to limit fatigue and keep the session under an hour. Score EEG from minute 1-6 (drop the first 60 s as ramp-in, as Katahira dropped trial 1). | Katahira; Knierim 2018; Rácz; Durcan |
| Repetitions | **2 blocks per condition = 6 task blocks**, plus 3-min EO rest + 1-min EC rest at the start and a 2-min EO rest between every block (washout, as Knierim's 60-s screens). Total ≈ 36 min task + ≈ 20 min rest/questionnaires ≈ 1 h. | Katahira (3/cond); Knierim (1/cond) |
| Order | Counterbalance with a 6-sequence Latin square over the 3 conditions (each condition in each position across subjects); the two blocks of the same condition never adjacent. Randomise which square row each subject gets. | Katahira (2 orders); Rácz (random); Durcan |
| N | **≥ 20 participants** completing all blocks (Ewing's demand effects were detected at N = 20 with η² ≈ 0.4-0.7; Katahira at N = 16; Knierim 2018's N = 7 found nothing). Run a G*Power sensitivity analysis for a repeated-measures ANOVA, 3 levels, α .05, power .8; with r ≈ .5 between levels, N = 20 detects f ≈ 0.28. Report it. | Ewing; Katahira; Durcan |
| Wearable | Wearer on the band throughout; log band time-sync to the Cyton via a shared marker (e.g., a key press that both the EEG stream and the phone app timestamp) at every block start/end. Include the band's raw HR/HRV: Rácz found HR boredom > flow < frustration and Sugino found RR-min tracks flow — these are the peripheral signals most likely to be shared by the wrist score. | Rácz; Sugino |

### 4.2 Self-report after every block

**Primary criterion: Flow Short Scale, 10 core items** (Rheinberg, Vollmeyer & Engeser 2003; English in Engeser & Rheinberg 2008), 7-point "Not at all … Very much". Wording as reproduced in the Finnish validation paper (Sci Rep 2022, PMC9681728):

1. I feel just the right amount of challenge *(challenge-skill antecedent; Finnish validation dropped it — analyse separately, Durcan 7b)*
2. My thoughts/activities run fluidly and smoothly *(fluency)*
3. I do not notice time passing *(absorption; sensitive to block duration)*
4. I have no difficulty concentrating *(fluency)*
5. My mind is completely clear *(fluency)*
6. I am totally absorbed in what I am doing *(absorption)*
7. The right thoughts/movements occur of their own accord *(fluency)*
8. I know what I have to do each step of the way *(fluency)*
9. I feel that I have everything under control *(fluency)*
10. I am completely lost in thought *(absorption)*

Supplementary (perceived importance, 3 items, not in the flow score): 11 Something important to me is at stake here; 12 I must not make any mistakes here; 13 I am worried about failing.

Plus **two manipulation-check items** on 9-point scales (Sugino): "How difficult was the task for you?" from 1 "far too easy" through 5 "just right" to 9 "far too hard"; and "How good were you at it?" 1-9. Blocks where the Flow condition is rated ≤ 3 or ≥ 7 on difficulty, or Boredom > 4, fail the manipulation check and are excluded in a sensitivity analysis (Durcan 7a).

Optional, once at the end: Flow State Scale-2 (Jackson & Eklund 2002; 36 items, 9 dimensions × 4, 5-point) for the whole session — used by Eschmann; too long per block. The Core Flow State Scale (10 items; Rosen 2024) is an alternative if the FSS is unavailable.

Administration: tablet or paper immediately at block end, before the rest period (Katahira: after every block; Knierim 2025: probes every 10-20 min). Keep item order fixed.

### 4.3 Expected ordering and pre-registered hypotheses

| Measure | Expected boredom / flow / overload | Source |
|---|---|---|
| FSS total (items 2-10) | **low / high / medium-low** (flow highest; overload lower than flow but usually above boredom on absorption) | Katahira item 6; Knierim 2018; Rácz |
| Difficulty item | 1-3 / 4-6 / 7-9 | design |
| Accuracy | ≈100 % / 50-70 % / < 25 % | Katahira |
| FlowEEG_1 (frontal theta z) | **low / high / high-or-slightly-lower** (H1: flow > boredom, d ≥ 0.6; H2 exploratory: flow ≥ overload) | Katahira; Ewing; Knierim 2025 |
| Upper alpha, central-parietal (A) | monotonic in one direction; **flow intermediate** (H3: |A_flow − A_mid| < |A_boredom − A_mid| and < |A_overload − A_mid|, by construction weak — test on held-out second blocks only) | Katahira; Ewing |
| FlowEEG_2 | low / high / low | construction |
| Engagement index EI | **low / mid / high** (monotonic) | Pope 1995 semantics; inference |
| Occipital alpha O1/O2 | high in boredom (disengaged gaze) / lower / lower | Ewing (alpha not affected at occipital), thesis snippet; weak |
| HR | boredom > flow < overload (U) | Rácz |
| Wearable flow score | low / high / medium (H4: Spearman ρ with FSS across the 6 blocks ≥ 0.4 on average per subject; H5: ρ with FlowEEG_1 ≥ 0.3) | inference |

### 4.4 Analysis plan

1. **Manipulation check**: repeated-measures ANOVA (or Friedman if non-normal, as Knierim 2018) on accuracy, difficulty item and FSS across the three conditions; Bonferroni/Holm pairwise. Must show FSS flow > boredom and flow > overload before any EEG claim is made.
2. **EEG condition effect**: block-mean FlowEEG_1, A, EI → repeated-measures ANOVA with condition (3) and repetition (2); report partial η² and pairwise Cohen's d_z. Planned contrasts: flow vs boredom (confirmatory), flow vs overload (exploratory). Greenhouse-Geisser correction.
3. **Trial/epoch-level association**: linear mixed model of 10-s FlowEEG_1 epochs on block FSS with random intercept and slope per subject (Knierim 2025); include a quadratic FSS term to test the inverted-U.
4. **Individual-level replication count** (Sugino): per subject, Spearman ρ between the 6 block FSS scores and the 6 block FlowEEG_1 means; report how many subjects show ρ > 0 and how many are significant — small N per subject means this is descriptive.
5. **Wearable validation**: (a) per-subject Spearman between the wearable's block-mean score and FSS; (b) between the wearable score and FlowEEG_1; (c) condition ANOVA on the wearable score identical to step 2; (d) epoch-level cross-correlation between the wearable score and FlowEEG_1 at the wearable's native cadence, with a lag search ±60 s (peripheral signals lag cortical ones). Pre-register the success criterion (e.g., the wearable separates flow from boredom with d_z ≥ 0.5 and ρ(FSS) ≥ 0.4 in ≥ 60 % of subjects).
6. **Robustness**: repeat step 2 with log-absolute instead of relative power, with ±150 µV instead of ±100 µV rejection, and with Fp1/Fp2 added to T; report whether conclusions change.
7. **Report**: epoch rejection rate per block and channel, impedances, number of ICA/ASR components removed, and the Durcan checklist items met/unmet.

---

## 5. Gaps and caveats in this compilation

- Full texts **not** read (abstract only): Alameda 2022 (both publisher and Granada preprint blocked), Berta 2013, Pope 1995, Wolf 2015, Lin 2025, Rosen 2024, Labonté-LeMoyne 2016, Bombeke 2018, Fairclough & Ewing 2017, Soltész 2012, the Creativity Research Journal 2023 paper. The Pope electrode sites (Cz, Pz, P3, P4) and band limits are from secondary reviews; the 1.67-s/20-s window figures are unverified.
- Knierim's 2018 LNISO chapter and any Knierim 2021-2023 flow EEG result could not be retrieved; the 2021 and 2023 Knierim papers found are hardware papers (OpenBCI + cEEGrid; OpenBCI headphones), not flow findings. The "frontal alpha/theta 2023" paper named in the brief was not found under that description.
- No study has used OpenBCI scalp electrodes at the stated 8 sites for flow; the nearest is Knierim 2025 (Cyton+Daisy with around-ear cEEGrids), which found theta (inverted-U) and beta-asymmetry effects but null alpha.
- No wrist-wearable vs EEG flow validation exists in the literature found; Rácz 2025 (prefrontal 4-ch + armband HR/IMU) is the closest.
- Katahira's "frontocentral alpha" in the title is supported in the paper only by correlations at the fronto-central ROI; the condition ANOVA reports "frontal areas" and the clean three-way ordering is at the **right-central** ROI (C4/C6/CP4/CP6). C4 on the Cyton is the right channel to carry that term.

### Sources (primary pages read)

- Katahira 2018: https://www.frontiersin.org/articles/10.3389/fpsyg.2018.00300/full ; https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5855042/fullTextXML
- Ewing 2016: https://www.frontiersin.org/articles/10.3389/fnhum.2016.00223/full ; https://pmc.ncbi.nlm.nih.gov/articles/PMC4870503
- Fairclough & Ewing 2017: https://researchonline.ljmu.ac.uk/id/eprint/5313/
- Pope 1995: https://ntrs.nasa.gov/citations/20050000198 ; https://api.openalex.org/works/doi:10.1016/0301-0511(95)05116-3
- Alameda 2022: https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1016/j.cortex.2022.06.005&resultType=core&format=json
- Berta 2013: https://api.openalex.org/works/doi:10.1109/TCIAIG.2013.2260340
- Knierim 2018: https://www.scitepress.org/PublishedPapers/2018/69267/pdf/index.html
- Knierim 2025: https://arxiv.org/html/2501.19217v1
- Sugino 2024: https://www.biorxiv.org/content/10.1101/2024.07.11.603158v1.full
- Lin 2025: https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:39749104&resultType=core&format=json
- Rácz 2025: https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11977251/fullTextXML
- Rosen 2024: https://researchdiscovery.drexel.edu/esploro/outputs/journalArticle/Creative-flow-as-optimized-processing-Evidence/991021853714104721
- Wolf 2015: https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:25616246&resultType=core&format=json
- Núñez Castellar 2019: https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1111/psyp.13347&resultType=core&format=json
- Labonté-LeMoyne 2016: https://api.openalex.org/works/doi:10.1145/2851581.2892356
- Eschmann 2022: https://pmc.ncbi.nlm.nih.gov/articles/PMC9360146/
- Durcan 2024: https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11332228/fullTextXML
- Flow Short Scale items: https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9681728/fullTextXML
- Bieri 2025: https://pmc.ncbi.nlm.nih.gov/articles/PMC12630671/
- Beiramvand 2025: https://arxiv.org/abs/2609.19737
- Soltész 2012: https://real.mtak.hu/75574/
- Eldaly 2026: https://scholars.cityu.edu.hk/en/publications/neural-symphony-of-flow-experience-evidence-for-high-dimensional-/
