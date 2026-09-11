# Exploratory architecture study on test blocks A + B — findings, 2026-09-09

**Status: EXPLORATORY, NOT PRE-REGISTERED.** Branch `explore-arch-2026` (from
`explore-blockB-2026` @ 07f11fd). Blocks A (2025) and B (2026) are examined data (block A
locked access 2026-08-28; block B consumed exploratorily 2026-09-08). Every number here is
exploratory and must be labelled so wherever it is used. Pre-declaration: RESEARCH_LOG
2026-09-09 09:52 UTC (commit 3cccd35), before the first training run.

**Question.** M2 (persistence-residual transformer, PRT) and B6 (log-space gradient
boosting) are interchangeable at the event grain across 77 fires the models never trained
on. Is that a property of *that* network, or of deep encoders in general under the same
persistence-residual wrapper and log-space representation?

**Design.** Six encoders under the identical wrapper (log-space residual on
log1p(raw persistence), gate α clamped [0, 0.5], init 0.1), representation (T1 pool,
robust normalization from dataset_v6_1 metadata), loss (LogHybridLoss fire_weight 7.0,
α 0.5) and recipe (AdamW 1e-3, wd 1e-4, OneCycle pct 0.3, batch 64, grad-clip 1.0,
patience 5, max 30 / min 2 epochs, checkpoint on val median SS_fire,t6): seeds 0–9,
10-member log-mean ensembles, **no selection step** (all six went to test), one scoring
pass per architecture after all training, via the locked `test_access_v1.py` machinery
imported unchanged. 60/60 runs completed; none failed; nothing was re-run or tuned after
a test score was seen. Identity check: block-A MAE_B6,t6 and MAE_M2,t6 reproduce the
locked access table (max |diff| 5.7e-14; `identity_check.json`).

**Multiple-comparison caveat (§1.6 of the brief).** Six architectures plus M2 were scored
on the same 77 fires. One may look best by chance. A Bonferroni threshold over six looks
is 0.05/6 ≈ 0.008; no pooled sign test reaches it. No winner is headlined; anything that
beats B6 here is a hypothesis for a 2027 pre-registration, not a claim.

## 1. Master table (event-level, MAE_fire,t6; wins = model lower MAE; p = one-sided exact binomial; W = Wilcoxon)

| model | params | block A (45) vs B6 | block B (32) vs B6 | pooled (77) vs B6 | pooled vs B1 | pooled vs M2 | median SS t+6 A / B / pooled |
|---|---|---|---|---|---|---|---|
| A1 GRU-Skip | 19,597 | 28/17 p=0.068 (W 0.17) | 12/20 p=0.945 | 40/37 p=0.410 (W 0.66) | 71/6 | 31/46 | +0.341 / +0.322 / +0.329 |
| A2 LSTM-Skip | 125,581 | 32/13 p=0.003 (W 0.14) | 12/20 p=0.945 | 44/33 p=0.127 (W 0.62) | 72/5 | 34/43 | +0.349 / +0.305 / +0.323 |
| A3 CNN-Skip | 19,341 | 30/15 p=0.018 (W 0.13) | 12/20 p=0.945 | 42/35 p=0.247 (W 0.73) | 72/4 | 36/41 | +0.344 / +0.307 / +0.310 |
| A4 MLP-Mixer | 9,037 | 27/18 p=0.116 (W 0.003) | 20/12 p=0.108 (W 0.010) | 47/30 p=0.034 (W <0.001) | 72/5 | 41/36 p=0.32 (W 0.003) | +0.339 / +0.340 / +0.339 |
| A5 PRT-noLSTM | 116,173 | 29/16 p=0.036 (W 0.009) | 11/21 p=0.975 | 40/37 p=0.410 (W 0.28) | 70/7 | 33/44 | +0.334 / +0.293 / +0.318 |
| A6 PatchTST-Skip | 29,533 | 30/15 p=0.018 (W 0.002) | 15/17 p=0.702 | 45/32 p=0.086 (W 0.018) | 72/5 | 36/41 | +0.338 / +0.320 / +0.336 |
| M2 PRT (reference) | 265,165 ×10 | 28/17 p=0.068 (W 0.08) | 15/17 p=0.702 | 43/34 p=0.181 (W 0.35) | 70/7 | — | +0.349 / +0.302 / +0.327 |
| B6 gbdt_log_T1 | — | — | — | — | 70/7 | 34/43 | +0.342 / +0.318 / +0.325 |
| M1 (2026 deposit, descriptive) | — | 26/19 p=0.186 | 12/20 p=0.945 | 38/39 p=0.590 | 74/3 | 35/42 | +0.335 / +0.307 / +0.314 |

Full table with mean ± std, negatives, minima, horizon medians and AMEND-8 strata:
`master_table.md`; per-event tables `{arch}_per_event.csv` (access-table columns);
statistics `master_results.json`.

Pooled horizon profile (median SS): every deep encoder sits far below B6 at t+1
(+0.02 to +0.13 vs +0.27), reaches parity at t+6 (+0.31 to +0.34 vs +0.33) and t+12
(+0.37 to +0.38 vs +0.37). The block-A/block-B profile of the M2 letter replicates for
all six.

Cross-architecture agreement (per-event ΔSS vs B6, Spearman): 0.13–0.83 between
encoders (A1–A2 0.83, A3–A6 0.76, A4 weakly related to all others). In 14 of 77 events
all seven deep models beat B6; in 6 none does; the count of deep models beating B6 per
event is spread nearly uniformly over 0–7. Median |ΔSS| vs B6 per model is 0.014–0.029:
the models differ from B6 by about one-tenth of the event-to-event spread of SS itself.

## 2. Validation table (12 evaluable val events; selection grain identical to e28)

| id | encoder | params | per-seed median SS range | seed-mean of medians | ensemble median | ensemble mean ± std | val wins vs B6 | best epochs | gate α (per seed) |
|---|---|---|---|---|---|---|---|---|---|
| A1 | GRU-Skip | 19,597 | +0.288 … +0.302 | +0.2942 | +0.2963 | +0.2948 ± 0.1475 | 6/12 | 5–19 | 0.33–0.50, 8 of 10 at the clamp |
| A2 | LSTM-Skip | 125,581 | +0.293 … +0.319 | +0.3016 | +0.2986 | +0.2937 ± 0.1550 | 6/12 | 6–23 | 0.31–0.50, 6 at the clamp |
| A3 | CNN-Skip | 19,341 | +0.296 … +0.299 | +0.2979 | +0.2982 | +0.2528 ± 0.1586 | 2/12 | 6–10 | 0.21–0.25 |
| A4 | MLP-Mixer | 9,037 | +0.291 … +0.333 | +0.3078 | +0.2997 | +0.3169 ± 0.1499 | 6/12 | 8–25 | 0.45–0.50, 9 at the clamp |
| A5 | PRT-noLSTM | 116,173 | +0.301 … +0.350 | +0.3247 | +0.3069 | +0.3256 ± 0.0998 | 9/12 | 5–30 | 0.31–0.50 |
| A6 | PatchTST-Skip | 29,533 | +0.277 … +0.293 | +0.2867 | +0.2887 | +0.2735 ± 0.1577 | 5/12 | 3–8 | 0.14–0.23 |
| M2 | PRT (e28 T1_robust) | 265,165 | +0.303 … +0.345 | +0.3193 (grid row s0–2: +0.3348) | +0.3131 | +0.3200 ± 0.1291 | 10/12 | 7–12 | 0.35–0.43 |
| B6 | gbdt_log_T1 | — | — | — | +0.3050 | +0.2913 ± 0.1273 | — | — | — |

Full per-seed values and per-event ensemble SS: `val_table.md`. Reading: all six
encoders land in the same validation band as the PRT members (+0.28 to +0.35); A6 stops
at epoch 3 in 7 of 10 seeds (within the expected 1–6 window) and holds the lowest gate.
**Validation does not rank the encoders on test:** A5 has the best val seed-mean of the
six and the weakest pooled test (40/37; block B 11/21); A6 has the lowest val seed-mean
and is among the best on test (45/32). With 12 val events, nothing here would have
justified a selection step, and none was made.

## 3. Post-peak-day diagnostic (77 events; `decisive_day_by_arch.md`, `decisive_day_{model}.csv`)

| model | pooled wins vs B6 | flips without the post-peak day | median post-peak share of AE | events with post-peak bias > 0 | sign test excl. post-peak day | sign test post-peak day alone |
|---|---|---|---|---|---|---|
| M2 | 43/34 | 9 | 0.11 | 15 | 44/33 p=0.127 | 27/32 p=0.78 |
| A1 | 40/37 | 9 | 0.11 | 19 | 45/32 p=0.086 | 27/32 p=0.78 |
| A2 | 44/33 | 13 | 0.12 | 19 | 43/34 p=0.181 | 25/34 p=0.90 |
| A3 | 42/35 | 11 | 0.10 | 6 | 43/34 p=0.181 | 31/28 p=0.40 |
| A4 | 47/30 | 9 | 0.11 | 20 | 48/29 p=0.020 | 29/30 p=0.60 |
| A5 | 40/37 | 11 | 0.11 | 19 | 43/34 p=0.181 | 23/36 p=0.97 |
| A6 | 45/32 | 9 | 0.10 | 10 | 46/31 p=0.055 | 33/26 p=0.22 |
| M1 | 38/39 | 10 | 0.10 | 22 | 36/41 p=0.75 | 35/25 p=0.12 |

The worst day of every model is the post-peak day in 12 of 77 events, identically across
encoders; the post-peak day flips the paired outcome vs B6 in 9–13 of 77 events for every
model; on that day alone no deep model beats B6 (best A6 33/26). Mean signed error on the
post-peak day is negative (under-forecast of the decline, −1,400 to −2,500 MW) for every
model including B6 (−2,339 / −1,739 MW on A / B).

**The three 2026 collapse events** (M2 trails B6 by > 0.3 skill), SS_fire,t6:

| event | B6 | M2 | A5 PRT-noLSTM | A1 | A2 | A3 | A4 | A6 | M1 |
|---|---|---|---|---|---|---|---|---|---|
| pt_cambra_e_carvalhal_2026_0702 | +0.501 | −0.050 | −0.060 | +0.372 | +0.361 | +0.461 | +0.455 | +0.517 | +0.453 |
| fr_porge_2026_0722 | +0.414 | −0.192 | −0.092 | +0.328 | +0.264 | +0.310 | +0.070 | +0.383 | +0.312 |
| pt_cerdeira_2026_0727 | +0.109 | −0.332 | −0.065 | −0.018 | +0.022 | +0.158 | −0.170 | +0.147 | +0.111 |

The collapses are reproduced by exactly one other encoder: A5, the PRT with its LSTM
removed. GRU, LSTM, CNN, MLP and PatchTST encoders under the identical wrapper score
+0.26 to +0.52 on cambra and porge. The failure therefore belongs to the PRT's
attention-over-projected-sequence + last-step readout block, not to the LSTM, not to the
wrapper, and not to the representation.

## 4. Reading (five lines, under the multiple-comparison caveat)

1. **No encoder separates from B6 at the event grain on 77 fires.** Pooled wins vs B6
   range from 40/37 to 47/30. The best pooled result (A4 MLP-Mixer, 9k parameters,
   47/30, p = 0.034 one-sided; Wilcoxon < 0.001) does not survive a six-way correction and
   its block halves are each p ≈ 0.11; A2's block-A 32/13 (p = 0.003) reverses to 12/20
   in block B. Seven looks at the same 77 events under a null of equivalence are expected
   to produce results of exactly this shape.
2. **The skill level is set by the wrapper and the representation, not the encoder.**
   Every encoder from 9k to 126k parameters beats persistence in ≥ 70 of 77 events,
   reproduces the horizon profile (B6 far ahead at t+1, parity from t+6), and lands within
   0.03 median skill of B6 and of each other. This is the block-A/B representation finding
   a third time, now across architectures, and it is consistent with FIRA's choice of a
   GBDT for hourly FRP intensity (Hung et al., GeoHealth 2025, 10.1029/2024GH001253).
3. **The 2026 collapses are PRT-specific, not wrapper-specific.** Only M2 and A5
   (PRT-noLSTM) fail on cambra / porge / cerdeira; the other five encoders do not. The
   shared component of M2 and A5 is the self-attention block over the projected sequence
   with a last-step readout; removing the LSTM does not remove the failure.
4. **The general post-peak-day weakness is shared by the whole family.** On the post-peak
   day alone none of the seven deep models beats B6; away from it, A4 (48/29, p = 0.020)
   and A6 (46/31, p = 0.055) are the strongest and M2 is 44/33. A "transition-aware"
   follow-up (residual gated on trajectory phase, or a GBDT/deep hybrid) is justified for
   the model class, not for one encoder; it is also where the only real room over B6 lies.
5. **Hypotheses for a 2027 pre-registration, not claims:** A4 MLP-Mixer and A6
   PatchTST-Skip as candidate primary models against B6 (both cheaper than the PRT by
   9–29×; A6 also holds the lowest gate α ≈ 0.14 and the most B6-like post-peak
   behaviour); dropping the pure attention readout (A5-type) from consideration. Any such
   pre-registration must state that these candidates were chosen after seeing blocks A + B.

Validation cannot arbitrate between them: the val ordering (A5 > A4 > A2 > A3 > A1 > A6)
is unrelated to the test ordering, which is itself within noise.

## 5. What this changes for the letter

Nothing in the block-A pre-registered result or the block-B exploratory disclosure. The
study strengthens the representation finding (three independent replications: 2025 fires,
2026 fires, six encoders) and localizes the 2026 collapse mechanism to the PRT's attention
readout. If cited in the letter, one sentence with the exploratory label; the material
belongs to the follow-up (transition-aware model, 2027 pre-registration).

## 6. Files (all exploratory; hashes in RESEARCH_LOG 2026-09-09)

| file | content |
|---|---|
| `master_table.md`, `master_results.json` | one row per model × {A 45, B 32, pooled 77}; horizon profile; AMEND-8 strata |
| `{A1..A6}_per_event.csv`, `reference_models_per_event.csv` | per-event tables, access-table columns |
| `val_table.md` | per-seed medians, ensembles, per-event val SS vs M2 and B6 |
| `identity_check.json` | block-A MAE identity vs the locked access table (PASSED, 5.7e-14) |
| `decisive_day_by_arch.md`, `decisive_day_{model}.csv` | post-peak-day diagnostic per predictor |
| `data/models/explore_arch/{arch}_s{0..9}_best.pt`, `_result.json`, `_val_preds.npy` | 60 runs (git-ignored) |
| `data/models/explore_arch/{arch}_ensemble_val_results.json`, `{arch}_freeze_hashes.json` | ensembles and freeze tables |
| `data/models/explore_arch/eval_cache/` | 17,428 windows, reference and per-architecture window-level predictions |
| scripts | `explore_arch_train.py`, `explore_arch_ensemble.py`, `explore_arch_eval.py`, `explore_arch_decisive.py` |

Not modified: `scripts/test_access_v1.py` (6b403afc…b1c4), `dataset_v6_1.h5`, its
manifest, `data/models/e28/*`, `TEST_ACCESS_RESULTS/`, `EXPLORATORY_BLOCKB_2026/`.
`scripts/verify_register.py`: PASS (86/86 values, 5/5 hashes) after the study.
