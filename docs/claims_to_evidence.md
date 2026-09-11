# Claims to evidence

Every quantitative claim in the letter, the file that carries it, and the field to read.
Column names in the deposited tables use the internal model labels, because the files are
published exactly as they were written and their hashes are part of the pre-registration:

| file label | letter name |
|---|---|
| `M2` | the PRT — the pre-registered 10-seed persistence-residual Transformer ensemble |
| `B6` | boosting — log-space gradient boosting, the pre-registered comparator |
| `B1` | persistence — flat persistence, the skill-score reference |
| `M1` | the archived ensemble of the earlier version of the study (descriptive only) |
| `A1 … A6` | GRU, LSTM, dilated CNN, MLP-Mixer, PRT-without-LSTM, PatchTST encoders |

Skill is `SS = 1 − MAE_model / MAE_persistence` at lead *t*+6 over fire-active windows, with
MAE in MW on the raw scale after inverse transform. An event is evaluable with at least five
fire-active windows.

---

## The pre-registered test (2025 block, single access, 28 August 2026)

Source: `results/access_2025/access_20260828T123059Z_results.json` and `..._per_event.csv`.

| Claim | Field | Value |
|---|---|---|
| 46 fires pass the quality gate; 45 are evaluable; 29 of the 30 primary and 31 of the 32 sensitivity events are evaluable | `counts` | `{gate_passing: 46, evaluable: 45, primary_evaluable: 29, sensitivity_evaluable: 31, N_min: 5}` |
| **Primary hypothesis is not rejected.** PRT lower MAE than boosting on 19 of 29, one short of the pre-declared region of ≥ 20 | `PRIMARY_sign_test_M2_vs_B6_primary_set` | wins 19, losses 10, *n* 29, *p* = 0.0680, `reject_at_005: false`; Wilcoxon 0.1846 |
| PRT beats persistence on 27 of 29 primary events | `SECONDARY_sign_test_M2_vs_B1_primary_set` | wins 27, losses 2, *p* = 8.12 × 10⁻⁷ |
| The sensitivity set gives 21 of 31 and carries no confirmatory weight | `SECONDARY_sign_test_M2_vs_B6_sensitivity_set` | wins 21, losses 10, *p* = 0.0354; Wilcoxon 0.0787 |
| Median skill over the 45 evaluable: PRT +0.349, boosting +0.342 | `AGG1_headline` | `M2_median_SS_t6` 0.34886, `B6_median_SS_t6` 0.34159; PRT mean +0.2874 ± 0.1664 |
| The archived earlier ensemble reaches +0.335 | `M1_DESCRIPTIVE` | `M1_median_SS_t6` 0.33447 |
| 11,926 forecast windows | `access_console_20260828.log` | `events=46 windows=11926` |

Per-event wins against persistence (41 of 45 for the PRT, 40 of 45 for boosting) and the count
of events with negative skill (4 and 5) are computed from `..._per_event.csv`; the stratified
results by lead, wind regime and volatility tertile are in `..._strata.json`.

## The 2026 block — exploratory

Examined on 8 September 2026, after the 2025 result was known. Not a confirmatory replication.
Source: `results/exploratory_2026/exploratory_blockB_20260908T140106Z_results.json`.

| Claim | Field | Value |
|---|---|---|
| All 32 fires are evaluable | `counts` | `{gate_passing: 32, evaluable: 32}` |
| PRT lower MAE than boosting on 15 of 32 — the 2025 direction does not repeat | `PRIMARY_sign_test_M2_vs_B6_primary_set` | wins 15, losses 17, *p* = 0.702 |
| Median skill +0.302 for the PRT against +0.318 for boosting | `AGG1_headline` | 0.30216 and 0.31791 |
| PRT beats persistence on 29 of 32 | `SECONDARY_sign_test_M2_vs_B1_primary_set` | wins 29, losses 3, *p* = 1.28 × 10⁻⁶ |

The three fires where the PRT is negative and boosting positive are identified in
`FINDINGS_blockB_2026-09-08.md` and labelled in Figure 3 of the letter.

## The encoder comparison — exploratory

Six encoders under the identical framework, representation, loss and training protocol,
pre-declared and scored once on the 77 evaluable fires of both blocks.
Source: `results/encoder_study/master_results.json`, field `stats.<model>.pooled`.

| Model | vs boosting (wins/losses, *p*) | vs persistence |
|---|---|---|
| PRT (`M2`) | 43 / 34, *p* = 0.181 | 70 / 7 |
| GRU (`A1`) | 40 / 37, *p* = 0.410 | 71 / 6 |
| LSTM (`A2`) | 44 / 33, *p* = 0.127 | 72 / 5 |
| dilated CNN (`A3`) | 42 / 35, *p* = 0.247 | 72 / 4 |
| MLP-Mixer (`A4`) | 47 / 30, *p* = 0.0338 | 72 / 5 |
| PRT without LSTM (`A5`) | 40 / 37, *p* = 0.410 | 70 / 7 |
| PatchTST (`A6`) | 45 / 32, *p* = 0.0856 | 72 / 5 |
| earlier ensemble (`M1`) | 38 / 39, *p* = 0.590 | 74 / 3 |

Six comparisons were made, so the Bonferroni threshold is 0.008. The MLP-Mixer's uncorrected
*p* = 0.034 does not survive it; its Wilcoxon companion (*p* = 1.5 × 10⁻⁴) does, and the letter
reports that without promoting it, because the pre-declared statistic is the sign test and its
per-block halves are *p* = 0.12 and 0.11. No encoder separates from boosting after correction.

Median skill by lead for every model is in the same file under `pooled.horizon_medians`; those
values are what Figure 3(b) draws.

**Consistency of the re-scoring.** `results/encoder_study/identity_check.json` reports
`max_abs_diff = 5.68 × 10⁻¹⁴` against the locked 2025 table over 45 events, tolerance 1 × 10⁻³.
The later runs reproduce the confirmatory numbers; they did not drift.

## Catalogue, sequestration and independence

Source: `catalogue/`.

| Claim | Where |
|---|---|
| 162 quality-screened events: 84 development (2007–2024), 46 in 2025, 32 in 2026 | `event_catalogue_162_export_2026-09-11.csv`, column `pool`; all five gate criteria are recomputed per event and all 162 pass |
| Sequestration is by ignition year alone | same file, `ignition_year` against `pool` |
| The development pool splits 71 train / 13 validation by a seeded draw | `split_v6_2026-08-25.csv`, `frozen_dev_pool_2026-08-25.csv` |
| 14 of the 2025 fires overlap a development-pool event and are removed from the primary statistic, remaining in the descriptive set | `pixel_disjointness_test_A_2025_2026-08-25.csv` and `_drops.json`; `set_membership` in the catalogue export |
| The two test blocks span 15 countries between them | `country` in the catalogue export, restricted to the two test pools |
| The development pool was curated, the test blocks were not | `dev_curation_list_2026-08-25.csv`, `dev_curation_selected_2026-08-25.csv` (the scoring rule and the 36 selected candidates), `EXCLUSIONS_E2_2026-08-25.md` |

The overlap rule itself: for two events with intersecting bounding boxes and overlapping
windows, `overlap_fraction = |P_A ∩ P_B| / min(|P_A|, |P_B|)` over fire-pixel positions rounded
to 0.05° inside the box intersection during the shared window; at or above 0.10 the pair is
treated as dependent and the event with fewer catalogue fire slots leaves the primary set.
Coordinates only — no FRP value enters the audit. It is stated in
`registration/prereg_test_blocks_v1_LOCKED.md` §9.5 and was executed before the access.

## Model and comparator specification

| Item | Where |
|---|---|
| PRT architecture, loss, optimizer, schedule, stopping rule | `scripts/e28_train_grid.py` |
| The validation-only selection and its three pre-declared rules | `scripts/e29_selection.py`, `models/prt_ensemble_m2/e29_selection.json` |
| The 10 ensemble members and their freeze hashes | `models/prt_ensemble_m2/`, `m2_freeze_hashes.json` |
| Comparator definition, retrained deterministically at scoring time | `scripts/test_access_v1.py`, `B6_CFG` and `b6_train_predict` |
| Comparator configuration search (four configurations, validation only) | `results/number_register.json`; the selection is recorded in the research log entry of 26 August 2026 |

The comparator is `HistGradientBoostingRegressor` on the 32 channels at the last input step,
target `log1p(FRP)`, inverse `expm1` clipped at zero, one model per lead;
`max_leaf_nodes = 63`, `learning_rate = 0.1`, `max_iter = 300`, early stopping on a 10 percent
internal split, `random_state = 0`. It was chosen from four configurations by validation median
skill and frozen after the PRT ensemble was frozen — built to beat the candidate, not the
reverse.
