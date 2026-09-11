# Pre-Registration — Sequestered Test Blocks for FRP Trajectory Forecasting
# STATUS: *** LOCKED (self-certified under G6′) — publicly registered before any
# test access; DOI, tag, and commit in §8 and the package registration record ***
# [Status text set per IA-1 Step 1; the lock takes force at the Step-4 public
# registration, whose identifiers are entered in the registration record.]

**Draft created:** 2026-08-24 (Task E1.5)
**Split rule declaration date:** 2026-08-24 — before any 2025/2026 gate results were known.
**Lock procedure (revised per IA-1/G6′):** this file is frozen, SHA-256-hashed,
and publicly registered (Zenodo deposit with DOI; annotated git tag + release on
the public frp-prt-grsl repository) with the public timestamp PRECEDING all
test-data access. Self-certified under G6′ (supervisor participation declined,
logged 2026-08-28); no independent human sign-off exists and none is claimed.
Two further deposits are REQUIRED by IA-1: the season-close deposit (block-B
composition + access declaration; access is forbidden before it is public) and
the post-access deposit (raw outputs within 72 hours, outcome-blind).

---

## 1. Purpose

GRSL-01633-2026 was rejected because its 4-event test set was accessed three times and
influenced model and dataset selection, and because bootstrap inference on 4 all-positive
events is tautological. This pre-registration defines a single-access evaluation on
sequestered test blocks. The temporal guarantees, stated precisely (IA-B3 —
four defensible claims, no more):
(i) no test-block FRP content, prediction, or forecast-derived statistic was
ever computed before the single access (sequestration, logged, verified);
(ii) block membership follows the ignition-year rule declared 2026-08-24,
before any 2025/2026 gate results existed;
(iii) no block-A or block-B event was ever onboarded or accessed by the
original project — M1's training pool contains only ≤ 2024 events, so M1
never saw any test event;
(iv) the subset of block B igniting after 2026-05-19 (M1 deposit) and after
2026-08-26T09:39:40Z (M2 freeze) additionally could not have existed at those
locks; those per-event counts are reported in the access outputs.
Results will be reported regardless of outcome (**publish-regardless commitment**);
a negative or nuanced outcome pivots the manuscript to a full-length venue, it does not
suppress the result.

## 2. Split rule (declared 2026-08-24)

| Block | Ignition year | Role |
|---|---|---|
| Development pool | ≤ 2024 | training, validation, all selection decisions |
| Test block A | 2025 | sequestered; never onboarded before this program |
| Test block B | 2026 | sequestered; membership by ignition year; the post-2026-05-19 / post-2026-08-26 igniting subset additionally postdates the model locks (counts reported at access) |

- Block membership follows ignition year ALONE. No observable (peak FRP, regime,
  intensity, volatility) may influence block assignment.
- No event appears in more than one block.
- Former test events (volos_2023, varnavas_2024, valmaior_2024, corinthos_2024) are
  RETIRED to the development pool: they were accessed three times during v6/v7/v8
  development and can never again serve as test. This migration is disclosed in the
  manuscript.
- Dev-pool internal split: deterministic seeded 85/15 train/val over the alphabetically
  sorted gate-passing event list, weighted toward recent years for val
  (forward-transfer proxy); procedure implemented in `scripts/e25_seeded_split.py`
  (SHA-256 `591deb027968187eee17081e080e3605caa23f52ba7f30fde235b0576b83ab73`).
  Val adequacy check (≥ 3 fire-rich events) with declared remediation: redraw at
  seed+1, logged. **Outcome (executed 2026-08-25, recorded per the staleness
  sweep):** draw 1 at the declared seed 20260824 FAILED adequacy check A4; the
  declared seed+1 remediation fired; **FINAL split at seed 20260825: 71 train /
  13 val** (`resubmission/split_v6_2026-08-25.csv`; ba_trebinje_2022 pinned
  train per D2). The split is FINAL and is not redrawn under any outcome.

## 3. Event admission (identical for all blocks)

1. **Candidacy (mechanical):** EFFIS burnt-area episode within the Mediterranean basin
   (MSG 0° disk), merged per the declared episode rule*, total area ≥ 3,000 ha
   (final text; D3, resolved 2026-08-28 under DEC-4 self-certification per
   IA-1/IA-B5), ignition
   year in block range, no spatial+temporal overlap with an already-catalogued event.
   *Episode rule: polygons with intersecting 0.25°-expanded bboxes and ignition dates
   within 10 days, same country, are one episode (union-find). Mega-episode
   sub-division (D4, declared 2026-08-25): any merged episode whose snapped bbox
   area ≥ 5.5 sq° is re-clustered among its members using STRICT (unexpanded)
   bbox intersection, same window/country, applied once; threshold sits between
   the largest legitimately-single onboarded episode (4.8 sq°) and the smallest
   over-merged complex (6.75 sq°), and touches no already-onboarded event.
   Conflict refinement (D7, declared 2026-08-25 before any affected event's FRP
   was seen): a catalogue conflict requires POSITIVE-AREA bbox intersection plus
   window overlap; zero-area edge/corner contact of snapped bboxes contains no
   pixels and is not a conflict.

   **Block-B candidacy cutoff and season close (IA-B2, pre-declared
   2026-08-28):** EFFIS episodes with earliest member FIREDATE ≤ 2026-10-31.
   The final EFFIS refresh and season-close procedure execute no earlier than
   2026-11-21 (≥ 21-day data-latency margin, per the right-censoring precedent
   of 2026-08-24). W-1 window extension may run to latest member FIREDATE
   + 9 d past the cutoff under the declared rule. Any 2026 ignition with
   earliest FIREDATE after 2026-10-31 is out-of-candidacy by this
   pre-declared rule; the count of such events is disclosed at access.
   Season close is thereby a date, not a decision.
2. **Geometry (mechanical):** bbox = episode envelope + 0.25° buffer snapped outward to
   the 0.25° ERA5 grid; window: see the W-1 rule below.

   **Window rule (W-1, corrected 2026-08-25 — disclosure per the D7 pattern):**
   the ORIGINAL rule (window = earliest FIREDATE − 1 d … earliest + 9 d) truncated
   temporally-chained episodes: a metadata-only audit (AMEND-3, EFFIS ignition
   dates vs downloaded windows; `b1_window_truncation_audit_2026-08-25*.csv`)
   found 10 events with member polygons igniting after the window (worst:
   pt_reriz_e_gafanhao_2024, ~99% of episode area outside), and 66 events in total
   whose last-igniter decay tail was clipped. CORRECTED RULE, uniform across all
   pools: **window = earliest member FIREDATE − 1 d … latest member FIREDATE
   + 9 d, hard cap 30 days total span.** Chains exceeding the cap are temporal
   over-merges (analogue of the D4 spatial rule) and KEEP the original
   earliest+9 d window with per-event disclosure (P-1): me_n_a_2017_0707 (64 d),
   mk_novaci_2024_0809 (33 d), dz_n_a_2026_0712 (38 d). Applied 2026-08-25 to all
   41 affected extendable dev + test-A events (13 + 28; every affected dev event
   is a TRAIN event — the validation set is untouched); the 17 affected block-B
   events extend at season close. **Full population accounting (66 affected
   events, per `w1_extension_list_2026-08-25.csv`): 41 extended now (28 test-A
   + 13 dev) + 3 P-1 over-cap chains keeping original windows (2 dev + 1
   block-B, dz_n_a_2026_0712) + 17 block-B deferred to season close + 5
   affected dev CANDIDATES that had already failed the frozen gate and are not
   in the frozen pool (no extension warranted) = 66.** The extension decision
   uses EFFIS metadata and
   the declared rule only — no FRP content. Extended events re-pass the frozen
   gate (byte-identical, hashed); any gate change under the uniform rule is
   logged, never remediated; the FINAL split is not redrawn under any outcome.
3. **Quality gate (frozen, five published criteria):** ≥ 15 fire-active hourly slots;
   ≥ 5-day duration (distinct calendar dates); ≥ 3 diurnally complete days; gap rate
   < 40%; MIR saturation < 30%. Canonical implementation:
   `scripts/frp_quality_gate.py`, SHA-256
   `e02db840cf77031d4ba32a33ed712cd126fc15d1d4d75fcaa94222135d725115`, regression-
   verified 2026-08-24 to reproduce the historical pass/fail record (54/54 catalogued
   PASS; 5/5 excluded-with-data FAIL on their documented criteria).
   **No intensity floor** (empirically rejected: `FINDINGS_2026-08-24.md` §4).
4. **Single-sensor rule:** LSA SAF SEVIRI FRP-PIXEL only (verified continuous through
   ≥ 2026-08-15, SATELLITE=MSG3, algorithm v2.4.0; `seviri_continuity_probe_2026-08-24.json`).
5. Test blocks are **exhaustive through the gate** — every candidate meeting 1–4 enters;
   zero discretionary selection. Development-pool curation remains free and disclosed.

## 4. Sequestration and single access

For test-block events, until the locked evaluation run: download, hourly build, ERA5
retrieval, gate metrics, and catalogue metadata ONLY. No persistence MAE, no baseline
or model predictions, no skill statistic, no diagnostic plots of forecast quality.
The evaluation runs ONCE per test block, executed by a pre-registered script (hash
recorded at lock) after all models and baselines are frozen and hashed. Any artifact
violating this section invalidates the corresponding block.

**Technical-failure protocol (pre-declared per the 2026-08-28 adversarial
review, B-5):** a run aborted by technical failure before the results files
are written does not constitute a completed access. The failure is logged
verbatim (full trace); no result-bearing content beyond the error trace may be
inspected; partial artifacts are preserved read-only; any script defect is
fixed, the script re-hashed with the diff disclosed (T-4 pattern), and the run
re-executed. **A run that writes its results files is THE access, regardless
of content.** Operational rules: named operator D. Valsamis; outputs are
written once to `resubmission/TEST_ACCESS_RESULTS/` and hashed + logged in
RESEARCH_LOG the same day; a publicly timestamped access declaration (the
season-close deposit, IA-1 Step 6) precedes execution.
[Supervisor-notification line superseded 2026-08-28 per IA-B4/G6′.]

## 5. Evaluation suite (frozen before access)

Models (A3):
- **M1 — RE-SCOPED (RAT-9, memo A6, 2026-08-27) to the Zenodo-archived
  ensemble of the rejected submission.** History, fully disclosed: (i) the
  prereg originally named M1 = the C-FROZEN and C-HONEST 10-seed ensembles
  from the 2026-05-19 lock; (ii) AMEND-5 verification (2026-08-27) found
  those checkpoints exist neither locally nor on Zenodo, while the deposit's
  ensemble is the LATER stage-2 "Row-2/PRT-Full" model → M1 was CUT (silently
  swapping model identity is forbidden); (iii) memo A6/RAT-9 then explicitly
  authorized re-scoping M1 to that Zenodo-archived model UNDER ITS TRUE
  IDENTITY, conditional on the complete inference pipeline reconstructing
  from the deposit alone with zero judgment calls. The condition was MET:
  deposit 10.5281/zenodo.20627737 (zip md5 771491878286355cd9b15d418aa195cb,
  vendored at resubmission/m1_deposit/) contains the 10 checkpoints
  (SHA-256 table frozen in the access script), normalization_params_v3.json,
  windfeats stats (also checkpoint-embedded), the canonical 32-feature order
  in its build_dataset.py (asserted at runtime), the PRT model class, and
  the log-mean ensembling rule in its eval_test.py. **Every M1 output is
  DESCRIPTIVE; no statistical test involves M1.** Manuscript note (per
  RAT-9.3, phrasing corrected per IA-B3): M1's dev-era training pool contains
  only ≤ 2024 events, so M1 never saw any test-block event; no block-A or
  block-B event was ever onboarded or accessed by the original project. (The
  earlier "postdate its lock by construction" phrasing was imprecise — 2025
  ignitions precede the 2026-05-19 deposit in calendar time; the operative
  guarantees are §1(i)–(iv).)
- **M2 — retrained PRT (FROZEN 2026-08-26): the T1×N-robust cell selected by the
  §5a single-shot comparison** (sole R-1' pairwise dominator; R-2' guard not
  fired; full trace in `e29_selection.json`). 10-seed deep ensemble, seeds
  [0..9], log-mean combination rule; PRT-Full paper hyperparameters, unchanged.
  Val (AGG-1 median SS_fire_t6): **+0.3131** (mean +0.3200 ± 0.1291, 12
  evaluable events, ALL positive, min +0.051); paired wins vs the FINAL B6
  (gbdt_log_T1, named post-B-2): **9/12** (vs the previously named B6
  ar_ridge_T0: 10/12).
  Complete deployed-configuration SHA-256 table:
  `data/models/e28/m2_freeze_hashes.json` (10 checkpoints + dataset_v6_1 +
  normalization params + manifest + training/selection/ensemble scripts +
  results), frozen 2026-08-26T09:39:40Z, entered in the §8 register at lock.

Baselines (suite evaluated 2026-08-25 on the v6.1 validation split, 12 evaluable
events, AGG-1 median SS_fire_t6 vs B1):
- B1 flat persistence (reference, SS ≡ 0)
- B2 diurnal persistence (lag-24): median +0.009
- B2b persistence × diurnal-shape factor (train-climatological shape): median −0.022
- B6(raw) AR Ridge — per training pool: T0 pool +0.1703 / T1 pool +0.024
- B7 GBDT raw target (histogram GBDT, sklearn HistGradientBoostingRegressor;
  lightgbm unavailable — disclosed; declared 4-config budget, selected
  max_leaf_nodes=31 / lr=0.1): T1 pool +0.024; T0 pool (same config, pool swap
  only, no new tuning) −0.010.

**ORDER B-2 log-space baseline extension (memo A4, executed 2026-08-26 AFTER
the M2 freeze — ordering disclosed):** the AR-T1 collapse was diagnosed as
heavy-tail squared-loss domination, and the winning §5a cell fixes exactly
that via log-space representation, yet the original baseline search offered
no log-space variant. Four variants were declared (RESEARCH_LOG 2026-08-26)
and run on the identical protocol (val only, AGG-1 median, gated
denominators, same RidgeCV settings / same declared GBDT 4-config budget):
- ar_ridge_log_T0 +0.2761 / ar_ridge_log_T1 +0.2767 (log1p lags → log1p
  target, expm1 inverse)
- gbdt_log_T0 +0.2945 / **gbdt_log_T1 +0.3050** (log1p target, full
  32-feature final-step input; selected config max_leaf_nodes=63 / lr=0.1)
Conservativity: M2 was frozen and hashed (2026-08-26T09:39:40Z) BEFORE B-2
was ordered or run — extending the baseline search post-freeze can only
raise or leave unchanged the bar M2 must clear at test.

**STRONGEST PRE-REGISTERED TEST BASELINE — FINAL (B-2 renaming rule applied
2026-08-26): B6 = gbdt_log_T1** — histogram GBDT (sklearn
HistGradientBoostingRegressor, max_leaf_nodes=63, learning_rate=0.1,
max_iter=300, early stopping 10%, random_state 0), log1p target with expm1
inverse, full 32-feature final-step input, trained per-horizon on the FULL
v6.1 train pool (71 events). Basis: highest val median SS_fire_t6 (+0.3050)
across the entire raw+log search, exceeding the previously named B6
(ar_ridge_T0, +0.1703; DEC-5 naming superseded per the pre-declared B-2
renaming rule; basis logged, RESEARCH_LOG 2026-08-26). Artifacts hashed:
scripts/b2_log_baselines.py `3ac578da…0551`, results JSON `27c1f967…76dd`.
Historical note (retained): the raw-target T1 AR Ridge degrades to +0.024 —
the W-1-recovered extreme-tail samples (150 GW) dominate its squared loss;
the log-space variants eliminate this pathology in the baselines exactly as
the robust cell eliminated it in the model — documented as the closing of
the heavy-tail loop.

Normalization and all fitting statistics: training split only.

### 5a. M2 selection procedure (Task E2 — val-only, pre-declared; DRAFT for sign-off)

Training-set variants declared BEFORE any comparison runs: T0 = current dev pool
(dataset_v5-equivalent under the new split), T1 = curated expanded pool (composition
targets, FINDINGS §6). Any additional variant requires a logged amendment BEFORE its
training starts. Each variant: identical architecture and hyperparameters (PRT-Full
paper config), 3 seeds; comparison on the seed-mean. All metrics computed on the
FIXED seeded validation set, drawn once after the dev pool is final.

**Selection metrics (top 3, in precedence order):**
- **M-1 (primary): event-level MEDIAN SS_fire_t6** across val events [AGG-1,
  2026-08-25: median replaces the mean for M-1 and every descriptive headline, on
  val and test alike; mean ± std demoted to a documented secondary table — the
  13-event val contains mechanically-drawn flat events where a ~21 MW persistence
  denominator turns a ~330 MW error into SS ≈ −16, making the mean a
  denominator-pathology statistic. The paired sign test (primary inference) is
  unaffected. Immutable once the first §5a run starts].
- **M-2: paired win fraction vs AR Ridge** — fraction of val events where the
  variant's MAE_fire_t6 < AR Ridge's (AR Ridge retrained per variant on the same
  training data). Mirrors the PRIMARY test statistic (paired sign direction);
  scale-robust (bejaia lesson).
- **M-3: worst-case robustness — minimum per-event SS_fire_t6** across val events.
  Guards against regime-specific collapse hidden by means (alexandroupolis lesson).

Recorded but non-decisive diagnostics: SS at t+1/t+3/t+12; MAE_all; per-event table;
inter-seed spread.

**Decision rules (top 3):**
- **R-1 (adoption): T1 is adopted iff it wins on ≥ 2 of the 3 metrics** (seed-mean
  comparison). Otherwise T0 is retained.
- **R-2 (regression guard):** regardless of R-1, T1 is REJECTED if its M-3 falls more
  than 0.05 SS below T0's — a mean improvement may not buy a materially worse
  worst-case event.
- **R-3 (single-shot comparison):** the comparison executes ONCE over the declared
  variant set. No new variants, hyperparameter edits, or metric changes after
  results are seen; any follow-up experiment requires a logged amendment and is
  reported as exploratory. Final M2 = 10-seed deep ensemble (log-mean rule) of the
  winning variant, retrained, then locked and hashed.

**AMENDMENT (AMEND-2, logged 2026-08-25, declared BEFORE any variant training):**
the variant space is the 2×2 grid **{T0, T1} × {N-legacy, N-robust}**, where
N-legacy = the v3-convention z-score normalization and N-robust = log1p-transform of
FRP-scale features followed by median/IQR scaling (exact definition fixed in the
dataset_v6 build code before training). Rationale: the v3→v4→v5 expansion degradation
is plausibly a heavy-tail z-standardization artifact (v5 frp_lag_1h std 6,186 MW vs
v3 4,397 MW); the scheme must be selectable on val, not revised after training.
Decision rule restated for 4 cells (replaces R-1's two-variant phrasing; R-2/R-3
unchanged in intent):
- **R-1':** winner = the cell that wins ≥ 2 of the 3 metrics against EVERY other cell
  pairwise; if no cell dominates, precedence M-1 > M-2 > M-3 breaks the tie.
- **R-2':** the winner is REJECTED in favour of the reference cell (T0 × N-legacy)
  if its M-3 falls more than 0.05 SS below that reference cell's M-3.
- **R-3':** single shot over the 4 declared cells, 3 seeds each (12 runs); nothing
  else trains before the selection is logged.
Disclosure: the M-2 selection metric remains "paired win fraction vs AR Ridge"
regardless of which baseline is later named strongest for the TEST statistic
(selection metric ≠ test reference; the test reference is named in §5 after the
baseline stage).

## 6. Statistics (pre-declared)

- **Primary (single hypothesis, α = 0.05, one-sided):** for per-event
  differences Δ_e = MAE_fire_t6(M2)_e − MAE_fire_t6(B6)_e over the primary
  event set, **H1: median(Δ_e) < 0** — equivalently, P(M2 has lower
  MAE_fire_t6 than B6 on a test event) > 1/2. Test: paired sign test on
  sign(Δ_e); ties (Δ_e = 0) dropped. B6 = gbdt_log_T1, the strongest
  pre-registered baseline per §5 after the B-2 renaming rule (previously
  ar_ridge_T0 under DEC-5).
  Primary event set: the DEC-3-audited, DEC-1-excluded pool — **block A: n = 30**
  as of 2026-08-26 (R-1 re-screen confirmed no change); block B added at season
  close under the identical mechanics. N_min (below) applies additionally at
  evaluation time.
- **Evaluability rule:** a test event enters the paired statistic only if it contributes
  ≥ 5 dataset-level fire-active forecast windows (calibrated on historical events:
  varnavas_2024 = 7 healthy, split_2017 = 0 broken). ALL gate-passing events are
  reported in the event table regardless; **both counts — gate-passing and
  evaluable — are reported.**
- Secondary (descriptive/supporting, no confirmatory claims): **the same paired
  sign test vs B1 naive persistence**; the DEC-1 sensitivity set (§6a, n = 32 for
  block A); **AGG-1 event-level MEDIAN SS_fire_t6 as the descriptive headline,
  with mean ± std as a secondary table**; Wilcoxon signed-rank on the same pairs;
  SS_fire_t6 vs persistence per event with volatility
  descriptor; event-level bootstrap CIs explicitly labeled coarse; MAE at
  t+1/t+3/t+6/t+12 per evaluation-standards format.

**Pre-registered interpretation (AMEND-7, 2026-08-27 — locked with this
document; the manuscript's reading of the primary outcome is fixed BEFORE the
answer exists):**
- If the primary sign test is significant: the deep framework adds a
  measurable event-level increment over the strongest classical baseline on
  sequestered multi-season data.
- If the primary is null: the result is reported as "no demonstrated
  event-level advantage of the deep framework over log-space gradient
  boosting"; the feasibility claim is supported by the pre-registered
  secondary vs-persistence comparison, reported with its exact p-value and
  explicitly labeled secondary — one pre-registered primary hypothesis, no
  multiplicity adjustment; the secondary licenses a feasibility statement,
  not a confirmatory advantage claim (wording per IA-A4). The representation
  finding (heavy-tail rescue across all model classes) stands as the paper's
  principal scientific result in either case.
- No post-hoc subgroup, horizon, or block re-analysis will be promoted to
  confirmatory status under any outcome.

**Pre-registered descriptive stratifications (AMEND-8, memo A6, approved with
tightenings T-1…T-4, 2026-08-27):** the single access additionally reports, for
M2 vs B6, three pre-declared strata — (a) per-horizon profile over the frozen
set {t+1, t+3, t+6, t+12}; (b) wind-regime conditional skill at t+6, windows
classified calm vs excursion by |delta_wind_3h| at the last input step against
the frozen TRAIN 1σ threshold (0.7994305491447449, computed from dataset_v6_1
train before hashing); (c) event-volatility tertiles at t+6, V = population
std of Δlog1p(frp_sum_mw) over consecutive 1-hour fire→fire slot pairs, with
frozen TRAIN tertile boundaries (0.8348539322108006 / 1.0851316749933935 over
the 71 train events). **T-1: per-stratum outputs are paired win counts and
median ΔSS only — no p-values, no confidence intervals, no language implying
a test; AMEND-7's no-promotion clause applies without exception. T-2: all
definitions and constants are frozen verbatim in the hashed access script;
nothing is computed from test data except the per-event/per-window values the
strata classify. T-3: all strata are archived as ONE table per run
(`*_strata.json` in TEST_ACCESS_RESULTS/); the manuscript (main text or
supplement) reproduces the FULL strata table, and prose excerpts are
permitted only alongside it (tightened per IA-A6). T-4: the modified script
was re-hashed and dry-run-revalidated on val; all hashes in §8.**
### 6a. D7-restored events — primary exclusion + sensitivity inclusion (DEC-1, 2026-08-25)

Three Test-A events were admitted only after the D7 conflict-rule refinement
(declared 2026-08-25): `es_rua_a_2025_0808`, `pt_trancoso_sao_pedro_2025_0808`,
`pt_real_2025_0728` (flagged `d7_restored = true` in the event catalogue; no other
event carries the flag).

**Timeline disclosed in full:** the inherited conflict rule (inclusive `<=`
comparisons) treated zero-area bbox edge-contact as a conflict and dropped these
three sub-episodes → the effect was observed during the E2.1 sub-division → D7 was
declared (a conflict requires POSITIVE-AREA bbox intersection; EFFIS burnt-area
metadata was known at declaration time, but NO FRP data for any affected event had
been downloaded) → the three events were re-admitted → each passed the frozen
five-criteria gate independently.

**Pre-declared handling — exact membership (arithmetic per the 2026-08-28
adversarial review, B-6):** the PRIMARY paired sign test EXCLUDES the
d7_restored events, so the primary claim carries zero timing-dependent
filtering decisions. A PRE-DECLARED SENSITIVITY ANALYSIS repeats the identical
statistic INCLUDING them. Of the three restored events, **two survive the
DEC-3 pixel-disjointness audit and form the sensitivity increment:
`es_rua_a_2025_0808` and `pt_trancoso_sao_pedro_2025_0808`; the third,
`pt_real_2025_0728`, was DROPPED by the DEC-3 audit's mechanical drop-smaller
rule (overlap_fraction 0.667 with pt_trancoso_sao_pedro) and appears in the
descriptive table only.** The arithmetic: 32 gate-passing DEC-3 survivors =
30 primary + 2 d7_restored. **Implementation binding (IA-B1): the access
event list carries an `in_sensitivity` flag; block-A values are frozen in
`resubmission/test_event_list_blockA_LOCKED.csv` (hashed in §8) and asserted
row-identical at runtime; the sensitivity sign test runs over exactly the
in_sensitivity events. For block B, in_sensitivity = in_primary — no
d7-restoration timing issue exists for B.** Both results are reported in the
manuscript
regardless of outcome. (Rationale of record: the primary excludes a small
fraction of the combined test set — 2 of 32 in block A's sensitivity set;
the sensitivity analysis preserves evaluation of the season's two most
intense fires.)

- **Fire-active mask (inlined verbatim per B-7, from
  `.claude/skills/evaluation-standards.md`):** `fire_active_mask =
  (targets > 0).any(axis=1)` — a forecast window is fire-active iff at least
  one non-zero FRP value appears anywhere in its 12-h target window; MAE_fire
  is the mean absolute error over fire-active windows ONLY. **Aggregation
  grain:** all statistics are computed per event first and aggregated at
  event level — never pooled across events at window level. Horizon set:
  {t+1, t+3, t+6, t+12}. The full protocol file is additionally hashed into
  §8.
- Target test size: ≥ 10 events combined across blocks (n = 10 all-positive:
  p ≈ 0.001; 9/10: p ≈ 0.011). ~~If block A alone yields ≥ 10, submission may
  proceed on block A with block B reported at revision or as follow-up.~~
  **SUPERSEDED by ACC-1 (memo A4, 2026-08-26; recorded per AMEND-6,
  2026-08-27): the test blocks are accessed exactly once, TOGETHER, after
  season close — there is no block-A-alone access or submission path.**

## 7. Reporting

Per-event table (both blocks): gate metrics, n_fire windows, volatility descriptor
(pers-MAE-based, computed only AT evaluation time), per-model MAE_fire at the four
horizons, SS vs persistence, paired direction vs strongest baseline. Aggregates at
event grain: MEDIAN primary (AGG-1), mean ± std secondary. Publish-regardless —
unconditional: results are reported whatever their direction or significance.
Any deviation from this document is reported as a protocol deviation in the
manuscript.

## 8. Locked-artifact register (filled at lock time)

All hashes SHA-256 unless noted; full 64-character values (register hygiene per
the 2026-08-28 adversarial review, A-b). Only the locked-prereg row is filled
at signature.

| Artifact | Hash |
|---|---|
| This pre-registration (locked version) | Self-reference exclusion (a file cannot embed its own hash — IA-1 Step 4 implemented per intent): the locked file's SHA-256 is computed at Step 4 over the file AS PUBLISHED (CRLF line endings, per IA-N7) and recorded in the lock package §7 registration record, RESEARCH_LOG, and the Zenodo deposit metadata. The only value not in this table, by necessity. |
| scripts/frp_quality_gate.py (frozen gate) | e02db840cf77031d4ba32a33ed712cd126fc15d1d4d75fcaa94222135d725115 |
| scripts/pixel_disjointness_audit.py v2 (v1 superseded — invalid run disclosed in §9.5; v1 hash e6709e6f1ac7c20af3f19095b7e52cc9b366d63a3d0c6989bcf83267b4de8b52) | cc1dbf86ecfa26b04f73ff9f49016daef73311b65ca5973fc8a39f39c33ba2c2 |
| M1 (original naming C-FROZEN/C-HONEST): SUPERSEDED — checkpoints unlocatable per AMEND-5 (2026-08-27); M1 re-scoped per RAT-9 to the Zenodo-deposit ensemble (rows below) | — |
| M2 freeze table data/models/e28/m2_freeze_hashes.json (frozen 2026-08-26T09:39:40Z; contains the 10 member checkpoint hashes, first 7097b786c38e73a634312975cb750e86c092d73dd60dc319a4dbb6aa3c479ca5, last 9cb79735814a097829bd8ba5cbe844c7e64435a470342aa1e58263c1a6536235) | 27b947348ff4ee493bc46c88a659ad89d1372bfcb9c572cf5ec98b222bf43cfa |
| dataset_v6_1.h5 (also inside the M2 freeze table — cross-reference, single source) | 0a1a7c8ac23d6525bac1f93e99ec6a9befd537653baaa7495e65693f6ff75724 |
| normalization_params_v6_1.json (also inside the M2 freeze table — cross-reference) | b55421f74665c57535fd4e7e82d2dc0633e4c96c426e049a35c6ff2a84fbb827 |
| scripts/e28_train_grid.py (M2 trainer) | 19e70a5051aa905a86bebb6a399835c437ef0e5959728b314bdb5c4d0f37b832 |
| scripts/e25_seeded_split.py (FINAL split procedure, §2) | 591deb027968187eee17081e080e3605caa23f52ba7f30fde235b0576b83ab73 |
| Raw baseline suite results: baselines_v6_1_full_val_results.json | d10a2de64dffaa8977139bf60a2c98c6e5f544618941cbce8d5b7ac35942f7b5 |
| Raw baseline suite table: baselines_v6_1_full_val_table.csv | 59dd8f5cb3d1117388d199c331705a9cb8f8e644cd356580c2768cd520b837ec |
| B-2 baseline extension: scripts/b2_log_baselines.py | 3ac578daba2da2c46fa802a9a0d9db27f4821e1c228e4b7726fd39e461ac0551 |
| B-2 results (contains FINAL B6 = gbdt_log_T1 definition + config): baselines_v6_1_b2_log_results.json | 27c1f967e548893a77f1567365e67a5edf01394449696dc4c8e0a7163ca776dd |
| B-2 table: baselines_v6_1_b2_log_table.csv | 2a702217f7b16c1711ef49697f617d1c03c0edab23825c46f7576871fb6edc53 |
| Evaluation protocol .claude/skills/evaluation-standards.md (fire-active mask + event grain inlined in §6a per B-7) | 147b1b3a5fae5172f35103a21a2183b8587d3ae6b047521fa6d77091d50365f3 |
| Access script scripts/test_access_v1.py — ORIGINAL (2026-08-26; superseded per T-4) | 93ed7538e4961ff3e2dea94240c2aa0ed933143d99eb73c68d1b47d2380329b3 |
| Access script scripts/test_access_v1.py — REV 2 (AMEND-8 strata + M1 re-scope, 2026-08-27; superseded per T-4 by the A-d environment-pin revision) | 8217e2b60f566f209ac767eea797989b53a8053a02c7e65118116eba6294e9ae |
| Access script scripts/test_access_v1.py — REV 3 (A-d environment pin, 2026-08-28; superseded by REV 4 per T-4) | 7d48a32aa0a6838e2ad156e8c95ccc7cdb114f4f1d8881bcb2527cfa2d462f2f |
| Environment lockfile resubmission/access_environment_lock.txt (A-d; python 3.11.5, numpy 1.26.4, pandas 2.1.4, scikit-learn 1.2.2, scipy 1.17.1, torch 2.6.0+cpu, h5py 3.8.0; the critical four asserted at runtime by the access script) | 3c602982003cc19e55f1d7d8937e304068ad25b7c7627fe7dba7d1ac5ba501e5 |
| M1 deposit (RAT-9): resubmission/m1_deposit/frp-prt-grsl-v1.0.0.zip (Zenodo 10.5281/zenodo.20627737) | md5 771491878286355cd9b15d418aa195cb (Zenodo-published checksum, verified on download) |
| M1 member checkpoints seed_000…009.pt — full SHA-256 table frozen as constants in the access script (first 63d2ed06bbc9affc248a596f6acab4c373fc017facfd0a0fba29d91b79ea7fc1, last d8e2d165911dab55349d8985c57a09b4bbeca23b9f4939e8404dcb6306d5b76c); deposit normalization_params_v3.json 3315db8327f3990d4d3cfbc72173ab694e5af3826bf456d7603494b4471784ef; windfeats_stats.json 6b4e23c96202cc5e7bb786a1e3fa24bd364b7703f46ecfaa4e074f87f6c79b65 | in-script constants + this row |
| §5a grid summary e28_grid_summary.json (12-run grid behind the selection; IA-A8-i) | ab88006baecbbbf7f2327d0dae077a17afc345200d37b9009a1fafeb41f99534 |
| M1 deposit zip SHA-256 (IA-A8-ii; complements the Zenodo md5, which is a provenance echo, not an integrity guarantee) | 305cad523e9f139e2a484c48a96fc5cbdce61ed99585567839bb770c7bf2739d |
| Frozen block-A composition resubmission/test_event_list_blockA_LOCKED.csv (46 rows; 30 primary / 32 sensitivity / 3 d7_restored flags; asserted row-identical at access; IA-B1.3) | b339f1795aaf977ebef0a2e8110d5e4df3c98e0cea0f902f6ec22a4f658b05e3 |
| Block-A input manifest resubmission/blockA_input_manifest.csv (SHA-256 of all 92 per-event hourly + ERA5 CSVs, 46 events, 0 missing; content-blind — sequestration untouched; IA-A8-iii; block-B analogue at season close) | 7163744d71085a1bdb3178b162d135265dfc316b367c1894d88f7e616e0122f0 |
| dataset_v6_1_manifest.csv (runtime-asserted; IA-A2) | cdff3201039e1846530568839c71254241a0c19881bf33b0bfff1bd1c367e1df |
| Access script scripts/test_access_v1.py — **REV 4, CURRENT** (IA-B1 sensitivity set + frozen block-A assertion; IA-A2 input-hash asserts; IA-A5 zero-window rows; IA-N5 paired Wilcoxon; dry-run revalidated 2026-08-28: env pin OK, input hashes OK, sensitivity_set == primary on val as expected, anchors reproduced exactly) | 6b403afc53b41a4f0c5bf75b45954ae39cdd35b06ee9d2ca6feef56ee897b1c4 |
| Access dry-run: REV 4 revalidation results (2026-08-28) | 20e21d886952e525b493ab08602e16ee8fc829ff193ad1f740e0cacd3a92df75 |
| Access dry-run: original results (2026-08-26, script REV 1) | bea1d7fe5e43587b48762fcc8dd9609d8e4781345cdb20c1181eeb202b7faa0f |
| Access dry-run: REV 2 revalidation results (2026-08-27) | 3fbe1ee981e4e4e9167d214efb77f9698b774eadc1dac36fed8da3acf73c97eb |
| Access dry-run: REV 3 revalidation results (2026-08-28; differs from REV 2 only by run_utc) | 75ac0978c34131ab0e42541970aeb597a9a89169c62628ffd6c604d79f8b8528 |
| Access dry-run: strata table (byte-identical across REV 2 and REV 3 runs) | 48fc8a445f06ce93a2377ca9054bbcd75cbc955dfc6cffed7841becfa05cd32a |

## 9. Open-item resolution record (converted per the 2026-08-28 adversarial
review, A-c — every item resolved except sign-off)

1. **RESOLVED 2026-08-28 (IA-1/G6′):** supervisor sign-off replaced by
   self-certified lock + public registration per IA-1 and G6′; gate package,
   area floor (final text, IA-B5), D4/D7, and DEC-4 are certified by the
   executor within the registered lock. [The former "block-A-alone submission
   timeline" item is SUPERSEDED by ACC-1 — see §6.]
2. **RESOLVED 2026-08-25:** Task D1 results → baseline suite finalized (§5;
   BASELINE_REPORT_2026-08-25.md); extended and closed by B-2 (2026-08-26).
3. **RESOLVED 2026-08-25:** E1.3 gate screening complete → block A 46
   gate-passers (→ 30/32 after DEC-3), block B 32 pending season close
   (E1_PROGRESS.md).
4. **RESOLVED 2026-08-26/27:** M2 trained + frozen (m2_freeze_hashes.json);
   access script authored, revalidated, hashed (§8).
5. **Event-independence rule — RESOLVED; executed 2026-08-25 (v2) — DEC-3:**
   the adjudication memo permits a pixel-COORDINATE-only overlap audit pre-lock
   (coordinates reveal strictly less than the peak-FRP gate metadata already held).
   Declared rule, fixed before execution: for each same-block pair with positive-area
   bbox intersection and overlapping windows, compute the shared fraction of rounded
   (0.05°) fire-pixel positions within the bbox-intersection region over the shared
   window; overlap_fraction = |P_A ∩ P_B| / min(|P_A|,|P_B|). Fraction < 0.10 →
   independent; ≥ 0.10 → dependent pair, and the event with FEWER catalogue
   fire-active slots is DROPPED from the paired statistic (kept in the descriptive
   table); transitive resolution in descending overlap order; merging only if two
   events are one fire by any reasonable reading, and only by escalation.
   Audit script: `scripts/pixel_disjointness_audit.py`, with a hard whitelist
   guard (only LATITUDE/LONGITUDE HDF5 datasets readable; any other access
   aborts). Results are append-only; the resulting test composition is fixed here
   at lock. **Version history (disclosed):** v1 (SHA-256 `e6709e6f…8b52`) was
   executed on block A 2026-08-25 after the W-1 extensions and found all 31
   overlapping pairs independent — but the run is INVALID: LSA SAF ListProduct
   files are full-disk pixel lists and v1 omitted the per-event bbox filter, so
   its denominators were disk-wide and every fraction spuriously ≈0 (v1 output
   preserved as `pixel_disjointness_test_A_2025_2026-08-25_v1_INVALID.csv`).
   v2 adds the bbox filter only — the DECLARED RULE is unchanged. v2 SHA-256
   `cc1dbf86ecfa26b04f73ff9f49016daef73311b65ca5973fc8a39f39c33ba2c2`,
   recorded here BEFORE the v2 run. The implementation defect was identified
   from the coordinate-count pattern alone (identical |P_A|=|P_B| disk-wide
   counts); no FRP content was involved.
