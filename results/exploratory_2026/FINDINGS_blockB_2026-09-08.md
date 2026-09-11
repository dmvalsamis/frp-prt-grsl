# What the 2026 fires (test block B) have to offer — exploratory findings, 2026-09-08

**Status: EXPLORATORY, NOT PRE-REGISTERED.** Ordered by the user on 2026-09-08 after two
warnings of the cost. Block B's sequestration is consumed by this work; every number below
must be labelled exploratory in any manuscript, and the access is disclosed in
RESEARCH_LOG (two entries, 2026-09-08). Branch `explore-blockB-2026`. Locked block-A outputs
in `TEST_ACCESS_RESULTS/` were not modified (hashes unchanged).

Protocol identity: all scoring reuses `scripts/test_access_v1.py` (sha256 6b403afc…b1c4)
unchanged — same windows, hash-verified M2 ensemble, deterministic B6 refit, M1 deposit
ensemble, B1 persistence, N_min = 5, AGG-1 medians, AMEND-8 strata; environment pin and
input hashes asserted at every run. Differences from the block-A access, all disclosed:
no DEC-3 pixel-disjointness audit and no W-1 window extension were applied to block B; the
32 gate-passing 2026 events (frozen gate, results of 2026-08-24; 4 FAIL excluded) are all
treated as "primary"; ERA5 for 8 events was pulled today (windows ending after 2026-08-15).

## 1. Headline: block B answers the primary question, in the negative

| test (one-sided sign test on per-event MAE_fire,t6) | block A (locked) | block B (32) | A-primary + B (61) |
|---|---|---|---|
| M2 vs B6 gradient boosting | 19/10, p = 0.068 | **15/17, p = 0.70** | 34/27, p = 0.22 |
| M2 vs B1 flat persistence | 27/2, p = 8e-07 | **29/3, p = 1e-06** | 56/5, p < 1e-9 |

Block B does not replicate M2's block-A lead; the direction reverses. Pooling does not
rescue the hypothesis: p rises from 0.068 (n = 29) to 0.22 (n = 61). Two things replicate
cleanly: feasibility against persistence, and the representation finding (every log-space
model, including the archived M1, lands at the same skill level on 2026 fires).

| AGG-1, event-level median SS_fire,t6 | A (45) | B (32) | A + B (77) |
|---|---|---|---|
| M2 | +0.3489 | +0.3022 | +0.3265 |
| B6 | +0.3416 | +0.3179 | +0.3251 |
| M1 (archived 2026 submission ensemble) | +0.3345 | +0.3073 | +0.3135 |
| M2 mean ± std | +0.2874 ± 0.1664 | +0.2445 ± 0.2206 | +0.2695 ± 0.1906 |

Horizon profile replicates exactly (M2 wins / B6 wins): A 2/43, 17/28, 28/17, 29/16;
B 3/29, 11/21, 15/17, 16/16 at t+1/3/6/12. B6 dominates the first hours; parity from t+6.

## 2. The three 2026 collapses and what they are

Three block-B fires are the only events in either block where M2 trails B6 by more than
0.3 skill: pt_cambra_e_carvalhal (M2 −0.050 vs B6 +0.501), fr_porge (−0.192 vs +0.414),
pt_cerdeira (−0.332 vs +0.109). Window-level profiles (`windows_blockB_*_profile.md`)
show one mechanism: 76–77 % of M2's total error in cambra and porge falls on the single UTC
day after the peak, where M2 forecasts continued growth (mean bias +2,890 and +6,743 MW at
t+6) while B6 forecasts the decline (bias −247 and −708 MW). In cerdeira the peak sits in the
lookback while persistence is zero, and M2's residual lifts a zero forecast to ~127 MW
against a truth of 8 MW (39 % of its error on that day, denominator 42 MW).

A first hypothesis — "burst-then-flicker" trajectory shape — was tested across all 77
events and **rejected**: tail share correlates positively with M2's margin overall
(Spearman +0.37, p = 0.002), and block A's 8 flicker-shaped fires went 5–3 for M2.
Shape does not explain the collapses (`shape_diagnostic_*.md`).

## 3. The general mechanism: the post-peak transition day decides

`decisive_day_*.md`, 77 events, window-level predictions:

| | A (45) | B (32) | all (77) |
|---|---|---|---|
| M2 wins | 28 | 15 | 43 |
| outcome flips if the post-peak day is removed | 5 | 4 | 9 |
| sign test excluding the post-peak day | 29/16, p = 0.036 | 15/17, p = 0.70 | 44/33, p = 0.13 |
| sign test on the post-peak day alone | 19/17, p = 0.43 | 8/15, p = 0.95 | 27/32, p = 0.78 |
| M2 over-forecasts (bias > 0) on the post-peak day | 8 | 7 | 15 |

Reading: M2's small edge lives away from the peak transition; B6 is better on the day after
the peak, and in 9 of 77 fires that one day alone decides the paired outcome (5 for B6,
4 for M2). This is the window-level face of the 2026-03 wind-excursion finding (M2's
trajectory-continuity prior breaks at regime changes), now seen on 77 fires the model never
trained on. It is a mechanism, not a rescue: excluding the post-peak day is post hoc, and
even then block B is 15/17.

## 4. What this means for the decision on how to move on

1. **The block-A-only letter remains defensible as written**, with block B disclosed as
   consumed exploratorily and its outcome reported (it strengthens the null, which is the
   letter's honest result). Hiding this run is not an option; it is logged.
2. **A block-B confirmatory access is no longer available.** The pre-registered
   season-close path (OPEN_QUESTIONS #15 option b) is closed by this work.
3. **Retraining with 2026 fires in the pool** (option c) would leave no untouched test data
   at all. If the goal is a bigger evaluation, the only clean route now is the 2027 season.
4. **The scientific yield is real:** (i) the representation finding replicates on a second
   independent year; (ii) forecastability at t+6 replicates (29/32); (iii) the deep model's
   advantage over GBDT does not exist at the event grain across 77 fires; (iv) the
   post-peak transition is where the two model classes differ, in B6's favour. Items
   (i)–(iii) are letter-grade statements once labelled exploratory; item (iv) is the seed of
   a follow-up (a transition-aware residual, or a GBDT/PRT hybrid gated on trajectory phase).

## 5. Files

| file | content |
|---|---|
| `blockB_inventory.csv` | 36 catalogued events, gate verdict, file presence, inclusion |
| `exploratory_blockB_20260908T133104Z_*` | run 1, 24 events (ERA5 available at 13:31 UTC) |
| `exploratory_blockB_20260908T140106Z_*` | run 2, all 32 gate-passing events |
| `exploratory_record_*.json` | run records (selection, input hashes, warning) |
| `combined_AB_20260908T140*_{json,md,_per_event.csv}` | pooled A+B analysis (32-event B) |
| `windows_blockB_20260908T134100Z*` | window-level predictions + daily error profiles, 5 events |
| `shape_diagnostic_*` | trajectory-shape hypothesis test, 77 events (rejected) |
| `decisive_day_20260908T140546Z*` | post-peak-day diagnostic, 77 events |
| `manuscript/figures/figX_paired_AB_exploratory.pdf` | paired scatter A+B, block-coded, with horizon wins |
| scripts | `explore_blockB_eval.py`, `explore_combined_AB.py`, `explore_blockB_windows.py`, `explore_shape_diagnostic.py`, `explore_decisive_day.py` |
