# Freeze Report — E2.4 → E2.7-LITE (per adjudication memo step 7)
**Date:** 2026-08-25 · Session scope executed in full; stopped at the memo's boundary.
All rulings applied as issued; three execution refinements from the chat acceptance
record incorporated (4-cell rule, trebinje pin, HDF5-level audit guard).

## 1. Final pool counts

| Pool | Events | Notes |
|---|---|---|
| Frozen dev pool | **84** (71 train / 13 val) | `frozen_dev_pool_2026-08-25.csv`; sequestration assert (≤2024) in code |
| — T0 subset | 43 train events | = T1 ∩ v5 pool |
| Test A (2025) | 46 gate-pass (52 onboarded) | 3 flagged `d7_restored`; primary excludes, sensitivity includes (prereg §6a) |
| Test B (2026) | 32 gate-pass (36 onboarded + 1 orphan) | season-close re-screen pending |

## 2. Hygiene results (catalogue backed up pre-change)
- 139 missing country codes filled (0 remain); `legacy_split` column preserves all
  historical labels; dev events carry `split='dev_pool'`.
- `d7_restored` flag on exactly 3 events. DEC-2 exclusion record issued:
  `EXCLUSIONS_E2_2026-08-25.md` (rule-version column included).
- No duplicate event names. One discrepancy documented: `ba_n_a_2026_0819` has an
  hourly CSV but no catalogue row (silent worker failure; it is the right-censored
  gate-FAIL already scheduled for season-close re-screen).

## 3. B1 window-truncation audit (AMEND-3) — MATERIAL FINDING
10 EFFIS-derived events have member polygons igniting after their downloaded window
(`b1_window_truncation_audit_2026-08-25{,_detail}.csv`):

| Event | Block | % burnt area outside window |
|---|---|---|
| pt_reriz_e_gafanhao_2024 | dev | **99.0%** — its "118k ha extreme-tail" label describes fires we never downloaded |
| dz_n_a_2026_0706 | B | 80.9% |
| dz_n_a_2026_0708 | B | 75.4% |
| me_n_a_2017_0707 | dev | 72.6% |
| mk_vasilevo_2025 | A | 68.2% |
| ba_n_a_2026_0731 | B | 61.6% |
| pt_lazarim_2025 | A | 40.6% |
| mk_novaci_2024 | dev | 31.7% |
| dz_n_a_2026_0712 | B | 27.2% |
| it_bivona_2026 | B | 5.4% |

Gate admission integrity unaffected (the gate judged the downloaded window).
**ESCALATION ITEM (your call, prereg content):** declare a mechanical
window-extension rule (e.g. end = latest member FIREDATE + 9d, capped) and
re-download the flagged events, or accept truncation with per-event disclosure.
Natural decision point: with DEC-6 at season close (block B holds 5 of the 10).

## 4. Split (E2.5) — FINAL
Criteria A1–A5 + weighting + remediation logged BEFORE the draw. Seed 20260824
failed A4 (>1 degenerate-slot val event); declared seed+1 remediation → **seed
20260825 passes all criteria. 71 train / 13 val; trebinje pinned train.**
Val: alexandroupolis GR, ba_n_a_2020_0409 BA, bejaia DZ, bonarcado IT,
es_carballeda_2017 ES(Oct), hr_velebit HR, huelva ES, kabylie DZ, morocco_rif MA,
mugla TR, pt_algarve2 PT, sierra ES, tn_n_a_2017 TN.
Mechanical outcome of note: legacy val events mandra/rhodes/caceres/split_2017 and
all four burned ex-test events landed in train.

## 5. dataset_v6 build manifest (E2.6)
- `dataset_v6.h5`: **14,519 samples (12,425 train / 2,094 val), 84 events, 32
  features, LOOKBACK=48, RAW inputs.** Manifest carries per-sample `in_T0`.
- §5a amended (AMEND-2) BEFORE the build: cells {T0,T1}×{N-legacy,N-robust},
  rules R-1'/R-2'/R-3' (pairwise 2-of-3 dominance; guard vs T0×N-legacy; single
  shot, 12 runs). Four normalization parameter sets stored, each fit on its own
  train subset. N-robust fixed: log1p for non-negative FRP-scale features,
  signed-log for frp_trend_6h, median/IQR(÷1.349) scaling, std fallback.
- Scale evidence for AMEND-2: T0 frp_lag_1h std 6,189 MW (≈v5's 6,186 ✓) vs
  T1 5,308 MW.
- Full `06_verify_era5.py` suite over the frozen pool ONLY (test blocks excluded —
  check 5 computes FRP diurnal stats): **GO, 5/5 PASS**. `era5_premerge_qc_v6.json`.
- 7-check readiness (adapted to raw layout, independent read-back incl. end-to-end
  spot-check vs source CSVs): **ALL PASS**. `dataset_v6_readiness.json`.

## 6. Legacy-baseline validation numbers (E2.7-LITE)
`baselines_v6_lite_val_results.json` / `_table.csv`. Guardrail held: nothing fed
back into pool, split, or adequacy.

| Statistic | AR Ridge (T1) | AR Ridge (T0) |
|---|---|---|
| SS_fire_t6 median (12 evaluable events) | **+0.1834** | **+0.1703** |
| SS_fire_t6 mean ± std | −1.58 ± 4.59 | −1.84 ± 5.24 |

- **Build VERIFIED**: the median matches the v5-era AR Ridge (+0.183 excl-outlier)
  almost exactly, and per-event values sit in v5-consistent ranges (alex +0.19,
  mugla +0.20, kabylie +0.21, huelva +0.32, es_carballeda Oct-2017 +0.32).
- The mean is dominated by 4 mechanically-drawn low-intensity val events
  (ba_n_a_2020_0409 SS −15.9 at persistence-MAE 21 MW; morocco_rif −2.27;
  bonarcado −1.36; bejaia −1.15) — the documented bejaia-class AR failure on flat
  events (which threatens AR, not the PRT, per FINDINGS §5). **Decision needed at
  D1/prereg: pre-declared robust val aggregation (median/trimmed) or outlier rule.**
- tn_n_a_2017_0808 contributes 0 dataset-level fire windows (split_2017
  phenomenon) → 12 evaluable val events.
- Reported deviation: AR-T1 marginally beats persistence at t+1 on the val mean
  (1,517 vs 1,567 MW), unlike v5-era; plausibly composition-driven; to be
  examined in the D1 session.

## 7. DEC-3 audit — drafted + hashed, NOT executed
Rule fixed in prereg (open item 5 → resolved-as-rule): 0.05°-rounded pixel-position
overlap over shared window within the bbox intersection; <0.10 independent; ≥0.10 →
drop-smaller (by fire_slots) from the paired statistic; transitive; merge only by
escalation. Script `scripts/pixel_disjointness_audit.py` with hard
LATITUDE/LONGITUDE-only whitelist guard. SHA-256
`e6709e6f1ac7c20af3f19095b7e52cc9b366d63a3d0c6989bcf83267b4de8b52` (prereg §8).

## 8. Updated prereg draft
`prereg_test_blocks_v1_DRAFT.md` now contains: §3.1 D4+D7 episode rules; §5a
AMEND-2 4-cell selection; §6a D7-restored primary-exclusion/sensitivity subsection
with full timeline; §9.5 resolved independence rule + audit hash; §8 register rows
for the gate and audit scripts. Still NOT LOCKED.

## 9. Decisions now on your desk
1. **Window-truncation rule** (§3 above) — extend-and-redownload vs disclose-only;
   block-B-heavy, natural fit with season close.
2. **Robust val aggregation** for M-1/headline (§6 above) — decide in the D1
   session before §5a training.
3. Standing: DEC-4 ratification at lock; DEC-6 season close; DEC-1 flag must be
   prominent in the sign-off package (memo standing item — noted in prereg §6a).

## Next session (per AMEND-1)
E2.7+D1 combined baseline stage on fresh context: diurnal persistence,
persistence×diurnal-shape, GBDT — all on the v6 val; strongest TEST baseline
named in the prereg. Then E2.8 (4 cells × 3 seeds), E2.9 selection, E2.10 ensemble.
