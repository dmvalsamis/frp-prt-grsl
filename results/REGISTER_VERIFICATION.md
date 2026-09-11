# Register verification -- T1 (2026-09-08)

Recomputed from `resubmission/TEST_ACCESS_RESULTS/access_20260828T123059Z_per_event.csv` alone (frozen constants read from `..._strata.json`; `..._results.json` used only as a second witness). No model, baseline, or dataset was touched; `scripts/test_access_v1.py` was NOT run. Script: `scripts/verify_register.py`.

## 1. File hashes (SHA-256) vs RESEARCH_LOG 2026-08-28

| file | recomputed | logged | match |
|---|---|---|---|
| access_20260828T123059Z_results.json | `9f3c1d8410ccb79f2e1841ed2e04f191152d818f8e216f8dc404e7e3b8262226` | `9f3c1d8410ccb79f2e1841ed2e04f191152d818f8e216f8dc404e7e3b8262226` | YES |
| access_20260828T123059Z_per_event.csv | `2c1dd44498b3fe76be7005b493eb0d2424c8347d601efcca1a505418fceb5e79` | `2c1dd44498b3fe76be7005b493eb0d2424c8347d601efcca1a505418fceb5e79` | YES |
| access_20260828T123059Z_strata.json | `414634555287b46d1ccbd862cffd5d108d70376b4f19ca9c34a625229fb435a0` | `414634555287b46d1ccbd862cffd5d108d70376b4f19ca9c34a625229fb435a0` | YES |
| access_record_20260828T123059Z.json | `9f1d6bf00eac0a71e95d781b34ed6e3b8423a4fc8b55dca3eb20e3bc5cb9b363` | `9f1d6bf00eac0a71e95d781b34ed6e3b8423a4fc8b55dca3eb20e3bc5cb9b363` | YES |
| access_console_20260828.log | `802ba2ccbab8765a26d2ad6f70de4547b4c583c6f2f3b727021f8853ba574eee` | `802ba2ccbab8765a26d2ad6f70de4547b4c583c6f2f3b727021f8853ba574eee` | YES |

Method notes: sign test = one-sided exact binomial (scipy `binomtest`, alternative='greater', ties dropped) on wins = events where MAE_M2 < MAE_comparator; Wilcoxon = scipy `wilcoxon` on the paired MAE differences, alternative='less'; std = sample (ddof=1, the convention used by the locked results.json); medians = `pandas.median`. Float values are matched at the register's own printed precision.

## 2. Recomputed vs registered

| section | quantity | recomputed | registered | match |
|---|---|---|---|---|
| counts | gate_passing_blockA | 46 | 46 | YES |
| counts | evaluable_blockA | 45 | 45 | YES |
| counts | primary_n | 29 | 29 | YES |
| counts | sensitivity_n | 31 | 31 | YES |
| counts | primary_declared_n (in_primary flag over all 46 rows) | 30 | 30 | YES |
| counts | sensitivity_declared_n (in_sensitivity flag over all 46 rows) | 32 | 32 | YES |
| counts | windows_blockA | 11926 | 11926 | YES |
| counts | fire_windows_blockA | 7257 | 7257 | YES |
| counts | fire_windows_calm | 5529 | 5529 | YES |
| counts | fire_windows_excursion | 1728 | 1728 | YES |
| counts | d7_restored | 3 | 3 | YES |
| counts | dec3_dropped_descriptive (evaluable and not in_sensitivity) | 14 | 14 | YES |
| counts | zero_fire_event | gr_n38e20_2025_0812 | gr_n38e20_2025_0812 | YES |
| counts | zero_fire_event_windows | 181 | 181 | YES |
| counts | N_min consistent (min n_fire evaluable >= 5; max n_fire non-evaluable < 5) | true | True | YES |
| primary M2 vs B6 | wins | 19 | 19 | YES |
| primary M2 vs B6 | losses | 10 | 10 | YES |
| primary M2 vs B6 | n | 29 | 29 | YES |
| primary M2 vs B6 | p_one_sided | 0.068023 | 0.0680 | YES |
| primary M2 vs B6 | wilcoxon_p | 0.184619 | 0.185 | YES |
| primary M2 vs B6 | reject_at_005 | false | False | YES |
| sensitivity M2 vs B6 | wins | 21 | 21 | YES |
| sensitivity M2 vs B6 | losses | 10 | 10 | YES |
| sensitivity M2 vs B6 | n | 31 | 31 | YES |
| sensitivity M2 vs B6 | p_one_sided | 0.0353778 | 0.0354 | YES |
| sensitivity M2 vs B6 | wilcoxon_p | 0.0787098 | 0.0787 | YES |
| primary-set M2 vs B1 | wins | 27 | 27 | YES |
| primary-set M2 vs B1 | losses | 2 | 2 | YES |
| primary-set M2 vs B1 | n | 29 | 29 | YES |
| primary-set M2 vs B1 | p_one_sided | 8.121e-07 | 8.1\times10^{-7} | YES |
| primary-set M2 vs B1 | wilcoxon_p | 1.192e-06 | 1.2\times10^{-6} | YES |
| AGG-1 (45 evaluable) | M2_median | 0.348862 | +0.3489 | YES |
| AGG-1 (45 evaluable) | M2_mean | 0.287381 | +0.2874 | YES |
| AGG-1 (45 evaluable) | M2_std (sample, ddof=1 -- the convention of the locked results.json) | 0.166357 | 0.1664 | YES |
| AGG-1 (45 evaluable) | B6_median | 0.341592 | +0.3416 | YES |
| AGG-1 (45 evaluable) | B6_mean | 0.267078 | +0.2671 | YES |
| AGG-1 (45 evaluable) | M1_median | 0.334465 | +0.3345 | YES |
| AGG-1 (45 evaluable) | M1_mean | 0.275731 | +0.2757 | YES |
| AGG-1 (45 evaluable) | M2_negative_events | 4 | 4 | YES |
| AGG-1 (45 evaluable) | B6_negative_events | 5 | 5 | YES |
| AGG-1 (45 evaluable) | M2_min | -0.174257 | -0.174 | YES |
| horizon medians M2 | t1 | -0.00087186 | -0.0009 | YES |
| horizon medians M2 | t3 | 0.189105 | +0.1891 | YES |
| horizon medians M2 | t6 | 0.348862 | +0.3489 | YES |
| horizon medians M2 | t12 | 0.390967 | +0.3910 | YES |
| horizon medians B6 | t1 | 0.271508 | +0.2715 | YES |
| horizon medians B6 | t3 | 0.243342 | +0.2433 | YES |
| horizon medians B6 | t6 | 0.341592 | +0.3416 | YES |
| horizon medians B6 | t12 | 0.394475 | +0.3945 | YES |
| strata horizon t1 | wins | 2 | 2 | YES |
| strata horizon t1 | losses | 43 | 43 | YES |
| strata horizon t1 | median_dSS | -0.250516 | -0.251 | YES |
| strata horizon t3 | wins | 17 | 17 | YES |
| strata horizon t3 | losses | 28 | 28 | YES |
| strata horizon t3 | median_dSS | -0.0090805 | -0.009 | YES |
| strata horizon t6 | wins | 28 | 28 | YES |
| strata horizon t6 | losses | 17 | 17 | YES |
| strata horizon t6 | median_dSS | 0.00436759 | +0.004 | YES |
| strata horizon t12 | wins | 29 | 29 | YES |
| strata horizon t12 | losses | 16 | 16 | YES |
| strata horizon t12 | median_dSS | 0.00675564 | +0.007 | YES |
| strata wind calm | wins | 27 | 27 | YES |
| strata wind calm | losses | 18 | 18 | YES |
| strata wind calm | median_dSS | 0.00459533 | +0.005 | YES |
| strata wind excursion | wins | 28 | 28 | YES |
| strata wind excursion | losses | 17 | 17 | YES |
| strata wind excursion | median_dSS | 0.00526919 | +0.005 | YES |
| strata volatility high | n | 16 | 16 | YES |
| strata volatility high | wins | 13 | 13 | YES |
| strata volatility high | losses | 3 | 3 | YES |
| strata volatility high | median_dSS | 0.0162561 | +0.016 | YES |
| strata volatility mid | n | 23 | 23 | YES |
| strata volatility mid | wins | 11 | 11 | YES |
| strata volatility mid | losses | 12 | 12 | YES |
| strata volatility mid | median_dSS | -0.00204864 | -0.002 | YES |
| strata volatility low | n | 6 | 6 | YES |
| strata volatility low | wins | 4 | 4 | YES |
| strata volatility low | losses | 2 | 2 | YES |
| strata volatility low | median_dSS | 0.0404243 | +0.040 | YES |
| strata volatility | vol_tertile labels consistent with frozen boundaries | true | True | YES |
| frozen constants (strata.json) | delta_wind_3h_sigma_train | 0.799431 | 0.7994 | YES |
| frozen constants (strata.json) | vol_tertile_lo | 0.834854 | 0.8349 | YES |
| frozen constants (strata.json) | vol_tertile_hi | 1.08513 | 1.0851 | YES |
| results.json witness | PRIMARY wins/losses/n | 19/10/29 | 19/10/29 | YES |
| results.json witness | M2_median_SS_t6 | 0.348862 | +0.3489 | YES |
| results.json witness | M1_median_SS_t6 | 0.334465 | +0.3345 | YES |

## 3. Verdict

**86/86 register values match; 5/5 file hashes match.** VERDICT: PASS -- the register is the same data as the repo's locked outputs.

Register file hash (as placed at `resubmission/number_register.json`): `12762d8696e4b567488869401f9252d3d226f2db5f92bc993ec61a9819dd0d75`

Not recomputable from per_event.csv (out of T1 scope, taken as-registered): the `validation` block (source: `resubmission/access_dryrun_val/`, `data/processed/baselines_v6_1_*`, `data/models/e28/e29_selection.json`), `model`, `registration`, and the dev-pool / block-B counts.
