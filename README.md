# Sub-daily FRP forecasting over Mediterranean wildfires — reproduction package (2026 resubmission)

Code, pre-registration, model checkpoints and every per-event result behind the IEEE GRSL
letter *"Sub-Daily Fire Radiative Power Forecasting over Mediterranean Wildfires: Pre-Registered
Evaluation and Two-Season Replication of Persistence-Residual and Gradient-Boosting Models"*
(under review; the title may change in revision).

This is **version 2** of the package. Version 1 (`v1.0.0`, June 2026) belongs to an earlier
version of the study whose four-event evaluation was not independent; it is preserved
untouched on the `main` branch and under tag `v1.0.0`, because the pre-registration of this
study verifies one archived model against it. Nothing here overwrites it.

Derived data (the model-ready cube and the per-event hourly tables) is deposited separately:
**Zenodo concept DOI [10.5281/zenodo.20627568](https://doi.org/10.5281/zenodo.20627568)**,
which always resolves to the latest version. See [`docs/data_access.md`](docs/data_access.md).

---

## What the letter claims, and where you can check it

The central result is a **null**: the deep model did not beat the classical comparator on the
pre-registered test. Everything needed to confirm that, including the material that would
expose it if the null had been massaged, is in this repository.

| Claim in the letter | Evidence here | How to check |
|---|---|---|
| The 2025 test block was fixed and hashed **before** it was ever scored | `registration/test_event_list_blockA_LOCKED.csv`, listed by SHA-256 inside `registration/lock_bundle_prereg_v1.zip` and in the annotated tag `prereg-lock-v1` | `sha256sum registration/test_event_list_blockA_LOCKED.csv` → `b339f179…05e3`; the same value appears in the tag message and in `results/access_2025/access_record_20260828T123059Z.json` |
| The hypothesis, rejection region, statistics and stratifications were fixed before access | `registration/prereg_test_blocks_v1_LOCKED.md` §5, §6, §6a | read it; its SHA-256 is recorded in the tag message and in `registration/REGISTRATION_RECORD.md` |
| The scoring code was frozen before access and not edited afterwards | `scripts/test_access_v1.py` | `sha256sum scripts/test_access_v1.py` → `6b403afc…7b1c4`, the value in the tag message, in the access record, and re-verified by the two later exploratory runs |
| The block was accessed **once** | `results/access_2025/access_record_20260828T123059Z.json` and `access_console_20260828.log` | one record exists; the console log shows the environment pin, the input-hash gate and the hash-verified checkpoints |
| **Primary test does not reject.** PRT attains lower MAE than boosting on 19 of 29 events, *p* = 0.068, one short of the pre-declared region (≥ 20) | `results/access_2025/access_20260828T123059Z_results.json` | `PRIMARY_sign_test_M2_vs_B6_primary_set`; recomputable from the per-event table |
| Both models beat persistence: PRT 27 of 29 primary (*p* = 8.1 × 10⁻⁷), 41 of 45 evaluable; boosting 40 of 45 | same JSON and `..._per_event.csv` | `SECONDARY_sign_test_M2_vs_B1_primary_set` |
| Median skill on the 45 evaluable 2025 fires: PRT +0.349, boosting +0.342 | same files | `AGG1_headline` |
| Replication on 32 fires of 2026 (**exploratory**): PRT lower MAE on 15 of 32; medians +0.302 vs +0.318 | `results/exploratory_2026/` | `exploratory_blockB_20260908T140106Z_results.json` |
| Six alternative encoders under the identical framework do not separate: wins against boosting run 40/37 to 47/30 over the 77 fires, and every encoder beats persistence on at least 70 of 77 (**exploratory**) | `results/encoder_study/master_results.json`, `master_table.md` | per-encoder per-event tables are in the same folder |
| The encoder study reproduces the 2025 numbers exactly, so the re-scoring did not drift | `results/encoder_study/identity_check.json` | block-A MAE agrees with the locked table to ~1e-14 |
| Every number printed in the letter comes from these files | `results/number_register.json` | `python scripts/verify_register.py` → 86/86 values and 5/5 file hashes match |
| Test blocks are separated from the development pool by ignition year alone, and overlapping fires are removed from the primary statistic | `catalogue/event_catalogue_162_export_2026-09-11.csv`, `catalogue/pixel_disjointness_test_A_2025_2026-08-25*` | 84 events ignite ≤ 2024, 46 in 2025, 32 in 2026; the 14 removed events remain in the descriptive set |

A fuller map, claim by claim with the exact field names, is in
[`docs/claims_to_evidence.md`](docs/claims_to_evidence.md).

## Layout

```
registration/   the pre-registration as locked, its amendment, the lock bundle (zip),
                the frozen 2025 event list and input manifest, the environment pin,
                and REGISTRATION_RECORD.md (timeline, hashes, and what is and is not
                independently verifiable)
scripts/        test_access_v1.py      the frozen scoring script used for the single access
                verify_register.py     recomputes every number the letter prints
                e28_train_grid.py      PRT training (architecture, loss, optimizer, stopping)
                e29_selection.py       the validation-only selection
                e210_ensemble.py       the 10-seed ensemble
                build_dataset_v6.py    cube assembly
                frp_quality_gate.py    the five-criterion admission gate
                explore_*.py           the exploratory 2026 and encoder analyses
                acquisition/           download and hourly-aggregation pipeline (credentials
                                       replaced by placeholders — use your own EUMETSAT account)
results/        access_2025/           the single pre-registered access
                exploratory_2026/      the 2026 replication, labelled exploratory throughout
                encoder_study/         six encoders under the identical framework
                number_register.json   every number in the letter, with its source
catalogue/      the 162-event export, the frozen development pool, the seeded split, the
                gate results, the pixel-disjointness audit, the curation record, exclusions
models/         prt_ensemble_m2/       the 10 checkpoints loaded at the single access,
                                       their training records and the freeze hash table
figures/        the three figures of the letter and the scripts that draw them
docs/           reproducibility, claims-to-evidence, data access, integrity hashes
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/verify_register.py          # 86/86 values, 5/5 file hashes  → PASS
```

That check needs no data download: it recomputes the letter's numbers from the deposited
per-event tables. To re-run the model end to end, fetch the cube from Zenodo first —
see [`docs/reproducibility.md`](docs/reproducibility.md).

## Honest limits of this package

- **The registration is self-certified.** It was committed and tagged 62 seconds before the
  access, but the repository holding that tag was private at the time, so no third-party
  server timestamp predates the access. `registration/REGISTRATION_RECORD.md` states exactly
  what is and is not independently checkable, and does not dress the tag up as more than it is.
- **The 2026 block and the encoder comparison are exploratory.** They were examined after the
  2025 result was known. Every number from them carries that label here and in the letter, and
  none of them is promoted to a confirmatory claim.
- **Raw third-party products are not redistributed.** LSA SAF FRP-PIXEL and ERA5 files come
  from EUMETSAT and the Copernicus Climate Data Store; the acquisition scripts regenerate the
  derived tables from them.

## Citation

See `CITATION.cff`. Please cite the letter, and the Zenodo data deposit if you use the cube
or the hourly tables.

## Acknowledgements

Derived from the EUMETSAT LSA SAF FRP-PIXEL product (MSG/SEVIRI) and from ERA5/ERA5T
(Copernicus Climate Change Service). Neither EUMETSAT nor C3S is responsible for any use of
these derived products. Supported by the HORIZON EUROPE ECHO project (grant 101225575).
