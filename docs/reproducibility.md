# Reproducibility

This document maps every figure, table, and headline number in the paper to the
exact script and command that produces it, and records the integrity protocol
(SHA-256-anchored test access, event-level no-leakage split).

## 0. Environment

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.11
pip install -r requirements.txt
# Point the loader at the directory holding dataset_v3.h5 (see data/README.md):
export FRP_DATA_DIR=/path/to/data   # not needed for figure regeneration
```

The paper run was **CPU-only** (torch 2.6.0+cpu, 16 cores). No GPU is required.
Determinism: `src/shared/seed.py::set_all_seeds` fixes `PYTHONHASHSEED`,
`random`, `numpy`, `torch`, and forces `cudnn.deterministic=True`,
`cudnn.benchmark=False`. The 10 ensemble seeds are `0..9` (`configs/seeds.txt`).

## 1. What runs WITHOUT the licensed data

These reproduce from artifacts shipped in the repo — no `dataset_v3.h5` needed,
and **no test-set access**:

| Output | Command | Reads |
|---|---|---|
| **Fig. 3** (forest plot) | `python scripts/make_forest_plot.py` | `figures/forest_plot_trajectories.json` (cache) |
| Conformal calibration table | `python scripts/conformal_calibration.py` | shipped val predictions in `results/partA/` **+** the val split of `dataset_v3.h5` (val only) |
| WindFeats SHA-256 check | see `configs/test_access_protocol.md` | `configs/windfeats_stats.json` |
| Architecture introspection (param counts, gate, dims) | `python scripts/ar_ridge_test_reference.py` is data-dependent; param counts come from `src/shared/prt_model.py` loaded against `models/prt_full_ensemble/seed_000.pt` | checkpoints |

> Note: `conformal_calibration.py` needs the **val** targets, so it requires
> `dataset_v3.h5` to be present (val split only); it never loads test.

## 2. What requires the licensed cube (`dataset_v3.h5`)

| Output | Command | Test access? |
|---|---|---|
| Assemble the cube | `python scripts/build_dataset.py` then `python scripts/readiness_check.py` | no |
| Train the PRT-Full 10-seed ensemble | `python scripts/train_prt_full.py` | no (val-only selection) |
| Held-out test evaluation (PRT-Full) | `python scripts/eval_test.py` | **yes — budgeted access #3** |
| AR-Ridge baseline test reference | `python scripts/ar_ridge_test_reference.py` | non-budget (fit on train only) |
| Backbone comparison ensembles (val) | `python scripts/backbone_trainer.py` | no |
| AR-Ridge validation baseline | `python scripts/baselines_ar_ridge.py` | no |

The PRT-Base/Tuned/Full bootstrap CIs are already committed in the result JSONs
(`bootstrap_95CI` / `bootstrap_95_CI` fields), computed event-level,
`n_boot=10000`, `np.random.default_rng(42)`.

`eval_test.py` must be called with the test seal explicitly bypassed and counts
as one of the **three** budgeted test accesses (`configs/test_access_protocol.md`).

## 3. Claim → code → result map

| Paper claim | Value | Produced by | Committed result |
|---|---|---|---|
| PRT-Base deep-ensemble test SS_fire,t6 | **+0.1969** (95% CI [+0.1115, +0.2905]) | budgeted test access #1 | `results/prt_base_tuned_test.json` → `C_FROZEN` |
| PRT-Tuned deep-ensemble test SS_fire,t6 | **+0.2402** (95% CI [+0.1535, +0.3451]) | budgeted test access #2 | `results/prt_base_tuned_test.json` → `C_HONEST` |
| **PRT-Full** deep-ensemble test SS_fire,t6 (deployed) | **+0.2998** (95% CI [+0.2074, +0.4285]) | `scripts/eval_test.py` (access #3) | `results/prt_full_test_ensemble.json` |
| Per-event PRT-Full test SS (Table I) | Volos +0.4986 (89); Varnavas +0.2856 (7); Val Maior +0.1964 (69); Corinth +0.2184 (21) | `scripts/eval_test.py` | `results/prt_full_test_ensemble.json` `per_event_SS` |
| PRT trainable parameters | **265,165** (base 32-ch); 265,645 deployed (37-ch) | `src/shared/prt_model.py` | Fig. 2 / Table II |
| Component ablation (val): SSAT / WindFeats / balanced sampler | SSAT ≈ +0.030 mean; WindFeats variance-reducer (~1.7×) | `scripts/train_prt_full.py` variants + aggregation | `results/ablation_validation.csv` |
| Backbone comparison (val, 10-seed ensembles) | gru_skip +0.2665; lstm_skip +0.2354; mlp_mixer +0.2366; cnn_skip +0.2059; prt +0.2548 | `scripts/backbone_trainer.py` | `results/backbone_validation.csv` |
| Conformal coverage (val) vs inter-seed Q25–Q75 | PRT-Full 0.5070 / 0.7996 / 0.8800 vs 0.2207 | `scripts/conformal_calibration.py` | `results/calibration_validation.csv` |
| AR-Ridge test reference (§IV-A) | event-mean SS = **−0.5764** (PRT-Full margin +0.8762) | `scripts/ar_ridge_test_reference.py` | `results/ar_ridge_test_reference.json` |
| Fig. 1 geographic map | — | QGIS (not script-generated) | `figures/geographic_distribution_map.png` |
| Fig. 2 architecture | — | hand-drawn from the spec in `configs/prt_full.yaml` / `src/shared/prt_model.py` | `figures/PRT_architecture.png` |
| Fig. 3 forest plot | per-event trajectories | `scripts/make_forest_plot.py` | `figures/forest_plot.png` |

> **AR-Ridge sign (honesty note).** The AR-Ridge baseline's event-mean test SS is
> **negative** (−0.5764), driven by the 7-fire-hour Varnavas event where the
> train-fitted linear model overpredicts a small fire. Excluding Varnavas it
> reaches only +0.0086 (≈ persistence). The baseline is at best persistence-level
> on this test set; it should not be described as a "competent baseline."

## 4. Ensembling rule (locked)

The deployed predictor is a deep ensemble over the 10 seeds:

```
per-seed pred -> clip(>=0) -> log1p -> mean over seeds -> expm1 -> clip(>=0)
```

Identical for val and test; implemented in `scripts/eval_test.py` and
`scripts/conformal_calibration.py`.

## 5. Test-access protocol & no-leakage split

The held-out test split is touched **exactly 3 times** in the whole project,
each pre-declared, under a SHA-256 anchor that proves the deployed model's design
was fixed before any test number was visible. The full **SHA-256 hash table**
(the manuscript's supplementary integrity anchor) is in
[`docs/integrity_hashes.md`](integrity_hashes.md) — 8/9 shared modules reproduce
the anchor byte-for-byte; `dataset_io.py` differs only by a repo-relative
path-resolution change, disclosed there. The 25/5/4 split is **event-level**
(no event spans two splits), stratified by intensity tier and wind regime. Full
detail: `configs/test_access_protocol.md`; selection honesty: `docs/event_selection.md`.

## 6. Calibration caveat (binding)

Inter-seed Q25–Q75 spread is **not** a predictive interval — its empirical
coverage on fire-active t+6 is ~0.22 against a ~0.50 target, i.e. the ensemble is
overconfident. Conformal calibration on validation recovers nominal coverage and
is the recommended path to calibrated uncertainty (`scripts/conformal_calibration.py`).
