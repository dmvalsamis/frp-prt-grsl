# PRT — Persistence-Residual Transformer for sub-daily FRP trajectory forecasting

Reference implementation, deployed model weights, and reproducibility materials
for the IEEE GRSL letter on **sub-daily Fire Radiative Power (FRP) trajectory
forecasting over Mediterranean wildfires**.

The model (**PRT**, Persistence-Residual Transformer) forecasts the next 12 hours
of FRP from a 48-hour history of MSG-SEVIRI FRP and ERA5 covariates. It learns a
**log-space residual on top of a persistence prior**, is trained with a
**Skill-Score-Aligned Training (SSAT)** objective, and uses five engineered
**WindFeats**. The headline result is reproducibility-first: a 10-seed deep
ensemble whose held-out skill has a bootstrap confidence interval that excludes
zero.

> **Honesty-first project.** This repo is explicit about what is reproducible vs.
> manual, reports the held-out test numbers as-is (including a baseline whose test
> skill is *negative*), and documents a SHA-256-anchored protocol limiting
> test-set access to exactly three pre-declared evaluations.

## Headline numbers (held-out test, deep ensemble, SS_fire,t6)

| Model | Test SS_fire,t6 | 95% CI | Source |
|---|---|---|---|
| PRT-Base (paper hyperparameters) | **+0.1969** | [+0.1115, +0.2905] | `results/prt_base_tuned_test.json` |
| PRT-Tuned | **+0.2402** | [+0.1535, +0.3451] | `results/prt_base_tuned_test.json` |
| **PRT-Full** (deployed) | **+0.2998** | [+0.2074, +0.4285] | `results/prt_full_test_ensemble.json` |

`SS_fire,t6 = 1 − MAE_model / MAE_persistence` at t+6 on fire-active samples,
event-level unweighted mean across events. PRT = **265,165** trainable parameters
(base 32-channel; 265,645 deployed with WindFeats).

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.11
pip install -r requirements.txt
```

CPU-only is sufficient (the paper run used no GPU).

## Reproduce

**Figure 3 (no data needed, no test access):**
```bash
python scripts/make_forest_plot.py     # regenerates figures/forest_plot.png from the shipped cache
```

**Everything else** (training, test evaluation, baselines) needs the derived
input cube `dataset_v3.h5`, published separately on **Zenodo (CC BY 4.0)** so its
license and DOI stay distinct from this MIT code:

```bash
python scripts/fetch_data.py           # pull dataset_v3.h5 + hourly tables into data/
python scripts/train_prt_full.py       # 10-seed PRT-Full ensemble (val-only selection)
python scripts/eval_test.py            # held-out test eval (budgeted access #3)
```

The cube is built from LSA SAF FRP-PIXEL + ERA5; you can also rebuild it from
those raw sources instead of fetching — see [`data/README.md`](data/README.md)
and [`data/DATA_LICENSE.md`](data/DATA_LICENSE.md).

A full figure/table/number → script map is in
[`docs/reproducibility.md`](docs/reproducibility.md).

## Repository layout

```
src/shared/      PRT model, SSAT loss, WindFeats, event-balanced sampler,
                 dataset loader, eval harness (skill score), seeds
scripts/         dataset build + reproducible data-quality gate, training,
                 held-out evaluation, AR-Ridge baseline, conformal calibration,
                 figure generation
configs/         frozen hyperparameters (prt_full.yaml), seeds, WindFeats stats,
                 SHA-256 test-access protocol
data/            dataset_v3_manifest.csv + normalization stats + acquisition guide
                 (the licensed dataset_v3.h5 is NOT shipped)
results/         per-seed + held-out test result JSON/CSV; validation ablation,
                 backbone, and calibration tables
figures/         the three paper figures + the forest-plot regeneration cache
models/          the deployed PRT-Full 10-seed checkpoints (prt_full_ensemble/)
docs/            reproducibility map; SHA-256 integrity hash table; honest
                 event-selection provenance
```

## Data availability

- **Code & trained weights:** this repository (MIT).
- **FRP target:** MSG-SEVIRI **FRP-PIXEL**, EUMETSAT **LSA SAF** — obtain from
  the LSA SAF data service (account required).
- **Meteorology:** **ERA5** (Copernicus C3S) — obtain from the Copernicus
  Climate Data Store. Acknowledge ECMWF/Copernicus.
- **Derived model-ready cube:** published on **Zenodo (CC BY 4.0)**, DOI
  **`10.5281/zenodo.20627568`** (concept DOI, always-latest); fetch with `python scripts/fetch_data.py`.
- Raw granules are **not** redistributed. See [`data/README.md`](data/README.md)
  and [`data/DATA_LICENSE.md`](data/DATA_LICENSE.md).

## How events were selected (reproducible vs. manual)

A fixed, coded **data-quality gate** (≥15 fire-active hourly slots, ≥5-day
duration, ≥3 complete diurnal days, gap rate < 40 %, saturation < 30 %) is the
reproducible inclusion rule and all 34 events pass it at the margin. But the
**candidate pool was hand-curated**, the **intensity floor was a single ad hoc
exclusion** (~874 MW; no coded threshold), and **bounding boxes were drawn by
hand** — so selection is *not* an automated database funnel. The 25/5/4 split is
event-level, stratified by intensity tier and wind regime (not chronological or
geographic). Full account: [`docs/event_selection.md`](docs/event_selection.md).

## Citation

See [`CITATION.cff`](CITATION.cff) (fill in author/ORCID and the paper DOI on
acceptance).

## License

MIT for the source code and trained weights (see [`LICENSE`](LICENSE)). The
underlying LSA SAF and Copernicus/ERA5 data products retain their own licenses.
