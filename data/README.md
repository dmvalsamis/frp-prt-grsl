# Data — what is shipped, what you must acquire

## Shipped in this repo (safe to redistribute)

| File | What it is |
|---|---|
| `dataset_v3_manifest.csv` | The authoritative event→split→window mapping. 34 events, 6,631 windows: **25 train / 5 val / 4 test** (5,210 / 863 / 558 windows). One row per 48→12 h forecast window, with `event_name` and `split`. |
| `normalization_params_v3.json` | Train-split-derived per-feature means/standard deviations used to z-standardize inputs and to denormalize the persistence prior. These are aggregate statistics, not raw observations. |

## The derived cube — fetch from Zenodo (recommended)

The assembled model-input cube **`dataset_v3.h5`** and the 121 per-event hourly
tables are not committed to this git repo. They are published as a separate
**Zenodo deposit (CC BY 4.0)** with its own DOI:

**Data DOI: <DATA DOI — fill after publishing the Zenodo record>**

Fetch them into this directory (stdlib only, no extra deps):

```bash
python scripts/fetch_data.py            # downloads dataset_v3.h5 + hourly/ here
# or:  FRP_ZENODO_RECORD=<id> python scripts/fetch_data.py
```

Licensing and required source acknowledgments are in `data/DATA_LICENSE.md`.

## Rebuild from raw sources instead (no Zenodo needed)

If you prefer to regenerate the cube from scratch, obtain the licensed sources
below, build the per-event hourly tables, and place them under `data/`, then run
the assembly scripts. The raw granules themselves are never redistributed here.

### Source 1 — Fire Radiative Power (the forecast target)
- **Product:** MSG-SEVIRI **FRP-PIXEL**, EUMETSAT **LSA SAF**.
- **Obtain:** register for an LSA SAF account and download FRP-PIXEL HDF5
  granules for each event's space–time bounding box from the LSA SAF data
  service (https://landsaf.ipma.pt/). Credentials are personal; this repo ships
  no credentials and no download scripts that embed them.

### Source 2 — Meteorological covariates
- **Product:** **ERA5** hourly single-level and pressure-level fields
  (Copernicus Climate Change Service, C3S).
- **Obtain:** via the Copernicus Climate Data Store (CDS) API
  (https://cds.climate.copernicus.eu/). Acknowledge ECMWF/Copernicus per the
  Copernicus licence.

### Rebuilding the cube
Once raw FRP and ERA5 are downloaded and converted to per-event hourly tables
(`{event}_hourly.csv`), the assembly is reproducible with the shipped scripts:

```bash
python scripts/build_dataset.py      # assemble dataset_v3.h5 (LOOKBACK=48, HORIZON=12)
python scripts/readiness_check.py    # 7 readiness checks must all pass
```

> The event-onboarding/download stage (LSA SAF fetch, hourly-table construction)
> is intentionally **not** included here: those scripts embed personal LSA SAF
> credentials and operate on licensed raw granules. They are not required to use
> the model — only to regenerate the cube from scratch.

## The reproducible data-quality gate (event inclusion)

Every event in the dataset had to pass a fixed quality gate, applied identically
to every candidate. This is the **reproducible** part of event selection
(implemented in `scripts/data_quality_gate.py`):

| Criterion | Threshold |
|---|---|
| Fire-active hourly slots | ≥ 15 |
| Duration | ≥ 5 days |
| Complete diurnal days (a day with > 50 % of its 15-min slots present) | ≥ 3 |
| Hourly missing-slot (gap) rate | < 0.40 |
| Mid-infrared saturation rate | < 0.30 |

All 34 retained events pass, and the gate binds at the margin (e.g.
`corinthos_2024` = exactly 15 fire slots; `varnavas_2024` = exactly 5 days).

**Honest scope note.** This quality gate is reproducible and coded, but it does
**not** by itself reproduce the 34-event set. Which fires entered the candidate
pool was a manual curation (documented major Mediterranean fires plus targeted
gap-filling), the low-intensity exclusion was a single ad hoc hand-exclusion
(one catalogued event at ~874 MW; no general minimum-FRP threshold is coded),
and bounding boxes were drawn by hand. See `docs/event_selection.md` for the
full, honest account.
