# Data access and licensing

## What is here, and what is on Zenodo

This repository holds code, the pre-registration, the model checkpoints and every per-event
result table. It does **not** hold the model-ready cube or the per-event hourly tables, which
are deposited on Zenodo:

**Concept DOI: [10.5281/zenodo.20627568](https://doi.org/10.5281/zenodo.20627568)** — always
resolves to the latest version. The version for this study contains:

```
cube/dataset_v6_1.h5                     model-ready cube: 48 h lookback, 12 h horizon,
                                         32 channels; train (71 events) and val (13 events)
cube/dataset_v6_1_manifest.csv           event-to-split manifest and window accounting
cube/normalization_params_v6_1.json      train-derived normalization statistics
hourly_frp/<event>_hourly.csv            per-event hourly area-aggregate FRP (MW), 162 events
hourly_era5/<event>_era5_hourly.csv      per-event hourly ERA5 covariates, 162 events
metadata/                                catalogue export, frozen pool, split, block lists
```

The two test blocks are deliberately **not** in the cube. They are assembled at evaluation time
from the hourly tables by `scripts/test_access_v1.py`, which is how the sequestration was kept
during the study.

The earlier version of this study deposited a 34-event cube (`dataset_v3.h5`) under the same
concept DOI. That version is superseded for the present letter but remains retrievable.

## Provenance of the underlying products

**Fire radiative power.** EUMETSAT LSA SAF FRP-PIXEL product from SEVIRI on Meteosat Second
Generation. Native 15-minute retrievals are summed over all detected fire pixels inside each
event's bounding box and aggregated to clock hours. Hours missing more than half their slots
are flagged; pixels saturated in the mid-infrared channel are kept at their reported, and
therefore underestimated, values, with the saturated fraction carried as a feature.

**Meteorology.** ERA5 and ERA5T from the Copernicus Climate Change Service, at 0.25° and hourly
resolution, reduced per event to a latitude-cosine-weighted area mean over the same box. Which
product applies depends on the block: the 2025 block is ERA5 final throughout, retrieved in
August 2026, around a year after the season. The 2026 block is **ERA5T**, the preliminary
release, for 31 of its 32 events — every window from late June 2026 onward — because those
windows fall inside ERA5's two-to-three-month finalisation lag. The single exception is a
February 2026 event, which is final. The letter states this; it matters because ERA5T values
can be revised.

Note that ERA5's latency also bounds what the reported skill means operationally: the
covariates are reanalysis, not forecast, so the skill is a ceiling under perfect meteorological
knowledge rather than an operational figure.

**Event footprints.** Burnt-area episodes from the European Forest Fire Information System,
merged by a declared union-find rule and buffered by 0.25° on the ERA5 grid.

## Why the raw products are not redistributed here

The raw LSA SAF HDF5 products and the raw ERA5 NetCDF files are third-party data under their
own terms, and redistribution is not ours to grant. `scripts/acquisition/` contains the
download and aggregation pipeline that regenerates the derived tables from them:

- **LSA SAF**: register at <https://landsaf.ipma.pt/> for an account, then put your own
  credentials in place of the `<<LSASAF_USERNAME_REDACTED>>` and
  `<<LSASAF_PASSWORD_REDACTED>>` placeholders in `02_download_all.py` and
  `_add_event_worker.py`. Do not commit them.
- **ERA5**: register for a Copernicus Climate Data Store account and configure `cdsapi`;
  `06_era5_download.py` handles the request, the ERA5/ERA5T stream merge and the quality
  checks.

The raw FRP-PIXEL archive for 162 events runs to tens of gigabytes, which is the other reason
the derived hourly tables are the practical starting point.

## Licence

Code in this repository: MIT (see `LICENSE`).

Derived data on Zenodo — the cube, the hourly tables and the catalogue metadata: CC BY 4.0.

Attribution required by the upstream providers: the FRP data are derived from the EUMETSAT
LSA SAF FRP-PIXEL product; the meteorological covariates are derived from ERA5/ERA5T, produced
by the Copernicus Climate Change Service. Neither EUMETSAT nor C3S is responsible for any use
of these derived products, and neither endorses this work.
