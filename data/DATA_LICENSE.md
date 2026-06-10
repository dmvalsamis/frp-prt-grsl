# Data license & attribution

The **code** in this repository is MIT (see `../LICENSE`). The **derived dataset**
it consumes is distributed separately on Zenodo under **CC BY 4.0** and is built
from two openly licensed Earth-observation sources that must be acknowledged.

- **Derived dataset (Zenodo, CC BY 4.0):** `dataset_v3.h5`, the manifest, the
  normalization stats, and the per-event hourly tables. DOI:
  **<DATA DOI — fill after publishing the Zenodo record>**. Fetch with
  `python scripts/fetch_data.py`.
- **ERA5** — Copernicus Climate Change Service (C3S) / ECMWF. Open Copernicus
  Licence; redistribution of derived products permitted with attribution.
- **MSG-SEVIRI FRP-PIXEL** — EUMETSAT LSA SAF. **CC BY 4.0** (EUMETSAT "Core"
  data policy); redistribution and derived products permitted with attribution
  (https://lsa-saf.eumetsat.int/en/data/data-access/). Attribute as
  "EUMETSAT LSA SAF FRP-PIXEL".

Required acknowledgment for any use:

> Contains modified Copernicus Climate Change Service information (ERA5). Neither
> the European Commission nor ECMWF is responsible for any use of this
> information. Fire Radiative Power is derived from the EUMETSAT LSA SAF
> FRP-PIXEL product (MSG/SEVIRI).

Raw LSA SAF HDF5 granules and ERA5 NetCDF files are not redistributed; see
`README.md` (this directory) for acquisition.
