"""Fetch the derived dataset (dataset_v3) from its Zenodo deposit into ``data/``.

The licensed-derived data layer (the model-ready cube, the manifest, the
normalization stats, and the per-event hourly tables) is published on Zenodo
(CC BY 4.0) with its own DOI rather than committed to this code repository.
This script downloads the Zenodo deposit archive and extracts it into the repo's
``data/`` directory so the training/evaluation scripts can find it (or set
``FRP_DATA_DIR`` to wherever you place it).

Usage
-----
    python scripts/fetch_data.py
    # override the record (e.g. to pin an exact version):
    FRP_ZENODO_RECORD=<recid> python scripts/fetch_data.py

The default record id is the Zenodo CONCEPT id (DOI 10.5281/zenodo.20627568),
which always resolves to the latest published version. Standard library only.
"""
import io
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

# Concept (all-versions) Zenodo record id — always resolves to the latest version.
# Override with the FRP_ZENODO_RECORD env var to pin a specific version.
ZENODO_RECORD_ID = os.environ.get("FRP_ZENODO_RECORD", "20627568")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("FRP_DATA_DIR", REPO_ROOT / "data"))
API = "https://zenodo.org/api/records/{}"

# Deposit-level docs that must NOT overwrite the repo's own data/ docs.
_SKIP = {"README.md", "LICENSE_AND_ATTRIBUTION.md"}


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[fetch_data] querying Zenodo record {ZENODO_RECORD_ID} ...")
    with urllib.request.urlopen(API.format(ZENODO_RECORD_ID)) as r:
        meta = json.load(r)

    files = meta.get("files", [])
    if not files:
        sys.exit("[fetch_data] record has no files (check the DOI / record id).")

    # The deposit is a single archive; fall back to the first file if not named .zip.
    entry = next((f for f in files if str(f.get("key", "")).endswith(".zip")), files[0])
    key = entry["key"]
    url = entry["links"].get("self") or entry["links"].get("download")

    print(f"[fetch_data] downloading {key} ({entry.get('size', '?')} bytes) ...")
    with urllib.request.urlopen(url) as resp:
        blob = resp.read()

    print(f"[fetch_data] extracting into {DATA_DIR} ...")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        members = [m for m in z.namelist()
                   if not m.endswith("/") and os.path.basename(m) not in _SKIP]
        z.extractall(DATA_DIR, members=members)

    cube = DATA_DIR / "dataset_v3.h5"
    if not cube.exists():
        sys.exit(f"[fetch_data] ERROR: dataset_v3.h5 not found after extraction in {DATA_DIR}")

    n_hourly = len(list((DATA_DIR / "hourly").glob("*_hourly.csv"))) if (DATA_DIR / "hourly").exists() else 0
    print(f"[fetch_data] done -> {DATA_DIR}")
    print(f"[fetch_data]   dataset_v3.h5, dataset_v3_manifest.csv, "
          f"normalization_params_v3.json, hourly/ ({n_hourly} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
