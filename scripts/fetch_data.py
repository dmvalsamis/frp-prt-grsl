"""Fetch the derived dataset (dataset_v3) from its Zenodo deposit into ``data/``.

The licensed-derived data layer (the model-ready cube, the manifest, the
normalization stats, and the per-event hourly tables) is hosted on Zenodo with
its own DOI rather than committed to this code repository. This script downloads
that record into the repo's ``data/`` directory so the training/evaluation
scripts can find it (or set ``FRP_DATA_DIR`` to wherever you place it).

Usage
-----
    python scripts/fetch_data.py
    # or override the record:
    FRP_ZENODO_RECORD=1234567 python scripts/fetch_data.py

No third-party dependencies (standard library only). Verifies MD5 checksums
reported by the Zenodo API when available.

After publishing the Zenodo deposit, set ZENODO_RECORD_ID below (or always pass
the FRP_ZENODO_RECORD env var) and update the DOI in README.md / data/README.md /
CITATION.cff.
"""
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

# <FILL: numeric Zenodo record id, e.g. "1234567">. The env var overrides this.
ZENODO_RECORD_ID = os.environ.get("FRP_ZENODO_RECORD", "<ZENODO_RECORD_ID>")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("FRP_DATA_DIR", REPO_ROOT / "data"))
API = "https://zenodo.org/api/records/{}"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if ZENODO_RECORD_ID.startswith("<"):
        sys.exit(
            "Zenodo record id not set. Publish the deposit, then either edit "
            "ZENODO_RECORD_ID in this file or run with FRP_ZENODO_RECORD=<id>."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "hourly").mkdir(exist_ok=True)

    print(f"[fetch_data] querying Zenodo record {ZENODO_RECORD_ID} ...")
    with urllib.request.urlopen(API.format(ZENODO_RECORD_ID)) as r:
        meta = json.load(r)

    files = meta.get("files", [])
    if not files:
        sys.exit("[fetch_data] record has no files (check the record id / DOI).")

    for f in files:
        key = f["key"]                       # e.g. dataset_v3.h5 or hourly/foo_hourly.csv
        url = f["links"].get("self") or f["links"].get("download")
        want_md5 = (f.get("checksum") or "").replace("md5:", "")
        dest = DATA_DIR / key
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists() and want_md5 and _md5(dest) == want_md5:
            print(f"[fetch_data] ok (cached): {key}")
            continue

        print(f"[fetch_data] downloading {key} ...")
        urllib.request.urlretrieve(url, dest)

        if want_md5:
            got = _md5(dest)
            if got != want_md5:
                sys.exit(f"[fetch_data] CHECKSUM MISMATCH for {key}: {got} != {want_md5}")

    print(f"[fetch_data] done -> {DATA_DIR}")
    print("[fetch_data] expected: dataset_v3.h5, dataset_v3_manifest.csv, "
          "normalization_params_v3.json, hourly/*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
