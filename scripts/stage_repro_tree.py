"""Stage this repository into the directory layout the analysis scripts expect.

The frozen scripts (`test_access_v1.py`, `verify_register.py`) were written against the
working tree of the study and address files by their paths there — `data/processed/`,
`data/models/e28/`, `resubmission/`. They are published byte-for-byte as they ran, because
their SHA-256 is part of the pre-registration, so the paths cannot be edited. This helper
builds the expected tree around them instead, from the files in this repository plus the
Zenodo data deposit.

    python scripts/stage_repro_tree.py --zenodo-zip frp-prt-data-deposit-v3.zip

After it runs:

    python scripts/verify_register.py
    python scripts/test_access_v1.py --access resubmission/test_event_list_blockA_LOCKED.csv \
                                     --confirm-single-access

Nothing is overwritten: the script refuses to replace an existing file whose content differs.
"""
import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def place(src: Path, dst: Path, quiet=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if sha(src) == sha(dst):
            return "identical"
        sys.exit(f"refusing to overwrite an existing, different file: {dst}")
    shutil.copy2(src, dst)
    if not quiet:
        print(f"  {dst.relative_to(ROOT)}")
    return "copied"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zenodo-zip", help="frp-prt-data-deposit-v3.zip from "
                                         "https://doi.org/10.5281/zenodo.22705927")
    ap.add_argument("--m1-archive", help="optional: the v1.0.0 source archive of this repository, "
                                         "needed only because the access script also scores the "
                                         "archived earlier ensemble (descriptive; no test uses it)")
    args = ap.parse_args()

    print("staging registration files into resubmission/ ...")
    for name in ("test_event_list_blockA_LOCKED.csv", "blockA_input_manifest.csv",
                 "access_environment_lock.txt", "prereg_test_blocks_v1_LOCKED.md"):
        src = ROOT / "registration" / name
        if src.exists():
            place(src, ROOT / "resubmission" / name)
    for name, sub in (("number_register.json", "results"),):
        src = ROOT / sub / name
        if src.exists():
            place(src, ROOT / "resubmission" / name)
    for sub, dest in (("access_2025", "TEST_ACCESS_RESULTS"),):
        d = ROOT / "results" / sub
        if d.is_dir():
            for f in sorted(d.iterdir()):
                place(f, ROOT / "resubmission" / dest / f.name, quiet=True)
            print(f"  resubmission/{dest}/ ({len(list(d.iterdir()))} files)")

    print("staging the PRT ensemble into data/models/e28/ ...")
    src_dir = ROOT / "models" / "prt_ensemble_m2"
    for s in range(10):
        ck = src_dir / f"seed_{s}.pt"
        if ck.exists():
            place(ck, ROOT / "data" / "models" / "e28" / f"T1_robust_s{s}_best.pt", quiet=True)
        rec = src_dir / f"seed_{s}_training_record.json"
        if rec.exists():
            place(rec, ROOT / "data" / "models" / "e28" / f"T1_robust_s{s}_result.json", quiet=True)
    for name in ("m2_freeze_hashes.json", "m2_ensemble_val_results.json",
                 "e28_grid_summary.json", "e29_selection.json"):
        p = src_dir / name
        if p.exists():
            place(p, ROOT / "data" / "models" / "e28" / name, quiet=True)
    print("  data/models/e28/ (10 checkpoints + records)")

    if args.zenodo_zip:
        z = Path(args.zenodo_zip)
        if not z.exists():
            sys.exit(f"not found: {z}")
        print(f"extracting the data deposit from {z.name} ...")
        proc = ROOT / "data" / "processed"
        proc.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():
                if m.endswith("/"):
                    continue
                name = Path(m).name
                if m.startswith("cube/"):
                    target = proc / name
                elif m.startswith(("hourly_frp/", "hourly_era5/")):
                    target = proc / name
                elif m.startswith("metadata/"):
                    target = ROOT / "resubmission" / name
                else:
                    continue
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(m) as fh, open(target, "wb") as out:
                    shutil.copyfileobj(fh, out)
        print(f"  data/processed/ (cube + per-event hourly tables)")
    else:
        print("no --zenodo-zip given: verify_register.py will still run; "
              "test_access_v1.py needs the cube.")

    if args.m1_archive:
        a = Path(args.m1_archive)
        dest = ROOT / "resubmission" / "m1_deposit"
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(a) as zf:
            zf.extractall(dest)
        print(f"  resubmission/m1_deposit/ (from {a.name})")

    print("\ndone. Next:")
    print("  python scripts/verify_register.py")
    if args.zenodo_zip:
        print("  python scripts/test_access_v1.py --access resubmission/test_event_list_blockA_LOCKED.csv "
              "--confirm-single-access")


if __name__ == "__main__":
    main()
