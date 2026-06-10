# Integrity hash table (SHA-256 anchor)

This is the supplementary **SHA-256 hash table** referenced in the manuscript
(§III-D and the code-availability note). It is the cryptographic anchor proving
that the deployed model's complete implementation was fixed **before** the
held-out test set was accessed — so the design provably could not have been
reshaped in light of the test result.

## What was hashed, and when

The hashes below were computed on **2026-05-19**, before the single PRT-Full
test-set access. They cover every source module that defines the deployed
(Row-2 / PRT-Full) design, plus the frozen WindFeats normalization statistics.

| Module (`src/shared/…`) | SHA-256 anchor (pre-test, 2026-05-19) |
|---|---|
| `prt_model.py` | `0b06edee0ce9cd3e84bb411196ac829627d7dbb1571e5faefabcbfe41a4b5463` |
| `ssat_loss.py` | `3d821d4ea575b35ecad74de14fdc10dad64855eb3c7c797ed59e523e7918dfac` |
| `wind_features.py` | `3b6aaaff21865e3f219a94624bdae6783b6b5a403c27c2a0e7075e36de1f8b7d` |
| `event_balanced_sampler.py` | `75063a78d834662268f679af20c5b7043ede2af12f5dedcd5d82b2ce7e599fd0` |
| `persistence_mae_table.py` | `d356bf6aa3ae903ba0d3f68e0634b9948a92929e5f32fe3af57384df7339f769` |
| `eval_harness.py` | `98be39b178685178347c777df3c81047cb12c20d3c41f3a724227d2bdce054c7` |
| `seed.py` | `3daf3ae7aba1a6c958143628e41fdb1836409114f7f6c6e94438986301423dac` |
| `train_ssat.py` | `5fb83e1d38ee11f1d53b4cc2a8890612f7cfa6b1a72280745b45bedfb02521e5` |
| `dataset_io.py` | `e542784bd90774fbef241dad561e9317e6e661c175d220ec46fff0d2800f6b6b` |

WindFeats statistics (canonical JSON of `{feature_names, means, stds}`, sorted
keys, no whitespace — `configs/windfeats_stats.json`):

```
b7d0fc8186d367d39b1b1072afa7cb444615d262fbee892a7630fa5a9ddf672a
```

## Public-release verification (honest disclosure)

The files shipped in this repository reproduce the anchor **8 of 9 modules
byte-for-byte**. One file was modified for the public release, and only
cosmetically:

| Module | Public-release file matches anchor? | Note |
|---|---|---|
| `prt_model.py`, `ssat_loss.py`, `wind_features.py`, `event_balanced_sampler.py`, `persistence_mae_table.py`, `eval_harness.py`, `seed.py`, `train_ssat.py` | **Yes — byte-for-byte** | unchanged from the locked workspace |
| `dataset_io.py` | **No — intentionally modified** | the only change is **path resolution**: the original hard-coded the author's absolute data path; the public copy resolves the data directory from the `FRP_DATA_DIR` environment variable (falling back to the repo's `data/`). The data-loading logic and the test-seal behaviour are unchanged. |

Public-release `dataset_io.py` SHA-256:
```
67374ff604b5f944a3943868c596319672eeb8d6adea8ff104fdba30f2d1a04b
```
The original anchored `dataset_io.py` (`e542784…`) is preserved unmodified in the
project's locked workspace as the integrity record.

## Reproduce the check

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
for f in sorted(Path("src/shared").glob("*.py")):
    print(hashlib.sha256(f.read_bytes()).hexdigest(), f.name)
d = json.load(open("configs/windfeats_stats.json"))
c = {k: d[k] for k in ("feature_names","means","stds")}
s = json.dumps(c, sort_keys=True, separators=(",",":"))
print(hashlib.sha256(s.encode()).hexdigest(), "windfeats_stats(canonical)")
PY
```

Expect 8/9 module hashes to match the anchor table above, with `dataset_io.py`
showing the disclosed public-release hash and the WindFeats canonical hash
matching exactly. (`prt_arg_model.py` is the negative-result ARG variant from
§IV-C; it is not part of the deployed design and is not in the anchor table.)
