# Reproducibility

Three levels, in increasing cost. The first needs nothing but this repository and takes a
minute; it is enough to confirm every number the letter prints.

## Level 1 — verify the reported numbers (no download)

```bash
pip install -r requirements.txt
python scripts/verify_register.py
```

Expected: `86/86 register values match; 5/5 file hashes match. VERDICT: PASS`.

The script recomputes each number in `results/number_register.json` from the deposited
per-event tables and compares it with the value the letter prints. It also re-hashes the five
locked artefacts. A mismatch anywhere is a failure, and the script says which entry failed.

You can check individual claims by hand from the same tables. For the primary test:

```python
import pandas as pd
from scipy.stats import binomtest
d = pd.read_csv("results/access_2025/access_20260828T123059Z_per_event.csv")
p = d[(d.evaluable) & (d.in_primary)][["MAE_M2_t6", "MAE_B6_t6"]].dropna()
w = (p.MAE_B6_t6 > p.MAE_M2_t6).sum(); l = (p.MAE_B6_t6 < p.MAE_M2_t6).sum()
print(w, l, binomtest(w, w + l, 0.5, alternative="greater").pvalue)   # 19 10 0.0680…
```

In the column names, `M2` is the PRT ensemble, `B6` the log-space gradient-boosting
comparator, `B1` flat persistence, and `M1` the archived earlier ensemble. The letter uses
descriptive names; the files keep the internal labels they were written with, because their
hashes are part of the registration.

## Level 2 — re-run the frozen scoring of the 2025 block

Needs the model-ready cube from Zenodo (about 17 MB).

```bash
# 1. fetch the data deposit: https://doi.org/10.5281/zenodo.22705927   (v3, 162 events)
# 2. build the directory layout the frozen scripts expect
python scripts/stage_repro_tree.py --zenodo-zip frp-prt-data-deposit-v3.zip
# 3. score
python scripts/test_access_v1.py --access resubmission/test_event_list_blockA_LOCKED.csv \
                                 --confirm-single-access
```

The script refuses to run unless its preconditions hold: the interpreter's `scikit-learn`,
`numpy`, `torch` and `h5py` match the pinned versions; the cube, its manifest and the ensemble
freeze table hash to their recorded values; the event list is row-identical to the frozen one;
and every checkpoint matches its recorded hash. Those gates are the point of the script, so do
not disable them — if one fails, the environment or a file differs, and the run is not the one
reported.

Two notes on what the run does. The comparator is retrained deterministically at scoring time
(`HistGradientBoostingRegressor`, `random_state=0`), so it is reproduced rather than loaded.
The script also scores the archived earlier ensemble for description; that requires the
`v1.0.0` source archive of this repository, which you can pass with `--m1-archive`. Without it
the script will stop at the M1 stage, after the primary and secondary tests have been written.

Expected primary outcome: 19 wins, 10 losses, *n* = 29, *p* = 0.0680 — the pre-declared
rejection region was ≥ 20 wins, so the test does not reject.

## Level 3 — retrain

```bash
python scripts/e28_train_grid.py      # the selection grid: 2 pools x 2 input scalings x 3 seeds
python scripts/e29_selection.py       # applies the three pre-declared selection rules
python scripts/e210_ensemble.py       # trains seeds 3-9 and builds the 10-seed ensemble
```

Training is CPU-only and took about 3.4 minutes per seed on the machine used (12 to 17 epochs
under early stopping). Seeds are set, but exact bit-level reproduction across different
PyTorch builds or hardware is not guaranteed; the ensemble median is stable across seeds, and
the per-seed validation results are in `models/prt_ensemble_m2/seed_*_training_record.json` for
comparison.

The hyperparameters were **not** tuned for this study: hidden width 96, 4 attention heads, 2
LSTM layers, dropout 0.1, AdamW at 1e-3 with weight decay 1e-4, OneCycle schedule, batch 64,
gradient clipping at 1.0, at most 30 epochs with early stopping after 5 without improvement on
the validation median skill. They were carried over from the earlier version of the study. The
grid varied only the training pool and the input scaling. The comparator, by contrast, received
a validation-only budget of four configurations. That asymmetry favours the comparator and is
stated in the letter.

## Rebuilding from raw products

`scripts/acquisition/` holds the download and aggregation pipeline. The LSA SAF credentials in
those files are replaced by placeholders; register for your own EUMETSAT account and substitute
them. See [`data_access.md`](data_access.md) for what the raw products are and where they come
from. Rebuilding the cube from scratch is the only path that does not rely on our derived
tables, and it is the slowest: the raw FRP-PIXEL archive for 162 events is tens of gigabytes.

## Environment

Pinned in `requirements.txt`; the versions asserted at access time were Python 3.11.5,
numpy 1.26.4, pandas 2.1.4, scikit-learn 1.2.2, scipy 1.17.1, torch 2.6.0+cpu, h5py 3.8.0
(`registration/access_environment_lock.txt`). scikit-learn's version matters more than the
others: the comparator is retrained at scoring time, and gradient-boosting implementations
change between releases.
