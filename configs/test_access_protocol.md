# Held-out test-set access protocol (SHA-256 anchored)

This project follows a pre-registered, cryptographically anchored protocol that
strictly limits how many times the held-out test split is touched. This is the
core integrity guarantee behind the reported numbers: the test set could not
have been used for model selection, because each access was pre-declared and
counted.

## The rule

- **The 4-event test split (`volos_2023`, `varnavas_2024`, `valmaior_2024`,
  `corinthos_2024`; 558 windows) is accessed exactly 3 times across the entire
  project.** No more.
- All model selection, hyperparameter choices, ablations, and calibration
  studies use the **validation split only** (5 events, 863 windows).
- In code, `src/shared/dataset_io.load_split('test')` **raises by default**. It
  only returns data when called with the explicit `unsealed=True` flag, and
  every such call corresponds to one of the three budgeted accesses below.

## The three accesses

| # | Configuration | What it produced | Result file |
|---|---|---|---|
| 1 | PRT-Base (C-FROZEN; paper hyperparameters, no SSAT/WindFeats) | deep-ensemble test SS, per-event, bootstrap CI | `results/prt_base_tuned_test.json` (`C_FROZEN`) |
| 2 | PRT-Tuned (C-HONEST; alt. hyperparameters, no SSAT/WindFeats) | deep-ensemble test SS, per-event, bootstrap CI | `results/prt_base_tuned_test.json` (`C_HONEST`) |
| 3 | PRT-Full (SSAT + WindFeats; deployed model) | deep-ensemble test SS, per-event, bootstrap CI | `results/prt_full_test_ensemble.json` |

The AR-Ridge baseline test reference (`results/ar_ridge_test_reference.json`,
`scripts/ar_ridge_test_reference.py`) is a **fixed linear baseline fit on the
training split only**; it tunes nothing against test and is therefore a
non-budget evaluation (it selects no model and cannot leak).

## The cryptographic anchor

Before the first test access, the full design of the deployed model
(architecture, loss, features, sampler, optimizer, seeds, ensembling, and the
test-evaluation procedure) was committed in writing, together with **SHA-256
hashes of every source module** that defines it and of the frozen WindFeats
statistics. Because the design was hashed *before* any test number was visible,
the design provably could not be reshaped in light of the test outcome.

Frozen WindFeats statistics SHA-256
(`configs/windfeats_stats.json`, canonical JSON, sorted keys, no whitespace):

```
b7d0fc8186d367d39b1b1072afa7cb444615d262fbee892a7630fa5a9ddf672a
```

Reproduce it:

```bash
python -c "import json,hashlib; d=json.load(open('configs/windfeats_stats.json')); \
c={k:d[k] for k in ('feature_names','means','stds')}; \
s=json.dumps(c,sort_keys=True,separators=(',',':')); \
print(hashlib.sha256(s.encode()).hexdigest())"
```

## No-leakage split

The 25/5/4 train/val/test split is **event-level**: every event contributes
windows to exactly one split, so no fire's history appears in more than one
partition. The split is stratified by intensity tier and wind regime; the four
most recent, highest-information events (2023–2024) are held out for test. It is
**not** a chronological or purely geographic cut (2023 events and Greek events
appear in all three splits). See `docs/event_selection.md`.
