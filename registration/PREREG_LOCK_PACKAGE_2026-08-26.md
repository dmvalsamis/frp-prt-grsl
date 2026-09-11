# Pre-Registration Lock Package — GRSL/JSTARS Resubmission
**Assembled:** 2026-08-26 (per adjudication memo A4) · **Status: SELF-CERTIFIED
LOCK IN PREPARATION (G6′, per memo IA-1 2026-08-28)** — the supervisor declined
sign-off participation (logged); the lock takes force at the Step-4 public
registration (Zenodo DOI canonical + git tag), whose identifiers are entered in
the §7 registration record. Block B finalizes afterward under procedures named
inside the lock (ACC-1.2 + IA-B2 cutoff), published as the REQUIRED
season-close deposit before any access.

---

## READ FIRST — The four sensitive disclosures

These are the four decisions a hostile reviewer will probe. Each is disclosed in
full in the prereg; each has a written rationale and a paper trail in
RESEARCH_LOG.md. None is hidden in an appendix.

**1. DEC-1 — D7-restored test events (timing + design).** Three block-A events
were restored by the D7 conflict-rule refinement (edge-touch ≠ conflict),
declared after the old rule's exclusions were observable in the metadata-only
sub-division output, before any FRP for the affected events was downloaded and
before block-A gate screening ran. Design response: the restored events are
EXCLUDED from the primary sign test (n = 30); the two that survive the DEC-3
audit (es_rua_a, pt_trancoso_sao_pedro — the season's two most intense fires)
form the labeled sensitivity increment (n = 32 = 30 + 2; the third, pt_real,
was dropped by the DEC-3 mechanical rule and appears in the descriptive table
only). The primary statistic never depends on a rule refined after its effect
was observable. (Prereg §6a; RESEARCH_LOG 2026-08-25.)

**2. W-1 — window-rule correction (P-1/P-2).** The original onboarding windows
truncated multi-polygon episodes (worst: pt_reriz, 99% of its fire record
outside the window). W-1 redefined windows (earliest−1d … latest FIREDATE+9d,
30-day cap) and re-downloaded 41 dev/test events BEFORE any training on v6.1;
over-cap chains keep original windows, disclosed (P-1); the rule is uniform
across all pools (P-2). Val split byte-identical before/after. (Prereg §3.2;
RESEARCH_LOG 2026-08-25/26.)

**3. DEC-3 v1 invalidation.** The first run of the pixel-disjointness audit was
INVALID (missing per-event bbox filter against full-disk LSA SAF pixel lists —
every overlap fraction spuriously ≈0). The defect was identified from
coordinate-count patterns alone (no FRP content), v1 output preserved, v2
re-hashed BEFORE its run, rule unchanged. v2 reversed the conclusion: 21/31
block-A pairs dependent → 15 drops (14 gate-passers + 1 non-gate-passer
no-op); 46 − 14 = 32 survivors = 30 primary + 2 d7_restored. An invalid audit that had
claimed full independence was caught and disclosed, not papered over. (Prereg
§9.5; RESEARCH_LOG 2026-08-25.)

**4. B-2 — post-freeze baseline extension (conservativity).** The log-space
baseline variants were run AFTER the M2 freeze (2026-08-26T09:39:40Z), on
adjudication order A4/B-2. Ordering is conservative by construction: the frozen
model cannot react to a stronger baseline; the extension can only raise the bar
M2 must clear. It DID raise it: **B6 renamed from ar_ridge_T0 (+0.1703) to
gbdt_log_T1 (+0.3050)** under the pre-declared renaming rule, and every
statistic below is stated against the stronger bar. (Prereg §5; RESEARCH_LOG
2026-08-26.)

---

## 1. One-page chronology

| Date (2026) | Event |
|---|---|
| 08-18 | GRSL-01633-2026 REJECTED (R2+AE: test-set independence; tautological bootstrap; weak baseline) |
| 08-24 | Resubmission program declared. **Split rule pre-declared before any 2025/2026 gate results**: dev = ignition ≤2024; test block A = 2025; block B = 2026. Frozen 5-part gate (`frp_quality_gate.py`, hash below) |
| 08-24 | Task E1: 64 gate-passing sequestered test events onboarded (metadata/gate/ERA5 only — no predictions, no skill statistics) |
| 08-25 | D4 mega-episode sub-division; D7 conflict refinement; dev-pool wave (+36 events); **dev pool FROZEN at 84 events; split FINAL (seed 20260825, declared seed+1 remediation): 71 train / 13 val** |
| 08-25 | DEC-1 (d7_restored: primary-exclude/sensitivity-include); DEC-3 pixel-disjointness audit (v1 invalid → v2: n = 30/32); W-1 window correction (41 events re-downloaded, re-gated) |
| 08-25 | dataset_v6 → v6.1 built (16,079 samples; RAW inputs; 4 normalization cells, per-variant-train-only fits); full baseline suite; DEC-5 names ar_ridge_T0 strongest (+0.1703) |
| 08-26 | §5a grid (pre-declared 4 cells × 3 seeds, 12/12 runs OK): **T1×robust sole pairwise dominator** — mechanical selection, no escalation. **M2 = 10-seed log-mean ensemble FROZEN** (hash table 09:39:40Z). Val median SS_fire_t6 **+0.3131**, all 12 events positive |
| 08-26 | ORDER B-2 (post-freeze, conservative): log-space baselines → **FINAL B6 = gbdt_log_T1 (+0.3050)**; M2 paired wins vs final B6 on val: **9/12**. Baseline search CLOSED |
| 08-26 | Access script `test_access_v1.py` written, dry-run-validated on val only, hashed. **THIS LOCK PACKAGE assembled** |
| ~Oct | Season close (DEC-6): block-B extensions, DEC-3 audit for B, deferred ERA5, re-screens — all under procedures named in the lock |
| after | **THE single test access** (ACC-1: one script, one run, both blocks) → results → manuscript → submission |

## 2. What is frozen (hash register)

The full register with every 64-character hash is prereg §8 (single source of
truth). Summary rows (full values, per A-b):

| Artifact | Hash |
|---|---|
| Quality gate `scripts/frp_quality_gate.py` | `e02db840cf77031d4ba32a33ed712cd126fc15d1d4d75fcaa94222135d725115` |
| Pixel-disjointness audit v2 (v1 `e6709e6f1ac7c20af3f19095b7e52cc9b366d63a3d0c6989bcf83267b4de8b52` invalid, disclosed) | `cc1dbf86ecfa26b04f73ff9f49016daef73311b65ca5973fc8a39f39c33ba2c2` |
| M2 freeze table `data/models/e28/m2_freeze_hashes.json` (contains the 10 member checkpoint hashes), frozen 2026-08-26T09:39:40Z | `27b947348ff4ee493bc46c88a659ad89d1372bfcb9c572cf5ec98b222bf43cfa` |
| dataset_v6_1.h5 | `0a1a7c8ac23d6525bac1f93e99ec6a9befd537653baaa7495e65693f6ff75724` |
| normalization_params_v6_1.json | `b55421f74665c57535fd4e7e82d2dc0633e4c96c426e049a35c6ff2a84fbb827` |
| B-2 script `scripts/b2_log_baselines.py` (defines FINAL B6) | `3ac578daba2da2c46fa802a9a0d9db27f4821e1c228e4b7726fd39e461ac0551` |
| B-2 results `baselines_v6_1_b2_log_results.json` | `27c1f967e548893a77f1567365e67a5edf01394449696dc4c8e0a7163ca776dd` |
| Evaluation protocol `.claude/skills/evaluation-standards.md` (mask/grain inlined in prereg §6a) | `147b1b3a5fae5172f35103a21a2183b8587d3ae6b047521fa6d77091d50365f3` |
| Access script `scripts/test_access_v1.py` — **REV 4, CURRENT** (memo IA-1: sensitivity-set conformance + frozen block-A assertion + input-hash asserts + zero-window rows; revalidated 2026-08-28; supersedes REV 1 `93ed7538…29b3`, REV 2 `8217e2b6…e9ae`, REV 3 `7d48a32a…2f2f` — full values in prereg §8) | `6b403afc53b41a4f0c5bf75b45954ae39cdd35b06ee9d2ca6feef56ee897b1c4` |
| Frozen block-A composition `test_event_list_blockA_LOCKED.csv` (46 rows = 30 primary + 2 sensitivity-increment + 14 DEC-3-dropped descriptive; runtime-asserted) | `b339f1795aaf977ebef0a2e8110d5e4df3c98e0cea0f902f6ec22a4f658b05e3` |
| Block-A input manifest `blockA_input_manifest.csv` (92 files, 0 missing) | `7163744d71085a1bdb3178b162d135265dfc316b367c1894d88f7e616e0122f0` |
| Environment lockfile `resubmission/access_environment_lock.txt` (A-d; critical four asserted at script startup) | `3c602982003cc19e55f1d7d8937e304068ad25b7c7627fe7dba7d1ac5ba501e5` |
| M1 deposit `resubmission/m1_deposit/` (Zenodo 10.5281/zenodo.20627737 archive) | zip md5 `771491878286355cd9b15d418aa195cb`; 10 checkpoint SHA-256s frozen in the access script |
| Access dry-run records (`access_dryrun_val/`) — REV 3 run results `75ac0978c34131ab0e42541970aeb597a9a89169c62628ffd6c604d79f8b8528` (REV 2 run `3fbe1ee9…97eb` superseded; differs only by run_utc); strata BYTE-IDENTICAL across REV 2/3 runs (IA-A3) | `48fc8a445f06ce93a2377ca9054bbcd75cbc955dfc6cffed7841becfa05cd32a` |
| Prereg locked text | [at signature — the only legitimately open row] |

## 3. Statistics (as pre-declared in prereg §6)

- **PRIMARY (single hypothesis, α = 0.05, one-sided):** for per-event
  differences Δ_e = MAE_fire_t6(M2)_e − MAE_fire_t6(B6)_e over the primary
  event set (block A n = 30, DEC-3-audited, DEC-1-excluded, + finalized block
  B), **H1: median(Δ_e) < 0** — equivalently, P(M2 has lower MAE_fire_t6 than
  B6 on a test event) > 1/2. Test: paired sign test on sign(Δ_e); ties
  (Δ_e = 0) dropped. B6 = gbdt_log_T1. N_min: an event enters iff ≥ 5
  fire-active windows; gate-passing AND evaluable counts both reported.
- Secondary/descriptive: same test vs B1 persistence; DEC-1 sensitivity set
  (n = 32 for A); AGG-1 median SS_fire_t6 headline with mean ± std secondary;
  Wilcoxon on the same pairs; per-event tables; MAE at t+1/3/6/12.
- **Honest power note (exact values, recorded pre-access):** M2's val paired
  win rate vs the FINAL B6 is 9/12 = 0.75. Rejection thresholds: **≥ 20 of
  n = 30** (block A alone-sized set) and **≥ 32 of n = 50** (combined-blocks
  estimate). Pass probability at p_true = 0.75: **0.894 at n = 30; 0.971 at
  n = 50**. At p_true = 0.60: **0.291 at n = 30; 0.336 at n = 50**. Passes in
  expectation at the val rate, not comfortably; materially below even odds if
  the true rate is nearer 0.6. The val estimate itself is weakly constrained
  (9/12; 95% Clopper–Pearson CI 0.43–0.95), which is why the table brackets
  p_true at 0.60 and 0.75 rather than trusting the point estimate (IA-A7).
  Stated before access, not after.

## 4. Publish-regardless clause (unconditional)

The outcome of the primary sign test — significant or not — is reported in the
manuscript as the confirmatory result, with the full per-event table for every
gate-passing test event. There is no file-drawer branch: a null result is
submitted as a null result with the same completeness.

## 5. Access design (ACC-1)

- Test blocks are accessed EXACTLY ONCE, together, after season close: one
  script (`scripts/test_access_v1.py`), one run, outputs archived under
  `resubmission/TEST_ACCESS_RESULTS/`.
- The script was validated on validation data only (`--dry-run-val`): it
  rebuilds each val event's windows from CSVs via the exact access code path,
  asserts equality with the frozen dataset_v6_1.h5 arrays, hash-verifies all
  10 M2 members against the freeze table before predicting, and reproduces the
  frozen val numbers. It refuses test access without `--confirm-single-access`.
- Block B finalizes at season close under procedures named here: W-1
  extensions (incl. dz_n_a_2026_0712 per P-1), frozen-gate re-screen of
  right-censored + orphan candidates, DEC-3 v2 audit for B, deferred ERA5.
  B's final n is determined by those rules, not judgment.

## 6. Pre-signature amendments (memo A5) — EXECUTED 2026-08-27

1. **AMEND-5 — M1 provenance: M1 CUT. [SUPERSEDED BY RAT-9 (§6a.1),
   2026-08-27 — retained as history.]** Verification against the
   immutable Zenodo deposit of the original submission
   (10.5281/zenodo.20627737 = frp-prt-grsl v1.0.0 archive, 2026-06-10) found
   the prereg-named M1 checkpoints (stage-1 C-FROZEN/C-HONEST, 2026-05-19
   lock) exist neither locally nor on Zenodo; the only Zenodo-archived
   ensemble is byte-identical to the LATER stage-2 "Row-2" redesign model —
   a different model identity. Per the amendment's own rule (ambiguous →
   cut), the access was scoped to M2 only at that time. ~~"the access
   evaluates M2 only"; "The access script needed no change; its hash
   stands."~~ [Both statements superseded by RAT-9: M1 was subsequently
   re-scoped under its true identity, and the access script was extended and
   re-hashed — see §6a.1 and the CURRENT script row in §2.]
2. **AMEND-6 — EXECUTED.** The "block-A-alone submission timeline" default is
   marked SUPERSEDED by ACC-1 (one access, both blocks, after season close)
   in prereg §6 and §9, dated.
3. **AMEND-7 — EXECUTED.** The pre-registered interpretation paragraph
   (significant / null / no post-hoc promotion) is added to prereg §6 and
   locks with the document.

## 6a. Pre-signature rulings (memo A6) — EXECUTED 2026-08-27

1. **RAT-9 — M1 RE-SCOPED (cut ratified, then conditional re-scope met).**
   M1 is now the Zenodo-archived ensemble of the rejected submission
   (10.5281/zenodo.20627737), under its true identity. The zero-judgment
   condition was verified: the deposit alone supplies the 10 checkpoints
   (SHA-256 table frozen in the access script), the v3 normalization stats,
   the wind-feature module with checkpoint-embedded statistics, the canonical
   feature order (runtime-asserted), the PRT class, and the log-mean rule.
   Deposit vendored at resubmission/m1_deposit/ (zip md5 verified against
   Zenodo). ALL M1 outputs are descriptive; no test involves M1. The
   manuscript states explicitly that the test blocks postdate M1's lock
   (2026-05-19) and deposit (2026-06-10) by construction.
2. **P-1 — hierarchy framing adopted** for the September manuscript brief,
   with the two corrections: no "first"/priority claims; no self-descriptors;
   B6's adversarial post-freeze construction stated wherever B6 appears.
3. **P-2 / AMEND-8 — descriptive stratifications added** (per-horizon,
   wind-regime, volatility tertiles) under T-1…T-4: win counts + median ΔSS
   only, constants frozen from TRAIN in the hashed script
   (Δwind 1σ = 0.79943; volatility tertiles 0.83485 / 1.08513), one archived
   strata table per run, script re-hashed + dry-run revalidated on val.
4. **VETO-3 recorded as standing:** no M2+B6 hybrid, no primary-comparison
   softening, no M2 retraining — refused regardless of proposer, permanently.

## 6b. Remaining at signature

1. DEC-4: ratification of operating defaults D1–D7 + P-1/P-2 — the one-page
   list with per-item status is `DEC4_RATIFICATION_ANNEX_2026-08-28.md`
   (D1–D5, D7, P-1/P-2 ratified as adopted; D6 ratified as SUPERSEDED by
   ACC-1). Folded into the signature; no separate action.
2. ~~Supervisor walkthrough + signature.~~ **SUPERSEDED 2026-08-28 (IA-B4/
   G6′): replaced by the IA-1 Step-4 public registration — new Zenodo record
   (canonical, DOI timestamp = lock time) + annotated git tag
   `prereg-lock-v1` + release on the public frp-prt-grsl repository; then
   the REQUIRED season-close deposit (Step 6) and REQUIRED outcome-blind
   post-access deposit (Step 8).**

## 7. Self-certification and public registration record (G6′, per IA-1)

```
Pre-registration locked as read, including §5 (frozen evaluation suite),
§6 (statistics), §8 (hash register), the four disclosures above, and
DEC-4 self-certification of operating defaults D1–D7 + P-1/P-2 per
DEC4_RATIFICATION_ANNEX_2026-08-28.md (D6 certified as superseded by ACC-1).

Self-certified under G6′ (supervisor participation declined, logged
2026-08-28; no independent human sign-off exists and none is claimed).
Lock integrity mechanism: public timestamped registration PRECEDING all
test-data access (IA-1 Step 4), plus the REQUIRED season-close deposit
(Step 6) and REQUIRED outcome-blind post-access deposit (Step 8).

Executor / self-certifier: D. Valsamis        Date: 2026-08-28
Locked prereg file: prereg_test_blocks_v1_LOCKED.md
Locked prereg SHA-256 (as published):
  9fc24cc719df69643d5a73f0881349be9646e04ff5c4339f6573ae8d88de7c5e
AMENDMENT registered with the lock (AMEND-9, block-A-only access; ACC-1
  superseded; block B remains sequestered):
  PREREG_AMENDMENT_9_2026-08-28.md, in lock_bundle_prereg_v1.zip (9 files;
  this package is inside the bundle, so the bundle SHA-256 is recorded
  OUTSIDE it — in RESEARCH_LOG and in the git tag annotation — per the §8
  self-reference protocol)
Registration venue (AMEND-9 §5): annotated git tag + public bundle on
  github.com/dmvalsamis/FRP_GSRL, PRECEDING the access; Zenodo DOI of the
  identical bundle REQUIRED before manuscript submission.
Zenodo DOI (required pre-submission):          ____________________
Git tag / commit SHA (FRP_GSRL):               prereg-lock-v1 @ 9a65d8e
Registration UTC:                              2026-08-28T12:30:09Z
[The git-tag lines are filled the moment the tag publishes; the lock takes
force at that public timestamp. The Zenodo line is filled when the DOI is
minted.]
Session log: RESEARCH_LOG.md 2026-08-24 → 2026-08-28
```
