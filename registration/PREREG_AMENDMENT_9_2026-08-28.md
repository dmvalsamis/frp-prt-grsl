# AMEND-9 — Block-A-Only Access (2026-08-28)
# Amendment to prereg_test_blocks_v1_LOCKED.md, executed BEFORE public
# registration and registered TOGETHER WITH the pre-registration in the same
# public deposit. The lock takes force at that registration; this amendment is
# therefore part of the locked text, not a post-lock deviation.

## Authority

Sole-author decision by the executor (D. Valsamis) under G6′ governance
(supervisor participation declined, logged 2026-08-28). Directive of record
(RESEARCH_LOG 2026-08-28): do not wait for the 2026 fire-season close;
finalize results on the data held now; execute the test; proceed to the
manuscript.

## What changes

1. **ACC-1 / AMEND-6 ("the test blocks are accessed exactly once, TOGETHER,
   after season close — there is no block-A-alone access or submission path")
   is SUPERSEDED.** The original per-block wording of §4 ("the evaluation runs
   ONCE per test block") is restored. The single pre-registered access for
   **test block A (2025)** executes immediately upon public registration of
   this bundle.
2. **Test block B (2026) remains fully sequestered and is NOT accessed.** No
   block-B event appears in the access event list; no block-B FRP content,
   prediction, or forecast-derived statistic is computed. The IA-B2 candidacy
   cutoff (earliest FIREDATE ≤ 2026-10-31) and season-close procedure
   (≥ 2026-11-21) remain in force for a possible future block-B access, which
   — if it ever runs — requires its own prior public season-close deposit and
   is reported as a separate, clearly-labeled evaluation. The manuscript of
   the present resubmission reports block A only and discloses block B's
   existence and continued sequestration.
3. **Primary event set** (prereg §6) is therefore **block A, n = 30**
   (in_primary = 1 in `test_event_list_blockA_LOCKED.csv`, SHA-256
   b339f1795aaf977ebef0a2e8110d5e4df3c98e0cea0f902f6ec22a4f658b05e3), with
   the N_min ≥ 5 evaluability rule applied at evaluation time as declared.
   The DEC-1 sensitivity set is n = 32 (in_sensitivity = 1). All 46
   gate-passing block-A events appear in the descriptive table. The
   pre-computed rejection region and power are unchanged from the prereg's
   n = 30 calculation: rejection at ≥ 20 wins of 30 (fewer with ties
   dropped); pass probability 0.894 at true P(win) = 0.75, 0.291 at 0.60.
4. **Access declaration (replaces the IA-1 Step-6 season-close deposit for
   block A):** the access is executed by operator D. Valsamis, on the pinned
   environment (resubmission/access_environment_lock.txt), with script
   `scripts/test_access_v1.py` REV 4 (SHA-256
   6b403afc53b41a4f0c5bf75b45954ae39cdd35b06ee9d2ca6feef56ee897b1c4 — byte
   unchanged by this amendment; the event-list argument alone scopes the run)
   and event list = the frozen `test_event_list_blockA_LOCKED.csv` itself.
   Outputs go once to `resubmission/TEST_ACCESS_RESULTS/`, are hashed and
   logged in RESEARCH_LOG the same day, and the raw outputs are publicly
   deposited within 72 h, outcome-blind (IA-1 Step 8, unchanged).
5. **Registration venue clarification:** the public timestamp anchoring this
   lock is an annotated git tag + public availability of the lock bundle on
   the program's public repository (github.com/dmvalsamis/FRP_GSRL), created
   BEFORE the access run. A Zenodo deposit of the identical bundle (DOI)
   remains REQUIRED before manuscript submission and is added to the
   registration record when minted. The prereg status line's "identifiers in
   §8" is satisfied via the lock package §7 registration record, per the
   §8 self-reference protocol.

## Why this is conservative (disclosed reasoning)

- Every substantive object was frozen BEFORE this amendment: M2 (2026-08-26,
  hash table), B6 (B-2, 2026-08-26), the access script (REV 4, 2026-08-28),
  the block-A composition (DEC-1/DEC-3, frozen CSV inside the hash
  perimeter), and all statistics/interpretation text (§6, AMEND-7/8). This
  amendment changes WHEN the access happens and WHICH pre-frozen block it
  covers — nothing about models, baselines, event membership, statistics, or
  interpretation.
- No test statistic existed when this amendment was written; the scheduling
  decision cannot be outcome-informed.
- Block-A membership is mechanical and exhaustive-through-gate; accessing A
  alone cannot cherry-pick events.
- The cost is honestly stated: n = 30 primary instead of ~30 + block B,
  i.e., less power than the combined design, accepted for schedule reasons.
  Block B is not consumed — it remains available, still sequestered, for a
  future strengthening evaluation.

## Manuscript disclosure obligation

The manuscript reports, verbatim in substance: the pre-registration
originally scheduled a single combined access after the 2026 season close;
before registration, the design was amended (this document, registered with
the prereg) to a block-A-only access, with the 2026 block remaining
sequestered and unevaluated.

— D. Valsamis, 2026-08-28, under G6′ self-certification.
