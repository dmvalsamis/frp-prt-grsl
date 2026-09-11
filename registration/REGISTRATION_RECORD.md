# Registration record — what happened, and what a reader can independently check

This file exists so that no reader has to take the phrase "pre-registered" on trust. It states
the timeline, the hashes, and — the part usually left out — the limits of what any of it proves.

## 1. Timeline of 28 August 2026 (UTC)

| # | Event | Time | Source of the timestamp |
|---|---|---|---|
| 1 | Commit `9a65d8ef25c3988f792c6ea9f5eef71db3c30709`, "AMEND-9: block-A-only access; lock bundle finalized for public registration" | 12:29:42 | git object (author = committer); **client clock** |
| 2 | Annotated tag `prereg-lock-v1` (tag object `70d11060e8a55c2a2e234911bf10611f922be497`) on that commit | 12:29:57 | git object (tagger); **client clock** |
| 3 | Push of the tag to the project repository | 12:30:09 | the executor's contemporaneous log entry and the lock package §7 |
| 4 | The single access starts | 12:30:59 | `results/access_2025/access_record_20260828T123059Z.json` |
| 5 | The access completes | 12:32:45 | `run_utc` in `access_20260828T123059Z_results.json` |
| 6 | Outputs committed | 12:33:58 | git object; client clock |

The interval from tag to access start is 62 seconds; from commit to access start, 77 seconds.

## 2. What this does and does not establish

**It establishes** that the hypothesis, the rejection region, the statistics, the event list and
the scoring code existed in a fixed, hashed form before any test statistic was computed, and
that the outputs match those hashes. The internal ordering is consistent across four independent
artefacts: the tag message, the lock bundle, the access record, and the research log.

**It does not establish** a third-party timestamp. The repository holding the tag was private at
the time of the access, so no public server recorded the push before the access occurred. Git
timestamps are written by the client and can be set to any value by whoever controls the machine.
A reader who does not trust the authors cannot distinguish this from a tag written afterwards.

The registration is therefore **self-certified**, and the letter says so. What raises it above a
bare assertion is that the null result is the one being reported: the pre-declared primary test
failed to reject, one event short of its rejection region, and that is what is published.

## 3. Hashes

| Artefact | SHA-256 |
|---|---|
| `lock_bundle_prereg_v1.zip` (9 files; the canonical, unambiguous identifier) | `06beb77fb131d2cdcccafb0bb5ce6800532ba5e7be9c88ffd16349607cb6f71c` |
| `prereg_test_blocks_v1_LOCKED.md` as recorded (CRLF line endings, 37,449 bytes) | `9fc24cc719df69643d5a73f0881349be9646e04ff5c4339f6573ae8d88de7c5e` |
| the same document with LF line endings (36,944 bytes) | `9f689d9bbf5fdbc5d9885c430454dae141974765caceec91d3ada7f557be0086` |
| `test_access_v1.py` (REV 4, the script used at the access) | `6b403afc53b41a4f0c5bf75b45954ae39cdd35b06ee9d2ca6feef56ee897b1c4` |
| `test_event_list_blockA_LOCKED.csv` (46 rows) | `b339f1795aaf977ebef0a2e8110d5e4df3c98e0cea0f902f6ec22a4f658b05e3` |
| `PREREG_AMENDMENT_9_2026-08-28.md` | `241636d8fe1002f274dcceffd0e6975c56e831a5377b288d80079abd102ae00f` |
| `access_20260828T123059Z_per_event.csv` (the locked result table) | `2c1dd44498b3fe76be7005b493eb0d2424c8347d601efcca1a505418fceb5e79` |

**On the two hashes of the registration document.** The value recorded everywhere is the one for
the file as it existed on a Windows checkout, with CRLF line endings. Git stores the same content
with LF, which hashes differently. Converting the stored blob's line endings back to CRLF
reproduces the recorded value exactly. The zip hash above avoids the issue entirely and is the
one to quote.

## 4. The amendment, and why it matters

`PREREG_AMENDMENT_9_2026-08-28.md` was written **before** the registration and travels inside the
same lock bundle. It narrowed the single access from "both blocks after the season closes" to the
2025 block alone, leaving the 2026 block sequestered. The cost is disclosed in the amendment
itself: a smaller primary set, and therefore less power — the declared power against an event-wise
win rate of 0.75, 0.70 and 0.65 is 0.83, 0.64 and 0.41. The comparator, the model, the script, the
composition and the interpretation were all frozen before this amendment, so it changed scheduling
and scope only, and no test statistic existed when it was written.

The 2026 block was later examined anyway, on 8 September 2026, on the author's instruction and
after the 2025 result was known. That consumed its sequestration. It is reported as exploratory
throughout, it is not a confirmatory replication, and it never will be.

## 5. Three scorings of the 2025 block, disclosed

The 2025 block has been scored three times: once confirmatory (28 August) and twice inside later
exploratory runs (8 and 9 September), which re-scored it to place the 2026 block and the six
encoders on a common footing. The later runs import the same frozen script and reproduce the
confirmatory PRT and boosting values to about 1e-14 (`results/encoder_study/identity_check.json`).
No selection was made on either block in any of the three, and every run is reported.

## 6. Model provenance

The ten checkpoints in `models/prt_ensemble_m2/` are the ensemble members loaded at the access.
Their hashes are listed in `m2_freeze_hashes.json` and were verified at load time by the access
script; the console log records each one. The ensemble was frozen on 2026-08-26 at 09:39:40 UTC,
before the comparator's configuration search closed, and two days before the registration.

The archived ensemble of the earlier study, scored inside the same access for description only,
is the one published in this repository's `v1.0.0` release and archived at Zenodo
10.5281/zenodo.20627737; the pre-registration verified it against that deposit by checksum. No
test in the letter involves it. That is why `v1.0.0` and the `main` branch are preserved
unchanged: rewriting them would break the provenance chain this registration depends on.
