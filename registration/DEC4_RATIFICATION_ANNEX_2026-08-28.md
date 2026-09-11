# DEC-4 Ratification Annex — Operating Defaults D1–D7 + P-1/P-2
**Purpose:** the one-page list the supervisor's signature ratifies (DEC-4).
These decisions were adopted under executor authority during Task E2
(2026-08-25 onward), each logged at adoption; the signature converts them from
"operating defaults pending ratification" to ratified protocol. Source of the
original table: `TASK_E2_PLAN_2026-08-25.md`; statuses updated 2026-08-28.

| # | Default (as adopted) | Basis | Status at signature |
|---|---|---|---|
| D1 | Gate package: NO intensity floor; published five-criteria gate byte-identical (`frp_quality_gate.py`, hashed); evaluability rule N_min = 5 dataset-level fire-active windows | FINDINGS_2026-08-24 §4 (empirical: intensity floor rejected; volatility, not intensity, predicts forecastability) | In force; gate applied to all 162 gate-passers; N_min in prereg §6 |
| D2 | ba_trebinje_2022 readmitted to the train pool | FINDINGS §4 (its exclusion was empirically unsupported) | Executed: pinned TRAIN in the FINAL split (seed 20260825) |
| D3 | Test-block candidacy area floor 3,000 ha | historical wave-filter precedent | In force; carried as the §3.1 provisional bracket, resolved to final text by this signature |
| D4 | Mega-episode sub-division: merged episode with snapped bbox ≥ 5.5 sq° re-clustered by STRICT bbox intersection; threshold sits between largest legitimate single (4.8 sq°) and smallest over-merge (6.75 sq°); declared before any affected gate results | E1.2 flag resolution | Executed 2026-08-25 (9 episodes → 46 sub-candidates); in prereg §3.1 |
| D5 | M2 selection metrics M-1/M-2/M-3 + rules R-1/R-2/R-3 as written in prereg §5a (extended by AMEND-2 to the 4-cell grid with R-1′/R-2′/R-3′) | supervisor: "work with what we have" (2026-08-25) | Executed 2026-08-26: 12/12 runs, T1×robust sole dominator, M2 frozen |
| D6 | Block-A-alone submission permitted if ≥ 10 evaluable events; block B at revision/follow-up | agreed 2026-08-24 | **SUPERSEDED by ACC-1 (memo A4, 2026-08-26): one access, both blocks together, after season close. Ratified as superseded — the signature ratifies the SUPERSESSION, not the original default** |
| D7 | Catalogue-conflict refinement: conflict requires POSITIVE-AREA bbox intersection + window overlap; zero-area edge/corner contact is not a conflict. Declared during E2.1 from metadata-only output, before any affected event's FRP was downloaded; restored 3 events | E2.1 discovery | Executed; DEC-1 handles the restored events (primary-exclude / 2-event sensitivity per prereg §6a) |
| P-1 | W-1 over-cap chains (span > 30 d) keep their ORIGINAL earliest+9d windows, disclosed per event: me_n_a_2017_0707 (64 d), mk_novaci_2024_0809 (33 d), dz_n_a_2026_0712 (38 d) | delegated ruling, memo A2 follow-up ("fix whatever we have to"), 2026-08-25 | In force; in prereg §3.2 |
| P-2 | W-1 window rule applied with FULL UNIFORMITY across all pools (66 affected events accounted: 41 extended + 3 P-1 + 17 block-B at season close + 5 gate-failed candidates) | same delegated ruling; ratified by memo A3 (RAT-1/RAT-2) | Executed for dev + test-A; block-B extensions are a season-close procedure named in the lock |

**Ratification line (revised 2026-08-28 per IA-1/G6′):** DEC-4
(self-certified 2026-08-28 under G6′, disclosed): the executor certifies
D1–D5, D7, and P-1/P-2 as adopted and D6 as superseded by ACC-1.
Certification is bound into the publicly registered lock; no independent
ratification was obtained and none is claimed.
