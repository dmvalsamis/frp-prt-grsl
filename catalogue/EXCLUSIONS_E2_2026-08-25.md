# E2 Exclusion Record — Permanent (DEC-2, adjudicated 2026-08-25)

Governance: adjudication memo 2026-08-25 (RESEARCH_LOG), DEC-2 approved as proposed
with the rule-version addendum. Evidence: `gate_results_e23_2026-08-25.csv`,
script `09_subdivide_mega_episodes.py` output (RESEARCH_LOG 2026-08-25 entries).

## 1. Gate-failed dev-wave events (permanent exclusion — canonical gate)

All six failed exactly one criterion: fire_slots ≥ 15. All other criteria passed
(gap 0.000, saturation 0.000, duration 11 d, complete days ≥ 9). Data retained on
disk (raw + hourly CSV); events remain in the catalogue with their gate verdict and
are excluded from every dataset build.

| Event | fire_slots | duration_d | complete_d | gap | sat | Failed | Governing rule |
|---|---|---|---|---|---|---|---|
| me_n_a_2020_0406 | 11 | 11 | 11 | 0.000 | 0.000 | fire_slots≥15 | canonical gate e02db840… (conflict rule n/a) |
| dz_n_a_2020_1104 | 0 | 11 | 11 | 0.000 | 0.000 | fire_slots≥15 | canonical gate (conflict rule n/a) |
| ba_n_a_2020_0317 | 13 | 11 | 11 | 0.000 | 0.000 | fire_slots≥15 | canonical gate (conflict rule n/a) |
| mk_veles_2019_0323 | 13 | 11 | 11 | 0.000 | 0.000 | fire_slots≥15 | canonical gate (conflict rule n/a) |
| me_n_a_2016_1208 | 0 | 11 | 11 | 0.000 | 0.000 | fire_slots≥15 | canonical gate (conflict rule n/a) |
| al_lezhe_2020_0729 | 3 | 11 | 11 | 0.000 | 0.000 | fire_slots≥15 | canonical gate (conflict rule n/a) |

## 2. Conflict-dropped D4 sub-episodes (spatially absorbed by onboarded neighbours)

Each has POSITIVE-AREA bbox intersection plus window overlap with an already-onboarded
catalogue event; their burnt area falls (partly) inside that neighbour's bbox/window,
so their fire activity is represented by the neighbour's FRP series. Never downloaded.
All eight were adjudicated UNDER D7 (positive-area rule) — these are genuine overlaps,
not edge-touch artifacts.

| Sub-episode | Absorbing event | Overlap (sq°) | Governing rule |
|---|---|---|---|
| pt_vieira_de_leiria_2017_1015 | lousa_2017 | 0.150 | D7 |
| pt_pinheiro_de_azere_2017_1008 | lousa_2017 | 0.600 | D7 |
| pt_mira_2017_1015 | lousa_2017 | 0.225 | D7 |
| pt_sobral_de_sao_migu_2025_0810 | es_caceres_2025_0808 | 0.062 | D7 |
| pt_carvicais_2025_0815 | es_puertas_2025_0813 | 0.375 | D7 |
| pt_quintas_de_sao_bar_2025_0815 | es_caceres_2025_0808 | 0.188 | D7 |
| pt_frechas_2025_0817 | es_puertas_2025_0813 | 0.125 | D7 |
| es_gallegos_del_rio_2025_0812 | es_puertas_2025_0813 | 0.188 | D7 |

Note: under the PRE-D7 (inclusive) rule, three additional sub-episodes were dropped on
zero-area edge contact — `es_rua_a_2025_0808`, `pt_trancoso_sao_pedro_2025_0808`,
`pt_real_2025_0728`. D7 re-admitted them; per DEC-1 they carry `d7_restored = true`
in the catalogue, are EXCLUDED from the primary sign test, and are covered by the
pre-declared sensitivity analysis (prereg §6a).

## 2b. W-1 window-staleness note (added 2026-08-25, ruling P-2)

The six gate-fail criteria values in §1 were computed on the ORIGINAL
earliest+9 d windows. Under the corrected uniform window rule (W-1), five of the
six would receive longer windows (me_n_a_2020_0406 +5 d, dz_n_a_2020_1104 +2 d,
ba_n_a_2020_0317 +2 d, mk_veles_2019_0323 +6 d, me_n_a_2016_1208 +1 d). Per P-2
they are NOT re-downloaded or re-screened: the dev pool is curated, not
exhaustive, so no admission obligation exists. Their exclusion record therefore
reflects the pre-W-1 window definition (governing rule column: canonical gate on
the original window). Test-block gate-fails carry no such staleness — no affected
test event was a gate-fail.

## 3. Dissolved parent episode

`it_aliminusa_2023_0718` (18 polygons, 24,048 ha aggregate, snapped bbox 5.69 sq°):
strict D4 re-clustering produced 17 sub-clusters, NONE reaching the 3,000 ha candidacy
floor (largest fragment 2,862 ha). The parent was an over-merge artifact of dispersed
small Sicilian fires with no dominant episode; nothing represents it, by construction
of the declared mechanical rules. Governing rules: D4 sub-division + 3,000 ha floor
(conflict rule n/a).

## 4. Superseded parent episodes (replaced by their sub-candidates)

The 9 episodes ≥ 5.5 sq° (2 test-A, 7 dev) are SUPERSEDED as candidates by their D4
sub-candidates (`candidates_subdivided_2026-08-25.csv`, parent_cand_id column). The
two test-A parents (`pt_trancoso_sao_pedro_2025_0726`, `es_una_de_quintana_2025_0802`)
were never onboarded; no parent has FRP data under its parent identity.
