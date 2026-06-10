# How the 34 events were selected — reproducible vs. manual

This project is honesty-first about its dataset. Event selection was **two-part**,
and only one part is reproducible. We state both plainly so a reviewer is not
misled into thinking the dataset came out of an automated database funnel.

## Part 1 — A reproducible data-quality gate (coded, and it binds)

Every candidate event had to pass a fixed gate, applied identically
(`scripts/data_quality_gate.py`):

| Criterion | Threshold |
|---|---|
| Fire-active hourly slots | ≥ 15 |
| Duration | ≥ 5 days |
| Complete diurnal days (> 50 % of 15-min slots present) | ≥ 3 |
| Gap (missing-slot) rate | < 0.40 |
| Mid-IR saturation rate | < 0.30 |

All 34 retained events pass; the gate binds at the margin (`corinthos_2024` = 15
fire slots, `varnavas_2024` = 5 days, `olympia_2021` = 3 complete days),
confirming it was the operative rule and not a post-hoc description. Documented
rejects fail it on at least one axis (0/3/14 fire slots; complete_days 0–2;
saturation > 30 %).

## Part 2 — A manually curated candidate pool (NOT reproducible)

The 34 events were **hand-picked**: documented major Mediterranean wildfire
events (2017–2024) chosen to span the basin's intensity-tier and wind-regime
range, plus two targeted gap-filling waves (N. Africa / Bora / France / Meltemi;
then PT / N. Macedonia / Algeria / Spanish spring). There is **no reproducible
"candidate N → 34" query** for this set. A systematic EFFIS query exists in the
broader project but drove *later* dataset versions, not the 34 used in the paper.

## Honest flags (what the paper must NOT claim)

1. **Candidate pool is manual**, not query-derived, for these 34 events.
2. **Intensity floor is ad hoc.** Exactly one catalogued event (~874 MW) was
   excluded by hand for low intensity; no general minimum-FRP threshold is
   coded. The retained minimum peak is ~2,160 MW.
3. **Bounding boxes are manual**, drawn around the documented perimeter with a
   ~0.25° buffer; box choice affects the fire-slot count.
4. **The split is event-level, stratified by intensity tier and wind regime —
   NOT chronological or geographic.** 2023 events appear in all three splits and
   Greece appears in all three. The test set does skew to the most recent
   seasons (2023–2024: Volos'23, Varnavas'24, Val Maior'24, Corinth'24), but
   that is a hold-out choice, not a clean time/space cut. Any manuscript wording
   implying a chronological/geographic partition should be corrected.

## Volume (the "small in events, large in data" point)

| Split | Events | 12 h windows | Fire-active windows |
|---|---|---|---|
| Train | 25 | 5,210 | 2,734 |
| Val | 5 | 863 | 580 |
| Test | 4 | 558 | 186 |
| **Total** | **34** | **6,631** | **3,500** |

34 events expand to 6,631 supervised 48→12 h windows (3,500 fire-active) over
~2,836 fire-active hourly FRP observations.

## Sources to cite
- **LSA SAF FRP-PIXEL** (EUMETSAT) — the FRP target and the detection floor that
  defines `fire_slots` (primary; every event traces to it).
- **EFFIS** MODIS burned-area perimeters — candidate areas (later waves).
- **GWIS / NASA FIRMS** — alternative candidate discovery.
- **Copernicus EMS** rapid-mapping polygons — used only for the Fig. 1 burn
  scars. No per-event EMSR/EFFIS/FIRMS identifier is stored in the catalogue;
  events carry internal names only.
