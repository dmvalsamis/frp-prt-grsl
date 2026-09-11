# EXPLORATORY post-peak-day diagnostic by architecture, blocks A + B (77 events) -- 20260909T105643Z

Every predictor is compared with B6 on the same windows (predictions cached by `explore_arch_eval.py`; B6 / B1 / M2 / M1 from the locked test_access_v1 machinery). Post-peak day = the UTC day after the event's peak FRP hour. 'flips' = events whose paired outcome vs B6 changes when that one day is removed (post hoc, descriptive). M1 is descriptive.

## block A (45)

| model | wins vs B6 | worst day = post-peak | median worst-day share | median post-peak share | flips without post-peak day | post-peak bias > 0 | post-peak bias > B6 bias | sign test excl. post-peak day | sign test post-peak day alone |
|---|---|---|---|---|---|---|---|---|---|
| M2 | 28/17 | 6 | 0.42 | 0.10 | 5 | 8 | 14 | 29/16 p=0.036 | 19/17 p=0.434 |
| A1 | 28/17 | 6 | 0.45 | 0.09 | 4 | 11 | 13 | 28/17 p=0.068 | 19/17 p=0.434 |
| A2 | 32/13 | 6 | 0.46 | 0.10 | 7 | 12 | 12 | 29/16 p=0.036 | 17/19 p=0.691 |
| A3 | 30/15 | 6 | 0.44 | 0.10 | 6 | 1 | 6 | 28/17 p=0.068 | 23/13 p=0.066 |
| A4 | 27/18 | 6 | 0.44 | 0.10 | 5 | 11 | 24 | 28/17 p=0.068 | 19/17 p=0.434 |
| A5 | 29/16 | 6 | 0.45 | 0.09 | 3 | 10 | 15 | 28/17 p=0.068 | 16/20 p=0.797 |
| A6 | 30/15 | 6 | 0.45 | 0.09 | 7 | 4 | 8 | 33/12 p=0.001 | 23/13 p=0.066 |
| M1 | 26/19 | 6 | 0.45 | 0.10 | 6 | 12 | 22 | 24/21 p=0.383 | 26/10 p=0.006 |

## block B (32)

| model | wins vs B6 | worst day = post-peak | median worst-day share | median post-peak share | flips without post-peak day | post-peak bias > 0 | post-peak bias > B6 bias | sign test excl. post-peak day | sign test post-peak day alone |
|---|---|---|---|---|---|---|---|---|---|
| M2 | 15/17 | 6 | 0.45 | 0.19 | 4 | 7 | 13 | 15/17 p=0.702 | 8/15 p=0.953 |
| A1 | 12/20 | 6 | 0.46 | 0.16 | 5 | 8 | 10 | 17/15 p=0.430 | 8/15 p=0.953 |
| A2 | 12/20 | 6 | 0.47 | 0.16 | 6 | 7 | 10 | 14/18 p=0.811 | 8/15 p=0.953 |
| A3 | 12/20 | 6 | 0.46 | 0.13 | 5 | 5 | 6 | 15/17 p=0.702 | 8/15 p=0.953 |
| A4 | 20/12 | 5 | 0.47 | 0.18 | 4 | 9 | 13 | 20/12 p=0.108 | 10/13 p=0.798 |
| A5 | 11/21 | 6 | 0.47 | 0.17 | 8 | 9 | 10 | 15/17 p=0.702 | 7/16 p=0.983 |
| A6 | 15/17 | 6 | 0.45 | 0.13 | 2 | 6 | 5 | 13/19 p=0.892 | 10/13 p=0.798 |
| M1 | 12/20 | 5 | 0.47 | 0.09 | 4 | 10 | 15 | 12/20 p=0.945 | 9/15 p=0.924 |

## pooled (77)

| model | wins vs B6 | worst day = post-peak | median worst-day share | median post-peak share | flips without post-peak day | post-peak bias > 0 | post-peak bias > B6 bias | sign test excl. post-peak day | sign test post-peak day alone |
|---|---|---|---|---|---|---|---|---|---|
| M2 | 43/34 | 12 | 0.45 | 0.11 | 9 | 15 | 27 | 44/33 p=0.127 | 27/32 p=0.783 |
| A1 | 40/37 | 12 | 0.46 | 0.11 | 9 | 19 | 23 | 45/32 p=0.086 | 27/32 p=0.783 |
| A2 | 44/33 | 12 | 0.46 | 0.12 | 13 | 19 | 22 | 43/34 p=0.181 | 25/34 p=0.904 |
| A3 | 42/35 | 12 | 0.45 | 0.10 | 11 | 6 | 12 | 43/34 p=0.181 | 31/28 p=0.397 |
| A4 | 47/30 | 11 | 0.45 | 0.11 | 9 | 20 | 37 | 48/29 p=0.020 | 29/30 p=0.603 |
| A5 | 40/37 | 12 | 0.45 | 0.11 | 11 | 19 | 25 | 43/34 p=0.181 | 23/36 p=0.966 |
| A6 | 45/32 | 12 | 0.45 | 0.10 | 9 | 10 | 13 | 46/31 p=0.055 | 33/26 p=0.217 |
| M1 | 38/39 | 11 | 0.46 | 0.10 | 10 | 22 | 37 | 36/41 p=0.753 | 35/25 p=0.123 |

## Post-peak-day signed bias at t+6 (mean over events with a post-peak day, MW)

| model | A: mean bias | A: median bias | B: mean bias | B: median bias | B6 mean bias A / B |
|---|---|---|---|---|---|
| M2 | -2,246 | -197 | -1,431 | -138 | -2,339 / -1,739 |
| A1 | -2,421 | -384 | -1,933 | -133 | -2,339 / -1,739 |
| A2 | -2,392 | -191 | -1,907 | -222 | -2,339 / -1,739 |
| A3 | -2,538 | -444 | -2,042 | -219 | -2,339 / -1,739 |
| A4 | -2,197 | -195 | -1,559 | -131 | -2,339 / -1,739 |
| A5 | -2,292 | -315 | -1,621 | -121 | -2,339 / -1,739 |
| A6 | -2,475 | -678 | -2,039 | -251 | -2,339 / -1,739 |
| M1 | -1,856 | -191 | -1,494 | -134 | -2,339 / -1,739 |

## The three 2026 collapse events (M2 vs B6 gap > 0.3 skill), SS_fire,t6 per model

| event | B6 | M2 | A1 | A2 | A3 | A4 | A5 | A6 | M1 |
|---|---|---|---|---|---|---|---|---|---|
| pt_cambra_e_carvalhal_2026_0702 | +0.501 | -0.050 | +0.372 | +0.361 | +0.461 | +0.455 | -0.060 | +0.517 | +0.453 |
| fr_porge_2026_0722 | +0.414 | -0.192 | +0.328 | +0.264 | +0.310 | +0.070 | -0.092 | +0.383 | +0.312 |
| pt_cerdeira_2026_0727 | +0.109 | -0.332 | -0.018 | +0.022 | +0.158 | -0.170 | -0.065 | +0.147 | +0.111 |
