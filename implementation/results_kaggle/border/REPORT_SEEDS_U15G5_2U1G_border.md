# Multi-seed results — U15G5_2U1G_border

Training seeds: [0, 1, 2] (n = 3). Each seed = its own imitation warm start, REINFORCE and fine-tuning runs, evaluated on the same 100 test instances (its own P0 defines B0). ± = 95% CI over seeds (t, 2 dof); per-seed values in brackets.

## Stage A — static roads


**beta = 0.9**

| policy | T (min) | E (kJ) | over budget (%) | per-seed T |
|---|---|---|---|---|
| P0 | 154.9 ± 14.6 | 2913 ± 319 | 100 ± 0 | [152, 162, 151] |
| P0+ (control) | 155.2 ± 6.1 | 2910 ± 154 | 78 ± 23 | [155, 158, 153] |
| P1 route filter | 123.9 ± 5.2 | 2171 ± 107 | 28 ± 16 | [123, 123, 126] |
| P2 fixed penalty | 159.1 ± 15.6 | 2987 ± 375 | 71 ± 27 | [158, 166, 153] |
| P4 budget-conditioned | 154.8 ± 6.5 | 2879 ± 118 | 69 ± 9 | [153, 153, 158] |
| P4 no budget input | 153.9 ± 8.5 | 2870 ± 255 | 73 ± 27 | [157, 155, 150] |
| P3 Lagrangian | 156.1 ± 6.8 | 2928 ± 133 | 76 ± 16 | [158, 157, 153] |

**beta = 1.0**

| policy | T (min) | E (kJ) | over budget (%) | per-seed T |
|---|---|---|---|---|
| P0 | 154.9 ± 14.6 | 2913 ± 319 | 0 ± 0 | [152, 162, 151] |
| P0+ (control) | 155.2 ± 6.1 | 2910 ± 154 | 47 ± 20 | [155, 158, 153] |
| P1 route filter | 123.4 ± 4.7 | 2181 ± 108 | 0 ± 0 | [122, 122, 126] |
| P2 fixed penalty | 159.1 ± 15.6 | 2987 ± 375 | 53 ± 11 | [158, 166, 153] |
| P4 budget-conditioned | 154.8 ± 6.5 | 2879 ± 118 | 48 ± 14 | [153, 153, 158] |
| P4 no budget input | 153.9 ± 8.5 | 2870 ± 255 | 45 ± 24 | [157, 155, 150] |

**Paired within seed, vs P0+ control (beta = 0.9)**

| policy | dT (min) | dE (kJ) | per-seed dE |
|---|---|---|---|
| P1 route filter | -31.3 ± 11.0 | -739 ± 261 | [-789, -810, -618] |
| P2 fixed penalty | 3.9 ± 9.7 | 78 ± 238 | [+53, +183, -4] |
| P3 Lagrangian | 0.9 ± 4.6 | 18 ± 48 | [+31, -4, +28] |
| P4 budget-conditioned | -0.4 ± 12.2 | -30 ± 271 | [-83, -103, +95] |
| P4 no budget input | -1.3 ± 5.6 | -40 ± 118 | [+9, -43, -86] |

## Stage B — blocked / slow roads (18 events), B = 0.9 B0

| policy | T (min) | E (kJ) | over budget (%) | replans | per-seed T |
|---|---|---|---|---|---|
| P0 open-loop (base) | 192.3 ± 21.3 | 3795 ± 457 | 100 ± 0 | 0.00 ± 0.00 | [189, 202, 186] |
| P0 periodic + road layer | 176.8 ± 4.3 | 3530 ± 133 | 92 ± 13 | 1.14 ± 0.07 | [176, 179, 176] |
| P0 event + road layer | 176.9 ± 4.3 | 3531 ± 132 | 92 ± 13 | 0.28 ± 0.02 | [176, 179, 176] |
| P0+ (control) open-loop | 185.1 ± 9.3 | 3630 ± 272 | 84 ± 17 | 0.00 ± 0.00 | [188, 186, 181] |
| P0+ (control) event + road layer | 172.4 ± 11.1 | 3419 ± 310 | 78 ± 19 | 0.29 ± 0.02 | [177, 171, 169] |
| P0+ event, road-distance assignment | 172.4 ± 11.1 | 3419 ± 310 | 78 ± 19 | 0.29 ± 0.02 | [177, 171, 169] |
| P4 open-loop | 183.1 ± 8.7 | 3561 ± 157 | 78 ± 5 | 0.00 ± 0.00 | [180, 182, 187] |
| P4 periodic + road layer | 169.3 ± 6.7 | 3330 ± 133 | 73 ± 11 | 1.17 ± 0.04 | [167, 169, 172] |
| P4 event + road layer | 169.3 ± 6.7 | 3331 ± 132 | 73 ± 11 | 0.37 ± 0.02 | [167, 169, 172] |
| P4 (road-event trained) event | 171.7 ± 13.9 | 3368 ± 292 | 76 ± 8 | 0.39 ± 0.09 | [175, 175, 165] |
| P3 (beta=0.9) event + road layer | 171.4 ± 5.2 | 3387 ± 99 | 79 ± 10 | 0.29 ± 0.03 | [174, 171, 169] |
| P4 event, road-distance assignment | 169.3 ± 6.7 | 3331 ± 132 | 73 ± 11 | 0.37 ± 0.02 | [167, 169, 172] |
| P0 event, road-distance assignment | 176.9 ± 4.3 | 3531 ± 132 | 92 ± 13 | 0.28 ± 0.02 | [176, 179, 176] |
| P1 route filter, event | 139.0 ± 6.1 | 2555 ± 119 | 45 ± 15 | 0.29 ± 0.03 | [138, 137, 142] |

**Paired within seed (all events)**

| comparison | dT (min) | dE (kJ) | per-seed dT |
|---|---|---|---|
| P0: event − periodic | 0.1 ± 0.2 | 1 ± 2 | [+0.1, +0.0, +0.0] |
| P4: event − periodic | 0.0 ± 0.1 | 0 ± 1 | [+0.0, +0.0, +0.0] |
| P0: event + layer − open loop | -15.4 ± 17.1 | -264 ± 344 | [-13.1, -23.1, -9.9] |
| P0+ control: event + layer − open loop | -12.7 ± 4.9 | -211 ± 114 | [-11.0, -14.9, -12.2] |
| P4: event + layer − open loop | -13.8 ± 2.0 | -230 ± 36 | [-13.0, -13.7, -14.6] |
| P4 − P0+ control (both event + layer) | -3.1 ± 17.3 | -88 ± 424 | [-10.4, -2.4, +3.5] |
| P3(0.9) − P0+ control (both event + layer) | -1.0 ± 6.1 | -32 ± 219 | [-3.8, +0.1, +0.6] |
| P4 road-event-trained − P4 | 2.3 ± 20.4 | 37 ± 424 | [+8.2, +5.8, -7.1] |
| P1 route filter − P0+ control | -33.4 ± 15.8 | -863 ± 405 | [-39.5, -33.7, -26.9] |

Replans event − periodic (P0): -0.86 ± 0.06 per mission.

