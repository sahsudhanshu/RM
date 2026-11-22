# Multi-seed results — U15G5_2U1G

Training seeds: [0, 1, 2] (n = 3). Each seed = its own imitation warm start, REINFORCE and fine-tuning runs, evaluated on the same 100 test instances (its own P0 defines B0). ± = 95% CI over seeds (t, 2 dof); per-seed values in brackets.

## Stage A — static roads


**beta = 0.9**

| policy | T (min) | E (kJ) | over budget (%) | per-seed T |
|---|---|---|---|---|
| P0 | 219.6 ± 144.7 | 3969 ± 2888 | 100 ± 0 | [186, 186, 287] |
| P0+ (control) | 218.7 ± 147.1 | 3970 ± 2892 | 81 ± 45 | [180, 189, 287] |
| P1 route filter | 188.8 ± 154.3 | 3189 ± 2830 | 30 ± 43 | [155, 151, 261] |
| P2 fixed penalty | 221.2 ± 141.3 | 3989 ± 2848 | 80 ± 44 | [190, 186, 287] |
| P4 budget-conditioned | 220.0 ± 143.8 | 3968 ± 2890 | 79 ± 50 | [187, 186, 287] |
| P4 no budget input | 221.9 ± 140.1 | 4007 ± 2811 | 81 ± 40 | [193, 185, 287] |
| P3 Lagrangian | 223.4 ± 136.6 | 4030 ± 2757 | 85 ± 34 | [192, 191, 287] |

**beta = 1.0**

| policy | T (min) | E (kJ) | over budget (%) | per-seed T |
|---|---|---|---|---|
| P0 | 219.6 ± 144.7 | 3969 ± 2888 | 0 ± 0 | [186, 186, 287] |
| P0+ (control) | 218.7 ± 147.1 | 3970 ± 2892 | 31 ± 68 | [180, 189, 287] |
| P1 route filter | 187.9 ± 151.9 | 3209 ± 2882 | 0 ± 0 | [154, 151, 259] |
| P2 fixed penalty | 221.2 ± 141.3 | 3989 ± 2848 | 35 ± 76 | [190, 186, 287] |
| P4 budget-conditioned | 220.0 ± 143.8 | 3968 ± 2890 | 31 ± 67 | [187, 186, 287] |
| P4 no budget input | 221.9 ± 140.1 | 4007 ± 2811 | 32 ± 71 | [193, 185, 287] |

**Paired within seed, vs P0+ control (beta = 0.9)**

| policy | dT (min) | dE (kJ) | per-seed dE |
|---|---|---|---|
| P1 route filter | -29.9 ± 17.5 | -781 ± 299 | [-649, -886, -808] |
| P2 fixed penalty | 2.5 ± 17.5 | 19 ± 346 | [+167, -109, +0] |
| P3 Lagrangian | 4.7 ± 16.2 | 60 ± 276 | [+188, -8, +0] |
| P4 budget-conditioned | 1.3 ± 12.3 | -2 ± 124 | [+47, -52, +0] |
| P4 no budget input | 3.2 ± 22.5 | 38 ± 384 | [+207, -95, +0] |

## Stage B — blocked / slow roads (18 events), B = 0.9 B0

| policy | T (min) | E (kJ) | over budget (%) | replans | per-seed T |
|---|---|---|---|---|---|
| P0 open-loop (base) | 248.3 ± 160.3 | 4689 ± 3299 | 100 ± 0 | 0.00 ± 0.00 | [210, 212, 323] |
| P0 periodic + road layer | 229.9 ± 129.2 | 4265 ± 2430 | 87 ± 26 | 2.77 ± 2.16 | [201, 199, 290] |
| P0 event + road layer | 230.1 ± 129.4 | 4269 ± 2436 | 87 ± 26 | 0.48 ± 0.10 | [201, 200, 290] |
| P0+ (control) open-loop | 242.6 ± 173.3 | 4580 ± 3550 | 86 ± 34 | 0.00 ± 0.00 | [196, 210, 323] |
| P0+ (control) event + road layer | 226.2 ± 138.5 | 4186 ± 2635 | 75 ± 14 | 0.44 ± 0.19 | [189, 200, 290] |
| P0+ event, road-distance assignment | 226.2 ± 138.5 | 4186 ± 2635 | 75 ± 14 | 0.44 ± 0.19 | [189, 200, 290] |
| P4 open-loop | 243.2 ± 171.4 | 4550 ± 3608 | 85 ± 37 | 0.00 ± 0.00 | [200, 207, 323] |
| P4 periodic + road layer | 226.2 ± 137.1 | 4149 ± 2683 | 74 ± 12 | 2.77 ± 2.17 | [194, 194, 290] |
| P4 event + road layer | 226.3 ± 137.5 | 4151 ± 2691 | 74 ± 12 | 0.60 ± 0.11 | [194, 194, 290] |
| P4 (road-event trained) event | 227.8 ± 135.1 | 4183 ± 2641 | 74 ± 22 | 0.61 ± 0.14 | [190, 203, 290] |
| P3 (beta=0.9) event + road layer | 230.6 ± 128.4 | 4235 ± 2510 | 79 ± 12 | 0.45 ± 0.16 | [201, 200, 290] |
| P4 event, road-distance assignment | 226.3 ± 137.5 | 4151 ± 2691 | 74 ± 12 | 0.60 ± 0.11 | [194, 194, 290] |
| P0 event, road-distance assignment | 230.1 ± 129.4 | 4269 ± 2436 | 87 ± 26 | 0.48 ± 0.10 | [201, 200, 290] |
| P1 route filter, event | 194.0 ± 124.7 | 3350 ± 2158 | 36 ± 12 | 0.41 ± 0.17 | [166, 164, 252] |

**Paired within seed (all events)**

| comparison | dT (min) | dE (kJ) | per-seed dT |
|---|---|---|---|
| P0: event − periodic | 0.2 ± 0.3 | 4 ± 6 | [+0.0, +0.2, +0.3] |
| P4: event − periodic | 0.1 ± 0.4 | 2 ± 9 | [+0.0, +0.0, +0.2] |
| P0: event + layer − open loop | -18.1 ± 31.0 | -420 ± 863 | [-9.7, -12.2, -32.5] |
| P0+ control: event + layer − open loop | -16.4 ± 34.8 | -394 ± 919 | [-6.7, -10.1, -32.5] |
| P4: event + layer − open loop | -16.9 ± 34.9 | -399 ± 921 | [-5.4, -12.8, -32.5] |
| P4 − P0+ control (both event + layer) | 0.1 ± 13.2 | -35 ± 207 | [+5.5, -5.1, -0.0] |
| P3(0.9) − P0+ control (both event + layer) | 4.4 ± 17.9 | 50 ± 410 | [+12.7, +0.5, +0.0] |
| P4 road-event-trained − P4 | 1.5 ± 15.6 | 32 ± 222 | [-3.8, +8.5, +0.0] |
| P1 route filter − P0+ control | -32.2 ± 19.9 | -836 ± 570 | [-23.2, -35.2, -38.3] |

Replans event − periodic (P0): -2.29 ± 2.06 per mission.

