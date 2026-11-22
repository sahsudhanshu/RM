# Multi-seed results — U15G5_2U1G_paper

Training seeds: [0, 1, 2] (n = 3). Each seed = its own imitation warm start, REINFORCE and fine-tuning runs, evaluated on the same 100 test instances (its own P0 defines B0). ± = 95% CI over seeds (t, 2 dof); per-seed values in brackets.

## Stage A — static roads


**beta = 0.9**

| policy | T (min) | E (kJ) | over budget (%) | per-seed T |
|---|---|---|---|---|
| P0 | 210.5 ± 15.6 | 3913 ± 365 | 100 ± 0 | [211, 217, 204] |
| P0+ (control) | 233.1 ± 93.2 | 4299 ± 1551 | 71 ± 14 | [212, 276, 211] |
| P1 route filter | 170.0 ± 6.8 | 2919 ± 141 | 21 ± 14 | [169, 173, 168] |
| P2 fixed penalty | 231.8 ± 91.8 | 4248 ± 1494 | 68 ± 12 | [274, 213, 208] |
| P4 budget-conditioned | 211.5 ± 8.4 | 3892 ± 207 | 66 ± 18 | [214, 208, 213] |
| P4 no budget input | 226.0 ± 52.0 | 4168 ± 846 | 70 ± 7 | [249, 220, 209] |
| P3 Lagrangian | 228.3 ± 25.9 | 4182 ± 401 | 71 ± 14 | [220, 240, 225] |

**beta = 1.0**

| policy | T (min) | E (kJ) | over budget (%) | per-seed T |
|---|---|---|---|---|
| P0 | 210.5 ± 15.6 | 3913 ± 365 | 0 ± 0 | [211, 217, 204] |
| P0+ (control) | 233.1 ± 93.2 | 4299 ± 1551 | 55 ± 12 | [212, 276, 211] |
| P1 route filter | 169.5 ± 6.8 | 2935 ± 143 | 0 ± 0 | [169, 173, 167] |
| P2 fixed penalty | 231.8 ± 91.8 | 4248 ± 1494 | 48 ± 13 | [274, 213, 208] |
| P4 budget-conditioned | 211.5 ± 8.4 | 3892 ± 207 | 48 ± 24 | [214, 208, 213] |
| P4 no budget input | 226.0 ± 52.0 | 4168 ± 846 | 56 ± 3 | [249, 220, 209] |

**Paired within seed, vs P0+ control (beta = 0.9)**

| policy | dT (min) | dE (kJ) | per-seed dE |
|---|---|---|---|
| P1 route filter | -63.1 ± 86.8 | -1380 ± 1414 | [-1046, -2037, -1056] |
| P2 fixed penalty | -1.3 ± 156.1 | -51 ± 2542 | [+991, -1054, -92] |
| P3 Lagrangian | -4.8 ± 68.3 | -117 ± 1170 | [+105, -658, +203] |
| P4 budget-conditioned | -21.5 ± 101.3 | -407 ± 1746 | [-42, -1217, +38] |
| P4 no budget input | -7.1 ± 117.4 | -131 ± 1864 | [+587, -910, -71] |

## Stage B — blocked / slow roads (18 events), B = 0.9 B0

| policy | T (min) | E (kJ) | over budget (%) | replans | per-seed T |
|---|---|---|---|---|---|
| P0 open-loop (base) | 234.3 ± 19.5 | 4503 ± 466 | 100 ± 0 | 0.00 ± 0.00 | [235, 242, 226] |
| P0 periodic + road layer | 218.9 ± 13.3 | 4154 ± 234 | 88 ± 7 | 2.40 ± 0.14 | [218, 225, 214] |
| P0 event + road layer | 218.9 ± 13.0 | 4154 ± 230 | 88 ± 7 | 0.38 ± 0.02 | [218, 225, 214] |
| P0+ (control) open-loop | 254.3 ± 94.6 | 4836 ± 1614 | 79 ± 11 | 0.00 ± 0.00 | [233, 298, 232] |
| P0+ (control) event + road layer | 224.8 ± 14.4 | 4291 ± 182 | 73 ± 14 | 0.38 ± 0.05 | [221, 231, 222] |
| P0+ event, road-distance assignment | 224.8 ± 14.4 | 4291 ± 182 | 73 ± 14 | 0.38 ± 0.05 | [221, 231, 222] |
| P4 open-loop | 230.9 ± 7.2 | 4381 ± 253 | 74 ± 20 | 0.00 ± 0.00 | [231, 228, 234] |
| P4 periodic + road layer | 220.3 ± 11.7 | 4156 ± 239 | 69 ± 25 | 2.49 ± 0.66 | [224, 215, 222] |
| P4 event + road layer | 220.3 ± 11.6 | 4157 ± 238 | 69 ± 25 | 0.58 ± 0.10 | [224, 215, 222] |
| P4 (road-event trained) event | 224.9 ± 7.8 | 4245 ± 68 | 74 ± 13 | 0.62 ± 0.16 | [227, 221, 226] |
| P3 (beta=0.9) event + road layer | 229.8 ± 27.3 | 4328 ± 474 | 74 ± 20 | 0.39 ± 0.07 | [232, 218, 239] |
| P4 event, road-distance assignment | 220.3 ± 11.6 | 4157 ± 238 | 69 ± 25 | 0.58 ± 0.10 | [224, 215, 222] |
| P0 event, road-distance assignment | 218.9 ± 13.0 | 4154 ± 230 | 88 ± 7 | 0.38 ± 0.02 | [218, 225, 214] |
| P1 route filter, event | 178.2 ± 7.8 | 3138 ± 183 | 33 ± 14 | 0.34 ± 0.09 | [176, 182, 177] |

**Paired within seed (all events)**

| comparison | dT (min) | dE (kJ) | per-seed dT |
|---|---|---|---|
| P0: event − periodic | 0.0 ± 0.3 | 1 ± 4 | [+0.0, -0.1, +0.2] |
| P4: event − periodic | 0.0 ± 0.1 | 1 ± 1 | [+0.0, +0.1, +0.0] |
| P0: event + layer − open loop | -15.3 ± 7.4 | -348 ± 243 | [-16.8, -17.3, -11.9] |
| P0+ control: event + layer − open loop | -29.5 ± 80.3 | -545 ± 1445 | [-12.0, -66.8, -9.8] |
| P4: event + layer − open loop | -10.6 ± 7.3 | -224 ± 61 | [-7.3, -13.0, -11.5] |
| P4 − P0+ control (both event + layer) | -4.4 ± 25.9 | -135 ± 379 | [+2.8, -16.4, +0.3] |
| P3(0.9) − P0+ control (both event + layer) | 5.0 ± 40.9 | 37 ± 623 | [+11.4, -13.6, +17.4] |
| P4 road-event-trained − P4 | 4.6 ± 3.8 | 89 ± 182 | [+3.3, +6.3, +4.1] |
| P1 route filter − P0+ control | -46.6 ± 6.6 | -1153 ± 16 | [-44.8, -49.6, -45.3] |

Replans event − periodic (P0): -2.03 ± 0.12 per mission.

