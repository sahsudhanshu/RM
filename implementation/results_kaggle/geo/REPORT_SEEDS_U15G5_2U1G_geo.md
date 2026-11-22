# Multi-seed results — U15G5_2U1G_geo

Training seeds: [0, 1, 2] (n = 3). Each seed = its own imitation warm start, REINFORCE and fine-tuning runs, evaluated on the same 100 test instances (its own P0 defines B0). ± = 95% CI over seeds (t, 2 dof); per-seed values in brackets.

## Stage A — static roads


**beta = 0.9**

| policy | T (min) | E (kJ) | over budget (%) | per-seed T |
|---|---|---|---|---|
| P0 | 355.3 ± 318.4 | 7413 ± 7240 | 100 ± 0 | [286, 277, 503] |
| P0+ (control) | 363.3 ± 301.2 | 7564 ± 6915 | 81 ± 41 | [289, 298, 503] |
| P1 route filter | 277.0 ± 258.5 | 5320 ± 5355 | 23 ± 10 | [217, 217, 397] |
| P2 fixed penalty | 356.9 ± 315.1 | 7416 ± 7233 | 84 ± 35 | [279, 288, 503] |
| P4 budget-conditioned | 358.0 ± 312.6 | 7468 ± 7118 | 84 ± 36 | [287, 283, 503] |
| P4 no budget input | 369.9 ± 287.1 | 7705 ± 6609 | 85 ± 33 | [300, 306, 503] |
| P3 Lagrangian | 362.4 ± 304.5 | 7555 ± 6964 | 84 ± 35 | [304, 280, 503] |

**beta = 1.0**

| policy | T (min) | E (kJ) | over budget (%) | per-seed T |
|---|---|---|---|---|
| P0 | 355.3 ± 318.4 | 7413 ± 7240 | 0 ± 0 | [286, 277, 503] |
| P0+ (control) | 363.3 ± 301.2 | 7564 ± 6915 | 38 ± 81 | [289, 298, 503] |
| P1 route filter | 276.6 ± 258.1 | 5329 ± 5364 | 0 ± 0 | [217, 216, 397] |
| P2 fixed penalty | 356.9 ± 315.1 | 7416 ± 7233 | 33 ± 71 | [279, 288, 503] |
| P4 budget-conditioned | 358.0 ± 312.6 | 7468 ± 7118 | 34 ± 76 | [287, 283, 503] |
| P4 no budget input | 369.9 ± 287.1 | 7705 ± 6609 | 37 ± 83 | [300, 306, 503] |

**Paired within seed, vs P0+ control (beta = 0.9)**

| policy | dT (min) | dE (kJ) | per-seed dE |
|---|---|---|---|
| P1 route filter | -86.3 ± 44.2 | -2244 ± 1577 | [-1777, -1988, -2967] |
| P2 fixed penalty | -6.4 ± 13.8 | -149 ± 323 | [-205, -241, +0] |
| P3 Lagrangian | -0.9 ± 40.6 | -9 ± 940 | [+365, -392, +0] |
| P4 budget-conditioned | -5.3 ± 19.8 | -96 ± 414 | [-0, -289, +0] |
| P4 no budget input | 6.5 ± 14.5 | 140 ± 355 | [+286, +135, +0] |

## Stage B — blocked / slow roads (18 events), B = 0.9 B0

| policy | T (min) | E (kJ) | over budget (%) | replans | per-seed T |
|---|---|---|---|---|---|
| P0 open-loop (base) | 397.3 ± 360.7 | 8486 ± 8356 | 100 ± 0 | 0.00 ± 0.00 | [318, 309, 565] |
| P0 periodic + road layer | 356.2 ± 273.0 | 7464 ± 5967 | 80 ± 28 | 3.15 ± 2.84 | [295, 290, 483] |
| P0 event + road layer | 356.2 ± 272.9 | 7464 ± 5966 | 80 ± 28 | 0.44 ± 0.18 | [295, 290, 483] |
| P0+ (control) open-loop | 403.2 ± 348.1 | 8584 ± 8146 | 86 ± 31 | 0.00 ± 0.00 | [317, 327, 565] |
| P0+ (control) event + road layer | 363.3 ± 257.9 | 7593 ± 5695 | 71 ± 13 | 0.45 ± 0.18 | [298, 308, 483] |
| P0+ event, road-distance assignment | 363.3 ± 257.9 | 7593 ± 5695 | 71 ± 13 | 0.45 ± 0.18 | [298, 308, 483] |
| P4 open-loop | 397.9 ± 359.4 | 8496 ± 8331 | 87 ± 29 | 0.00 ± 0.00 | [317, 312, 565] |
| P4 periodic + road layer | 356.2 ± 272.9 | 7439 ± 6021 | 70 ± 7 | 3.25 ± 2.64 | [293, 293, 483] |
| P4 event + road layer | 356.2 ± 272.9 | 7439 ± 6021 | 70 ± 7 | 0.64 ± 0.08 | [293, 293, 483] |
| P4 (road-event trained) event | 360.3 ± 264.0 | 7544 ± 5796 | 72 ± 12 | 0.66 ± 0.10 | [296, 302, 483] |
| P3 (beta=0.9) event + road layer | 358.4 ± 268.9 | 7497 ± 5918 | 70 ± 7 | 0.44 ± 0.18 | [304, 288, 483] |
| P4 event, road-distance assignment | 356.2 ± 272.9 | 7439 ± 6021 | 70 ± 7 | 0.64 ± 0.08 | [293, 293, 483] |
| P0 event, road-distance assignment | 356.2 ± 272.9 | 7464 ± 5966 | 80 ± 28 | 0.44 ± 0.18 | [295, 290, 483] |
| P1 route filter, event | 264.3 ± 155.0 | 5057 ± 2859 | 27 ± 33 | 0.36 ± 0.06 | [230, 227, 336] |

**Paired within seed (all events)**

| comparison | dT (min) | dE (kJ) | per-seed dT |
|---|---|---|---|
| P0: event − periodic | 0.0 ± 0.1 | 0 ± 2 | [+0.0, +0.0, -0.0] |
| P4: event − periodic | 0.0 ± 0.1 | 0 ± 1 | [+0.0, +0.0, +0.0] |
| P0: event + layer − open loop | -41.1 ± 87.9 | -1022 ± 2392 | [-22.8, -18.7, -81.9] |
| P0+ control: event + layer − open loop | -39.9 ± 90.3 | -992 ± 2452 | [-19.1, -18.8, -81.9] |
| P4: event + layer − open loop | -41.7 ± 86.7 | -1057 ± 2316 | [-24.1, -19.1, -81.9] |
| P4 − P0+ control (both event + layer) | -7.1 ± 19.6 | -154 ± 399 | [-5.8, -15.6, +0.0] |
| P3(0.9) − P0+ control (both event + layer) | -4.9 ± 34.8 | -96 ± 821 | [+6.0, -20.7, +0.0] |
| P4 road-event-trained − P4 | 4.1 ± 10.8 | 105 ± 255 | [+3.8, +8.7, +0.0] |
| P1 route filter − P0+ control | -99.0 ± 103.8 | -2536 ± 2854 | [-68.6, -81.6, -146.6] |

Replans event − periodic (P0): -2.71 ± 2.66 per mission.

