# Multi-seed results — U15G5_4U2G

Training seeds: [0, 1, 2] (n = 3). Each seed = its own imitation warm start, REINFORCE and fine-tuning runs, evaluated on the same 100 test instances (its own P0 defines B0). ± = 95% CI over seeds (t, 2 dof); per-seed values in brackets.

## Stage A — static roads


**beta = 0.9**

| policy | T (min) | E (kJ) | over budget (%) | per-seed T |
|---|---|---|---|---|
| P0 | 191.8 ± 40.6 | 6819 ± 1765 | 100 ± 0 | [182, 182, 211] |
| P0+ (control) | 195.7 ± 32.5 | 6988 ± 1402 | 80 ± 43 | [189, 187, 211] |
| P1 route filter | 147.5 ± 49.6 | 4730 ± 1676 | 19 ± 26 | [139, 133, 170] |
| P2 fixed penalty | 194.1 ± 35.8 | 6990 ± 1418 | 84 ± 39 | [185, 187, 211] |
| P4 budget-conditioned | 336.4 ± 607.9 | 11985 ± 21517 | 89 ± 25 | [618, 180, 211] |
| P4 no budget input | 195.3 ± 33.8 | 6967 ± 1467 | 79 ± 46 | [190, 185, 211] |
| P4x budget + pace | 192.4 ± 40.8 | 6857 ± 1740 | 82 ± 40 | [187, 179, 211] |
| P3 Lagrangian | 193.3 ± 37.5 | 6904 ± 1587 | 83 ± 44 | [184, 185, 211] |

**beta = 1.0**

| policy | T (min) | E (kJ) | over budget (%) | per-seed T |
|---|---|---|---|---|
| P0 | 191.8 ± 40.6 | 6819 ± 1765 | 0 ± 0 | [182, 182, 211] |
| P0+ (control) | 195.7 ± 32.5 | 6988 ± 1402 | 38 ± 83 | [189, 187, 211] |
| P1 route filter | 146.2 ± 48.2 | 4767 ± 1709 | 0 ± 0 | [139, 132, 168] |
| P2 fixed penalty | 194.1 ± 35.8 | 6990 ± 1418 | 36 ± 79 | [185, 187, 211] |
| P4 budget-conditioned | 336.4 ± 607.9 | 11985 ± 21517 | 36 ± 93 | [618, 180, 211] |
| P4 no budget input | 195.3 ± 33.8 | 6967 ± 1467 | 36 ± 78 | [190, 185, 211] |
| P4x budget + pace | 192.4 ± 40.8 | 6857 ± 1740 | 26 ± 61 | [187, 179, 211] |

**Paired within seed, vs P0+ control (beta = 0.9)**

| policy | dT (min) | dE (kJ) | per-seed dE |
|---|---|---|---|
| P1 route filter | -48.2 ± 17.5 | -2258 ± 383 | [-2198, -2434, -2143] |
| P2 fixed penalty | -1.5 ± 6.8 | 2 ± 248 | [-97, +103, +0] |
| P3 Lagrangian | -2.4 ± 6.5 | -84 ± 224 | [-179, -71, +0] |
| P4 budget-conditioned | 140.7 ± 620.7 | 4997 ± 22156 | [+15294, -303, +0] |
| P4 no budget input | -0.4 ± 3.2 | -21 ± 238 | [+62, -126, +0] |

## Stage B — blocked / slow roads (18 events), B = 0.9 B0

| policy | T (min) | E (kJ) | over budget (%) | replans | per-seed T |
|---|---|---|---|---|---|
| P0 open-loop (base) | 219.3 ± 42.8 | 7853 ± 1954 | 100 ± 0 | 0.00 ± 0.00 | [210, 208, 239] |
| P0 periodic + road layer | 200.6 ± 38.8 | 7080 ± 1802 | 77 ± 14 | 1.29 ± 0.65 | [193, 190, 219] |
| P0 event + road layer | 200.7 ± 38.8 | 7082 ± 1807 | 78 ± 14 | 0.22 ± 0.08 | [193, 190, 219] |
| P0+ (control) open-loop | 218.0 ± 45.4 | 7854 ± 1956 | 84 ± 36 | 0.00 ± 0.00 | [208, 207, 239] |
| P0+ (control) event + road layer | 202.0 ± 35.8 | 7155 ± 1639 | 69 ± 11 | 0.22 ± 0.08 | [195, 193, 219] |
| P0+ event, road-distance assignment | 202.3 ± 33.4 | 7161 ± 1491 | 70 ± 9 | 0.23 ± 0.10 | [196, 193, 218] |
| P4 open-loop | 370.5 ± 647.0 | 13512 ± 23910 | 91 ± 20 | 0.00 ± 0.00 | [670, 202, 239] |
| P4 periodic + road layer | 205.5 ± 45.3 | 7323 ± 2191 | 71 ± 18 | 1.50 ± 0.69 | [213, 185, 219] |
| P4 event + road layer | 205.6 ± 45.3 | 7328 ± 2191 | 72 ± 18 | 0.35 ± 0.18 | [213, 185, 219] |
| P4 (road-event trained) event | 196.0 ± 49.2 | 6899 ± 2207 | 66 ± 20 | 0.30 ± 0.10 | [182, 188, 219] |
| P3 (beta=0.9) event + road layer | 198.2 ± 44.4 | 6976 ± 2021 | 68 ± 13 | 0.22 ± 0.09 | [186, 190, 219] |
| P4 event, road-distance assignment | 204.7 ± 44.0 | 7266 ± 2092 | 71 ± 14 | 0.35 ± 0.14 | [212, 185, 218] |
| P0 event, road-distance assignment | 200.3 ± 37.6 | 7055 ± 1717 | 77 ± 11 | 0.23 ± 0.10 | [192, 191, 218] |
| P1 route filter, event | 152.9 ± 32.5 | 4912 ± 1206 | 25 ± 14 | 0.20 ± 0.09 | [152, 140, 166] |

**Paired within seed (all events)**

| comparison | dT (min) | dE (kJ) | per-seed dT |
|---|---|---|---|
| P0: event − periodic | 0.0 ± 0.1 | 3 ± 7 | [+0.0, +0.0, +0.1] |
| P4: event − periodic | 0.1 ± 0.0 | 5 ± 2 | [+0.1, +0.1, +0.1] |
| P0: event + layer − open loop | -18.6 ± 4.2 | -771 ± 290 | [-17.2, -18.2, -20.5] |
| P0+ control: event + layer − open loop | -16.0 ± 9.9 | -699 ± 434 | [-12.9, -14.6, -20.5] |
| P4: event + layer − open loop | -164.9 ± 628.5 | -6184 ± 22904 | [-457.1, -17.1, -20.5] |
| P4 − P0+ control (both event + layer) | 3.6 ± 33.7 | 173 ± 1665 | [+18.6, -7.9, +0.0] |
| P3(0.9) − P0+ control (both event + layer) | -3.9 ± 11.8 | -178 ± 516 | [-9.2, -2.5, +0.0] |
| P4 road-event-trained − P4 | -9.6 ± 47.7 | -429 ± 2323 | [-31.7, +3.0, +0.0] |
| P1 route filter − P0+ control | -49.1 ± 14.4 | -2243 ± 824 | [-42.4, -52.5, -52.4] |

Replans event − periodic (P0): -1.06 ± 0.57 per mission.

