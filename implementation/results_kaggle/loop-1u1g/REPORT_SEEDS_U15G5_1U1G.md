# Multi-seed results — U15G5_1U1G

Training seeds: [0, 1, 2] (n = 3). Each seed = its own imitation warm start, REINFORCE and fine-tuning runs, evaluated on the same 100 test instances (its own P0 defines B0). ± = 95% CI over seeds (t, 2 dof); per-seed values in brackets.

## Stage A — static roads


**beta = 0.9**

| policy | T (min) | E (kJ) | over budget (%) | per-seed T |
|---|---|---|---|---|
| P0 | 216.1 ± 6.5 | 1580 ± 57 | 100 ± 0 | [214, 219, 215] |
| P0+ (control) | 215.6 ± 21.2 | 1540 ± 72 | 73 ± 9 | [211, 225, 210] |
| P1 route filter | 193.2 ± 3.7 | 1373 ± 33 | 44 ± 16 | [194, 194, 192] |
| P2 fixed penalty | 213.6 ± 11.3 | 1557 ± 82 | 81 ± 12 | [209, 213, 218] |
| P4 budget-conditioned | 211.3 ± 3.9 | 1538 ± 40 | 77 ± 5 | [210, 211, 213] |
| P4 no budget input | 211.1 ± 3.8 | 1541 ± 29 | 77 ± 2 | [209, 213, 211] |
| P4x budget + pace | 217.0 ± 16.6 | 1581 ± 128 | 80 ± 21 | [212, 214, 225] |
| P3 Lagrangian | 211.8 ± 9.5 | 1543 ± 71 | 78 ± 6 | [208, 211, 216] |

**beta = 1.0**

| policy | T (min) | E (kJ) | over budget (%) | per-seed T |
|---|---|---|---|---|
| P0 | 216.1 ± 6.5 | 1580 ± 57 | 0 ± 0 | [214, 219, 215] |
| P0+ (control) | 215.6 ± 21.2 | 1540 ± 72 | 43 ± 2 | [211, 225, 210] |
| P1 route filter | 192.9 ± 3.5 | 1374 ± 34 | 0 ± 0 | [194, 194, 191] |
| P2 fixed penalty | 213.6 ± 11.3 | 1557 ± 82 | 39 ± 17 | [209, 213, 218] |
| P4 budget-conditioned | 211.3 ± 3.9 | 1538 ± 40 | 37 ± 17 | [210, 211, 213] |
| P4 no budget input | 211.1 ± 3.8 | 1541 ± 29 | 42 ± 16 | [209, 213, 211] |
| P4x budget + pace | 217.0 ± 16.6 | 1581 ± 128 | 46 ± 37 | [212, 214, 225] |

**Paired within seed, vs P0+ control (beta = 0.9)**

| policy | dT (min) | dE (kJ) | per-seed dE |
|---|---|---|---|
| P1 route filter | -22.4 ± 19.2 | -167 ± 50 | [-147, -188, -166] |
| P2 fixed penalty | -2.0 ± 24.7 | 17 ± 105 | [+1, -15, +65] |
| P3 Lagrangian | -3.7 ± 24.5 | 4 ± 104 | [-5, -33, +49] |
| P4 budget-conditioned | -4.2 ± 21.4 | -1 ± 69 | [-2, -29, +26] |
| P4 no budget input | -4.5 ± 18.4 | 1 ± 58 | [+6, -24, +21] |

## Stage B — blocked / slow roads (18 events), B = 0.9 B0

| policy | T (min) | E (kJ) | over budget (%) | replans | per-seed T |
|---|---|---|---|---|---|
| P0 open-loop (base) | 235.3 ± 8.4 | 1845 ± 81 | 100 ± 0 | 0.00 ± 0.00 | [233, 239, 233] |
| P0 periodic + road layer | 232.4 ± 6.6 | 1806 ± 66 | 99 ± 0 | 4.67 ± 0.11 | [231, 235, 230] |
| P0 event + road layer | 232.4 ± 6.6 | 1806 ± 66 | 99 ± 0 | 0.61 ± 0.01 | [231, 235, 230] |
| P0+ (control) open-loop | 228.9 ± 21.1 | 1724 ± 70 | 82 ± 8 | 0.00 ± 0.00 | [225, 239, 223] |
| P0+ (control) event + road layer | 222.8 ± 2.6 | 1692 ± 33 | 82 ± 9 | 0.45 ± 0.05 | [223, 224, 222] |
| P0+ event, road-distance assignment | 222.8 ± 2.6 | 1692 ± 33 | 82 ± 9 | 0.45 ± 0.05 | [223, 224, 222] |
| P4 open-loop | 224.7 ± 2.7 | 1723 ± 21 | 85 ± 5 | 0.00 ± 0.00 | [224, 225, 226] |
| P4 periodic + road layer | 222.8 ± 3.3 | 1698 ± 25 | 85 ± 5 | 4.58 ± 0.12 | [222, 222, 224] |
| P4 event + road layer | 222.8 ± 3.3 | 1698 ± 25 | 85 ± 5 | 0.80 ± 0.05 | [222, 222, 224] |
| P4 (road-event trained) event | 223.1 ± 7.8 | 1693 ± 61 | 83 ± 3 | 0.79 ± 0.00 | [219, 224, 225] |
| P3 (beta=0.9) event + road layer | 222.7 ± 8.0 | 1695 ± 47 | 84 ± 7 | 0.43 ± 0.03 | [220, 222, 226] |
| P4 event, road-distance assignment | 222.8 ± 3.3 | 1698 ± 25 | 85 ± 5 | 0.80 ± 0.05 | [222, 222, 224] |
| P0 event, road-distance assignment | 232.4 ± 6.6 | 1806 ± 66 | 99 ± 0 | 0.61 ± 0.01 | [231, 235, 230] |
| P1 route filter, event | 204.9 ± 4.1 | 1512 ± 41 | 63 ± 6 | 0.64 ± 0.04 | [205, 206, 203] |

**Paired within seed (all events)**

| comparison | dT (min) | dE (kJ) | per-seed dT |
|---|---|---|---|
| P0: event − periodic | 0.0 ± 0.0 | 0 ± 0 | [+0.0, +0.0, +0.0] |
| P4: event − periodic | 0.0 ± 0.0 | 0 ± 0 | [+0.0, +0.0, +0.0] |
| P0: event + layer − open loop | -2.9 ± 2.3 | -39 ± 26 | [-2.0, -3.8, -3.1] |
| P0+ control: event + layer − open loop | -6.2 ± 19.1 | -32 ± 39 | [-1.8, -15.1, -1.7] |
| P4: event + layer − open loop | -1.9 ± 1.3 | -24 ± 9 | [-1.7, -2.5, -1.5] |
| P4 − P0+ control (both event + layer) | 0.0 ± 5.9 | 6 ± 51 | [-1.1, -1.6, +2.7] |
| P3(0.9) − P0+ control (both event + layer) | -0.0 ± 10.3 | 3 ± 74 | [-3.0, -1.8, +4.7] |
| P4 road-event-trained − P4 | 0.3 ± 6.1 | -5 ± 40 | [-2.5, +2.2, +1.1] |
| P1 route filter − P0+ control | -17.9 ± 1.5 | -179 ± 8 | [-18.0, -17.3, -18.5] |

Replans event − periodic (P0): -4.06 ± 0.10 per mission.

