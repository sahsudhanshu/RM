# Multi-seed results — U15G5_2U2G

Training seeds: [0, 1, 2] (n = 3). Each seed = its own imitation warm start, REINFORCE and fine-tuning runs, evaluated on the same 100 test instances (its own P0 defines B0). ± = 95% CI over seeds (t, 2 dof); per-seed values in brackets.

## Stage A — static roads


**beta = 0.9**

| policy | T (min) | E (kJ) | over budget (%) | per-seed T |
|---|---|---|---|---|
| P0 | 183.9 ± 78.9 | 3005 ± 1152 | 100 ± 0 | [158, 175, 219] |
| P0+ (control) | 179.7 ± 85.2 | 2956 ± 1240 | 80 ± 48 | [161, 159, 219] |
| P1 route filter | 153.5 ± 98.0 | 2292 ± 1307 | 26 ± 49 | [129, 133, 199] |
| P2 fixed penalty | 185.4 ± 72.9 | 3024 ± 1091 | 79 ± 45 | [168, 169, 219] |
| P4 budget-conditioned | 182.4 ± 79.4 | 2977 ± 1196 | 79 ± 47 | [164, 164, 219] |
| P4 no budget input | 179.9 ± 85.4 | 2938 ± 1283 | 80 ± 46 | [164, 156, 219] |
| P4x budget + pace | 183.5 ± 77.0 | 2995 ± 1157 | 78 ± 49 | [165, 166, 219] |
| P3 Lagrangian | 182.7 ± 79.5 | 2993 ± 1175 | 81 ± 45 | [168, 160, 219] |

**beta = 1.0**

| policy | T (min) | E (kJ) | over budget (%) | per-seed T |
|---|---|---|---|---|
| P0 | 183.9 ± 78.9 | 3005 ± 1152 | 0 ± 0 | [158, 175, 219] |
| P0+ (control) | 179.7 ± 85.2 | 2956 ± 1240 | 32 ± 70 | [161, 159, 219] |
| P1 route filter | 152.1 ± 95.5 | 2315 ± 1354 | 0 ± 0 | [128, 132, 196] |
| P2 fixed penalty | 185.4 ± 72.9 | 3024 ± 1091 | 33 ± 71 | [168, 169, 219] |
| P4 budget-conditioned | 182.4 ± 79.4 | 2977 ± 1196 | 33 ± 72 | [164, 164, 219] |
| P4 no budget input | 179.9 ± 85.4 | 2938 ± 1283 | 29 ± 66 | [164, 156, 219] |
| P4x budget + pace | 183.5 ± 77.0 | 2995 ± 1157 | 32 ± 69 | [165, 166, 219] |

**Paired within seed, vs P0+ control (beta = 0.9)**

| policy | dT (min) | dE (kJ) | per-seed dE |
|---|---|---|---|
| P1 route filter | -26.2 ± 14.8 | -664 ± 97 | [-707, -653, -631] |
| P2 fixed penalty | 5.7 ± 13.0 | 68 ± 187 | [+56, +149, +0] |
| P3 Lagrangian | 2.9 ± 10.2 | 37 ± 143 | [+103, +7, +0] |
| P4 budget-conditioned | 2.7 ± 6.3 | 21 ± 45 | [+34, +28, +0] |
| P4 no budget input | 0.2 ± 8.1 | -18 ± 74 | [-1, -52, +0] |

## Stage B — blocked / slow roads (18 events), B = 0.9 B0

| policy | T (min) | E (kJ) | over budget (%) | replans | per-seed T |
|---|---|---|---|---|---|
| P0 open-loop (base) | 206.0 ± 80.0 | 3536 ± 1178 | 100 ± 0 | 0.00 ± 0.00 | [180, 196, 242] |
| P0 periodic + road layer | 194.9 ± 75.3 | 3264 ± 1068 | 89 ± 7 | 2.85 ± 2.14 | [171, 184, 229] |
| P0 event + road layer | 194.1 ± 71.1 | 3253 ± 998 | 89 ± 8 | 0.42 ± 0.08 | [171, 185, 226] |
| P0+ (control) open-loop | 197.1 ± 96.8 | 3372 ± 1522 | 84 ± 41 | 0.00 ± 0.00 | [177, 172, 242] |
| P0+ (control) event + road layer | 187.4 ± 83.7 | 3156 ± 1214 | 77 ± 34 | 0.38 ± 0.19 | [170, 166, 226] |
| P0+ event, road-distance assignment | 186.4 ± 83.8 | 3124 ± 1229 | 76 ± 31 | 0.35 ± 0.18 | [169, 165, 225] |
| P4 open-loop | 199.3 ± 91.8 | 3382 ± 1497 | 83 ± 38 | 0.00 ± 0.00 | [179, 177, 242] |
| P4 periodic + road layer | 190.7 ± 82.6 | 3185 ± 1240 | 78 ± 24 | 2.84 ± 2.10 | [172, 171, 229] |
| P4 event + road layer | 190.9 ± 82.7 | 3190 ± 1242 | 78 ± 24 | 0.62 ± 0.23 | [172, 171, 229] |
| P4 (road-event trained) event | 193.6 ± 76.9 | 3205 ± 1206 | 78 ± 21 | 0.63 ± 0.20 | [176, 175, 229] |
| P3 (beta=0.9) event + road layer | 189.5 ± 79.9 | 3171 ± 1196 | 78 ± 23 | 0.35 ± 0.25 | [176, 166, 226] |
| P4 event, road-distance assignment | 190.1 ± 82.4 | 3167 ± 1246 | 76 ± 25 | 0.61 ± 0.22 | [171, 171, 228] |
| P0 event, road-distance assignment | 192.9 ± 71.3 | 3221 ± 1015 | 87 ± 5 | 0.38 ± 0.10 | [171, 183, 225] |
| P1 route filter, event | 161.0 ± 99.1 | 2443 ± 1319 | 34 ± 33 | 0.36 ± 0.41 | [137, 139, 207] |

**Paired within seed (all events)**

| comparison | dT (min) | dE (kJ) | per-seed dT |
|---|---|---|---|
| P0: event − periodic | -0.8 ± 4.4 | -11 ± 71 | [+0.2, +0.3, -2.8] |
| P4: event − periodic | 0.2 ± 0.2 | 6 ± 2 | [+0.2, +0.1, +0.3] |
| P0: event + layer − open loop | -11.9 ± 9.2 | -283 ± 202 | [-8.4, -11.7, -15.7] |
| P0+ control: event + layer − open loop | -9.6 ± 13.2 | -216 ± 309 | [-7.0, -6.2, -15.7] |
| P4: event + layer − open loop | -8.4 ± 9.2 | -192 ± 255 | [-6.7, -5.9, -12.6] |
| P4 − P0+ control (both event + layer) | 3.4 ± 4.8 | 35 ± 57 | [+1.7, +5.5, +3.1] |
| P3(0.9) − P0+ control (both event + layer) | 2.1 ± 7.5 | 15 ± 88 | [+5.6, +0.7, +0.0] |
| P4 road-event-trained − P4 | 2.7 ± 5.9 | 14 ± 123 | [+4.3, +3.9, +0.0] |
| P1 route filter − P0+ control | -26.4 ± 17.4 | -713 ± 132 | [-33.0, -27.0, -19.1] |

Replans event − periodic (P0): -2.43 ± 2.06 per mission.

