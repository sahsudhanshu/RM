# Multi-seed results — U15G5_2U1G

Training seeds: [0, 1, 2] (n = 3). Each seed = its own imitation warm start, REINFORCE and fine-tuning runs, evaluated on the same 100 test instances (its own P0 defines B0). ± = 95% CI over seeds (t, 2 dof); per-seed values in brackets.

## Stage A — static roads


**beta = 0.9**

| policy | T (min) | E (kJ) | over budget (%) | per-seed T |
|---|---|---|---|---|
| P0 | 219.6 ± 144.7 | 3969 ± 2888 | 100 ± 0 | [186, 186, 287] |
| P0+ (control) | 220.2 ± 144.5 | 3971 ± 2896 | 78 ± 52 | [180, 194, 287] |
| P1 route filter | 188.8 ± 154.3 | 3189 ± 2830 | 30 ± 43 | [155, 151, 261] |
| P4 budget-conditioned | 221.8 ± 140.7 | 4006 ± 2825 | 79 ± 50 | [195, 184, 287] |
| Distilled search, time only | 219.8 ± 144.4 | 3994 ± 2837 | 85 ± 32 | [184, 189, 287] |
| Distilled search, budget-blind | 218.6 ± 147.3 | 3942 ± 2951 | 77 ± 51 | [188, 181, 287] |
| Distilled search, budget-aware | 220.4 ± 143.8 | 3984 ± 2869 | 78 ± 47 | [193, 181, 287] |
| P3 high-cap lambda (0.9) | 220.8 ± 142.5 | 4002 ± 2822 | 80 ± 44 | [192, 184, 287] |

**beta = 1.0**

| policy | T (min) | E (kJ) | over budget (%) | per-seed T |
|---|---|---|---|---|
| P0 | 219.6 ± 144.7 | 3969 ± 2888 | 0 ± 0 | [186, 186, 287] |
| P0+ (control) | 220.2 ± 144.5 | 3971 ± 2896 | 32 ± 71 | [180, 194, 287] |
| P1 route filter | 187.9 ± 151.9 | 3209 ± 2882 | 0 ± 0 | [154, 151, 259] |
| P4 budget-conditioned | 221.8 ± 140.7 | 4006 ± 2825 | 33 ± 72 | [195, 184, 287] |
| Distilled search, time only | 219.8 ± 144.4 | 3994 ± 2837 | 30 ± 67 | [184, 189, 287] |
| Distilled search, budget-blind | 218.6 ± 147.3 | 3942 ± 2951 | 30 ± 64 | [188, 181, 287] |
| Distilled search, budget-aware | 220.4 ± 143.8 | 3984 ± 2869 | 33 ± 72 | [193, 181, 287] |
| P3 high-cap lambda (1.0) | 219.4 ± 146.8 | 3962 ± 2932 | 31 ± 70 | [177, 194, 287] |

**Paired within seed, vs P0+ control (beta = 0.9)**

| policy | dT (min) | dE (kJ) | per-seed dE |
|---|---|---|---|
| P1 route filter | -31.4 ± 23.6 | -782 ± 383 | [-617, -922, -808] |
| P4 budget-conditioned | 1.6 ± 30.4 | 35 ± 597 | [+291, -186, +0] |

## Stage B — blocked / slow roads (18 events), B = 0.9 B0

| policy | T (min) | E (kJ) | over budget (%) | replans | per-seed T |
|---|---|---|---|---|---|
| P0 open-loop (base) | 248.3 ± 160.3 | 4689 ± 3299 | 100 ± 0 | 0.00 ± 0.00 | [210, 212, 323] |
| P0 periodic + road layer | 229.9 ± 129.2 | 4265 ± 2430 | 87 ± 26 | 2.77 ± 2.16 | [201, 199, 290] |
| P0 event + road layer | 230.1 ± 129.4 | 4269 ± 2436 | 87 ± 26 | 0.48 ± 0.10 | [201, 200, 290] |
| P0+ (control) open-loop | 242.7 ± 173.3 | 4545 ± 3627 | 84 ± 38 | 0.00 ± 0.00 | [195, 211, 323] |
| P0+ (control) event + road layer | 226.2 ± 139.8 | 4146 ± 2741 | 73 ± 17 | 0.43 ± 0.22 | [185, 204, 290] |
| P0+ event, road-distance assignment | 226.2 ± 139.8 | 4146 ± 2741 | 73 ± 17 | 0.43 ± 0.22 | [185, 204, 290] |
| P4 open-loop | 244.6 ± 169.0 | 4593 ± 3528 | 84 ± 37 | 0.00 ± 0.00 | [212, 199, 323] |
| P4 periodic + road layer | 229.6 ± 130.9 | 4217 ± 2566 | 74 ± 14 | 2.85 ± 1.98 | [206, 193, 290] |
| P4 event + road layer | 229.7 ± 131.2 | 4220 ± 2571 | 74 ± 14 | 0.63 ± 0.10 | [206, 193, 290] |
| Distilled search, time only, event + road layer | 228.9 ± 132.1 | 4247 ± 2491 | 78 ± 9 | 0.47 ± 0.13 | [195, 201, 290] |
| Distilled search, budget-blind, event + road layer | 224.9 ± 140.7 | 4121 ± 2756 | 72 ± 12 | 0.44 ± 0.20 | [194, 191, 290] |
| Distilled search, budget-aware, event + road layer | 225.2 ± 140.3 | 4127 ± 2742 | 71 ± 8 | 0.60 ± 0.13 | [197, 189, 290] |
| P4 event, road-distance assignment | 229.7 ± 131.2 | 4220 ± 2571 | 74 ± 14 | 0.63 ± 0.10 | [206, 193, 290] |
| P0 event, road-distance assignment | 230.1 ± 129.4 | 4269 ± 2436 | 87 ± 26 | 0.48 ± 0.10 | [201, 200, 290] |
| P1 route filter, event | 194.0 ± 124.7 | 3350 ± 2158 | 36 ± 12 | 0.41 ± 0.17 | [166, 164, 252] |

**Paired within seed (all events)**

| comparison | dT (min) | dE (kJ) | per-seed dT |
|---|---|---|---|
| P0: event − periodic | 0.2 ± 0.3 | 4 ± 6 | [+0.0, +0.2, +0.3] |
| P4: event − periodic | 0.1 ± 0.3 | 4 ± 5 | [+0.1, +0.0, +0.2] |
| P0: event + layer − open loop | -18.1 ± 31.0 | -420 ± 863 | [-9.7, -12.2, -32.5] |
| P0+ control: event + layer − open loop | -16.5 ± 34.5 | -399 ± 913 | [-9.8, -7.3, -32.5] |
| P4: event + layer − open loop | -14.9 ± 38.0 | -373 ± 966 | [-6.0, -6.1, -32.5] |
| P4 − P0+ control (both event + layer) | 3.5 ± 40.7 | 74 ± 887 | [+21.4, -10.8, -0.0] |
| P1 route filter − P0+ control | -32.2 ± 28.1 | -796 ± 730 | [-19.2, -39.2, -38.3] |

Replans event − periodic (P0): -2.29 ± 2.06 per mission.

